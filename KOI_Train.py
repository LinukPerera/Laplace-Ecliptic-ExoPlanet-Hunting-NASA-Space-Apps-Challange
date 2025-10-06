# Modified RF_Model.py for KOI with added joblib.dumps

# !pip install --upgrade scikit-learn==1.2.2 imbalanced-learn==0.10.1 numpy==1.24.3 joblib==1.2.0

# !pip install imbalanced-learn

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
import logging
import warnings
import joblib
import os
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check GPU availability
try:
    import cupy
    logger.info("CuPy installed, GPU available.")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Pin to P100
except ImportError:
    logger.warning("CuPy not installed, falling back to CPU.")
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

try:
    # Step 1: Load & Clean
    df = pd.read_csv('/kaggle/input/exoplanetslaplaceecliptic/KOIFiltered.csv')
    identifiers = df[['kepoi_name', 'kepler_name', 'koi_disposition']].copy()  # Save for predictions
    drop_cols = ['kepid', 'kepoi_name', 'kepler_name', 'koi_datalink_dvr', 'koi_datalink_dvs', 'koi_comment']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
    
    # Multi-class target (0=CONFIRMED, 1=CANDIDATE, 2=FALSE)
    le = LabelEncoder()
    df['target'] = le.fit_transform(df['koi_disposition'])
    joblib.dump(le, '/kaggle/working/koi_le.pkl')
    df = df.drop('koi_disposition', axis=1)
    logger.info(f"Shape: {df.shape}, Target dist: \n{df['target'].value_counts(normalize=True)}")

    # Step 2: Handle Asymmetric Errors
    error_pairs = [(col.replace('_err1', ''), col, col.replace('_err2', '')) 
                   for col in df.columns if '_err1' in col]
    for param, err1, err2 in error_pairs:
        if err1 in df.columns and err2 in df.columns:
            try:
                df[f'{param}_err_avg'] = (np.abs(df[err1].fillna(0)) + np.abs(df[err2].fillna(0))) / 2
                df = df.drop([err1, err2], axis=1)
            except Exception as e:
                logger.error(f"Error processing {param}: {e}")
    logger.info("Error columns averaged.")

    # Step 3: Identify Numeric and Categorical Columns
    # Protect koi_quarters as categorical
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'koi_quarters' in df.columns and 'koi_quarters' not in cat_cols:
        cat_cols.append('koi_quarters')
    logger.info(f"Initial categorical cols: {cat_cols}")
    
    num_cols = []
    for col in df.columns:
        if col != 'target' and col not in cat_cols:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].dtype in ['float64', 'int64'] and df[col].notna().any():
                    num_cols.append(col)
                else:
                    logger.warning(f"Column {col} is non-numeric or entirely NaN after conversion.")
            except Exception as e:
                logger.warning(f"Column {col} failed numeric conversion: {e}")
                if col not in cat_cols:
                    cat_cols.append(col)
    logger.info(f"Numeric cols: {len(num_cols)}, Categorical cols: {len(cat_cols)}")

    valid_cat_cols = [col for col in cat_cols if df[col].notna().any()]  # Moved here before imputation
    joblib.dump(num_cols, '/kaggle/working/koi_num_cols.pkl')
    joblib.dump(valid_cat_cols, '/kaggle/working/koi_valid_cat_cols.pkl')

    # Step 4: Imputation
    try:
        if num_cols:
            imputer_num = SimpleImputer(strategy='median')
            imputed_data = imputer_num.fit_transform(df[num_cols])
            joblib.dump(imputer_num, '/kaggle/working/koi_imputer_num.pkl')
            df[num_cols] = pd.DataFrame(imputed_data, columns=num_cols, index=df.index)
            logger.info("Numeric imputation completed.")
        else:
            logger.warning("No numeric columns found for imputation.")
        
        if cat_cols:
            logger.info(f"Valid categorical cols for imputation: {valid_cat_cols}")
            if valid_cat_cols:
                imputer_cat = SimpleImputer(strategy='most_frequent')
                imputed_cat_data = imputer_cat.fit_transform(df[valid_cat_cols])
                joblib.dump(imputer_cat, '/kaggle/working/koi_imputer_cat.pkl')
                df[valid_cat_cols] = pd.DataFrame(imputed_cat_data, columns=valid_cat_cols, index=df.index)
                logger.info("Categorical imputation completed.")
            else:
                logger.warning("No valid categorical columns for imputation.")
        else:
            logger.warning("No categorical columns found for imputation.")
    except Exception as e:
        logger.error(f"Imputation failed: {e}")
        raise

    # Step 5: Encoding
    try:
        if valid_cat_cols:
            df = pd.get_dummies(df, columns=valid_cat_cols, drop_first=True, prefix='cat')
        if 'koi_quarters' in df.columns:
            try:
                df['koi_quarters_count'] = df['koi_quarters'].astype(str).str.count('1').fillna(0)
                df = df.drop('koi_quarters', axis=1)
            except Exception as e:
                logger.error(f"Failed to process koi_quarters: {e}")
                df = df.drop('koi_quarters', axis=1, errors='ignore')
        logger.info("Encoding completed.")
        columns_after_encoding = df.columns.tolist()
        joblib.dump(columns_after_encoding, '/kaggle/working/koi_columns_after_encoding.pkl')
    except Exception as e:
        logger.error(f"Encoding failed: {e}")
        raise

    # Step 6: Feature Engineering
    try:
        df['ror_ratio'] = df.get('koi_ror', 0) / df.get('koi_srad', 1).clip(lower=1e-6)
        df['log_period'] = np.log1p(df.get('koi_period', 0))
        num_cols = df.select_dtypes(include=[np.number]).columns.drop('target', errors='ignore')
        logger.info("Feature engineering completed.")
        clip_num_cols = df.select_dtypes(include=[np.number]).columns.drop('target', errors='ignore').tolist()
        joblib.dump(clip_num_cols, '/kaggle/working/koi_clip_num_cols.pkl')
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        raise

    # Step 7: Outlier Handling & Feature Selection
    try:
        Q1 = df[num_cols].quantile(0.25)
        Q3 = df[num_cols].quantile(0.75)
        IQR = Q3 - Q1
        clip_bounds = pd.Series({'lower': Q1 - 1.5 * IQR, 'upper': Q3 + 1.5 * IQR})
        joblib.dump(clip_bounds, '/kaggle/working/koi_clip_bounds.pkl')
        df[num_cols] = df[num_cols].clip(lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR, axis=1)
        logger.info("Outliers clipped.")

        # Drop highly correlated features
        corr_matrix = df[num_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
        joblib.dump(to_drop, '/kaggle/working/koi_to_drop.pkl')
        df = df.drop(to_drop, axis=1)
        num_cols = [c for c in num_cols if c not in to_drop]
        logger.info(f"Dropped {len(to_drop)} correlated features.")

        # Feature selection with XGBoost
        X_temp = df[num_cols]
        y_temp = df['target']
        xgb_temp = xgb.XGBClassifier(random_state=42, device='cuda')
        xgb_temp.fit(X_temp, y_temp)
        importances = pd.Series(xgb_temp.feature_importances_, index=num_cols).sort_values(ascending=False)
        
        # Automated feature removal based on importance threshold
        importance_threshold = 0.01
        top_features = importances[importances >= importance_threshold].index.tolist()
        if not top_features or len(top_features) < 10:
            logger.warning(f"Too few important features ({len(top_features)}), using top 50.")
            top_features = importances.head(50).index.tolist()
        joblib.dump(top_features, '/kaggle/working/koi_top_features.pkl')
        X = df[top_features]
        logger.info(f"Selected {len(top_features)} features: {top_features[:5]}")
    except Exception as e:
        logger.error(f"Feature selection failed: {e}")
        raise

    # Step 8: Balance & Split
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, df['target'], test_size=0.2, 
                                                            stratify=df['target'], random_state=42)
        identifiers_test = identifiers.iloc[X_test.index].copy()  # For predictions
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        scaler = StandardScaler()
        X_train_res = scaler.fit_transform(X_train_res)
        joblib.dump(scaler, '/kaggle/working/koi_scaler.pkl')
        X_test = scaler.transform(X_test)
        logger.info("Data split and balanced.")
    except Exception as e:
        logger.error(f"Balancing/splitting failed: {e}")
        raise

    # Step 9: Hyperparameter Tuning with Optuna
    def objective(trial):
        try:
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
            }
            model = xgb.XGBClassifier(random_state=42, device='cuda', **params)
            cv_scores = cross_val_score(model, X_train_res, y_train_res, cv=StratifiedKFold(5), 
                                      scoring='f1_weighted')
            return cv_scores.mean()
        except Exception as e:
            logger.error(f"Optuna trial failed: {e}")
            return np.nan

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)  # ~6-7 min on P100
    best_params = study.best_params
    logger.info(f"Best params: {best_params}")

    # Step 10: Train & Evaluate
    try:
        model = xgb.XGBClassifier(random_state=42, device='cuda', **best_params)
        model.fit(X_train_res, y_train_res)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Save predictions with identifiers
        pred_df = identifiers_test.copy()
        pred_df['koi_disposition_pred'] = le.inverse_transform(y_pred)
        pred_df.to_csv('/kaggle/working/predictions.csv', index=False)
        logger.info("Predictions saved to /kaggle/working/predictions_updated_koi.csv")
        
        # Evaluate
        print("Classification Report:\n", classification_report(y_test, y_pred, 
                                                              target_names=le.classes_))
        print(f"F1 (weighted): {f1_score(y_test, y_pred, average='weighted'):.4f}")
        print(f"AUC (OvR): {roc_auc_score(y_test, y_pred_proba, multi_class='ovr'):.4f}")
        
        # Feature Importances
        importances = pd.Series(model.feature_importances_, index=top_features).sort_values(ascending=False)
        importances.head(20).plot(kind='bar', figsize=(10,6))
        plt.title('Top 20 Feature Importances')
        plt.savefig('/kaggle/working/importances_updated_KOI.png')
        plt.show()
        
        # Save model
        joblib.dump(model, '/kaggle/working/koi_updated_xgb_model.pkl')
        logger.info("Model saved to /kaggle/working/koi_updated_xgb_model.pkl")
    except Exception as e:
        logger.error(f"Training/evaluation failed: {e}")
        raise

except Exception as e:
    logger.error(f"Pipeline failed: {e}")
    raise