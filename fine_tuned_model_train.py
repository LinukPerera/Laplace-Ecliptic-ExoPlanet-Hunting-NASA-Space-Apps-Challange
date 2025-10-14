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
import time  # Added for timing the training loops
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check GPU availability
try:
    import cupy
    logger.info("CuPy installed, GPU available.")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
except ImportError:
    logger.warning("CuPy not installed, falling back to CPU.")
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Column mappings (unchanged)
mapping_k2 = {
    'kepid': 'tic_id',
    'kepoi_name': 'epic_candname',
    'kepler_name': 'pl_name',
    'koi_disposition': 'disposition',
    'koi_period': 'pl_orbper',
    'koi_time0bk': 'pl_tranmid',
    'koi_eccen': 'pl_orbeccen',
    'koi_longp': 'pl_orblper',
    'koi_impact': 'pl_imppar',
    'koi_duration': 'pl_trandur',
    'koi_ingress': 'pl_trandurlim',
    'koi_depth': 'pl_trandep',
    'koi_ror': 'pl_ratror',
    'koi_srho': 'st_dens',
    'koi_prad': 'pl_rade',
    'koi_sma': 'pl_orbsmax',
    'koi_incl': 'pl_orbincl',
    'koi_teq': 'pl_eqt',
    'koi_insol': 'pl_insol',
    'koi_dor': 'pl_ratdor',
    'koi_steff': 'st_teff',
    'koi_slogg': 'st_logg',
    'koi_smet': 'st_met',
    'koi_srad': 'st_rad',
    'koi_smass': 'st_mass',
    'koi_sage': 'st_age',
    'ra': 'ra',
    'dec': 'dec',
    'koi_kepmag': 'sy_kepmag',
    'koi_gmag': 'sy_gmag',
    'koi_rmag': 'sy_rmag',
    'koi_imag': 'sy_imag',
    'koi_zmag': 'sy_zmag',
    'koi_jmag': 'sy_jmag',
    'koi_hmag': 'sy_hmag',
    'koi_kmag': 'sy_kmag',
}

mapping_tess = {
    'koi_period': 'pl_orbper',
    'koi_time0bk': 'pl_tranmid',
    'koi_duration': 'pl_trandurh',
    'koi_depth': 'pl_trandep',
    'koi_prad': 'pl_rade',
    'koi_insol': 'pl_insol',
    'koi_teq': 'pl_eqt',
    'koi_steff': 'st_teff',
    'koi_slogg': 'st_logg',
    'koi_srad': 'st_rad',
    'ra': 'ra',
    'dec': 'dec',
    'koi_kepmag': 'st_tmag',
}

def process_dataset(df, prefix, mapping=None, fit=False, le=None, is_koi=False):
    df = df.copy()
    if mapping:
        df.rename(columns=mapping, inplace=True)
    if 'target' not in df.columns:
        disposition_col = 'koi_disposition' if is_koi else 'disposition' if 'disposition' in df.columns else 'tfopwg_disp'
        if le is None and not fit:
            raise ValueError("LE must be provided for transform")
        if not is_koi:
            if 'disposition' in df.columns:  # K2 dataset
                df[disposition_col] = df[disposition_col].replace({
                    'CONFIRMED': 'CONFIRMED',
                    'CANDIDATE': 'CANDIDATE',
                    'FALSE POSITIVE': 'FALSE POSITIVE',
                    'REFUTED': 'FALSE POSITIVE'
                })
            elif 'tfopwg_disp' in df.columns:  # TESS dataset
                df[disposition_col] = df[disposition_col].replace({
                    'KP': 'CONFIRMED',
                    'PC': 'CANDIDATE',
                    'APC': 'CANDIDATE',
                    'FP': 'FALSE POSITIVE'
                })
        if fit:
            le = LabelEncoder()
            df['target'] = le.fit_transform(df[disposition_col])
            joblib.dump(le, f'{prefix}_le.pkl')
        else:
            unique_labels = df[disposition_col].unique()
            le_classes = le.classes_
            unmapped = [label for label in unique_labels if label not in le_classes and pd.notna(label)]
            if unmapped:
                logger.warning(f"Unmapped labels found: {unmapped}. Mapping to 'FALSE POSITIVE'...")
                df[disposition_col] = df[disposition_col].apply(lambda x: x if x in le_classes else 'FALSE POSITIVE')
            df['target'] = le.transform(df[disposition_col])
        df = df.drop(disposition_col, axis=1)
    y = df['target']
    df = df.drop('target', axis=1)
    drop_cols = ['kepid', 'kepoi_name', 'kepler_name', 'koi_datalink_dvr', 'koi_datalink_dvs', 'koi_comment'] if is_koi else []
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    if fit:
        if is_koi:
            error_pairs = [(col.replace('_err1', ''), col, col.replace('_err1', '_err2')) for col in df.columns if '_err1' in col]
            for param, err1, err2 in error_pairs:
                if err1 in df.columns and err2 in df.columns:
                    df[f'{param}_err_avg'] = (np.abs(df[err1].fillna(0)) + np.abs(df[err2].fillna(0))) / 2
                    df = df.drop([err1, err2], axis=1)

        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        if is_koi and 'koi_quarters' in df.columns and 'koi_quarters' not in cat_cols:
            cat_cols.append('koi_quarters')
        num_cols = [col for col in df.columns if col not in cat_cols and df[col].notna().any() and pd.to_numeric(df[col], errors='coerce').notna().any()]
        valid_cat_cols = [col for col in cat_cols if df[col].notna().any()]
        joblib.dump(num_cols, f'{prefix}_num_cols.pkl')
        joblib.dump(valid_cat_cols, f'{prefix}_valid_cat_cols.pkl')
    else:
        num_cols = joblib.load(f'{prefix}_num_cols.pkl')
        valid_cat_cols = joblib.load(f'{prefix}_valid_cat_cols.pkl')

    for col in num_cols:
        if col not in df.columns:
            df[col] = np.nan
    for col in valid_cat_cols:
        if col not in df.columns:
            df[col] = np.nan

    if fit:
        if num_cols:
            imputer_num = SimpleImputer(strategy='median')
            df[num_cols] = imputer_num.fit_transform(df[num_cols])
            joblib.dump(imputer_num, f'{prefix}_imputer_num.pkl')
        if valid_cat_cols:
            imputer_cat = SimpleImputer(strategy='most_frequent')
            df[valid_cat_cols] = imputer_cat.fit_transform(df[valid_cat_cols])
            joblib.dump(imputer_cat, f'{prefix}_imputer_cat.pkl')
    else:
        if num_cols:
            imputer_num = joblib.load(f'{prefix}_imputer_num.pkl')
            df[num_cols] = imputer_num.transform(df[num_cols])
        if valid_cat_cols and os.path.exists(f'{prefix}_imputer_cat.pkl'):
            imputer_cat = joblib.load(f'{prefix}_imputer_cat.pkl')
            df[valid_cat_cols] = imputer_cat.transform(df[valid_cat_cols])

    if fit:
        if valid_cat_cols:
            df = pd.get_dummies(df, columns=valid_cat_cols, drop_first=True, prefix='cat')
        if is_koi and 'koi_quarters' in df.columns:
            df['koi_quarters_count'] = df['koi_quarters'].astype(str).str.count('1').fillna(0)
            df = df.drop('koi_quarters', axis=1)
        columns_after_encoding = df.columns.tolist()
        joblib.dump(columns_after_encoding, f'{prefix}_columns_after_encoding.pkl')
    else:
        if valid_cat_cols:
            df = pd.get_dummies(df, columns=valid_cat_cols, drop_first=True, prefix='cat')
        columns_after_encoding = joblib.load(f'{prefix}_columns_after_encoding.pkl')
        df = df.reindex(columns=columns_after_encoding, fill_value=0)

    # Feature engineering
    if is_koi:
        ror = 'koi_ror'
        srad = 'koi_srad'
        period = 'koi_period'
    else:
        ror = 'pl_ratror' if 'pl_ratror' in df.columns else 'pl_rade'
        srad = 'st_rad'
        period = 'pl_orbper'
    srad_series = df[srad] if srad in df.columns else pd.Series(np.ones(len(df)), index=df.index)
    df['ror_ratio'] = df.get(ror, 0) / srad_series.clip(lower=1e-6)
    df['log_period'] = np.log1p(df.get(period, 0))

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if fit:
        joblib.dump(num_cols, f'{prefix}_clip_num_cols.pkl')

    if fit:
        Q1 = df[num_cols].quantile(0.25)
        Q3 = df[num_cols].quantile(0.75)
        IQR = Q3 - Q1
        clip_bounds = {'lower': Q1 - 1.5 * IQR, 'upper': Q3 + 1.5 * IQR}
        joblib.dump(clip_bounds, f'{prefix}_clip_bounds.pkl')
    else:
        clip_bounds = joblib.load(f'{prefix}_clip_bounds.pkl')
    for col in num_cols:
        if col in clip_bounds['lower']:
            df[col] = df[col].clip(lower=clip_bounds['lower'][col], upper=clip_bounds['upper'][col])

    if fit:
        corr_matrix = df[num_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
        joblib.dump(to_drop, f'{prefix}_to_drop.pkl')
    else:
        to_drop = joblib.load(f'{prefix}_to_drop.pkl')
    df = df.drop(columns=[col for col in to_drop if col in df.columns], errors='ignore')

    if fit:
        xgb_temp = xgb.XGBClassifier(random_state=42, device='cuda')
        xgb_temp.fit(df, y)
        importances = pd.Series(xgb_temp.feature_importances_, index=df.columns).sort_values(ascending=False)
        importance_threshold = 0.01
        top_features = importances[importances >= importance_threshold].index.tolist()
        if len(top_features) < 10:
            top_features = importances.head(50).index.tolist()
        joblib.dump(top_features, f'{prefix}_top_features.pkl')
    else:
        top_features = joblib.load(f'{prefix}_top_features.pkl')
    X = df[top_features]

    return X, y, le if fit else le

# Main script (starting from K2 fine-tuning)
try:
    # Check for prerequisite files
    required_files = [
        '/kaggle/working/koi_X_val.pkl',
        '/kaggle/working/koi_y_val.pkl',
        '/kaggle/working/koi_le.pkl',
        '/kaggle/working/koi_scaler.pkl',
        '/kaggle/working/koi_best_params.pkl',
        '/kaggle/working/koi_model.pkl',
        '/kaggle/working/koi_num_cols.pkl',
        '/kaggle/working/koi_valid_cat_cols.pkl',
        '/kaggle/working/koi_columns_after_encoding.pkl',
        '/kaggle/working/koi_clip_num_cols.pkl',
        '/kaggle/working/koi_clip_bounds.pkl',
        '/kaggle/working/koi_to_drop.pkl',
        '/kaggle/working/koi_top_features.pkl'
    ]
    for file in required_files:
        if not os.path.exists(file):
            logger.error(f"Required file {file} is missing!")
            raise FileNotFoundError(f"Required file {file} is missing!")

    # Load KOI validation set
    logger.info("Loading KOI validation set...")
    X_koi_val = joblib.load('/kaggle/working/koi_X_val.pkl')
    y_koi_val = joblib.load('/kaggle/working/koi_y_val.pkl')
    le = joblib.load('/kaggle/working/koi_le.pkl')
    logger.info("KOI validation set loaded successfully.")

    # Fine-tune on K2, validate on TESS + KOI val
    logger.info("Processing K2 dataset...")
    df_k2 = pd.read_csv('/kaggle/input/exoplanetslaplaceecliptic/K2edited.csv')
    drop_cols_k2 = ['rowid', 'pl_name', 'hostname', 'pl_letter', 'k2_name', 'epic_hostname', 'epic_candname', 'hd_name', 'hip_name', 'tic_id', 'gaia_id', 'default_flag', 'disp_refname', 'disc_refname', 'disc_pubdate', 'disc_locale', 'disc_facility', 'disc_telescope', 'disc_instrument', 'pl_refname', 'st_refname', 'sy_refname', 'rowupdate', 'pl_pubdate', 'releasedate', 'k2_campaigns']
    df_k2 = df_k2.drop(columns=[col for col in drop_cols_k2 if col in df_k2.columns], errors='ignore')
    X_k2, y_k2, _ = process_dataset(df_k2, 'koi', mapping=mapping_k2, fit=False, le=le)
    X_k2 = joblib.load('/kaggle/working/koi_scaler.pkl').transform(X_k2)
    logger.info("K2 dataset processed successfully.")

    logger.info("Processing TESS dataset...")
    df_tess = pd.read_csv('/kaggle/input/exoplanetslaplaceecliptic/TessEdited.csv')
    drop_cols_tess = ['toi', 'toipfx', 'tid', 'ctoi_alias', 'rastr', 'decstr', 'toi_created', 'rowupdate']
    df_tess = df_tess.drop(columns=[col for col in drop_cols_tess if col in df_tess.columns], errors='ignore')
    X_tess, y_tess, _ = process_dataset(df_tess, 'koi', mapping=mapping_tess, fit=False, le=le)
    X_tess = joblib.load('/kaggle/working/koi_scaler.pkl').transform(X_tess)
    logger.info("TESS dataset processed successfully.")

    logger.info("Combining validation sets...")
    X_val_comb = np.vstack((X_tess, X_koi_val))
    y_val_comb = np.concatenate((y_tess, y_koi_val))
    logger.info("Validation sets combined.")

    logger.info("Applying SMOTE to K2 data...")
    smote = SMOTE(random_state=42)
    X_k2_res, y_k2_res = smote.fit_resample(X_k2, y_k2)
    logger.info("SMOTE applied to K2 data.")

    logger.info("Loading model and parameters for K2 fine-tuning...")
    best_params = joblib.load('/kaggle/working/koi_best_params.pkl')
    best_params['learning_rate'] *= 0.1
    model = joblib.load('/kaggle/working/koi_model.pkl')
    logger.info("Model and parameters loaded.")

    # K2 fine-tuning with timing and logging
    logger.info("Starting K2 fine-tuning...")
    start_time = time.time()
    model.fit(X_k2_res, y_k2_res, eval_set=[(X_val_comb, y_val_comb)], early_stopping_rounds=10, verbose=True)
    end_time = time.time()
    logger.info(f"K2 fine-tuning completed in {end_time - start_time:.2f} seconds.")
    joblib.dump(model, '/kaggle/working/koi_k2_finetuned_model.pkl')

    logger.info("Evaluating K2 fine-tuned model...")
    y_pred_val = model.predict(X_val_comb)
    y_pred_proba_val = model.predict_proba(X_val_comb)
    print("K2 Fine-tune Validation Report:\n", classification_report(y_val_comb, y_pred_val, target_names=le.classes_))
    print(f"K2 Fine-tune F1 (weighted): {f1_score(y_val_comb, y_pred_val, average='weighted'):.4f}")
    print(f"K2 Fine-tune AUC (OvR): {roc_auc_score(y_val_comb, y_pred_proba_val, multi_class='ovr'):.4f}")

    logger.info("Saving K2 feature importances plot...")
    importances = pd.Series(model.feature_importances_, index=joblib.load('/kaggle/working/koi_top_features.pkl')).sort_values(ascending=False)
    importances.head(20).plot(kind='bar', figsize=(10,6))
    plt.title('Top 20 Feature Importances (K2 Fine-tuned)')
    plt.savefig('/kaggle/working/importances_k2_finetuned.png')
    plt.close()
    logger.info("K2 feature importances plot saved.")

    # TESS fine-tuning
    logger.info("Splitting TESS dataset for fine-tuning...")
    X_tess_train, X_tess_val, y_tess_train, y_tess_val = train_test_split(X_tess, y_tess, test_size=0.2, stratify=y_tess, random_state=42)
    X_val_comb = np.vstack((X_tess_val, X_koi_val))
    y_val_comb = np.concatenate((y_tess_val, y_koi_val))
    logger.info("TESS dataset split and validation set combined.")

    logger.info("Applying SMOTE to TESS training data...")
    smote = SMOTE(random_state=42)
    X_tess_train_res, y_tess_train_res = smote.fit_resample(X_tess_train, y_tess_train)
    logger.info("SMOTE applied to TESS training data.")

    logger.info("Loading model for TESS fine-tuning...")
    best_params['learning_rate'] *= 0.1
    model = joblib.load('/kaggle/working/koi_k2_finetuned_model.pkl')
    logger.info("Model loaded for TESS fine-tuning.")

    # TESS fine-tuning with timing and logging
    logger.info("Starting TESS fine-tuning...")
    start_time = time.time()
    model.fit(X_tess_train_res, y_tess_train_res, eval_set=[(X_val_comb, y_val_comb)], early_stopping_rounds=10, verbose=True)
    end_time = time.time()
    logger.info(f"TESS fine-tuning completed in {end_time - start_time:.2f} seconds.")
    joblib.dump(model, '/kaggle/working/koi_k2_tess_finetuned_model.pkl')

    logger.info("Evaluating TESS fine-tuned model...")
    y_pred_val = model.predict(X_val_comb)
    y_pred_proba_val = model.predict_proba(X_val_comb)
    print("TESS Fine-tune Validation Report:\n", classification_report(y_val_comb, y_pred_val, target_names=le.classes_))
    print(f"TESS Fine-tune F1 (weighted): {f1_score(y_val_comb, y_pred_val, average='weighted'):.4f}")
    print(f"TESS Fine-tune AUC (OvR): {roc_auc_score(y_val_comb, y_pred_proba_val, multi_class='ovr'):.4f}")

    logger.info("Saving TESS feature importances plot...")
    importances = pd.Series(model.feature_importances_, index=joblib.load('/kaggle/working/koi_top_features.pkl')).sort_values(ascending=False)
    importances.head(20).plot(kind='bar', figsize=(10,6))
    plt.title('Top 20 Feature Importances (TESS Fine-tuned)')
    plt.savefig('/kaggle/working/importances_k2_tess_finetuned.png')
    plt.close()
    logger.info("TESS feature importances plot saved.")

except Exception as e:
    logger.error(f"Script failed: {e}")
    raise
