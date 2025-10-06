# Install compatible libraries
# no GPU ACC

!pip install scikit-learn==1.2.2 imbalanced-learn==0.10.1 numpy==1.24.3 joblib==1.2.0 --force-reinstall

# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Check versions
print("Library Versions:")
!pip show scikit-learn
!pip show imbalanced-learn
!pip show numpy
!pip show joblib

# Load dataset
df = pd.read_csv('/kaggle/input/exoplanetslaplaceecliptic/KOI.csv')
print("\nDataset Shape:", df.shape)
print("Dataset Columns:", list(df.columns))

# Create mapping for planet identification
id_mapping = df[['kepoi_name', 'kepler_name']].copy()
id_mapping['original_index'] = df.index

# Cleaning: Drop non-predictive columns if they exist
drop_cols = ['kepid', 'kepoi_name', 'kepler_name', 'koi_comment', 'koi_datalink_dvr', 'koi_datalink_dvs']
df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

# Target and features
target = 'koi_disposition'
if target not in df.columns:
    raise ValueError(f"Target column '{target}' not found in dataset")
X = df.drop(target, axis=1)
y = LabelEncoder().fit_transform(df[target])  # Encode: e.g., 0=CONFIRMED, 1=CANDIDATE, 2=FALSE POSITIVE

# Column types
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
print("\nCategorical Columns:", cat_cols)
print("Numerical Columns:", num_cols)

# Handle special columns (bitmasks)
if 'koi_quarters' in X.columns:
    X['koi_quarters_count'] = X['koi_quarters'].apply(lambda x: str(x).count('1') if pd.notnull(x) else 0)
    num_cols.append('koi_quarters_count')
    if 'koi_quarters' in cat_cols:
        cat_cols.remove('koi_quarters')

# Combine asymmetric errors (only for existing columns)
error_pairs = [
    ('koi_period_err1', 'koi_period_err2'),
    ('koi_prad_err1', 'koi_prad_err2'),
    ('koi_depth_err1', 'koi_depth_err2'),
    ('koi_duration_err1', 'koi_duration_err2'),
    ('koi_time0bk_err1', 'koi_time0bk_err2'),
    ('koi_impact_err1', 'koi_impact_err2'),
    ('koi_teq_err1', 'koi_teq_err2'),
    ('koi_insol_err1', 'koi_insol_err2'),
    ('koi_srad_err1', 'koi_srad_err2'),
    ('koi_steff_err1', 'koi_steff_err2'),
    ('koi_smass_err1', 'koi_smass_err2'),
    ('koi_slogg_err1', 'koi_slogg_err2')
]
valid_error_pairs = [(pos, neg) for pos, neg in error_pairs if pos in X.columns and neg in X.columns]
for pos, neg in valid_error_pairs:
    X[pos.replace('_err1', '_err')] = (X[pos].abs() + X[neg].abs()) / 2
    X = X.drop([pos, neg], axis=1)
    num_cols = [c for c in num_cols if c not in [pos, neg]]
    num_cols.append(pos.replace('_err1', '_err'))

X = X.dropna(axis=1, how='all')  # Drop fully empty columns
print("\nColumns after Error Pair Processing:", list(X.columns))

# Update column types after processing
cat_cols = [col for col in X.select_dtypes(include=['object']).columns if col in X.columns]
num_cols = [col for col in X.select_dtypes(include=['float64', 'int64']).columns if col in X.columns]
print("Updated Categorical Columns:", cat_cols)
print("Updated Numerical Columns:", num_cols)

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', IterativeImputer(random_state=42, max_iter=10)),
            ('scaler', StandardScaler())
        ]), num_cols),
        ('cat', Pipeline(steps=[
            ('imputer', IterativeImputer(random_state=42, max_iter=10)),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_cols)
    ])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42, shuffle=True
)
train_indices = X_train.index
test_indices = X_test.index

# Pipeline with RFE
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('selector', RFE(estimator=RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=30)),
    ('balancer', SMOTE(random_state=42)),
    ('classifier', StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(random_state=42)),
            ('gb', GradientBoostingClassifier(random_state=42))
        ],
        final_estimator=LogisticRegression(),
        cv=5
    ))
])

# Tune hyperparameters
param_grid = {
    'selector__n_features_to_select': [20, 30],
    'classifier__rf__n_estimators': [100],
    'classifier__gb__n_estimators': [100]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=1)
grid_search.fit(X_train, y_train)

# Evaluate
y_pred = grid_search.predict(X_test)
label_encoder = LabelEncoder().fit(df[target])
print(f"\nBest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Map predictions to kepoi_name
results = pd.DataFrame({
    'original_index': test_indices,
    'koi_disposition_pred': label_encoder.inverse_transform(y_pred),
    'koi_disposition_true': label_encoder.inverse_transform(y_test)
})
results = results.merge(id_mapping, on='original_index', how='left')
print("\nPrediction Results with Identifiers (First 10):")
print(results[['kepoi_name', 'kepler_name', 'koi_disposition_true', 'koi_disposition_pred']].head(10))

# Extract selected features
preprocessed_cols = preprocessor.get_feature_names_out()
selected_mask = grid_search.best_estimator_.named_steps['selector'].get_support()
selected_features = preprocessed_cols[selected_mask]
print("\nSelected Top Features:", list(selected_features))

# Save results
results.to_csv('/kaggle/working/predictions.csv', index=False)