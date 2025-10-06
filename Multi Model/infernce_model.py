import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get the absolute path of the directory containing inference.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Column mappings from KOI to K2 and TESS
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

# Weights, biased towards KOI and K2
weights = {'koi': 0.4, 'k2': 0.4, 'tess': 0.2}

def load_preprocessor(prefix):
    try:
        imputer_num = joblib.load(os.path.join(BASE_DIR, f'{prefix}_imputer_num.pkl'))
        imputer_cat = joblib.load(os.path.join(BASE_DIR, f'{prefix}_imputer_cat.pkl')) if os.path.exists(os.path.join(BASE_DIR, f'{prefix}_imputer_cat.pkl')) else None
        num_cols = joblib.load(os.path.join(BASE_DIR, f'{prefix}_num_cols.pkl'))
        valid_cat_cols = joblib.load(os.path.join(BASE_DIR, f'{prefix}_valid_cat_cols.pkl')) if os.path.exists(os.path.join(BASE_DIR, f'{prefix}_valid_cat_cols.pkl')) else []
        columns_after_encoding = joblib.load(os.path.join(BASE_DIR, f'{prefix}_columns_after_encoding.pkl'))
        clip_bounds = joblib.load(os.path.join(BASE_DIR, f'{prefix}_clip_bounds.pkl'))
        clip_num_cols = joblib.load(os.path.join(BASE_DIR, f'{prefix}_clip_num_cols.pkl'))
        to_drop = joblib.load(os.path.join(BASE_DIR, f'{prefix}_to_drop.pkl'))
        top_features = joblib.load(os.path.join(BASE_DIR, f'{prefix}_top_features.pkl'))
        scaler = joblib.load(os.path.join(BASE_DIR, f'{prefix}_scaler.pkl'))
        le = joblib.load(os.path.join(BASE_DIR, f'{prefix}_le.pkl'))
        logger.info(f"Loaded preprocessor files for {prefix}")
        return {
            'imputer_num': imputer_num,
            'imputer_cat': imputer_cat,
            'num_cols': num_cols,
            'valid_cat_cols': valid_cat_cols,
            'columns_after_encoding': columns_after_encoding,
            'clip_bounds': clip_bounds,
            'clip_num_cols': clip_num_cols,
            'to_drop': to_drop,
            'top_features': top_features,
            'scaler': scaler,
            'le': le
        }
    except FileNotFoundError as e:
        logger.error(f"Missing preprocessor file for {prefix}: {e}")
        raise ValueError(f"Cannot proceed without preprocessor files for {prefix}. Please ensure all required .pkl files are in {BASE_DIR}.")

def process_data(df, prefix, mapping=None, is_koi=False, is_k2=False, is_tess=False):
    df = df.copy()
    if mapping:
        df.rename(columns=mapping, inplace=True)
    drop_cols = []
    if is_koi:
        drop_cols = ['kepid', 'kepoi_name', 'kepler_name', 'koi_datalink_dvr', 'koi_datalink_dvs', 'koi_comment']
        error_pairs = [(col.replace('_err1', ''), col, col.replace('_err1', '_err2')) for col in df.columns if '_err1' in col]
        for param, err1, err2 in error_pairs:
            if err1 in df.columns and err2 in df.columns:
                df[f'{param}_err_avg'] = (np.abs(df[err1].fillna(0)) + np.abs(df[err2].fillna(0))) / 2
                df = df.drop([err1, err2], axis=1)
    elif is_k2:
        drop_cols = ['rowid', 'pl_name', 'hostname', 'pl_letter', 'k2_name', 'epic_hostname', 'epic_candname', 'hd_name', 'hip_name', 'tic_id', 'gaia_id', 'default_flag', 'disp_refname', 'disc_refname', 'disc_pubdate', 'disc_locale', 'disc_facility', 'disc_telescope', 'disc_instrument', 'pl_refname', 'st_refname', 'sy_refname', 'rowupdate', 'pl_pubdate', 'releasedate', 'k2_campaigns']
    elif is_tess:
        drop_cols = ['toi', 'toipfx', 'tid', 'ctoi_alias', 'rastr', 'decstr', 'toi_created', 'rowupdate']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    try:
        preproc = load_preprocessor(prefix)
    except ValueError as e:
        raise

    for col in preproc['num_cols']:
        if col not in df.columns:
            df[col] = np.nan
    for col in preproc['valid_cat_cols']:
        if col not in df.columns:
            df[col] = np.nan

    df[preproc['num_cols']] = preproc['imputer_num'].transform(df[preproc['num_cols']])
    if preproc['valid_cat_cols'] and preproc['imputer_cat']:
        df[preproc['valid_cat_cols']] = preproc['imputer_cat'].transform(df[preproc['valid_cat_cols']])

    df = pd.get_dummies(df, columns=preproc['valid_cat_cols'], drop_first=True, prefix='cat')
    if is_koi and 'koi_quarters' in df.columns:
        df['koi_quarters_count'] = df['koi_quarters'].astype(str).str.count('1').fillna(0)
        df = df.drop('koi_quarters', axis=1)

    df = df.reindex(columns=preproc['columns_after_encoding'], fill_value=0)

    if is_koi:
        df['ror_ratio'] = df.get('koi_ror', 0) / df.get('koi_srad', 1).clip(lower=1e-6)
        df['log_period'] = np.log1p(df.get('koi_period', 0))
    elif is_k2:
        df['ror_ratio'] = df.get('pl_ratror', 0) / df.get('st_rad', 1).clip(lower=1e-6)
        df['log_period'] = np.log1p(df.get('pl_orbper', 0))
    elif is_tess:
        df['ror_ratio'] = df.get('pl_rade', 0) / df.get('st_rad', 1).clip(lower=1e-6)
        df['log_period'] = np.log1p(df.get('pl_orbper', 0))

    for col in preproc['clip_num_cols']:
        if col in df.columns:
            df[col] = df[col].clip(lower=preproc['clip_bounds']['lower'][col], upper=preproc['clip_bounds']['upper'][col])

    df = df.drop(columns=[col for col in preproc['to_drop'] if col in df.columns], errors='ignore')

    X = df[preproc['top_features']]
    X = preproc['scaler'].transform(X)

    return X, preproc['le']

def run_inference(df, model_code=None):
    if df.empty:
        logger.error("Input DataFrame is empty.")
        raise ValueError("Input DataFrame is empty.")

    # Try to select identifiers; fallback to index if none are found
    available_identifiers = []
    possible_identifiers = {
        'koi': ['kepid', 'kepoi_name', 'kepler_name'],
        'k2': ['tic_id', 'epic_candname', 'pl_name'],
        'tess': ['toi', 'tid']
    }
    
    for source, cols in possible_identifiers.items():
        found_cols = [col for col in cols if col in df.columns]
        if found_cols:
            available_identifiers.extend(found_cols)
    
    if available_identifiers:
        identifiers = df[available_identifiers].copy()
    else:
        identifiers = pd.DataFrame({'row_index': df.index})
        logger.warning("No identifier columns found in input DataFrame. Using row index as identifier.")

    # Load models with error handling
    models = {}
    probas = {}
    available_weights = {}
    for prefix, weight in weights.items():
        try:
            model_file = os.path.join(BASE_DIR, f'{prefix}_updated_xgb_model.pkl' if prefix == 'koi' else f'{prefix}_xgb_model.pkl')
            if not os.path.exists(model_file):
                raise FileNotFoundError(f"Model file {model_file} does not exist.")
            models[prefix] = joblib.load(model_file)
            available_weights[prefix] = weight
            logger.info(f"Loaded model for {prefix}: {model_file}")
        except FileNotFoundError as e:
            logger.warning(f"Model file for {prefix} not found: {e}. Skipping {prefix} model.")
            available_weights[prefix] = 0

    # Check if any models were loaded
    if not models:
        logger.error(f"No models could be loaded. Please ensure model files exist in {BASE_DIR}.")
        raise ValueError("No models available for inference.")

    # Normalize weights if some models are missing
    total_weight = sum(available_weights.values())
    if total_weight > 0:
        for prefix in available_weights:
            available_weights[prefix] = available_weights[prefix] / total_weight
    else:
        logger.error("No valid models available after weight normalization.")
        raise ValueError("No valid models available for inference.")

    # Process and predict
    for prefix in models:
        try:
            X, le = process_data(df, prefix, 
                                mapping=mapping_k2 if prefix == 'k2' else mapping_tess if prefix == 'tess' else None,
                                is_koi=(prefix == 'koi'), 
                                is_k2=(prefix == 'k2'), 
                                is_tess=(prefix == 'tess'))
            probas[prefix] = models[prefix].predict_proba(X)
            if prefix == 'koi':
                le_koi = le
            elif prefix == 'k2':
                le_k2 = le
            elif prefix == 'tess':
                le_tess = le
            logger.info(f"Generated predictions for {prefix}")
        except Exception as e:
            logger.warning(f"Failed to process or predict for {prefix}: {e}. Skipping {prefix} model.")
            available_weights[prefix] = 0
            probas.pop(prefix, None)

    # Recalculate weights if processing failed for any model
    total_weight = sum(available_weights.values())
    if total_weight == 0:
        logger.error("No models produced valid predictions.")
        raise ValueError("No valid predictions available.")

    for prefix in available_weights:
        available_weights[prefix] = available_weights[prefix] / total_weight if total_weight > 0 else 0

    # Select reference label encoder (KOI > K2 > TESS)
    if 'koi' in probas:
        le_reference = le_koi
        logger.info("Using KOI model labels for alignment.")
    elif 'k2' in probas:
        le_reference = le_k2
        logger.warning("KOI model missing; using K2 model labels for alignment.")
    elif 'tess' in probas:
        le_reference = le_tess
        logger.warning("KOI and K2 models missing; using TESS model labels for alignment.")
    else:
        logger.error("No valid predictions available for label alignment.")
        raise ValueError("No valid predictions available.")

    # Define common labels based on reference label encoder
    common_labels = le_reference.classes_
    num_samples = list(probas.values())[0].shape[0]

    # Initialize aligned probability arrays
    aligned_probas = {prefix: np.zeros((num_samples, len(common_labels))) for prefix in probas}

    # Map probabilities to common labels
    for prefix in probas:
        le = le_koi if prefix == 'koi' and 'koi' in probas else le_k2 if prefix == 'k2' and 'k2' in probas else le_tess if prefix == 'tess' and 'tess' in probas else None
        for i, label in enumerate(le.classes_):
            if label in common_labels:
                j = np.where(common_labels == label)[0][0]
                aligned_probas[prefix][:, j] = probas[prefix][:, i]

    # Weighted average probabilities
    total_proba = sum(available_weights[prefix] * aligned_probas[prefix] for prefix in probas)

    # Final prediction
    y_final = np.argmax(total_proba, axis=1)
    disposition_pred = le_reference.inverse_transform(y_final)

    # Format output as list of dictionaries
    predictions = [
        {
            "keid": str(i),
            "kepler_name": f"sample_{i}",
            "result": disposition_pred[i]
        }
        for i in range(len(disposition_pred))
    ]

    return predictions

if __name__ == "__main__":
    import sys
    DEFAULT_INPUT_CSV = os.path.join(BASE_DIR, 'KOISample.csv')
    input_csv = DEFAULT_INPUT_CSV
    if len(sys.argv) >= 2:
        input_csv = sys.argv[1]
    
    try:
        if not os.path.exists(input_csv):
            logger.error(f"Default or provided input CSV file not found: {input_csv}")
            sys.exit(1)
        df = pd.read_csv(input_csv)
        if df.empty:
            logger.error("Input CSV file is empty.")
            sys.exit(1)
        predictions = run_inference(df)
        logger.info(f"Predictions: {predictions}")
    except Exception as e:
        logger.error(f"Failed to run inference: {e}")
        sys.exit(1)