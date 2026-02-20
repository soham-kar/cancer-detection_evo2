"""
Stratified Training for Evo2 on CHANGE-seq Hard Cases
Trains Ridge regression on Evo2 embeddings, evaluates on hard case subset
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr
import logging

# Paths
OUTPUT_DIR = Path(__file__).parent
TRAIN_PATH = OUTPUT_DIR / "stratified_train.csv"
TEST_PATH = OUTPUT_DIR / "stratified_test.csv"
FEATURES_CACHE = Path(__file__).parent / "features_cache"

logging.basicConfig(level=logging.INFO, format='%(message)s')


def load_data(csv_path: Path, cache_dir: Path) -> tuple:
    """
    Load embeddings and labels from cache
    Returns: X (embeddings), y (log_reads), metadata (DataFrame)
    """
    df = pd.read_csv(csv_path)
    logging.info(f"Loading {len(df)} samples from {csv_path.name}")
    
    embeddings = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        site_id = row['site_id']
        npz_path = cache_dir / f"seq_{int(site_id)}.npz"
        
        if not npz_path.exists():
            continue
            
        try:
            data = np.load(npz_path, allow_pickle=True)
            # Concatenate global + center embeddings (1024-dim total)
            parts = []
            if 'global_embedding' in data:
                parts.append(data['global_embedding'])
            if 'center_embedding' in data:
                parts.append(data['center_embedding'])
            
            if parts:
                emb = np.concatenate(parts)
                embeddings.append(emb)
                valid_indices.append(idx)
        except Exception as e:
            logging.warning(f"Failed to load {npz_path}: {e}")
    
    # Filter DataFrame to match embeddings
    filtered_df = df.loc[valid_indices].reset_index(drop=True)
    X = np.stack(embeddings)
    y = filtered_df['log_reads'].values
    
    logging.info(f"✓ Loaded {len(X)} embeddings (dim: {X.shape[1]})")
    return X, y, filtered_df


def train_ridge(X_train: np.ndarray, y_train: np.ndarray) -> Ridge:
    """Train Ridge regression with cross-validated alpha"""
    # Alpha range: 0.1 (less regularization) to 10.0 (more)
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train, y_train)
    logging.info(f"✓ Trained Ridge (alpha={model.alpha}, n_features={X_train.shape[1]})")
    return model


def evaluate(model: Ridge, X_test: np.ndarray, y_test: np.ndarray, df_test: pd.DataFrame) -> dict:
    """Evaluate on full test set and hard case subset"""
    y_pred = model.predict(X_test)
    
    # Overall metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    spearman, spearman_p = spearmanr(y_test, y_pred)
    
    # Hard case metrics (0-3 mismatches)
    hard_mask = df_test['distance'] <= 3
    if hard_mask.sum() > 10:
        y_hard_true = y_test[hard_mask]
        y_hard_pred = y_pred[hard_mask]
        r2_hard = r2_score(y_hard_true, y_hard_pred)
        spearman_hard, spearman_hard_p = spearmanr(y_hard_true, y_hard_pred)
    else:
        r2_hard = spearman_hard = spearman_hard_p = np.nan
    
    # Easy case metrics (4+ mismatches)
    easy_mask = df_test['distance'] >= 4
    if easy_mask.sum() > 10:
        y_easy_true = y_test[easy_mask]
        y_easy_pred = y_pred[easy_mask]
        spearman_easy, _ = spearmanr(y_easy_true, y_easy_pred)
    else:
        spearman_easy = np.nan
    
    results = {
        'overall': {
            'n_samples': len(y_test),
            'mse': mse,
            'r2': r2,
            'spearman': spearman,
            'spearman_p': spearman_p
        },
        'hard_cases': {
            'n_samples': hard_mask.sum(),
            'r2': r2_hard,
            'spearman': spearman_hard,
            'spearman_p': spearman_hard_p
        },
        'easy_cases': {
            'n_samples': easy_mask.sum(),
            'spearman': spearman_easy
        }
    }
    
    return results, y_pred


def compare_to_baseline(df_test: pd.DataFrame) -> dict:
    """Baseline: predict using mismatch count only"""
    y_true = df_test['log_reads'].values
    y_pred_baseline = -df_test['distance'].values * 0.5  # Simple: more mismatches = less cleavage
    
    spearman_base, _ = spearmanr(y_true, y_pred_baseline)
    
    # Hard case baseline
    hard_mask = df_test['distance'] <= 3
    if hard_mask.sum() > 10:
        spearman_base_hard, _ = spearmanr(y_true[hard_mask], y_pred_baseline[hard_mask])
    else:
        spearman_base_hard = np.nan
    
    return {
        'overall_spearman': spearman_base,
        'hard_cases_spearman': spearman_base_hard
    }


def main():
    logging.info("=" * 60)
    logging.info("Evo2 Training on CHANGE-seq Hard Cases")
    logging.info("=" * 60)
    
    # Load data
    X_train, y_train, df_train = load_data(TRAIN_PATH, FEATURES_CACHE)
    X_test, y_test, df_test = load_data(TEST_PATH, FEATURES_CACHE)
    
    # Train model
    model = train_ridge(X_train, y_train)
    
    # Evaluate
    results, y_pred = evaluate(model, X_test, y_test, df_test)
    baseline = compare_to_baseline(df_test)
    
    # Print results
    logging.info("\n" + "=" * 60)
    logging.info("RESULTS")
    logging.info("=" * 60)
    
    logging.info(f"\n📊 Overall Test Set (n={results['overall']['n_samples']}):")
    logging.info(f"   R²: {results['overall']['r2']:.3f}")
    logging.info(f"   Spearman r: {results['overall']['spearman']:.3f} (p={results['overall']['spearman_p']:.2e})")
    logging.info(f"   Baseline (mismatch count): {baseline['overall_spearman']:.3f}")
    
    logging.info(f"\n🎯 Hard Cases Only (0-3 mm, n={results['hard_cases']['n_samples']}):")
    logging.info(f"   R²: {results['hard_cases']['r2']:.3f}")
    logging.info(f"   Spearman r: {results['hard_cases']['spearman']:.3f} (p={results['hard_cases']['spearman_p']:.2e})")
    logging.info(f"   Baseline (mismatch count): {baseline['hard_cases_spearman']:.3f}")
    
    # Interpretation
    logging.info("\n" + "=" * 60)
    logging.info("INTERPRETATION")
    logging.info("=" * 60)
    
    evo2_gain = results['hard_cases']['spearman'] - baseline['hard_cases_spearman']
    
    if evo2_gain > 0.15:
        logging.info(f"✅ WIN: Evo2 gains +{evo2_gain:.3f} Spearman on hard cases")
        logging.info("   → Foundation model captures chromatin/mismatch chemistry")
    elif evo2_gain > 0.05:
        logging.info(f"⚠️ MARGINAL: Evo2 gains +{evo2_gain:.3f} Spearman")
        logging.info("   → May need more hard case samples or model tuning")
    else:
        logging.info(f"❌ NO GAIN: Evo2 matches baseline (+{evo2_gain:.3f})")
        logging.info("   → Publish binary Kleinstiver critique; Evo2 adds no value here")
    
    # Save predictions
    df_test['predicted_log_reads'] = y_pred
    df_test.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
    logging.info(f"\n✓ Predictions saved to test_predictions.csv")


if __name__ == "__main__":
    main()
