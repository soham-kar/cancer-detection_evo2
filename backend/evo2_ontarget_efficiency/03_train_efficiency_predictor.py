"""
03_train_efficiency_predictor.py

Train Ridge regression to predict on-target efficiency from Evo2 embeddings.
Evaluates zero-shot generalization to held-out cell type.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr
from pathlib import Path
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

DATA_DIR = Path(__file__).parent / "data"
TRAIN_FEATURES = Path(__file__).parent / "features_train"
TEST_FEATURES = Path(__file__).parent / "features_test"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_embeddings(coords_path: Path, features_dir: Path):
    """Load embeddings and labels."""
    coords = pd.read_csv(coords_path)
    
    embeddings = []
    efficiencies = []
    valid_ids = []
    
    for _, row in coords.iterrows():
        sample_id = row['sample_id']
        npz_path = features_dir / f"sample_{sample_id}.npz"
        
        if npz_path.exists():
            data = np.load(npz_path)
            embeddings.append(data['center_embedding'])
            efficiencies.append(row['efficiency'])
            valid_ids.append(sample_id)
    
    if len(embeddings) == 0:
        logging.warning(f"No embeddings found in {features_dir}")
        return None, None
    
    X = np.stack(embeddings)
    y = np.array(efficiencies)
    
    logging.info(f"Loaded {len(X)} samples from {features_dir}")
    logging.info(f"  Embedding dim: {X.shape[1]}")
    logging.info(f"  Efficiency range: {y.min():.3f} - {y.max():.3f}")
    
    return X, y


def train_and_evaluate(X_train, y_train, X_test, y_test, alpha=1.0):
    """Train Ridge and evaluate zero-shot performance."""
    # Train
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    
    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metrics
    train_metrics = compute_metrics(y_train, y_pred_train, "Train")
    test_metrics = compute_metrics(y_test, y_pred_test, "Test (zero-shot)")
    
    return model, y_pred_test, train_metrics, test_metrics


def compute_metrics(y_true, y_pred, label):
    """Compute regression metrics."""
    spearman_r, spearman_p = spearmanr(y_true, y_pred)
    pearson_r, pearson_p = pearsonr(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    
    logging.info(f"\n{label} Metrics:")
    logging.info(f"  Spearman r: {spearman_r:.4f} (p={spearman_p:.2e})")
    logging.info(f"  Pearson r:  {pearson_r:.4f} (p={pearson_p:.2e})")
    logging.info(f"  R²:         {r2:.4f}")
    logging.info(f"  MSE:        {mse:.4f}")
    
    return {
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "r2": r2,
        "mse": mse
    }


def interpret_results(test_spearman_r):
    """Interpret zero-shot performance."""
    logging.info("\n" + "=" * 60)
    logging.info("INTERPRETATION")
    logging.info("=" * 60)
    
    # DeepHF baseline on endogenous loci: r ~ 0.45
    DEEPHF_BASELINE = 0.45
    
    if test_spearman_r >= 0.65:
        logging.info("✅ SUCCESS: Evo2 significantly beats DeepHF (r > 0.65)")
        logging.info("   → Zero-shot generalization works!")
        logging.info("   → Publication: Nature Biotechnology")
        return "success"
    elif test_spearman_r >= 0.50:
        logging.info("⚠️ MARGINAL: Better than DeepHF but not transformative")
        logging.info(f"   → Improvement: {test_spearman_r - DEEPHF_BASELINE:.3f}")
        logging.info("   → Publication: Bioinformatics / Nucleic Acids Research")
        return "marginal"
    else:
        logging.info(f"❌ BELOW TARGET: r = {test_spearman_r:.3f} < 0.50")
        logging.info(f"   → DeepHF baseline: {DEEPHF_BASELINE}")
        logging.info("   → Chromatin may not be the key factor")
        return "failure"


def plot_results(y_test, y_pred, test_metrics, output_path):
    """Plot prediction vs actual."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scatter plot
    ax = axes[0]
    ax.scatter(y_test, y_pred, alpha=0.6, s=30)
    ax.plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect')
    ax.set_xlabel('True Efficiency', fontsize=12)
    ax.set_ylabel('Predicted Efficiency', fontsize=12)
    ax.set_title(
        f"Zero-Shot On-Target Prediction\n"
        f"Spearman r = {test_metrics['spearman_r']:.3f}",
        fontsize=14
    )
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Comparison bar plot
    ax = axes[1]
    methods = ['DeepHF\n(baseline)', 'CRISPRon\n(baseline)', 'Evo2\n(zero-shot)']
    r_values = [0.45, 0.42, test_metrics['spearman_r']]
    colors = ['gray', 'gray', 'steelblue']
    
    bars = ax.bar(methods, r_values, color=colors)
    ax.axhline(0.65, color='green', linestyle='--', label='Target (r=0.65)')
    ax.set_ylabel('Spearman r on Endogenous Loci')
    ax.set_title('Zero-Shot Generalization Comparison')
    ax.set_ylim(0, 1)
    ax.legend()
    
    # Value labels
    for bar, val in zip(bars, r_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved figure: {output_path}")


def main():
    logging.info("=" * 60)
    logging.info("On-Target Efficiency Predictor Training")
    logging.info("=" * 60)
    
    # Load data
    train_coords = DATA_DIR / "train_coords.csv"
    test_coords = DATA_DIR / "test_coords.csv"
    
    if not train_coords.exists():
        logging.error("Run 01_prepare_deephf_data.py first")
        return
    
    X_train, y_train = load_embeddings(train_coords, TRAIN_FEATURES)
    X_test, y_test = load_embeddings(test_coords, TEST_FEATURES)
    
    if X_train is None or X_test is None:
        logging.error("No embeddings found. Run 02_modal_extract.py first")
        return
    
    # Train and evaluate
    model, y_pred, train_metrics, test_metrics = train_and_evaluate(
        X_train, y_train, X_test, y_test
    )
    
    # Interpret
    result = interpret_results(test_metrics['spearman_r'])
    
    # Save results
    results_df = pd.DataFrame([{
        **test_metrics,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "interpretation": result
    }])
    results_df.to_csv(RESULTS_DIR / "training_results.csv", index=False)
    
    # Plot
    plot_results(y_test, y_pred, test_metrics, RESULTS_DIR / "zero_shot_results.png")
    
    logging.info(f"\n✅ Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
