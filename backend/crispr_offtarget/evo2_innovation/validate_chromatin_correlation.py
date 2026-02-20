
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_data():
    base_dir = Path("data/features")
    
    print("Loading Evo2 features...")
    with open(base_dir / "evo2_feats_1k.json", 'r') as f:
        evo_data = json.load(f)
        
    print("Loading ATAC signals...")
    with open(base_dir / "atac_sig_1k.json", 'r') as f:
        atac_data = json.load(f)
        
    print(f"Evo2 samples: {len(evo_data)}")
    print(f"ATAC samples: {len(atac_data)}")
    
    # Filter valid samples
    X_global = []
    X_center = []
    y_signal = []
    
    valid_count = 0
    for i in range(min(len(evo_data), len(atac_data))):
        e = evo_data[i]
        a = atac_data[i]
        
        if e.get('success') and a is not None and len(a) == 100:
            X_global.append(e['global_embedding'])
            X_center.append(e['center_embedding'])
            y_signal.append(a)
            valid_count += 1
            
    print(f"Valid paired samples: {valid_count}")
    
    Xg = np.array(X_global)
    Xc = np.array(X_center)
    Ys = np.array(y_signal)
    
    print(f"X_global shape: {Xg.shape}, dtype: {Xg.dtype}")
    print(f"X_center shape: {Xc.shape}, dtype: {Xc.dtype}")
    print(f"y_signal shape: {Ys.shape}, dtype: {Ys.dtype}")
    
    if np.isnan(Xg).any(): print("WARNING: NaNs in X_global")
    if np.isnan(Xc).any(): print("WARNING: NaNs in X_center")
    if np.isnan(Ys).any(): print("WARNING: NaNs in y_signal")
    
    # Fill NaNs if any (shouldn't be)
    Xg = np.nan_to_num(Xg)
    Xc = np.nan_to_num(Xc)
    Ys = np.nan_to_num(Ys)
    
    return Xg, Xc, Ys

def analyze_correlation():
    X_global, X_center, y_signals = load_data()
    
    if len(y_signals) < 10:
        print("ERROR: Not enough data points!")
        return
    
    # Target 1: Total Accessibility (Sum of bins)
    y_total = np.sum(y_signals, axis=1)
    # Log transform for stability
    y_total_log = np.log1p(y_total)
    
    print("\n--- Correlation Analysis: Predicting Total Accessibility ---")
    
    # Model A: Global Embedding
    model = Ridge(alpha=1.0)
    y_pred_global = cross_val_predict(model, X_global, y_total_log, cv=5)
    corr_global, _ = pearsonr(y_total_log, y_pred_global)
    print(f"Global Embedding -> Total ATAC: Pearson R = {corr_global:.4f}")
    
    # Model B: Center Embedding
    y_pred_center = cross_val_predict(model, X_center, y_total_log, cv=5)
    corr_center, _ = pearsonr(y_total_log, y_pred_center)
    print(f"Center Embedding -> Total ATAC: Pearson R = {corr_center:.4f}")
    
    # Model C: Combined
    X_combined = np.hstack([X_global, X_center])
    y_pred_comb = cross_val_predict(model, X_combined, y_total_log, cv=5)
    corr_comb, _ = pearsonr(y_total_log, y_pred_comb)
    print(f"Combined Embeddings -> Total ATAC: Pearson R = {corr_comb:.4f}")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(y_total_log, y_pred_comb, alpha=0.5)
    plt.xlabel("True Log Total ATAC")
    plt.ylabel("Predicted Log Total ATAC")
    plt.title(f"Zero-Shot Chromatin Prediction (R={corr_comb:.2f})")
    plt.plot([y_total_log.min(), y_total_log.max()], [y_total_log.min(), y_total_log.max()], 'r--')
    plt.grid(True)
    plt.savefig("results/chromatin_validation_scatter.png")
    print("\nSaved scatter plot to results/chromatin_validation_scatter.png")
    
    return corr_comb

if __name__ == "__main__":
    analyze_correlation()
