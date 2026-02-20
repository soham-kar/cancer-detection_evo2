
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import os

def load_data():
    print("Loading features...")
    with open('data/features/evo2_feats_1k.json') as f:
        feats = json.load(f)
    
    print("Loading ATAC targets...")
    with open('data/features/atac_sig_1k.json') as f:
        atac = json.load(f)
        
    # X: Global Embeddings (8kb context representation)
    # feats is a list of dicts.
    X = np.array([item['global_embedding'] for item in feats])
    
    # y: ATAC signal. 
    # atac is a list of lists (signal tracks). 
    # For a simple scaler prediction, let's look at the MEAN signal first, 
    # or train a multi-output regressor on the bins.
    # Let's verify the shape given the inspection showed ~128 bins?
    # The inspection showed lists of floats. Let's assume fixed length.
    y_tracks = np.array(atac)
    
    # Target: Mean accessibility often correlates with "openness" score
    y_mean = np.mean(y_tracks, axis=1)
    
    print(f"Features shape: {X.shape}")
    print(f"Targets (Tracks) shape: {y_tracks.shape}")
    print(f"Targets (Mean) shape: {y_mean.shape}")
    
    return X, y_tracks, y_mean

def train_and_evaluate(X, y, name="Mean ATAC"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nTraining Ridge Regression for {name}...")
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    # Metrics
    if y.ndim == 1:
        r2 = r2_score(y_test, y_pred)
        p_corr, _ = pearsonr(y_test, y_pred)
        print(f"{name} Results - R2: {r2:.4f}, Pearson: {p_corr:.4f}")
        
        # Plot
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel(f"Actual {name}")
        plt.ylabel(f"Predicted {name}")
        plt.title(f"Evo2 Embedding vs {name}\nR={p_corr:.3f}")
        plt.tight_layout()
        plt.savefig(f"results/nature_figures/chromatin_val_{name.replace(' ', '_')}.png")
        print(f"Saved plot to results/nature_figures/chromatin_val_{name.replace(' ', '_')}.png")
        
    else:
        # Multi-output
        r2 = r2_score(y_test, y_pred, multioutput='uniform_average')
        # Correlation per sample
        corrs = [pearsonr(y_test[i], y_pred[i])[0] for i in range(len(y_test))]
        mean_corr = np.mean(corrs)
        print(f"{name} Results - Avg R2: {r2:.4f}, Avg Sample Correlation: {mean_corr:.4f}")
        
        # Plot a few random tracks
        plt.figure(figsize=(15, 5))
        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.plot(y_test[i], label='Actual', alpha=0.7)
            plt.plot(y_pred[i], label='Predicted', linestyle='--', alpha=0.7)
            plt.title(f"Sample {i}\nR={corrs[i]:.2f}")
            plt.legend()
        plt.tight_layout()
        plt.savefig("results/nature_figures/chromatin_val_tracks.png")
        print("Saved track comparison plot")

def main():
    os.makedirs('results/nature_figures', exist_ok=True)
    
    try:
        X, y_tracks, y_mean = load_data()
    except Exception as e:
        print(f"Data loading failed: {e}")
        return

    # 1. Predict Mean Accessibility (Scalar)
    # This validates if the embedding "knows" if the region is open or closed
    train_and_evaluate(X, y_mean, name="Mean_Accessiblity")
    
    # 2. Predict Signal Shape (Vector)
    # This validates if the embedding captures the chromatin profile shape
    # Only if tracks are consistent length (which they appeared to be)
    train_and_evaluate(X, y_tracks, name="Signal_Shape")

if __name__ == "__main__":
    main()
