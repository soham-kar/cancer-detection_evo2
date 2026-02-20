"""
Configuration for Evo2 CRISPR Innovation Pipeline
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results" / "evo2_innovation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Model settings
EVO2_MODEL = "evo2_7b"  # Options: evo2_7b, evo2_40b_8k
HIDDEN_DIM = 512  # Evo2 7B hidden dimension
BEST_LAYER = 25   # Layer for feature extraction (empirically 20-30 work best)

# Training settings
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 100
PATIENCE = 10  # Early stopping patience
SEED = 42

# Feature extraction settings
CONTEXT_WINDOW = 5  # ±5bp around mismatch for context
PAM_POSITIONS = (20, 23)  # PAM region indices

# Seed weights (for comparison baseline)
SEED_WEIGHTS = [0.5]*7 + [1.0]*10 + [3.0]*3

# Mismatch type penalties (biophysical)
MISMATCH_TYPES = {
    # Wobble pairs (tolerable)
    ('G', 'T'): 0.3,
    ('T', 'G'): 0.3,
    ('A', 'C'): 0.3,
    ('C', 'A'): 0.3,
    # Purine-purine (bad)
    ('A', 'G'): 0.8,
    ('G', 'A'): 0.8,
    # Pyrimidine-pyrimidine (moderate)
    ('C', 'T'): 0.6,
    ('T', 'C'): 0.6,
    # Default
    'default': 1.0
}
