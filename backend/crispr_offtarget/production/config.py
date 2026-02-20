"""
Configuration for CRISPR Off-Target Prediction

Centralizes all parameters for consistent behavior across scripts.
Based on thesis findings (11.5x threshold variation, population calibration).
"""

from pathlib import Path
from typing import Dict


class CRISPRConfig:
    """Configuration for CRISPR off-target prediction"""
    
    # ==================== PATHS ====================
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    RESULTS_DIR = BASE_DIR / "results"
    FIGURES_DIR = RESULTS_DIR / "figures"
    
    # ==================== EVO2 SETTINGS ====================
    MODEL_NAME = "evo2_7b"
    PRECISION = "bfloat16"
    BATCH_SIZE = 50
    CONTEXT_WINDOW = 512  # bp on each side
    
    # ==================== SCORING PARAMETERS ====================
    # Position-specific weights (from CRISPR literature)
    # Position-specific weights (from CRISPR literature)
    # Positions 18-20 (indices 17-19) are the CORE SEED (PAM-adjacent)
    SEED_POSITIONS = [17, 18, 19]  
    
    # Positions 13-17 (indices 12-16) are PAM-proximal secondary
    PAM_PROXIMAL_POSITIONS = [12, 13, 14, 15, 16] 
    
    FIVE_PRIME_POSITIONS = list(range(7))  # 0-indexed (positions 1-7)
    
    SEED_WEIGHT = 3.0
    PAM_WEIGHT = 2.0
    FIVE_PRIME_WEIGHT = 0.5
    DEFAULT_WEIGHT = 1.0
    
    # ==================== POPULATION PENALTIES ====================
    # From thesis: population_thresholds.py findings
    POPULATION_PENALTIES: Dict[str, float] = {
        'AFR': -0.015,  # African (most stringent)
        'EUR': -0.003,  # European
        'EAS': -0.006,  # East Asian
        'AMR': -0.010,  # Admixed American
        'SAS': -0.008,  # South Asian
        'ASJ': -0.004,  # Ashkenazi Jewish
        'FIN': -0.003,  # Finnish
        'NFE': -0.003,  # Non-Finnish European
        'OTH': -0.005,  # Other
    }
    
    # ==================== CONFIDENCE THRESHOLDS ====================
    # From thesis: Honest uncertainty quantification
    HIGH_CONF_STD_THRESHOLD = 0.05  # std dev of delta scores
    MODERATE_CONF_STD_THRESHOLD = 0.03
    HIGH_CONF_MEAN_THRESHOLD = 0.04  # mean |delta|
    MODERATE_CONF_MEAN_THRESHOLD = 0.02
    MAX_MISMATCHES_FOR_HIGH_CONF = 3
    
    # ==================== VALIDATION SETTINGS ====================
    GUIDE_SEQ_READ_THRESHOLD = 10  # read count > 10 = validated
    AUROC_TARGET = 0.70
    AUPRC_TARGET = 0.30
    
    # ==================== COST SETTINGS ====================
    COST_PER_OFFTARGET = 0.012  # USD
    GPU_HOURLY_RATE = 3.50  # H100 on Modal
    
    # ==================== gRNA DEFAULTS ====================
    GRNA_LENGTH = 20
    PAM_SEQUENCE = "NGG"
    MAX_MISMATCHES = 6


# Singleton instance
config = CRISPRConfig()


def get_position_weights() -> list:
    """Get full 20-position weight array"""
    weights = [config.DEFAULT_WEIGHT] * config.GRNA_LENGTH
    
    for pos in config.FIVE_PRIME_POSITIONS:
        weights[pos] = config.FIVE_PRIME_WEIGHT
    
    for pos in config.SEED_POSITIONS:
        weights[pos] = config.SEED_WEIGHT
    
    for pos in config.PAM_PROXIMAL_POSITIONS:
        weights[pos] = config.PAM_WEIGHT
    
    return weights


def get_population_penalty(population: str) -> float:
    """Get population-specific penalty from config"""
    return config.POPULATION_PENALTIES.get(population.upper(), 0.0)


if __name__ == "__main__":
    print("CRISPR Configuration Summary")
    print("="*50)
    print(f"Base Dir: {config.BASE_DIR}")
    print(f"Model: {config.MODEL_NAME}")
    print(f"Seed Weight: {config.SEED_WEIGHT}x")
    print(f"AUROC Target: {config.AUROC_TARGET}")
    print(f"Cost/Off-target: ${config.COST_PER_OFFTARGET}")
    print(f"\nPosition Weights: {get_position_weights()}")
    print(f"\nPopulation Penalties: {config.POPULATION_PENALTIES}")
