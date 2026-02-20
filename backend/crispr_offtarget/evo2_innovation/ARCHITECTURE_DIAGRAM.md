# Evo2 Innovation - Architecture Diagram

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         EVO2 CHROMATIN-AWARE CRISPR                      │
│                    Quantitative Off-Target Prediction                    │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: DATA PREPARATION                                                │
└─────────────────────────────────────────────────────────────────────────┘

    CHANGE-seq Dataset
    (Quantitative Cleavage Data)
           │
           ├─ Guide RNA sequences (20bp)
           ├─ Off-target sites (chr:pos)
           ├─ Read counts (quantitative)
           └─ ATAC-seq BigWig files
           │
           ▼
    extract_8kb_sequences.py
           │
           ├─ Fetch from hg38 genome
           ├─ Center on cleavage site
           └─ Extract ±4kb context
           │
           ▼
    ┌──────────────────────────┐
    │  8kb Genomic Sequences   │
    │  + Metadata (JSON)       │
    └──────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: FEATURE EXTRACTION (Modal.com H100 GPU)                        │
└─────────────────────────────────────────────────────────────────────────┘

    modal_extract.py
           │
           ├─────────────────────┬─────────────────────┐
           │                     │                     │
           ▼                     ▼                     ▼
    ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
    │  Evo2 7B    │      │  Tokenizer  │      │   ATAC-seq  │
    │  Inference  │      │  (encode)   │      │  Extraction │
    └─────────────┘      └─────────────┘      └─────────────┘
           │                     │                     │
           │                     │                     │
           ▼                     ▼                     ▼
    ┌─────────────────────────────────────────────────────┐
    │         Hidden States [layers × seq_len × 512]      │
    └─────────────────────────────────────────────────────┘
           │
           ├─────────────────────┬─────────────────────┐
           │                     │                     │
           ▼                     ▼                     ▼
    ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
    │   Global    │      │   Center    │      │  Evo2 Score │
    │  Embedding  │      │  Embedding  │      │  (Scalar)   │
    │   [512]     │      │   [512]     │      │             │
    └─────────────┘      └─────────────┘      └─────────────┘
           │                     │                     │
           └─────────────────────┴─────────────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │  Cached Features (NPZ)   │
                    │  - global_emb [512]      │
                    │  - center_emb [512]      │
                    │  - evo2_score [1]        │
                    │  - atac_signal [100]     │
                    └──────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: MODEL TRAINING                                                  │
└─────────────────────────────────────────────────────────────────────────┘

    chromatin_model.py
           │
           ▼
    ┌─────────────────────────────────────────────────────┐
    │         ChromatinAwareCRISPR (Multi-Task)           │
    └─────────────────────────────────────────────────────┘
           │
           ├─────────────────────┬─────────────────────┐
           │                     │                     │
           ▼                     ▼                     ▼
    ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
    │  Chromatin  │      │  Cleavage   │      │   Fusion    │
    │    Head     │      │    Head     │      │    Layer    │
    │             │      │             │      │             │
    │ center_emb  │      │ global_emb  │      │  Combined   │
    │     ↓       │      │     +       │      │  Features   │
    │  Linear     │      │ atac_pred   │      │             │
    │     ↓       │      │     ↓       │      │             │
    │   ReLU      │      │  Linear     │      │             │
    │     ↓       │      │     ↓       │      │             │
    │ Dropout     │      │   ReLU      │      │             │
    │     ↓       │      │     ↓       │      │             │
    │  Linear     │      │  Linear     │      │             │
    │     ↓       │      │     ↓       │      │             │
    │ [100 bins]  │      │    [1]      │      │             │
    └─────────────┘      └─────────────┘      └─────────────┘
           │                     │
           │                     │
           ▼                     ▼
    ┌─────────────┐      ┌─────────────┐
    │  ATAC-seq   │      │  Cleavage   │
    │ Prediction  │      │ Prediction  │
    │  (100 bins) │      │ (log reads) │
    └─────────────┘      └─────────────┘
           │                     │
           │                     │
           ▼                     ▼
    ┌─────────────┐      ┌─────────────┐
    │  MSE Loss   │      │  MSE Loss   │
    │  (ATAC)     │      │ (Cleavage)  │
    └─────────────┘      └─────────────┘
           │                     │
           └─────────────────────┘
                     │
                     ▼
            ┌─────────────────┐
            │   Total Loss    │
            │ = L_cleave +    │
            │   λ * L_atac    │
            └─────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 4: EVALUATION                                                      │
└─────────────────────────────────────────────────────────────────────────┘

    evaluate.py
           │
           ├─────────────────────┬─────────────────────┬─────────────────────┐
           │                     │                     │                     │
           ▼                     ▼                     ▼                     ▼
    ┌─────────────┐      ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
    │  Evo2 Model │      │  Heuristic  │      │  Ensemble   │      │  Baseline   │
    │             │      │  (Weighted  │      │  (Evo2 +    │      │  (Mismatch  │
    │             │      │  Mismatch)  │      │  Biophys)   │      │   Count)    │
    └─────────────┘      └─────────────┘      └─────────────┘      └─────────────┘
           │                     │                     │                     │
           └─────────────────────┴─────────────────────┴─────────────────────┘
                                         │
                                         ▼
                              ┌──────────────────────┐
                              │  Spearman Correlation│
                              │  Pearson Correlation │
                              │  MSE / MAE           │
                              └──────────────────────┘
                                         │
                                         ▼
                              ┌──────────────────────┐
                              │  Performance Report  │
                              │  - Evo2: ρ = 0.7?    │
                              │  - Heuristic: ρ = 0.5│
                              │  - Improvement: +0.2 │
                              └──────────────────────┘

```

## Data Flow Details

### 1. Input Data Format

```json
{
  "seq_id": "site_123",
  "chrom": "chr1",
  "center": 12345678,
  "sequence_8kb": "ATCG...",
  "grna_target_seq": "GTCACCTCCAATGACTAGGG",
  "reads": 456.7,
  "normalized_reads": 0.234
}
```

### 2. Feature Extraction Output

```json
{
  "seq_id": "site_123",
  "evo2_score": -1.084908,
  "seq_length": 8000,
  "success": true,
  "source": "cache"
}
```

### 3. Cached Features (NPZ)

```python
{
  "evo2_score": float,           # Scalar
  "global_embedding": [512],     # Float16
  "center_embedding": [512],     # Float16
  "atac_signal": [100]           # Float32 (optional)
}
```

### 4. Model Input Batch

```python
{
  "global_emb": torch.Tensor([batch, 512]),
  "center_emb": torch.Tensor([batch, 512]),
  "log_cleavage": torch.Tensor([batch]),      # Target
  "atac_signal": torch.Tensor([batch, 100])   # Target
}
```

### 5. Model Output

```python
{
  "log_cleavage": torch.Tensor([batch, 1]),   # Predicted
  "atac_pred": torch.Tensor([batch, 100])     # Predicted
}
```

## Key Design Decisions

### 1. Why 8kb Context?
- **Chromatin accessibility** is influenced by distal regulatory elements
- **Evo2's strength** is long-range context (1M bp capable)
- **Biological relevance**: Enhancers can be 10-100kb away
- **Practical limit**: 8kb balances context vs computational cost

### 2. Why Multi-Task Learning?
- **Regularization**: ATAC prediction prevents overfitting
- **Mechanistic**: Forces model to learn chromatin → cleavage relationship
- **Interpretability**: Can visualize predicted chromatin patterns
- **Auxiliary signal**: ATAC-seq provides additional supervision

### 3. Why Separate Global and Center Embeddings?
- **Global (8kb)**: Captures cell state, chromatin context
- **Center (1kb)**: Captures local binding site features
- **Biological**: Cleavage depends on both local and global factors
- **Empirical**: Improves performance vs single embedding

### 4. Why Float16 for Caching?
- **Storage**: 50% reduction vs Float32
- **Precision**: Sufficient for embeddings (not gradients)
- **Speed**: Faster I/O for large datasets
- **Cost**: Reduces Modal volume storage costs

## Performance Bottlenecks

### 1. Feature Extraction (Slowest)
```
Single sequence: ~10 seconds on H100
Bottleneck: Evo2 forward pass (8kb sequence)
Solution: Batch processing (2-4 sequences)
```

### 2. ATAC-seq Extraction
```
Single region: ~1 second
Bottleneck: BigWig file I/O
Solution: Batch queries, cache results
```

### 3. Model Training (Fast)
```
10K samples: ~1 hour on single GPU
Bottleneck: Data loading (if not cached)
Solution: Pre-extract all features, use DataLoader
```

### 4. Inference (Fastest)
```
Single prediction: ~0.1 seconds
Bottleneck: Feature loading from cache
Solution: In-memory caching for repeated queries
```

## Cost Breakdown

### Feature Extraction (H100)
```
Cost: $4/hour
Speed: 360 sequences/hour (10 sec each)
Per sequence: $0.011
1000 sequences: $11
```

### Alternative: L40S
```
Cost: $2/hour
Speed: 180 sequences/hour (20 sec each)
Per sequence: $0.011 (same!)
1000 sequences: $11
```

### Storage (Modal Volumes)
```
Per sequence: 1 KB (compressed)
1000 sequences: 1 MB
Cost: Negligible (<$0.01/month)
```

### Training (Single GPU)
```
Cost: $2/hour (L40S sufficient)
Duration: 1 hour for 10K samples
Total: $2 per training run
```

## Comparison to Alternatives

### vs. Binary Classification (Main Pipeline)
```
Main Pipeline:
- Task: Binary (on/off target)
- Context: 20bp
- Baseline: AUROC 0.92
- Evo2 improvement: +0.02
- Cost: Low
- Use case: Safety screening

Innovation Pipeline:
- Task: Quantitative (cleavage rate)
- Context: 8kb
- Baseline: Spearman 0.3-0.5
- Evo2 improvement: +0.2-0.3 (expected)
- Cost: High
- Use case: Therapeutic dosing
```

### vs. Traditional ML (Random Forest)
```
Random Forest:
- Features: Mismatch positions, GC content
- Performance: Spearman 0.4-0.5
- Training: Minutes
- Inference: Milliseconds
- Interpretability: High

Evo2 Innovation:
- Features: 8kb embeddings + chromatin
- Performance: Spearman 0.6-0.8 (target)
- Training: Hours
- Inference: Seconds
- Interpretability: Medium (via chromatin)
```

### vs. Deep Learning (CNN/Transformer)
```
CNN/Transformer:
- Input: One-hot encoded DNA (8kb)
- Architecture: Custom conv/attention layers
- Performance: Unknown (not benchmarked)
- Training: Days
- Inference: Seconds

Evo2 Innovation:
- Input: Evo2 embeddings (pre-trained)
- Architecture: Lightweight MLP heads
- Performance: Leverages 8.8T token pretraining
- Training: Hours (transfer learning)
- Inference: Seconds
```

## Future Enhancements

### 1. Attention Mechanisms
```python
class AttentionPooling(nn.Module):
    def forward(self, embeddings, positions):
        # Learn to weight mismatch positions
        weights = self.attention(embeddings)
        return (embeddings * weights).sum(dim=0)
```

### 2. Multi-Gene Calibration
```python
# Gene-specific thresholds
thresholds = {
    'BRCA1': {'benign': -0.5, 'pathogenic': 0.5},
    'TP53': {'benign': -0.3, 'pathogenic': 0.7}
}
```

### 3. Ensemble with Experimental Data
```python
# Combine Evo2 with CHANGE-seq, GUIDE-seq, CIRCLE-seq
final_score = (
    0.5 * evo2_score +
    0.3 * change_seq_score +
    0.2 * guide_seq_score
)
```

### 4. Active Learning
```python
# Select most uncertain predictions for wet-lab validation
uncertainty = std(ensemble_predictions)
top_uncertain = argsort(uncertainty)[-100:]
# Send to lab for CHANGE-seq validation
```

## Conclusion

The architecture is **well-designed** with clear biological motivation. The main risk is whether the performance improvement justifies the computational cost. Next step is to run medium-scale testing (100 samples) to validate the approach.

**Key Success Metric**: Spearman correlation > 0.6 on held-out test set
