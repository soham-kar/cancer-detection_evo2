# GSE149363 CHANGE-seq + Epigenetic Data

## Status
- Downloading: ⏳

## Data Structure

GSE149363 is a **SuperSeries** containing:

| SubSeries | Data Type | Files Needed |
|-----------|-----------|--------------|
| GSE149295 | CHANGE-seq + ChIP-seq | Processed counts |
| GSE149361 | ATAC-seq | bigWig tracks |
| GSE149362 | RNA-seq | Not needed for now |

## Downloaded Files

### CHANGE-seq (Cleavage Counts)
- [ ] Supplementary tables from Nature paper

### ATAC-seq (Chromatin Accessibility)
- [ ] bigWig files from GSE149361

### ChIP-seq (Histone Marks)
- [ ] H3K4me1
- [ ] H3K4me3
- [ ] H3K27ac

## Usage

After download, run:
```python
python parse_change_seq.py
```
