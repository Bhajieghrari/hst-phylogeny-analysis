# Documentation Directory

This directory contains additional documentation for the analysis pipeline.

## Contents

### Analysis Workflow

1. **Sequence Curation** - `scripts/sequence_curation.py`
   - Input: Raw FASTA sequences from UniProt
   - Output: Curated FASTA sequences with quality filtering

2. **Multiple Sequence Alignment** - `scripts/alignment_trim.py`
   - Input: Curated FASTA sequences
   - Output: Aligned and trimmed sequences

3. **Motif Analysis** - `scripts/motif_analysis.py`
   - Input: Aligned sequences
   - Output: MEME motifs, matrices, clusters

4. **Structural Analysis** - `scripts/structural_compare.py`
   - Input: PDB models
   - Output: RMSD matrices, RMSF values, clusters

5. **Figure Generation** - `scripts/figure_generation.py`
   - Input: Results directory
   - Output: Publication-ready figures

## Requirements

See `requirements.txt` for Python dependencies.

## Citation

Please cite the manuscript if you use these scripts.

## Contact

bheghrari@yahoo.com