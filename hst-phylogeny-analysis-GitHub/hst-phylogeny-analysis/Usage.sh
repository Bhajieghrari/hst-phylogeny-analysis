# 1. Sequence curation
python scripts/sequence_curation.py --input raw_sequences.fasta --output curated.fasta

# 2. Multiple sequence alignment
python scripts/alignment_trim.py --input curated.fasta --output aligned.fasta

# 3. Motif discovery
python scripts/motif_analysis.py --input aligned.fasta --output motifs/

# 4. Phylogenetic analysis
iqtree2 -s aligned.fasta -m JTT+G4 -B 1000 -alrt 1000

# 5. Structural comparison
python scripts/structural_compare.py --models structures/ --output results/

# 6. Generate figures
python scripts/figure_generation.py --results results/ --output figures/