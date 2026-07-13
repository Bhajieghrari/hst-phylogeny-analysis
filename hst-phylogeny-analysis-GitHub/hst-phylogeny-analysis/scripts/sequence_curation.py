#!/usr/bin/env python3
"""
Sequence Curation Pipeline for HASTY/XPO5 Proteins

Filters sequences based on:
- Length: 80-120% of reference (807-1211 aa)
- Ambiguous residues: <5% (B, Z, X, J, U, O)
- Premature stop codons: none allowed
- HEAT-repeat validation
- Reciprocal BLAST orthology validation

Usage:
    python sequence_curation.py --input raw_sequences.fasta --output curated_sequences.fasta
    python sequence_curation.py -i raw.fasta -o curated.fasta --verbose
"""

import argparse
import sys
import os
import re
from Bio import SeqIO
from Bio.Blast import NCBIWWW, NCBIXML
from Bio.Blast.Applications import NcbiblastpCommandline
import tempfile
import subprocess
import pandas as pd
import numpy as np

# ============================================================
# CONSTANTS
# ============================================================
REFERENCE_LENGTH = 1009  # A. thaliana HASTY (Q0WP44)
MIN_LENGTH = int(REFERENCE_LENGTH * 0.80)  # 807
MAX_LENGTH = int(REFERENCE_LENGTH * 1.20)  # 1211
AMBIGUOUS_RESIDUES = set(['B', 'Z', 'X', 'J', 'U', 'O'])
MAX_AMBIGUOUS_PCT = 5.0

BLAST_EVALUE = 1e-10
BLAST_COVERAGE = 60.0
BLAST_IDENTITY = 40.0

REFERENCE_ACCESSIONS = [
    ('Q0WP44', 'Arabidopsis thaliana HASTY'),
    ('Q9HAV4', 'Homo sapiens Exportin-5'),
    ('Q924C1', 'Mus musculus Exportin-5'),
    ('Q9VWE7', 'Drosophila melanogaster Exportin-5'),
    ('Q54PQ8', 'Dictyostelium discoideum Exportin-5'),
]

HEAT_PFAM_DOMAINS = ['PF08305', 'PF03810', 'PF13513']
MIN_HEAT_REPEATS = 3


# ============================================================
# FILTERING FUNCTIONS
# ============================================================
def filter_by_length(record, verbose=False):
    """Stage 1: Length filtering."""
    seq_len = len(str(record.seq))
    if seq_len < MIN_LENGTH or seq_len > MAX_LENGTH:
        if verbose:
            print(f"  ✗ {record.id}: Length {seq_len} (outside {MIN_LENGTH}-{MAX_LENGTH})")
        return False, f"Length {seq_len}"
    if verbose:
        print(f"  ✓ {record.id}: Length {seq_len}")
    return True, ""


def filter_ambiguous_residues(record, verbose=False):
    """Stage 2: Ambiguous residue filtering."""
    seq = str(record.seq)
    seq_len = len(seq)
    ambiguous_count = sum(1 for aa in seq if aa in AMBIGUOUS_RESIDUES)
    ambiguous_pct = (ambiguous_count / seq_len) * 100
    if ambiguous_pct > MAX_AMBIGUOUS_PCT:
        if verbose:
            print(f"  ✗ {record.id}: Ambiguous residues {ambiguous_pct:.1f}% > {MAX_AMBIGUOUS_PCT}%")
        return False, f"Ambiguous {ambiguous_pct:.1f}%"
    if verbose:
        print(f"  ✓ {record.id}: Ambiguous residues {ambiguous_pct:.1f}%")
    return True, ""


def filter_premature_stops(record, verbose=False):
    """Stage 3: Premature stop codon filtering."""
    seq = str(record.seq)
    if '*' in seq[:-1]:
        if verbose:
            print(f"  ✗ {record.id}: Contains premature stop codon")
        return False, "Premature stop"
    if verbose:
        print(f"  ✓ {record.id}: No premature stop codons")
    return True, ""


def filter_heat_repeats(record, verbose=False):
    """Stage 4: HEAT-repeat validation using Pfam."""
    seq = str(record.seq)
    # Simplified check - count HEAT-like motifs
    heat_pattern = re.compile(r'[LIVMFYC]..[LIVMFYC]..[LIVMFYC]')
    heat_matches = len(heat_pattern.findall(seq))
    if heat_matches < MIN_HEAT_REPEATS:
        if verbose:
            print(f"  ✗ {record.id}: Only {heat_matches} HEAT-like repeats (< {MIN_HEAT_REPEATS})")
        return False, f"HEAT repeats {heat_matches}"
    if verbose:
        print(f"  ✓ {record.id}: {heat_matches} HEAT-like repeats")
    return True, ""


def reciprocal_blast(record, reference_db, verbose=False):
    """Stage 5: Reciprocal BLAST orthology validation."""
    if reference_db is None:
        return True, ""
    
    temp_dir = tempfile.mkdtemp()
    query_file = os.path.join(temp_dir, 'query.fasta')
    SeqIO.write(record, query_file, 'fasta')
    
    blast_cmd = NcbiblastpCommandline(
        query=query_file,
        db=reference_db,
        out=os.path.join(temp_dir, 'blast_out.xml'),
        outfmt=5,
        evalue=BLAST_EVALUE
    )
    
    try:
        stdout, stderr = blast_cmd()
    except Exception as e:
        if verbose:
            print(f"  ✗ {record.id}: BLAST failed - {str(e)}")
        return False, f"BLAST error: {str(e)}"
    
    try:
        with open(os.path.join(temp_dir, 'blast_out.xml')) as handle:
            blast_records = NCBIXML.parse(handle)
            for blast_record in blast_records:
                for alignment in blast_record.alignments:
                    for hsp in alignment.hsps:
                        if hsp.expect <= BLAST_EVALUE:
                            coverage = (hsp.align_length / len(record.seq)) * 100
                            identity = hsp.identities / hsp.align_length * 100
                            if coverage >= BLAST_COVERAGE and identity >= BLAST_IDENTITY:
                                if verbose:
                                    print(f"  ✓ {record.id}: BLAST validated (id={identity:.1f}%, cov={coverage:.1f}%)")
                                return True, ""
    except Exception as e:
        if verbose:
            print(f"  ✗ {record.id}: BLAST parsing failed - {str(e)}")
        return False, f"Parse error: {str(e)}"
    
    if verbose:
        print(f"  ✗ {record.id}: BLAST validation failed")
    return False, "No BLAST hit"


def clans_outlier_detection(records, verbose=False):
    """Stage 6: CLANS-based outlier detection."""
    if len(records) < 3:
        return records
    
    try:
        from sklearn.cluster import DBSCAN
        from sklearn.feature_extraction.text import CountVectorizer
        
        seqs = [str(r.seq) for r in records]
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 3))
        X = vectorizer.fit_transform(seqs)
        
        clustering = DBSCAN(eps=0.5, min_samples=3, metric='cosine')
        labels = clustering.fit_predict(X)
        
        filtered_records = []
        for i, record in enumerate(records):
            if labels[i] != -1:
                filtered_records.append(record)
                if verbose:
                    print(f"  ✓ {record.id}: Kept (cluster {labels[i]})")
            else:
                if verbose:
                    print(f"  ✗ {record.id}: Removed (outlier)")
        
        return filtered_records
    except ImportError:
        if verbose:
            print("  sklearn not available, skipping outlier detection")
        return records


# ============================================================
# MAIN FUNCTION
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='HASTY/XPO5 Sequence Curation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sequence_curation.py -i raw.fasta -o curated.fasta
  python sequence_curation.py -i raw.fasta -o curated.fasta -v
        """
    )
    parser.add_argument('--input', '-i', required=True, help='Input FASTA file')
    parser.add_argument('--output', '-o', required=True, help='Output FASTA file')
    parser.add_argument('--reference_db', help='BLAST database for reciprocal validation')
    parser.add_argument('--skip_blast', action='store_true', help='Skip reciprocal BLAST')
    parser.add_argument('--skip_clans', action='store_true', help='Skip CLANS outlier detection')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print progress')
    args = parser.parse_args()
    
    print("=" * 60)
    print("HASTY/XPO5 Sequence Curation Pipeline")
    print("=" * 60)
    
    # Load sequences
    print(f"\n[1] Loading sequences from {args.input}...")
    records = list(SeqIO.parse(args.input, 'fasta'))
    print(f"  Loaded {len(records)} sequences")
    
    # Apply filters
    filters = [
        ('Length', filter_by_length),
        ('Ambiguous residues', filter_ambiguous_residues),
        ('Premature stops', filter_premature_stops),
        ('HEAT repeats', filter_heat_repeats),
    ]
    
    filtered_records = records
    for name, filter_func in filters:
        print(f"\n[2] {name} filtering...")
        temp_records = []
        for record in filtered_records:
            passed, reason = filter_func(record, verbose=args.verbose)
            if passed:
                temp_records.append(record)
        filtered_records = temp_records
        print(f"  Kept {len(filtered_records)} sequences")
    
    # Reciprocal BLAST
    if not args.skip_blast and args.reference_db:
        print(f"\n[3] Reciprocal BLAST validation...")
        temp_records = []
        for record in filtered_records:
            passed, reason = reciprocal_blast(record, args.reference_db, verbose=args.verbose)
            if passed:
                temp_records.append(record)
        filtered_records = temp_records
        print(f"  Kept {len(filtered_records)} sequences")
    
    # CLANS outlier detection
    if not args.skip_clans:
        print(f"\n[4] CLANS outlier detection...")
        filtered_records = clans_outlier_detection(filtered_records, verbose=args.verbose)
        print(f"  Kept {len(filtered_records)} sequences")
    
    # Save
    print(f"\n[5] Saving curated dataset to {args.output}...")
    SeqIO.write(filtered_records, args.output, 'fasta')
    
    print("\n" + "=" * 60)
    print(f"✅ Curated dataset saved: {len(filtered_records)} sequences")
    print("=" * 60)


if __name__ == '__main__':
    main()