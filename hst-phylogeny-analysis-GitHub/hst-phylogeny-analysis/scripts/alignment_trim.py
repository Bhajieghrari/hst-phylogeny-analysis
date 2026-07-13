#!/usr/bin/env python3
"""
Multiple Sequence Alignment and Trimming Pipeline

Generates MSA using 5 different algorithms and selects the best based on quality metrics.

Usage:
    python alignment_trim.py --input curated_sequences.fasta --output aligned_trimmed.fasta
    python alignment_trim.py -i curated.fasta -o aligned.fasta -v
"""

import argparse
import os
import sys
import subprocess
import tempfile
from Bio import AlignIO, SeqIO
from Bio.Align import MultipleSeqAlignment
import numpy as np
import pandas as pd
import math


# ============================================================
# ALIGNMENT TOOLS
# ============================================================
ALIGNMENT_TOOLS = {
    'mafft': {
        'cmd': 'mafft --auto --thread -1 {input} > {output}',
        'description': 'MAFFT (FFT-NS-2)'
    },
    'muscle': {
        'cmd': 'muscle -in {input} -out {output}',
        'description': 'MUSCLE'
    },
    'clustalo': {
        'cmd': 'clustalo -i {input} -o {output} --threads -1',
        'description': 'Clustal Omega'
    },
    'tcoffee': {
        'cmd': 't_coffee {input} -output fasta_aln -outfile {output}',
        'description': 'T-Coffee'
    },
    'kalign': {
        'cmd': 'kalign -i {input} -o {output}',
        'description': 'Kalign'
    }
}


# ============================================================
# FUNCTIONS
# ============================================================
def run_alignment(input_file, output_file, tool, verbose=False):
    """Run alignment using specified tool."""
    if tool not in ALIGNMENT_TOOLS:
        print(f"  Warning: Unknown tool '{tool}', skipping")
        return False
    
    cmd = ALIGNMENT_TOOLS[tool]['cmd'].format(input=input_file, output=output_file)
    if verbose:
        print(f"  Running {tool}...")
    
    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        if verbose:
            print(f"  Error running {tool}: {e.stderr.decode() if e.stderr else str(e)}")
        return False


def calculate_alignment_metrics(alignment_file, verbose=False):
    """Calculate quality metrics for an alignment."""
    try:
        alignment = AlignIO.read(alignment_file, 'fasta')
    except Exception as e:
        if verbose:
            print(f"  Error reading alignment: {e}")
        return None
    
    seq_count = len(alignment)
    alignment_length = alignment.get_alignment_length()
    
    # Gap ratio
    gap_counts = []
    for i in range(alignment_length):
        column = alignment[:, i]
        gaps = sum(1 for c in column if c == '-')
        gap_counts.append(gaps)
    total_gap_ratio = sum(gap_counts) / (seq_count * alignment_length) if seq_count > 0 else 0
    
    # Entropy
    entropies = []
    for i in range(alignment_length):
        column = alignment[:, i]
        aa_counts = {}
        for aa in column:
            if aa != '-':
                aa_counts[aa] = aa_counts.get(aa, 0) + 1
        if aa_counts:
            total = sum(aa_counts.values())
            entropy = -sum((count/total) * math.log2(count/total) for count in aa_counts.values())
            entropies.append(entropy)
    avg_entropy = np.mean(entropies) if entropies else 0
    
    # SP-score (simplified)
    sequences = [str(record.seq) for record in alignment]
    sp_score = 0
    comparisons = 0
    for i in range(len(sequences)):
        for j in range(i+1, len(sequences)):
            seq1 = sequences[i]
            seq2 = sequences[j]
            matches = sum(1 for a, b in zip(seq1, seq2) if a == b and a != '-')
            sp_score += matches
            comparisons += 1
    sp_score = sp_score / comparisons if comparisons > 0 else 0
    
    return {
        'seq_count': seq_count,
        'alignment_length': alignment_length,
        'gap_ratio': total_gap_ratio,
        'avg_entropy': avg_entropy,
        'sp_score': sp_score
    }


def trim_alignment(input_file, output_file, verbose=False):
    """Trim alignment using TrimAl."""
    cmd = f"trimal -in {input_file} -out {output_file} -gappyout -cons 60 -automated1"
    if verbose:
        print(f"  Trimming alignment with TrimAl...")
    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        if verbose:
            print(f"  Error trimming alignment: {e.stderr.decode() if e.stderr else str(e)}")
        return False


def select_best_alignment(alignment_dir, verbose=False):
    """Select the best alignment based on quality metrics."""
    metrics = []
    
    for tool in ALIGNMENT_TOOLS.keys():
        aln_file = os.path.join(alignment_dir, f"{tool}.aln")
        if os.path.exists(aln_file):
            metric = calculate_alignment_metrics(aln_file, verbose)
            if metric:
                metric['tool'] = tool
                metrics.append(metric)
    
    if not metrics:
        return None
    
    df = pd.DataFrame(metrics)
    if verbose:
        print("\n=== Alignment Quality Metrics ===")
        print(df.to_string(index=False))
    
    # Score: higher SP-score, lower gap_ratio and entropy
    df['score'] = (df['sp_score'] * 0.4) - (df['gap_ratio'] * 0.3) - (df['avg_entropy'] * 0.3)
    best_tool = df.loc[df['score'].idxmax(), 'tool']
    
    if verbose:
        print(f"\n✅ Best alignment: {best_tool} (score={df['score'].max():.2f})")
    
    return best_tool


# ============================================================
# MAIN FUNCTION
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Multiple Sequence Alignment and Trimming Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python alignment_trim.py -i curated.fasta -o aligned.fasta
  python alignment_trim.py -i curated.fasta -o aligned.fasta -v
        """
    )
    parser.add_argument('--input', '-i', required=True, help='Input FASTA file')
    parser.add_argument('--output', '-o', required=True, help='Output aligned FASTA file')
    parser.add_argument('--tools', nargs='+', default=list(ALIGNMENT_TOOLS.keys()), 
                       help='Alignment tools to use')
    parser.add_argument('--skip_trimming', action='store_true', help='Skip trimming')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print progress')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Multiple Sequence Alignment and Trimming Pipeline")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp()
    
    # Run alignments
    print(f"\n[1] Running alignments...")
    for tool in args.tools:
        output_file = os.path.join(temp_dir, f"{tool}.aln")
        run_alignment(args.input, output_file, tool, verbose=args.verbose)
    
    # Select best alignment
    print(f"\n[2] Selecting best alignment...")
    best_tool = select_best_alignment(temp_dir, verbose=args.verbose)
    if not best_tool:
        print("Error: No alignment succeeded")
        sys.exit(1)
    
    best_file = os.path.join(temp_dir, f"{best_tool}.aln")
    
    # Trim alignment
    if not args.skip_trimming:
        print(f"\n[3] Trimming alignment...")
        trimmed_file = os.path.join(temp_dir, "trimmed.aln")
        trim_alignment(best_file, trimmed_file, verbose=args.verbose)
        final_file = trimmed_file
    else:
        final_file = best_file
    
    # Copy to output
    with open(final_file, 'r') as src, open(args.output, 'w') as dst:
        dst.write(src.read())
    
    print(f"\n✅ Alignment saved to {args.output}")
    print(f"   Tool used: {best_tool}")
    if not args.skip_trimming:
        print(f"   Trimmed: Yes (TrimAl -gappyout -cons 60)")


if __name__ == '__main__':
    main()