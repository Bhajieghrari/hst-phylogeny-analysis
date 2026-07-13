#!/usr/bin/env python3
"""
Motif Discovery and Analysis Pipeline using MEME Suite

Usage:
    python motif_analysis.py --input aligned_trimmed.fasta --output motif_results/
    python motif_analysis.py -i aligned.fasta -o motifs/ -v
"""

import argparse
import os
import sys
import subprocess
import tempfile
import pandas as pd
import numpy as np
import re
import xml.etree.ElementTree as ET
from Bio import SeqIO
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# FUNCTIONS
# ============================================================
def run_meme(input_file, output_dir, max_motifs=20, min_width=6, max_width=50, verbose=False):
    """Run MEME motif discovery."""
    if verbose:
        print(f"  Running MEME...")
    
    cmd = f"meme {input_file} -protein -nmotifs {max_motifs} -minw {min_width} -maxw {max_width} -oc {output_dir}"
    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        if verbose:
            print(f"  Error running MEME: {e.stderr.decode() if e.stderr else str(e)}")
        return False


def parse_meme_output(meme_dir, verbose=False):
    """Parse MEME output and extract motif information."""
    motif_file = os.path.join(meme_dir, 'meme.xml')
    if not os.path.exists(motif_file):
        if verbose:
            print(f"  Error: MEME output not found in {meme_dir}")
        return None
    
    try:
        tree = ET.parse(motif_file)
        root = tree.getroot()
        
        motifs = []
        for motif in root.findall('.//motif'):
            motif_id = motif.get('id')
            width = int(motif.get('width'))
            sites = int(motif.get('sites'))
            evalue = float(motif.get('evalue'))
            
            consensus = motif.find('consensus')
            consensus_seq = consensus.text if consensus is not None else ''
            
            motifs.append({
                'id': motif_id,
                'width': width,
                'sites': sites,
                'evalue': evalue,
                'consensus': consensus_seq
            })
        
        motifs.sort(key=lambda x: x['evalue'])
        if verbose:
            print(f"  Found {len(motifs)} motifs")
        
        return motifs
    except Exception as e:
        if verbose:
            print(f"  Error parsing MEME output: {e}")
        return None


def create_motif_matrix(sequences, motifs, verbose=False):
    """Create presence/absence matrix of motifs in sequences."""
    seq_records = list(SeqIO.parse(sequences, 'fasta'))
    seq_ids = [str(r.id) for r in seq_records]
    
    matrix = []
    for record in seq_records:
        seq = str(record.seq)
        row = {}
        for motif in motifs:
            # Simple motif matching
            pattern = motif['consensus'].replace('?', '.').replace('[', '[').replace(']', ']')
            found = len(re.findall(pattern, seq)) > 0
            row[motif['id']] = 1 if found else 0
        matrix.append(row)
    
    df = pd.DataFrame(matrix, index=seq_ids)
    return df


def hierarchical_clustering_motifs(motif_matrix, verbose=False):
    """Perform hierarchical clustering based on motif presence/absence."""
    X = motif_matrix.values
    
    # Jaccard distance
    n = X.shape[0]
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            intersection = np.sum(np.logical_and(X[i], X[j]))
            union = np.sum(np.logical_or(X[i], X[j]))
            distance_matrix[i, j] = 1 - (intersection / union) if union > 0 else 1
            distance_matrix[j, i] = distance_matrix[i, j]
    
    linkage_matrix = linkage(distance_matrix, method='average')
    
    if verbose:
        print(f"  Hierarchical clustering completed")
    
    return linkage_matrix, distance_matrix


def pca_motif_analysis(motif_matrix, verbose=False):
    """Perform PCA on motif presence/absence matrix."""
    X = motif_matrix.values
    X_scaled = StandardScaler().fit_transform(X)
    
    pca = PCA(n_components=min(10, X.shape[0], X.shape[1]))
    pca_result = pca.fit_transform(X_scaled)
    
    if verbose:
        explained_variance = pca.explained_variance_ratio_
        print(f"  PCA completed: PC1={explained_variance[0]:.2%}, PC2={explained_variance[1]:.2%}")
    
    return pca_result, pca.explained_variance_ratio_


def plot_motif_heatmap(motif_matrix, output_file, verbose=False):
    """Plot heatmap of motif presence/absence."""
    plt.figure(figsize=(20, 40))
    sns.heatmap(motif_matrix, cmap=['white', 'blue'], cbar_kws={'label': 'Motif Present'})
    plt.title('Motif Presence/Absence Heatmap')
    plt.xlabel('Motifs')
    plt.ylabel('Sequences')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    if verbose:
        print(f"  Heatmap saved to {output_file}")


def plot_dendrogram(linkage_matrix, labels, output_file, verbose=False):
    """Plot dendrogram for motif clustering."""
    plt.figure(figsize=(20, 40))
    dendrogram(linkage_matrix, labels=labels, orientation='left', leaf_font_size=6)
    plt.title('Hierarchical Clustering Based on Motif Profiles')
    plt.xlabel('Jaccard Distance')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    if verbose:
        print(f"  Dendrogram saved to {output_file}")


# ============================================================
# MAIN FUNCTION
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Motif Discovery and Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python motif_analysis.py -i aligned.fasta -o motifs/
  python motif_analysis.py -i aligned.fasta -o motifs/ -v
        """
    )
    parser.add_argument('--input', '-i', required=True, help='Input aligned FASTA file')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    parser.add_argument('--max_motifs', type=int, default=20, help='Maximum number of motifs')
    parser.add_argument('--min_width', type=int, default=6, help='Minimum motif width')
    parser.add_argument('--max_width', type=int, default=50, help='Maximum motif width')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print progress')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    print("=" * 60)
    print("Motif Discovery and Analysis Pipeline")
    print("=" * 60)
    
    # Run MEME
    print(f"\n[1] Running MEME motif discovery...")
    meme_dir = os.path.join(args.output, 'meme')
    run_meme(args.input, meme_dir, args.max_motifs, args.min_width, args.max_width, verbose=args.verbose)
    
    # Parse MEME output
    print(f"\n[2] Parsing MEME output...")
    motifs = parse_meme_output(meme_dir, verbose=args.verbose)
    if not motifs:
        print("Error: No motifs found")
        sys.exit(1)
    
    # Create motif matrix
    print(f"\n[3] Creating motif presence/absence matrix...")
    motif_matrix = create_motif_matrix(args.input, motifs, verbose=args.verbose)
    matrix_file = os.path.join(args.output, 'motif_matrix.csv')
    motif_matrix.to_csv(matrix_file)
    print(f"  Matrix saved to {matrix_file}")
    
    # Hierarchical clustering
    print(f"\n[4] Performing hierarchical clustering...")
    linkage_matrix, distance_matrix = hierarchical_clustering_motifs(motif_matrix, verbose=args.verbose)
    
    # PCA analysis
    print(f"\n[5] Performing PCA analysis...")
    pca_result, explained_variance = pca_motif_analysis(motif_matrix, verbose=args.verbose)
    pca_file = os.path.join(args.output, 'pca_results.csv')
    pca_df = pd.DataFrame(pca_result[:, :2], columns=['PC1', 'PC2'])
    pca_df.index = motif_matrix.index
    pca_df.to_csv(pca_file)
    print(f"  PCA results saved to {pca_file}")
    
    # Generate plots
    print(f"\n[6] Generating plots...")
    heatmap_file = os.path.join(args.output, 'motif_heatmap.png')
    plot_motif_heatmap(motif_matrix, heatmap_file, verbose=args.verbose)
    
    dendrogram_file = os.path.join(args.output, 'motif_dendrogram.png')
    plot_dendrogram(linkage_matrix, motif_matrix.index.tolist(), dendrogram_file, verbose=args.verbose)
    
    print(f"\n✅ Motif analysis completed. Results in {args.output}")


if __name__ == '__main__':
    main()