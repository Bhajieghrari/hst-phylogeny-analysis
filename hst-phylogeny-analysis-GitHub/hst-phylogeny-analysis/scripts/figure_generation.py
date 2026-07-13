#!/usr/bin/env python3
"""
Publication-Ready Figure Generation Pipeline

Generates all figures for the manuscript from analysis results.

Usage:
    python figure_generation.py --results results/ --output figures/
    python figure_generation.py -r results/ -o figures/ -v
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import Phylo
from scipy.cluster.hierarchy import linkage, dendrogram
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


# ============================================================
# FIGURE GENERATION FUNCTIONS
# ============================================================
def generate_figure1_motif_clustering(results_dir, output_dir, verbose=False):
    """Figure 1: Motif-based hierarchical clustering."""
    motif_file = os.path.join(results_dir, 'motifs', 'motif_matrix.csv')
    if not os.path.exists(motif_file):
        if verbose:
            print(f"    Warning: {motif_file} not found")
        return
    
    motif_df = pd.read_csv(motif_file, index_col=0)
    
    # Calculate distance matrix
    X = motif_df.values
    n = X.shape[0]
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            intersection = np.sum(np.logical_and(X[i], X[j]))
            union = np.sum(np.logical_or(X[i], X[j]))
            distance_matrix[i, j] = 1 - (intersection / union) if union > 0 else 1
            distance_matrix[j, i] = distance_matrix[i, j]
    
    linkage_matrix = linkage(distance_matrix, method='average')
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 15))
    
    # Heatmap
    sns.heatmap(motif_df, cmap='Blues', cbar_kws={'label': 'Motif Present'}, ax=axes[0])
    axes[0].set_title('Motif Presence/Absence Heatmap', fontsize=14, fontweight='bold')
    
    # Dendrogram
    dendrogram(linkage_matrix, labels=motif_df.index, orientation='left', ax=axes[1],
               leaf_font_size=6)
    axes[1].set_title('Hierarchical Clustering (UPGMA-like)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Jaccard Distance')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Figure_1.png'), dpi=300, bbox_inches='tight')
    plt.close()
    if verbose:
        print("  ✓ Figure 1: Motif hierarchical clustering")


def generate_figure2_motif_heatmap(results_dir, output_dir, verbose=False):
    """Figure 2: Motif distribution heatmap."""
    motif_file = os.path.join(results_dir, 'motifs', 'motif_matrix.csv')
    if not os.path.exists(motif_file):
        if verbose:
            print(f"    Warning: {motif_file} not found")
        return
    
    motif_df = pd.read_csv(motif_file, index_col=0)
    
    fig, ax = plt.subplots(figsize=(15, 20))
    sns.heatmap(motif_df, cmap=['white', '#1f77b4'], cbar=False, ax=ax)
    ax.set_title('Motif Distribution Across HST/XPO5 Proteins', fontsize=14, fontweight='bold')
    ax.set_xlabel('Motifs', fontsize=12)
    ax.set_ylabel('Protein Sequences', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Figure_2.png'), dpi=300, bbox_inches='tight')
    plt.close()
    if verbose:
        print("  ✓ Figure 2: Motif distribution heatmap")


def generate_figure3_phylogeny(results_dir, output_dir, verbose=False):
    """Figure 3: ML phylogeny."""
    tree_file = None
    for f in os.listdir(os.path.join(results_dir, 'phylogeny')):
        if f.endswith('.treefile') or f.endswith('.tree'):
            tree_file = os.path.join(results_dir, 'phylogeny', f)
            break
    
    if not tree_file:
        if verbose:
            print("    Warning: Tree file not found")
        return
    
    try:
        tree = Phylo.read(tree_file, 'newick')
        fig, ax = plt.subplots(figsize=(15, 20))
        Phylo.draw(tree, axes=ax, do_show=False)
        ax.set_title('Maximum-Likelihood Phylogeny', fontsize=14, fontweight='bold')
        ax.set_xlabel('Substitutions per site', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'Figure_3.png'), dpi=300, bbox_inches='tight')
        plt.close()
        if verbose:
            print("  ✓ Figure 3: ML phylogeny")
    except Exception as e:
        if verbose:
            print(f"    Error: {e}")


def generate_figure6_rmsd_heatmap(results_dir, output_dir, verbose=False):
    """Figure 6: RMSD heatmap."""
    rmsd_file = os.path.join(results_dir, 'structures', 'rmsd_matrix.csv')
    if not os.path.exists(rmsd_file):
        if verbose:
            print("    Warning: RMSD matrix not found")
        return
    
    rmsd_df = pd.read_csv(rmsd_file, index_col=0)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(rmsd_df, annot=True, fmt='.1f', cmap='RdYlBu_r',
                cbar_kws={'label': 'RMSD (Å)'}, ax=ax)
    ax.set_title('Pairwise RMSD Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('Protein Models', fontsize=12)
    ax.set_ylabel('Protein Models', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Figure_6.png'), dpi=300, bbox_inches='tight')
    plt.close()
    if verbose:
        print("  ✓ Figure 6: RMSD heatmap")


def generate_figure7_rmsf_plot(results_dir, output_dir, verbose=False):
    """Figure 7: RMSF plot."""
    rmsf_file = os.path.join(results_dir, 'structures', 'rmsf_values.csv')
    if not os.path.exists(rmsf_file):
        if verbose:
            print("    Warning: RMSF values not found")
        return
    
    rmsf_df = pd.read_csv(rmsf_file)
    rmsf_values = rmsf_df['RMSF'].values
    
    fig, ax = plt.subplots(figsize=(15, 6))
    
    p25 = np.percentile(rmsf_values, 25)
    p75 = np.percentile(rmsf_values, 75)
    colors = ['blue' if r < p25 else 'red' if r > p75 else 'grey' for r in rmsf_values]
    
    ax.scatter(range(len(rmsf_values)), rmsf_values, c=colors, alpha=0.6, s=10)
    ax.axhline(y=p25, color='blue', linestyle='--', label='25th percentile')
    ax.axhline(y=p75, color='red', linestyle='--', label='75th percentile')
    ax.set_xlabel('Residue Position', fontsize=12)
    ax.set_ylabel('RMSF (Å)', fontsize=12)
    ax.set_title('Residue-Specific RMSF', fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Figure_7.png'), dpi=300, bbox_inches='tight')
    plt.close()
    if verbose:
        print("  ✓ Figure 7: RMSF plot")


def generate_figure8_clashes(results_dir, output_dir, verbose=False):
    """Figure 8: Steric clashes bar plot."""
    clash_file = os.path.join(results_dir, 'structures', 'steric_clashes.csv')
    if not os.path.exists(clash_file):
        if verbose:
            print("    Warning: Clash data not found")
        return
    
    clash_df = pd.read_csv(clash_file)
    clash_df['Clashes_per_100'] = (clash_df['Total_Clashes'] / 1000) * 100
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(clash_df))
    width = 0.35
    
    ax.bar(x - width/2, clash_df['Total_Clashes'], width, color='royalblue', label='Total Clashes')
    ax.bar(x + width/2, clash_df['Clashes_per_100'], width, color='coral', label='Clashes per 100 residues')
    ax.set_xlabel('Protein Models', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Steric Clash Analysis', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(clash_df['Model'], rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Figure_8.png'), dpi=300, bbox_inches='tight')
    plt.close()
    if verbose:
        print("  ✓ Figure 8: Steric clashes")


def generate_figure9_cluster_dendrogram(results_dir, output_dir, verbose=False):
    """Figure 9: Structural clustering dendrogram."""
    rmsd_file = os.path.join(results_dir, 'structures', 'rmsd_matrix.csv')
    if not os.path.exists(rmsd_file):
        if verbose:
            print("    Warning: RMSD matrix not found")
        return
    
    rmsd_df = pd.read_csv(rmsd_file, index_col=0)
    linkage_matrix = linkage(rmsd_df.values, method='average')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    dendrogram(linkage_matrix, labels=rmsd_df.columns, orientation='top',
               leaf_rotation=90, leaf_font_size=8, ax=ax)
    ax.set_title('Structural Clustering', fontsize=14, fontweight='bold')
    ax.set_xlabel('Protein Models', fontsize=12)
    ax.set_ylabel('Cα RMSD Distance (Å)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Figure_9.png'), dpi=300, bbox_inches='tight')
    plt.close()
    if verbose:
        print("  ✓ Figure 9: Structural clustering")


# ============================================================
# MAIN FUNCTION
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Figure Generation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python figure_generation.py -r results/ -o figures/
  python figure_generation.py -r results/ -o figures/ -v
        """
    )
    parser.add_argument('--results', '-r', required=True, help='Results directory')
    parser.add_argument('--output', '-o', required=True, help='Output directory for figures')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print progress')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    print("=" * 60)
    print("Publication-Ready Figure Generation")
    print("=" * 60)
    
    figures = [
        (generate_figure1_motif_clustering, 'Figure 1: Motif hierarchical clustering'),
        (generate_figure2_motif_heatmap, 'Figure 2: Motif distribution heatmap'),
        (generate_figure3_phylogeny, 'Figure 3: ML phylogeny'),
        (generate_figure6_rmsd_heatmap, 'Figure 6: RMSD heatmap'),
        (generate_figure7_rmsf_plot, 'Figure 7: RMSF plot'),
        (generate_figure8_clashes, 'Figure 8: Steric clashes'),
        (generate_figure9_cluster_dendrogram, 'Figure 9: Structural clustering'),
    ]
    
    for func, desc in figures:
        print(f"\n[Generating] {desc}")
        func(args.results, args.output, verbose=args.verbose)
    
    print(f"\n✅ All figures generated. Output in {args.output}")


if __name__ == '__main__':
    main()