#!/usr/bin/env python3
"""
Structural Comparison and Analysis Pipeline

Compares 3D structures using RMSD, RMSF, and clustering.

Usage:
    python structural_compare.py --models structures/ --output results/
    python structural_compare.py -m structures/ -o results/ -v
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from Bio import PDB
from Bio.PDB import PDBParser, Superimposer, PPBuilder, NeighborSearch
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# FUNCTIONS
# ============================================================
def parse_structure(pdb_file):
    """Parse PDB structure and extract Cα atoms."""
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('id', pdb_file)
        model = structure[0]
        ca_atoms = []
        for chain in model:
            for residue in chain:
                if PDB.is_aa(residue):
                    try:
                        ca = residue['CA']
                        ca_atoms.append(ca)
                    except KeyError:
                        continue
        return structure, ca_atoms
    except Exception as e:
        return None, None


def calculate_rmsd(structures, structure_names, verbose=False):
    """Calculate pairwise RMSD between all structures."""
    n = len(structures)
    rmsd_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            ca_i = [atom for atom in structures[i] if atom.get_id() == 'CA']
            ca_j = [atom for atom in structures[j] if atom.get_id() == 'CA']
            
            min_len = min(len(ca_i), len(ca_j))
            if min_len > 10:
                sup = Superimposer()
                sup.set_atoms(ca_i[:min_len], ca_j[:min_len])
                rmsd_matrix[i, j] = sup.rms
                rmsd_matrix[j, i] = sup.rms
            else:
                rmsd_matrix[i, j] = 100.0
                rmsd_matrix[j, i] = 100.0
    
    rmsd_df = pd.DataFrame(rmsd_matrix, index=structure_names, columns=structure_names)
    
    if verbose:
        print(f"  RMSD matrix calculated: {rmsd_matrix.min():.2f} - {rmsd_matrix.max():.2f} Å")
    
    return rmsd_df


def calculate_rmsf(structures, verbose=False):
    """Calculate RMSF (positional variability) across structures."""
    # Align all structures to the first one
    ref_structure = structures[0]
    aligned_structures = []
    
    for i, structure in enumerate(structures):
        if i == 0:
            aligned_structures.append(structure)
            continue
        
        sup = Superimposer()
        ref_ca = [atom for atom in ref_structure if atom.get_id() == 'CA']
        target_ca = [atom for atom in structure if atom.get_id() == 'CA']
        
        min_len = min(len(ref_ca), len(target_ca))
        if min_len > 10:
            sup.set_atoms(ref_ca[:min_len], target_ca[:min_len])
            # BUGFIX: `structure` is already a plain list of Cα Atom objects
            # (built in main() via `structures.append(ca_atoms)`), not a
            # Bio.PDB Structure/Model/Chain object. Lists have no
            # `.get_atoms()` method, so the original call
            # `sup.apply(structure.get_atoms())` raised an AttributeError
            # for every structure after the first one. Superimposer.apply()
            # accepts any iterable of Atom objects, so we pass the list
            # directly.
            sup.apply(structure)
        aligned_structures.append(structure)
    
    # Calculate RMSF
    common_positions = []
    min_len = min(len([a for a in s if a.get_id() == 'CA']) for s in structures)
    
    for pos in range(min_len):
        positions = []
        for s in structures:
            ca_atoms = [a for a in s if a.get_id() == 'CA']
            if pos < len(ca_atoms):
                positions.append(ca_atoms[pos].get_coord())
        
        if len(positions) > 1:
            std_dev = np.std(positions, axis=0)
            rmsf = np.sqrt(np.mean(std_dev**2))
            common_positions.append(rmsf)
        else:
            common_positions.append(0)
    
    rmsf_values = np.array(common_positions)
    
    if verbose:
        print(f"  RMSF calculated: {rmsf_values.min():.2f} - {rmsf_values.max():.2f} Å")
    
    return rmsf_values


def calculate_steric_clashes(structure):
    """Calculate number of steric clashes in a structure.

    BUGFIX: the original implementation only counted a clash between two
    atoms if they belonged to *different chains* (it compared
    `atom.get_parent().get_parent()`, i.e. residue -> chain). For a
    monomeric (single-chain) protein model -- the common case for these
    HASTY/XPO5 models -- every atom belongs to the same chain, so the old
    code always returned 0 regardless of any real clashes present.

    This version instead excludes clashes between atoms in the same
    residue or in immediately sequence-adjacent residues (normal bonded
    neighbors), while still counting genuine steric clashes within a
    chain. It also uses Bio.PDB's NeighborSearch (KD-tree) instead of an
    O(n^2) nested Python loop, which is both correct and dramatically
    faster for larger structures.
    """
    atoms = list(structure.get_atoms())
    clash_threshold = 2.0

    ns = NeighborSearch(atoms)
    close_pairs = ns.search_all(clash_threshold, level='A')

    clash_count = 0
    for atom1, atom2 in close_pairs:
        res1 = atom1.get_parent()
        res2 = atom2.get_parent()
        if res1 is res2:
            continue  # same residue, not a clash

        chain1 = res1.get_parent()
        chain2 = res2.get_parent()
        if chain1 is chain2:
            try:
                seq_sep = abs(res1.get_id()[1] - res2.get_id()[1])
            except (TypeError, IndexError):
                seq_sep = None
            if seq_sep is not None and seq_sep <= 1:
                continue  # bonded/adjacent residue, not a clash

        clash_count += 1

    return clash_count


def structural_clustering(rmsd_matrix, n_clusters=4):
    """Perform hierarchical clustering based on RMSD."""
    distance_matrix = rmsd_matrix.values
    linkage_matrix = linkage(distance_matrix, method='average')
    
    from scipy.cluster.hierarchy import fcluster
    clusters = fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')
    
    return linkage_matrix, clusters


def plot_rmsd_heatmap(rmsd_matrix, output_file, verbose=False):
    """Plot RMSD heatmap."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(rmsd_matrix, annot=True, fmt='.1f', cmap='RdYlBu_r',
                cbar_kws={'label': 'RMSD (Å)'})
    plt.title('Pairwise RMSD Heatmap')
    plt.xlabel('Protein Models')
    plt.ylabel('Protein Models')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    if verbose:
        print(f"  RMSD heatmap saved to {output_file}")


def plot_rmsf(rmsf_values, output_file, verbose=False):
    """Plot RMSF values."""
    plt.figure(figsize=(15, 6))
    
    p25 = np.percentile(rmsf_values, 25)
    p75 = np.percentile(rmsf_values, 75)
    colors = ['blue' if r < p25 else 'red' if r > p75 else 'grey' for r in rmsf_values]
    
    plt.scatter(range(len(rmsf_values)), rmsf_values, c=colors, alpha=0.6, s=10)
    plt.axhline(y=p25, color='blue', linestyle='--', label='25th percentile')
    plt.axhline(y=p75, color='red', linestyle='--', label='75th percentile')
    plt.xlabel('Residue Position')
    plt.ylabel('RMSF (Å)')
    plt.title('Residue-Specific RMSF')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    if verbose:
        print(f"  RMSF plot saved to {output_file}")


def plot_cluster_dendrogram(linkage_matrix, structure_names, output_file, verbose=False):
    """Plot cluster dendrogram."""
    plt.figure(figsize=(12, 8))
    dendrogram(linkage_matrix, labels=structure_names, orientation='top',
               leaf_rotation=90, leaf_font_size=8)
    plt.title('Structural Clustering')
    plt.xlabel('Protein Models')
    plt.ylabel('Cα RMSD Distance (Å)')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    if verbose:
        print(f"  Cluster dendrogram saved to {output_file}")


# ============================================================
# MAIN FUNCTION
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Structural Comparison Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python structural_compare.py -m structures/ -o results/
  python structural_compare.py -m structures/ -o results/ -v
        """
    )
    parser.add_argument('--models', '-m', required=True, help='Directory containing PDB models')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    parser.add_argument('--n_clusters', type=int, default=4, help='Number of clusters')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print progress')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    print("=" * 60)
    print("Structural Comparison and Analysis Pipeline")
    print("=" * 60)
    
    # Find PDB files
    pdb_files = [f for f in os.listdir(args.models) if f.endswith('.pdb')]
    pdb_files.sort()
    
    if not pdb_files:
        print(f"Error: No PDB files found in {args.models}")
        sys.exit(1)
    
    print(f"\n[1] Found {len(pdb_files)} PDB models")
    
    # Parse structures
    print(f"\n[2] Parsing structures...")
    structures = []
    structure_names = []
    
    for pdb_file in pdb_files:
        pdb_path = os.path.join(args.models, pdb_file)
        name = os.path.splitext(pdb_file)[0]
        structure, ca_atoms = parse_structure(pdb_path)
        if structure and ca_atoms:
            structures.append(ca_atoms)
            structure_names.append(name)
            if args.verbose:
                print(f"  ✓ {name}: {len(ca_atoms)} Cα atoms")
    
    if len(structures) < 2:
        print("Error: At least 2 structures required")
        sys.exit(1)
    
    # Calculate RMSD
    print(f"\n[3] Calculating pairwise RMSD...")
    rmsd_matrix = calculate_rmsd(structures, structure_names, verbose=args.verbose)
    rmsd_matrix.to_csv(os.path.join(args.output, 'rmsd_matrix.csv'))
    
    # Calculate RMSF
    print(f"\n[4] Calculating RMSF...")
    rmsf_values = calculate_rmsf(structures, verbose=args.verbose)
    pd.DataFrame({'RMSF': rmsf_values}).to_csv(os.path.join(args.output, 'rmsf_values.csv'))
    
    # Calculate steric clashes
    print(f"\n[5] Analyzing steric clashes...")
    clash_data = []
    for pdb_file, structure in zip(pdb_files, structures):
        pdb_path = os.path.join(args.models, pdb_file)
        struct, _ = parse_structure(pdb_path)
        if struct:
            clash_count = calculate_steric_clashes(struct)
            name = os.path.splitext(pdb_file)[0]
            clash_data.append({'Model': name, 'Total_Clashes': clash_count})
    pd.DataFrame(clash_data).to_csv(os.path.join(args.output, 'steric_clashes.csv'), index=False)
    
    # Structural clustering
    print(f"\n[6] Structural clustering...")
    linkage_matrix, clusters = structural_clustering(rmsd_matrix, args.n_clusters)
    cluster_df = rmsd_matrix.copy()
    cluster_df['Cluster'] = clusters
    cluster_df.to_csv(os.path.join(args.output, 'clusters.csv'))
    
    # Generate plots
    print(f"\n[7] Generating plots...")
    plot_rmsd_heatmap(rmsd_matrix, os.path.join(args.output, 'rmsd_heatmap.png'), verbose=args.verbose)
    plot_rmsf(rmsf_values, os.path.join(args.output, 'rmsf_plot.png'), verbose=args.verbose)
    plot_cluster_dendrogram(linkage_matrix, structure_names, 
                           os.path.join(args.output, 'cluster_dendrogram.png'), verbose=args.verbose)
    
    print(f"\n✅ Structural analysis completed. Results in {args.output}")


if __name__ == '__main__':
    main()