#!/usr/bin/env python3
"""
Setup script for hst-phylogeny-analysis package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hst-phylogeny-analysis",
    version="1.0.0",
    author="Behzad Hajieghrari",
    author_email="bheghrari@yahoo.com",
    description="Molecular Evolution and Phylogeny of the Plant HASTY Family - Analysis Scripts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # TODO: replace YOUR_USERNAME with the actual GitHub username/org before publishing
    url="https://github.com/YOUR_USERNAME/hst-phylogeny-analysis",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.12",
    install_requires=[
        "biopython>=1.86",
        "numpy>=1.26.4",
        "pandas>=2.0.3",
        "matplotlib>=3.8.2",
        "seaborn>=0.13.2",
        "scikit-learn>=1.4.0",
        "scipy>=1.12.0",
    ],
    entry_points={
        "console_scripts": [
            "curate_sequences=scripts.sequence_curation:main",
            "align_sequences=scripts.alignment_trim:main",
            "find_motifs=scripts.motif_analysis:main",
            "compare_structures=scripts.structural_compare:main",
            "generate_figures=scripts.figure_generation:main",
        ],
    },
)