# NanoGenLM

**NanoGenLM: A paired-size benchmark for alloy nanoparticle stability across material classes**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20112995.svg)](https://doi.org/10.5281/zenodo.20112995)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

NanoGenLM is a computational pipeline for the systematic generation and DFTB-based relaxation of alloy nanoparticle structures across multiple material classes. Starting from 75 elements covered by the PTBP parameter set, the pipeline generates, validates, and relaxes thousands of nanoparticle structures to produce ML-ready datasets.

**Key numbers:**
- 3,184 unique compositions generated (perovskite, Heusler, hydride, HEA)
- 399 DFT-validated structures from Materials Project
- 81,423 nanoparticles carved (R = 5–30 Å)
- 2,488 DFTB+-relaxed nanoparticles (R = 5, 6 Å)
- 2,346 paired structures in the final dataset

## Pipeline

```
PTBP Parameters (75 elements)
    ↓
Composition Generation + Scientific Filters
    ↓
Materials Project Download + Prototype CIFs
    ↓
Bulk DFTB+ Relaxation (3,137 passed)
    ↓
Spherical Nanoparticle Carving (Atomsk)
    ↓
NP DFTB+ Relaxation (2,488 passed)
    ↓
ML Dataset (2,346 NPs, 1,173 compositions)
```

## Repository Structure

```
NanoGenLM/
├── scripts/
│   ├── 01_generate_unitcells.py
│   ├── 01b_download_mp.py
│   ├── 02_bulk_relax.py
│   ├── 03_carve_nanoparticles.py
│   ├── 04_relax_nanoparticles.py
│   ├── 05_collect_dataset.py
│   ├── 06_plot_all_figures.py
│   ├── 06_plot_fig1_fig2_final.py
│   ├── 07_plot_SI_figures.py
│   └── 08_plot_ML_figures.py
├── bulk_relax_results_v2.csv
├── carve_results.csv
├── compositions_master.csv
├── compositions_master_v2.csv
├── np_relax_results.csv
├── LICENSE
└── README.md
```
The full dataset, including relaxed structures, ASE databases, XYZ files, and archived nanoparticle/unit-cell files, is available on Zenodo: https://doi.org/10.5281/zenodo.20112995.

## Requirements

### Software
- **DFTB+** (≥ 22.1) with SCC support
- **Atomsk** (≥ 0.12) for supercell generation and carving
- **Python** ≥ 3.9

### Python packages
```bash
pip install numpy pandas pymatgen ase scipy matplotlib scikit-learn umap-learn
```

### DFTB parameters
- **PTBP** (Periodic Table of the Tight-Binding Parameters)
  - Reference: [J. Chem. Theory Comput. 2024](https://doi.org/10.1021/acs.jctc.4c00228)
  - 75 elements across s-, p-, and d-blocks

### Optional
- **mp-api** or `requests` for Materials Project download (requires free API key)

## Quick Start

### 1. Generate unit cells
```bash
python scripts/01_generate_unitcells.py --base .
```

### 2. Download Materials Project structures (optional, requires API key)
```bash
export MP_API_KEY="your_key_here"
python scripts/01b_download_mp.py --base .
```

### 3. Bulk DFTB+ relaxation
```bash
python scripts/02_bulk_relax.py --base . --workers 10
```

### 4. Carve nanoparticles
```bash
python scripts/03_carve_nanoparticles.py --base . --workers 8
```

### 5. Relax nanoparticles
```bash
python scripts/04_relax_nanoparticles.py --base . --workers 10
```

### 6. Collect ML dataset
```bash
python scripts/05_collect_dataset.py --base .
```

### 7. Generate figures
```bash
python scripts/06_plot_all_figures.py --base .
python scripts/06_plot_fig1_fig2_final.py
python scripts/07_plot_SI_figures.py --base .
python scripts/08_plot_ML_figures.py --base .
```

## Dataset Description

### Final paired dataset (`nature_comm_dataset.csv`)
- **2,346 records** (1,173 compositions × 2 radii)
- Only compositions where both R = 5 and R = 6 Å relaxation succeeded
- Columns include: formula, class, energy, structural descriptors, source, confidence

### NeurIPS dataset (`neurips_dataset.csv`)
- **2,488 records** (1,315 compositions)
- Includes all successfully relaxed NPs (single-radius included)

### Material classes

| Class | Compositions | NPs | Structure | Source |
|-------|-------------|-----|-----------|--------|
| Perovskite (ABX₃) | 451 | 902 | Pm-3m | MP + prototype |
| Heusler (X₂YZ) | 598 | 1,196 | Fm-3m | MP + prototype |
| Hydride (ABH₃) | 124 | 248 | Pm-3m | MP + prototype |

### Scientific filters applied
- **Perovskite**: Goldschmidt tolerance factor (0.71 ≤ t ≤ 1.10), octahedral factor (0.41 ≤ μ ≤ 0.73)
- **HEA**: VEC-based phase assignment, atomic size mismatch δ < 6.6%
- **Bulk relaxation**: Volume change < 30% (passed), ≥ 30% (suspect, low confidence)

## Computational Details

- **DFTB method**: Self-consistent charge DFTB with PTBP parameters
- **Bulk relaxation**: Lattice + atomic position optimization, 4×4×4 k-points
- **NP relaxation**: Atomic position optimization only (non-periodic), Fermi smearing at 600–1000 K
- **Nanoparticle carving**: Spherical cut from relaxed supercells using Atomsk, margin = 5 Å

## Citation

If you use this dataset or code, please cite the associated manuscript:

Mustafa Kurban and Erchin Serpedin, *NanoGenLM: A paired-size benchmark for alloy nanoparticle stability across material classes*, submitted, 2026.

Dataset DOI: https://doi.org/10.5281/zenodo.20112995


## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

The dataset is released under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Acknowledgments
- PTBP parameters: [Cui et al., J. Chem. Theory Comput. 2024](https://doi.org/10.1021/acs.jctc.4c00228)
- Materials Project: [Jain et al., APL Mater. 2013](https://doi.org/10.1063/1.4812323)
- DFTB+: [Hourahine et al., J. Chem. Phys. 2020](https://doi.org/10.1063/1.5143190)
- Atomsk: [Hirel, Comput. Phys. Commun. 2015](https://doi.org/10.1016/j.cpc.2015.07.012)
- CrystaLLM (inspiration): [Antunes et al., Nat. Commun. 2024](https://doi.org/10.1038/s41467-024-54639-7)
