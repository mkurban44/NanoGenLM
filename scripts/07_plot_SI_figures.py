#!/usr/bin/env python3
"""
NanoGenLM — Supplementary Figures (S1–S5)
Usage: python3 07_plot_SI_figures.py --base .
Saves to figures-SI/ as pdf + png + svg at 1200 DPI.
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 9, 'axes.titlesize': 9,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'legend.fontsize': 7, 'legend.title_fontsize': 7.5,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'figure.dpi': 1200, 'savefig.dpi': 1200,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
    'axes.grid': False, 'legend.frameon': False,
})

COLORS = {
    'perovskite': '#0072B2', 'heusler': '#D55E00', 'hydride': '#009E73',
    'hea': '#CC79A7',
    'mp': '#4053D3', 'proto': '#DECE00',
    'r5': '#56B4E9', 'r6': '#0072B2',
    'gray': '#999999',
}
CLASS_ORDER = ['perovskite', 'heusler', 'hydride']
CLASS_NAMES = {'perovskite': 'Perovskite', 'heusler': 'Heusler', 'hydride': 'Hydride', 'hea': 'HEA'}
DC = 180 / 25.4
SC = 88 / 25.4


def add_panel_label(ax, label, x=-0.14, y=1.06):
    ax.text(x, y, f'({label})', transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top', ha='left')


def save(fig, name, outdir):
    for fmt in ['pdf', 'png', 'svg']:
        fig.savefig(outdir / f"{name}.{fmt}", format=fmt)
    plt.close(fig)
    print(f"  Saved: {name}")


# ================================================================
# FIG S1: Bulk relax volume change distribution
# ================================================================
def figS1(bulk, outdir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DC, DC * 0.38))

    # (a) Histogram by class (perovskite, heusler, hydride)
    bins = np.linspace(0, 50, 50)
    for cls in CLASS_ORDER:
        sub = bulk[(bulk['class'] == cls) & (bulk['status'].isin(['passed', 'suspect']))]
        dv = sub['volume_change_pct'].dropna()
        ax1.hist(dv, bins=bins, alpha=0.55, color=COLORS[cls],
                label=CLASS_NAMES[cls], edgecolor='white', linewidth=0.3, density=True)
    ax1.axvline(30, color='black', linewidth=0.8, linestyle='--', alpha=0.6)
    ax1.text(28, ax1.get_ylim()[1] * 0.9, 'Suspect\nthreshold', ha='right',
            fontsize=6, color='black', alpha=0.7)
    ax1.set_xlabel('Volume change after bulk relaxation (%)')
    ax1.set_ylabel('Density')
    ax1.set_xlim(0, 50)
    ax1.legend(loc='upper right')
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    add_panel_label(ax1, 'a', x=-0.18)

    # (b) Including HEA
    for cls in ['perovskite', 'heusler', 'hydride']:
        sub = bulk[(bulk['class'] == cls) & (bulk['status'].isin(['passed', 'suspect']))]
        dv = sub['volume_change_pct'].dropna()
        if len(dv) == 0:
            continue
        ax2.hist(dv, bins=bins, alpha=0.45, color=COLORS[cls],
                label=CLASS_NAMES[cls], edgecolor='white', linewidth=0.3, density=True)
    ax2.axvline(30, color='black', linewidth=0.8, linestyle='--', alpha=0.6)
    ax2.set_xlabel('Volume change after bulk relaxation (%)')
    ax2.set_ylabel('Density')
    ax2.set_xlim(0, 50)
    ax2.legend(loc='upper right')
    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    add_panel_label(ax2, 'b', x=-0.12)

    fig.tight_layout(w_pad=3)
    save(fig, 'figS1_volume_change', outdir)


# ================================================================
# FIG S2: MP vs Prototype energy comparison
# ================================================================
def figS2(nc, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(DC, DC * 0.35), sharey=False)

    for idx, cls in enumerate(CLASS_ORDER):
        ax = axes[idx]
        sub = nc[nc['class'] == cls]
        mp_e = sub[sub['source'] == 'materials_project']['energy_per_atom_eV'].dropna()
        pr_e = sub[sub['source'] == 'prototype']['energy_per_atom_eV'].dropna()

        bins = np.linspace(sub['energy_per_atom_eV'].min() - 2,
                          sub['energy_per_atom_eV'].max() + 2, 30)

        if len(mp_e) > 0:
            ax.hist(mp_e, bins=bins, alpha=0.6, color=COLORS['mp'],
                   label=f'MP (n={len(mp_e)})', edgecolor='white', linewidth=0.3, density=True)
        if len(pr_e) > 0:
            ax.hist(pr_e, bins=bins, alpha=0.5, color=COLORS['proto'],
                   label=f'Proto (n={len(pr_e)})', edgecolor='white', linewidth=0.3, density=True)

        ax.set_xlabel('Energy per atom (eV)')
        if idx == 0:
            ax.set_ylabel('Density')
        ax.set_title(CLASS_NAMES[cls], fontweight='500', pad=8)
        ax.legend(loc='upper left', fontsize=6)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))

    add_panel_label(axes[0], 'a', x=-0.18)
    add_panel_label(axes[1], 'b', x=-0.18)
    add_panel_label(axes[2], 'c', x=-0.18)

    fig.tight_layout(w_pad=2.5)
    save(fig, 'figS2_mp_vs_proto', outdir)


# ================================================================
# FIG S3: HEA bulk results (VEC vs energy, phase distribution)
# ================================================================
def figS3(bulk, outdir):
    # Load master CSV for VEC and delta info
    master_path = bulk_path.parent / "compositions_master.csv"
    if not master_path.exists():
        print("  WARNING: compositions_master.csv not found, skipping figS3")
        return

    master = pd.read_csv(master_path)
    hea_master = master[master['class'] == 'hea']
    hea_bulk = bulk[bulk['class'] == 'hea']

    # Merge
    merged = hea_bulk.merge(hea_master[['formula', 'vec', 'delta_percent', 'predicted_phase']],
                            on='formula', how='left')
    merged = merged[merged['status'].isin(['passed', 'suspect'])]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DC, DC * 0.38))

    # (a) VEC vs energy
    for phase, color, marker in [('bcc', '#D55E00', 'o'), ('fcc', '#0072B2', 's'), ('mixed', '#CC79A7', '^')]:
        sub = merged[merged['predicted_phase'] == phase]
        if len(sub) > 0:
            ax1.scatter(sub['vec'], sub['energy_per_atom_eV'], s=8, alpha=0.4,
                       color=color, label=f'{phase.upper()} (n={len(sub)})',
                       edgecolors='none', marker=marker)
    ax1.set_xlabel('Valence Electron Concentration (VEC)')
    ax1.set_ylabel('Energy per atom (eV)')
    ax1.legend(loc='lower left', markerscale=2)
    ax1.axvline(6.87, color='gray', linewidth=0.6, linestyle=':', alpha=0.5)
    ax1.axvline(8.0, color='gray', linewidth=0.6, linestyle=':', alpha=0.5)
    ax1.text(5.5, ax1.get_ylim()[1] * 0.95, 'BCC', fontsize=6, color='gray', ha='center')
    ax1.text(7.4, ax1.get_ylim()[1] * 0.95, 'Mixed', fontsize=6, color='gray', ha='center')
    ax1.text(9.0, ax1.get_ylim()[1] * 0.95, 'FCC', fontsize=6, color='gray', ha='center')
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    add_panel_label(ax1, 'a', x=-0.18)

    # (b) Delta vs energy
    ax2.scatter(merged['delta_percent'], merged['energy_per_atom_eV'], s=8, alpha=0.3,
               color=COLORS['hea'], edgecolors='none')
    ax2.set_xlabel('Atomic size mismatch \u03B4 (%)')
    ax2.set_ylabel('Energy per atom (eV)')
    ax2.axvline(6.6, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax2.text(6.5, ax2.get_ylim()[0] * 0.85, 'δ cutoff', fontsize=6, color='black', alpha=0.7, ha='right')
    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    add_panel_label(ax2, 'b', x=-0.12)

    fig.tight_layout(w_pad=3)
    save(fig, 'figS3_hea_bulk', outdir)


# ================================================================
# FIG S4: Atom count vs energy
# ================================================================
def figS4(nc, outdir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DC, DC * 0.38))

    # (a) R=5
    r5 = nc[nc['R'] == 5]
    for cls in CLASS_ORDER:
        sub = r5[r5['class'] == cls]
        ax1.scatter(sub['n_atoms'], sub['energy_per_atom_eV'], s=8, alpha=0.4,
                   color=COLORS[cls], label=CLASS_NAMES[cls], edgecolors='none')
    ax1.set_xlabel('Number of atoms')
    ax1.set_ylabel('Energy per atom (eV)')
    ax1.set_title('R = 5 \u00C5', fontweight='500', pad=8)
    ax1.legend(loc='upper right', markerscale=2)
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    add_panel_label(ax1, 'a', x=-0.18)

    # (b) R=6
    r6 = nc[nc['R'] == 6]
    for cls in CLASS_ORDER:
        sub = r6[r6['class'] == cls]
        ax2.scatter(sub['n_atoms'], sub['energy_per_atom_eV'], s=8, alpha=0.4,
                   color=COLORS[cls], label=CLASS_NAMES[cls], edgecolors='none')
    ax2.set_xlabel('Number of atoms')
    ax2.set_ylabel('Energy per atom (eV)')
    ax2.set_title('R = 6 \u00C5', fontweight='500', pad=8)
    ax2.legend(loc='upper right', markerscale=2)
    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    add_panel_label(ax2, 'b', x=-0.12)

    fig.tight_layout(w_pad=3)
    save(fig, 'figS4_atoms_vs_energy', outdir)


# ================================================================
# FIG S5: Most/least stable compositions
# ================================================================
def figS5(nc, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(DC, DC * 0.42))

    for idx, cls in enumerate(CLASS_ORDER):
        ax = axes[idx]
        sub = nc[(nc['class'] == cls) & (nc['R'] == 6)].copy()
        sub = sub.sort_values('energy_per_atom_eV')

        top5 = sub.head(5)
        bot5 = sub.tail(5)
        combined = pd.concat([top5, bot5])

        y_pos = np.arange(10)
        colors_bar = [COLORS[cls]] * 5 + ['#CCCCCC'] * 5
        alphas = [0.9] * 5 + [0.5] * 5

        bars = ax.barh(y_pos, combined['energy_per_atom_eV'].values,
                       color=colors_bar, edgecolor='white', linewidth=0.3, height=0.7)
        for bar, a in zip(bars, alphas):
            bar.set_alpha(a)

        labels = combined['formula'].values
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=5.5)
        ax.set_xlabel('Energy per atom (eV)')
        ax.set_title(CLASS_NAMES[cls], fontweight='500', pad=8)
        ax.invert_yaxis()

        # Separator line
        ax.axhline(4.5, color='black', linewidth=0.5, linestyle='--', alpha=0.4)
        ax.text(ax.get_xlim()[1] * 0.98, 2, 'Most\nstable', fontsize=5,
                ha='right', va='center', color=COLORS[cls], alpha=0.7)
        lx = 0.65 if idx < 2 else 0.72
        ax.text(lx, 0.30, 'Least\nstable', fontsize=5, transform=ax.transAxes,
                ha='right', va='center', color='#999', alpha=0.7)

        ax.xaxis.set_minor_locator(AutoMinorLocator(2))

    add_panel_label(axes[0], 'a', x=-0.22)
    add_panel_label(axes[1], 'b', x=-0.16)
    add_panel_label(axes[2], 'c', x=-0.16)

    fig.tight_layout(w_pad=2)
    save(fig, 'figS5_stability_ranking', outdir)


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=".")
    args = parser.parse_args()

    base = Path(args.base)

    nc_path = base / "dataset" / "nature_comm_dataset.csv"
    if not nc_path.exists():
        nc_path = base / "nature_comm_dataset.csv"
    bulk_path = base / "bulk_relax_results_v2.csv"

    if not nc_path.exists():
        print(f"ERROR: {nc_path} not found")
        exit(1)
    if not bulk_path.exists():
        print(f"ERROR: {bulk_path} not found")
        exit(1)

    nc = pd.read_csv(nc_path)
    bulk = pd.read_csv(bulk_path)
    print(f"Loaded: {len(nc)} NP records, {len(bulk)} bulk records")

    outdir = base / "figures-SI"
    outdir.mkdir(exist_ok=True)

    figS1(bulk, outdir)
    figS2(nc, outdir)
    figS3(bulk, outdir)
    figS4(nc, outdir)
    figS5(nc, outdir)

    print(f"\nAll SI figures saved to {outdir}/")
