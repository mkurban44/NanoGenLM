#!/usr/bin/env python3
"""
NanoGenLM — All 6 Publication Figures (Nature Style, 1200 DPI)
Usage: python3 06_plot_all_figures.py --base .
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
    'mp': '#4053D3', 'proto': '#DECE00',
    'r5': '#56B4E9', 'r6': '#0072B2',
    'high': '#009E73', 'medium': '#E69F00', 'low': '#CC79A7',
    'gray': '#999999',
}
CLASS_ORDER = ['perovskite', 'heusler', 'hydride']
CLASS_NAMES = {'perovskite': 'Perovskite', 'heusler': 'Heusler', 'hydride': 'Hydride'}
SC = 88 / 25.4
DC = 180 / 25.4


def add_panel_label(ax, label, x=-0.14, y=1.06):
    ax.text(x, y, f'({label})', transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top', ha='left')


def save_fig(fig, name, outdir):
    for fmt in ['pdf', 'png', 'svg']:
        fig.savefig(outdir / f"{name}.{fmt}", format=fmt)
    plt.close(fig)
    print(f"  Saved: {name}")


# ================================================================
# FIGURE 1: Dataset Overview (FIXED)
# ================================================================
def fig1_dataset_overview(nc, outdir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DC, DC * 0.38),
                                    gridspec_kw={'width_ratios': [1, 1.1]})
    mp_counts, proto_counts = [], []
    for cls in CLASS_ORDER:
        sub = nc[nc['class'] == cls]
        mp_counts.append(len(sub[sub['source'] == 'materials_project']))
        proto_counts.append(len(sub[sub['source'] == 'prototype']))

    x = np.arange(3)
    w = 0.55
    ax1.bar(x, mp_counts, w, label='Materials Project', color=COLORS['mp'], edgecolor='white', linewidth=0.3)
    ax1.bar(x, proto_counts, w, bottom=mp_counts, label='Prototype', color=COLORS['proto'], edgecolor='white', linewidth=0.3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Perovskite', 'Heusler', 'Hydride'])
    ax1.set_ylabel('Number of nanoparticles')
    ax1.legend(loc='upper right', borderpad=0.4)
    ax1.set_ylim(0, 1450)
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    for i, (mp, pr) in enumerate(zip(mp_counts, proto_counts)):
        ax1.text(i, mp + pr + 25, str(mp + pr), ha='center', va='bottom', fontsize=7, fontweight='500')
    add_panel_label(ax1, 'a', x=-0.18, y=1.06)

    conf_order = ['high', 'medium', 'low']
    conf_labels = ['High', 'Medium', 'Low']
    bar_h = 0.50
    for i, cls in enumerate(CLASS_ORDER):
        sub = nc[nc['class'] == cls]
        counts = [len(sub[sub['confidence_bulk'] == c]) for c in conf_order]
        total = sum(counts)
        fracs = [c / total * 100 for c in counts]
        left = 0
        for j, (frac, conf) in enumerate(zip(fracs, conf_order)):
            ax2.barh(i, frac, bar_h, left=left, color=COLORS[conf],
                    edgecolor='white', linewidth=0.4,
                    label=conf_labels[j] if i == 0 else '')
            if frac > 15:
                ax2.text(left + frac / 2, i, f'{frac:.0f}%',
                        ha='center', va='center', fontsize=7,
                        color='white', fontweight='600')
            left += frac
    ax2.set_yticks(range(3))
    ax2.set_yticklabels([CLASS_NAMES[c] for c in CLASS_ORDER])
    ax2.set_xlabel('Confidence distribution (%)')
    ax2.set_xlim(0, 102)
    ax2.invert_yaxis()
    ax2.legend(loc='center', ncol=3, columnspacing=1.2, handletextpad=0.4,
               frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 0.333))
    add_panel_label(ax2, 'b', x=-0.16, y=1.06)

    fig.tight_layout(w_pad=3)
    save_fig(fig, 'fig1_dataset_overview', outdir)


# ================================================================
# FIGURE 2: Energy Distribution (FIXED)
# ================================================================
def fig2_energy_distribution(nc, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(DC, DC * 0.40), sharey=False)
    for idx, cls in enumerate(CLASS_ORDER):
        ax = axes[idx]
        sub = nc[nc['class'] == cls]
        data_r5 = sub[sub['R'] == 5]['energy_per_atom_eV'].dropna().values
        data_r6 = sub[sub['R'] == 6]['energy_per_atom_eV'].dropna().values

        positions = [1, 2]
        vp = ax.violinplot([data_r5, data_r6], positions=positions,
                           showmeans=False, showmedians=False, showextrema=False)
        for i, body in enumerate(vp['bodies']):
            body.set_facecolor([COLORS['r5'], COLORS['r6']][i])
            body.set_alpha(0.25)
            body.set_edgecolor([COLORS['r5'], COLORS['r6']][i])
            body.set_linewidth(0.8)

        bp = ax.boxplot([data_r5, data_r6], positions=positions,
                       widths=0.18, patch_artist=True, showfliers=False, zorder=3)
        for i, (patch, color) in enumerate(zip(bp['boxes'], [COLORS['r5'], COLORS['r6']])):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(0.5)
        for element in ['whiskers', 'caps']:
            for line in bp[element]:
                line.set_color('black')
                line.set_linewidth(0.5)
        for line in bp['medians']:
            line.set_color('white')
            line.set_linewidth(1)

        ax.set_xticks(positions)
        ax.set_xticklabels(['R = 5 Å', 'R = 6 Å'])
        ax.set_title(CLASS_NAMES[cls], fontweight='500', pad=10)
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        if idx == 0:
            ax.set_ylabel('Energy per atom (eV)')

        n5, m5 = len(data_r5), np.mean(data_r5)
        n6, m6 = len(data_r6), np.mean(data_r6)
        ax.text(0.03, 0.04, f'R=5\nn={n5}\nμ={m5:.1f}',
                transform=ax.transAxes, fontsize=6, va='bottom', ha='left',
                color=COLORS['r5'], fontweight='500', linespacing=1.3)
        ax.text(0.97, 0.04, f'R=6\nn={n6}\nμ={m6:.1f}',
                transform=ax.transAxes, fontsize=6, va='bottom', ha='right',
                color=COLORS['r6'], fontweight='500', linespacing=1.3)

    add_panel_label(axes[0], 'a', x=-0.18, y=1.10)
    add_panel_label(axes[1], 'b', x=-0.12, y=1.10)
    add_panel_label(axes[2], 'c', x=-0.12, y=1.10)
    fig.tight_layout(w_pad=2.5)
    save_fig(fig, 'fig2_energy_distribution', outdir)


# ================================================================
# FIGURE 3: R5 vs R6 Scatter
# ================================================================
def fig3_energy_correlation(nc, outdir):
    fig, ax = plt.subplots(figsize=(SC * 1.1, SC * 1.1))
    r5 = nc[nc['R'] == 5][['formula', 'class', 'energy_per_atom_eV', 'source']].rename(
        columns={'energy_per_atom_eV': 'E_R5'})
    r6 = nc[nc['R'] == 6][['formula', 'energy_per_atom_eV']].rename(
        columns={'energy_per_atom_eV': 'E_R6'})
    paired = r5.merge(r6, on='formula')

    for cls in CLASS_ORDER:
        sub = paired[paired['class'] == cls]
        ax.scatter(sub['E_R5'], sub['E_R6'], s=12, alpha=0.5,
                  color=COLORS[cls], label=CLASS_NAMES[cls], edgecolors='none', zorder=2)

    all_x = paired['E_R5'].values
    all_y = paired['E_R6'].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(all_x, all_y)
    r2 = r_value ** 2
    x_line = np.array([all_x.min() - 5, all_x.max() + 5])
    ax.plot(x_line, slope * x_line + intercept, '-', color='black', linewidth=0.8, alpha=0.7, zorder=1)
    ax.plot(x_line, x_line, '--', color=COLORS['gray'], linewidth=0.6, alpha=0.5, zorder=0)
    ax.text(0.05, 0.95, f'$R^2$ = {r2:.3f}\nslope = {slope:.3f}\nn = {len(paired)}',
            transform=ax.transAxes, fontsize=7.5, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLORS['gray'],
                      alpha=0.9, linewidth=0.5))
    ax.set_xlabel('$E$(R = 5 Å) (eV/atom)')
    ax.set_ylabel('$E$(R = 6 Å) (eV/atom)')
    ax.legend(loc='lower right', markerscale=1.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    lims = [min(all_x.min(), all_y.min()) - 5, max(all_x.max(), all_y.max()) + 5]
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    save_fig(fig, 'fig3_energy_correlation', outdir)


# ================================================================
# FIGURE 4: Size Effect + Structure
# ================================================================
def fig4_size_effect(nc, outdir):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(DC, DC * 0.35))

    r5 = nc[nc['R'] == 5][['formula', 'class', 'energy_per_atom_eV']].rename(columns={'energy_per_atom_eV': 'E_R5'})
    r6 = nc[nc['R'] == 6][['formula', 'energy_per_atom_eV']].rename(columns={'energy_per_atom_eV': 'E_R6'})
    paired = r5.merge(r6, on='formula')
    paired['dE'] = paired['E_R6'] - paired['E_R5']

    bins = np.linspace(-40, 40, 40)
    for cls in CLASS_ORDER:
        sub = paired[paired['class'] == cls]
        ax1.hist(sub['dE'], bins=bins, alpha=0.55, color=COLORS[cls],
                label=CLASS_NAMES[cls], edgecolor='white', linewidth=0.3, density=True)
    ax1.axvline(0, color='black', linewidth=0.6, linestyle='--', alpha=0.5)
    ax1.set_xlabel('$\\Delta E$ = $E$(R=6) − $E$(R=5) (eV/atom)')
    ax1.set_ylabel('Density')
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    add_panel_label(ax1, 'a')

    data_nn, colors_nn = [], []
    for cls in CLASS_ORDER:
        data_nn.append(nc[nc['class'] == cls]['nn_mean'].dropna().values)
        colors_nn.append(COLORS[cls])
    bp2 = ax2.boxplot(data_nn, positions=range(len(CLASS_ORDER)),
                      widths=0.45, patch_artist=True, showfliers=True,
                      flierprops=dict(marker='.', markersize=2, alpha=0.3))
    for patch, color in zip(bp2['boxes'], colors_nn):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor('black')
        patch.set_linewidth(0.5)
    for element in ['whiskers', 'caps']:
        for line in bp2[element]:
            line.set_color('black')
            line.set_linewidth(0.5)
    for line in bp2['medians']:
        line.set_color('white')
        line.set_linewidth(1.2)
    ax2.set_xticks(range(len(CLASS_ORDER)))
    ax2.set_xticklabels([CLASS_NAMES[c] for c in CLASS_ORDER])
    ax2.set_ylabel('Mean NN distance (Å)')
    ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
    add_panel_label(ax2, 'b')

    x = np.arange(len(CLASS_ORDER))
    w = 0.3
    r5_means = [nc[(nc['class'] == cls) & (nc['R'] == 5)]['n_atoms'].mean() for cls in CLASS_ORDER]
    r6_means = [nc[(nc['class'] == cls) & (nc['R'] == 6)]['n_atoms'].mean() for cls in CLASS_ORDER]
    r5_stds = [nc[(nc['class'] == cls) & (nc['R'] == 5)]['n_atoms'].std() for cls in CLASS_ORDER]
    r6_stds = [nc[(nc['class'] == cls) & (nc['R'] == 6)]['n_atoms'].std() for cls in CLASS_ORDER]
    ax3.bar(x - w/2, r5_means, w, yerr=r5_stds, label='R = 5 Å',
           color=COLORS['r5'], edgecolor='white', linewidth=0.3,
           error_kw=dict(linewidth=0.5, capsize=2, capthick=0.5))
    ax3.bar(x + w/2, r6_means, w, yerr=r6_stds, label='R = 6 Å',
           color=COLORS['r6'], edgecolor='white', linewidth=0.3,
           error_kw=dict(linewidth=0.5, capsize=2, capthick=0.5))
    ax3.set_xticks(x)
    ax3.set_xticklabels([CLASS_NAMES[c] for c in CLASS_ORDER])
    ax3.set_ylabel('Number of atoms')
    ax3.legend(loc='upper left')
    ax3.yaxis.set_minor_locator(AutoMinorLocator(2))
    add_panel_label(ax3, 'c')

    fig.tight_layout(w_pad=2.5)
    save_fig(fig, 'fig4_size_and_structure', outdir)


# ================================================================
# FIGURE 5: Bulk vs NP Energy
# ================================================================
def fig5_bulk_vs_np(nc, outdir):
    fig, ax = plt.subplots(figsize=(SC * 1.1, SC * 1.1))
    nc_valid = nc.dropna(subset=['bulk_energy_per_atom_eV', 'energy_per_atom_eV'])
    nc_valid = nc_valid[nc_valid['bulk_energy_per_atom_eV'] != '']
    nc_valid['bulk_E'] = nc_valid['bulk_energy_per_atom_eV'].astype(float)

    if len(nc_valid) == 0:
        print("  WARNING: No bulk-NP paired data. Skipping fig5.")
        plt.close(fig)
        return

    for cls in CLASS_ORDER:
        sub = nc_valid[nc_valid['class'] == cls]
        if len(sub) > 0:
            ax.scatter(sub['bulk_E'], sub['energy_per_atom_eV'], s=12, alpha=0.5,
                      color=COLORS[cls], label=CLASS_NAMES[cls], edgecolors='none')

    all_vals = np.concatenate([nc_valid['bulk_E'].values, nc_valid['energy_per_atom_eV'].values])
    lims = [all_vals.min() - 5, all_vals.max() + 5]
    ax.plot(lims, lims, '--', color=COLORS['gray'], linewidth=0.6, alpha=0.5)
    slope, intercept, r_value, _, _ = stats.linregress(nc_valid['bulk_E'].values, nc_valid['energy_per_atom_eV'].values)
    x_line = np.array(lims)
    ax.plot(x_line, slope * x_line + intercept, '-', color='black', linewidth=0.8, alpha=0.7)
    ax.text(0.05, 0.95, f'$R^2$ = {r_value**2:.3f}\nn = {len(nc_valid)}',
            transform=ax.transAxes, fontsize=7.5, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=COLORS['gray'], alpha=0.9, linewidth=0.5))
    ax.set_xlabel('Bulk energy per atom (eV)')
    ax.set_ylabel('NP energy per atom (eV)')
    ax.legend(loc='lower right', markerscale=1.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    save_fig(fig, 'fig5_bulk_vs_np', outdir)


# ================================================================
# FIGURE 6: Morphology
# ================================================================
def fig6_morphology(nc, outdir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DC, DC * 0.38))

    data_asp, colors_asp = [], []
    for cls in CLASS_ORDER:
        vals = nc[nc['class'] == cls]['asphericity'].dropna()
        vals = vals[vals < vals.quantile(0.98)]
        data_asp.append(vals.values)
        colors_asp.append(COLORS[cls])
    bp = ax1.boxplot(data_asp, positions=range(len(CLASS_ORDER)),
                     widths=0.45, patch_artist=True, showfliers=False)
    for patch, color in zip(bp['boxes'], colors_asp):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor('black')
        patch.set_linewidth(0.5)
    for element in ['whiskers', 'caps']:
        for line in bp[element]:
            line.set_color('black')
            line.set_linewidth(0.5)
    for line in bp['medians']:
        line.set_color('white')
        line.set_linewidth(1.2)
    ax1.set_xticks(range(len(CLASS_ORDER)))
    ax1.set_xticklabels([CLASS_NAMES[c] for c in CLASS_ORDER])
    ax1.set_ylabel('Asphericity (Å²)')
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    add_panel_label(ax1, 'a')

    for cls in CLASS_ORDER:
        sub = nc[nc['class'] == cls]
        ax2.scatter(sub['radius_of_gyration'], sub['energy_per_atom_eV'],
                   s=8, alpha=0.4, color=COLORS[cls], label=CLASS_NAMES[cls], edgecolors='none')
    ax2.set_xlabel('Radius of gyration (Å)')
    ax2.set_ylabel('Energy per atom (eV)')
    ax2.legend(loc='upper right', markerscale=2)
    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
    add_panel_label(ax2, 'b')

    fig.tight_layout(w_pad=3)
    save_fig(fig, 'fig6_morphology', outdir)


# ================================================================
# MAIN
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=".")
    args = parser.parse_args()

    base = Path(args.base)
    nc_path = base / "dataset" / "nature_comm_dataset.csv"
    if not nc_path.exists():
        nc_path = base / "nature_comm_dataset.csv"
    if not nc_path.exists():
        print(f"ERROR: Cannot find nature_comm_dataset.csv")
        return

    nc = pd.read_csv(nc_path)
    print(f"Loaded: {len(nc)} records")

    outdir = base / "figures"
    outdir.mkdir(exist_ok=True)

    fig1_dataset_overview(nc, outdir)
    fig2_energy_distribution(nc, outdir)
    fig3_energy_correlation(nc, outdir)
    fig4_size_effect(nc, outdir)
    fig5_bulk_vs_np(nc, outdir)
    fig6_morphology(nc, outdir)

    print(f"\nAll 6 figures saved to {outdir}/")


if __name__ == "__main__":
    main()
