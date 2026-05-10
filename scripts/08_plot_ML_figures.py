#!/usr/bin/env python3
"""
NanoGenLM — ML Analysis Figures (Fig 9 & 10)
Usage: python3 08_plot_ML_figures.py --base .
Saves to figures-ML/ as pdf + png + svg at 1200 DPI.

Requirements: pip install umap-learn scikit-learn
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy import stats
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.lines import Line2D

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
    'r5': '#56B4E9', 'r6': '#0072B2',
    'gray': '#999999',
}
CLASS_ORDER = ['perovskite', 'heusler', 'hydride']
CLASS_NAMES = {'perovskite': 'Perovskite', 'heusler': 'Heusler', 'hydride': 'Hydride'}
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


def prepare_features(nc):
    """Extract numerical features for ML."""
    # Composition features: element counts from composition_json
    comp_features = []
    for _, row in nc.iterrows():
        try:
            comp = json.loads(row['composition_json'])
        except:
            comp = {}
        comp_features.append(comp)

    # All unique elements
    all_elements = sorted(set(el for comp in comp_features for el in comp))

    # Build feature matrix
    feature_cols = ['n_atoms', 'R', 'n_elements', 'nn_mean', 'nn_min', 'nn_max',
                    'nn_std', 'radius_of_gyration', 'asphericity', 'surface_fraction']

    X_struct = nc[feature_cols].copy()

    # Add element fractions
    for el in all_elements:
        X_struct[f'frac_{el}'] = [
            comp.get(el, 0) / max(sum(comp.values()), 1)
            for comp in comp_features
        ]

    # Add class encoding
    for cls in CLASS_ORDER:
        X_struct[f'is_{cls}'] = (nc['class'] == cls).astype(int)

    # Clean
    X_struct = X_struct.fillna(0)

    return X_struct, all_elements


# ================================================================
# FIG 9: UMAP Structural Space Map
# ================================================================
def fig9_umap(nc, outdir):
    try:
        from umap import UMAP
    except ImportError:
        print("  WARNING: umap-learn not installed. pip install umap-learn")
        print("  Falling back to PCA...")
        from sklearn.decomposition import PCA
        use_pca = True
    else:
        use_pca = False

    # Features for dimensionality reduction
    feat_cols = ['n_atoms', 'nn_mean', 'nn_min', 'nn_max', 'nn_std',
                 'radius_of_gyration', 'asphericity', 'surface_fraction',
                 'energy_per_atom_eV']

    df = nc[feat_cols + ['class', 'R']].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feat_cols])

    if use_pca:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
        embedding = reducer.fit_transform(X_scaled)
        ax1_label = f'PC1 ({reducer.explained_variance_ratio_[0]*100:.1f}%)'
        ax2_label = f'PC2 ({reducer.explained_variance_ratio_[1]*100:.1f}%)'
        method_name = 'PCA'
    else:
        reducer = UMAP(n_components=2, n_neighbors=30, min_dist=0.3, random_state=42)
        embedding = reducer.fit_transform(X_scaled)
        ax1_label = 'UMAP 1'
        ax2_label = 'UMAP 2'
        method_name = 'UMAP'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DC, DC * 0.40))

    # (a) Colored by class
    for cls in CLASS_ORDER:
        mask = df['class'].values == cls
        ax1.scatter(embedding[mask, 0], embedding[mask, 1],
                   s=6, alpha=0.4, color=COLORS[cls],
                   label=CLASS_NAMES[cls], edgecolors='none', rasterized=True)

    ax1.set_xlabel(ax1_label)
    ax1.set_ylabel(ax2_label)
    ax1.legend(loc='best', markerscale=3)
    add_panel_label(ax1, 'a', x=-0.16)

    # (b) Colored by energy
    sc = ax2.scatter(embedding[:, 0], embedding[:, 1],
                     c=df['energy_per_atom_eV'].values,
                     s=6, alpha=0.5, cmap='RdYlBu', edgecolors='none', rasterized=True)
    cbar = plt.colorbar(sc, ax=ax2, shrink=0.85, aspect=25, pad=0.02)
    cbar.set_label('Energy per atom (eV)', fontsize=8)
    cbar.ax.tick_params(labelsize=6.5)

    ax2.set_xlabel(ax1_label)
    ax2.set_ylabel(ax2_label)
    add_panel_label(ax2, 'b', x=-0.14)

    # R=5 vs R=6 markers in legend for panel a
    r5_mask = df['R'].values == 5
    r6_mask = df['R'].values == 6

    fig.tight_layout(w_pad=2.5)
    save(fig, f'fig9_{method_name.lower()}_map', outdir)

    return method_name


# ================================================================
# FIG 10: Energy Prediction Model
# ================================================================
def fig10_prediction(nc, outdir):
    X, all_elements = prepare_features(nc)
    y = nc['energy_per_atom_eV'].values

    # Remove any NaN
    valid = ~np.isnan(y) & ~X.isnull().any(axis=1)
    X = X[valid]
    y = y[valid]
    classes = nc['class'].values[valid]

    # Train/test split (80/20, stratified by class)
    X_train, X_test, y_train, y_test, cls_train, cls_test = train_test_split(
        X, y, classes, test_size=0.2, random_state=42, stratify=classes
    )

    print(f"  Training: {len(X_train)}, Testing: {len(X_test)}")

    # Train GBR
    model = GradientBoostingRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)

    print(f"  R\u00b2 train: {r2_train:.4f}")
    print(f"  R\u00b2 test:  {r2_test:.4f}")
    print(f"  MAE test: {mae_test:.3f} eV/atom")

    # Feature importance (top 15)
    feat_names = X.columns.tolist()
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    top_names = [feat_names[i] for i in top_idx]
    top_imp = importances[top_idx]

    # Clean feature names for display
    display_names = []
    for name in top_names:
        name = name.replace('frac_', '').replace('is_', '').replace('_', ' ')
        name = name.replace('nn mean', '\u27E8d\u2099\u2099\u27E9')
        name = name.replace('nn min', 'd\u2099\u2099 min')
        name = name.replace('nn max', 'd\u2099\u2099 max')
        name = name.replace('nn std', '\u03C3(d\u2099\u2099)')
        name = name.replace('n atoms', 'N\u2090\u209C\u2092\u2098')
        name = name.replace('n elements', 'N\u2091\u2097')
        name = name.replace('radius of gyration', 'R\u209A')
        name = name.replace('surface fraction', 'f\u209B\u1D64\u1D63\u1DA0')
        display_names.append(name)

    # ---- PLOT ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DC, DC * 0.42),
                                    gridspec_kw={'width_ratios': [1.1, 1]})

    # (a) Actual vs Predicted scatter
    for cls in CLASS_ORDER:
        mask = cls_test == cls
        ax1.scatter(y_test[mask], y_pred_test[mask], s=10, alpha=0.5,
                   color=COLORS[cls], label=CLASS_NAMES[cls],
                   edgecolors='none', zorder=2, rasterized=True)

    # Identity line
    lims = [min(y_test.min(), y_pred_test.min()) - 3,
            max(y_test.max(), y_pred_test.max()) + 3]
    ax1.plot(lims, lims, '--', color=COLORS['gray'], linewidth=0.6, alpha=0.5, zorder=0)

    # Regression line
    slope, intercept, r_val, _, _ = stats.linregress(y_test, y_pred_test)
    x_fit = np.array(lims)
    ax1.plot(x_fit, slope * x_fit + intercept, '-', color='black',
             linewidth=0.8, alpha=0.6, zorder=1)

    ax1.text(0.05, 0.95,
             f'$R^2$ = {r2_test:.3f}\nMAE = {mae_test:.2f} eV/atom\nn = {len(y_test)}',
             transform=ax1.transAxes, fontsize=7.5, va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor=COLORS['gray'], alpha=0.9, linewidth=0.5))

    ax1.set_xlabel('DFTB+ energy per atom (eV)')
    ax1.set_ylabel('Predicted energy per atom (eV)')
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_aspect('equal')
    ax1.legend(loc='lower right', markerscale=1.5)
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    add_panel_label(ax1, 'a', x=-0.16)

    # (b) Feature importance
    y_pos = np.arange(len(top_names))
    bars = ax2.barh(y_pos, top_imp, height=0.65, edgecolor='white', linewidth=0.3)

    # Color bars by type
    for i, name in enumerate(top_names):
        if name.startswith('frac_'):
            bars[i].set_facecolor('#E69F00')
            bars[i].set_alpha(0.8)
        elif name.startswith('is_'):
            bars[i].set_facecolor('#CC79A7')
            bars[i].set_alpha(0.8)
        elif name in ['R']:
            bars[i].set_facecolor('#56B4E9')
            bars[i].set_alpha(0.8)
        else:
            bars[i].set_facecolor('#0072B2')
            bars[i].set_alpha(0.8)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(display_names, fontsize=6.5)
    ax2.invert_yaxis()
    ax2.set_xlabel('Feature importance')
    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))

    # Legend for bar colors
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#0072B2', markersize=6, label='Structural'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#E69F00', markersize=6, label='Composition'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#CC79A7', markersize=6, label='Class'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#56B4E9', markersize=6, label='Size (R)'),
    ]
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=5.5,
              handletextpad=0.3, borderpad=0.3)

    add_panel_label(ax2, 'b', x=-0.22)

    fig.tight_layout(w_pad=3)
    save(fig, 'fig10_ml_prediction', outdir)

    return r2_test, mae_test


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
    if not nc_path.exists():
        print(f"ERROR: {nc_path} not found")
        exit(1)

    nc = pd.read_csv(nc_path)
    print(f"Loaded: {len(nc)} records")

    outdir = base / "figures-ML"
    outdir.mkdir(exist_ok=True)

    print("\n--- Fig 9: Structural space map ---")
    method = fig9_umap(nc, outdir)

    print("\n--- Fig 10: Energy prediction ---")
    r2, mae = fig10_prediction(nc, outdir)

    print(f"\nAll ML figures saved to {outdir}/")
    print(f"Summary: {method} map + GBR prediction (R\u00b2={r2:.3f}, MAE={mae:.2f} eV/atom)")
