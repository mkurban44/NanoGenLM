#!/usr/bin/env python3
"""NanoGenLM — Fig 1 (Pipeline) & Fig 2 (Periodic Table) — Final"""
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BASE = Path(__file__).parent
NC_CSV = BASE / "dataset" / "nature_comm_dataset.csv"
OUTDIR = BASE / "figures"
OUTDIR.mkdir(exist_ok=True)

nc = pd.read_csv(NC_CSV)
print(f"Loaded {len(nc)} records")

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica','Arial','DejaVu Sans'],
    'font.size': 8, 'axes.linewidth': 0.6,
    'figure.dpi': 1200, 'savefig.dpi': 1200,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.08,
})
DC = 180 / 25.4

# ================================================================
# FIGURE 1: Pipeline
# ================================================================
def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(DC, DC * 0.52))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis('off')

    C = {
        'param':'#37474F','comp':'#546E7A','mp':'#4053D3',
        'proto':'#C49B00','relax':'#D55E00','result':'#2E7D32',
        'carve':'#5D4037','output':'#0072B2','arrow':'#888888',
        'annot':'#555555','bg_box':'#F7F7F7',
        'conf_h':'#009E73','conf_m':'#E69F00','conf_l':'#CC79A7',
    }

    def box(x, y, w, h, title, sub='', color='#333'):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25",
                           facecolor=color, edgecolor='white', linewidth=0.6, zorder=2)
        ax.add_patch(b)
        if sub:
            ax.text(x+w/2, y+h/2+1.2, title, ha='center', va='center',
                    fontsize=7.5, fontweight='600', color='white', zorder=3)
            ax.text(x+w/2, y+h/2-1.2, sub, ha='center', va='center',
                    fontsize=5.5, color='white', alpha=0.9, zorder=3)
        else:
            ax.text(x+w/2, y+h/2, title, ha='center', va='center',
                    fontsize=7.5, fontweight='600', color='white', zorder=3)

    def arr(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=C['arrow'], lw=1.0), zorder=1)

    c1, c2, c3 = 12, 38, 64
    bw, bh = 22, 8

    # ROW 1
    y = 50
    box(c1-bw/2, y, bw, bh, 'PTBP Parameters', '75 elements (s, p, d)', C['param'])
    box(c2-bw/2, y, bw, bh, 'Composition Space', '3,184 unique formulas', C['comp'])
    box(c3-bw/2, y, bw, bh, 'Materials Project', '399 DFT-optimized CIFs', C['mp'])

    fx, fy = 78, y+5.5
    ax.text(fx, fy, 'Scientific Filters', fontsize=6.5, fontweight='600', color=C['annot'])
    for i, t in enumerate(['Goldschmidt (t, μ)', 'VEC → BCC / FCC', 'δ < 6.6% (size mismatch)', 'Confidence labeling']):
        ax.text(fx+0.5, fy-1.8*(i+1), f'▸ {t}', fontsize=5, color=C['annot'])

    arr(c1, y, c1, y-4)
    arr(c2, y, c2, y-4)
    arr(c3, y, c3, y-4)

    # ROW 2
    y = 37
    box(c1-bw/2, y, bw, bh, 'Prototype CIFs', '2,785 generated', C['proto'])
    box(c2-bw/2, y, bw, bh, 'Bulk DFTB+ Relax', '3,137 passed (85%)', C['relax'])
    box(c3-bw/2, y, bw, bh, 'Relaxed Unit Cells', '3,157 validated CIFs', C['result'])

    arr(c1+bw/2, y+bh/2, c2-bw/2, y+bh/2)
    arr(c2+bw/2, y+bh/2, c3-bw/2, y+bh/2)

    # Confidence box
    fx2 = 78
    conf_box_y = y - 0.5
    conf_box_h = 8.5
    ax.add_patch(FancyBboxPatch((fx2, conf_box_y), 20, conf_box_h, boxstyle="round,pad=0.2",
                                facecolor=C['bg_box'], edgecolor='#CCC', linewidth=0.5, zorder=2))
    ax.text(fx2+10, y+5.8, 'Confidence', ha='center', fontsize=5.5, fontweight='600', color=C['annot'], zorder=3)
    for i, (lb, pct, col) in enumerate([('High','54%',C['conf_h']),
                                         ('Med','28%',C['conf_m']),
                                         ('Low','18%',C['conf_l'])]):
        bx = fx2+1 + i*6.3
        ax.add_patch(plt.Rectangle((bx, y+1.2), 5.5, 3, facecolor=col, alpha=0.75, zorder=3))
        ax.text(bx+2.75, y+2.7, f'{lb}\n{pct}', ha='center', va='center',
                fontsize=4.2, color='white', fontweight='600', zorder=4)
    arr(c3+bw/2, y+bh/2, fx2, y+bh/2)

    arr(c2, y, c2, y-5)

    # ROW 3
    y = 23
    box(c1-bw/2, y, bw, bh, 'Spherical Carve', '81,423 NPs (R = 5–30 Å)', C['carve'])
    box(c2-bw/2, y, bw, bh, 'NP DFTB+ Relax', '2,488 passed (R = 5, 6 Å)', C['relax'])
    box(c3-bw/2, y, bw, bh, 'Relaxed NPs', '1,173 compositions × 2 R', C['result'])

    arr(c1+bw/2, y+bh/2, c2-bw/2, y+bh/2)
    arr(c2+bw/2, y+bh/2, c3-bw/2, y+bh/2)

    fx3 = 78
    ax.text(fx3, y+5.5, 'Software', fontsize=6.5, fontweight='600', color=C['annot'])
    for i, t in enumerate(['DFTB+ / PTBP', 'Atomsk (supercell + carve)', 'ASE (I/O + analysis)', 'pymatgen (symmetry)']):
        ax.text(fx3+0.5, y+5.5-1.8*(i+1), f'▸ {t}', fontsize=5, color=C['annot'])

    arr(c2, y, c2, y-5)

    # ROW 4
    y = 7
    ow = 40
    box(c2-ow/2, y, ow, 9, 'NanoGenLM Dataset', '2,346 NPs · 1,173 compositions · 3 material classes', C['output'])

    for i, (cls, col, n) in enumerate([('Perovskite','#0072B2','952'),
                                        ('Heusler','#D55E00','1,279'),
                                        ('Hydride','#009E73','257')]):
        dx = 18 + i*24
        ax.add_patch(plt.Circle((dx, 3), 1.2, facecolor=col, alpha=0.85, zorder=3))
        ax.text(dx+2.2, 3, f'{cls} ({n} NPs)', va='center', fontsize=5.5, color=C['annot'], zorder=3)

    for fmt in ['pdf', 'png', 'svg']:
        fig.savefig(OUTDIR / f"fig1_pipeline.{fmt}", format=fmt)
    plt.close(fig)
    print("  Saved: fig1_pipeline")


# ================================================================
# FIGURE 2: Periodic Table
# ================================================================
def fig2_periodic_table():
    PT = {
        'H':(0,0,1),'He':(0,17,2),
        'Li':(1,0,3),'Be':(1,1,4),'B':(1,12,5),'C':(1,13,6),
        'N':(1,14,7),'O':(1,15,8),'F':(1,16,9),'Ne':(1,17,10),
        'Na':(2,0,11),'Mg':(2,1,12),'Al':(2,12,13),'Si':(2,13,14),
        'P':(2,14,15),'S':(2,15,16),'Cl':(2,16,17),'Ar':(2,17,18),
        'K':(3,0,19),'Ca':(3,1,20),'Sc':(3,2,21),'Ti':(3,3,22),
        'V':(3,4,23),'Cr':(3,5,24),'Mn':(3,6,25),'Fe':(3,7,26),
        'Co':(3,8,27),'Ni':(3,9,28),'Cu':(3,10,29),'Zn':(3,11,30),
        'Ga':(3,12,31),'Ge':(3,13,32),'As':(3,14,33),'Se':(3,15,34),
        'Br':(3,16,35),'Kr':(3,17,36),
        'Rb':(4,0,37),'Sr':(4,1,38),'Y':(4,2,39),'Zr':(4,3,40),
        'Nb':(4,4,41),'Mo':(4,5,42),'Tc':(4,6,43),'Ru':(4,7,44),
        'Rh':(4,8,45),'Pd':(4,9,46),'Ag':(4,10,47),'Cd':(4,11,48),
        'In':(4,12,49),'Sn':(4,13,50),'Sb':(4,14,51),'Te':(4,15,52),
        'I':(4,16,53),'Xe':(4,17,54),
        'Cs':(5,0,55),'Ba':(5,1,56),'La':(5,2,57),
        'Hf':(5,3,72),'Ta':(5,4,73),'W':(5,5,74),
        'Re':(5,6,75),'Os':(5,7,76),'Ir':(5,8,77),'Pt':(5,9,78),
        'Au':(5,10,79),'Hg':(5,11,80),'Tl':(5,12,81),'Pb':(5,13,82),
        'Bi':(5,14,83),'Po':(5,15,84),'At':(5,16,85),'Rn':(5,17,86),
        'Ra':(6,1,88),'Th':(6,3,90),
    }

    PTBP = {
        'H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg',
        'Al','Si','P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr',
        'Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr',
        'Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd',
        'In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Lu',
        'Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb',
        'Bi','Po','At','Rn','Ra','Th'
    }

    counts = Counter()
    for s in nc['elements'].dropna():
        for el in s.split(';'):
            el = el.strip()
            if el:
                counts[el] += 1
    max_c = max(counts.values()) if counts else 1

    fig, ax = plt.subplots(figsize=(DC, DC * 0.42))
    ax.set_xlim(-1, 19)
    ax.set_ylim(-0.5, 8.8)
    ax.invert_yaxis()
    ax.axis('off')
    ax.set_aspect('equal')

    cmap = plt.cm.YlOrRd
    c_ptbp_only = '#DAEAF6'
    c_none = '#F2F2F2'

    for sym, (row, col, Z) in PT.items():
        x, y = col, row
        cnt = counts.get(sym, 0)
        in_ptbp = sym in PTBP
        in_data = cnt > 0

        if in_data:
            intensity = np.log1p(cnt) / np.log1p(max_c)
            fc = cmap(0.15 + 0.80 * intensity)
            tc = 'white' if intensity > 0.45 else '#222'
        elif in_ptbp:
            fc = c_ptbp_only
            tc = '#444'
        else:
            fc = c_none
            tc = '#CCC'

        ec = 'white' if in_data else '#DDD'
        ax.add_patch(plt.Rectangle((x+0.04, y+0.04), 0.92, 0.92,
                                    facecolor=fc, edgecolor=ec, linewidth=0.4, zorder=2))
        ax.text(x+0.10, y+0.15, str(Z), fontsize=2.8, color=tc, alpha=0.6, va='top', ha='left', zorder=3)
        fs = 7 if len(sym)==1 else 5.8
        fw = '700' if in_data else '400'
        ax.text(x+0.5, y+0.46, sym, fontsize=fs, fontweight=fw,
                ha='center', va='center', color=tc, zorder=3)
        if in_data:
            ax.text(x+0.5, y+0.82, str(cnt), fontsize=3.2, color=tc, alpha=0.75,
                    ha='center', va='center', zorder=3)

    ax.text(0.5, -0.3, 's-block', fontsize=6, color='#AAA', ha='center', fontweight='500')
    ax.text(6.5, 2.5, 'd-block', fontsize=6, color='#AAA', ha='center', fontweight='500')
    ax.text(14.5, 0.0, 'p-block', fontsize=6, color='#AAA', ha='center', fontweight='500')

    ly = 7.5
    items = [
        (cmap(0.85), 'High usage in dataset'),
        (cmap(0.30), 'Low usage in dataset'),
        (c_ptbp_only, 'PTBP covered, not used'),
        (c_none, 'Not in PTBP'),
    ]
    for i, (col, label) in enumerate(items):
        lx = 0.5 + i * 4.7
        ax.add_patch(plt.Rectangle((lx, ly), 0.55, 0.55,
                                    facecolor=col, edgecolor='#AAA', linewidth=0.3, zorder=3))
        ax.text(lx + 0.75, ly + 0.28, label, va='center', fontsize=5, color='#555', zorder=3)

    n_used = len([e for e in counts if counts[e] > 0])
    ax.text(9.25, -0.3, f'PTBP: 75 elements   |   In dataset: {n_used} elements   |   '
            f'Total NPs: {len(nc):,}',
            ha='center', fontsize=5.5, color='#777')

    for fmt in ['pdf', 'png', 'svg']:
        fig.savefig(OUTDIR / f"fig2_periodic_table.{fmt}", format=fmt)
    plt.close(fig)
    print("  Saved: fig2_periodic_table")


fig1_pipeline()
fig2_periodic_table()
print("\nDone!")
