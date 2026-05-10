#!/usr/bin/env python3
"""
NanoGenLM — Step 01: Unit Cell CIF Generation
Generates CIF files for 4 material classes with scientific filters and confidence labels.

Filters:
  - Perovskite: Goldschmidt tolerance factor (t) + octahedral factor (μ)
  - Heusler: X≠Y, valid radii
  - HEA: VEC (BCC/FCC assignment), atomic size mismatch δ < 6.6%
  - Hydride: tolerance factor + octahedral factor

Output: unit_cells/<class>/<formula>.cif + compositions_master.csv
"""
import numpy as np
import csv, json, sys
from itertools import combinations
from pathlib import Path
from pymatgen.core import Structure, Lattice
from pymatgen.io.cif import CifWriter
import warnings
warnings.filterwarnings("ignore")

# ================================================================
# OUTPUT DIRECTORY
# ================================================================
BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./NanoGenLM")
UC_DIR = BASE / "unit_cells"
for cls in ["perovskite", "heusler", "hea", "hydride"]:
    (UC_DIR / cls).mkdir(parents=True, exist_ok=True)

# ================================================================
# RADII & CONSTANTS
# ================================================================
shannon_XII = {
    "Li": 1.06, "Na": 1.39, "K": 1.64, "Rb": 1.72, "Cs": 1.88,
    "Ca": 1.34, "Sr": 1.44, "Ba": 1.61,
}

shannon_VI = {
    "Li": 0.760, "Na": 1.020, "Mg": 0.720, "Ca": 1.000, "Sr": 1.180, "Ba": 1.350,
    "Sc": 0.745, "Ti": 0.605, "V": 0.640, "Cr": 0.615, "Mn": 0.645,
    "Fe": 0.645, "Co": 0.545, "Ni": 0.690, "Cu": 0.730, "Zn": 0.740,
    "Y": 0.900, "Zr": 0.720, "Nb": 0.640, "Mo": 0.650,
    "Ru": 0.620, "Rh": 0.665, "Pd": 0.860,
    "Hf": 0.710, "Ta": 0.640, "W": 0.620, "Re": 0.630,
    "Os": 0.630, "Ir": 0.625, "Pt": 0.625,
    "Al": 0.535, "Ga": 0.620, "In": 0.800, "Sn": 0.690, "Sb": 0.600,
    "B": 0.270, "Si": 0.400, "Ge": 0.530,
}

shannon_anion_VI = {"O": 1.40, "F": 1.33, "Cl": 1.81, "N": 1.46, "S": 1.84, "H": 1.40}

metallic_r = {
    "Li": 1.52, "Be": 1.12, "Na": 1.86, "Mg": 1.60, "Al": 1.43,
    "Si": 1.18, "K": 2.27, "Ca": 1.97, "Sc": 1.62, "Ti": 1.47,
    "V": 1.34, "Cr": 1.28, "Mn": 1.27, "Fe": 1.26, "Co": 1.25,
    "Ni": 1.24, "Cu": 1.28, "Zn": 1.34, "Ga": 1.35, "Ge": 1.22,
    "As": 1.21, "Se": 1.16, "Rb": 2.48, "Sr": 2.15, "Y": 1.80,
    "Zr": 1.60, "Nb": 1.46, "Mo": 1.39, "Ru": 1.34,
    "Rh": 1.34, "Pd": 1.37, "Ag": 1.44, "Cd": 1.51, "In": 1.67,
    "Sn": 1.58, "Sb": 1.61, "Hf": 1.59, "Ta": 1.46, "W": 1.39,
    "Re": 1.37, "Os": 1.35, "Ir": 1.36, "Pt": 1.39, "Au": 1.44,
    "Pb": 1.75, "Bi": 1.56,
}

# VEC values for HEA
vec_values = {
    "Al": 3, "Si": 4, "Ti": 4, "V": 5, "Cr": 6, "Mn": 7,
    "Fe": 8, "Co": 9, "Ni": 10, "Cu": 11, "Zn": 12,
    "Zr": 4, "Nb": 5, "Mo": 6, "Hf": 4, "Ta": 5, "W": 6,
    "Pd": 10, "Ag": 11, "Pt": 10, "Au": 11,
}

element_bcc_a = {
    "Ti": 3.31, "V": 3.03, "Cr": 2.88, "Mn": 2.87, "Fe": 2.87,
    "Nb": 3.30, "Mo": 3.15, "Ta": 3.30, "W": 3.16,
    "Zr": 3.58, "Hf": 3.56, "Al": 3.31, "Si": 3.08,
    "Co": 2.82, "Ni": 2.81, "Cu": 2.95,
}
element_fcc_a = {
    "Al": 4.05, "Ti": 4.11, "V": 3.82, "Cr": 3.68,
    "Mn": 3.63, "Fe": 3.65, "Co": 3.54, "Ni": 3.52, "Cu": 3.61,
    "Zr": 4.53, "Nb": 4.17, "Mo": 3.97, "Pd": 3.89, "Ag": 4.09,
    "Hf": 4.47, "Ta": 4.17, "W": 3.98, "Pt": 3.92, "Au": 4.08,
}

# MaxAngularMomentum
s_block = {"H","He","Li","Be","Na","Mg","K","Ca","Rb","Sr","Cs","Ba","Ra"}
p_block = {"B","C","N","O","F","Ne","Al","Si","P","S","Cl","Ar",
           "Ga","Ge","As","Se","Br","Kr","In","Sn","Sb","Te","I","Xe",
           "Tl","Pb","Bi","Po","At","Rn"}
d_block = {"Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
           "Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
           "La","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Th"}

def get_ang_mom(el):
    if el in s_block: return "s"
    if el in p_block: return "p"
    if el in d_block: return "d"
    return "p"

# ================================================================
# STRUCTURE GENERATORS
# ================================================================

def make_perovskite(A, B, X):
    r_A = shannon_XII.get(A)
    r_B = shannon_VI.get(B)
    r_X = shannon_anion_VI.get(X)
    if None in (r_A, r_B, r_X):
        return None, {"reject": "missing_radii"}

    a = 2 * (r_B + r_X)
    t = (r_A + r_X) / (np.sqrt(2) * (r_B + r_X))
    mu = r_B / r_X  # octahedral factor

    # Filters
    if t < 0.71 or t > 1.10:
        return None, {"reject": f"t={t:.3f}"}
    if mu < 0.41 or mu > 0.73:
        return None, {"reject": f"mu={mu:.3f}"}

    # Confidence
    if 0.9 <= t <= 1.0 and 0.45 <= mu <= 0.65:
        conf = "high"
    elif 0.8 <= t <= 1.05:
        conf = "medium"
    else:
        conf = "low"

    lattice = Lattice.cubic(a)
    species = [A, B, X, X, X]
    coords = [[.5,.5,.5],[0,0,0],[.5,0,0],[0,.5,0],[0,0,.5]]
    struct = Structure(lattice, species, coords)
    formula = f"{A}{B}{X}3"
    elements = sorted(set([A, B, X]))
    ang_mom = {el: get_ang_mom(el) for el in elements}

    info = {
        "formula": formula, "class": "perovskite", "sg": "Pm-3m",
        "a": round(a, 4), "n_atoms_cell": 5,
        "t_factor": round(t, 4), "mu_factor": round(mu, 4),
        "confidence": conf, "elements": elements,
        "ang_mom": ang_mom, "reject": None,
    }
    return struct, info


def make_heusler(X, Y, Z):
    if X == Y:
        return None, {"reject": "X==Y"}
    r_X = metallic_r.get(X)
    r_Y = metallic_r.get(Y)
    r_Z = metallic_r.get(Z)
    if None in (r_X, r_Y, r_Z):
        return None, {"reject": "missing_radii"}

    a = (4 / np.sqrt(3)) * np.mean([r_X + r_Y, r_X + r_Z])

    # Confidence based on lattice param range
    if 5.4 <= a <= 6.5:
        conf = "high"
    elif 5.0 <= a <= 7.0:
        conf = "medium"
    else:
        conf = "low"

    lattice = Lattice.cubic(a)
    species = [Y]*4 + [Z]*4 + [X]*8
    coords = [
        [0,0,0],[.5,.5,0],[.5,0,.5],[0,.5,.5],
        [.5,.5,.5],[0,0,.5],[0,.5,0],[.5,0,0],
        [.25,.25,.25],[.75,.75,.25],[.75,.25,.75],[.25,.75,.75],
        [.75,.75,.75],[.25,.25,.75],[.25,.75,.25],[.75,.25,.25],
    ]
    struct = Structure(lattice, species, coords)
    formula = f"{X}2{Y}{Z}"
    elements = sorted(set([X, Y, Z]))
    ang_mom = {el: get_ang_mom(el) for el in elements}

    info = {
        "formula": formula, "class": "heusler", "sg": "Fm-3m",
        "a": round(a, 4), "n_atoms_cell": 16,
        "confidence": conf, "elements": elements,
        "ang_mom": ang_mom, "reject": None,
    }
    return struct, info


def make_hea(elements, n_el=None):
    n_el = len(elements)

    # --- VEC filter → structure type assignment ---
    vecs = [vec_values.get(el) for el in elements]
    if None in vecs:
        return None, {"reject": "missing_VEC"}
    avg_vec = np.mean(vecs)

    if avg_vec >= 8.0:
        stype = "fcc"
    elif avg_vec <= 6.87:
        stype = "bcc"
    else:
        stype = "mixed"
        # Still generate as BCC but flag low confidence
    actual_stype = "bcc" if stype in ("bcc", "mixed") else "fcc"

    # --- Atomic size mismatch δ ---
    radii = [metallic_r.get(el) for el in elements]
    if None in radii:
        return None, {"reject": "missing_radii"}
    r_avg = np.mean(radii)
    delta = 100 * np.sqrt(np.mean([(r/r_avg - 1)**2 for r in radii]))

    if delta > 6.6:
        return None, {"reject": f"delta={delta:.2f}%"}

    # --- Omega parameter (thermodynamic) ---
    # Simplified: just use delta for now
    if delta < 4.0:
        conf = "high"
    elif delta < 5.5:
        conf = "medium"
    else:
        conf = "low"
    if stype == "mixed":
        conf = "low"  # Override: mixed region always low confidence

    # --- Build structure ---
    if actual_stype == "bcc":
        a_vals = [element_bcc_a.get(el) for el in elements]
        if None in a_vals:
            r_vals = [metallic_r.get(el, 1.35) for el in elements]
            a_prim = np.mean(r_vals) * 4 / np.sqrt(3)
        else:
            a_prim = np.mean(a_vals)
        a_s = a_prim * 2
        base = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    base.append([i/2, j/2, k/2])
                    base.append([(i+.5)/2, (j+.5)/2, (k+.5)/2])
        n_sites = 16
    else:
        a_vals = [element_fcc_a.get(el) for el in elements]
        if None in a_vals:
            r_vals = [metallic_r.get(el, 1.35) for el in elements]
            a_prim = np.mean(r_vals) * 2 * np.sqrt(2)
        else:
            a_prim = np.mean(a_vals)
        a_s = a_prim * 2
        base = []
        fcc = [[0,0,0],[.5,.5,0],[.5,0,.5],[0,.5,.5]]
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for b in fcc:
                        base.append([(i+b[0])/2, (j+b[1])/2, (k+b[2])/2])
        n_sites = 32

    np.random.seed(hash(tuple(sorted(elements))) % 2**31)
    species = []
    per = n_sites // n_el
    rem = n_sites % n_el
    for idx, el in enumerate(elements):
        species.extend([el] * (per + (1 if idx < rem else 0)))
    np.random.shuffle(species)

    lattice = Lattice.cubic(a_s)
    struct = Structure(lattice, species, base)
    formula = "".join(sorted(elements))
    els = sorted(set(elements))
    ang_mom = {el: get_ang_mom(el) for el in els}

    info = {
        "formula": formula, "class": "hea", "sg": f"{actual_stype.upper()}_SQS",
        "a": round(a_prim, 4), "a_super": round(a_s, 4),
        "n_atoms_cell": n_sites,
        "vec": round(avg_vec, 2), "delta_percent": round(delta, 2),
        "predicted_phase": stype,
        "confidence": conf, "elements": els,
        "ang_mom": ang_mom, "reject": None,
    }
    return struct, info


# ================================================================
# COMPOSITION SPACES
# ================================================================

A_pero = ["Li", "Na", "K", "Rb", "Cs", "Ca", "Sr", "Ba"]
B_pero = ["Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
          "Y","Zr","Nb","Mo","Ru","Rh","Pd","Hf","Ta","W","Re","Os","Ir","Pt"]
X_pero = ["O", "F", "N", "S", "Cl"]

X_heus = ["Fe","Co","Ni","Cu","Mn","Cr","Ti","V"]
Y_heus = ["Ti","V","Cr","Mn","Fe","Co","Ni","Zr","Nb","Mo","Hf","Ta","W"]
Z_heus = ["Al","Si","Ga","Ge","Sn","Sb","In","Bi"]

hea_pool = ["Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zr","Nb","Mo","Hf","Ta","W","Al"]

A_hyd = ["Li", "Na", "K", "Rb", "Cs", "Ca", "Sr", "Ba"]
B_hyd = ["Mg","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
         "Y","Zr","Nb","Mo","Pd","Al","Ga","In","Sn"]

# ================================================================
# GENERATE ALL
# ================================================================

master = []  # all records (accepted + rejected)
accepted = 0
rejected = 0
stats = {"perovskite": [0,0], "heusler": [0,0], "hea": [0,0], "hydride": [0,0]}

# --- 1. Perovskites ---
print("Generating perovskites...")
for A in A_pero:
    for B in B_pero:
        for X in X_pero:
            struct, info = make_perovskite(A, B, X)
            if struct is None:
                rejected += 1
                stats["perovskite"][1] += 1
                continue
            fname = f"{info['formula']}.cif"
            CifWriter(struct).write_file(str(UC_DIR / "perovskite" / fname))
            master.append(info)
            accepted += 1
            stats["perovskite"][0] += 1

print(f"  Perovskite: {stats['perovskite'][0]} accepted, {stats['perovskite'][1]} rejected")

# --- 2. Heusler ---
print("Generating Heuslers...")
for X in X_heus:
    for Y in Y_heus:
        for Z in Z_heus:
            struct, info = make_heusler(X, Y, Z)
            if struct is None:
                rejected += 1
                stats["heusler"][1] += 1
                continue
            fname = f"{info['formula']}.cif"
            CifWriter(struct).write_file(str(UC_DIR / "heusler" / fname))
            master.append(info)
            accepted += 1
            stats["heusler"][0] += 1

print(f"  Heusler: {stats['heusler'][0]} accepted, {stats['heusler'][1]} rejected")

# --- 3. HEA (4 and 5 element) ---
print("Generating HEAs...")
for n_combo in [4, 5]:
    for combo in combinations(hea_pool, n_combo):
        struct, info = make_hea(list(combo))
        if struct is None:
            rejected += 1
            stats["hea"][1] += 1
            continue
        stype = info["predicted_phase"]
        if stype == "mixed":
            stype = "bcc"
        fname = f"{''.join(sorted(combo))}_{stype}.cif"
        CifWriter(struct).write_file(str(UC_DIR / "hea" / fname))
        master.append(info)
        accepted += 1
        stats["hea"][0] += 1

print(f"  HEA: {stats['hea'][0]} accepted, {stats['hea'][1]} rejected")

# --- 4. Hydrides ---
print("Generating hydrides...")
for A in A_hyd:
    for B in B_hyd:
        struct, info = make_perovskite(A, B, "H")
        if struct is None:
            rejected += 1
            stats["hydride"][1] += 1
            continue
        info["class"] = "hydride"
        info["formula"] = f"{A}{B}H3"
        fname = f"{info['formula']}.cif"
        CifWriter(struct).write_file(str(UC_DIR / "hydride" / fname))
        master.append(info)
        accepted += 1
        stats["hydride"][0] += 1

print(f"  Hydride: {stats['hydride'][0]} accepted, {stats['hydride'][1]} rejected")

# ================================================================
# WRITE MASTER CSV
# ================================================================
csv_path = BASE / "compositions_master.csv"
fieldnames = [
    "formula", "class", "sg", "a", "n_atoms_cell",
    "t_factor", "mu_factor", "vec", "delta_percent", "predicted_phase",
    "confidence", "elements", "ang_mom"
]

with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader()
    for rec in master:
        row = {k: rec.get(k, "") for k in fieldnames}
        row["elements"] = ";".join(rec.get("elements", []))
        row["ang_mom"] = json.dumps(rec.get("ang_mom", {}))
        w.writerow(row)

# ================================================================
# SUMMARY
# ================================================================
print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"  Total accepted:  {accepted}")
print(f"  Total rejected:  {rejected}")
print(f"  Acceptance rate: {accepted/(accepted+rejected)*100:.1f}%")
print(f"\n  By class:")
for cls in ["perovskite", "heusler", "hea", "hydride"]:
    a, r = stats[cls]
    print(f"    {cls:<12s}  {a:>5d} accepted  {r:>5d} rejected  ({a/(a+r)*100:.0f}%)")

# Confidence distribution
conf_dist = {"high": 0, "medium": 0, "low": 0}
for rec in master:
    conf_dist[rec["confidence"]] += 1
print(f"\n  Confidence distribution:")
for c in ["high", "medium", "low"]:
    print(f"    {c:<8s}  {conf_dist[c]:>5d}  ({conf_dist[c]/accepted*100:.1f}%)")

print(f"\n  Output: {UC_DIR}/")
print(f"  Master CSV: {csv_path}")
print(f"{'='*60}")
