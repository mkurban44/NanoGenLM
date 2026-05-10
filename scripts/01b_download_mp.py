#!/usr/bin/env python3
"""
NanoGenLM — Step 01b: Download structures from Materials Project (REST API)
No mp-api dependency — uses requests directly.

Usage:
    export MP_API_KEY="your_key"
    python3 01b_download_mp.py --base .
"""
import os, sys, csv, json, argparse, time, shutil
import numpy as np
import requests
from pathlib import Path
from pymatgen.core import Structure, Lattice
from pymatgen.io.cif import CifWriter


MP_BASE = "https://api.materialsproject.org"


def mp_search(formula, api_key):
    """Search MP for a formula, return best structure or None."""
    url = f"{MP_BASE}/materials/summary/"
    headers = {"X-API-KEY": api_key}
    params = {
        "formula": formula,
        "_fields": "material_id,structure,energy_above_hull,formation_energy_per_atom,is_stable",
        "_limit": 5,
    }
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code == 429:
            return "rate_limited"
        if resp.status_code != 200:
            return None
        data = resp.json().get("data", [])
        if not data:
            return None
        # Pick lowest energy_above_hull
        best = min(data, key=lambda d: d.get("energy_above_hull", 999) or 999)
        struct = Structure.from_dict(best["structure"])
        return {
            "structure": struct,
            "mp_id": best.get("material_id", ""),
            "e_above_hull": best.get("energy_above_hull"),
            "formation_e": best.get("formation_energy_per_atom"),
            "is_stable": best.get("is_stable"),
        }
    except Exception as e:
        return None


def make_hea_struct(elements, phase):
    """Generate HEA supercell structure."""
    metallic_r = {
        "Li": 1.52, "Al": 1.43, "Si": 1.18, "Ti": 1.47,
        "V": 1.34, "Cr": 1.28, "Mn": 1.27, "Fe": 1.26, "Co": 1.25,
        "Ni": 1.24, "Cu": 1.28, "Zn": 1.34, "Ga": 1.35, "Ge": 1.22,
        "Zr": 1.60, "Nb": 1.46, "Mo": 1.39, "Ru": 1.34,
        "Pd": 1.37, "Ag": 1.44, "Hf": 1.59, "Ta": 1.46, "W": 1.39,
        "Pt": 1.39, "Au": 1.44,
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

    n_el = len(elements)
    if phase == "bcc":
        a_vals = [element_bcc_a.get(el) for el in elements]
        if None in a_vals:
            r_vals = [metallic_r.get(el, 1.35) for el in elements]
            a_prim = np.mean(r_vals) * 4 / np.sqrt(3)
        else:
            a_prim = np.mean(a_vals)
        a_s = a_prim * 2
        coords = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    coords.append([i/2, j/2, k/2])
                    coords.append([(i+.5)/2, (j+.5)/2, (k+.5)/2])
        n_sites = 16
    else:
        a_vals = [element_fcc_a.get(el) for el in elements]
        if None in a_vals:
            r_vals = [metallic_r.get(el, 1.35) for el in elements]
            a_prim = np.mean(r_vals) * 2 * np.sqrt(2)
        else:
            a_prim = np.mean(a_vals)
        a_s = a_prim * 2
        coords = []
        fcc = [[0,0,0],[.5,.5,0],[.5,0,.5],[0,.5,.5]]
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for b in fcc:
                        coords.append([(i+b[0])/2, (j+b[1])/2, (k+b[2])/2])
        n_sites = 32

    np.random.seed(hash(tuple(sorted(elements))) % 2**31)
    species = []
    per = n_sites // n_el
    rem = n_sites % n_el
    for idx, el in enumerate(elements):
        species.extend([el] * (per + (1 if idx < rem else 0)))
    np.random.shuffle(species)

    lattice = Lattice.cubic(a_s)
    return Structure(lattice, species, coords), n_sites


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=".")
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("MP_API_KEY")
    if not api_key:
        print("ERROR: Set MP_API_KEY or use --api-key")
        sys.exit(1)

    base = Path(args.base)
    master_csv = base / "compositions_master.csv"
    if not master_csv.exists():
        print(f"ERROR: {master_csv} not found")
        sys.exit(1)

    mp_dir = base / "unit_cells_mp"
    for cls in ["perovskite", "heusler", "hea", "hydride"]:
        (mp_dir / cls).mkdir(parents=True, exist_ok=True)

    rows = []
    with open(master_csv) as f:
        for row in csv.DictReader(f):
            rows.append(row)

    print(f"{'='*60}")
    print(f"NanoGenLM — Materials Project Download (REST API)")
    print(f"{'='*60}")

    # Separate HEA vs non-HEA
    non_hea = [r for r in rows if r["class"] != "hea"]
    hea_rows = [r for r in rows if r["class"] == "hea"]

    # Deduplicate formulas
    unique = {}
    for r in non_hea:
        if r["formula"] not in unique:
            unique[r["formula"]] = r

    total = len(unique)
    print(f"  Non-HEA formulas to search: {total}")
    print(f"  HEA formulas to generate:   {len(hea_rows)}")

    # Search MP
    mp_found = {}
    not_found = []
    errors = 0
    t0 = time.time()

    for i, (formula, row) in enumerate(unique.items(), 1):
        result = mp_search(formula, api_key)

        if result == "rate_limited":
            print(f"  [{i}/{total}] Rate limited, waiting 60s...")
            time.sleep(60)
            result = mp_search(formula, api_key)

        if result and result != "rate_limited":
            mp_found[formula] = result
            mark = "MP"
        else:
            not_found.append(formula)
            mark = "--"

        if i % 50 == 0 or i == total:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (total - i) / rate / 60 if rate > 0 else 0
            print(f"  [{i:>5d}/{total}] found={len(mp_found)} miss={len(not_found)} "
                  f"({rate:.1f}/s ETA:{eta:.0f}m)")

        time.sleep(0.15)  # ~6 req/s to stay under rate limit

    # Write MP CIFs
    print(f"\nWriting CIF files...")
    all_results = []

    for formula, data in mp_found.items():
        row = unique[formula]
        cls = row["class"]
        cif_path = mp_dir / cls / f"{formula}.cif"
        try:
            CifWriter(data["structure"]).write_file(str(cif_path))
            all_results.append({
                "formula": formula, "class": cls, "source": "materials_project",
                "mp_id": data["mp_id"],
                "e_above_hull": data["e_above_hull"],
                "formation_e": data["formation_e"],
                "is_stable": data["is_stable"],
                "confidence": "high",
                "n_atoms_cell": len(data["structure"]),
                "sg": data["structure"].get_space_group_info()[0],
                "elements": row["elements"], "ang_mom": row["ang_mom"],
            })
        except Exception as e:
            not_found.append(formula)

    # Fallback: copy prototype CIFs for not-found
    proto_count = 0
    for formula in not_found:
        row = unique.get(formula)
        if not row:
            continue
        cls = row["class"]
        src = base / "unit_cells" / cls / f"{formula}.cif"
        dst = mp_dir / cls / f"{formula}.cif"
        if src.exists() and not dst.exists():
            shutil.copy2(str(src), str(dst))
            proto_count += 1
            all_results.append({
                "formula": formula, "class": cls, "source": "prototype",
                "mp_id": "", "e_above_hull": "", "formation_e": "",
                "is_stable": "", "confidence": row.get("confidence", "medium"),
                "n_atoms_cell": row.get("n_atoms_cell", ""),
                "sg": row.get("sg", ""),
                "elements": row["elements"], "ang_mom": row["ang_mom"],
            })

    # Generate HEA
    print(f"\nGenerating {len(hea_rows)} HEA structures...")
    for row in hea_rows:
        formula = row["formula"]
        elements = row["elements"].split(";")
        phase = row.get("predicted_phase", "bcc")
        if phase == "mixed":
            phase = "bcc"

        struct, n_sites = make_hea_struct(elements, phase)
        fname = f"{formula}_{phase}.cif"
        CifWriter(struct).write_file(str(mp_dir / "hea" / fname))

        all_results.append({
            "formula": formula, "class": "hea", "source": "prototype",
            "mp_id": "", "e_above_hull": "", "formation_e": "",
            "is_stable": "", "confidence": row.get("confidence", "medium"),
            "n_atoms_cell": n_sites,
            "sg": f"{phase.upper()}_SQS",
            "elements": row["elements"], "ang_mom": row["ang_mom"],
            "predicted_phase": phase, "cif_filename": fname,
        })

    # Write master CSV v2
    csv_out = base / "compositions_master_v2.csv"
    fieldnames = [
        "formula", "class", "source", "mp_id", "e_above_hull",
        "formation_e", "is_stable", "confidence", "n_atoms_cell",
        "sg", "elements", "ang_mom", "predicted_phase", "cif_filename"
    ]
    with open(csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_results)

    # Summary
    mp_count = sum(1 for r in all_results if r["source"] == "materials_project")
    proto_total = sum(1 for r in all_results if r["source"] == "prototype")

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Materials Project:  {mp_count}")
    print(f"  Prototype:          {proto_total}")
    print(f"  Total:              {len(all_results)}")
    print(f"\n  By class:")
    for cls in ["perovskite", "heusler", "hea", "hydride"]:
        n = sum(1 for r in all_results if r["class"] == cls)
        n_mp = sum(1 for r in all_results if r["class"] == cls and r["source"] == "materials_project")
        print(f"    {cls:<12s}  total={n:>5d}  MP={n_mp:>5d}  proto={n-n_mp:>5d}")
    print(f"\n  Output: {mp_dir}/")
    print(f"  Master CSV: {csv_out}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
