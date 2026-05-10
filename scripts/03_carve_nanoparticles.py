#!/usr/bin/env python3
"""
NanoGenLM — Step 03: Nanoparticle Carve + DFTB Setup
Carves spherical NPs from relaxed bulk CIFs, prepares DFTB input files.

R = 5, 6, 7 Å       → carve + dftb_in.hsd (will be relaxed)
R = 8, 9, ..., 30 Å  → carve + dftb_in.hsd (files only, for future use)

Usage:
    python3 03_carve_nanoparticles.py --base . [--workers 10]
"""
import os, sys, csv, json, subprocess, argparse, time, math
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from ase.io import read, write

SK_PREFIX = "/mnt/d/DFTB_shared/skfiles/ParameterSets/ptbp/complete_set/"
ATOMSK_BIN = "atomsk"
MARGIN = 5.0  # Å

# Radii
R_RELAX = [5, 6, 7]
R_FUTURE = list(range(8, 31))
R_ALL = R_RELAX + R_FUTURE


def get_supercell_dims(a, b, c, r_max):
    """Calculate supercell dimensions to cover sphere of radius r_max + margin."""
    L = 2 * r_max + MARGIN
    nx = max(4, math.ceil(L / a))
    ny = max(4, math.ceil(L / b))
    nz = max(4, math.ceil(L / c))
    return nx, ny, nz


def write_np_dftb_hsd(work_dir, elements, ang_mom_dict, is_metallic=False):
    """Write DFTB input for nanoparticle (non-periodic, no k-points)."""
    lines = []
    for el in sorted(elements):
        lines.append(f'    {el} = "{ang_mom_dict[el]}"')
    ang_mom_block = "\n".join(lines)
    filling_temp = 1000.0 if is_metallic else 600.0

    hsd = f"""Geometry = GenFormat {{
  <<< "np.gen"
}}
Hamiltonian = DFTB {{
  SCC = Yes
  MaxSCCIterations = 1500
  SCCTolerance = 1e-6
  SlaterKosterFiles = Type2FileNames {{
    Prefix    = "{SK_PREFIX}"
    Separator = "-"
    Suffix    = ".skf"
  }}
  MaxAngularMomentum {{
{ang_mom_block}
  }}
  Filling = Fermi {{
    Temperature [Kelvin] = {filling_temp}
  }}
  Mixer = Broyden {{
    MixingParameter = 0.05
  }}
}}
Driver = GeometryOptimization {{
  Optimizer = Rational {{}}
  MovedAtoms = 1:-1
  MaxSteps = 2000
  Convergence {{
    GradElem = 1e-3
    DispElem = 1e-3
  }}
}}
Options {{
  WriteChargesAsText = Yes
  WriteResultsTag = Yes
}}
ParserOptions {{
  IgnoreUnprocessedNodes = Yes
}}
"""
    (work_dir / "dftb_in.hsd").write_text(hsd)


def carve_single(args):
    """Carve NPs at all radii for a single composition."""
    formula, cls, cif_path, ang_mom_str, confidence, np_base = args

    ang_mom = json.loads(ang_mom_str)
    elements = sorted(ang_mom.keys())
    is_metallic = cls in ("hea", "heusler")

    result = {
        "formula": formula, "class": cls, "confidence": confidence,
        "status": "ok", "reason": "",
        "radii_done": [], "atoms_per_radius": {},
    }

    try:
        # 1. Read relaxed CIF to get lattice params
        atoms_bulk = read(str(cif_path))
        cell = atoms_bulk.get_cell()
        a = np.linalg.norm(cell[0])
        b = np.linalg.norm(cell[1])
        c = np.linalg.norm(cell[2])

        # 2. Calculate max supercell needed (for R=30)
        r_max = max(R_ALL)
        nx, ny, nz = get_supercell_dims(a, b, c, r_max)

        # 3. Create supercell with atomsk
        comp_dir = Path(np_base) / cls / formula
        comp_dir.mkdir(parents=True, exist_ok=True)
        supercell_file = comp_dir / "supercell.xsf"

        proc = subprocess.run(
            [ATOMSK_BIN, str(cif_path), "-duplicate", str(nx), str(ny), str(nz),
             str(supercell_file)],
            capture_output=True, text=True, timeout=120
        )
        if proc.returncode != 0 and not supercell_file.exists():
            result["status"] = "failed"
            result["reason"] = f"atomsk_supercell: {proc.stderr[-200:]}"
            return result

        # 4. Carve spheres at each radius
        for R in R_ALL:
            tag = f"R{R:02d}"
            r_dir = comp_dir / tag
            r_dir.mkdir(parents=True, exist_ok=True)

            xyz_file = r_dir / f"np_{tag}.xyz"
            cfg_file = r_dir / f"np_{tag}.cfg"

            # atomsk: select atoms outside sphere, remove them
            proc = subprocess.run(
                [ATOMSK_BIN, str(supercell_file),
                 "-select", "out", "sphere", "0.5*box", "0.5*box", "0.5*box", str(R),
                 "-rmatom", "select",
                 str(cfg_file), "xyz"],
                capture_output=True, text=True, timeout=60
            )

            # Read xyz, write gen (non-periodic)
            if xyz_file.exists():
                try:
                    np_atoms = read(str(xyz_file))
                    n_atoms = len(np_atoms)

                    if n_atoms < 3:
                        # Too few atoms, skip
                        continue

                    np_atoms.pbc = False
                    write(str(r_dir / "np.gen"), np_atoms, format="gen")

                    # Write dftb_in.hsd
                    write_np_dftb_hsd(r_dir, elements, ang_mom, is_metallic)

                    # Write atom count
                    (r_dir / "N.txt").write_text(f"{n_atoms}\n")

                    result["radii_done"].append(R)
                    result["atoms_per_radius"][str(R)] = n_atoms

                except Exception as e:
                    result["reason"] += f"R{R}:{str(e)[:50]}; "
            else:
                result["reason"] += f"R{R}:no_xyz; "

        # Clean up supercell (large file)
        if supercell_file.exists():
            supercell_file.unlink()

    except subprocess.TimeoutExpired:
        result["status"] = "failed"
        result["reason"] = "timeout"
    except Exception as e:
        result["status"] = "failed"
        result["reason"] = str(e)[:300]

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=".")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--class-filter", default="")
    args = parser.parse_args()

    base = Path(args.base)
    np_base = base / "nanoparticles"
    np_base.mkdir(exist_ok=True)

    # Read bulk relax results to get passed + suspect
    relax_csv = base / "bulk_relax_results_v2.csv"
    master_csv = base / "compositions_master_v2.csv"

    if not relax_csv.exists():
        print(f"ERROR: {relax_csv} not found")
        sys.exit(1)
    if not master_csv.exists():
        print(f"ERROR: {master_csv} not found")
        sys.exit(1)

    # Load ang_mom from master CSV
    ang_mom_map = {}
    with open(master_csv) as f:
        for row in csv.DictReader(f):
            ang_mom_map[row["formula"]] = row["ang_mom"]

    # Load relax results — only passed + suspect
    jobs = []
    with open(relax_csv) as f:
        for row in csv.DictReader(f):
            if row["status"] not in ("passed", "suspect"):
                continue

            cls = row["class"]
            formula = row["formula"]

            if args.class_filter and cls != args.class_filter:
                continue

            # Find relaxed CIF
            cif_path = base / "unit_cells_relaxed" / cls / f"{formula}.cif"
            if not cif_path.exists():
                continue

            # Skip if already done (check if R07 directory has np.gen)
            if (np_base / cls / formula / "R07" / "np.gen").exists():
                continue

            ang_mom = ang_mom_map.get(formula, "")
            if not ang_mom:
                continue

            confidence = row.get("confidence_update", "medium")
            jobs.append((formula, cls, str(cif_path), ang_mom, confidence, str(np_base)))

    if args.limit > 0:
        jobs = jobs[:args.limit]

    total = len(jobs)
    if total == 0:
        print("No jobs to run.")
        return

    print(f"{'='*60}")
    print(f"NanoGenLM — Nanoparticle Carve")
    print(f"{'='*60}")
    print(f"  Compositions: {total}")
    print(f"  Workers:      {args.workers}")
    print(f"  Radii:        {R_ALL[0]}-{R_ALL[-1]} A ({len(R_ALL)} sizes)")
    print(f"  Relax radii:  {R_RELAX}")
    print(f"  Future radii: {R_FUTURE[0]}-{R_FUTURE[-1]}")
    print(f"  Margin:       {MARGIN} A")
    print(f"  Output:       {np_base}/")
    print(f"{'='*60}")

    results = []
    ok = failed = 0
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(carve_single, job): job[0] for job in jobs}

        for i, future in enumerate(as_completed(futures), 1):
            formula = futures[future]
            try:
                res = future.result()
            except Exception as e:
                res = {"formula": formula, "status": "failed",
                       "reason": str(e)[:200], "radii_done": [], "atoms_per_radius": {}}
            results.append(res)

            st = res.get("status", "failed")
            if st == "ok":
                ok += 1
            else:
                failed += 1

            elapsed = time.time() - t_start
            rate = i / elapsed if elapsed > 0 else 0
            eta = (total - i) / rate / 60 if rate > 0 else 0

            n_radii = len(res.get("radii_done", []))
            atoms_r7 = res.get("atoms_per_radius", {}).get("7", "?")
            mk = "OK" if st == "ok" else "XX"
            rsn = res.get("reason", "")[:40]

            if i % 50 == 0 or i == total or st != "ok":
                print(f"  [{i:>5d}/{total}] {mk} {res.get('class',''):>10s}/{formula:<18s} "
                      f"radii={n_radii:>2d}/{len(R_ALL)} n@R7={atoms_r7} "
                      f"({rate*3600:.0f}/hr ETA:{eta:.0f}m) {rsn}")

    # Write summary CSV
    csv_path = base / "carve_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["formula", "class", "confidence", "status", "reason",
                     "n_radii_done"] + [f"atoms_R{r:02d}" for r in R_ALL])
        for res in results:
            atoms = res.get("atoms_per_radius", {})
            w.writerow([
                res.get("formula", ""),
                res.get("class", ""),
                res.get("confidence", ""),
                res.get("status", ""),
                res.get("reason", ""),
                len(res.get("radii_done", [])),
            ] + [atoms.get(str(r), "") for r in R_ALL])

    tt = time.time() - t_start

    # Count total NP files created
    total_np = 0
    total_relax_np = 0
    for res in results:
        radii = res.get("radii_done", [])
        total_np += len(radii)
        total_relax_np += len([r for r in radii if r in R_RELAX])

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  Compositions OK:   {ok}")
    print(f"  Compositions fail: {failed}")
    print(f"  Total NP created:  {total_np}")
    print(f"  NP for relax:      {total_relax_np} (R={R_RELAX})")
    print(f"  NP future only:    {total_np - total_relax_np} (R={R_FUTURE[0]}-{R_FUTURE[-1]})")
    print(f"  Time: {tt/60:.1f} min")

    # Per-class summary
    print(f"\n  By class:")
    for cls in ["perovskite", "heusler", "hea", "hydride"]:
        sub = [r for r in results if r.get("class") == cls and r.get("status") == "ok"]
        if not sub:
            continue
        n_comp = len(sub)
        # Average atoms at R=7
        r7_atoms = [r.get("atoms_per_radius", {}).get("7", 0) for r in sub]
        r7_atoms = [x for x in r7_atoms if x > 0]
        avg_r7 = np.mean(r7_atoms) if r7_atoms else 0
        print(f"    {cls:<12s}  compositions={n_comp:>5d}  avg_atoms@R7={avg_r7:.0f}")

    print(f"\n  Summary CSV: {csv_path}")
    print(f"  NP directory: {np_base}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
