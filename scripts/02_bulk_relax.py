#!/usr/bin/env python3
"""
NanoGenLM — Step 02: Bulk DFTB+ Relaxation (v3)
Uses unit_cells_mp/ and compositions_master_v2.csv

Usage:
    python3 02_bulk_relax.py --base . --workers 10 [--limit 10]
"""
import os, sys, json, csv, subprocess, argparse, time
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from ase.io import read, write

SK_PREFIX = "/mnt/d/DFTB_shared/skfiles/ParameterSets/ptbp/complete_set/"
DFTB_BIN = "dftb+"
MAX_VOLUME_CHANGE = 0.30


def make_dftb_hsd(elements, ang_mom_dict, is_metallic=False, n_atoms=5):
    lines = []
    for el in sorted(elements):
        lines.append(f'    {el} = "{ang_mom_dict[el]}"')
    ang_mom_block = "\n".join(lines)
    filling_temp = 1000.0 if is_metallic else 600.0

    # Auto k-points based on number of atoms
    if n_atoms > 30:
        kpoints = """  KPointsAndWeights = SupercellFolding {
    1 0 0
    0 1 0
    0 0 1
    0.5 0.5 0.5
  }"""
    elif n_atoms > 10:
        kpoints = """  KPointsAndWeights = SupercellFolding {
    2 0 0
    0 2 0
    0 0 2
    0.5 0.5 0.5
  }"""
    else:
        kpoints = """  KPointsAndWeights = SupercellFolding {
    4 0 0
    0 4 0
    0 0 4
    0.5 0.5 0.5
  }"""

    return f"""Geometry = GenFormat {{
  <<< "geo.gen"
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
{kpoints}
}}
Driver = GeometryOptimization {{
  Optimizer = Rational {{}}
  MovedAtoms = 1:-1
  MaxSteps = 500
  LatticeOpt = Yes
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


def run_single_relax(args):
    formula, cls, cif_path, ang_mom_str, work_base, source = args
    ang_mom = json.loads(ang_mom_str)
    elements = sorted(ang_mom.keys())
    is_metallic = cls in ("hea", "heusler")
    is_supercell = cls == "hea"
    work_dir = Path(work_base) / cls / formula
    work_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "formula": formula, "class": cls, "source": source,
        "status": "failed", "reason": "",
        "energy_eV": None, "energy_per_atom_eV": None,
        "a_initial": None, "a_relaxed": None,
        "volume_change_pct": None, "confidence_update": None,
        "n_atoms": None, "elapsed_sec": None,
    }
    t0 = time.time()

    try:
        atoms_raw = read(str(cif_path))
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.core import Structure as PMGStruct
        pmg = AseAtomsAdaptor.get_structure(atoms_raw)
        pmg_prim = pmg.get_primitive_structure(tolerance=0.1)
        atoms = AseAtomsAdaptor.get_atoms(pmg_prim)
        n_atoms = len(atoms)
        result["n_atoms"] = n_atoms
        vol_init = atoms.get_volume()
        cell_init = atoms.get_cell()
        a_init = np.mean([np.linalg.norm(cell_init[i]) for i in range(3)])
        result["a_initial"] = round(a_init, 4)

        write(str(work_dir / "geo.gen"), atoms, format="gen")
        (work_dir / "dftb_in.hsd").write_text(
            make_dftb_hsd(elements, ang_mom, is_metallic, n_atoms))

        proc = subprocess.run(
            [DFTB_BIN], cwd=str(work_dir),
            capture_output=True, text=True, timeout=600
        )
        if proc.returncode != 0:
            result["reason"] = f"exit_{proc.returncode}"
            err_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
            (work_dir / "stderr.log").write_text(err_text[-3000:])
            result["elapsed_sec"] = round(time.time() - t0, 1)
            return result

        geo_end = work_dir / "geo_end.gen"
        if not geo_end.exists():
            result["reason"] = "no_geo_end"
            result["elapsed_sec"] = round(time.time() - t0, 1)
            return result

        atoms_relaxed = read(str(geo_end), format="gen")
        cell_relax = atoms_relaxed.get_cell()
        vol_relax = atoms_relaxed.get_volume()
        a_relax = np.mean([np.linalg.norm(cell_relax[i]) for i in range(3)])
        vol_change = abs(vol_relax - vol_init) / vol_init

        result["a_relaxed"] = round(a_relax, 4)
        result["volume_change_pct"] = round(vol_change * 100, 2)

        detailed = work_dir / "detailed.out"
        if detailed.exists():
            for line in detailed.read_text().split("\n"):
                if "Total energy:" in line:
                    try:
                        result["energy_eV"] = round(float(line.split()[-2]), 6)
                        result["energy_per_atom_eV"] = round(result["energy_eV"] / n_atoms, 6)
                    except:
                        pass

        if vol_change > MAX_VOLUME_CHANGE:
            result["status"] = "suspect"
            result["reason"] = f"vol_{vol_change*100:.0f}pct"
            result["confidence_update"] = "low"
        else:
            result["status"] = "passed"
            if vol_change < 0.10:
                result["confidence_update"] = "high"
            elif vol_change < 0.20:
                result["confidence_update"] = "medium"
            else:
                result["confidence_update"] = "low"

        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.io.cif import CifWriter
        pmg_struct = AseAtomsAdaptor.get_structure(atoms_relaxed)
        relaxed_dir = Path(work_base).parent / "unit_cells_relaxed" / cls
        relaxed_dir.mkdir(parents=True, exist_ok=True)
        CifWriter(pmg_struct).write_file(str(relaxed_dir / f"{formula}.cif"))

    except subprocess.TimeoutExpired:
        result["reason"] = "timeout"
    except Exception as e:
        result["reason"] = str(e)[:200]

    result["elapsed_sec"] = round(time.time() - t0, 1)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=".")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--class-filter", default="")
    args = parser.parse_args()

    base = Path(args.base)
    work_base = base / "bulk_relax"
    work_base.mkdir(exist_ok=True)

    # Use v2 master CSV
    master_csv = base / "compositions_master_v2.csv"
    if not master_csv.exists():
        print(f"ERROR: {master_csv} not found. Run 01b_download_mp.py first.")
        sys.exit(1)

    # CIF source directory
    cif_base = base / "unit_cells_mp"
    if not cif_base.exists():
        print(f"ERROR: {cif_base} not found.")
        sys.exit(1)

    jobs = []
    with open(master_csv) as f:
        for row in csv.DictReader(f):
            cls = row["class"]
            formula = row["formula"]
            ang_mom = row["ang_mom"]
            source = row.get("source", "unknown")

            if args.class_filter and cls != args.class_filter:
                continue

            # Find CIF file — handle HEA filenames with phase suffix
            if cls == "hea":
                cif_fname = row.get("cif_filename", "")
                if cif_fname:
                    cif_path = cif_base / cls / cif_fname
                else:
                    phase = row.get("predicted_phase", "bcc")
                    if phase == "mixed":
                        phase = "bcc"
                    cif_path = cif_base / cls / f"{formula}_{phase}.cif"
            else:
                cif_path = cif_base / cls / f"{formula}.cif"

            if not cif_path.exists():
                continue

            # Skip already done
            if (work_base / cls / formula / "geo_end.gen").exists():
                continue

            jobs.append((formula, cls, str(cif_path), ang_mom, str(work_base), source))

    if args.limit > 0:
        jobs = jobs[:args.limit]
    total = len(jobs)
    if total == 0:
        print("No jobs to run.")
        return

    print(f"{'='*60}")
    print(f"NanoGenLM Bulk DFTB+ Relaxation v3")
    print(f"  Jobs: {total}  Workers: {args.workers}")
    print(f"  CIF source: {cif_base}/")
    print(f"{'='*60}")

    results = []
    passed = failed = suspect = 0
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_relax, job): job[0] for job in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            formula = futures[future]
            try:
                res = future.result()
            except Exception as e:
                res = {"formula": formula, "status": "error", "reason": str(e)[:200],
                       "class": "", "source": "", "volume_change_pct": None,
                       "energy_per_atom_eV": None, "elapsed_sec": None, "n_atoms": None}
            results.append(res)

            st = res.get("status", "error")
            if st == "passed":
                passed += 1
            elif st == "suspect":
                suspect += 1
            else:
                failed += 1

            elapsed = time.time() - t_start
            rate = i / elapsed if elapsed > 0 else 0
            eta = (total - i) / rate / 60 if rate > 0 else 0

            dv = res.get("volume_change_pct")
            ep = res.get("energy_per_atom_eV")
            na = res.get("n_atoms")
            dv_s = f"{dv:.1f}" if dv is not None else "N/A"
            ep_s = f"{ep:.4f}" if ep is not None else "N/A"
            na_s = f"{na}" if na is not None else "?"
            sec = res.get("elapsed_sec", "?")
            rsn = res.get("reason", "")
            rcls = res.get("class", "")
            src = res.get("source", "")[:5]
            mk = {"passed": "OK", "suspect": "!!"}.get(st, "XX")

            if i % 50 == 0 or i == total or st != "passed":
                print(f"  [{i:>5d}/{total}] {mk} {rcls:>10s}/{formula:<18s} "
                      f"n={na_s:>3s} dV={dv_s:>6s}% E={ep_s:>10s} eV/at "
                      f"[{src}] ({sec}s) [{rate*3600:.0f}/hr ETA:{eta:.0f}m] {rsn}")

    csv_path = base / "bulk_relax_results_v2.csv"
    if results:
        fnames = ["formula","class","source","status","reason","energy_eV",
                  "energy_per_atom_eV","a_initial","a_relaxed","n_atoms",
                  "volume_change_pct","confidence_update","elapsed_sec"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(results)

    tt = time.time() - t_start
    n = max(len(results), 1)

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  Passed:  {passed:>5d} ({passed/n*100:.1f}%)")
    print(f"  Suspect: {suspect:>5d} ({suspect/n*100:.1f}%)")
    print(f"  Failed:  {failed:>5d} ({failed/n*100:.1f}%)")
    print(f"  Time: {tt/60:.1f} min ({tt/3600:.1f} hr)")

    # Per-class breakdown
    print(f"\n  By class:")
    for cls in ["perovskite", "heusler", "hea", "hydride"]:
        sub = [r for r in results if r.get("class") == cls]
        if not sub:
            continue
        p = sum(1 for r in sub if r["status"] == "passed")
        s = sum(1 for r in sub if r["status"] == "suspect")
        f = sum(1 for r in sub if r["status"] == "failed")
        print(f"    {cls:<12s}  passed={p:>5d}  suspect={s:>4d}  failed={f:>4d}")

    # By source
    print(f"\n  By source:")
    for src in ["materials_project", "prototype"]:
        sub = [r for r in results if r.get("source") == src]
        if not sub:
            continue
        p = sum(1 for r in sub if r["status"] == "passed")
        print(f"    {src:<20s}  total={len(sub):>5d}  passed={p:>5d} ({p/max(len(sub),1)*100:.0f}%)")

    print(f"\n  Results CSV: {csv_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
