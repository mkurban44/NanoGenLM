#!/usr/bin/env python3
"""
NanoGenLM — Step 04: Nanoparticle DFTB+ Relaxation (v2)
R=5,6 only. Perovskite + Heusler + Hydride (no HEA).
Uses corrected DFTB input format (tested on TiFe NP).

Usage:
    python3 04_relax_nanoparticles.py --base . --workers 10 [--limit 20]
"""
import os, sys, csv, json, subprocess, argparse, time
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

SK_PREFIX = "/mnt/d/DFTB_shared/skfiles/ParameterSets/ptbp/complete_set/"
DFTB_BIN = "dftb+"
R_RELAX = [5, 6]
SKIP_CLASSES = ["hea"]


def write_corrected_hsd(work_dir, elements, ang_mom_dict, is_metallic=False):
    """Write DFTB input matching the working TiFe NP format."""
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
  Filling = Fermi {{ Temperature [Kelvin] = {filling_temp} }}
  Mixer = Broyden {{ MixingParameter = 0.2 }}
}}

Options       = {{ WriteChargesAsText = Yes }}
ParserOptions = {{ IgnoreUnprocessedNodes = Yes }}

Driver = GeometryOptimization {{
  MaxSteps    = 2000
  Convergence = {{ GradTolerance = 1e-3 }}
}}
"""
    (work_dir / "dftb_in.hsd").write_text(hsd)


def run_np_relax(args):
    formula, cls, r_dir_str, R, ang_mom_str = args
    r_dir = Path(r_dir_str)

    ang_mom = json.loads(ang_mom_str)
    elements = sorted(ang_mom.keys())
    is_metallic = cls == "heusler"

    result = {
        "formula": formula, "class": cls, "R": R,
        "status": "failed", "reason": "",
        "n_atoms": None, "energy_eV": None, "energy_per_atom_eV": None,
        "elapsed_sec": None,
    }
    t0 = time.time()

    try:
        gen_file = r_dir / "np.gen"
        if not gen_file.exists():
            result["reason"] = "missing_np_gen"
            result["elapsed_sec"] = round(time.time() - t0, 1)
            return result

        # Read atom count
        n_file = r_dir / "N.txt"
        if n_file.exists():
            n_atoms = int(n_file.read_text().strip())
        else:
            from ase.io import read
            atoms = read(str(gen_file), format="gen")
            n_atoms = len(atoms)
        result["n_atoms"] = n_atoms

        # Rewrite dftb_in.hsd with corrected format
        write_corrected_hsd(r_dir, elements, ang_mom, is_metallic)

        # Timeout based on atom count
        if n_atoms > 150:
            timeout = 1800
        elif n_atoms > 80:
            timeout = 900
        else:
            timeout = 1200

        # Run DFTB+
        proc = subprocess.run(
            [DFTB_BIN], cwd=str(r_dir),
            capture_output=True, text=True, timeout=timeout
        )

        if proc.returncode != 0:
            result["reason"] = f"exit_{proc.returncode}"
            err = (proc.stdout or "") + "\n" + (proc.stderr or "")
            (r_dir / "stderr.log").write_text(err[-3000:])
            result["elapsed_sec"] = round(time.time() - t0, 1)
            return result

        geo_end = r_dir / "geo_end.gen"
        if not geo_end.exists():
            result["reason"] = "no_geo_end"
            result["elapsed_sec"] = round(time.time() - t0, 1)
            return result

        # Parse energy
        detailed = r_dir / "detailed.out"
        if detailed.exists():
            text = detailed.read_text()
            for line in text.split("\n"):
                if "Total energy:" in line:
                    try:
                        result["energy_eV"] = round(float(line.split()[-2]), 6)
                        result["energy_per_atom_eV"] = round(
                            result["energy_eV"] / n_atoms, 6)
                    except:
                        pass

            # Check convergence
            if "SCC converged" in text and "Geometry converged" in text:
                result["status"] = "passed"
            elif "SCC converged" in text:
                result["status"] = "partial"
                result["reason"] = "scc_ok_geo_not"
            else:
                result["status"] = "failed"
                result["reason"] = "scc_not_converged"

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
    np_base = base / "nanoparticles"

    if not np_base.exists():
        print(f"ERROR: {np_base} not found.")
        sys.exit(1)

    # Load ang_mom from master CSV
    ang_mom_map = {}
    master_csv = base / "compositions_master_v2.csv"
    if master_csv.exists():
        with open(master_csv) as f:
            for row in csv.DictReader(f):
                ang_mom_map[row["formula"]] = row["ang_mom"]

    # Find jobs
    jobs = []
    for cls_dir in sorted(np_base.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls = cls_dir.name

        if cls in SKIP_CLASSES:
            continue
        if args.class_filter and cls != args.class_filter:
            continue

        for comp_dir in sorted(cls_dir.iterdir()):
            if not comp_dir.is_dir():
                continue
            formula = comp_dir.name
            ang_mom = ang_mom_map.get(formula, "")
            if not ang_mom:
                continue

            for R in R_RELAX:
                tag = f"R{R:02d}"
                r_dir = comp_dir / tag

                if not (r_dir / "np.gen").exists():
                    continue
                if (r_dir / "geo_end.gen").exists():
                    continue

                jobs.append((formula, cls, str(r_dir), R, ang_mom))

    if args.limit > 0:
        jobs = jobs[:args.limit]

    total = len(jobs)
    if total == 0:
        print("No NP relaxation jobs to run.")
        return

    r_counts = {}
    cls_counts = {}
    for _, cls, _, R, _ in jobs:
        r_counts[R] = r_counts.get(R, 0) + 1
        cls_counts[cls] = cls_counts.get(cls, 0) + 1

    print(f"{'='*60}")
    print(f"NanoGenLM — NP DFTB+ Relaxation v2")
    print(f"{'='*60}")
    print(f"  Total jobs: {total}")
    print(f"  Workers:    {args.workers}")
    print(f"  Radii:      {R_RELAX}")
    print(f"  Skipping:   {SKIP_CLASSES}")
    for R in R_RELAX:
        print(f"  R={R} A:     {r_counts.get(R, 0)} NPs")
    for cls in sorted(cls_counts):
        print(f"  {cls}: {cls_counts[cls]} NPs")
    print(f"{'='*60}")

    results = []
    passed = partial = failed = 0
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_np_relax, job): (job[0], job[3]) for job in jobs}

        for i, future in enumerate(as_completed(futures), 1):
            formula, R = futures[future]
            try:
                res = future.result()
            except Exception as e:
                res = {"formula": formula, "R": R, "status": "error",
                       "reason": str(e)[:200], "class": "", "n_atoms": None,
                       "energy_per_atom_eV": None, "elapsed_sec": None}
            results.append(res)

            st = res.get("status", "error")
            if st == "passed":
                passed += 1
            elif st == "partial":
                partial += 1
            else:
                failed += 1

            elapsed = time.time() - t_start
            rate = i / elapsed if elapsed > 0 else 0
            eta = (total - i) / rate / 60 if rate > 0 else 0

            na = res.get("n_atoms")
            ep = res.get("energy_per_atom_eV")
            na_s = f"{na}" if na is not None else "?"
            ep_s = f"{ep:.4f}" if ep is not None else "N/A"
            sec = res.get("elapsed_sec", "?")
            rsn = res.get("reason", "")
            rcls = res.get("class", "")
            mk = {"passed": "OK", "partial": "~~"}.get(st, "XX")

            if i % 50 == 0 or i == total or st not in ("passed", "partial"):
                print(f"  [{i:>5d}/{total}] {mk} {rcls:>10s}/{formula:<18s} "
                      f"R={res.get('R','?')} n={na_s:>4s} E={ep_s:>10s} eV/at "
                      f"({sec}s) [{rate*3600:.0f}/hr ETA:{eta:.0f}m] {rsn}")

    # Write results
    csv_path = base / "np_relax_results.csv"
    if results:
        fnames = ["formula", "class", "R", "status", "reason",
                  "n_atoms", "energy_eV", "energy_per_atom_eV", "elapsed_sec"]
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
    print(f"  Partial: {partial:>5d} ({partial/n*100:.1f}%)")
    print(f"  Failed:  {failed:>5d} ({failed/n*100:.1f}%)")
    print(f"  Time:    {tt/60:.1f} min ({tt/3600:.1f} hr)")

    print(f"\n  By radius:")
    for R in R_RELAX:
        sub = [r for r in results if r.get("R") == R]
        p = sum(1 for r in sub if r["status"] in ("passed", "partial"))
        f = sum(1 for r in sub if r["status"] not in ("passed", "partial"))
        if sub:
            avg_atoms = np.mean([r["n_atoms"] for r in sub if r.get("n_atoms")])
            print(f"    R={R} A:  ok={p:>5d}  failed={f:>4d}  avg_atoms={avg_atoms:.0f}")

    print(f"\n  By class:")
    for cls in ["perovskite", "heusler", "hydride"]:
        sub = [r for r in results if r.get("class") == cls]
        if not sub:
            continue
        p = sum(1 for r in sub if r["status"] in ("passed", "partial"))
        print(f"    {cls:<12s}  ok={p:>5d}/{len(sub)}")

    print(f"\n  Results: {csv_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
