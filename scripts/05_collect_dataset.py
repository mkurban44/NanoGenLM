#!/usr/bin/env python3
"""
NanoGenLM — Step 05: ML Dataset Collection
Collects all relaxed NP results into two datasets:
  1. NeurIPS:      All passed/partial NPs (maximizes data)
  2. NatureComm:   Only compositions with BOTH R=5 and R=6 passed (paired data)

Usage:
    python3 05_collect_dataset.py --base .
"""
import os, sys, csv, json, argparse
import numpy as np
from pathlib import Path
from ase.io import read
from ase.db import connect

R_RELAX = [5, 6]


def parse_detailed_out(detailed_path):
    """Extract energy, convergence, Fermi level from detailed.out."""
    info = {
        "total_energy_eV": None,
        "fermi_eV": None,
        "scc_converged": False,
        "geo_converged": False,
        "band_energy_eV": None,
        "repulsive_energy_eV": None,
    }
    if not detailed_path.exists():
        return info

    text = detailed_path.read_text()
    for line in text.split("\n"):
        if "Total energy:" in line and "eV" in line:
            try:
                info["total_energy_eV"] = float(line.split()[-2])
            except:
                pass
        if "Fermi level:" in line and "eV" in line:
            try:
                info["fermi_eV"] = float(line.split()[-2])
            except:
                pass
        if "Band energy:" in line and "eV" in line:
            try:
                info["band_energy_eV"] = float(line.split()[-2])
            except:
                pass
        if "Repulsive energy:" in line and "eV" in line:
            try:
                info["repulsive_energy_eV"] = float(line.split()[-2])
            except:
                pass
    if "SCC converged" in text:
        info["scc_converged"] = True
    if "Geometry converged" in text:
        info["geo_converged"] = True
    return info


def compute_structural_descriptors(atoms):
    """Compute basic structural descriptors for a NP."""
    positions = atoms.get_positions()
    n_atoms = len(atoms)

    # Center of mass
    com = positions.mean(axis=0)

    # Radius of gyration
    r_vecs = positions - com
    rg = np.sqrt(np.mean(np.sum(r_vecs**2, axis=1)))

    # Max radius from COM
    r_max = np.max(np.linalg.norm(r_vecs, axis=1))

    # Nearest neighbor distances
    from scipy.spatial.distance import pdist, squareform
    dists = squareform(pdist(positions))
    np.fill_diagonal(dists, 999.0)
    nn_dists = np.min(dists, axis=1)

    # Asphericity
    gyration_tensor = np.zeros((3, 3))
    for r in r_vecs:
        gyration_tensor += np.outer(r, r)
    gyration_tensor /= n_atoms
    eigenvalues = np.sort(np.linalg.eigvalsh(gyration_tensor))
    asphericity = eigenvalues[2] - 0.5 * (eigenvalues[0] + eigenvalues[1])

    # Element composition
    symbols = atoms.get_chemical_symbols()
    unique_elements = sorted(set(symbols))
    composition = {el: symbols.count(el) for el in unique_elements}
    surface_fraction = np.sum(nn_dists > np.mean(nn_dists) * 1.2) / n_atoms

    return {
        "radius_of_gyration": round(rg, 4),
        "max_radius": round(r_max, 4),
        "nn_mean": round(np.mean(nn_dists), 4),
        "nn_min": round(np.min(nn_dists), 4),
        "nn_max": round(np.max(nn_dists), 4),
        "nn_std": round(np.std(nn_dists), 4),
        "asphericity": round(asphericity, 4),
        "surface_fraction_approx": round(surface_fraction, 4),
        "composition": composition,
        "unique_elements": unique_elements,
        "n_elements": len(unique_elements),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=".")
    args = parser.parse_args()

    base = Path(args.base)
    np_base = base / "nanoparticles"
    dataset_dir = base / "dataset"
    dataset_dir.mkdir(exist_ok=True)

    # Load NP relax results
    relax_csv = base / "np_relax_results.csv"
    if not relax_csv.exists():
        print(f"ERROR: {relax_csv} not found")
        sys.exit(1)

    # Load bulk relax results for reference
    bulk_csv = base / "bulk_relax_results_v2.csv"
    bulk_data = {}
    if bulk_csv.exists():
        with open(bulk_csv) as f:
            for row in csv.DictReader(f):
                bulk_data[row["formula"]] = row

    # Load master CSV for metadata
    master_csv = base / "compositions_master_v2.csv"
    master_data = {}
    if master_csv.exists():
        with open(master_csv) as f:
            for row in csv.DictReader(f):
                master_data[row["formula"]] = row

    # Read NP relax results
    np_results = []
    with open(relax_csv) as f:
        for row in csv.DictReader(f):
            if row["status"] in ("passed", "partial"):
                np_results.append(row)

    print(f"{'='*60}")
    print(f"NanoGenLM — ML Dataset Collection")
    print(f"{'='*60}")
    print(f"  Passed/Partial NPs: {len(np_results)}")

    # Group by formula
    by_formula = {}
    for row in np_results:
        f = row["formula"]
        if f not in by_formula:
            by_formula[f] = {}
        R = int(row["R"])
        by_formula[f][R] = row

    # Identify paired (both R=5 and R=6) vs unpaired
    paired = []
    unpaired = []
    for formula, radii_dict in by_formula.items():
        has_5 = 5 in radii_dict
        has_6 = 6 in radii_dict
        if has_5 and has_6:
            paired.append(formula)
        else:
            unpaired.append(formula)

    print(f"  Paired (R=5+R=6):   {len(paired)} compositions")
    print(f"  Unpaired (single):  {len(unpaired)} compositions")
    print(f"  Total compositions: {len(by_formula)}")

    # ============================================================
    # COLLECT DATA
    # ============================================================
    all_records = []  # For NeurIPS
    skip_classes = ["hea"]
    processed = 0
    errors = 0

    for formula, radii_dict in sorted(by_formula.items()):
        for R, row in sorted(radii_dict.items()):
            cls = row["class"]
            if cls in skip_classes:
                continue

            r_dir = np_base / cls / formula / f"R{R:02d}"
            geo_end = r_dir / "geo_end.gen"

            if not geo_end.exists():
                errors += 1
                continue

            try:
                atoms = read(str(geo_end), format="gen")
                n_atoms = len(atoms)

                # Parse detailed.out
                detailed = parse_detailed_out(r_dir / "detailed.out")

                # Structural descriptors
                desc = compute_structural_descriptors(atoms)

                # Bulk reference data
                bulk = bulk_data.get(formula, {})
                master = master_data.get(formula, {})

                record = {
                    # Identity
                    "formula": formula,
                    "class": cls,
                    "R": R,
                    "n_atoms": n_atoms,
                    "n_elements": desc["n_elements"],
                    "elements": ";".join(desc["unique_elements"]),
                    "composition_json": json.dumps(desc["composition"]),

                    # Source & confidence
                    "source": master.get("source", ""),
                    "mp_id": master.get("mp_id", ""),
                    "confidence_bulk": bulk.get("confidence_update", ""),
                    "confidence_initial": master.get("confidence", ""),

                    # NP energies
                    "total_energy_eV": detailed["total_energy_eV"],
                    "energy_per_atom_eV": round(detailed["total_energy_eV"] / n_atoms, 6) if detailed["total_energy_eV"] else None,
                    "fermi_eV": detailed["fermi_eV"],
                    "band_energy_eV": detailed["band_energy_eV"],
                    "repulsive_energy_eV": detailed["repulsive_energy_eV"],

                    # Bulk reference
                    "bulk_energy_per_atom_eV": bulk.get("energy_per_atom_eV", ""),
                    "bulk_a_relaxed": bulk.get("a_relaxed", ""),

                    # Convergence
                    "scc_converged": detailed["scc_converged"],
                    "geo_converged": detailed["geo_converged"],

                    # Structural descriptors
                    "radius_of_gyration": desc["radius_of_gyration"],
                    "max_radius": desc["max_radius"],
                    "nn_mean": desc["nn_mean"],
                    "nn_min": desc["nn_min"],
                    "nn_max": desc["nn_max"],
                    "nn_std": desc["nn_std"],
                    "asphericity": desc["asphericity"],
                    "surface_fraction": desc["surface_fraction_approx"],

                    # File paths (relative)
                    "geo_end_path": str(r_dir / "geo_end.gen"),
                    "np_gen_path": str(r_dir / "np.gen"),

                    # Paired flag
                    "is_paired": formula in paired,
                }

                all_records.append(record)
                processed += 1

            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  ERROR: {cls}/{formula}/R{R:02d}: {str(e)[:100]}")

        if processed % 200 == 0 and processed > 0:
            print(f"  Processed: {processed}...")

    print(f"\n  Processed: {processed}")
    print(f"  Errors: {errors}")

    # ============================================================
    # WRITE NEURIPS DATASET (ALL)
    # ============================================================
    neurips_records = all_records
    neurips_csv = dataset_dir / "neurips_dataset.csv"
    fieldnames = list(all_records[0].keys()) if all_records else []

    with open(neurips_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(neurips_records)

    # ============================================================
    # WRITE NATURE COMM DATASET (PAIRED ONLY)
    # ============================================================
    natcomm_records = [r for r in all_records if r["is_paired"]]
    natcomm_csv = dataset_dir / "nature_comm_dataset.csv"

    with open(natcomm_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(natcomm_records)

    # ============================================================
    # WRITE ASE DATABASES
    # ============================================================
    print(f"\n  Writing ASE databases...")

    # NeurIPS DB
    neurips_db_path = dataset_dir / "neurips_all.db"
    if neurips_db_path.exists():
        neurips_db_path.unlink()
    db_neurips = connect(str(neurips_db_path))

    # NatureComm DB
    natcomm_db_path = dataset_dir / "nature_comm_paired.db"
    if natcomm_db_path.exists():
        natcomm_db_path.unlink()
    db_natcomm = connect(str(natcomm_db_path))

    for rec in all_records:
        geo_path = rec["geo_end_path"]
        try:
            atoms = read(geo_path, format="gen")
            kvp = {
                "formula": rec["formula"],
                "material_class": rec["class"],
                "R": rec["R"],
                "n_atoms": rec["n_atoms"],
                "total_energy_eV": rec["total_energy_eV"] or 0.0,
                "energy_per_atom_eV": rec["energy_per_atom_eV"] or 0.0,
                "source": rec["source"],
                "confidence_bulk": rec["confidence_bulk"],
                "nn_mean": rec["nn_mean"],
                "nn_min": rec["nn_min"],
                "asphericity": rec["asphericity"],
                "is_paired": rec["is_paired"],
            }

            db_neurips.write(atoms, **kvp)
            if rec["is_paired"]:
                db_natcomm.write(atoms, **kvp)
        except Exception as e:
            pass

    # ============================================================
    # WRITE XYZ FILES (for generative model training)
    # ============================================================
    print(f"  Writing XYZ files...")
    xyz_dir = dataset_dir / "xyz_files"
    xyz_dir.mkdir(exist_ok=True)

    for rec in all_records:
        geo_path = rec["geo_end_path"]
        try:
            atoms = read(geo_path, format="gen")
            fname = f"{rec['class']}_{rec['formula']}_R{rec['R']:02d}.xyz"
            from ase.io import write
            write(str(xyz_dir / fname), atoms, format="xyz",
                  comment=f"E={rec['energy_per_atom_eV']} eV/at class={rec['class']} R={rec['R']}")
        except:
            pass

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'='*60}")
    print(f"DATASET SUMMARY")
    print(f"{'='*60}")

    print(f"\n  NeurIPS Dataset (ALL):")
    print(f"    Records:      {len(neurips_records)}")
    n_comp = len(set(r["formula"] for r in neurips_records))
    print(f"    Compositions: {n_comp}")
    for cls in ["perovskite", "heusler", "hydride"]:
        sub = [r for r in neurips_records if r["class"] == cls]
        n_c = len(set(r["formula"] for r in sub))
        print(f"      {cls:<12s}  {len(sub):>5d} NPs from {n_c} compositions")
    print(f"    CSV: {neurips_csv}")
    print(f"    DB:  {neurips_db_path}")

    print(f"\n  Nature Communications Dataset (PAIRED R=5+R=6):")
    print(f"    Records:      {len(natcomm_records)}")
    n_comp_p = len(set(r["formula"] for r in natcomm_records))
    print(f"    Compositions: {n_comp_p}")
    for cls in ["perovskite", "heusler", "hydride"]:
        sub = [r for r in natcomm_records if r["class"] == cls]
        n_c = len(set(r["formula"] for r in sub))
        print(f"      {cls:<12s}  {len(sub):>5d} NPs from {n_c} compositions")
    print(f"    CSV: {natcomm_csv}")
    print(f"    DB:  {natcomm_db_path}")

    # Energy statistics
    print(f"\n  Energy Statistics (eV/atom):")
    for ds_name, records in [("NeurIPS", neurips_records), ("NatComm", natcomm_records)]:
        energies = [r["energy_per_atom_eV"] for r in records if r["energy_per_atom_eV"]]
        if energies:
            print(f"    {ds_name}: mean={np.mean(energies):.2f}  "
                  f"min={np.min(energies):.2f}  max={np.max(energies):.2f}")

    # Size statistics
    print(f"\n  Size Statistics:")
    for R in R_RELAX:
        sub = [r for r in neurips_records if r["R"] == R]
        atoms_list = [r["n_atoms"] for r in sub]
        if atoms_list:
            print(f"    R={R}: n={len(sub)}  atoms: mean={np.mean(atoms_list):.0f}  "
                  f"min={np.min(atoms_list)}  max={np.max(atoms_list)}")

    print(f"\n  XYZ files: {xyz_dir}/ ({len(list(xyz_dir.glob('*.xyz')))} files)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
