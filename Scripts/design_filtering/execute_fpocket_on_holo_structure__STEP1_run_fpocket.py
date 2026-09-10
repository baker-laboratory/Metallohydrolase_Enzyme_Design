#!/usr/bin/env python3
"""
execute_fpocket_on_structure.py

This script runs fpocket on a given PDB file and post-processes the results to
identify ligand-binding pockets, compute occupancy metrics, filter pocket info,
and relocate outputs.

Workflow:
 1. Change into the PDB's directory.
 2. Run fpocket on the PDB basename.
 3. Parse the input PDB for ligand residues (HETATM), grouping by ligand ID.
 4. Parse the fpocket output PDB for pocket sphere coordinates, grouped by pocket ID.
 5. Assign each ligand atom to its nearest pocket sphere and collect counts.
 6. Compute occupancy metrics:
    - ligand_<ligand>_occupation_of_this_pocket = fraction of ligand atoms in pocket
    - pocket_occupation_of_this_ligand_<ligand> = fraction of pocket spheres contacted
 7. Filter the fpocket info.txt file to keep only ligand-binding pockets and append metrics.
 8. Move the filtered info.txt and fpocket output PDB back to the input PDB directory.
 9. Delete the fpocket-generated output directory.

Usage:
    python execute_fpocket_on_structure.py --input_pdb /path/to/structure.pdb

Example:
    python execute_fpocket_on_structure.py \
      --input_pdb /path/to/structures/group1_...__1_0.pdb
"""
import argparse
import subprocess
import os
import sys
import math
import shutil


def run_fpocket(fpocket_exe, pdb_basename, params):
    cmd = [
        fpocket_exe,
        "--file", pdb_basename,
        "--min_spheres_per_pocket", str(params.min_spheres_per_pocket),
        "--number_apol_asph_pocket", str(params.number_apol_asph_pocket),
        "--ratio_apol_spheres_pocket", str(params.ratio_apol_spheres_pocket),
    ]
    print(f"[VERBOSE] Running fpocket in {os.getcwd()}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print("[VERBOSE] fpocket completed successfully.")


def parse_ligand_atoms(pdb_path):
    ligands = {}
    print(f"[VERBOSE] Parsing ligand atoms from {pdb_path}")
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("HETATM"):
                resname = line[17:20].strip()
                chain = line[21].strip()
                resseq = line[22:26].strip()
                lid = f"{resname}_{chain}_{resseq}"
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                ligands.setdefault(lid, []).append((x, y, z))
    for lid, coords in ligands.items():
        print(f"[VERBOSE] Found ligand {lid} with {len(coords)} atoms.")
    return ligands


def parse_pocket_spheres(out_pdb_path):
    spheres = {}
    print(f"[VERBOSE] Parsing pocket spheres from {out_pdb_path}")
    with open(out_pdb_path) as f:
        for line in f:
            if line.startswith("HETATM"):
                pid = int(line[22:26].strip())
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                spheres.setdefault(pid, []).append((x, y, z))
    print(f"[VERBOSE] Found pockets: {sorted(spheres.keys())}")
    return spheres


def assign_and_compute_metrics(ligand_groups, pocket_spheres):
    ligand_totals = {lid: len(coords) for lid, coords in ligand_groups.items()}
    pocket_sizes = {pid: len(coords) for pid, coords in pocket_spheres.items()}
    counts = {}

    print("[VERBOSE] Assigning ligand atoms to pockets and computing counts...")
    for lid, atoms in ligand_groups.items():
        for lx, ly, lz in atoms:
            best = (None, float('inf'))
            for pid, spheres in pocket_spheres.items():
                for sx, sy, sz in spheres:
                    d = math.dist((lx, ly, lz), (sx, sy, sz))
                    if d < best[1]: best = (pid, d)
            pid = best[0]
            counts.setdefault(pid, {}).setdefault(lid, 0)
            counts[pid][lid] += 1

    metrics = {}
    for pid, lig_counts in counts.items():
        metrics[pid] = {}
        for lid, c in lig_counts.items():
            ligand_occ = c / ligand_totals[lid]
            pocket_occ = c / pocket_sizes[pid]
            metrics[pid][lid] = (ligand_occ, pocket_occ)
            print(f"[VERBOSE] Pocket {pid}, Ligand {lid}: ligand_occ={ligand_occ:.3f}, pocket_occ={pocket_occ:.3f}")
    return metrics


def filter_and_annotate_info(info_txt_path, binding_pockets, metrics):
    print(f"[VERBOSE] Filtering and annotating {info_txt_path} for pockets {binding_pockets}")
    with open(info_txt_path) as f: lines = f.readlines()
    blocks, current = [], []
    for line in lines:
        if line.startswith("Pocket "):
            if current: blocks.append(current)
            current = [line]
        else:
            current.append(line)
    if current: blocks.append(current)

    with open(info_txt_path, 'w') as outf:
        for blk in blocks:
            pid = int(blk[0].split()[1])
            if pid in binding_pockets:
                outf.writelines(blk)
                for lid, (lig_occ, poc_occ) in metrics.get(pid, {}).items():
                    outf.write(f"\tligand_{lid}_occupation_of_this_pocket: {lig_occ:.3f}\n")
                    outf.write(f"\tpocket_occupation_of_this_ligand_{lid}: {poc_occ:.3f}\n")
                outf.write("\n")
    print(f"[VERBOSE] Annotated info written to {info_txt_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run fpocket and compute ligand-pocket occupancy metrics."
    )
    parser.add_argument("--input_pdb", required=True)
    parser.add_argument("--fpocket_exe", default=os.environ.get("FPOCKET") or shutil.which("fpocket") or "fpocket")
    parser.add_argument("--min_spheres_per_pocket", type=int, default=20)
    parser.add_argument("--number_apol_asph_pocket", type=int, default=3)
    parser.add_argument("--ratio_apol_spheres_pocket", type=int, default=0)
    args = parser.parse_args()

    pdb_path = os.path.abspath(args.input_pdb)
    if not os.path.isfile(pdb_path):
        print(f"[ERROR] Input PDB not found: {pdb_path}", file=sys.stderr)
        sys.exit(1)

    pdb_dir, pdb_basename = os.path.split(pdb_path)
    os.chdir(pdb_dir)
    print(f"[VERBOSE] CWD → {pdb_dir}")

    run_fpocket(args.fpocket_exe, pdb_basename, args)

    base = os.path.splitext(pdb_basename)[0]
    out_dir = f"{base}_out"
    out_pdb = os.path.join(out_dir, f"{base}_out.pdb")
    info_txt = os.path.join(out_dir, f"{base}_info.txt")

    ligand_groups = parse_ligand_atoms(pdb_basename)
    pocket_spheres = parse_pocket_spheres(out_pdb)
    metrics = assign_and_compute_metrics(ligand_groups, pocket_spheres)
    binding_pockets = sorted(metrics.keys())
    filter_and_annotate_info(info_txt, binding_pockets, metrics)

    # Move outputs back to input PDB directory
    dest_info = os.path.basename(info_txt)
    dest_pdb = os.path.basename(out_pdb)
    shutil.move(info_txt, dest_info)
    shutil.move(out_pdb, dest_pdb)
    print(f"[VERBOSE] Moved {info_txt} and {out_pdb} to {pdb_dir}")

    # Clean up fpocket directory
    shutil.rmtree(out_dir)
    print(f"[VERBOSE] Removed fpocket output directory {out_dir}")

    print("[VERBOSE] Post-processing and relocation complete.")

if __name__ == "__main__":
    main()
