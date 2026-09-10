#!/usr/bin/env python3
"""
contact_counter__MAIN.py

Wrapper to generate and execute the contact_counter__STEP1_calculate_contacts.py script
on one or more PDB files, forwarding all relevant flags.

Usage:
    python contact_counter__MAIN.py /path/to/file1.pdb [file2.pdb ...] \
        [--ligands LIG1 LIG2 ...] \
        [--split-ligands] \
        [--include-hydrogens] \
        [--cutoffs 4 5 6 7.2] \
        [--return_HETATM_coordinates_from_cutoffs] \
        [--keep_original_csv_ligANDcutoff_separated_by_row] \
        [--step2-eps-list 0.05 0.1 0.2] \
        [--step2-min-list 3 5 10] \
        [--step2-angle-thresholds 10 15 20 25] \
        [--step2-sphere-samples 1000] \
        [--keep_contactcutoff]

Dependencies:
    Python 3, subprocess, argparse
"""

from pathlib import Path
import os
import argparse
import subprocess
import shlex
import sys
import glob

# Sibling helper scripts resolve next to this file, so the whole set can be
# relocated together.
_SCRIPT_DIR = Path(__file__).resolve().parent

STEP1_SCRIPT = str(_SCRIPT_DIR / "contact_counter__STEP1_calculate_contacts.py")

STEP2_SCRIPT    = str(_SCRIPT_DIR / "contact_counter__STEP2_parse_coord_clouds_for_metrics.py")

# STEP2 used to be launched through a lab container at
# /software/containers/crispy.sif. That image was never published and is gone.
# STEP2 needs only numpy/pandas/scipy/scikit-learn, all of which the zinc_hydro
# environment provides, so the runner defaults to this interpreter; set
# ZINC_HYDRO_SIF to route it through a container instead.
for _anc in _SCRIPT_DIR.parents:
    if (_anc / "env_config.py").is_file():
        sys.path.insert(0, str(_anc))
        break
import env_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and run contact_counter__STEP1_calculate_contacts.py on PDB(s)."
    )
    parser.add_argument(
        "pdb",
        nargs='+',
        help="Path(s) to one or more input PDB files."
    )
    parser.add_argument(
        "--ligands",
        nargs='+',
        help="Residue names of ligands to pass to the STEP1 script."
    )
    parser.add_argument(
        "--split-ligands",
        action="store_true",
        help="Forward --split-ligands flag to the STEP1 script."
    )
    parser.add_argument(
        "--include-hydrogens",
        action="store_true",
        help="Forward --include-hydrogens flag to the STEP1 script."
    )
    parser.add_argument(
        "--cutoffs",
        nargs='+',
        type=float,
        default=[4.0, 5.0, 6.0],
        help="List of distance cutoffs (Å) for the STEP1 script. Default: 4.0 5.0 6.0"
    )
    parser.add_argument(
        "--return_HETATM_coordinates_from_cutoffs",
        action="store_true",
        help="Forward --return_HETATM_coordinates_from_cutoffs to the STEP1 script."
    )
    parser.add_argument(
        "--keep_original_csv_ligANDcutoff_separated_by_row",
        action="store_true",
        help="Forward --keep_original_csv_ligANDcutoff_separated_by_row to the STEP1 script."
    )
    # STEP2 parameters
    parser.add_argument(
        "--step2-eps-list",
        nargs='+',
        type=float,
        default=None,
        help="List of DBSCAN eps values to pass to STEP2 (e.g. 0.05 0.1 0.2)."
    )
    parser.add_argument(
        "--step2-min-list",
        nargs='+',
        type=int,
        default=None,
        help="List of DBSCAN min_samples values to pass to STEP2 (e.g. 3 5 10)."
    )
    parser.add_argument(
        "--step2-angle-thresholds",
        nargs='+',
        type=float,
        default=None,
        help="List of half-angle thresholds to pass to STEP2 (e.g. 10 15 20 25)."
    )
    parser.add_argument(
        "--step2-sphere-samples",
        type=int,
        default=None,
        help="Number of sphere samples to pass to STEP2 (e.g. 1000)."
    )
    parser.add_argument(
        "--keep_contactcutoff",
        action="store_true",
        help="Do NOT delete the __CONTACTcutoff_*A.pdb files after STEP2."
    )
    return parser.parse_args()


def build_command(pdb_file, args):
    # Use the same Python interpreter
    python_exec = sys.executable if sys.executable else "python3"
    cmd = [python_exec, STEP1_SCRIPT, pdb_file]

    if args.ligands:
        cmd += ["--ligands"] + args.ligands
    if args.split_ligands:
        cmd.append("--split-ligands")
    if args.include_hydrogens:
        cmd.append("--include-hydrogens")
    if args.cutoffs:
        cmd += ["--cutoffs"] + [str(c) for c in args.cutoffs]
    if args.return_HETATM_coordinates_from_cutoffs:
        cmd.append("--return_HETATM_coordinates_from_cutoffs")
    if args.keep_original_csv_ligANDcutoff_separated_by_row:
        cmd.append("--keep_original_csv_ligANDcutoff_separated_by_row")

    return cmd


def main():
    args = parse_args()

    for pdb_file in args.pdb:
        if not os.path.isfile(pdb_file):
            print(f"[ERROR] PDB file not found: {pdb_file}")
            continue

        cmd = build_command(pdb_file, args)
        print(f"[INFO] Executing: {' '.join(shlex.quote(x) for x in cmd)}")
        try:
            # STEP1
            subprocess.run(cmd, check=True)
            print("[INFO] STEP1 completed successfully.")

            # STEP2 (only if coord dumps requested)
            if args.return_HETATM_coordinates_from_cutoffs:
                step2_cmd = env_config.resolve_runner() + [
                    STEP2_SCRIPT,
                    pdb_file
                ]
                # forward STEP2 lists if provided
                if args.step2_eps_list is not None:
                    step2_cmd += ["--eps-list"] + [str(e) for e in args.step2_eps_list]
                if args.step2_min_list is not None:
                    step2_cmd += ["--min-list"] + [str(m) for m in args.step2_min_list]
                if args.step2_angle_thresholds is not None:
                    step2_cmd += ["--angle-thresholds"] + [str(a) for a in args.step2_angle_thresholds]
                if args.step2_sphere_samples is not None:
                    step2_cmd += ["--sphere-samples", str(args.step2_sphere_samples)]

                print(f"[INFO] Executing STEP2: {' '.join(shlex.quote(x) for x in step2_cmd)}")
                subprocess.run(step2_cmd, check=True)
                print("[INFO] STEP2 completed successfully.")

                # Cleanup: delete the per-cutoff PDBs unless user asked to keep them
                if not args.keep_contactcutoff:
                    pattern = os.path.splitext(pdb_file)[0] + "__CONTACTcutoff_*A.pdb"
                    for f in glob.glob(pattern):
                        try:
                            os.remove(f)
                            print(f"[INFO] Deleted cutoff cloud {f}")
                        except Exception as e:
                            print(f"[WARN] Could not delete {f}: {e}")

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Command failed for {pdb_file} with exit code {e.returncode}")


if __name__ == "__main__":
    main()
