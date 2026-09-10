#!/usr/bin/env python3
"""
prepare_fpocket_run.py

This script prepares and executes two steps for fpocket analysis:

STEP1: Run fpocket on the input PDB (fpocket wrapper script).
STEP2: Post-process fpocket outputs (compile metrics script).

It will always produce an aggregated CSV (`<base>_info_aggregated.csv`), even if STEP1 or STEP2 fail.

Usage:
    python prepare_fpocket_run.py \
      --input_pdb /path/to/structure.pdb \
      [--fpocket_exe /path/to/fpocket] \
      [--min_spheres_per_pocket 20] \
      [--number_apol_asph_pocket 3] \
      [--ratio_apol_spheres_pocket 0] \
      [--step1_script_path /path/to/STEP1.py] \
      [--step2_script_path /path/to/STEP2.py] \
      [--do_not_delete_raw_csv] \
      [--do_not_delete_modified_fpocket_txt] \
      [--do_not_delete_fpocket_pdb]


It will execute:
  python STEP1.py --input_pdb <...> [flags]
  python STEP2.py --fpocket_pdb_pockets_file <base>_out.pdb --fpocket_txt_file <base>_info.txt [--do_not_delete_raw_csv]

Upon any failure, it writes an aggregated CSV with only an `error_message` column.
"""
from pathlib import Path
import argparse
import subprocess
import sys
import os
import shutil
import pandas as pd

# Sibling helper scripts resolve next to this file, so the whole set can be
# relocated together.
_SCRIPT_DIR = Path(__file__).resolve().parent


def write_error_csv(agg_csv, msg):
    # Write a CSV with an error_message column
    df = pd.DataFrame([{'error_message': msg}])
    df.to_csv(agg_csv, index=False)
    print(f"[INFO] Wrote error aggregated CSV to {agg_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Run fpocket STEP1 and STEP2 pipelines with guaranteed output."
    )
    parser.add_argument(
        "--input_pdb", required=True,
        help="Path to the input PDB file."
    )
    parser.add_argument(
        "--fpocket_exe",
        default=os.environ.get("FPOCKET") or shutil.which("fpocket") or "fpocket",
        help="Path to the fpocket executable to pass through (STEP1)."
    )
    parser.add_argument(
        "--min_spheres_per_pocket", type=int, default=20,
        help="--min_spheres_per_pocket for STEP1."
    )
    parser.add_argument(
        "--number_apol_asph_pocket", type=int, default=3,
        help="--number_apol_asph_pocket for STEP1."
    )
    parser.add_argument(
        "--ratio_apol_spheres_pocket", type=int, default=0,
        help="--ratio_apol_spheres_pocket for STEP1."
    )
    parser.add_argument(
        "--step1_script_path",
        default=str(_SCRIPT_DIR / "execute_fpocket_on_holo_structure__STEP1_run_fpocket.py"),
        help="Path to STEP1 fpocket wrapper script."
    )
    parser.add_argument(
        "--step2_script_path",
        default=str(_SCRIPT_DIR / "execute_fpocket_on_holo_structure__STEP2_compile_fpocket_metrics.py"),
        help="Path to STEP2 compile metrics script."
    )
    parser.add_argument(
        "--do_not_delete_raw_csv", action="store_true",
        help="Pass --do_not_delete_raw_csv to STEP2 to keep raw CSV."
    )
    parser.add_argument(
        "--do_not_delete_modified_fpocket_txt", action="store_true",
        help="Do not delete the modified fpocket info TXT at end."
    )
    parser.add_argument(
        "--do_not_delete_fpocket_pdb",
        action="store_true",
        help="Do not delete the fpocket-generated PDB (`<base>_out.pdb`) after STEP2."
    )
    args = parser.parse_args()

    # Derive base and paths
    base = os.path.splitext(os.path.basename(args.input_pdb))[0]
    info_txt = f"{base}_info.txt"
    agg_csv = f"{base}_info_aggregated.csv"

    try:
        # STEP1
        step1_cmd = [
            sys.executable,
            args.step1_script_path,
            "--input_pdb", args.input_pdb,
            "--fpocket_exe", args.fpocket_exe,
            "--min_spheres_per_pocket", str(args.min_spheres_per_pocket),
            "--number_apol_asph_pocket", str(args.number_apol_asph_pocket),
            "--ratio_apol_spheres_pocket", str(args.ratio_apol_spheres_pocket),
        ]
        print(f"[VERBOSE] Running STEP1: {' '.join(step1_cmd)}")
        subprocess.run(step1_cmd, check=True)
        print("[VERBOSE] STEP1 completed successfully.")

        # STEP2
        out_pdb = f"{base}_out.pdb"
        step2_cmd = [
            sys.executable,
            args.step2_script_path,
            "--fpocket_pdb_pockets_file", out_pdb,
            "--fpocket_txt_file", info_txt,
        ]
        if args.do_not_delete_raw_csv:
            step2_cmd.append("--do_not_delete_raw_csv")
        print(f"[VERBOSE] Running STEP2: {' '.join(step2_cmd)}")
        subprocess.run(step2_cmd, check=True)
        print("[VERBOSE] STEP2 completed successfully.")

        # Delete modified info TXT unless skipped
        if not args.do_not_delete_modified_fpocket_txt:
            if os.path.exists(info_txt):
                os.remove(info_txt)
                print(f"[VERBOSE] Deleted modified info file {info_txt}")

        # Delete the fpocket output PDB unless user asked to keep it
        if not args.do_not_delete_fpocket_pdb:
            if os.path.exists(out_pdb):
                os.remove(out_pdb)
                print(f"[VERBOSE] Deleted fpocket output PDB {out_pdb}")

    except subprocess.CalledProcessError as e:
        # On STEP1/2 failure, write error CSV
        write_error_csv(agg_csv, f"Process failed: exit code {e.returncode}")
        sys.exit(e.returncode)
    except Exception as ex:
        write_error_csv(agg_csv, f"Unexpected error: {str(ex)}")
        sys.exit(1)

    # If reached here, aggregated CSV should already exist from STEP2
    # If not, ensure it's present
    if not os.path.exists(agg_csv):
        write_error_csv(agg_csv, "Aggregated CSV missing after successful STEP2")

if __name__ == "__main__":
    main()
