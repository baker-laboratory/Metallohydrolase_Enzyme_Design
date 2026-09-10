#!/usr/bin/env python3
"""
Script: check_pdbs_for_remark666_lines.py
Author: Seth Woodbury (woodbuse@uw.edu)
Date: 2025-01-22

Description:
    This script checks PDB files for 'REMARK 666' lines. You may provide:
      1) --pdb_input: a single file or glob pattern (e.g., 'my_path/struct_*.pdb')
      2) --pdb_dir_input: a directory containing PDB files.

    You can also specify:
      --number_of_minimal_remark666_lines_to_expect <int>
      --number_of_maximal_remark666_lines_to_expect <int>

    The script will:
      - Identify all relevant PDB files from the input.
      - For each PDB, count how many lines begin with 'REMARK 666'.
      - Check whether that count meets or exceeds the minimum and/or does not exceed the maximum, if specified.
      - Print a summary of any that fail the requirement and a final aggregate report.

Example Usage:

    # Check a specific directory with no min or max specified:
    python check_pdbs_for_remark666_lines.py \
        --pdb_dir_input /path/to/pdbs/

    # Check a glob pattern:
    python check_pdbs_for_remark666_lines.py \
        --pdb_input "/path/to/pdbs/*group1*.pdb" \
        --number_of_minimal_remark666_lines_to_expect 3 \
        --number_of_maximal_remark666_lines_to_expect 10

    # Provide both a directory and a glob pattern (script merges them):
    python check_pdbs_for_remark666_lines.py \
        --pdb_dir_input /path/to/pdbs/ \
        --pdb_input "/other_pdbs/special/*.pdb" \
        --number_of_minimal_remark666_lines_to_expect 2
"""

import argparse
import glob
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description="Check PDB files for 'REMARK 666' lines and report any that do not meet min/max criteria."
    )
    parser.add_argument(
        "--pdb_input",
        type=str,
        help="A single PDB file or a glob pattern (e.g., 'path/*.pdb') to check."
    )
    parser.add_argument(
        "--pdb_dir_input",
        type=str,
        help="A directory containing one or more PDB files (all '*.pdb' files in that directory will be checked)."
    )
    parser.add_argument(
        "--number_of_minimal_remark666_lines_to_expect",
        type=int,
        default=None,
        help="Minimum number of 'REMARK 666' lines expected in each PDB. "
             "If not specified, no minimum check is performed."
    )
    parser.add_argument(
        "--number_of_maximal_remark666_lines_to_expect",
        type=int,
        default=None,
        help="Maximum number of 'REMARK 666' lines expected in each PDB. "
             "If not specified, no maximum check is performed."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Collect PDB files from user input
    pdb_files = set()

    if args.pdb_input:
        # expand the glob pattern
        matched_files = glob.glob(args.pdb_input)
        for mf in matched_files:
            if os.path.isfile(mf):
                pdb_files.add(os.path.abspath(mf))

    if args.pdb_dir_input:
        if os.path.isdir(args.pdb_dir_input):
            dir_matched_files = glob.glob(os.path.join(args.pdb_dir_input, "*.pdb"))
            for dm in dir_matched_files:
                pdb_files.add(os.path.abspath(dm))
        else:
            print(f"[WARNING] --pdb_dir_input provided but '{args.pdb_dir_input}' is not a valid directory.", file=sys.stderr)

    pdb_files = sorted(pdb_files)  # sort for consistent ordering

    # If no files found, exit early
    if not pdb_files:
        print("No PDB files found. Exiting.")
        sys.exit(0)

    print(f"Identified {len(pdb_files)} PDB file(s) to check.\n")

    min_lines = args.number_of_minimal_remark666_lines_to_expect
    max_lines = args.number_of_maximal_remark666_lines_to_expect

    # Keep track of results
    invalid_pdbs = []
    valid_count = 0

    # Check each PDB file for REMARK 666 lines
    for pdb_file in pdb_files:
        count_remark666 = 0

        try:
            with open(pdb_file, "r") as f:
                for line in f:
                    # We only check if it *starts* with 'REMARK 666'
                    # (assuming that is the standard format)
                    if line.startswith("REMARK 666"):
                        count_remark666 += 1
        except Exception as e:
            print(f"[ERROR] Could not read file '{pdb_file}': {e}", file=sys.stderr)
            invalid_pdbs.append(pdb_file)
            continue

        # Evaluate against min/max
        meets_min = (min_lines is None) or (count_remark666 >= min_lines)
        meets_max = (max_lines is None) or (count_remark666 <= max_lines)

        if meets_min and meets_max:
            valid_count += 1
        else:
            invalid_pdbs.append(pdb_file)

    total = len(pdb_files)
    invalid_count = len(invalid_pdbs)

    # Print invalid results
    if invalid_count > 0:
        print("The following PDBs do NOT meet the REMARK 666 criteria:\n")
        for ipdb in invalid_pdbs:
            print(f"  - {ipdb}")
        print("")

    # Summarize
    print(
        f"Summary: {valid_count}/{total} PDB(s) have the expected number of 'REMARK 666' lines.\n"
        f"{invalid_count}/{total} PDB(s) are missing 'REMARK 666' lines or exceed the constraints."
    )

if __name__ == "__main__":
    main()
