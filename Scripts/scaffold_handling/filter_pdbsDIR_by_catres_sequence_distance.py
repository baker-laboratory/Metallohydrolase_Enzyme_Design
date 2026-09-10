#!/usr/bin/env python3

"""
Usage example:

  Scripts/scaffold_handling/filter_pdbsDIR_by_catres_sequence_distance.py \
     --input_dir_of_pdbs_with_remark666_lines /path/to/pdb_dir \
     --catalytic_residue_pair_types "HIS HIS" "ASP HIS" \
     --max_sequence_distances 10 6 \
     --min_sequence_distances 4 5

Explanation:
  - For the pair "HIS HIS", the allowed sequence distance is >=4 and <=10.
  - For the pair "ASP HIS", the allowed sequence distance is >=5 and <=6.
  - If either pair in a given PDB file meets its respective (min ≤ distance ≤ max)
    criterion, the structure passes. (Only one pair has to pass.)
  - The script copies the .pdb (and the .trb if present) into
    `sequence_space_filtered_structures` inside the input directory.
"""

import os
import argparse
import shutil
import re
from multiprocessing import Pool, cpu_count

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Filter PDB files by minimal sequence space distance among catalytic residues (REMARK 666)."
    )
    parser.add_argument(
        "--input_dir_of_pdbs_with_remark666_lines",
        type=str,
        required=True,
        help="Directory containing PDB files with REMARK 666 lines."
    )
    parser.add_argument(
        "--catalytic_residue_pair_types",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Residue pairs to check, e.g. 'HIS HIS' or 'ASP HIS'. "
            "Pass multiple pairs by separating them with spaces, e.g. "
            "--catalytic_residue_pair_types 'HIS HIS' 'ASP HIS'."
        )
    )
    parser.add_argument(
        "--max_sequence_distances",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional maximum allowed sequence distance(s) for the specified residue pair(s). "
            "If provided, must match the number of pairs in --catalytic_residue_pair_types. "
            "If omitted, no maximum distance check is performed."
        )
    )
    parser.add_argument(
        "--min_sequence_distances",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional minimum allowed sequence distance(s) for the specified residue pair(s). "
            "If provided, must match the number of pairs in --catalytic_residue_pair_types. "
            "If omitted, no minimum distance check is performed."
        )
    )
    return parser.parse_args()

def gather_catalytic_positions(pdb_file):
    """
    From a single PDB file, gather all (resname, chain, resid) found in REMARK 666 lines.

    Returns:
        A list of tuples: [(resname, chain, resid), ...]
        where resname is e.g. 'HIS', chain is e.g. 'A', resid is an integer.
    """
    catalytic_positions = []
    with open(pdb_file, 'r') as f:
        for line in f:
            # We only look at REMARK 666 lines:
            if line.startswith("REMARK 666"):
                # Example line:
                #   REMARK 666 MATCH TEMPLATE X SZD    0 MATCH MOTIF A HIS   56  1  1
                # We'll extract chain, residue name, and residue number:
                match = re.search(r"MATCH MOTIF\s+([A-Za-z0-9])\s+([A-Z]{3})\s+(\d+)", line)
                if match:
                    chain = match.group(1)      # e.g. 'A'
                    resname = match.group(2)   # e.g. 'HIS'
                    resid = int(match.group(3))  # e.g. 56
                    catalytic_positions.append((resname, chain, resid))
    return catalytic_positions

def get_min_distance_info_for_pair(catalytic_positions, pair):
    """
    Compute the minimal sequence distance among the given catalytic_positions
    for a specific pair of residues, e.g. ("HIS", "HIS") or ("ASP", "HIS").

    Args:
        catalytic_positions (list): list of (resname, chain, resid)
        pair (tuple): (resname1, resname2), e.g. ("HIS", "ASP")

    Returns:
        (min_dist, (res1, chain1, resid1, res2, chain2, resid2)) or (None, None)
          - min_dist: integer for the minimal distance found
          - the second item is a tuple describing which two residues gave that min_dist
            in the format (resname, chain, resid, resname, chain, resid)
          - if no valid pairs found, returns (None, None)
    """
    r1, r2 = pair
    r1 = r1.upper()
    r2 = r2.upper()

    # Group positions by residue name
    positions_by_resname = {}
    for resname, chain, resid in catalytic_positions:
        positions_by_resname.setdefault(resname.upper(), []).append((chain, resid))

    # If one or both residues are not present, return None
    if r1 not in positions_by_resname or r2 not in positions_by_resname:
        return None, None

    positions_1 = positions_by_resname[r1]
    positions_2 = positions_by_resname[r2]
    min_dist = None
    best_pair_info = None

    if r1 == r2:
        # Compare among the same list, skipping self-pairs (i < j)
        for i in range(len(positions_1)):
            for j in range(i + 1, len(positions_1)):
                chain1, resid1 = positions_1[i]
                chain2, resid2 = positions_1[j]
                dist = abs(resid1 - resid2)
                if min_dist is None or dist < min_dist:
                    min_dist = dist
                    best_pair_info = (r1, chain1, resid1, r2, chain2, resid2)
    else:
        # Compare across the two lists
        for (chain1, resid1) in positions_1:
            for (chain2, resid2) in positions_2:
                dist = abs(resid1 - resid2)
                if min_dist is None or dist < min_dist:
                    min_dist = dist
                    best_pair_info = (r1, chain1, resid1, r2, chain2, resid2)

    return min_dist, best_pair_info

def process_pdb_file(args):
    """
    Worker function to process a single PDB file.

    Args (tuple):
        (pdb_file, pair_types, min_distances, max_distances, output_dir)

    Returns:
        (pdb_file, passes_filter_boolean)
    """
    pdb_file, pair_types, min_distances, max_distances, output_dir = args

    print(f"Processing file: {pdb_file}")
    catalytic_positions = gather_catalytic_positions(pdb_file)

    # If no catalytic positions found, automatically fail
    if not catalytic_positions:
        print(f"    No REMARK 666 lines found in {pdb_file}. Skipping.")
        return (pdb_file, False)

    # For each pair type, compute min distance and compare to the optional thresholds.
    # We only need to pass if ANY one of them meets its respective criteria.
    passes_filter = False

    for i, pair_str in enumerate(pair_types):
        pair_list = pair_str.split()  # e.g. ["HIS", "HIS"]
        if len(pair_list) != 2:
            print(f"    Warning: Pair '{pair_str}' is not two tokens. Skipping.")
            continue

        r1, r2 = pair_list[0], pair_list[1]
        min_dist, best_pair_info = get_min_distance_info_for_pair(catalytic_positions, (r1, r2))

        # Determine the thresholds for this pair if provided
        min_req = None if (min_distances is None) else min_distances[i]
        max_req = None if (max_distances is None) else max_distances[i]

        if min_dist is not None:
            # Print the pair that gave this min_dist
            if best_pair_info is not None:
                (rr1, ch1, rs1, rr2, ch2, rs2) = best_pair_info
                print(f"    Pair {pair_str}: Closest is {rr1} {ch1} {rs1} and {rr2} {ch2} {rs2} => distance = {min_dist}")

            # Now check if it meets the optional min and/or max
            meets_min = True  # default to True if no min_req
            meets_max = True  # default to True if no max_req

            if min_req is not None:
                meets_min = (min_dist >= min_req)
            if max_req is not None:
                meets_max = (min_dist <= max_req)

            if meets_min and meets_max:
                print(f"    ==> PASS for pair {pair_str}")
                passes_filter = True
                # Since only one pair has to pass, break out of the loop
                break
            else:
                if not meets_min and min_req is not None:
                    print(f"    Fails minimum threshold ({min_req}), distance is {min_dist}")
                if not meets_max and max_req is not None:
                    print(f"    Exceeds maximum threshold ({max_req}), distance is {min_dist}")
        else:
            print(f"    Pair {pair_str} not found in {pdb_file}.")

    if passes_filter:
        # Copy PDB to the output directory
        base_name = os.path.basename(pdb_file)
        shutil.copy2(pdb_file, os.path.join(output_dir, base_name))

        # Copy corresponding .trb if it exists
        trb_file = os.path.splitext(pdb_file)[0] + ".trb"
        if os.path.isfile(trb_file):
            trb_name = os.path.basename(trb_file)
            shutil.copy2(trb_file, os.path.join(output_dir, trb_name))

        return (pdb_file, True)
    else:
        print(f"    FAIL: No pairs met their thresholds for {pdb_file}.")
        return (pdb_file, False)

def main():
    print("### ENSURE THAT YOUR PDB FILES HAVE REMARK 666 LINES ###")

    # Parse arguments
    args = parse_arguments()
    input_dir = args.input_dir_of_pdbs_with_remark666_lines
    pair_types = args.catalytic_residue_pair_types
    max_distances = args.max_sequence_distances
    min_distances = args.min_sequence_distances

    # Validate lengths if provided
    if (max_distances is not None) and (len(pair_types) != len(max_distances)):
        raise ValueError(
            "Number of --catalytic_residue_pair_types must match the number of --max_sequence_distances."
        )
    if (min_distances is not None) and (len(pair_types) != len(min_distances)):
        raise ValueError(
            "Number of --catalytic_residue_pair_types must match the number of --min_sequence_distances."
        )

    print(f"Input directory: {input_dir}")
    print(f"Catalytic residue pair types: {pair_types}")
    print(f"Max sequence distances: {max_distances if max_distances else 'None (no max check)'}")
    print(f"Min sequence distances: {min_distances if min_distances else 'None (no min check)'}")

    # Make output directory
    output_dir = os.path.join(input_dir, "sequence_space_filtered_structures")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    else:
        print(f"Output directory already exists: {output_dir}")

    # Gather all PDB files in the input directory
    pdb_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(".pdb")
    ]
    print(f"Found {len(pdb_files)} PDB files to process.")

    # Prepare arguments for parallel processing
    task_args = [
        (pdb_file, pair_types, min_distances, max_distances, output_dir)
        for pdb_file in pdb_files
    ]

    # Use multiprocessing to process files in parallel
    num_cpus = cpu_count()
    print(f"Using up to {num_cpus} CPUs for parallel processing...")

    if len(pdb_files) > 0:
        with Pool(processes=num_cpus) as pool:
            results = pool.map(process_pdb_file, task_args)

        # Gather which files passed
        passed_files = [r[0] for r in results if r[1]]
        print(f"\nFinished processing. {len(passed_files)} of {len(pdb_files)} PDB files passed the filter.")
    else:
        print("No PDB files found. Exiting.")

if __name__ == "__main__":
    main()
