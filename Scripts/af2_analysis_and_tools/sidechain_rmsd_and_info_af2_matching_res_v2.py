#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 2022
Updated on Tue Apr 30 2024

@author: ikalvet, updated by Seth Woodbury and Donghyo Kim

Computes per-atom-group RMSDs for catalytic residues between predicted models
and the designs they came from, and appends them to a scorefile.

Differs from sidechain_rmsd_and_info_af2_matching_res.py in two ways:

  --suffix_by_extension_to_match_with_previousFILE takes the exact suffix the
  prediction step appends. The script strips that one suffix to recover the
  reference filename, instead of searching candidate prefixes -- unambiguous
  where several stages have each added a suffix, and considerably faster.

  Catalytic residues are reindexed in REMARK 666 order rather than by residue
  number, so the cat_<AA><N>_* columns stay aligned across structures.
"""

import argparse
import os
import glob
import pandas as pd
import numpy as np
import pyrosetta
import multiprocessing
from multiprocessing import Pool
import json
import time
import re


# Initialize PyRosetta with custom or default parameters
def init_pyrosetta(params_files):
    expanded_params = []
    if params_files:
        for param in params_files:
            expanded_params.extend(glob.glob(param))  # Expand wildcards
    if expanded_params:
        params_str = ' '.join(expanded_params)
        options = f'-extra_res_fa {params_str} -mute all'
    else:
        options = '-mute all -beta_nov16'
    pyrosetta.init(options)
    print("PyRosetta initialized with options:", options)


# Configure command line argument parser
def configure_parser():
    parser = argparse.ArgumentParser(description="Process and analyze PDB files with PyRosetta.")
    parser.add_argument('--params', nargs='+', help='Params files for PyRosetta initialization.')
    parser.add_argument('--ref_pdbs_dir', type=str, required=True, help='Directory containing reference PDBs.')
    parser.add_argument('--scorefile', type=str, required=True, help='CSV file with scoring data.')
    parser.add_argument('--column_name_for_pdb_path', type=str, default='pdb_path', help='Column name for PDB paths.')
    parser.add_argument('--max_possible_num_of_suffixes', type=int, default=1,
                        help='Maximum matches allowed per reference PDB (used only if no suffix_by_extension is given).')
    parser.add_argument('--output_file', type=str, default='scores_updated.sc', help='Output CSV file name.')
    parser.add_argument('--atom_groups', type=str, help='JSON string defining atom groups and labels.')

    # NEW FLAG: If provided, bypass prefix-based matching and do suffix-based matching
    parser.add_argument('--suffix_by_extension_to_match_with_previousFILE', type=str, default=None,
                        help="If provided, remove this suffix from the PDB's basename (once, at the very end), then "
                             "append .pdb to find the reference file. Raise error if not found.")
    return parser


# Load data from scorefile
def load_data(scorefile):
    data = pd.read_csv(scorefile)
    print(f"Loaded {len(data)} entries from the score file '{scorefile}'.")
    return data


# NEW FUNCTION: Suffix-based matching
def match_pdbs_by_suffix(ref_pdbs_dir, pdb_paths, suffix_to_remove):
    """
    For each target PDB, remove `suffix_to_remove` from the end of its basename, then
    append '.pdb' to find the reference. Raise error if the resulting file does not exist.
    Returns:
      matches: dict { target_pdb -> ref_pdb_fullpath }
      unmatched_targets: list of any that fail (though we raise FileNotFoundError by default)
    """
    start_time = time.time()
    matches = {}
    unmatched_targets = []

    for target_pdb in pdb_paths:
        base_no_ext = os.path.splitext(os.path.basename(target_pdb))[0]

        # Use a regex to remove suffix exactly at the end (once). If not present, raise an error.
        pattern = re.escape(suffix_to_remove) + r'$'  # suffix must appear at end
        truncated = re.sub(pattern, '', base_no_ext)
        if truncated == base_no_ext:
            # Suffix not found at the end
            unmatched_targets.append(target_pdb)
            continue

        # Construct the candidate reference path
        ref_candidate = os.path.join(ref_pdbs_dir, f"{truncated}.pdb")
        if not os.path.isfile(ref_candidate):
            # Per your request: raise an error if it doesn't exist
            raise FileNotFoundError(
                f"ERROR: After removing suffix '{suffix_to_remove}' from '{base_no_ext}', "
                f"the expected reference PDB '{ref_candidate}' does not exist."
            )
        matches[target_pdb] = ref_candidate

    elapsed_time = time.time() - start_time
    print(f"Suffix-based matching completed in {elapsed_time:.2f} seconds")
    print(f"Found {len(matches)} matches; {len(unmatched_targets)} unmatched (missing suffix).")
    if unmatched_targets:
        print(f"Unmatched (suffix not found at end): {unmatched_targets}")
    return matches, unmatched_targets


# Original prefix-based matching
def match_pdbs_prefix(ref_pdbs_dir, pdb_paths, max_possible_num_of_suffixes=1):
    """
    If no suffix is specified, we do the old approach of matching references
    by checking if target_pdb_basename.startswith(ref_basename).
    """
    start_time = time.time()

    # Prepare reference PDB map: basename -> full path
    ref_pdbs = {os.path.splitext(os.path.basename(p))[0]: p
                for p in glob.glob(os.path.join(ref_pdbs_dir, '*.pdb'))}
    ref_usage_count = {key: 0 for key in ref_pdbs.keys()}

    target_pdbs = [os.path.basename(p) for p in pdb_paths]

    matches = {}
    unmatched_targets = []

    processed_count = 0
    total_files = len(target_pdbs)
    for target_pdb in target_pdbs:
        target_base = os.path.splitext(target_pdb)[0]
        match_found = False
        # iterate over a copy to allow modifying dictionary
        for ref_base, ref_path in list(ref_pdbs.items()):
            if (target_base.startswith(ref_base)
                    and ref_usage_count[ref_base] < max_possible_num_of_suffixes):
                matches[target_pdb] = ref_path
                ref_usage_count[ref_base] += 1
                match_found = True
                if ref_usage_count[ref_base] >= max_possible_num_of_suffixes:
                    del ref_pdbs[ref_base]  # remove from further consideration
                break
        if not match_found:
            unmatched_targets.append(target_pdb)

        processed_count += 1
        if (processed_count % 1000 == 0) or (processed_count == total_files):
            elapsed_time = time.time() - start_time
            print(f"{processed_count}/{total_files} matched; "
                  f"{total_files - processed_count} remaining. "
                  f"{len(ref_pdbs)} refs left. Time elapsed: {elapsed_time:.2f} s")

    # Final
    print(f"Total prefix-based matching completed in {time.time() - start_time:.2f} seconds")
    print(f"Found {len(matches)} matches. {len(unmatched_targets)} unmatched: {unmatched_targets}")
    return matches, unmatched_targets


def get_catalytic_residues(pdb_file):
    """
    Parse REMARK 666 lines at the top of the PDB, stopping at the first ATOM record.
    Return a list of dicts *in the order they appear*:
      [
        {
          'res_index_in_666': 1-based index in the remark,
          'resnum': <integer residue number from line>,
          'name3': <3-letter code (HIS, GLU, ASP, etc.)>,
          'chain': <chain ID>
        },
        ...
      ]
    """
    with open(pdb_file, 'r') as file:
        lines = file.readlines()

    catalytic_residues = []
    remark_index = 0
    for line in lines:
        if line.startswith("ATOM"):
            # Stop reading after the first ATOM line
            break
        if "REMARK 666" in line:
            # example line format:
            # REMARK 666 MATCH TEMPLATE X SZD    0 MATCH MOTIF A HIS   91  1  1
            # We'll parse out chain = parts[9], residue_name = parts[10], residue_number = parts[11], ...
            parts = line.split()
            remark_index += 1
            chain = parts[9]
            residue_name = parts[10]
            residue_number = int(parts[11])

            catalytic_residues.append({
                'res_index_in_666': remark_index,
                'resnum': residue_number,
                'name3': residue_name,
                'chain': chain
            })

    print(f"\n\nCatalytic residues found in {pdb_file}:")
    for r in catalytic_residues:
        print(f"  REMARK order {r['res_index_in_666']}, "
              f"{r['name3']} {r['resnum']} (chain {r['chain']})")

    return catalytic_residues


def detailed_rmsd_calculation(ref_pose, af2_pose, catalytic_residues_list, atom_groups):
    """
    Calculate RMSD for specific atoms in each catalytic residue.
    *Preserve the order from the REMARK 666 lines* by iterating the list in that order.
    For each residue, label them as "cat_<name3><i>_<label>" where <i> is the remark order (1-based).
    """
    rmsd_results = {}
    print("Starting RMSD calculations...")

    for residue_info in catalytic_residues_list:
        i = residue_info['res_index_in_666']
        residue_number = residue_info['resnum']
        residue_name = residue_info['name3']

        label_prefix = f"cat_{residue_name}{i}"
        print(f"Processing remarkIndex={i}, residueNum={residue_number} ({residue_name})")

        # Check if we have an atom group definition for this residue_name
        if residue_name in atom_groups:
            for config in atom_groups[residue_name]:
                atom_names = config['atoms']
                label_suffix = config['label']
                rmsd_label = f"{label_prefix}_{label_suffix}"

                euclidean_deviation_values = []
                for atom_name in atom_names:
                    # Safely check if pose has that residue index and that atom
                    if (residue_number <= ref_pose.size() and
                        ref_pose.residue(residue_number).has(atom_name) and
                        residue_number <= af2_pose.size() and
                        af2_pose.residue(residue_number).has(atom_name)):
                        ref_xyz = ref_pose.residue(residue_number).xyz(atom_name)
                        af2_xyz = af2_pose.residue(residue_number).xyz(atom_name)
                        deviation = (ref_xyz - af2_xyz).norm()
                        euclidean_deviation_values.append(deviation)
                        print(f"  Atom: {atom_name}, Deviation: {deviation:.3f} Å")
                    else:
                        print(f"  WARNING: Could not find atom '{atom_name}' in residue {residue_number}")

                if euclidean_deviation_values:
                    # RMS of the Euclidean distances
                    rmsd = np.sqrt(np.mean(np.square(euclidean_deviation_values)))
                    rmsd_results[rmsd_label] = rmsd
                    print(f"  => {rmsd_label} RMSD: {rmsd:.3f} Å")
        else:
            print(f"No atom group configuration for {residue_name}.")

    print("RMSD results computed (in original remark order).")
    print("Final keys in rmsd_results:")
    for k in rmsd_results:
        print(f"  {k} -> {rmsd_results[k]:.3f} Å")

    return rmsd_results


# Function to update the scorefile with RMSD results
def update_scorefile(score_data, list_of_dicts, output_path):
    """
    list_of_dicts is a list of:
      {
         'pdb_path': <the target pdb>,
         'cat_HIS1_bb_rmsd': value,
         'cat_GLU2_bb_rmsd': value,
         ...
      }

    We do a naive concat of these new columns to the original DataFrame.
    The assumption: The 'score_data' and the new RMSD columns have the same row order
    or we must re-merge them carefully. But typically you'd do a merge on pdb_path
    if you want 1:1 alignment by path.
    """
    # We strongly recommend merging by 'pdb_path' rather than just concatenating columns by index.
    # Here is a MERGE approach:
    updates_df = pd.DataFrame(list_of_dicts)
    # merge on 'pdb_path' (left join keeps original rows; you might want inner join).
    updated_data = pd.merge(score_data, updates_df, on="pdb_path", how="left")

    updated_data.to_csv(output_path, index=False)
    print(f"Updated scorefile saved to {output_path}.")


# Multiprocessing routine
def multiprocessing_handler_detailed(matches, atom_groups, score_data, output_file):
    start_time = time.time()
    num_processes = multiprocessing.cpu_count()
    print(f"Using {num_processes} processes.")

    # Prepare the list of tasks:
    #   pairs = [((ref_pdb, af2_pdb), atom_groups), ...]
    pairs = [((ref_pdb, af2_pdb), atom_groups) for (af2_pdb, ref_pdb) in matches.items()]
    total_files = len(pairs)

    results = []
    with Pool(processes=num_processes) as pool:
        for i, result in enumerate(pool.imap(process_detailed_match, pairs), 1):
            results.append(result)
            if i % 50 == 0 or i == total_files:
                elapsed_time = time.time() - start_time
                avg_time_per_file = elapsed_time / i
                files_left = total_files - i
                est_time_left = avg_time_per_file * files_left
                print(f"{i}/{total_files} RMSDs done; ~{est_time_left:.1f} s left.")

    # Once all results are processed, update the score file
    update_scorefile(score_data, results, output_file)


def process_detailed_match(args):
    """
    Args: ((ref_pdb, af2_pdb), atom_groups)
    Returns: { 'pdb_path': af2_pdb, 'cat_XXX1_bb_rmsd': val, ... }
    """
    (ref_pdb, af2_pdb), atom_groups = args
    print(f"Ref: {ref_pdb}\nAF2: {af2_pdb}")

    ref_pose = pyrosetta.rosetta.core.import_pose.pose_from_file(ref_pdb)
    af2_pose = pyrosetta.rosetta.core.import_pose.pose_from_file(af2_pdb)

    # Get catalytic residues in the order they appear
    catalytic_residues_list = get_catalytic_residues(ref_pdb)
    # Compute RMSDs preserving that order
    rmsd_dict = detailed_rmsd_calculation(ref_pose, af2_pose, catalytic_residues_list, atom_groups)

    # Include the pdb_path for merging
    return {'pdb_path': af2_pdb, **rmsd_dict}


def main():
    args = configure_parser().parse_args()
    init_pyrosetta(args.params)

    score_data = load_data(args.scorefile)
    pdb_paths = score_data[args.column_name_for_pdb_path].tolist()

    # 1) If user provided suffix_by_extension, do suffix-based matching
    # 2) Otherwise, do the old prefix-based matching
    if args.suffix_by_extension_to_match_with_previousFILE:
        print("Performing suffix-based matching...")
        matches, unmatched_targets = match_pdbs_by_suffix(
            args.ref_pdbs_dir,
            pdb_paths,
            args.suffix_by_extension_to_match_with_previousFILE
        )
    else:
        print("Performing prefix-based matching...")
        matches, unmatched_targets = match_pdbs_prefix(
            args.ref_pdbs_dir,
            pdb_paths,
            args.max_possible_num_of_suffixes
        )

    # Load JSON for the atom groups
    atom_groups = json.loads(args.atom_groups) if args.atom_groups else {}

    if matches:
        multiprocessing_handler_detailed(matches, atom_groups, score_data, args.output_file)
        print("Detailed processing of matches completed.")
    else:
        print("No matches found to process.")

    print("Script execution completed.")


if __name__ == "__main__":
    main()
