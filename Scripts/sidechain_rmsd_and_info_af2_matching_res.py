#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 2022
Updated on Tue Apr 30 2024

@author: ikalvet & updated by Seth Woodbury + Donghyo Kim

NOTE: WE NEED TO FIX THIS SCRIPT SUCH THAT THE REINDEXING KEEPS THE OG BLOCK ORDER FROM THE REMARK 666 LINES OR SOMETHING
RN IT CURRENTLY REINDEXES BASED ON WHAT RESIDUE NUMBER IT IS!
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
import re  # Import regex module

# Initialize PyRosetta with custom or default parameters
def init_pyrosetta(params_files):
    expanded_params = []
    for param in params_files:
        expanded_params.extend(glob.glob(param))  # Expand each wildcard and add to the list
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
    parser.add_argument('--max_possible_num_of_suffixes', type=int, default=1, help='Maximum number of matches allowed per reference PDB. Default is 1.')
    parser.add_argument('--output_file', type=str, default='scores_updated.sc', help='Suffix for the output score file.')
    parser.add_argument('--atom_groups', type=str, help='JSON string defining atom groups and labels')
    return parser

# Load data from scorefile
def load_data(scorefile):
    data = pd.read_csv(scorefile)
    print(f"Loaded {len(data)} entries from the score file.")
    return data

# Match reference PDBs to target PDBs using suffixes
def match_pdbs(ref_pdbs_dir, pdb_paths, max_possible_num_of_suffixes=1):
    start_time = time.time()  # Start timing the function
    
    # Prepare reference PDB map: basename without extension to full path
    ref_pdbs = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(ref_pdbs_dir, '*.pdb'))}
    ref_usage_count = {key: 0 for key in ref_pdbs.keys()}  # Track usage of each ref PDB

    # Prepare target PDBs for matching
    target_pdbs = [os.path.basename(p) for p in pdb_paths]

    # Initialize matching results
    matches = {}
    unmatched_targets = []

    # Matching process
    processed_count = 0
    total_files = len(pdb_paths)
    for target_pdb in pdb_paths:
        target_base = os.path.splitext(os.path.basename(target_pdb))[0]
        match_found = False
        for ref_base, ref_path in list(ref_pdbs.items()):  # Iterate over a copy to allow modifying the dictionary
            if target_base.startswith(ref_base) and ref_usage_count[ref_base] < max_possible_num_of_suffixes:
            #if ref_base.startswith(target_base) and ref_usage_count[ref_base] < max_possible_num_of_suffixes:
                matches[target_pdb] = ref_path
                ref_usage_count[ref_base] += 1
                match_found = True
                if ref_usage_count[ref_base] >= max_possible_num_of_suffixes:
                    del ref_pdbs[ref_base]  # Remove ref PDB from further consideration
                break
        if not match_found:
            unmatched_targets.append(target_pdb)

        processed_count += 1
        if processed_count % 1000 == 0 or processed_count == total_files:  # Changed from 100 to 1000
            elapsed_time = time.time() - start_time
            print(f"{processed_count}/{total_files} files matched, {total_files - processed_count} files remaining, {len(ref_pdbs)} reference PDBs still available for matching. Time elapsed: {elapsed_time:.2f} seconds")

    # Final report
    print(f"Total matching completed in {time.time() - start_time:.2f} seconds")
    print(f"Found {len(matches)} matches. {len(unmatched_targets)} targets remain unmatched. Unmatched targets: {unmatched_targets}")
    #print(matches) #debug

    return matches, unmatched_targets

def get_catalytic_residues(pdb_file):
    with open(pdb_file, 'r') as file:
        lines = file.readlines()
    
    catalytic_residues = {}
    for line in lines:
        if "ATOM" in line:
            break
        if "REMARK 666" in line:
            parts = line.split()
            chain = parts[9]
            residue_name = parts[10]
            residue_number = int(parts[11])
            catalytic_residues[residue_number] = {'name3': residue_name, 'chain': chain}

    print('')
    print('')
    print(f"Catalytic residues identified in {pdb_file}: {catalytic_residues}")
    return catalytic_residues

# Expanded RMSD calculation to handle specific atoms and residues
def detailed_rmsd_calculation(ref_pose, af2_pose, catalytic_residues_dict, atom_groups):
    """
    Calculate RMSD for specific atoms and catalytic residues between two protein structures.
    :param ref_pose: Pose object for the reference protein structure.
    :param af2_pose: Pose object for the predicted protein structure.
    :param catalytic_residues_dict: Dictionary with residue numbers as keys and dicts with additional info as values.
    :param atom_groups: Dictionary defining atom groups and their labels for RMSD calculations.
    :return: Dictionary with residue indices as keys and RMSD values, labeled accordingly.
    """
    rmsd_results = {}
    print("Starting RMSD calculations...")

    # Iterate over each residue using the keys directly from the dictionary
    for residue_index, residue_info in catalytic_residues_dict.items():
        residue_name = residue_info['name3']
        label_prefix = f"cat_{residue_name}{residue_index}"
        print(f"Processing residue {residue_index} ({residue_name})")

        if residue_name in atom_groups:
            for config in atom_groups[residue_name]:
                atom_names = config['atoms']
                label_suffix = config['label']
                rmsd_label = f"{label_prefix}_{label_suffix}"
                
                # Calculate RMSD for the specific atoms
                euclidean_deviation_values = []
                for atom_name in atom_names:
                    if ref_pose.residue(residue_index).has(atom_name) and af2_pose.residue(residue_index).has(atom_name):
                        ref_xyz = ref_pose.residue(residue_index).xyz(atom_name)
                        af2_xyz = af2_pose.residue(residue_index).xyz(atom_name)
                        deviation = (ref_xyz - af2_xyz).norm()
                        euclidean_deviation_values.append(deviation)
                        print(f"Atom: {atom_name}, Euclidean Deviation: {deviation} A")

                if euclidean_deviation_values:
                    rmsd = np.sqrt(np.mean(np.square(euclidean_deviation_values)))
                    rmsd_results[rmsd_label] = rmsd
                    print(f"Residue {residue_index}, {label_suffix} RMSD: {rmsd}")
        else:
            print(f"No atom group configuration for {residue_name}.")

    print("Generated RMSD results keys before sorting and reindexing:")
    print(rmsd_results.keys())

    # Reindexing the results starting from 1
    print("###############################################")
    print("########## REINDEXING THE RESULTS... ##########")
    print("###############################################")
    reindexed_results = {}
    residue_index_map = {}
    index = 1
    for key in sorted(rmsd_results.keys(), key=lambda x: int(re.match(r"cat_[A-Z]{3}(\d+)_", x).group(1))):
        original_residue_num = re.match(r"cat_[A-Z]{3}(\d+)_", key).group(1)
        if original_residue_num not in residue_index_map:
            residue_index_map[original_residue_num] = index
            index += 1
        new_key = re.sub(r'(\d+)(_)', f'{residue_index_map[original_residue_num]}\\2', key)
        reindexed_results[new_key] = rmsd_results[key]
        print(f"Key '{key}' reindexed to '{new_key}'.")

    print("Reindexed RMSD results:")
    print(reindexed_results)
    print("###############################################")
    return reindexed_results

# Function to update the scorefile with RMSD results
def update_scorefile(score_data, reindexed_results, output_path):
    updates = []
    for result in reindexed_results:
        pdb_path = result['pdb_path']
        update = {}
        for key, value in result.items():
            if key != 'pdb_path':
                update[key] = value
        updates.append(update)

    updates_df = pd.DataFrame(updates)
    updated_data = pd.concat([score_data, updates_df], axis=1)
    updated_data.to_csv(output_path, index=False)
    print(f"Updated scorefile saved to {output_path}.")

# Updating the multiprocessing handler to accommodate detailed RMSD calculations
def multiprocessing_handler_detailed(matches, atom_groups, score_data, output_file):
    start_time = time.time()
    num_processes = multiprocessing.cpu_count()
    print(f"Using {num_processes} processes!!")

    # Convert matches to a list of tuples suitable for processing
    pairs = [((matches[af2_pdb], af2_pdb), atom_groups) for af2_pdb in matches]
    total_files = len(pairs)

    with Pool(processes=num_processes) as pool:
        # Process each pair using a pool of workers; use imap_unordered for potentially faster processing
        results = []
        for i, result in enumerate(pool.imap(process_detailed_match, pairs), 1):
            results.append(result)
            if i % 500 == 0 or i == total_files:  # Reporting every 500 files or when processing is complete
                elapsed_time = time.time() - start_time
                avg_time_per_file = elapsed_time / i
                files_left = total_files - i
                estimated_time_left = avg_time_per_file * files_left
                print(f"{i}/{total_files} PDB RMSDs calculated, {files_left} left, "
                      f"Time elapsed: {elapsed_time:.2f} s, Avg time per file: {avg_time_per_file:.2f} s, "
                      f"Estimated time left: {estimated_time_left:.2f} s")

    # Once all results are processed, update the score file
    update_scorefile(score_data, results, output_file)

def process_detailed_match(args):
    ((ref_pdb, af2_pdb), atom_groups) = args  # Ensure this matches the structure passed from multiprocessing handler
    ref_pose = pyrosetta.rosetta.core.import_pose.pose_from_file(ref_pdb)
    af2_pose = pyrosetta.rosetta.core.import_pose.pose_from_file(af2_pdb)
    catalytic_residues = get_catalytic_residues(ref_pdb)  # Ensure this function returns the correct format
    reindexed_results = detailed_rmsd_calculation(ref_pose, af2_pose, catalytic_residues, atom_groups)
    return {'pdb_path': af2_pdb, **reindexed_results}

# Main function to orchestrate the processing
def main():
    args = configure_parser().parse_args()
    init_pyrosetta(args.params)
    score_data = load_data(args.scorefile)
    pdb_paths = score_data[args.column_name_for_pdb_path].tolist()
    matches, unmatched_targets = match_pdbs(args.ref_pdbs_dir, pdb_paths, args.max_possible_num_of_suffixes)

    # Directly use the dictionary, no need to parse JSON again
    atom_groups = json.loads(args.atom_groups)

    if matches:
        detailed_results = multiprocessing_handler_detailed(matches, atom_groups, score_data, args.output_file)
        print("Detailed processing of matches completed.")
    else:
        print("No matches found to process.")

    print("Script execution completed.")

if __name__ == "__main__":
    main()
