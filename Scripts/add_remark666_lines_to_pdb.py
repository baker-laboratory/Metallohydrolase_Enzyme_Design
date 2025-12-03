"""
Created 2024-04-24 by Seth Woodbury (woodbuse@uw.edu)
This script maps and copies remark 666 + header lines from pdbs in a reference directory to a new directory

#IMPORTANT NOTE: This script requires that the new pdbs have an added suffix that utilizes the old name (forward mapping)
This cannot map backwards yet, really because I have no purpose for it. It is meant to propagate REMARK 666 + other important
lines down the pipeline from pdb to pdb. Enjoy. Please contact me if you add/want to add anything as I want to make this as general as possible.
"""
# Example command using the --find_suffixes_from_random_sample flag:
# python your_script_name.py \
#   --new_pdb_dir "/path/to/new_pdbs" \
#   --reference_old_pdb_dir "/path/to/old_pdbs" \
#   --debug \
#   --find_suffixes_from_random_sample 30 \
#   --remove_unwanted_lines_from_new_pdbs "REMARK SomeUnwantedLine" "REMARK AnotherUnwantedLine" \
#   --additional_lines_from_old_pdbs_to_copy "DATE" "DIG"
#
# This command will randomly select 30 old files (if available) to determine the suffixes used in new file names.

import os
import glob
import multiprocessing
import time
from pathlib import Path
import argparse
import random

def collect_headers(old_file, additional_lines_prefixes):
    """
    Collects header lines from the old PDB file that start with specific prefixes.
    Optionally includes additional prefixes provided by the user.
    
    Parameters:
        old_file (str): Path to the old PDB file.
        additional_lines_prefixes (list): List of additional prefixes to include in headers.
    
    Returns:
        list: A list of header lines to be added to new PDB files.
    """
    headers_to_add = []
    seen_headers = set()
    with open(old_file, 'r') as file:
        for line in file:
            if line.startswith(tuple(["HEADER", "REMARK", "HETNAM", "LINK"] + additional_lines_prefixes)) and line not in seen_headers:
                headers_to_add.append(line)
                seen_headers.add(line)
    return headers_to_add

def copy_remark_lines(old_file, suffixes, new_pdbs_dir, headers_to_add, unwanted_lines):
    """
    Copies necessary header lines from old PDB files to new PDB files, removing unwanted lines.
    
    Parameters:
        old_file (str): Path to the old PDB file.
        suffixes (dict): Dictionary mapping old file bases to new file suffixes.
        new_pdbs_dir (str): Directory containing new PDB files.
        headers_to_add (list): List of headers to add to the new files.
        unwanted_lines (list): Lines to remove from new files if present.
    
    Returns:
        list: Results detailing what was done for each file.
    """
    old_base = Path(old_file).stem
    detailed_results = []
    for suffix in suffixes.get(old_base, []):
        new_file_path = Path(new_pdbs_dir) / f"{old_base}{suffix}.pdb"
        was_modified = False
        if new_file_path.exists():
            with open(new_file_path, 'r+') as new_file:
                content = new_file.readlines()
                # Remove unwanted specific lines if they exist
                content = [line for line in content if line.strip() not in [ul.strip() for ul in unwanted_lines]]
                content_set = set(content)
                
                # Determine which headers need to be added (avoiding duplication)
                headers_needed = [header for header in headers_to_add if header not in content_set]
                unique_headers = []
                seen = set()
                for header in headers_needed:
                    if header not in seen:
                        seen.add(header)
                        unique_headers.append(header)
                
                # Write back to file if there are any unique headers to add
                if unique_headers:
                    new_file.seek(0, 0)
                    new_file.writelines(unique_headers + content)
                    was_modified = True

            # Track the file modification status
            detailed_result = f"Headers copied from {old_file} to {new_file_path}: {''.join(unique_headers)}" if unique_headers else f"No new headers needed copying for {new_file_path}."
            detailed_results.append((new_file_path, was_modified, detailed_result))
        else:
            detailed_results.append((new_file_path, False, f"File not found: {new_file_path}"))
    return detailed_results

def add_remark666_lines_to_pdb_files(new_pdbs_dir, reference_old_pdbs_dir, debug, remove_unwanted_lines, additional_lines_prefixes, sample_size=None):
    """
    Main function to process PDB files, copying specific header lines from old to new PDBs, handling unwanted lines.
    Optionally considers a random sample of old files if sample_size is provided.
    
    Parameters:
        new_pdbs_dir (str): Directory containing new PDB files.
        reference_old_pdbs_dir (str): Directory containing old reference PDB files.
        debug (bool): Flag to enable detailed debug output.
        remove_unwanted_lines (list): List of lines to remove from new PDB files.
        additional_lines_prefixes (list): Additional line prefixes to copy from old PDBs.
        sample_size (int): Optional number of old files to sample for suffix identification.
    """
    print("\nStarting the processing of PDB files...")
    old_pdb_files = glob.glob(os.path.join(reference_old_pdbs_dir, '*.pdb'))
    new_pdb_files = glob.glob(os.path.join(new_pdbs_dir, '*.pdb'))
    print(f"Found {len(old_pdb_files)} old PDB files and {len(new_pdb_files)} new PDB files.\n")

    start_time = time.time()
    all_suffixes, cumulative_suffixes = identify_suffixes(old_pdb_files, new_pdb_files, debug, sample_size)
    
    num_cpus = multiprocessing.cpu_count()
    print(f"\nUsing {num_cpus} CPUs for parallel processing...")
    with multiprocessing.Pool(processes=num_cpus) as pool:
        results = pool.starmap(copy_remark_lines, [(old_file, all_suffixes, new_pdbs_dir, collect_headers(old_file, additional_lines_prefixes), remove_unwanted_lines) for old_file in old_pdb_files])

    total_modified = 0
    unmodified_files = []
    for result in results:
        for file_path, was_modified, detailed_result in result:
            if was_modified:
                total_modified += 1
                print(detailed_result)
            else:
                unmodified_files.append((file_path, detailed_result))

    # Reporting on file modifications
    print(f"\nTotal modified new PDB files: {total_modified}/{len(new_pdb_files)}")
    if total_modified == 0:
        print("No files were modified.")

    # Reporting on unmodified files
    if unmodified_files:
        print("\nUnmodified files:")
        for file, reason in unmodified_files:
            print(f"{file}")
            print(f"REASON: {reason}")
            print("")

    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds.")
    print(f"Cumulative unique suffixes found across all samples: {sorted(cumulative_suffixes)}\n")
    print(f"Number of cumulative unique suffixes: {len(cumulative_suffixes)}")
    print(f"\nTotal modified new PDB files: {total_modified}/{len(new_pdb_files)}")

def identify_suffixes(old_files, new_files, debug, sample_size=None):
    """
    Identifies suffixes added to new PDB files based on old PDB file names. Optionally uses a random sample of old files.
    
    Parameters:
        old_files (list): List of old PDB file paths.
        new_files (list): List of new PDB file paths.
        debug (bool): Flag to enable debug output.
        sample_size (int): Optional size of random sample of old files to use for suffix identification.
    
    Returns:
        dict: A dictionary where each old file base maps to the list of all unique suffixes found, and
        set: A set of all unique suffixes identified across sampled files.
    """
    if sample_size is not None and sample_size < len(old_files):
        sampled_files = random.sample(old_files, sample_size)
        if debug:
            print(f"\nUsing a random sample of {sample_size} old files for suffix identification.")
    else:
        sampled_files = old_files
    
    start_time = time.time()
    all_suffixes = {}
    cumulative_suffixes = set()
    
    # First, find suffixes based on the sampled or all files
    for old_file in sampled_files:
        old_base = Path(old_file).stem
        suffixes = []
        for new_file in new_files:
            new_base = Path(new_file).stem
            if new_base.startswith(old_base):
                suffix = new_base[len(old_base):]
                suffixes.append(suffix)
                cumulative_suffixes.add(suffix)
        all_suffixes[old_base] = suffixes
        if debug:
            print(f"\nOld file base '{old_base}' mapped with suffixes: {suffixes}")
    
    # Now map every old file base to the cumulative set of suffixes
    for old_file in old_files:
        old_base = Path(old_file).stem
        all_suffixes[old_base] = list(cumulative_suffixes)

    print(f"\nSuffix identification completed in {time.time() - start_time:.2f} seconds.")
    return all_suffixes, cumulative_suffixes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy specific lines from old PDBs to new PDBs.")
    parser.add_argument('--new_pdb_dir', type=str, required=True, help='Directory with new PDB files.')
    parser.add_argument('--reference_old_pdb_dir', type=str, required=True, help='Directory with reference old PDB files.')
    parser.add_argument('--debug', action='store_true', help='Enable verbose debugging output.')
    parser.add_argument('--remove_unwanted_lines_from_new_pdbs', nargs='*', 
                        default=["REMARK AtomGroup Unnamed + Unnamed"], 
                        help='List of unwanted lines to remove from new PDB files. eg: "BAD LINE" "TERRIBLE LINE" \
                              Defaults to removing "REMARK AtomGroup Unnamed + Unnamed".')
    parser.add_argument('--additional_lines_from_old_pdbs_to_copy', nargs='*', default=[], 
                        help='Additional line prefixes from old PDBs to copy to new PDB headers. eg: "GOOD LINE" "GREAT LINE"')
    parser.add_argument('--find_suffixes_from_random_sample', type=int, 
                        help='Optionally find suffixes from a random sample of old PDB files. Provide an integer to specify sample size.')
    args = parser.parse_args()

    add_remark666_lines_to_pdb_files(args.new_pdb_dir, args.reference_old_pdb_dir, args.debug, 
                                     args.remove_unwanted_lines_from_new_pdbs, args.additional_lines_from_old_pdbs_to_copy,
                                     sample_size=args.find_suffixes_from_random_sample)
