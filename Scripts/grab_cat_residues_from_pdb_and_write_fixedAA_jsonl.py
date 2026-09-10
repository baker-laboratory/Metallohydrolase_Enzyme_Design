"""
@author: Seth Woodbury + ChatGTP
woodbuse@uw.edu
"""
import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor
import logging
from argparse import ArgumentParser
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

def grab_cat_residues_from_pdb_and_write_fixedAA_jsonl(input_dir, output_path):
    try:
        # Find all PDB files in the specified directory
        pdb_files = glob.glob(os.path.join(input_dir, '*.pdb'))
        logging.info(f"Found {len(pdb_files)} PDB files to process.")

        # Use ThreadPoolExecutor to process files in parallel
        results = []
        with ThreadPoolExecutor() as executor:
            future_to_file = {executor.submit(parse_pdb_for_remark666, pdb_file): pdb_file for pdb_file in pdb_files}
            for future in as_completed(future_to_file):
                results.append(future.result())

        # Write the results to the output JSONL file
        write_jsonl(output_path, results)
        logging.info(f"Finished writing to {output_path}")

    except Exception as e:
        logging.error("An error occurred during processing:")
        logging.error(traceback.format_exc())

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_pdb_for_remark666(file_path):
    relevant_residues = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("REMARK 666 MATCH TEMPLATE") and "MOTIF" in line:
                parts = line.split()
                chain_letter = parts[parts.index("MOTIF") + 1]
                amino_acid_code = parts[parts.index("MOTIF") + 2]
                residue_number = parts[parts.index(amino_acid_code) + 1]
                relevant_residues.append(f"{chain_letter}{residue_number}")
    return file_path, relevant_residues


def write_jsonl(output_path, results):
    """
    Write the results to a JSONL file.
    """
    with open(output_path, 'w') as file:
        file.write("{")
        for i, result in enumerate(results):
            values = " ".join(result[1])
            if i == 0:
                file.write(f'\n"{result[0]}": "{values}"')
            else:
                file.write(f',\n"{result[0]}": "{values}"')
        file.write("\n}")

def grab_cat_residues_from_pdb_and_write_fixedAA_jsonl(input_dir, output_path):
    """
    Process PDB files in the given directory to extract data and write to a .jsonl file.
    """
    # Find all PDB files in the specified directory
    pdb_files = glob.glob(os.path.join(input_dir, '*.pdb'))
    logging.info(f"Found {len(pdb_files)} PDB files to process.")

    # Use ThreadPoolExecutor to process files in parallel
    results = []
    with ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(parse_pdb_for_remark666, pdb_file): pdb_file for pdb_file in pdb_files}
        for future in as_completed(future_to_file):
            results.append(future.result())

    # Write the results to the output JSONL file
    write_jsonl(output_path, results)
    logging.info(f"Finished writing to {output_path}")
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = ArgumentParser(description="Extract and write fixed AA from PDB files to JSONL")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing PDB files")
    parser.add_argument('--output_jsonl', type=str, required=True, help="Output JSONL file path")
    args = parser.parse_args()
    grab_cat_residues_from_pdb_and_write_fixedAA_jsonl(args.input_dir, args.output_jsonl)