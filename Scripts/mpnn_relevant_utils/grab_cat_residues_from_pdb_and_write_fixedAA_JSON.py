#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Seth M. Woodbury
Date: 2025-01-20

Extract catalytic (or any special) residues from PDB files by parsing REMARK 666
lines containing "MATCH TEMPLATE" and "MOTIF". Build a single JSON dictionary.

DEFAULT (new / recommended output format):
{
    "/path/to/pdb1.pdb": ["A106", "A104", "A26", "A108"],
    "/path/to/pdb2.pdb": ["A59", "A43", "A66", "A62"],
    ...
}

LEGACY (optional) output format (enabled with --legacy_string_values):
{
    "/path/to/pdb1.pdb": "A106 A104 A26 A108",
    "/path/to/pdb2.pdb": "A59 A43 A66 A62",
    ...
}
"""

import os
import glob
import json
import logging
import traceback
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_pdb_for_remark666(file_path):
    """
    Parse a single PDB file to find lines that start with 'REMARK 666 MATCH TEMPLATE'
    and contain the keyword 'MOTIF'. Extract the chain letter, amino acid code, and
    residue number from these lines.

    Returns:
        tuple(str, list[str]):
            A tuple (pdb_file_path, relevant_residues_list) where relevant_residues_list
            is a list of 'chainLetterResidueIndex' strings, e.g. ['A106', 'A104'].
    """
    relevant_residues = []
    try:
        with open(file_path, "r") as file:
            for line in file:
                # Check if line matches our signature
                if line.startswith("REMARK 666 MATCH TEMPLATE") and "MOTIF" in line:
                    parts = line.split()
                    # We expect the sequence: "REMARK 666 MATCH TEMPLATE ... MOTIF <chain> <AA> <resNum>"
                    if "MOTIF" in parts:
                        motif_idx = parts.index("MOTIF")
                        # Defensive bounds checks
                        if motif_idx + 3 < len(parts):
                            chain_letter = parts[motif_idx + 1]     # e.g. 'A'
                            # amino_acid_code = parts[motif_idx + 2] # e.g. 'HIS' (kept for completeness)
                            residue_number = parts[motif_idx + 3]   # e.g. '106'
                            relevant_residues.append(f"{chain_letter}{residue_number}")
    except Exception as exc:
        logging.warning(f"Failed to parse file {file_path}. Error: {exc}")

    return file_path, relevant_residues


def grab_cat_residues_from_pdb_and_write_fixedAA_json(
    input_dir: str,
    output_json: str,
    legacy_string_values: bool = False,
):
    """
    Process all PDB files in the given directory in parallel to find special/catalytic
    residues.

    DEFAULT output:
        { pdb_file_path : ["A106","A104",...], ... }

    LEGACY output (if legacy_string_values=True):
        { pdb_file_path : "A106 A104 ...", ... }

    Args:
        input_dir (str): Directory containing *.pdb files.
        output_json (str): Path to the output JSON file (one dictionary).
        legacy_string_values (bool): If True, write values as a single space-separated
            string (old behavior). If False, write values as a JSON list of tokens
            (recommended).
    """
    try:
        # 1) Find all PDB files
        pdb_files = glob.glob(os.path.join(input_dir, "*.pdb"))
        logging.info(f"Found {len(pdb_files)} PDB files in '{input_dir}' to process.")

        # 2) Use ThreadPoolExecutor to parse files in parallel
        results_dict = {}
        with ThreadPoolExecutor() as executor:
            future_to_file = {executor.submit(parse_pdb_for_remark666, pdb): pdb for pdb in pdb_files}
            for future in as_completed(future_to_file):
                try:
                    pdb_path, residues_list = future.result()

                    # Keep current logic, only change the serialization format:
                    if legacy_string_values:
                        # Old behavior: space-separated string
                        results_dict[pdb_path] = " ".join(residues_list)
                    else:
                        # New default: JSON list of tokens
                        results_dict[pdb_path] = residues_list

                except Exception as exc:
                    logging.error(f"Error processing {future_to_file[future]}: {exc}")
                    logging.error(traceback.format_exc())

        # 3) Write the single dictionary to JSON (no trailing commas)
        with open(output_json, "w") as outfile:
            json.dump(results_dict, outfile, indent=4)
        logging.info(
            "Success: wrote fixed residues JSON to '%s' (format=%s).",
            output_json,
            "legacy_string_values" if legacy_string_values else "list_values_default",
        )
    except Exception:
        logging.error("An error occurred during processing:")
        logging.error(traceback.format_exc())


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Extract catalytic residues from PDB files and write to a single JSON dictionary."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing PDB files to parse.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Path to output JSON file (will contain one dict).",
    )
    parser.add_argument(
        "--legacy_string_values",
        action="store_true",
        help=(
            "Write JSON values as a single space-separated string (OLD behavior). "
            "Default is NEW behavior: values are JSON lists like ['A74','A76',...]."
        ),
    )
    args = parser.parse_args()

    grab_cat_residues_from_pdb_and_write_fixedAA_json(
        args.input_dir,
        args.output_json,
        legacy_string_values=args.legacy_string_values,
    )
