#!/usr/bin/env python3
"""
build_full_residue_from_tips_INSTALL_MISSING_ATOMS.py

Single entrypoint that performs the three “INSTALL_MISSING_ATOMS” stages:

  1) Combine aligned Rosetta residue PDBs into one file
     (PART1 logic: build_full_residue_from_tips_PART1_INSTALL_MISSING_ATOMS.py)

  2) Merge that combined Rosetta file with the DFT tip-atom PDB by matching
     atoms and appending any unmatched Rosetta atoms
     (PART2 logic: build_full_residue_from_tips_PART2_INSTALL_MISSING_ATOMS.py)

  3) Fix PDB spacing and write a clean final PDB
     (PART3 logic: build_full_residue_from_tips_PART3_INSTALL_MISSING_ATOMS.py)

CLI is compatible with your existing wrapper call, e.g.:

  python build_full_residue_from_tips_INSTALL_MISSING_ATOMS.py \
      -aligned_rosetta_pdbs_for_residues A94_inputpdb_TEMP_aligned_rosetta.pdb A96_inputpdb_TEMP_aligned_rosetta.pdb \
      -dft_tip_atoms_of_residues_only residues_TEMP.pdb \
      -output_pdb final_residue_construction.pdb
"""

import os
import math
import argparse
from pathlib import Path
from collections import defaultdict


# =========================
# PART 1: COMBINE PDB FILES
# =========================

def combine_residues_into_pdb(aligned_rosetta_pdbs, output_file):
    """
    Combine multiple aligned Rosetta PDB files into a single PDB file.

    - Keeps only ATOM/HETATM lines
    - Sorts files by any digits in the filename
    - Renumbers atom serials consecutively across all residues
    """
    # Sort residues by residue number embedded in the file names
    def extract_residue_number(file_name):
        digits = ''.join(c for c in os.path.basename(file_name) if c.isdigit())
        return int(digits) if digits else 0

    sorted_pdbs = sorted(aligned_rosetta_pdbs, key=extract_residue_number)

    atom_counter = 1
    with open(output_file, 'w') as out_f:
        for pdb_file in sorted_pdbs:
            if not os.path.exists(pdb_file):
                print(f"Warning: File {pdb_file} does not exist. Skipping.")
                continue

            with open(pdb_file, 'r') as f:
                for line in f:
                    if line.startswith("ATOM") or line.startswith("HETATM"):
                        # Overwrite atom serial (columns 7–11) with running counter
                        new_line = line[:6] + f"{atom_counter:5d}" + line[11:]
                        out_f.write(new_line)
                        atom_counter += 1

    print(f"Combined PDB written to {output_file}")


# ==========================================
# PART 2: MATCH DFT TIP ATOMS TO ROSETTA PDB
# ==========================================

def parse_pdb(file_path):
    """
    Parse a PDB file and return a dictionary:

        { (chain_id, residue_number): [atom_dict, ...], ... }

    Each atom_dict contains:
        atom_name, element, residue_name, chain_id, residue_number,
        x, y, z, line

    Hydrogens (element == 'H') are excluded.
    """
    residues = defaultdict(list)

    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom = {
                    "atom_name":      line[12:16].strip(),
                    "element":        line[76:78].strip(),
                    "residue_name":   line[17:20].strip(),
                    "chain_id":       line[21].strip(),
                    "residue_number": int(line[22:26].strip()),
                    "x":              float(line[30:38].strip()),
                    "y":              float(line[38:46].strip()),
                    "z":              float(line[46:54].strip()),
                    "line":           line
                }

                # Exclude hydrogens
                if atom["element"] == "H":
                    continue

                key = (atom["chain_id"], atom["residue_number"])
                residues[key].append(atom)

    return residues


def calculate_distance(atom1, atom2):
    """Calculate the Euclidean distance between two atoms."""
    return math.sqrt(
        (atom1["x"] - atom2["x"]) ** 2 +
        (atom1["y"] - atom2["y"]) ** 2 +
        (atom1["z"] - atom2["z"]) ** 2
    )


def match_atoms(residue_dft, residue_rosetta):
    """
    Match atoms in the DFT residue to the closest atoms in the Rosetta residue,
    considering element types.

    Returns:
      matched_atoms: dict { dft_line_str : rosetta_atom_name }
      unmatched_rosetta_atoms: list of rosetta atom dicts that weren't used
    """
    matched_atoms = {}
    used_atom_names = set()

    for atom_dft in residue_dft:
        closest_atom = None
        min_distance = float("inf")

        for atom_ros in residue_rosetta:
            if atom_ros["atom_name"] in used_atom_names:
                continue
            if atom_dft["element"] != atom_ros["element"]:
                continue

            distance = calculate_distance(atom_dft, atom_ros)
            if distance < min_distance:
                min_distance = distance
                closest_atom = atom_ros

        if closest_atom:
            matched_atoms[atom_dft["line"]] = closest_atom["atom_name"]
            used_atom_names.add(closest_atom["atom_name"])

    unmatched = [atom for atom in residue_rosetta if atom["atom_name"] not in used_atom_names]
    return matched_atoms, unmatched


def merge_rosetta_with_dft(
    aligned_rosetta_single_pdb_of_residues: str,
    dft_tip_atoms_of_residues_only: str,
    output_pdb: str
):
    """
    Combine the single-file aligned Rosetta residue PDB with the DFT tip-atom PDB.

    - For each residue key (chain, resnum) present in DFT:
        * Match each DFT atom to closest Rosetta atom of same element
        * Rename DFT atom_name to Rosetta atom_name
        * Append any unmatched Rosetta atoms for that residue
    - If a DFT residue isn't found in Rosetta, its atoms are copied as-is.
    """
    rosetta_residues = parse_pdb(aligned_rosetta_single_pdb_of_residues)
    dft_residues = parse_pdb(dft_tip_atoms_of_residues_only)

    output_lines = []

    for residue_id, dft_atoms in dft_residues.items():
        rosetta_atoms = rosetta_residues.get(residue_id, [])

        if not rosetta_atoms:
            print(f"Warning: Residue {residue_id} in DFT file not found in Rosetta file.")
            output_lines.extend([atom["line"] for atom in dft_atoms])
            continue

        print(f"Processing residue {residue_id}...")
        matched_atoms, unmatched_rosetta_atoms = match_atoms(dft_atoms, rosetta_atoms)

        # Update atom names in DFT residue and write to output
        for atom in dft_atoms:
            updated_line = atom["line"]
            if atom["line"] in matched_atoms:
                atom_name = matched_atoms[atom["line"]]
                updated_line = updated_line[:12] + atom_name.ljust(4) + updated_line[16:]
                print(f"Matched {atom['atom_name']} -> {atom_name}")
            output_lines.append(updated_line)

        # Append unmatched Rosetta atoms
        for atom in unmatched_rosetta_atoms:
            output_lines.append(atom["line"])
            print(f"Appending unmatched atom {atom['atom_name']} to residue {residue_id}.")

    with open(output_pdb, 'w') as out_f:
        out_f.writelines(output_lines)

    print(f"Output written to {output_pdb}.")


# ==========================
# PART 3: FIX PDB SPACING
# ==========================

def fix_pdb_spacing(input_pdb, output_pdb):
    """
    Fixes the spacing of PDB lines in the input file and writes the corrected
    lines to the output file.

    Any non-ATOM/HETATM lines are copied as-is.
    """
    pdb_format = (
        "{:<6}{:>5}  {:<3} {:<3} {:<1}{:>4}    "
        "{:>8.3f}{:>8.3f}{:>8.3f}"
        "{:>6.2f}{:>6.2f}           {:<2} "
    )

    with open(input_pdb, 'r') as infile, open(output_pdb, 'w') as outfile:
        for line in infile:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                record     = line[0:6].strip()
                atom_serial = int(line[6:11].strip())
                atom_name   = line[12:16].strip()
                res_name    = line[17:20].strip()
                chain_id    = line[21:22].strip()
                res_seq     = int(line[22:26].strip())
                x           = float(line[30:38].strip())
                y           = float(line[38:46].strip())
                z           = float(line[46:54].strip())
                occ_str     = line[54:60].strip()
                tf_str      = line[60:66].strip()
                occupancy   = float(occ_str) if occ_str else 1.00
                temp_factor = float(tf_str) if tf_str else 0.00
                element     = line[76:78].strip()

                outfile.write(
                    pdb_format.format(
                        record, atom_serial, atom_name, res_name,
                        chain_id, res_seq, x, y, z, occupancy,
                        temp_factor, element
                    ) + "\n"
                )
            else:
                outfile.write(line)


# ====================
# TOP-LEVEL ORCHESTRY
# ====================

def install_missing_atoms_pipeline(
    aligned_rosetta_pdbs_for_residues,
    dft_tip_atoms_of_residues_only,
    final_output_pdb
):
    """
    Run PART1 -> PART2 -> PART3 in order:

      1) combine_residues_into_pdb(...)
      2) merge_rosetta_with_dft(...)
      3) fix_pdb_spacing(...)
    """
    # Intermediate filenames (same ones used before)
    part1_out = "single_file_aligned_rosetta_pdb_residues_TEMP.pdb"
    part2_out = "SEMI_FINAL_RESIDUE_CONSTRUCTION_TEMP.pdb"

    print("\n### PART1: COMBINING ALIGNED ROSETTA RESIDUES ###")
    combine_residues_into_pdb(aligned_rosetta_pdbs_for_residues, part1_out)

    print("\n### PART2: MERGING WITH DFT TIP-ATOM PDB ###")
    merge_rosetta_with_dft(part1_out, dft_tip_atoms_of_residues_only, part2_out)

    print("\n### PART3: FIXING PDB SPACING ###")
    fix_pdb_spacing(part2_out, final_output_pdb)

    print(f"\n### INSTALL_MISSING_ATOMS PIPELINE COMPLETED ###")
    print(f"Final output PDB: {final_output_pdb}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine aligned Rosetta residues with DFT tip atoms and fix PDB formatting."
    )
    parser.add_argument(
        "-aligned_rosetta_pdbs_for_residues",
        nargs="+",
        required=True,
        help="List of *_inputpdb_TEMP_aligned_rosetta.pdb files (one per residue).",
    )
    parser.add_argument(
        "-dft_tip_atoms_of_residues_only",
        required=True,
        help="PDB with DFT-optimized tip atoms for the residues (e.g. residues_TEMP.pdb).",
    )
    parser.add_argument(
        "-output_pdb",
        required=True,
        help="Final output PDB containing reconstructed residues.",
    )

    args = parser.parse_args()

    install_missing_atoms_pipeline(
        args.aligned_rosetta_pdbs_for_residues,
        args.dft_tip_atoms_of_residues_only,
        args.output_pdb,
    )


if __name__ == "__main__":
    main()
