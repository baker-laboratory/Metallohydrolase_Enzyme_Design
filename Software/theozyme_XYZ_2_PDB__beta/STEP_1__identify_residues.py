#!/usr/bin/env python3
"""
Identify amino-acid residues from an input PDB by:
1. Splitting each residue into its own PDB file.
2. Converting each residue PDB to SMILES using Open Babel.
3. Matching each fragment SMILES against a small amino acid library
   using simple atom-count filtering and an alignment-based score.
"""

import os
import re
import argparse
import subprocess
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

# -----------------------------------------------------------------------------
# GLOBAL CONFIGURATION
# -----------------------------------------------------------------------------

DEBUGGING: bool = False

# Path to Open Babel executable
OBABEL_PATH: str = "/home/woodbuse/conda_envs/openbabel_env/bin/obabel"

# Amino acid SMILES and atom-count features
AMINO_ACID_DATA: Dict[str, Dict[str, object]] = {
    "ALA": {"smiles": "C[C@H](N)C=O", "features": {"C": 3, "N": 1, "O": 1}},
    "CYS": {"smiles": "N[C@H](C=O)CS", "features": {"C": 3, "N": 1, "O": 1, "S": 1}},
    "ASP": {"smiles": "N[C@H](C=O)CC(=O)O", "features": {"C": 4, "N": 1, "O": 3}},
    "GLU": {"smiles": "N[C@H](C=O)CCC(=O)O", "features": {"C": 5, "N": 1, "O": 3}},
    "PHE": {"smiles": "N[C@H](C=O)Cc1ccccc1", "features": {"C": 9, "N": 1, "O": 1}},
    "GLY": {"smiles": "NCC=O", "features": {"C": 2, "N": 1, "O": 1}},
    "HIS": {"smiles": "N[C@H](C=O)Cc1c[nH]cn1", "features": {"C": 6, "N": 3, "O": 1}},
    "ILE": {"smiles": "CC[C@H](C)[C@H](N)C=O", "features": {"C": 6, "N": 1, "O": 1}},
    "LYS": {"smiles": "NCCCC[C@H](N)C=O", "features": {"C": 6, "N": 2, "O": 1}},
    "LEU": {"smiles": "CC(C)C[C@H](N)C=O", "features": {"C": 6, "N": 1, "O": 1}},
    "MET": {"smiles": "CSCC[C@H](N)C=O", "features": {"C": 5, "N": 1, "O": 1, "S": 1}},
    "ASN": {"smiles": "NC(=O)C[C@H](N)C=O", "features": {"C": 4, "N": 2, "O": 2}},
    "PRO": {"smiles": "O=C[C@@H]1CCCN1", "features": {"C": 5, "N": 1, "O": 1}},
    "GLN": {"smiles": "NC(=O)CC[C@H](N)C=O", "features": {"C": 5, "N": 2, "O": 2}},
    "ARG": {"smiles": "N=C(N)NCCC[C@H](N)C=O", "features": {"C": 6, "N": 4, "O": 1}},
    "SER": {"smiles": "N[C@H](C=O)CO", "features": {"C": 3, "N": 1, "O": 2}},
    "THR": {"smiles": "C[C@@H](O)[C@H](N)C=O", "features": {"C": 4, "N": 1, "O": 2}},
    "VAL": {"smiles": "CC(C)[C@H](N)C=O", "features": {"C": 5, "N": 1, "O": 1}},
    "TRP": {
        "smiles": "N[C@H](C=O)Cc1c[nH]c2ccccc12",
        "features": {"C": 11, "N": 2, "O": 1},
    },
    "TYR": {"smiles": "N[C@H](C=O)Cc1ccc(O)cc1", "features": {"C": 9, "N": 1, "O": 2}},
}

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS: SMILES ALIGNMENT AND MATCHING
# -----------------------------------------------------------------------------

def align_smiles(fragment: str, candidate: str) -> int:
    """
    Align fragment SMILES with candidate SMILES using a simple scoring algorithm.

    Scoring:
        MATCH_SCORE   = +2 for exact character match
        MISMATCH_SCORE = -1 for character mismatch
        GAP_PENALTY    = -2 for gaps (insert/delete)
    """
    MATCH_SCORE = 2       # Reward for an exact atom match
    MISMATCH_SCORE = -1   # Penalty for atom mismatch
    GAP_PENALTY = -2      # Penalty for alignment gaps

    n = len(fragment)
    m = len(candidate)

    # Initialize scoring matrix
    score_matrix: List[List[int]] = [[0] * (m + 1) for _ in range(n + 1)]

    # Fill scoring matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = score_matrix[i - 1][j - 1] + (
                MATCH_SCORE if fragment[i - 1] == candidate[j - 1] else MISMATCH_SCORE
            )
            delete = score_matrix[i - 1][j] + GAP_PENALTY
            insert = score_matrix[i][j - 1] + GAP_PENALTY
            score_matrix[i][j] = max(match, delete, insert)

    best_score = score_matrix[n][m]

    if DEBUGGING:
        print(
            f"\n--- Alignment Matrix for Fragment: {fragment} "
            f"and Candidate: {candidate} ---"
        )
        for row in score_matrix:
            print(" ".join(f"{val:4}" for val in row))
        print(
            f"\nBest alignment score between '{fragment}' and "
            f"'{candidate}': {best_score}"
        )

    return best_score


def find_best_substructure_match(
    fragment_smiles: str,
    candidate_amino_acids: Dict[str, str],
) -> str:
    """
    Find the best substructure match for a fragment SMILES among candidate amino acids.

    Args:
        fragment_smiles: SMILES string of the fragment.
        candidate_amino_acids: dict mapping amino acid name -> SMILES string.

    Returns:
        The amino acid name with the highest alignment score, or "unknown".
    """
    best_match = "unknown"
    highest_score = float("-inf")

    if DEBUGGING:
        print("\n### STARTING SUBSTRUCTURE MATCHING ###")
        print(f"Fragment SMILES: {fragment_smiles}")
        print(f"Candidate amino acids for matching: {candidate_amino_acids}\n")

    for aa_name, aa_smiles in candidate_amino_acids.items():
        alignment_score = align_smiles(fragment_smiles, aa_smiles)

        if DEBUGGING:
            print(
                f"\nCalculated alignment score for candidate {aa_name} "
                f"with SMILES '{aa_smiles}': {alignment_score}"
            )

        if alignment_score > highest_score:
            highest_score = alignment_score
            best_match = aa_name

    if DEBUGGING:
        print("\n### MATCHING SUMMARY ###")
        print(
            f"Best match for fragment SMILES '{fragment_smiles}' is "
            f"'{best_match}' with a score of {highest_score}"
        )

    return best_match

# -----------------------------------------------------------------------------
# PDB PARSING AND RESIDUE WRITING
# -----------------------------------------------------------------------------

def parse_pdb(input_pdb: str) -> Dict[int, List[str]]:
    """
    Parse an input PDB file into a dictionary of residues.

    The residues are indexed sequentially based on changes in the residue
    number field (columns 23-26 in PDB format).

    Returns:
        Dict[int, List[str]] mapping residue_index -> list of lines.
    """
    residues: Dict[int, List[str]] = {}
    current_residue: Optional[int] = None
    current_residue_lines: List[str] = []
    residue_index = 1

    with open(input_pdb, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                residue_num = int(line[22:26].strip())

                if residue_num != current_residue:
                    if current_residue is not None:
                        residues[residue_index] = current_residue_lines
                        residue_index += 1

                    current_residue = residue_num
                    current_residue_lines = []

                current_residue_lines.append(line)

        if current_residue_lines:
            residues[residue_index] = current_residue_lines

    return residues


def write_residue_pdbs(residues: Dict[int, List[str]]) -> None:
    """
    Write each residue to its own PDB file and convert it to SMILES with Open Babel.

    Output files:
        pdb{index}_TEMP.pdb
        pdb{index}_TEMP_smiles.smi
    """
    print("### RESIDUE FILE PROCESSING ###")

    for index, lines in residues.items():
        pdb_filename = f"pdb{index}_TEMP.pdb"
        smiles_filename = f"pdb{index}_TEMP_smiles.smi"

        # Write PDB for this residue
        with open(pdb_filename, "w") as f:
            f.writelines(lines)
        print(f"Created {pdb_filename}")

        # Convert PDB to SMILES
        try:
            subprocess.run(
                [OBABEL_PATH, "-ipdb", pdb_filename, "-ocan", "-O", smiles_filename],
                check=True,
            )
            print(f"Converted {pdb_filename} to SMILES as {smiles_filename}")
        except subprocess.CalledProcessError as e:
            print(f"Error converting {pdb_filename} to SMILES: {e}")

# -----------------------------------------------------------------------------
# SMILES ATOM COUNTING AND RESIDUE MATCHING
# -----------------------------------------------------------------------------

def count_atoms(smiles: str) -> Dict[str, int]:
    """
    Count occurrences of C, N, O, and S atoms in a SMILES string (case-insensitive).
    """
    atom_counts = Counter(char.upper() for char in smiles if char.upper() in "CNOS")
    return dict(atom_counts)


def match_residue_smiles(
    tip_atom_residues_3letter: Optional[List[str]] = None,
) -> None:
    """
    Match fragment SMILES files to amino acids based on atom counts and SMILES alignment.

    Reads all files matching 'pdb*_TEMP_smiles.smi' in the current directory, applies
    coarse atom-count filtering, then uses alignment scoring to pick the best match.

    Writes:
        residue_map.txt with lines "pdb{index}_TEMP_smiles = AA"
    """
    residue_map: Dict[str, str] = {}

    # Normalize optional 3-letter input list to uppercase for case-insensitive matching
    if tip_atom_residues_3letter is not None:
        tip_atom_residues_3letter = [aa.upper() for aa in tip_atom_residues_3letter]

    print("### RESIDUE MATCHING ###")

    # Collect all indices that actually have SMILES files (obvious fix from len(os.listdir()))
    indices: List[int] = []
    for path in Path(".").glob("pdb*_TEMP_smiles.smi"):
        match = re.search(r"pdb(\d+)_TEMP_smiles\.smi", path.name)
        if match:
            indices.append(int(match.group(1)))

    for index in sorted(indices):
        smi_filename = f"pdb{index}_TEMP_smiles.smi"
        if not os.path.exists(smi_filename):
            continue

        with open(smi_filename, "r") as f:
            # Take first token as SMILES
            first_line = f.readline()
            if not first_line:
                print(f"Warning: empty SMILES file {smi_filename}")
                continue
            smiles = first_line.split()[0].strip()

        fragment_atom_counts = count_atoms(smiles)
        print(f"\nFragment {smi_filename} atom counts: {fragment_atom_counts}")

        candidates: List[str] = []
        candidate_smiles: Dict[str, str] = {}

        # Coarse filter by atom counts and (optionally) tip_atom_residues_3letter
        for aa, data in AMINO_ACID_DATA.items():
            aa_atom_counts: Dict[str, int] = data["features"]
            if all(
                fragment_atom_counts.get(atom, 0) <= aa_atom_counts.get(atom, 0)
                for atom in fragment_atom_counts
            ):
                if tip_atom_residues_3letter is None or aa in tip_atom_residues_3letter:
                    candidates.append(aa)
                    candidate_smiles[aa] = data["smiles"]

        print(f"Candidates for {smi_filename} after filtering: {candidates}")

        if candidates:
            if DEBUGGING:
                print("\n### Running inline substructure matching ###")
            best_match = find_best_substructure_match(smiles, candidate_smiles)
        else:
            best_match = "unknown"

        residue_key = f"pdb{index}_TEMP_smiles"
        residue_map[residue_key] = best_match
        print(f"Matched {smi_filename} to {best_match}")

    # Write residue mapping to file
    with open("residue_map.txt", "w") as f:
        for pdb_name, aa in residue_map.items():
            f.write(f"{pdb_name} = {aa}\n")

    print("\n### RESIDUE MAP CREATED ###")

# -----------------------------------------------------------------------------
# CLI ENTRY POINT
# -----------------------------------------------------------------------------

def main() -> None:
    """
    Command-line interface:
        1. Parse input PDB.
        2. Write per-residue PDBs and SMILES.
        3. Match residue SMILES to amino acids.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Parse an input PDB file, create separate PDB files for each residue, "
            "convert them to SMILES, and match against amino acid SMILES."
        )
    )
    parser.add_argument(
        "-input_pdb",
        required=True,
        help="Input PDB file",
    )
    parser.add_argument(
        "-tip_atom_residues_3letter",
        nargs="+",
        help="Optional 3-letter codes for residues with specified side chains",
    )

    args = parser.parse_args()

    print("### STARTING RESIDUE IDENTIFICATION ###")
    residues = parse_pdb(args.input_pdb)
    write_residue_pdbs(residues)
    match_residue_smiles(args.tip_atom_residues_3letter)
    print("### RESIDUE IDENTIFICATION COMPLETE ###")


if __name__ == "__main__":
    main()
