#!/usr/bin/env python

"""
Build and superimpose standard residue PDBs onto residues extracted from an input PDB.

Workflow:
    1. Read residue identities from the input PDB.
    2. Generate standard residue PDBs using PyRosetta (optionally tweaking chi angles).
    3. Extract the corresponding residues from the input PDB to separate PDB files.
    4. Superimpose standard residues onto the extracted residues via Kabsch-based alignment.
"""

import os
import argparse
import itertools
from typing import Dict, List, Tuple, Any

import numpy as np

from pyrosetta import init
from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.rosetta.core.conformation import ResidueFactory

# -----------------------------------------------------------------------------
# PyRosetta initialization
# -----------------------------------------------------------------------------

# Initialize PyRosetta once at import
init("-ignore_unrecognized_res true -load_PDB_components false")

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def infer_element_from_pdb_line(line: str) -> str:
    """
    Infer the element symbol from a PDB line.

    Priority:
        1. Columns 77–78 (PDB element field).
        2. Inferred from atom name (first alphabetic character, with special handling of 'H*').

    Returns:
        Uppercase element symbol (e.g., 'C', 'N', 'O', 'H').
    """
    element = line[76:78].strip()
    if element:
        return element.upper()

    atom_name = line[12:16].strip()
    element = "".join(filter(str.isalpha, atom_name)).strip()

    if not element:
        return ""

    # Handle cases like 'HG' where H* should map to hydrogen,
    # versus other multi-letter element symbols.
    if len(element) > 1 and element[0].upper() == "H":
        return "H"

    return element[0].upper()


# -----------------------------------------------------------------------------
# PDB residue reading and generation
# -----------------------------------------------------------------------------

def read_pdb_residues(pdb_file: str) -> List[Dict[str, Any]]:
    """
    Scan a PDB file and collect unique residues present in ATOM records.

    Returns:
        List of dicts, each with keys: resname, chain, resnum, id.
    """
    residues_info: List[Dict[str, Any]] = []

    with open(pdb_file, "r") as f:
        for line in f:
            if line.startswith("ATOM"):
                res_name = line[17:20].strip()
                chain_id = line[21]
                res_num = int(line[22:26])
                residue_id: Tuple[str, int] = (chain_id, res_num)

                # Avoid duplicates
                if not any(res_info["id"] == residue_id for res_info in residues_info):
                    residues_info.append(
                        {
                            "resname": res_name,
                            "chain": chain_id,
                            "resnum": res_num,
                            "id": residue_id,
                        }
                    )

    return residues_info


def generate_residue_pdbs(residues_info: List[Dict[str, Any]]) -> None:
    """
    For each residue in residues_info, create a standard residue PDB using PyRosetta.

    - Optionally modifies chi angles for certain residue types to get "diverse"
      conformations that are easier to map.
    - Writes {resname}{resnum}_rosetta_TEMP.pdb and then post-processes it via modify_pdb_file.
    """
    residue_type_set = Pose().residue_type_set_for_pose()

    for res_info in residues_info:
        res_name = res_info["resname"]
        chain_id = res_info["chain"]
        res_num = res_info["resnum"]

        try:
            res_type = residue_type_set.name_map(res_name)
        except Exception:
            print(
                f"Unknown residue type '{res_name}'. "
                f"Skipping residue {res_num}{chain_id}."
            )
            continue

        # Create a residue and a new pose
        residue = ResidueFactory.create_residue(res_type)
        pose = Pose()
        pose.append_residue_by_jump(residue, 1)

        # ---------------------------------------------------------------------
        # OPTIONAL GEOMETRY MODIFICATIONS
        # Some standard residue geometries are not ideal for mapping, so we
        # tweak chi angles to get more diverse conformations.
        #
        # Copy the patterns below if you need to add / adjust more residue types.
        # ---------------------------------------------------------------------

        # Rotate chi angles if the residue is HIS
        if res_name in ["HIS"]:
            current_chi1 = pose.chi(1, 1)
            pose.set_chi(1, 1, current_chi1 - 120.0)
            current_chi2 = pose.chi(2, 1)
            pose.set_chi(2, 1, current_chi2 - 120.0)
            print("##### !!!!!!!!!!!!!!!!!!!!!!! #####")
            print(
                "##### STANDARD RESIDUE MODIFIED "
                "IN build_full_residue_from_tips.py #####"
            )

        # Rotate chi angles if the residue is ASP or GLU
        if res_name in ["ASP", "GLU"]:
            current_chi1 = pose.chi(1, 1)
            pose.set_chi(1, 1, current_chi1 - 90.0)
            current_chi2 = pose.chi(2, 1)
            pose.set_chi(2, 1, current_chi2 + 180.0)
            print("##### !!!!!!!!!!!!!!!!!!!!!!! #####")
            print(
                "##### STANDARD RESIDUE MODIFIED "
                "IN build_full_residue_from_tips.py #####"
            )

        # Rotate chi angles if the residue is TYR
        if res_name in ["TYR"]:
            current_chi1 = pose.chi(1, 1)
            pose.set_chi(1, 1, current_chi1 - 120.0)
            current_chi2 = pose.chi(2, 1)
            pose.set_chi(2, 1, current_chi2 + 120.0)
            print("##### !!!!!!!!!!!!!!!!!!!!!!! #####")
            print(
                "##### STANDARD RESIDUE MODIFIED "
                "IN build_full_residue_from_tips.py #####"
            )

        # Rotate chi angles if the residue is LYS
        if res_name in ["LYS"]:
            current_chi1 = pose.chi(1, 1)
            pose.set_chi(1, 1, current_chi1 - 180.0)
            current_chi2 = pose.chi(2, 1)
            pose.set_chi(2, 1, current_chi2 + 180.0)
            current_chi3 = pose.chi(3, 1)
            pose.set_chi(3, 1, current_chi3 + 180.0)
            current_chi4 = pose.chi(4, 1)
            pose.set_chi(4, 1, current_chi4 + 180.0)
            print("##### !!!!!!!!!!!!!!!!!!!!!!! #####")
            print(
                "##### STANDARD RESIDUE MODIFIED "
                "IN build_full_residue_from_tips.py #####"
            )

        # Define the output filename and write the PDB
        output_filename = f"{res_name}{res_num}_rosetta_TEMP.pdb"
        pose.dump_pdb(output_filename)

        # Post-process: set chain ID, residue number, and remove hydrogens
        modify_pdb_file(output_filename, chain_id, res_num)

        print(f"Generated standard residue PDB file: {output_filename}")


# -----------------------------------------------------------------------------
# PDB extraction and editing
# -----------------------------------------------------------------------------

def extract_residue_from_input_pdb(
    input_pdb: str,
    residues_info: List[Dict[str, Any]],
) -> None:
    """
    Extract each residue from the input PDB into its own PDB file.

    - Renames atoms to element+index (e.g., C1, C2, N1) per residue.
    - Skips hydrogens.
    - Writes {resname}{resnum}_inputpdb_TEMP.pdb for each residue.
    """
    with open(input_pdb, "r") as f:
        lines = f.readlines()

    for res_info in residues_info:
        res_name = res_info["resname"]
        chain_id = res_info["chain"]
        res_num = res_info["resnum"]

        residue_lines: List[str] = []
        atom_counts: Dict[str, int] = {}

        for line in lines:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                line_res_num = int(line[22:26])
                line_chain_id = line[21]

                element = infer_element_from_pdb_line(line)
                if element.upper() == "H":
                    continue  # Skip hydrogens

                if line_chain_id == chain_id and line_res_num == res_num:
                    atom_counts[element] = atom_counts.get(element, 0) + 1
                    temp_atom_name = f"{element}{atom_counts[element]}"

                    # Update the line with the temporary atom name
                    new_line = line[:12] + f"{temp_atom_name:<4}" + line[16:]
                    residue_lines.append(new_line)

        if residue_lines:
            output_filename = f"{res_name}{res_num}_inputpdb_TEMP.pdb"
            with open(output_filename, "w") as f_out:
                f_out.writelines(residue_lines)
            print(f"Extracted residue from input PDB: {output_filename}")
        else:
            print(f"No lines found for residue {res_name}{res_num} in input PDB.")


def modify_pdb_file(pdb_filename: str, chain_id: str, res_num: int) -> None:
    """
    Modify a PDB file in-place:

        - Remove hydrogens.
        - Set the chain ID and residue number for all ATOM/HETATM lines.
    """
    with open(pdb_filename, "r") as f:
        lines = f.readlines()

    with open(pdb_filename, "w") as f_out:
        for line in lines:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                element = infer_element_from_pdb_line(line)
                if element.upper() == "H":
                    continue  # Skip writing hydrogens

                # Modify chain ID (col 22) and residue number (cols 23–26)
                line = line[:21] + chain_id + f"{res_num:>4}" + line[26:]
                f_out.write(line)
            else:
                f_out.write(line)


# -----------------------------------------------------------------------------
# Atom reading, transformation, and coordinate updating
# -----------------------------------------------------------------------------

def read_pdb_atoms(
    pdb_filename: str,
    include_hydrogens: bool = False,
) -> List[Dict[str, Any]]:
    """
    Read ATOM/HETATM records from a PDB file.

    Args:
        pdb_filename: PDB file to read.
        include_hydrogens: If False, hydrogens are skipped.

    Returns:
        List of atom dicts with keys: atom_name, element, coords (np.array), line.
    """
    atoms: List[Dict[str, Any]] = []

    with open(pdb_filename, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_name = line[12:16].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])

                element = infer_element_from_pdb_line(line)

                if not include_hydrogens and element.upper() == "H":
                    continue  # Skip hydrogens

                atom = {
                    "atom_name": atom_name,
                    "element": element.upper(),
                    "coords": np.array([x, y, z]),
                    "line": line,
                }
                atoms.append(atom)

    return atoms


def apply_transformation(
    atoms: List[Dict[str, Any]],
    rotation_matrix: np.ndarray,
    translation_vector: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    Apply a rigid-body transformation to atoms.

    Args:
        atoms: List of atom dicts (with 'coords').
        rotation_matrix: 3x3 rotation matrix.
        translation_vector: 3D translation vector.

    Returns:
        New list of atom dicts with transformed coordinates.
    """
    transformed_atoms: List[Dict[str, Any]] = []

    for atom in atoms:
        original_coords = atom["coords"]
        transformed_coords = np.dot(original_coords, rotation_matrix) + translation_vector

        transformed_atom = atom.copy()
        transformed_atom["coords"] = transformed_coords
        transformed_atoms.append(transformed_atom)

    return transformed_atoms


def update_pdb_coordinates(pdb_filename: str, atoms: List[Dict[str, Any]]) -> None:
    """
    Update coordinates in a PDB file from a list of atom dicts.

    - ATOM/HETATM lines with hydrogen elements are skipped.
    - Non-ATOM/HETATM lines are written unchanged.
    """
    with open(pdb_filename, "r") as f:
        lines = f.readlines()

    atom_index = 0

    with open(pdb_filename, "w") as f_out:
        for line in lines:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                element = infer_element_from_pdb_line(line)
                if element.upper() == "H":
                    continue

                atom = atoms[atom_index]
                x, y, z = atom["coords"]

                # Update the coordinates in the line (cols 31–54)
                new_line = line[:30] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + line[54:]
                f_out.write(new_line)
                atom_index += 1
            else:
                f_out.write(line)


# -----------------------------------------------------------------------------
# Superimposition (Kabsch)
# -----------------------------------------------------------------------------

def calculate_superimposition(
    matched_atoms: List[Tuple[Dict[str, Any], Dict[str, Any]]],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute the optimal rotation and translation (Kabsch algorithm) to align
    standard residue atoms onto input residue atoms.

    Args:
        matched_atoms: List of (standard_atom, input_atom) pairs.

    Returns:
        rotation_matrix (3x3), translation_vector (3,), rmsd.
    """
    # Prepare coordinate arrays
    P = np.array([pair[0]["coords"] for pair in matched_atoms])  # Standard residue
    Q = np.array([pair[1]["coords"] for pair in matched_atoms])  # Input residue

    # Center the coordinates
    P_mean = P.mean(axis=0)
    Q_mean = Q.mean(axis=0)
    P_centered = P - P_mean
    Q_centered = Q - Q_mean

    # Compute covariance matrix
    C = np.dot(P_centered.T, Q_centered)

    # Singular Value Decomposition
    V, S, Wt = np.linalg.svd(C)

    # Compute rotation matrix (handle potential reflection)
    d = np.sign(np.linalg.det(np.dot(Wt.T, V.T)))
    D = np.diag([1, 1, d])
    U = np.dot(np.dot(Wt.T, D), V.T)

    # Compute translation vector
    translation = Q_mean - np.dot(P_mean, U)

    # Compute RMSD
    P_transformed = np.dot(P_centered, U)
    diff = P_transformed - Q_centered
    rmsd = np.sqrt((diff**2).sum() / len(P))

    return U, translation, rmsd


def superimpose_residues(
    residues_info: List[Dict[str, Any]],
    debug: bool = False,
) -> None:
    """
    For each residue:
        - Read standard and input PDBs.
        - Match atoms element-wise by exhaustive combination/permutation.
        - Use the best mapping to compute a transformation and apply it to the standard PDB.
    """
    for res_info in residues_info:
        res_name = res_info["resname"]
        res_num = res_info["resnum"]
        chain_id = res_info["chain"]

        standard_pdb = f"{res_name}{res_num}_rosetta_TEMP.pdb"
        input_pdb = f"{res_name}{res_num}_inputpdb_TEMP.pdb"

        if not os.path.isfile(standard_pdb):
            print(
                f"Standard PDB file {standard_pdb} not found for residue "
                f"{res_name}{res_num}. Skipping."
            )
            continue

        if not os.path.isfile(input_pdb):
            print(
                f"Input PDB file {input_pdb} not found for residue "
                f"{res_name}{res_num}. Skipping."
            )
            continue

        # Read atoms from both PDB files (exclude hydrogens)
        standard_atoms = read_pdb_atoms(standard_pdb, include_hydrogens=False)
        input_atoms = read_pdb_atoms(input_pdb, include_hydrogens=False)

        if len(input_atoms) < 3:
            print(
                f"Not enough atoms in input residue {res_name}{res_num} "
                f"for superimposition."
            )
            continue

        matched_atoms: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

        # Match by element
        for element in {atom["element"] for atom in input_atoms}:
            std_atoms = [atom for atom in standard_atoms if atom["element"] == element]
            inp_atoms = [atom for atom in input_atoms if atom["element"] == element]

            N_s = len(std_atoms)
            N_i = len(inp_atoms)

            if N_s == 0 or N_i == 0:
                continue  # No matching atoms

            if N_s < N_i:
                print(
                    f"Not enough standard atoms for element {element} in "
                    f"residue {res_name}{res_num}. Skipping element."
                )
                continue

            if debug:
                print(f"Element {element}: {N_s} standard atoms, {N_i} input atoms")

            # Generate all possible combinations of standard atoms
            std_combinations = itertools.combinations(std_atoms, N_i)
            best_rmsd = float("inf")
            best_mapping: List[Tuple[Dict[str, Any], Dict[str, Any]]] | None = None

            # For each combination, generate all permutations
            for std_combo in std_combinations:
                permutations = itertools.permutations(std_combo)
                for perm in permutations:
                    pairs = list(zip(perm, inp_atoms))
                    rotation_matrix, translation_vector, rmsd = calculate_superimposition(
                        pairs
                    )

                    if rmsd < best_rmsd:
                        best_rmsd = rmsd
                        best_mapping = pairs

            if best_mapping is not None:
                matched_atoms.extend(best_mapping)
                if debug:
                    print(
                        f"Best RMSD for element {element} in residue "
                        f"{res_name}{res_num}: {best_rmsd:.4f} Å"
                    )
                    for std_atom, inp_atom in best_mapping:
                        print(
                            f"Matched standard atom {std_atom['atom_name']} "
                            f"to input atom {inp_atom['atom_name']}"
                        )
            else:
                print(
                    f"No valid mappings found for element {element} "
                    f"in residue {res_name}{res_num}."
                )

        if len(matched_atoms) < 3:
            print(
                f"Not enough matched atoms for residue {res_name}{res_num}. "
                f"Skipping superimposition."
            )
            continue

        # Perform superimposition
        rotation_matrix, translation_vector, rmsd = calculate_superimposition(
            matched_atoms
        )

        # Apply transformation to standard atoms (excluding hydrogens)
        best_transformed_atoms = apply_transformation(
            standard_atoms,
            rotation_matrix,
            translation_vector,
        )

        # Write updated coordinates back to the standard PDB file
        update_pdb_coordinates(standard_pdb, best_transformed_atoms)

        print(
            f"Superimposed residue {res_name}{res_num} (chain {chain_id}) and "
            f"updated coordinates in {standard_pdb}."
        )
        print(f"RMSD after superimposition: {rmsd:.4f} Å")


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def main(input_pdb: str, debug: bool = False) -> None:
    """
    Main pipeline:
        1. Identify residues in input PDB.
        2. Generate standard residue PDBs with PyRosetta.
        3. Extract matching residues from input PDB.
        4. Superimpose standard residues onto extracted residues.
    """
    residues_info = read_pdb_residues(input_pdb)

    if not residues_info:
        print("No residues found in the input PDB file.")
        return

    generate_residue_pdbs(residues_info)
    extract_residue_from_input_pdb(input_pdb, residues_info)
    superimpose_residues(residues_info, debug=debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate standard residue PDBs, extract residues from an input PDB, "
            "and superimpose them."
        )
    )
    parser.add_argument(
        "-input_pdb",
        required=True,
        help="Input PDB file",
    )
    parser.add_argument(
        "-debug",
        action="store_true",
        help="Enable debug mode",
    )

    args = parser.parse_args()
    main(args.input_pdb, debug=args.debug)
