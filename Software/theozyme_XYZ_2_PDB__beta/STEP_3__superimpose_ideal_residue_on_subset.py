"""
Superimpose and align residues between an input PDB and a Rosetta-generated PDB
using exhaustive atom-pair search and the Kabsch algorithm.

Features:
- Atom parsing from PDB.
- Element-wise exhaustive pairing search (N, C, O, S).
- Optional backbone omission modes.
- Optionally logs all DEBUG output to a file.
- Optional histidine debug mode to test a specific expected pairing.
"""

import os
import sys
import copy
import argparse
from math import factorial
from contextlib import redirect_stdout, nullcontext
from itertools import combinations, permutations, product

import numpy as np

### FUNCTION TO PARSE PDB FILES ###
def parse_pdb(pdb_file):
    """Parse the PDB file and extract atom data."""
    atoms = []
    with open(pdb_file, 'r') as file:
        for line in file:
            if line.startswith("ATOM"):
                atom_index = int(line[6:11].strip())
                atom_name = line[12:16].strip()
                atom_type = line[76:78].strip()
                coords = list(map(float, [line[30:38], line[38:46], line[46:54]]))
                atoms.append({"index": atom_index, "name": atom_name, "type": atom_type, "coords": coords})
    return atoms

### KABSCH ALGORITHM ###
def kabsch(P, Q):
    """Align using the Kabsch algorithm."""
    P, Q = np.array(P), np.array(Q)
    centroid_P = P.mean(axis=0)
    centroid_Q = Q.mean(axis=0)
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    H = P_centered.T @ Q_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    return R, centroid_P, centroid_Q

### PROXIMITY CHECK ###
def is_reasonable_pairing(input_atoms, rosetta_atoms, pairing, threshold=57.5): # DEBUG : TRY CHANGING THRESHOLD IF NO SOLUTIONS... NOT SURE WHY THIS IS BROKEN
    """Check if the pairing is reasonable based on distance threshold."""
    for input_atom, rosetta_atom in pairing:
        distance = np.linalg.norm(np.array(input_atom["coords"]) - np.array(rosetta_atom["coords"]))
        if distance > threshold:
            return False
    return True

### FINDING BEST PAIRINGS ###
def find_best_pairing(input_atoms, rosetta_atoms, expected_pairing_names=None):
    """Find the best pairing between input and Rosetta atoms."""
    best_rmsd = float('inf')
    best_pairing = None
    best_transformation = None
    best_transformed_atom = None
    permutations_searched = 0
    expected_permutation_index = None
    expected_transformation = None

    # Categorize atoms by type
    atom_types = ["N", "C", "O", "S"]
    input_atoms_by_type = {atom_type: [] for atom_type in atom_types}
    rosetta_atoms_by_type = {atom_type: [] for atom_type in atom_types}

    for atom in input_atoms:
        if atom["type"] in atom_types:
            input_atoms_by_type[atom["type"]].append(atom)

    for atom in rosetta_atoms:
        if atom["type"] in atom_types:
            rosetta_atoms_by_type[atom["type"]].append(atom)

    # Calculate total combinations
    total_combinations = 1
    for atom_type in atom_types:
        num_input = len(input_atoms_by_type[atom_type])
        num_rosetta = len(rosetta_atoms_by_type[atom_type])
        if num_input > num_rosetta:
            print(f"ERROR: Not enough {atom_type} atoms in Rosetta PDB to match input PDB.")
            sys.exit(1)
        total_combinations *= factorial(num_rosetta) // factorial(num_rosetta - num_input)

    print(f"DEBUG: Total pairing space: {total_combinations} combinations.")

    # Generate all possible pairings
    all_pairings = []
    for atom_type in atom_types:
        input_atoms_list = input_atoms_by_type[atom_type]
        rosetta_atoms_list = rosetta_atoms_by_type[atom_type]
        num_input_atoms = len(input_atoms_list)
        atom_pairings = []

        if num_input_atoms == 0:
            atom_pairings.append([])
            continue

        rosetta_combinations = combinations(rosetta_atoms_list, num_input_atoms)

        for rosetta_combo in rosetta_combinations:
            for perm in permutations(rosetta_combo):
                atom_pairings.append(list(zip(input_atoms_list, perm)))
        all_pairings.append(atom_pairings)
    print(f"DEBUG: Total pairings sampled across all atom types: {sum(len(pairings) for pairings in all_pairings)}")

    # Generate all combinations across atom types
    permutation_index = 0
    for pairing_combination in product(*all_pairings):
        permutations_searched += 1
        pairing = [pair for sublist in pairing_combination for pair in sublist]

        if not is_reasonable_pairing(input_atoms, rosetta_atoms, pairing):
            print(f"DEBUG: Skipping unreasonable pairing {[(p['name'], q['name']) for p, q in pairing]}")
            continue

        # Extract P and Q
        P = [input_atom["coords"] for input_atom, _ in pairing]
        Q = [rosetta_atom["coords"] for _, rosetta_atom in pairing]

        # Compute RMSD
        R, centroid_P, centroid_Q = kabsch(P, Q)
        transformed_atoms = apply_transformation(rosetta_atoms, R, centroid_P, centroid_Q)
        transformed_Q = []
        for _, rosetta_atom in pairing:
            for transformed_atom in transformed_atoms:
                if transformed_atom["name"] == rosetta_atom["name"]:
                    transformed_Q.append(transformed_atom["coords"])
        rmsd = np.sqrt(np.mean(np.sum((np.array(P) - np.array(transformed_Q))**2, axis=1)))
        print(f"DEBUG: Evaluating pairing {[(p['name'], q['name']) for p, q in pairing]}: RMSD = {rmsd}")

        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_pairing = [(input_atom["name"], rosetta_atom["name"]) for input_atom, rosetta_atom in pairing]
            best_transformation = (R, centroid_P, centroid_Q)
            best_transformed_atom = apply_transformation(rosetta_atoms, R, centroid_P, centroid_Q)

        # Check for expected pairing
        if expected_pairing_names:
            current_pairing_names = [(input_atom["name"], rosetta_atom["name"]) for input_atom, rosetta_atom in pairing]
            if set(current_pairing_names) == set(expected_pairing_names):
                expected_permutation_index = permutations_searched
                expected_transformation = (R, centroid_P, centroid_Q)
                print(f"DEBUG: Expected pairing found at permutation {permutations_searched} with RMSD {rmsd}")

    print(f"\nTotal permutations searched: {permutations_searched}")
    if not best_pairing:
        print("ERROR: No valid pairing found.")
        sys.exit(1)

    return best_rmsd, best_pairing, best_transformation, best_transformed_atom, permutations_searched, expected_permutation_index, expected_transformation

### APPLY TRANSFORMATION ###
def apply_transformation(rosetta_atoms, R, centroid_P, centroid_Q):
    """Apply Kabsch transformation to Rosetta atoms."""
    transformed_atoms = []
    for atom in rosetta_atoms:
        coords = np.array(atom["coords"])
        transformed_coords = (coords - centroid_Q) @ R + centroid_P
        transformed_atom = copy.deepcopy(atom)
        transformed_atom["coords"] = transformed_coords.tolist()
        transformed_atoms.append(transformed_atom)
    return transformed_atoms

### WRITE TO PDB ###
def write_pdb(atoms, output_file):
    """Write atoms to a PDB file."""
    with open(output_file, 'w') as file:
        for atom in atoms:
            file.write(
                f"ATOM  {atom['index']:>5} {atom['name']:<4} HIS A   1"
                f"   {atom['coords'][0]:8.3f}{atom['coords'][1]:8.3f}{atom['coords'][2]:8.3f}"
                f"  1.00  0.00           {atom['type']}\n"
            )

### MAIN EXECUTION ###
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Superimpose and align residues.")
    parser.add_argument("-i", "--input", required=True, help="Input PDB file")
    parser.add_argument("-r", "--rosetta", required=True, help="Rosetta PDB file")
    parser.add_argument("-o", "--output", required=True, help="Output PDB file")
    parser.add_argument("-test_histidine_debug", action='store_true', help="Test specific histidine pairing")
    parser.add_argument("-make_permutation_file", action='store_true', help="Save permutations debug log to a file")
    parser.add_argument("--omit_all_backbone", action="store_true",help="Skip all four backbone atoms (N, CA, C, O) from matching")
    parser.add_argument("--omit_backbone_but_keep_CA", action="store_true", help="Skip backbone atoms N, C, O but keep CA in the matching")
    args = parser.parse_args()

    # Prepare log file if -make_permutation_file is set
    input_basename = os.path.splitext(os.path.basename(args.input))[0]
    permutation_log_file = None
    if args.make_permutation_file:
        permutation_log_file = f"{input_basename}_permutation_log.txt"

    # Redirect output to file if logging
    with open(permutation_log_file, 'w') if permutation_log_file else nullcontext() as f:
        with redirect_stdout(f) if f else nullcontext():

            # 1) parse once, keep full list and a separate match list
            input_atoms         = parse_pdb(args.input)
            all_rosetta_atoms   = parse_pdb(args.rosetta)
            rosetta_atoms_match = all_rosetta_atoms[:]   # this one we’ll filter for matching

            # —— omit‐backbone filtering on the match list ——
            if args.omit_all_backbone or args.omit_backbone_but_keep_CA:
                if args.omit_all_backbone:
                    drop_names = {"N", "CA", "C", "O"}
                else:  # omit_backbone_but_keep_CA
                    drop_names = {"N", "C", "O"}

                # filter only the match copy
                rosetta_atoms_match = [
                    atom for atom in rosetta_atoms_match
                    if atom["name"] not in drop_names
                ]
            # —— end omit‐backbone filtering ——


            # Debugging output for parsed atoms
            print("DEBUG: Parsed input PDB atoms:")
            for atom in input_atoms:
                print(f"  {atom['name']} ({atom['type']}): {atom['coords']}")
            print("DEBUG: Parsed Rosetta PDB atoms:")
            for atom in rosetta_atoms_match:
                print(f"  {atom['name']} ({atom['type']}): {atom['coords']}")

            # Define expected pairing if debug flag is set
            expected_pairing_names = None
            if args.test_histidine_debug:
                expected_pairing = [
                    ("N1", "NE2"),
                    ("N2", "ND1"),
                    ("C1", "CE1"),
                    ("C2", "CD2"),
                    ("C3", "CG"),
                    ("C4", "CB")
                ]
                expected_pairing_names = expected_pairing
            # Find the best pairing and perform transformation
            best_rmsd, best_pairing, (R, centroid_P, centroid_Q), _, permutations_searched, expected_permutation_index, expected_transformation = \
                find_best_pairing(input_atoms, rosetta_atoms_match, expected_pairing_names)

            for input_name, rosetta_name in best_pairing:
                print(f"  {input_name} -> {rosetta_name}")

            # Generate output filenames based on input PDB basename
            output_pdb = f"{input_basename}_{args.output}"

            # Apply the transformation and write the output
            #write_pdb(best_transformed_atom, output_pdb)
            # apply the found transform to all atoms (backbone + sidechains)
            full_transformed_atoms = apply_transformation(all_rosetta_atoms, R, centroid_P, centroid_Q)
            write_pdb(full_transformed_atoms, output_pdb)


            # If test_histidine_debug flag is set, compute RMSD for specific pairing
            if args.test_histidine_debug:
                print("\nTEST HISTIDINE DEBUG MODE ACTIVATED")
                # Extract atoms based on the expected pairing
                P = []
                Q = []
                for input_name, rosetta_name in expected_pairing:
                    input_atom = next(
                        (atom for atom in input_atoms if atom["name"] == input_name),
                        None
                    )
                    rosetta_atom = next(
                        (atom for atom in rosetta_atoms_match if atom["name"] == rosetta_name),
                        None
                    )
                    # only append if both atoms exist
                    if input_atom and rosetta_atom:
                        P.append(input_atom["coords"])
                        Q.append(rosetta_atom["coords"])
                    else:
                        print(f"ERROR: Atom {input_name} or {rosetta_name} not found.")
                        sys.exit(1)

                # Compute RMSD for the expected pairing
                R_expected, centroid_P_exp, centroid_Q_exp = kabsch(P, Q)

                # Apply transformation based on the expected pairing and write output PDB
                transformed_atoms_expected = apply_transformation(
                    all_rosetta_atoms, R_expected, centroid_P_exp, centroid_Q_exp
                )
                transformed_Q = []
                for _, rosetta_atom in expected_pairing:
                    for transformed_atom in transformed_atoms_expected:
                        if transformed_atom["name"] == rosetta_atom:
                            transformed_Q.append(transformed_atom["coords"])
                rmsd = np.sqrt(np.mean(np.sum((np.array(P) - np.array(transformed_Q))**2, axis=1)))
                print(f"RMSD for the expected pairing: {rmsd}")
                
                output_pdb_expected = f"{input_basename}_expected_{args.output}"
                write_pdb(transformed_atoms_expected, output_pdb_expected)
                print(f"Aligned structure for expected pairing written to {output_pdb_expected}")

                # Print the expected pairing
                print("Expected Pairing:")
                for input_name, rosetta_name in expected_pairing:
                    print(f"  {input_name} -> {rosetta_name}")

                # Report whether the expected pairing was evaluated
                if expected_permutation_index:
                    print(f"\nThe expected pairing was evaluated at permutation {expected_permutation_index}/{permutations_searched}.")
                else:
                    print("\nWARNING: The expected pairing was not evaluated during the permutation search.")
