#!/usr/bin/env python3

"""
Script: standardize_GLU_ASP_tip_atom_labeling_based_on_proximity_to_atomOFinterest.py

Description:
    1. Reads a PDB file (--input_pdb).
    2. Finds a specific ligand (--ligand_code) and ligand atom
       (--ligand_atom_for_close_proximity_to_OE2glu_and_OD2asp).
    3. For every ASP/GLU in the PDB:
       - Checks whether OD2/OE2 is closer to the ligand's reference atom than OD1/OE1.
       - If not, swaps OD1 <-> OD2 or OE1 <-> OE2.
    4. Writes out a new PDB (same lines in the same order, including remarks, etc.),
       except for changed ATOM/HETATM lines in ASP/GLU that got swapped.
    5. Logs messages about what it found and what it changed.

Example:
    python standardize_GLU_ASP_tip_atom_labeling_based_on_proximity_to_atomOFinterest.py \
        --input_pdb my_input.pdb \
        --ligand_code SZA \
        --ligand_atom_for_close_proximity_to_OE2glu_and_OD2asp H1
"""

import os
import argparse
import sys


def distance_squared(coord1, coord2):
    """Return the squared distance between two 3D coordinates."""
    return (
        (coord1[0] - coord2[0]) ** 2 +
        (coord1[1] - coord2[1]) ** 2 +
        (coord1[2] - coord2[2]) ** 2
    )


def _replace_atom_name(pdb_line, new_name):
    """
    Replace the atom name (cols 13–16) in an ATOM/HETATM line with new_name,
    preserving column alignment by padding to width 4.
    """
    left = pdb_line[:13]
    right = pdb_line[17:]
    new_name_padded = new_name.ljust(4)
    return left + new_name_padded + right


def swap_atom_names_in_lines(atom_records, chain, resnum, atom1, atom2):
    """
    Swap atom1 and atom2 names for the given (chain, resnum) in atom_records.

    atom_records: dict[line_index] -> {
        'resname', 'resnum', 'atomname', 'chain', 'coord', 'original_line'
    }
    """
    for idx, record in atom_records.items():
        if record['chain'] == chain and record['resnum'] == resnum:
            if record['atomname'] == atom1:
                new_line = _replace_atom_name(record['original_line'], atom2)
                record['atomname'], record['original_line'] = atom2, new_line
            elif record['atomname'] == atom2:
                new_line = _replace_atom_name(record['original_line'], atom1)
                record['atomname'], record['original_line'] = atom1, new_line


def main():
    parser = argparse.ArgumentParser(
        description="Standardize ASP/GLU tip atom names (OD2/OE2) based on proximity to a given ligand atom."
    )
    parser.add_argument("--input_pdb", required=True, help="Path to the input PDB file.")
    parser.add_argument("--ligand_code", required=True, help="3-letter code for the ligand (e.g. SZA).")
    parser.add_argument(
        "--ligand_atom_for_close_proximity_to_OE2glu_and_OD2asp",
        required=True,
        help="Ligand atom name (e.g. H1) to measure proximity for OE2/OD2."
    )
    args = parser.parse_args()

    pdb_file = args.input_pdb
    ligand_code = args.ligand_code
    ligand_atom_of_interest = args.ligand_atom_for_close_proximity_to_OE2glu_and_OD2asp

    if not os.path.isfile(pdb_file):
        print(f"[ERROR] PDB file '{pdb_file}' does not exist.")
        sys.exit(1)

    # Read all lines so we can write them back in order.
    with open(pdb_file, "r") as f:
        all_lines = f.readlines()

    # line_index -> record for ATOM/HETATM lines
    atom_records = {}
    ligand_atom_coord = None

    # First pass: parse ATOM/HETATM lines
    for i, line in enumerate(all_lines):
        line_type = line[:6].strip()
        if line_type in ("ATOM", "HETATM"):
            atomname = line[12:16].strip()
            resname = line[17:20].strip()
            chain = line[21].strip()
            resnum = line[22:26].strip()
            x_str, y_str, z_str = line[30:38].strip(), line[38:46].strip(), line[46:54].strip()

            try:
                x, y, z = float(x_str), float(y_str), float(z_str)
            except ValueError:
                # Malformed coordinates, skip
                continue

            atom_records[i] = {
                'resname': resname,
                'resnum': resnum,
                'atomname': atomname,
                'chain': chain,
                'coord': (x, y, z),
                'original_line': line,
            }

    # Find the coordinate of the specified ligand atom (first occurrence)
    for i, record in atom_records.items():
        if record['resname'] == ligand_code and record['atomname'] == ligand_atom_of_interest:
            ligand_atom_coord = record['coord']
            print(
                f"### Found ligand {ligand_code} (resnum {record['resnum']}, chain {record['chain']}) "
                f"with atom {ligand_atom_of_interest}. ###"
            )
            break

    if ligand_atom_coord is None:
        print(
            f"### WARNING: Could not find atom '{ligand_atom_of_interest}' in ligand '{ligand_code}' in {pdb_file}."
        )
        print("### Distance checks won't be performed, but we'll still output a 'standardized' file. ###")

    # Build dict of ASP/GLU residues keyed by (chain, resnum, resname) -> {atomname: line_index}
    residues_dict = {}
    for i, record in atom_records.items():
        rname = record['resname']
        if rname in ("ASP", "GLU"):
            key = (record['chain'], record['resnum'], rname)
            residues_dict.setdefault(key, {})
            residues_dict[key][record['atomname']] = i

    found_asp_glu = []

    # For each ASP/GLU, check distances and swap if needed
    for (chain, rnum, rname), atoms_dict in residues_dict.items():
        if rname == "ASP":
            found_asp_glu.append(f"ASP {rnum}")
            od1_idx, od2_idx = atoms_dict.get("OD1"), atoms_dict.get("OD2")
            if od1_idx is not None and od2_idx is not None and ligand_atom_coord is not None:
                od1_coord = atom_records[od1_idx]['coord']
                od2_coord = atom_records[od2_idx]['coord']
                dist_od1_sq = distance_squared(od1_coord, ligand_atom_coord)
                dist_od2_sq = distance_squared(od2_coord, ligand_atom_coord)

                if dist_od2_sq <= dist_od1_sq:
                    print(
                        f"### ASP {rnum} OD2 is already closer to {ligand_code} atom {ligand_atom_of_interest} "
                        "| DOING NOTHING ###"
                    )
                else:
                    print(
                        f"### ASP {rnum} OD1 is closer to {ligand_code} atom {ligand_atom_of_interest} than OD2 "
                        "| INVERTING THE NAMES ###"
                    )
                    swap_atom_names_in_lines(atom_records, chain, rnum, "OD1", "OD2")

        elif rname == "GLU":
            found_asp_glu.append(f"GLU {rnum}")
            oe1_idx, oe2_idx = atoms_dict.get("OE1"), atoms_dict.get("OE2")
            if oe1_idx is not None and oe2_idx is not None and ligand_atom_coord is not None:
                oe1_coord = atom_records[oe1_idx]['coord']
                oe2_coord = atom_records[oe2_idx]['coord']
                dist_oe1_sq = distance_squared(oe1_coord, ligand_atom_coord)
                dist_oe2_sq = distance_squared(oe2_coord, ligand_atom_coord)

                if dist_oe2_sq <= dist_oe1_sq:
                    print(
                        f"### GLU {rnum} OE2 is already closer to {ligand_code} atom {ligand_atom_of_interest} "
                        "| DOING NOTHING ###"
                    )
                else:
                    print(
                        f"### GLU {rnum} OE1 is closer to {ligand_code} atom {ligand_atom_of_interest} than OE2 "
                        "| INVERTING THE NAMES ###"
                    )
                    swap_atom_names_in_lines(atom_records, chain, rnum, "OE1", "OE2")

    if found_asp_glu:
        print("### found", ", ".join(found_asp_glu), "###")

    # Write the output file, preserving all non-ATOM/HETATM lines as-is
    out_pdb_file = os.path.splitext(pdb_file)[0] + ".pdb"
    with open(out_pdb_file, "w") as out_f:
        for i, line in enumerate(all_lines):
            if i in atom_records:
                out_f.write(atom_records[i]['original_line'])
            else:
                out_f.write(line)

    print(f"### Wrote updated PDB to {out_pdb_file} ###")


if __name__ == "__main__":
    main()
