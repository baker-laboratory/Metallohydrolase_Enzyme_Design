#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UNIFIED SCRIPT:
    Extract ligands from a PDB, generate a fully-bonded MOL2 via Open Babel,
    and call Rosetta's molfile_to_params.py to make params and PDBs.

    This combines the functionality of:
      1) extract_ligands_from_SINGLE_pdb_and_create_PARAMS.py
      2) make_FullyBonded_mol2_file_from_singleXYZ_ThatCanHave_multipleXYZinside.py
      3) mol2_with_confs_to_params.py

NEW FEATURE:
    --preserve_pdb_ligand_atom_order

    When OFF (default):
        Legacy pipeline:
            PDB  --> XYZ (HETATM-only) --> MOL2 (via Open Babel, -ixyz)
               --> fully-bonded MOL2   --> Rosetta molfile_to_params.py

    When ON:
        "PDB-preserve" pipeline:
            PDB (cropped to HETATM selection) --> PDB_ligand_only
               --> MOL2 (via Open Babel, -ipdb)
               --> fully-bonded MOL2
               --> Rosetta molfile_to_params.py

    The PDB-preserve pipeline is designed to keep ligand atom names and
    ordering from the original PDB as much as Open Babel and Rosetta allow,
    while the legacy pipeline behaves exactly as before.

USAGE (example):
    python unified_ligand_params_pipeline.py \
        --input_single_pdb some_structure.pdb \
        --ligands_to_extract_via_3letter_code LIG \
        --output_dir_for_params_stuff /path/to/out \
        --desired_ligand_3letter_code LIG \
        --preserve_pdb_ligand_atom_order

"""

import argparse
import os
import shutil
import sys
import subprocess
from math import sqrt, inf
import re

###############################################################################
# PATH CONSTANTS (EDIT HERE IF NEEDED)
###############################################################################

from pathlib import Path as _Path
for _anc in _Path(__file__).resolve().parents:
    if (_anc / "repo_paths.py").is_file():
        sys.path.insert(0, str(_anc)); break
import repo_paths  # noqa: E402

OPENBABEL_BIN = repo_paths.OBABEL
MOLFILE_TO_PARAMS_SCRIPT = repo_paths.MOLFILE_TO_PARAMS

###############################################################################
# ARGPARSE
###############################################################################

def parse_args():
    p = argparse.ArgumentParser(
        description="Unified ligand → MOL2 → Rosetta params pipeline (legacy XYZ or PDB-preserve mode)."
    )
    # Core inputs
    p.add_argument('--input_single_pdb', required=True,
                   help="Path to input PDB file (will be scanned for HETATM ligands).")
    p.add_argument('--ligands_to_extract_via_3letter_code',
                   nargs='*', default=None,
                   help=("Optional list of 3-letter ligand codes to extract (e.g. ZN1 LIG). "
                         "If omitted, ALL HETATMs are extracted."))
    p.add_argument('--output_dir_for_params_stuff', required=True,
                   help="Directory for XYZ (if used), intermediate MOL2, and final params/PDB files.")
    p.add_argument('--desired_ligand_3letter_code', required=True,
                   help="3-letter code used in naming output MOL2, params, and PDB files.")

    # Pipeline behavior
    p.add_argument('--preserve_pdb_ligand_atom_order', action='store_true',
                   help=("NEW: Use PDB-preserve pipeline (PDB→MOL2) to better retain original ligand "
                         "atom names and order. If not set, legacy PDB→XYZ→MOL2 pipeline is used."))
    p.add_argument('--stop_after_XYZ_is_made', action='store_true',
                   help=("If set, stop after generating XYZ (legacy pipeline only); "
                         "print downstream commands without executing."))
    p.add_argument('--stop_after_MOL2_is_made', action='store_true',
                   help="If set, stop after generating (and bond-fixing) MOL2; do not call Rosetta.")
    p.add_argument('--skip_bond_fix', action='store_true',
                   help="If set, skip the bond-fixing / connectivity step on the MOL2.")

    # Misc
    p.add_argument('--verbose', action='store_true',
                   help="Print extra debug information.")
    return p.parse_args()

###############################################################################
# PDB UTILITIES
###############################################################################

def read_pdb_hetatm_with_lines(pdb_path):
    """
    Read all HETATM lines from the PDB, returning a list of dicts:
      {
        'line': original_line,
        'resname': str,
        'element': str,
        'x': float,
        'y': float,
        'z': float
      }
    """
    het_atoms = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("HETATM"):
                continue
            resname = line[17:20].strip()
            # element: use columns 76-78, fall back to first letter of atom name
            element = line[76:78].strip() or line[12:14].strip()[0]
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            het_atoms.append({
                "line": line.rstrip("\n"),
                "resname": resname,
                "element": element,
                "x": x,
                "y": y,
                "z": z,
            })
    return het_atoms

def write_xyz_from_hetatoms(atoms, xyz_path):
    """
    Legacy behavior: write a simple XYZ file with only element and coordinates.
    """
    with open(xyz_path, 'w') as out:
        out.write(f"{len(atoms)}\n")
        out.write("Extracted ligand XYZ\n")
        for a in atoms:
            out.write(f"{a['element']:2s} {a['x']:12.6f} {a['y']:12.6f} {a['z']:12.6f}\n")

def write_pdb_ligand_only(atoms, pdb_path):
    """
    New behavior: write a PDB containing only the selected HETATM lines,
    preserving the exact original lines and order.
    """
    with open(pdb_path, 'w') as out:
        out.write("REMARK  Generated ligand-only PDB from input_single_pdb\n")
        for a in atoms:
            out.write(a["line"] + "\n")

###############################################################################
# MOL2 PARSING / BOND FIXING (from make_FullyBonded_mol2_file_*.py)
###############################################################################

def standardize_residue_labels(mol2_file, resid=1, resname="UNL1"):
    """
    Safely rewrite subst_id and subst_name in MOL2 ATOM records
    WITHOUT breaking column alignment.
    """
    with open(mol2_file) as f:
        lines = f.readlines()

    out = []
    in_atom = False

    for line in lines:
        if line.startswith("@<TRIPOS>ATOM"):
            in_atom = True
            out.append(line)
            continue
        if line.startswith("@<TRIPOS>"):
            in_atom = False
            out.append(line)
            continue

        if in_atom and line.strip():
            fields = line.split()
            if len(fields) < 9:
                out.append(line)
                continue

            # overwrite only these fields
            fields[6] = str(resid)
            fields[7] = resname

            # reformat safely
            new_line = (
                f"{int(fields[0]):>7d} "
                f"{fields[1]:<8s}"
                f"{float(fields[2]):>10.4f}"
                f"{float(fields[3]):>10.4f}"
                f"{float(fields[4]):>10.4f} "
                f"{fields[5]:<6s} "
                f"{int(fields[6]):>3d} "
                f"{fields[7]:<6s} "
                f"{float(fields[8]):>10.4f}\n"
            )
            out.append(new_line)
        else:
            out.append(line)

    with open(mol2_file, "w") as f:
        f.writelines(out)

def parse_mol2(mol2_file):
    """
    Parse .mol2 file to extract atom and bond data for each molecule.
    Returns a list of molecule dicts:
       [
         {
           "atoms": {atom_id: {"name": ..., "coords": (x,y,z), "element": ..., "bonds": []}, ...},
           "bonds": [(a1, a2), (a1, a2), ...]
         },
         ...
       ]
    """
    molecules = []
    current_molecule = {"atoms": {}, "bonds": []}
    with open(mol2_file, "r") as file:
        in_atom_section = False
        in_bond_section = False

        for line in file:
            if line.startswith("@<TRIPOS>MOLECULE"):
                # Start of a new molecule
                if current_molecule["atoms"]:
                    molecules.append(current_molecule)
                current_molecule = {"atoms": {}, "bonds": []}
                in_atom_section = False
                in_bond_section = False

            elif line.startswith("@<TRIPOS>ATOM"):
                in_atom_section = True
                in_bond_section = False

            elif line.startswith("@<TRIPOS>BOND"):
                in_atom_section = False
                in_bond_section = True

            elif line.startswith("@<TRIPOS>"):
                in_atom_section = False
                in_bond_section = False

            elif in_atom_section:
                parts = line.split()
                if len(parts) >= 6:
                    atom_id = int(parts[0])
                    atom_name = parts[1]
                    x, y, z = map(float, parts[2:5])
                    element = parts[5]
                    current_molecule["atoms"][atom_id] = {
                        "name": atom_name,
                        "coords": (x, y, z),
                        "element": element,
                        "bonds": []
                    }
                else:
                    print(f"Warning: Malformed atom line: {line.strip()}")

            elif in_bond_section:
                parts = line.split()
                if len(parts) >= 3:
                    # The first column is bond ID (ignored), next two are atom IDs
                    try:
                        a1 = int(parts[1])
                        a2 = int(parts[2])
                        if a1 in current_molecule["atoms"] and a2 in current_molecule["atoms"]:
                            current_molecule["bonds"].append((a1, a2))
                            current_molecule["atoms"][a1]["bonds"].append(a2)
                            current_molecule["atoms"][a2]["bonds"].append(a1)
                        else:
                            print(f"Warning: Bond references non-existent atoms: {[a1, a2]}")
                    except ValueError:
                        print(f"Warning: Could not parse bond line: {line.strip()}")
                else:
                    print(f"Warning: Malformed bond line: {line.strip()}")

        # Append last molecule if it has data
        if current_molecule["atoms"]:
            molecules.append(current_molecule)

    return molecules

def find_nearest_heteroatom(target_atom, atoms):
    """
    Find the nearest non-hydrogen atom to the target_atom.
    Return its ID, or None if not found.
    """
    min_dist = float("inf")
    nearest_id = None
    tx, ty, tz = target_atom["coords"]
    for other_id, other_atom in atoms.items():
        if other_atom is target_atom:
            continue
        if other_atom["element"].upper() == "H":
            continue

        ox, oy, oz = other_atom["coords"]
        dist = sqrt((tx - ox)**2 + (ty - oy)**2 + (tz - oz)**2)
        if dist < min_dist:
            min_dist = dist
            nearest_id = other_id

    return nearest_id

def gather_new_bonds(molecules):
    """
    Ensure each atom has at least one bond. If not, bond it to nearest non-H.
    Return a dict of *newly* added bonds for each molecule index (1-based):
      { 1: [(6,2), (a,b)], 2: [], 3: [...], ... }
    """
    new_bonds_dict = {}
    for i, mol in enumerate(molecules, start=1):
        new_bonds = []
        atoms = mol["atoms"]
        for atom_id, atom in atoms.items():
            if not atom["bonds"]:
                print(f"Molecule {i}: Atom {atom_id} ({atom['name']}) has no bonds.")
                nearest_het_id = find_nearest_heteroatom(atom, atoms)
                if nearest_het_id:
                    print(f"  Creating bond between atom {atom_id} and {nearest_het_id}.")
                    atom["bonds"].append(nearest_het_id)
                    atoms[nearest_het_id]["bonds"].append(atom_id)
                    mol["bonds"].append((atom_id, nearest_het_id))
                    new_bonds.append((atom_id, nearest_het_id))
        new_bonds_dict[i] = new_bonds
    return new_bonds_dict

def connect_fragments(molecules, new_bonds_dict):
    """
    For each molecule, if its atoms split into >1 connected component,
    add one bond between the closest pair of non-H atoms in each adjacent component.
    """
    for idx, mol in enumerate(molecules, start=1):
        atoms = mol["atoms"]
        # 1) Build list of components via simple DFS/BFS
        visited = set()
        components = []
        for a0 in atoms:
            if a0 in visited:
                continue
            stack = [a0]
            comp = set()
            while stack:
                a = stack.pop()
                if a in visited:
                    continue
                visited.add(a)
                comp.add(a)
                for nbr in atoms[a]["bonds"]:
                    if nbr not in visited:
                        stack.append(nbr)
            components.append(comp)

        # 2) If more than one, connect them in series
        if len(components) > 1:
            for compA, compB in zip(components, components[1:]):
                best_pair = None
                best_dist2 = inf
                for a in compA:
                    if atoms[a]["element"].upper() == "H":
                        continue
                    x1, y1, z1 = atoms[a]["coords"]
                    for b in compB:
                        if atoms[b]["element"].upper() == "H":
                            continue
                        x2, y2, z2 = atoms[b]["coords"]
                        d2 = (x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2
                        if d2 < best_dist2:
                            best_dist2 = d2
                            best_pair = (a, b)
                if best_pair:
                    a, b = best_pair
                    atoms[a]["bonds"].append(b)
                    atoms[b]["bonds"].append(a)
                    mol["bonds"].append((a, b))
                    new_bonds_dict[idx].append((a, b))
                    print(f"  Connected fragment of mol {idx}: adding bond {a}–{b}")

def partial_update_mol2(mol2_file, new_bonds_dict):
    """
    Read the .mol2 line by line, modifying only:
      - The "atom/bond count" line (2nd line after @<TRIPOS>MOLECULE) if we have new bonds.
      - The BOND section to append new bond lines.

    Write updated result back to the same .mol2 file.
    """
    with open(mol2_file, 'r') as f:
        lines = f.readlines()

    updated_lines = []
    line_index = 0
    total = len(lines)

    molecule_index = 0
    bond_line_pattern = re.compile(r'^\s*(\d+)\s+(\d+)\s+(\d+)\s+(.*)')

    while line_index < total:
        line = lines[line_index]
        stripped = line.strip()

        if stripped.startswith('@<TRIPOS>MOLECULE'):
            molecule_index += 1
            updated_lines.append(line)
            line_index += 1

            if line_index < total:
                mol_name_line = lines[line_index]
                updated_lines.append(mol_name_line)
                line_index += 1
            else:
                break

            if line_index < total:
                mol_count_line = lines[line_index]
                parts = mol_count_line.strip().split()
                if molecule_index in new_bonds_dict and new_bonds_dict[molecule_index]:
                    try:
                        old_bond_count = int(parts[1])
                        add_bonds = len(new_bonds_dict[molecule_index])
                        new_bond_count = old_bond_count + add_bonds
                        parts[1] = str(new_bond_count)
                        updated_mol_count_line = " " + " ".join(parts) + "\n"
                        updated_lines.append(updated_mol_count_line)
                    except (IndexError, ValueError):
                        updated_lines.append(mol_count_line)
                else:
                    updated_lines.append(mol_count_line)

                line_index += 1
            else:
                break

            continue

        elif stripped.startswith('@<TRIPOS>BOND'):
            updated_lines.append(line)
            line_index += 1

            bond_lines = []
            last_bond_id = 0
            while line_index < total:
                peek = lines[line_index]
                if peek.strip().startswith('@<TRIPOS>'):
                    break
                bond_lines.append(peek)
                match = bond_line_pattern.match(peek)
                if match:
                    b_id = int(match.group(1))
                    if b_id > last_bond_id:
                        last_bond_id = b_id
                line_index += 1

            updated_lines.extend(bond_lines)

            if molecule_index in new_bonds_dict:
                for (a1, a2) in new_bonds_dict[molecule_index]:
                    last_bond_id += 1
                    new_bond_line = f" {last_bond_id:>5} {a1:>5} {a2:>5} {1:>4}\n"
                    updated_lines.append(new_bond_line)

            continue

        updated_lines.append(line)
        line_index += 1

    with open(mol2_file, 'w') as f:
        f.writelines(updated_lines)

    print(f"[partial_update_mol2] Wrote updated file => {mol2_file}")

def update_mol2_with_bonds(mol2_file):
    """
    1) Parse .mol2 to get molecules
    2) Identify new bonds (unbonded atoms -> nearest heteroatom)
    3) Ensure single-fragment connectivity per molecule
    4) Partially rewrite .mol2: increment bond count, append new bond lines
    5) Standardize residue labels to `1 UNL1`
    """
    molecules = parse_mol2(mol2_file)
    print(f"Parsed {len(molecules)} molecules from '{mol2_file}'.")
    new_bonds_dict = gather_new_bonds(molecules)
    connect_fragments(molecules, new_bonds_dict)
    partial_update_mol2(mol2_file, new_bonds_dict)
    standardize_residue_labels(mol2_file)

###############################################################################
# MOL2 GENERATION (WRAPPER AROUND OPEN BABEL)
###############################################################################

def generate_mol2_file(input_path, input_format, output_basename, output_dir, verbose=False):
    """
    Use Open Babel to convert input_path to a MOL2 file.
    input_format: "xyz" or "pdb" etc., passed as -i{format} to obabel.
    """
    if output_dir is None:
        output_dir = os.path.dirname(input_path) or "."

    os.makedirs(output_dir, exist_ok=True)
    output_mol2 = os.path.join(output_dir, f"{output_basename}.mol2")

    cmd = [
        OPENBABEL_BIN,
        f"-i{input_format}", input_path,
        "-omol2", "-O", output_mol2
    ]
    print("[generate_mol2_file] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("[generate_mol2_file] Wrote:", output_mol2)
    return output_mol2

###############################################################################
# ROSETTA molfile_to_params WRAPPER
###############################################################################

def run_molfile_to_params(lig_code, input_mol2, verbose=False):
    """
    Execute Rosetta's molfile_to_params.py with the user-provided arguments.
    Then merge {lig_code}.pdb on top of {lig_code}_conformers.pdb.
    (Same logic as mol2_with_confs_to_params.py)
    """
    cmd = [
        "python",
        MOLFILE_TO_PARAMS_SCRIPT,
        "--name", lig_code,
        "--conformers-in-one-file", input_mol2,
        "--root_atom=1",
        "--clobber",
    ]
    print("[mol2_with_confs_to_params] Running command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    main_pdb_path = f"{lig_code}.pdb"
    conf_pdb_path = f"{lig_code}_conformers.pdb"

    if not os.path.isfile(main_pdb_path):
        print(f"WARNING: File {main_pdb_path} not found. Cannot merge.")
        return
    if not os.path.isfile(conf_pdb_path):
        print(f"WARNING: File {conf_pdb_path} not found. Cannot merge.")
        return

    with open(main_pdb_path, "r") as f_main:
        main_lines = f_main.readlines()
    with open(conf_pdb_path, "r") as f_conf:
        conf_lines = f_conf.readlines()

    with open(conf_pdb_path, "w") as f_conf_out:
        f_conf_out.writelines(main_lines)
        f_conf_out.writelines(conf_lines)

    print(f"[mol2_with_confs_to_params] Merged contents of '{main_pdb_path}' on top of '{conf_pdb_path}'.")

###############################################################################
# MAIN PIPELINE
###############################################################################

def main():
    args = parse_args()

    # 1) Read HETATMs from input PDB
    print(f"[INFO] Reading PDB: {args.input_single_pdb}")
    het_atoms = read_pdb_hetatm_with_lines(args.input_single_pdb)
    print(f"[INFO] Found {len(het_atoms)} HETATM entries in PDB")

    # 2) Filter by 3-letter codes if provided
    if args.ligands_to_extract_via_3letter_code:
        desired = set(args.ligands_to_extract_via_3letter_code)
        filtered_atoms = [a for a in het_atoms if a["resname"] in desired]
        print(f"[INFO] Filtering to codes {desired}: {len(filtered_atoms)} atoms remain")
    else:
        filtered_atoms = het_atoms
        print(f"[INFO] No ligand filter codes given, using ALL HETATM atoms")

    if not filtered_atoms:
        print("[ERROR] No HETATM atoms passed the filter; nothing to do.")
        sys.exit(1)

    out_dir = args.output_dir_for_params_stuff
    os.makedirs(out_dir, exist_ok=True)

    # 3) Decide pipeline mode
    if args.preserve_pdb_ligand_atom_order:
        pipeline_mode = "pdb_preserve"
        print("[INFO] Using PDB-preserve pipeline (PDB → MOL2).")
    else:
        pipeline_mode = "xyz_legacy"
        print("[INFO] Using legacy XYZ pipeline (PDB → XYZ → MOL2).")

    # 4) Generate XYZ or ligand-only PDB
    xyz_path = None
    ligand_pdb_path = None

    if pipeline_mode == "xyz_legacy":
        xyz_fname = f"saved_xyz_cropped_ligand_fromSINGLEpdb__lig_{args.desired_ligand_3letter_code}.xyz"
        xyz_path = os.path.join(out_dir, xyz_fname)
        print(f"[INFO] Writing XYZ to: {xyz_path}")
        write_xyz_from_hetatoms(filtered_atoms, xyz_path)

        if args.stop_after_XYZ_is_made:
            # mimic legacy behavior: print downstream commands and exit
            mol2_basename = args.desired_ligand_3letter_code
            mol2_script_cmd = (
                f"python make_FullyBonded_mol2_file_from_singleXYZ_ThatCanHave_multipleXYZinside.py "
                f"--input_xyz_that_may_contain_multipleXYZinside {xyz_path} "
                f"--output_mol2_file_basename {mol2_basename}"
            )
            params_cmd = (
                f"python mol2_with_confs_to_params.py "
                f"--lig_3letter_code {args.desired_ligand_3letter_code} "
                f"--input_mol2_with_confs {mol2_basename}.mol2"
            )
            print("\n[INFO] --stop_after_XYZ_is_made set; not executing downstream steps.")
            print(f"[CMD] {mol2_script_cmd}")
            print(f"[CMD] {params_cmd}")
            sys.exit(0)

    else:  # pdb_preserve pipeline
        ligand_pdb_fname = f"ligand_only_fromSINGLEpdb__lig_{args.desired_ligand_3letter_code}.pdb"
        ligand_pdb_path = os.path.join(out_dir, ligand_pdb_fname)
        print(f"[INFO] Writing ligand-only PDB to: {ligand_pdb_path}")
        write_pdb_ligand_only(filtered_atoms, ligand_pdb_path)

    # 5) Generate MOL2 with Open Babel
    mol2_basename = args.desired_ligand_3letter_code
    if pipeline_mode == "xyz_legacy":
        input_path = xyz_path
        input_format = "xyz"
    else:
        input_path = ligand_pdb_path
        input_format = "pdb"

    print("")
    print("############################################## MOL2 GENERATION ##############################################")
    mol2_path = generate_mol2_file(
        input_path=input_path,
        input_format=input_format,
        output_basename=mol2_basename,
        output_dir=out_dir,
        verbose=args.verbose
    )

    # 6) Fix bonds / connectivity if requested
    if args.skip_bond_fix:
        print("[INFO] --skip_bond_fix set; skipping bond connectivity updates on MOL2.")
    else:
        print("")
        print("############################################## BOND FIX / CONNECTIVITY ##############################################")
        update_mol2_with_bonds(mol2_path)

    if args.stop_after_MOL2_is_made:
        print("\n[INFO] --stop_after_MOL2_is_made set; not running Rosetta molfile_to_params.py.")
        print(f"[INFO] Final MOL2 at: {mol2_path}")
        sys.exit(0)

    # 7) Run Rosetta molfile_to_params.py
    print("")
    print("############################################## ROSETTA molfile_to_params.py ##############################################")
    # Change into output directory so Rosetta drops params/PDBs there
    cwd_orig = os.getcwd()
    os.chdir(out_dir)
    try:
        run_molfile_to_params(
            lig_code=args.desired_ligand_3letter_code,
            input_mol2=os.path.basename(mol2_path),
            verbose=args.verbose
        )
    finally:
        os.chdir(cwd_orig)

    print("")
    print("############################################## DONE ##############################################")
    print(f"[INFO] Ligand extraction and parameter generation complete. Outputs are in: {out_dir}")

if __name__ == "__main__":
    main()
