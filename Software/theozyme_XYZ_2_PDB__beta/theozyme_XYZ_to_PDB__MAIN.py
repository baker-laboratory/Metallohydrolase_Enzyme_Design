#!/usr/bin/env python3
"""
Convert theozyme XYZ to formatted PDB with ligand/residue separation,
residue identity annotation, Rosetta side-chain rebuilding, and final
Theozyme+ligand assembly.

This script:
  1. Splits an XYZ theozyme into residue and ligand XYZ files.
  2. Converts both to PDBs (Open Babel), cleans formatting, and assigns chains.
  3. Renumbers residues; identifies residue types via SMILES (Rosetta helper).
  4. Replaces UNL with the correct amino acid codes.
  5. Builds Rosetta residue “ideal” models and aligns them to DFT tip atoms.
  6. Propagates residue name / chain / number from Rosetta to aligned tips.
  7. Installs missing atoms and builds a final residue-only PDB.
  8. Combines residues + ligand into a final theozyme PDB.
  9. Optionally standardizes GLU/ASP OE1/OE2 and OD1/OD2 based on ligand proximity.
 10. Optionally cleans up intermediate files.

Helper STEP scripts are assumed to live in the same directory:
  STEP_1__identify_residues.py
  STEP_2__build_full_residue_from_subset.py
  STEP_3__superimpose_ideal_residue_on_subset.py
  STEP_4__install_missing_atoms.py
  STEP_5__standardize_GLU_ASP_tip_Os.py
"""

import argparse
import os
import string
import glob
import shutil
import sys
import re
from pathlib import Path

#######################################
### CONFIG & PATHS (RELATIVE SETUP) ###
#######################################

SCRIPT_DIR = Path(__file__).resolve().parent

# External helper scripts (by STEP order)
IDENTIFY_RESIDUES_SCRIPT = SCRIPT_DIR / "STEP_1__identify_residues.py"
BUILD_FROM_TIPS_SCRIPT = SCRIPT_DIR / "STEP_2__build_full_residue_from_subset.py"
OPTIMAL_SUPERIMPOSE_SCRIPT = SCRIPT_DIR / "STEP_3__superimpose_ideal_residue_on_subset.py"
INSTALL_MISSING_ATOMS_SCRIPT = SCRIPT_DIR / "STEP_4__install_missing_atoms.py"
STANDARDIZE_GLU_ASP_SCRIPT = SCRIPT_DIR / "STEP_5__standardize_GLU_ASP_tip_Os.py"

# Container / tools
#APPTAINER = "/software/containers/crispy.sif" (MUST BE INPUT AS ARGUMENT)
#OBABEL_PATH = "/home/woodbuse/conda_envs/openbabel_env/bin/obabel" (MUST BE INPUT AS ARGUMENT)


#############################
### CORE MAIN WORKFLOW    ###
#############################

def main(
    input_xyz,
    ligand_atom_ranges,
    lig_3letter_code,
    ligand_chain,
    tip_atom_residues_3letter,
    DO_NOT_pass_tip_atom_residues_3letter_to_help_identifier,
    KEEP_temp_smiles_and_res_pdbs_for_debug,
    KEEP_temp_residue_and_temp_ligand_files_for_debug,
    ligand_atom_for_close_proximity_to_OE2glu_and_OD2asp=None,
):
    """
    High-level orchestration of the theozyme XYZ → theozyme PDB workflow.
    """

    ###########################################
    # 1. SPLIT XYZ INTO RESIDUE / LIGAND XYZ #
    ###########################################

    ligand_atoms = parse_atom_ranges(ligand_atom_ranges)

    temp_residue_xyz_file = create_temp_xyz(input_xyz, ligand_atoms, exclude=True)
    temp_ligand_xyz_file = create_temp_xyz(input_xyz, ligand_atoms, exclude=False)

    ###########################################
    # 2. CONVERT XYZ → PDB (Open Babel)      #
    #    + CLEAN INITIAL PDB FORMATTING      #
    ###########################################

    # Residues → PDB
    residue_pdb_file = temp_residue_xyz_file.replace("TEMP.xyz", "residues_TEMP.pdb")

    # Use --separate then post-process into a single PDB with distinct residue IDs
    os.system(f"{OBABEL_PATH} -ixyz {temp_residue_xyz_file} --separate -opdb -O {residue_pdb_file}")

    processed = []
    with open(residue_pdb_file, 'r') as infile:
        model_idx = 0
        atom_serial = 1
        for line in infile:
            if line.startswith("MODEL"):
                model_idx += 1
                continue
            if line.startswith("ENDMDL"):
                continue
            if line.startswith(("ATOM", "HETATM")):
                record = line[:6]       # columns 1–6
                atom_and_rest = line[11:22]  # original atom name / res / chain segment
                tail = line[26:]        # everything from col 27 onward
                new_line = f"{record}{atom_serial:5d}{atom_and_rest}{model_idx:4d}{tail}"
                processed.append(new_line)
                atom_serial += 1

    with open(residue_pdb_file, 'w') as outfile:
        outfile.writelines(processed)

    # Ligand → PDB
    ligand_pdb_file = temp_ligand_xyz_file.replace("TEMP.xyz", "ligand_TEMP.pdb")
    os.system(f"{OBABEL_PATH} -ixyz {temp_ligand_xyz_file} -opdb -O {ligand_pdb_file}")

    # Remove raw XYZ intermediates
    os.remove(temp_residue_xyz_file)
    os.remove(temp_ligand_xyz_file)

    # Clean both TEMP PDB files
    clean_pdb_file(residue_pdb_file)
    clean_pdb_file(ligand_pdb_file)

    # For residues: HETATM→ATOM, assign chain letters
    update_residues_pdb(residue_pdb_file)

    # For ligand: force HETATM, set code/chain/resnum, unique atom names
    update_ligand_pdb(ligand_pdb_file, lig_3letter_code, ligand_chain)
    modify_pdb_atom_names(ligand_pdb_file)

    print("Temporary PDB files created:")
    print(f"  Residues only: {residue_pdb_file}")
    print(f"  Ligand only (updated): {ligand_pdb_file}")

    ###########################################
    # 3. RENUMBER RESIDUES + ATOMS           #
    ###########################################

    renumber_pdb_inplace(residue_pdb_file)
    print("PASSED RESIDUE RENUMBERING\n")

    ###########################################
    # 4. IDENTIFY RESIDUES (STEP 1 SCRIPT)   #
    #    + APPLY AA CODES TO residues_TEMP   #
    ###########################################

    if DO_NOT_pass_tip_atom_residues_3letter_to_help_identifier:
        os.system(
            f"{APPTAINER} {IDENTIFY_RESIDUES_SCRIPT} "
            f"-input_pdb {residue_pdb_file} "
            f"-tip_atom_residues_3letter {' '.join(tip_atom_residues_3letter)}"
        )
    else:
        os.system(
            f"{APPTAINER} {IDENTIFY_RESIDUES_SCRIPT} "
            f"-input_pdb {residue_pdb_file}"
        )

    print("PASSED RESIDUE IDENTIFICATION\n")

    # Read mapping from residue_map.txt
    with open("residue_map.txt", "r") as f:
        residue_map = [line.strip().split(' = ') for line in f]
        residue_mapping_dict = {
            int(res.split('_TEMP_smiles')[0][3:]): aa for res, aa in residue_map
        }

    print("\n### RESIDUE MAPPING FROM FILE ###")
    for res_num, aa_code in residue_mapping_dict.items():
        print(f"Residue {res_num}: {aa_code}")

    # Replace UNL with correct AA codes in residue_pdb_file
    with open(residue_pdb_file, "r") as infile:
        lines = infile.readlines()

    with open(residue_pdb_file, "w") as outfile:
        current_residue = None
        for line in lines:
            if line.startswith(("ATOM", "HETATM")):
                residue_index = int(line[22:26].strip())
                if residue_index != current_residue:
                    current_residue = residue_index
                aa_code = residue_mapping_dict.get(current_residue, "UNL")
                updated_line = line[:17] + f"{aa_code:<3}" + line[20:]
                outfile.write(updated_line)
            else:
                outfile.write(line)

    print(f"\n### {residue_pdb_file} has been updated with correct amino acid codes ###\n")

    # Clean up STEP 1 intermediates unless requested to keep
    if not KEEP_temp_smiles_and_res_pdbs_for_debug:
        print("\n### CLEANING UP TEMPORARY FILES (SMILES + RES PDBS) ###")

        if os.path.exists("residue_map.txt"):
            os.remove("residue_map.txt")
            print("Deleted residue_map.txt")

        for file in os.listdir("."):
            if file.endswith("_TEMP_smiles.smi"):
                os.remove(file)
                print(f"Deleted {file}")

        for file in os.listdir("."):
            if file.startswith("pdb") and file.endswith("_TEMP.pdb"):
                os.remove(file)
                print(f"Deleted {file}")
    else:
        print("\n### KEEPING TEMPORARY SMILES AND RESIDUE PDB FILES FOR DEBUGGING ###")

    #########################################################
    # 5. BUILD ROSETTA RESIDUES FROM TIPS (STEP 2 SCRIPT)  #
    #########################################################

    os.system(f"{APPTAINER} {BUILD_FROM_TIPS_SCRIPT} -input_pdb {residue_pdb_file}")
    print("\n### BUILT ROSETTA RESIDUE FILES FOR ALIGNMENT WITH THE CORRESPONDING TIPS ###\n")

    # At this point you can manually tweak chi angles in the generated Rosetta PDBs if desired.
    # (Exit early here if you want interactive debugging.)
    # sys.exit("Stopping here")

    #########################################################
    # 6. SUPERIMPOSE IDEAL RESIDUES ON TIPS (STEP 3 SCRIPT) #
    #########################################################

    input_files = glob.glob("*_inputpdb_TEMP.pdb")
    rosetta_files = glob.glob("*_rosetta_TEMP.pdb")

    pairs = []
    for input_file in input_files:
        prefix = input_file.split("_inputpdb_TEMP.pdb")[0]
        rosetta_file = f"{prefix}_rosetta_TEMP.pdb"
        if rosetta_file in rosetta_files:
            pairs.append((input_file, rosetta_file))

    print("\n### MATCHING PAIRS FOUND ###")
    for input_file, rosetta_file in pairs:
        print(f"Input: {input_file}, Rosetta: {rosetta_file}")

    # Only match side chains (omit backbone, keep CA)
    build_script = f"{APPTAINER} {OPTIMAL_SUPERIMPOSE_SCRIPT}"
    for input_file, rosetta_file in pairs:
        command = (
            f"{build_script} "
            f"-i {input_file} "
            f"-r {rosetta_file} "
            f"-o aligned_rosetta.pdb "
            f"--omit_backbone_but_keep_CA"
        )
        os.system(command)
        print(f"\n### EXECUTED: {command} ###")

    print("\n### ROSETTA RESIDUES SUCCESSFULLY ALIGNED WITH TIPS ###")

    #########################################################
    # 7. COPY RES NAME / CHAIN / RESNUM FROM ROSETTA        #
    #    → ALIGNED TIP PDBS                                 #
    #########################################################

    rosetta_files = glob.glob("*_rosetta_TEMP.pdb")
    aligned_rosetta_files = glob.glob("*_inputpdb_TEMP_aligned_rosetta.pdb")

    aligned_pairs = []
    for rosetta_file in rosetta_files:
        prefix = rosetta_file.split("_rosetta_TEMP.pdb")[0]
        aligned_file = f"{prefix}_inputpdb_TEMP_aligned_rosetta.pdb"
        if aligned_file in aligned_rosetta_files:
            aligned_pairs.append((rosetta_file, aligned_file))

    print("\n### MATCHING ALIGNED PAIRS FOUND ###")
    for rosetta_file, aligned_file in aligned_pairs:
        print(f"Rosetta: {rosetta_file}, Aligned: {aligned_file}")

    for rosetta_file, aligned_file in aligned_pairs:
        print(f"\n### CORRECTING RESIDUE/CHAIN/NUMBER IN {aligned_file} USING TEMPLATE {rosetta_file} ###")
        correct_residue_chain_and_number(rosetta_file, aligned_file)

    print("\n### RENAMED TIP ATOM ALIGNED ROSETTA RESIDUES ###")

    aligned_rosetta_files = glob.glob("*_inputpdb_TEMP_aligned_rosetta.pdb")

    print("\n### ALIGNED ROSETTA PDB FILES ###")
    for aligned_file in aligned_rosetta_files:
        print(aligned_file)

    #########################################################
    # 8. INSTALL MISSING ATOMS (STEP 4 SCRIPT WRAPPER)      #
    #########################################################

    final_output_pdb = "final_residue_construction.pdb"
    aligned_files_argument = " ".join(aligned_rosetta_files)

    command = (
        f"python {INSTALL_MISSING_ATOMS_SCRIPT} "
        f"-aligned_rosetta_pdbs_for_residues {aligned_files_argument} "
        f"-dft_tip_atoms_of_residues_only {residue_pdb_file} "
        f"-output_pdb {final_output_pdb}"
    )
    os.system(command)
    print(f"\n### EXECUTED: {command} ###")
    print(f"Output PDB: {final_output_pdb}")

    #########################################################
    # 9. COMBINE RESIDUES + LIGAND INTO THEOZYME PDB        #
    #########################################################

    new_ligand_pdb = ligand_pdb_file.replace("ligand_TEMP", "only")
    shutil.copy(ligand_pdb_file, new_ligand_pdb)
    print(f"\n### COPIED: {ligand_pdb_file} to {new_ligand_pdb} ###")

    residues_pdb = final_output_pdb
    output_pdb = os.path.basename(input_xyz).replace(".xyz", f"__lig_{lig_3letter_code}_artificialBB_theozyme.pdb")

    combine_residues_and_ligand(residues_pdb, ligand_pdb_file, output_pdb)
    print(f"Combined PDB Output: {output_pdb}")

    #########################################################
    # 10. OPTIONAL GLU/ASP OE1/OE2 & OD1/OD2 STANDARDIZATION #
    #########################################################

    input_xyz_dir = os.path.dirname(input_xyz)
    pdb_theozymes_folder = os.path.join(input_xyz_dir, "pdb_theozymes")
    pdb_ligands_only_folder = os.path.join(input_xyz_dir, "pdb_theozymes_ligands_only")

    os.makedirs(pdb_theozymes_folder, exist_ok=True)
    os.makedirs(pdb_ligands_only_folder, exist_ok=True)

    if ligand_atom_for_close_proximity_to_OE2glu_and_OD2asp:
        command = (
            f"python {STANDARDIZE_GLU_ASP_SCRIPT} "
            f"--input_pdb {output_pdb} "
            f"--ligand_code {lig_3letter_code} "
            f"--ligand_atom_for_close_proximity_to_OE2glu_and_OD2asp "
            f"{ligand_atom_for_close_proximity_to_OE2glu_and_OD2asp}"
        )
        print(f"\n### EXECUTING STANDARDIZATION: {command}")
        os.system(command)
        print("### DONE WITH STANDARDIZATION ###\n")

    #########################################################
    # 10b. ADD REMARK 666 LINES TO FINAL THEOZYME PDB       #
    #########################################################

    add_remark666_to_pdb(output_pdb, lig_3letter_code)
    print(f"\n### ADDED REMARK 666 LINE TEMPLATE ###")

    #########################################################
    # 11. MOVE FINAL OUTPUTS & CLEANUP INTERMEDIATE FILES   #
    #########################################################

    # Move final theozyme
    shutil.move(output_pdb, os.path.join(pdb_theozymes_folder, output_pdb))
    print(f"\n### MOVED: {output_pdb} to {pdb_theozymes_folder}/ ###")

    # Move ligand-only PDB
    only_ligand_pdb = ligand_pdb_file.replace("ligand_TEMP", "only")
    shutil.move(
        only_ligand_pdb,
        os.path.join(pdb_ligands_only_folder, os.path.basename(only_ligand_pdb))
    )
    print(f"\n### MOVED: {only_ligand_pdb} to {pdb_ligands_only_folder}/ ###")

    if not KEEP_temp_residue_and_temp_ligand_files_for_debug:
        print("\n### CLEANING UP INTERMEDIATE FILES ###")

        files_to_delete = [
            "SEMI_FINAL_RESIDUE_CONSTRUCTION_TEMP.pdb",
            "single_file_aligned_rosetta_pdb_residues_TEMP.pdb",
            "final_residue_construction.pdb",
            ligand_pdb_file,
            residue_pdb_file,
        ]

        patterns_to_delete = [
            "*_inputpdb_TEMP.pdb",
            "*_inputpdb_TEMP_aligned_rosetta.pdb",
            "*_rosetta_TEMP.pdb",
        ]

        for file in files_to_delete:
            if os.path.exists(file):
                os.remove(file)
                print(f"Deleted: {file}")

        for pattern in patterns_to_delete:
            for file in glob.glob(pattern):
                os.remove(file)
                print(f"Deleted: {file}")
    else:
        print("\n### KEEPING INTERMEDIATE FILES FOR DEBUGGING ###")

    print("\n### POST-SCRIPT CLARITY DEBUG V1: If stuff is not correctly aligned in outputs, "
          "consider changing standard residue chi angles in the STEP_2 helper (build_full_residue_from_subset). ###")
    print("### POST-SCRIPT CLARITY DEBUG V2: If the backbone of a residue is missing, consider "
          "adjusting the threshold in 'is_reasonable_pairing' in STEP_3 (superimpose_ideal_residue_on_subset). ###\n")
    print("### SCRIPT COMPLETED SUCCESSFULLY :D ###")


########################################
### COMBINE RESIDUES + LIGAND HELPER ###
########################################

def combine_residues_and_ligand(residues_pdb, ligand_pdb, output_pdb):
    """
    Merge residue PDB and ligand PDB into a single file,
    renumbering atom serials and residue indices sequentially.
    Mirrors combine_residuesPDB_w_ligandPDB.py behavior.
    """

    def parse_atoms(file_path, start_atom_serial, start_res_seq):
        atoms = []
        res_map = {}
        res_seq_counter = start_res_seq

        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    atom_name = line[12:16]
                    res_name = line[17:20]
                    chain_id = line[21]
                    res_seq = int(line[22:26])

                    res_key = (chain_id, res_seq)
                    if res_key not in res_map:
                        res_map[res_key] = res_seq_counter
                        res_seq_counter += 1

                    new_res_seq = res_map[res_key]

                    atoms.append({
                        "record_type": line[0:6],
                        "atom_serial": start_atom_serial,
                        "atom_name": atom_name,
                        "res_name": res_name,
                        "chain_id": chain_id,
                        "res_seq": new_res_seq,
                        "x": float(line[30:38]),
                        "y": float(line[38:46]),
                        "z": float(line[46:54]),
                        "occupancy": float(line[54:60] or 1.0),
                        "temp_factor": float(line[60:66] or 0.0),
                        "element": line[76:78].strip(),
                    })
                    start_atom_serial += 1

        return atoms, start_atom_serial, res_seq_counter

    all_atoms = []
    atom_serial_counter = 1
    res_seq_counter = 1

    # Residues first
    res_atoms, atom_serial_counter, res_seq_counter = parse_atoms(
        residues_pdb, atom_serial_counter, res_seq_counter
    )
    all_atoms.extend(res_atoms)

    # Then ligand
    lig_atoms, atom_serial_counter, res_seq_counter = parse_atoms(
        ligand_pdb, atom_serial_counter, res_seq_counter
    )
    all_atoms.extend(lig_atoms)

    with open(output_pdb, 'w') as out:
        for atom in all_atoms:
            line = "{:<6}{:>5} {:<4}{:1}{:>3} {:1}{:>4}{:1}   {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}          {:>2}".format(
                atom["record_type"],
                atom["atom_serial"],
                atom["atom_name"],
                "",
                atom["res_name"],
                atom["chain_id"],
                atom["res_seq"],
                "",
                atom["x"],
                atom["y"],
                atom["z"],
                atom["occupancy"],
                atom["temp_factor"],
                atom["element"]
            )
            out.write(line.rstrip() + "\n")

    print(f"Combined residues ({residues_pdb}) and ligand ({ligand_pdb}) into {output_pdb}")


###########################
### RENUMBER PDB HELPER ###
###########################

def _parse_pdb_for_renumber(input_pdb):
    records = []
    conect_records = []
    master_line = None

    with open(input_pdb, 'r') as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                record = {
                    "record_type": line[0:6].strip(),
                    "atom_serial": int(line[6:11].strip()),
                    "atom_name": line[12:16],
                    "alt_loc": line[16],
                    "res_name": line[17:20],
                    "chain_id": line[21],
                    "res_seq": int(line[22:26].strip()),
                    "i_code": line[26],
                    "x": float(line[30:38].strip()),
                    "y": float(line[38:46].strip()),
                    "z": float(line[46:54].strip()),
                    "occupancy": float(line[54:60].strip() or 1.0),
                    "temp_factor": float(line[60:66].strip() or 0.0),
                    "element": line[76:78].strip(),
                    "charge": line[78:80].strip(),
                    "line": line
                }
                records.append(record)
            elif line.startswith("CONECT"):
                conect_records.append(line.strip())
            elif line.startswith("MASTER"):
                master_line = line.strip()
    return records, conect_records, master_line


def _group_and_renumber(records):
    res_map = {}
    new_records = []
    atom_serial_counter = 1
    res_serial_counter = 1

    for rec in records:
        res_key = (rec["chain_id"], rec["res_seq"], rec["i_code"])
        if res_key not in res_map:
            res_map[res_key] = res_serial_counter
            res_serial_counter += 1

        new_res_seq = res_map[res_key]
        new_rec = rec.copy()
        new_rec["atom_serial"] = atom_serial_counter
        new_rec["res_seq"] = new_res_seq
        new_records.append(new_rec)
        atom_serial_counter += 1

    return new_records, res_serial_counter - 1


def _update_conect(conect_records, atom_map):
    updated_conect_lines = []

    for line in conect_records:
        if not line.startswith("CONECT"):
            continue

        parts = line.split()
        updated_parts = [parts[0]]

        for atom_str in parts[1:]:
            try:
                old_serial = int(atom_str)
            except ValueError:
                updated_parts.append(atom_str)
                continue

            new_serial = atom_map.get(old_serial)
            if new_serial is not None:
                updated_parts.append(f"{new_serial:4d}")
            else:
                updated_parts.append(atom_str)

        updated_line = " ".join(updated_parts).ljust(80)
        updated_conect_lines.append(updated_line)

    return updated_conect_lines


def _update_master_line(master_line, num_atoms, num_conect):
    if not master_line:
        master_line = "MASTER        0    0    0    0    0    0    0    0    0    0"
    pattern = r"(MASTER\s+)(\d+)(\s+)(\d+)(.*)"
    match = re.match(pattern, master_line)
    if match:
        prefix, _, middle, _, suffix = match.groups()
        new_line = f"{prefix}{num_atoms:5d}{middle}{num_conect:5d}{suffix}"
        return new_line.ljust(80)
    return master_line.ljust(80)


def renumber_pdb_inplace(input_pdb):
    """
    In-place version of renumber_pdb.py:
    - Renumbers atom serials 1..N
    - Renumbers residues 1..M by appearance
    - Fixes CONECT and MASTER.
    """
    records, conect_records, master_line = _parse_pdb_for_renumber(input_pdb)

    old_to_new_atom_serial = {}
    for new_serial, rec in enumerate(records, start=1):
        old_to_new_atom_serial[rec["atom_serial"]] = new_serial

    new_records, _ = _group_and_renumber(records)
    updated_conect_lines = _update_conect(conect_records, old_to_new_atom_serial)

    with open(input_pdb, 'w') as f:
        for rec in new_records:
            line = "{:<6}{:>5} {:<4}{:1}{:>3} {:1}{:>4}{:1}   {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}          {:>2}{:>2}".format(
                rec["record_type"],
                rec["atom_serial"],
                rec["atom_name"],
                rec["alt_loc"],
                rec["res_name"],
                rec["chain_id"],
                rec["res_seq"],
                rec["i_code"],
                rec["x"],
                rec["y"],
                rec["z"],
                rec["occupancy"],
                rec["temp_factor"],
                rec["element"],
                rec["charge"]
            )
            f.write(line.rstrip() + "\n")

        for conect_line in updated_conect_lines:
            f.write(conect_line.rstrip() + "\n")

        master_line_updated = _update_master_line(master_line, len(new_records), len(updated_conect_lines))
        f.write(master_line_updated.rstrip() + "\n")

    print(f"Renumbered PDB written in place to: {input_pdb}")


###############################
### CORRECT RESIDUE HELPERS ###
###############################

def parse_residue_information(pdb_file):
    """
    Read the first ATOM/HETATM line in `pdb_file` and return (res_name, chain_id, res_num).
    """
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                res_name = line[17:20].strip()
                chain_id = line[21].strip()
                res_num = line[22:26].strip()
                print(
                    f"Identified residue information from {pdb_file}: "
                    f"Residue={res_name}, Chain={chain_id}, Residue Number={res_num}"
                )
                return res_name, chain_id, res_num
    raise ValueError(f"No ATOM/HETATM lines found in {pdb_file}.")


def update_residue_chain_and_number_in_place(target_pdb, res_name, chain_id, res_num):
    """
    Update residue name, chain ID, and residue number in `target_pdb` in place.
    """
    updated_lines = []
    with open(target_pdb, 'r') as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                original_line = line
                updated_line = (
                    line[:17] + f"{res_name:>3}" +
                    line[20] +
                    f"{chain_id}" +
                    f"{res_num:>4}" +
                    line[26:]
                )
                updated_lines.append(updated_line)
                print(
                    "Updated line:\n"
                    f"Original: {original_line.strip()}\n"
                    f"Updated:  {updated_line.strip()}"
                )
            else:
                updated_lines.append(line)

    with open(target_pdb, 'w') as f:
        f.writelines(updated_lines)

    print(f"Updated PDB file written in place to {target_pdb}")


def correct_residue_chain_and_number(rosetta_template_pdb, aligned_pdb):
    """
    Read residue info from `rosetta_template_pdb` and apply it to `aligned_pdb` in place.
    """
    res_name, chain_id, res_num = parse_residue_information(rosetta_template_pdb)
    update_residue_chain_and_number_in_place(aligned_pdb, res_name, chain_id, res_num)
    print(f"### CORRECTED RESIDUE INFO IN {aligned_pdb} BASED ON {rosetta_template_pdb} ###")


#######################
### XYZ/PDB HELPERS ###
#######################

def parse_atom_ranges(ligand_range):
    """Parse the ligand atom range string into a set of integers."""
    atoms = []
    for part in ligand_range.split(","):
        if "-" in part:
            start, end = map(int, part.split("-"))
            atoms.extend(range(start, end + 1))
        else:
            atoms.append(int(part))
    return set(atoms)


def create_temp_xyz(input_xyz, ligand_atoms, exclude=True):
    """
    Create a temporary XYZ file with specified atoms removed or kept.
    If exclude=True, all atoms in ligand_atoms are removed (residues only).
    If exclude=False, only ligand_atoms are kept (ligand only).
    """
    suffix = "_TEMP.xyz" if exclude else "_ligand_TEMP.xyz"
    temp_xyz = input_xyz.replace(".xyz", suffix)

    with open(input_xyz, "r") as infile, open(temp_xyz, "w") as outfile:
        lines = infile.readlines()
        # atom_count in header is ignored; we recompute
        selected_lines = [
            line for i, line in enumerate(lines[2:], start=1)
            if (i in ligand_atoms) != exclude
        ]

        outfile.write(f"{len(selected_lines)}\n")
        outfile.write(lines[1])  # comment line unchanged
        outfile.writelines(selected_lines)

    return temp_xyz


def update_residues_pdb(residue_pdb_file):
    """
    Update residues PDB file:
      - Change HETATM to ATOM
      - Replace numeric chain IDs with A, B, C, ... (1→A, 2→B, ...)
    """
    int_to_letter = {i: letter for i, letter in enumerate(string.ascii_uppercase, start=1)}

    with open(residue_pdb_file, "r") as file:
        lines = file.readlines()

    with open(residue_pdb_file, "w") as file:
        for line in lines:
            if line.startswith("HETATM"):
                line = "ATOM  " + line[6:]

            if line.startswith("ATOM  "):
                chain_position = int(line[22:26].strip())
                chain_letter = int_to_letter.get(chain_position, " ")
                line = f"{line[:21]}{chain_letter}{chain_position:4d}{line[26:]}"

            file.write(line)


def update_ligand_pdb(ligand_pdb_file, lig_3letter_code, ligand_chain):
    """
    Update ligand PDB file to:
      - Force HETATM
      - Set 3-letter code
      - Set chain ID
      - Set residue number to 9
      - Keep coordinates / occupancy / B-factor / element
    """
    with open(ligand_pdb_file, "r") as file:
        lines = file.readlines()

    with open(ligand_pdb_file, "w") as file:
        for line in lines:
            if line.startswith("ATOM  "):
                line = "HETATM" + line[6:]
            if line.startswith("HETATM"):
                atom_serial = line[6:11].strip()
                atom_name = line[12:16].strip()
                res_name = lig_3letter_code
                chain_id = ligand_chain
                res_seq = 9
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                occupancy = float(line[54:60].strip())
                temp_factor = float(line[60:66].strip())
                element = line[76:78].strip()

                line = (
                    f"HETATM{int(atom_serial):5d} {atom_name:<3} {res_name} {chain_id}{res_seq:4d}"
                    f"    {x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{temp_factor:6.2f}           {element:<2}\n"
                )

                if line.endswith("ZN\n"):
                    line = line[:-5] + " ZN \n"

            file.write(line)


def modify_pdb_atom_names(input_pdb):
    """
    Modify ligand atom names to ensure uniqueness by appending counters
    (e.g., C1, C2, O1, O2, ZN1, ZN2, ...).
    """
    with open(input_pdb, 'r') as infile:
        lines = infile.readlines()

    atom_counters = {}
    with open(input_pdb, 'w') as outfile:
        for line in lines:
            if line.startswith("HETATM"):
                atom_type = line[12:14].strip()
                atom_counters[atom_type] = atom_counters.get(atom_type, 0) + 1
                unique_atom_name = f"{atom_type}{atom_counters[atom_type]}"
                line = f"{line[:12]} {unique_atom_name:<4}{line[16:]}"
            outfile.write(line)


def clean_pdb_file(pdb_file):
    """Remove COMPND/AUTHOR lines and convert all remaining lines to uppercase."""
    with open(pdb_file, "r") as infile:
        lines = infile.readlines()

    with open(pdb_file, "w") as outfile:
        for line in lines:
            if not line.startswith(("COMPND", "AUTHOR")):
                outfile.write(line.upper())

################################
### REMARK 666 HELPER FUNCS  ###
################################

def collect_protein_residues(atom_lines):
    """
    Return a sorted list of unique protein residues as:
        [(chain_id, resname, resnum), ...]
    sorted by (chain_id, resnum).
    """
    residues = set()
    for line in atom_lines:
        if not line.startswith("ATOM"):
            continue
        chain = line[21]
        resname = line[17:20].strip()
        resnum = int(line[22:26])
        residues.add((chain, resname, resnum))
    return sorted(residues, key=lambda x: (x[0], x[2]))


def make_remark666_for_residues(residues, ligand_code):
    """
    Create REMARK 666 lines for each residue in `residues`.

    Format matches prepare_PDB_structure_into_theozyme.py:
    REMARK 666 MATCH TEMPLATE X LIG    0 MATCH MOTIF A HIS  94   1  1
    """
    remark_lines = []
    for idx, (chain, resname, resnum) in enumerate(residues, start=1):
        line = (
            f"REMARK 666 MATCH TEMPLATE X {ligand_code:<3}    0 MATCH MOTIF "
            f"{chain} {resname:<3} {resnum:>4}{idx:>4}{1:>3}\n"
        )
        remark_lines.append(line)
    return remark_lines


def add_remark666_to_pdb(pdb_path, ligand_code):
    """
    Read `pdb_path`, detect unique protein residues from ATOM records,
    build REMARK 666 lines, and prepend them to the file.

    Overwrites the file in place.
    """
    with open(pdb_path, "r") as f:
        lines = f.readlines()

    # Collect ATOM lines (protein) to define residues
    atom_lines = [l for l in lines if l.startswith("ATOM")]

    if not atom_lines:
        print(f"[WARN] add_remark666_to_pdb: No ATOM lines found in {pdb_path}; skipping REMARK 666.")
        return

    residues = collect_protein_residues(atom_lines)
    remark666_block = make_remark666_for_residues(residues, ligand_code)

    # Prepend REMARK 666 block to existing contents
    new_lines = remark666_block + lines

    with open(pdb_path, "w") as f:
        f.writelines(new_lines)

    print(f"[INFO] Added {len(remark666_block)} REMARK 666 lines to {pdb_path}")


########################
### CLI ENTRY POINT  ###
########################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert theozyme XYZ to formatted PDB with ligand and residue separation."
    )
    parser.add_argument(
        "--input_xyz",
        required=True,
        help="Input XYZ file for theozyme structure"
    )
    parser.add_argument(
        "--ligand_atom_ranges",
        required=True,
        help="Ranges of ligand atoms, e.g., '1-77'"
    )
    parser.add_argument(
        "--ligand_3letter_code",
        required=True,
        help="3-letter code for the ligand in the PDB file"
    )
    parser.add_argument(
        "--ligand_chain",
        default="Z",
        help="Chain ID for the ligand (default: Z)"
    )
    parser.add_argument(
        "--tip_atom_residues_3letter",
        nargs="+",
        required=True,
        help="3-letter codes for residues with specified side chains"
    )
    parser.add_argument(
        "--DO_NOT_pass_tip_atom_residues_3letter_to_help_identifier",
        action="store_false",
        help="Disable passing tip_atom_residues_3letter to STEP_1 identify_residues.py "
             "(default: pass them)."
    )
    parser.add_argument(
        "--KEEP_temp_smiles_and_res_pdbs_for_debug",
        action="store_true",
        help="Keep temporary SMILES and residue PDB files from STEP_1 for debugging."
    )
    parser.add_argument(
        "--KEEP_temp_residue_and_temp_ligand_files_for_debug",
        action="store_true",
        help="Keep intermediate residue and ligand files (e.g. *_TEMP.pdb) for debugging."
    )
    parser.add_argument(
        "--ligand_atom_for_close_proximity_to_OE2glu_and_OD2asp",
        default=None,
        help="Ligand atom name used for proximity-based standardization of ASP/GLU tip atoms "
             "(e.g. H1). If provided, STEP_5 is invoked."
    )

    # in your argparse setup
    parser.add_argument(
        "--apptainer",
        default="/software/containers/crispy.sif",
        help="Path to the Apptainer/Singularity image (default: /software/containers/crispy.sif)",
    )
    parser.add_argument(
        "--obabel_path",
        default="/home/woodbuse/conda_envs/openbabel_env/bin/obabel",
        help="Path to the obabel executable.",
    )

    args = parser.parse_args()
    APPTAINER = args.apptainer
    OBABEL_PATH = args.obabel_path

    args = parser.parse_args()
    main(
        args.input_xyz,
        args.ligand_atom_ranges,
        args.ligand_3letter_code,
        args.ligand_chain,
        args.tip_atom_residues_3letter,
        args.DO_NOT_pass_tip_atom_residues_3letter_to_help_identifier,
        args.KEEP_temp_smiles_and_res_pdbs_for_debug,
        args.KEEP_temp_residue_and_temp_ligand_files_for_debug,
        args.ligand_atom_for_close_proximity_to_OE2glu_and_OD2asp,
    )
