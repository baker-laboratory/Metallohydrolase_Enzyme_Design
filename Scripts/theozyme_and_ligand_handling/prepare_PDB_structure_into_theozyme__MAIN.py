#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author:
    Seth M. Woodbury, David Baker Lab, University of Washington

Email:
    woodbuse@uw.edu

Date:
    2025-05-14

Script: prepare_PDB_structure_into_theozyme__MAIN.py

Purpose:
    Wrap and execute the cleanup (step1) and reorder REMARK666 (step2) scripts by constructing
    and running the appropriate Python commands.

    New pre-cleaning behaviors:
      1) Optionally strip residue insertion-code suffixes (e.g. 132N -> 132 ).
      2) Detect non-canonical amino acids (ncAAs) based on a dictionary.
      3) Optionally:
           • Protect ncAAs from ligandization by temporarily forcing HETATM -> ATOM.
           • Leave ncAAs as ATOM permanently.
           • Fragment ncAAs into canonical AA + ligand (e.g., KCX -> LYS + LIG).

Requirements:
    - Python3
    - The step1 and step2 scripts at the hardcoded paths below

Usage:
    python prepare_PDB_structure_into_theozyme__MAIN.py \
        --input_pdb /path/to/input.pdb \
        --output_pdb_path /path/to/output.pdb \
        --ligand_complex_3_letter_name LIG \
        [--remark666_residue_front_order A244 A199] \
        [--remark666_residue_back_order A207 A143] \
        [--disable_resseq_suffix_cleanup] \
        [--protect_ncAA_from_ligandization] \
        [--leave_ncAA_as_ATOM] \
        [--frag_ncAA_into_cAA_plus_lig [A169 A72 ...]]
"""

from pathlib import Path
import os, sys, argparse, subprocess, string, math

# Sibling helper scripts resolve next to this file, so the whole set can be
# relocated together.
_SCRIPT_DIR = Path(__file__).resolve().parent

# Paths to the step1 and step2 scripts
STEP1_SCRIPT = str(_SCRIPT_DIR / "prepare_PDB_structure_into_theozyme__STEP1__cleanPDB_and_addREMARK666.py")
STEP2_SCRIPT = str(_SCRIPT_DIR / "prepare_PDB_structure_into_theozyme__STEP2__reorder_REMARK666_lines.py")

###############################################################################
# ncAA DICTIONARY (EDIT THIS SECTION TO ADD/CHANGE NON-CANONICAL AAs)
###############################################################################

# Each entry:
#   key = ncAA residue name (3-letter)
#   value = {
#       "canonical_resname": <cAA 3-letter>,
#       "atom_name_map": { <ncAA_atom_name>: <cAA_atom_name> or None },
#       "ligand_resname": <3-letter ligand name for "extra" atoms>
#   }
#
# Any ncAA atoms whose names are in atom_name_map with a non-None value will
# be turned into canonical ATOM lines (same chain / residue number).
# All remaining atoms from that ncAA residue will be turned into HETATM with
# a new (chain, resseq) and residue name ligand_resname.
#
# NOTE: This KCX mapping is a starting point; tweak as needed.
NONCANONICAL_AA_MAP = {
    "KCX": {
        "canonical_resname": "LYS",
        "atom_name_map": {
            # Backbone
            "N": "N", "CA": "CA", "C": "C", "O": "O",
            # Side chain
            "CB": "CB", "CG": "CG", "CD": "CD", "CE": "CE", "NZ": "NZ",
            # Add / edit more as needed, e.g. "H": "H", etc.
        },
        "ligand_resname": "LIG",
    },
}

###############################################################################
# PROTEIN / SIDECHAIN POLAR HYDROGEN DICTIONARIES
# (EDIT THIS SECTION TO TUNE WHICH POLAR Hs ARE CANONICAL / PROTECTED)
###############################################################################

PROTEIN_RESNAMES = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS",
    "ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP",
    "TYR","VAL"
}

# Mapping: resname -> heavy_atom_name -> list of canonical H atom names
# (You can extend/edit this any time.)
PROTECTED_POLAR_H_MAP = {
    "SER": {"OG":  ["HG"]},
    "THR": {"OG1": ["HG1"]},
    "TYR": {"OH":  ["HH"]},
    "CYS": {"SG":  ["HG"]},
    "TRP": {"NE1": ["HE1"]},
    "HIS": {"ND1": ["HD1"], "NE2": ["HE2"]},
    "LYS": {"NZ":  ["HZ1", "HZ2", "HZ3"]},
    "ARG": {
        "NE":  ["HE"],
        "NH1": ["HH11", "HH12"],
        "NH2": ["HH21", "HH22"],
    },
    "ASN": {"ND2": ["HD21", "HD22"]},
    "GLN": {"NE2": ["HE21", "HE22"]},
    "ASP": {"OD1": ["HD1"], "OD2": ["HD2"]},
    "GLU": {"OE1": ["HE1"], "OE2": ["HE2"]},
}


###############################################################################
# HELPER FUNCTIONS
###############################################################################

def parse_args():
    p = argparse.ArgumentParser(description="Wrapper to run step1 cleanup and step2 reorder of PDB")
    p.add_argument('--input_pdb', required=True, help='Path to the input PDB file')
    p.add_argument('--output_pdb_path', required=True, help='Destination path for the cleaned PDB file')
    p.add_argument('--ligand_complex_3_letter_name', required=True, help='Three-letter code to group all HETATM entries')
    p.add_argument('--remark666_residue_front_order', nargs='*', default=[], help='Optional list of ChainResidue entries (e.g. A57) to place first')
    p.add_argument('--remark666_residue_back_order', nargs='*', default=[], help='Optional list of ChainResidue entries (e.g. A207) to place last')
    p.add_argument('--disable_resseq_suffix_cleanup', action='store_true',
                   help='If set, DO NOT strip residue insertion-code suffixes (e.g. 132N -> 132 ).')
    p.add_argument('--protect_ncAA_from_ligandization', action='store_true',
                   help='Temporarily force ncAA HETATM -> ATOM for step1/2, then revert those back to HETATM at the end.')
    p.add_argument('--leave_ncAA_as_ATOM', action='store_true',
                   help='Force ncAA HETATM -> ATOM and keep them as ATOM permanently (no revert).')
    p.add_argument('--frag_ncAA_into_cAA_plus_lig', nargs='*',
                   help='Fragment ncAA into canonical AA + ligand. Optionally pass specific residues like A169 A72; if no residues given, all ncAAs are fragmented.')
    p.add_argument('--disable_intelligent_hstrip', action='store_true', help='If set, do NOT intelligently strip hydrogens from canonical ncAA fragments.')    
    p.add_argument('--keep_precleaned_pdb', action='store_true', help='If set, keep the intermediate precleaned PDB (useful for debugging).')  
    p.add_argument('--add_CA_to_labeled_frag', action='store_true', help='Add CA atoms to canonical residues missing CA by reusing CB hydrogens (experimental).')      
    p.add_argument('--protect_sidechain_polarH', nargs='*',help=('Protect sidechain polar hydrogens by renaming H* -> Q* before step1/2. '
              'Optionally pass residues like A32 A25; if no residues are given, applies to all protein residues.'))
    return p.parse_args()


def run_script(cmd, description):
    print(f"[INFO] Executing: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)


def strip_insertion_code(line):
    """Remove insertion-code letter (column 27) for ATOM/HETATM, e.g. 132N -> 132 ."""
    if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 27: return line
    if line[26] != "":  # column 27
        if line[26] != " ":
            line = line[:26] + " " + line[27:]
    return line


def detect_ncaa(lines):
    """Return dict of ncAA residues found: {(resname, chain, resseq, icode): {'record_types': {ATOM/HETATM}}}."""
    ncaa_residues = {}
    for line in lines:
        if not line.startswith(("ATOM  ", "HETATM")): continue
        resname = line[17:20].strip()
        if resname not in NONCANONICAL_AA_MAP: continue
        chain = line[21]
        resseq = line[22:26].strip()
        icode = (line[26].strip() or " ")
        key = (resname, chain, resseq, icode)
        info = ncaa_residues.setdefault(key, {"record_types": set()})
        info["record_types"].add(line[0:6].strip())
    return ncaa_residues


def gather_used_positions(lines):
    """Collect all (chain, resseq, icode) positions used by ATOM/HETATM."""
    used = set()
    for line in lines:
        if not line.startswith(("ATOM  ", "HETATM")): continue
        chain = line[21]
        resseq = line[22:26].strip()
        icode = (line[26].strip() or " ")
        used.add((chain, resseq, icode))
    return used


def new_lig_position_generator(used):
    """Yield new (chain, resseq, icode) combos not in 'used', deterministic A1..Z1, A2..Z2, etc."""
    for resseq_int in range(1, 9999):
        resseq_str = str(resseq_int)
        for chain in string.ascii_uppercase:
            key = (chain, resseq_str, " ")
            if key not in used:
                used.add(key)
                yield key


def replace_field(line, start, end, new):
    """Replace substring [start:end] (0-based, end-excl) with 'new' (same length)."""
    if len(new) != (end - start):
        raise ValueError(f"replace_field length mismatch: {end - start} vs {len(new)}")
    return line[:start] + new + line[end:]


def build_frag_filters(arg_list):
    """
    Translate --frag_ncAA_into_cAA_plus_lig argument into:
      None                 -> no fragmentation
      {'all': True, ...}   -> fragment all ncAA
      {'all': False, 'targets': {(chain, resseq), ...}} -> fragment these ncAA only
    """
    if arg_list is None: return None
    if len(arg_list) == 0: return {"all": True, "targets": set()}

    targets = set()
    for token in arg_list:
        token = token.strip()
        if not token: continue
        chain = token[0]
        resseq = "".join(ch for ch in token[1:] if ch.isdigit())
        if not resseq:
            print(f"[WARN] Could not parse residue identifier '{token}' (expected like A169); skipping.", file=sys.stderr)
            continue
        targets.add((chain, resseq))
    return {"all": False, "targets": targets}


def frag_ncaa(lines, frag_filters, enable_intelligent_hstrip=True):
    """
    Fragment ncAA residues into canonical AA + ligand, based on NONCANONICAL_AA_MAP.

    If enable_intelligent_hstrip is True:
      - For each fragmented ncAA residue:
          * Hydrogens whose nearest heavy atom is canonical -> STRIPPED (deleted).
          * Hydrogens whose nearest heavy atom is ligand-side -> LIGANDIZED.
    """
    if frag_filters is None: return lines

    used = gather_used_positions(lines)
    pos_gen = new_lig_position_generator(used)
    lig_pos_for_res = {}  # key: (chain, resseq, icode) of ncAA -> (chain, resseq, icode) for ligand

    target_all = frag_filters.get("all", False)
    targets = frag_filters.get("targets", set())

    # ----------------------------------------------------------------------
    # Precompute hydrogen handling: which Hs to strip vs force into ligand
    # ----------------------------------------------------------------------
    strip_H = set()              # indices of lines to drop
    force_lig_H_to_ligand = set()  # indices of lines forced to ligand branch

    if enable_intelligent_hstrip:
        groups = {}  # key: (resname, chain, resseq, icode) -> [(idx, line), ...]
        for idx, line in enumerate(lines):
            if not line.startswith(("ATOM  ", "HETATM")): continue
            resname = line[17:20].strip()
            if resname not in NONCANONICAL_AA_MAP: continue
            chain = line[21]
            resseq = line[22:26].strip()
            icode = (line[26].strip() or " ")
            if not (target_all or (chain, resseq) in targets): continue
            key = (resname, chain, resseq, icode)
            groups.setdefault(key, []).append((idx, line))

        for (resname, chain, resseq, icode), grp in groups.items():
            rule = NONCANONICAL_AA_MAP[resname]
            canon_map = rule["atom_name_map"]

            heavy = []      # (idx, x, y, z, is_canonical_heavy)
            hydrogens = []  # (idx, x, y, z)

            for idx, line in grp:
                atom_name = line[12:16].strip()
                # Element detection: prefer element column, fall back to first char of atom name
                elem = line[76:78].strip() if len(line) >= 78 else ""
                if not elem:
                    elem = atom_name[0].upper() if atom_name else ""

                # Coordinates
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    # If coords are weird, skip for H-handling (we'll just fall back to default behavior)
                    continue

                if elem == "H":
                    hydrogens.append((idx, x, y, z))
                else:
                    is_canon = atom_name in canon_map and canon_map[atom_name] is not None
                    heavy.append((idx, x, y, z, is_canon))

            if not heavy:  # no heavy atoms to anchor to; skip special H handling
                continue

            for h_idx, hx, hy, hz in hydrogens:
                best_d2 = None
                best_is_canon = False
                for a_idx, ax, ay, az, is_canon in heavy:
                    d2 = (hx - ax)**2 + (hy - ay)**2 + (hz - az)**2
                    if best_d2 is None or d2 < best_d2:
                        best_d2 = d2
                        best_is_canon = is_canon
                if best_d2 is None:
                    continue
                if best_is_canon:
                    strip_H.add(h_idx)  # belongs to canonical fragment -> strip
                else:
                    force_lig_H_to_ligand.add(h_idx)  # belongs to ligand fragment -> ligandize

    # ----------------------------------------------------------------------
    # Main fragmentation pass
    # ----------------------------------------------------------------------
    new_lines = []
    for idx, line in enumerate(lines):
        if not line.startswith(("ATOM  ", "HETATM")):
            new_lines.append(line); continue

        # Drop hydrogens we decided to strip
        if idx in strip_H:
            continue

        resname = line[17:20].strip()
        if resname not in NONCANONICAL_AA_MAP:
            new_lines.append(line); continue

        chain = line[21]
        resseq = line[22:26].strip()
        icode = (line[26].strip() or " ")
        res_key_for_filter = (chain, resseq)

        if not (target_all or res_key_for_filter in targets):
            new_lines.append(line); continue

        rule = NONCANONICAL_AA_MAP[resname]
        canon = rule["canonical_resname"]
        canon_map = rule["atom_name_map"]
        ligand_resname = rule["ligand_resname"]
        atom_name = line[12:16].strip()

        # Decide if this atom should be canonical or ligand
        #  - Some hydrogens are explicitly forced into ligand
        #  - Others follow atom_name_map (canonical) vs "everything else" (ligand)
        if idx in force_lig_H_to_ligand or atom_name not in canon_map or canon_map[atom_name] is None:
            # -----------------------------
            # Ligand branch
            # -----------------------------
            lig_key = (chain, resseq, icode)
            if lig_key not in lig_pos_for_res:
                lig_pos_for_res[lig_key] = next(pos_gen)
            lig_chain, lig_resseq, lig_icode = lig_pos_for_res[lig_key]
            resseq_field = lig_resseq.rjust(4)

            line2 = line
            line2 = replace_field(line2, 0, 6, "HETATM")
            line2 = replace_field(line2, 17, 20, ligand_resname.rjust(3))
            line2 = replace_field(line2, 21, 22, lig_chain)
            line2 = replace_field(line2, 22, 26, resseq_field)
            line2 = replace_field(line2, 26, 27, lig_icode)
            new_lines.append(line2)
        else:
            # -----------------------------
            # Canonical AA branch
            # -----------------------------
            new_name = canon_map[atom_name].rjust(4)
            line2 = line
            line2 = replace_field(line2, 0, 6, "ATOM  ")
            line2 = replace_field(line2, 12, 16, new_name)
            line2 = replace_field(line2, 17, 20, canon.rjust(3))
            new_lines.append(line2)

    return new_lines

def convert_ncaa_hetatm_to_atom(lines):
    """
    For ncAAs, convert HETATM -> ATOM and return:
      new_lines, protected_atoms
    where protected_atoms is a set of (chain, resseq, icode, atom_name).
    """
    protected_atoms = set()
    new_lines = []
    for line in lines:
        if not line.startswith("HETATM"):
            new_lines.append(line); continue
        resname = line[17:20].strip()
        if resname not in NONCANONICAL_AA_MAP:
            new_lines.append(line); continue
        chain = line[21]
        resseq = line[22:26].strip()
        icode = (line[26].strip() or " ")
        atom_name = line[12:16].strip()
        protected_atoms.add((chain, resseq, icode, atom_name))
        line2 = replace_field(line, 0, 6, "ATOM  ")
        new_lines.append(line2)
    return new_lines, protected_atoms


def revert_protected_ncaa_atom_to_hetatm(lines, protected_atoms):
    """Revert ncAA atoms we previously forced to ATOM back to HETATM."""
    if not protected_atoms: return lines
    new_lines = []
    for line in lines:
        if not line.startswith("ATOM  "):
            new_lines.append(line); continue
        chain = line[21]
        resseq = line[22:26].strip()
        icode = (line[26].strip() or " ")
        atom_name = line[12:16].strip()
        if (chain, resseq, icode, atom_name) in protected_atoms:
            line = replace_field(line, 0, 6, "HETATM")
        new_lines.append(line)
    return new_lines


def preclean_pdb(input_pdb, precleaned_pdb, args):
    """
    Pre-cleaning:
      - Detect ncAAs and print summary.
      - Optionally fragment ncAAs into cAA+lig.
      - Optionally strip insertion-code suffixes (unless disabled).
      - Optionally convert ncAA HETATM -> ATOM (protect/leave).
    Returns:
      precleaned_pdb_path, protected_atoms
    """
    with open(input_pdb, 'r') as fh: lines = fh.readlines()

    # 1) Detect ncAAs (always) and tell user where to edit dictionary.
    ncaa_residues = detect_ncaa(lines)
    if ncaa_residues:
        print("[INFO] ncAA detection (edit NONCANONICAL_AA_MAP near top of this script to change):")
        for (resname, chain, resseq, icode), info in sorted(ncaa_residues.items()):
            icode_str = "" if icode == " " else icode
            recs = ",".join(sorted(info["record_types"]))
            print(f"       - {resname} at {chain}{resseq}{icode_str} (records: {recs})")
    else:
        print("[INFO] No ncAA residues detected. Edit NONCANONICAL_AA_MAP near top of this script to add more.")

    # 2) Fragment ncAAs if requested
    frag_filters = build_frag_filters(args.frag_ncAA_into_cAA_plus_lig)
    if frag_filters is not None:
        print("[INFO] Fragmenting ncAAs into canonical AA + ligand according to NONCANONICAL_AA_MAP.")
        if args.disable_intelligent_hstrip:
            print("[INFO] Intelligent hydrogen stripping for ncAA fragments DISABLED (per user flag).")
        else:
            print("[INFO] Intelligent hydrogen stripping for ncAA fragments ENABLED (default).")
        lines = frag_ncaa(lines, frag_filters, enable_intelligent_hstrip=not args.disable_intelligent_hstrip)


    # 3) Strip insertion-code suffixes (unless disabled)
    if not args.disable_resseq_suffix_cleanup:
        print("[INFO] Stripping residue insertion-code suffixes (e.g. 132N -> 132 ).")
        lines = [strip_insertion_code(l) for l in lines]
    else:
        print("[INFO] Skipping residue insertion-code cleanup (requested by user).")

    # 4) Protect / leave ncAAs as ATOM
    protected_atoms = set()
    if args.protect_ncAA_from_ligandization or args.leave_ncAA_as_ATOM:
        print("[INFO] Converting ncAA HETATM -> ATOM before step1/2 (protect/leave flags).")
        lines, protected_atoms = convert_ncaa_hetatm_to_atom(lines)

    # 5) Add CA atoms to canonical residues missing CA (using CB hydrogens), if requested
    if args.add_CA_to_labeled_frag:
        print("[INFO] add_CA_to_labeled_frag flag detected; attempting to add CA atoms to canonical residues missing CA.")
        lines = add_CA_to_labeled_frag(lines)
    else:
        print("[INFO] add_CA_to_labeled_frag not requested; skipping CA insertion.")

    # 6) Protect sidechain polar hydrogens (rename H* -> Q*) if requested
    if args.protect_sidechain_polarH is not None:
        print("[INFO] protect_sidechain_polarH flag detected; marking sidechain polar H atoms with Q prefix.")
        lines = protect_sidechain_polarH(lines, args.protect_sidechain_polarH)
    else:
        print("[INFO] protect_sidechain_polarH not requested; skipping sidechain H protection.")


    with open(precleaned_pdb, 'w') as fh: fh.writelines(lines)
    return precleaned_pdb, protected_atoms


def protect_sidechain_polarH(lines, residue_tokens):
    """
    Rename selected sidechain polar Hs from H* -> Q* so they survive step1/2, then
    later Q* will be reverted to H*.

    residue_tokens:
      - None => feature OFF
      - []   => apply to ALL protein residues
      - ["A32", "A25", ...] => apply only to those (chain,resseq) combos
    """
    if residue_tokens is None:
        return lines

    # Interpret residue list
    if len(residue_tokens) == 0:
        apply_all = True
        target_res = set()
        print("[INFO] protect_sidechain_polarH: applying to ALL protein residues.")
    else:
        apply_all = False
        target_res = set()
        for token in residue_tokens:
            token = token.strip()
            if not token: continue
            chain = token[0]
            resseq = "".join(ch for ch in token[1:] if ch.isdigit())
            if not resseq:
                print(f"[WARN] protect_sidechain_polarH: could not parse residue '{token}' (expected like A32); skipping.", file=sys.stderr)
                continue
            target_res.add((chain, resseq))
        print(f"[INFO] protect_sidechain_polarH: applying to residues: {sorted(target_res)}")

    # Group per residue: collect heavy + H atoms
    residues = {}  # key: (chain, resseq, icode, resname) -> dict(heavy=[...], H=[...])
    for idx, line in enumerate(lines):
        if not line.startswith("ATOM  "):  # protein only
            continue
        resname = line[17:20].strip()
        if resname not in PROTEIN_RESNAMES:
            continue
        chain = line[21]
        resseq = line[22:26].strip()
        icode = (line[26].strip() or " ")
        if not (apply_all or (chain, resseq) in target_res):
            continue

        atom_name = line[12:16].strip()
        elem = line[76:78].strip() if len(line) >= 78 else ""
        if not elem:
            elem = atom_name[0].upper() if atom_name else ""

        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue

        key = (chain, resseq, icode, resname)
        bucket = residues.setdefault(key, {"heavy": [], "H": []})
        if elem == "H":
            bucket["H"].append((idx, x, y, z, atom_name))
        else:
            bucket["heavy"].append((idx, x, y, z, atom_name, elem))

    # Decide which Hs to protect (and possibly canonicalize) and build atom-name updates
    name_updates = {}
    protected_count = 0

    for (chain, resseq, icode, resname), grp in residues.items():
        heavy = grp["heavy"]
        hydrogens = grp["H"]
        if not heavy or not hydrogens:
            continue

        for h_idx, hx, hy, hz, h_name in hydrogens:
            # Find nearest heavy atom (ANY heavy), then decide if it's allowed
            best_d2 = None
            best_heavy_name = None
            best_heavy_elem = None
            for a_idx, ax, ay, az, a_name, a_elem in heavy:
                d2 = (hx - ax)**2 + (hy - ay)**2 + (hz - az)**2
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    best_heavy_name = a_name
                    best_heavy_elem = a_elem

            if best_heavy_name is None:
                # No heavy atom found for some reason; skip this H
                continue

            # If the *nearest* heavy atom is carbon or backbone N/C/CA/O (except PRO N),
            # then do NOT treat this as a polar sidechain H -> skip it.
            if best_heavy_elem == "C":
                continue
            if best_heavy_name in ("N", "C", "CA", "O") and not (resname == "PRO" and best_heavy_name == "N"):
                continue

            # Try to canonicalize H name based on which heavy atom it is near
            canonical_base = h_name  # default: keep current
            polar_map = PROTECTED_POLAR_H_MAP.get(resname, {})
            canon_list = polar_map.get(best_heavy_name, [])

            if canon_list:
                if h_name in canon_list:
                    canonical_base = h_name
                elif len(canon_list) == 1:
                    canonical_base = canon_list[0]
                else:
                    print(f"[WARN] protect_sidechain_polarH: {resname} {chain}{resseq}{icode} H '{h_name}' near {best_heavy_name} not recognized among canonical {canon_list}; leaving name as-is.")
            else:
                # No canonical mapping known for this heavy atom; just warn once per H and keep name
                print(f"[WARN] protect_sidechain_polarH: no canonical polar-H mapping for {resname} {chain}{resseq}{icode} heavy '{best_heavy_name}'; keeping H name '{h_name}'.")

            # Build Q* name (first letter forced to Q)
            base = canonical_base.strip()
            if not base:
                base = h_name.strip() or "H"
            if base[0] == "H":
                q_base = "Q" + base[1:]
            else:
                q_base = "Q" + base
            q_name = q_base[:4].rjust(4)

            name_updates[h_idx] = q_name
            protected_count += 1


    if protected_count:
        print(f"[INFO] protect_sidechain_polarH: protected {protected_count} sidechain polar hydrogens (H* -> Q*).")
    else:
        print("[INFO] protect_sidechain_polarH: no sidechain polar hydrogens met criteria; nothing changed.")

    # Apply name updates
    new_lines = []
    for idx, line in enumerate(lines):
        if idx in name_updates:
            line = replace_field(line, 12, 16, name_updates[idx])
        new_lines.append(line)

    return new_lines

def revert_protected_sidechain_polarH(lines):
    """
    Revert Q* sidechain polar hydrogens back to H* (Q -> H in atom name prefix).
    Any ATOM/HETATM whose atom name starts with 'Q' is mapped back to 'H'.
    """
    new_lines = []
    for line in lines:
        if not line.startswith(("ATOM  ", "HETATM")):
            new_lines.append(line); continue
        atom_name = line[12:16].strip()
        if atom_name.startswith("Q"):
            h_base = "H" + atom_name[1:]
            new_name = h_base[:4].rjust(4)
            line = replace_field(line, 12, 16, new_name)
        new_lines.append(line)
    return new_lines

def add_CA_to_labeled_frag(lines, ca_cb_bond_length=1.53):
    """
    For canonical protein residues (PROTEIN_RESNAMES) that are missing a CA atom:
      - Require that a CB atom exists.
      - Require that at least one hydrogen is CB-bound (nearest heavy atom in same residue is CB).
      - For each CB-bound H, propose a CA position by extending CB->H to a typical CA–CB bond length.
      - Choose the candidate CA position that maximizes the minimum distance to all other heavy atoms
        (to minimize clashes).
      - Replace that H line with a CA carbon at the chosen coordinates.

    If a residue is missing CA but lacks CB OR lacks CB-bound hydrogens, this function raises an error.
    """
    # Gather global heavy-atom coordinates for clash checking
    heavy_global = []
    for idx, line in enumerate(lines):
        if not line.startswith(("ATOM  ", "HETATM")): continue
        atom_name = line[12:16].strip()
        elem = line[76:78].strip() if len(line) >= 78 else ""
        if not elem:
            elem = atom_name[0].upper() if atom_name else ""
        if elem == "H": continue
        try:
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
        except ValueError:
            continue
        heavy_global.append((idx, x, y, z))

    # Group per canonical residue: only ATOM records, PROTEIN_RESNAMES
    residues = {}  # key: (chain, resseq, icode, resname) -> list of atom dicts
    for idx, line in enumerate(lines):
        if not line.startswith("ATOM  "): continue
        resname = line[17:20].strip()
        if resname not in PROTEIN_RESNAMES: continue
        chain = line[21]
        resseq = line[22:26].strip()
        icode = (line[26].strip() or " ")
        atom_name = line[12:16].strip()
        elem = line[76:78].strip() if len(line) >= 78 else ""
        if not elem:
            elem = atom_name[0].upper() if atom_name else ""
        try:
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
        except ValueError:
            continue
        key = (chain, resseq, icode, resname)
        residues.setdefault(key, []).append({
            "idx": idx, "atom_name": atom_name, "elem": elem,
            "x": x, "y": y, "z": z
        })

    new_lines = list(lines)
    for (chain, resseq, icode, resname), atoms in residues.items():
        # Skip residues that already have CA
        if any(a["atom_name"] == "CA" for a in atoms):
            continue

        # Require CB
        cb_atoms = [a for a in atoms if a["atom_name"] == "CB"]
        if not cb_atoms:
            raise RuntimeError(f"add_CA_to_labeled_frag: residue {resname} {chain}{resseq}{icode} is missing CA and CB; cannot construct CA.")

        cb = cb_atoms[0]
        heavy_same = [a for a in atoms if a["elem"] != "H"]
        H_atoms = [a for a in atoms if a["elem"] == "H"]

        if not H_atoms:
            raise RuntimeError(f"add_CA_to_labeled_frag: residue {resname} {chain}{resseq}{icode} is missing CA and has no hydrogens; cannot construct CA.")

        # Identify hydrogens whose nearest heavy atom in this residue is CB
        cb_hydrogens = []
        for a in H_atoms:
            hx, hy, hz = a["x"], a["y"], a["z"]
            best_d2 = None
            best_heavy = None
            for h in heavy_same:
                dx = hx - h["x"]; dy = hy - h["y"]; dz = hz - h["z"]
                d2 = dx*dx + dy*dy + dz*dz
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    best_heavy = h
            if best_heavy and best_heavy["atom_name"] == "CB":
                cb_hydrogens.append(a)

        if not cb_hydrogens:
            raise RuntimeError(f"add_CA_to_labeled_frag: residue {resname} {chain}{resseq}{icode} is missing CA and has no CB-bound hydrogens; cannot construct CA.")

        # For each CB-bound H, propose a CA position and score by min distance to other heavy atoms
        best_choice = None  # (h_idx, ca_x, ca_y, ca_z, min_d)
        for h in cb_hydrogens:
            hx, hy, hz = h["x"], h["y"], h["z"]
            dx = hx - cb["x"]; dy = hy - cb["y"]; dz = hz - cb["z"]
            d = math.sqrt(dx*dx + dy*dy + dz*dz)
            if d == 0.0:
                continue
            scale = ca_cb_bond_length / d
            ca_x = cb["x"] + dx * scale
            ca_y = cb["y"] + dy * scale
            ca_z = cb["z"] + dz * scale

            # Compute minimum distance to all other heavy atoms (excluding this residue's CB and this H)
            min_d = None
            for (g_idx, gx, gy, gz) in heavy_global:
                if g_idx == cb["idx"] or g_idx == h["idx"]:
                    continue
                ddx = ca_x - gx; ddy = ca_y - gy; ddz = ca_z - gz
                dist = math.sqrt(ddx*ddx + ddy*ddy + ddz*ddz)
                if min_d is None or dist < min_d:
                    min_d = dist

            if min_d is None:
                continue
            if best_choice is None or min_d > best_choice[4]:
                best_choice = (h["idx"], ca_x, ca_y, ca_z, min_d)

        if best_choice is None:
            raise RuntimeError(f"add_CA_to_labeled_frag: could not find a non-clashing CA placement for residue {resname} {chain}{resseq}{icode}.")

        h_idx, ca_x, ca_y, ca_z, min_d = best_choice

        # Replace the chosen H line with a CA carbon at the chosen coordinates
        line = new_lines[h_idx]
        # Normalize to at least 80 chars (for element field), preserve newline
        newline_char = "\n" if line.endswith("\n") else ""
        core = line.rstrip("\n")
        if len(core) < 80:
            core = core.ljust(80)
        line = core + newline_char

        # Atom name " CA "
        ca_name_field = "CA".rjust(4)
        line = replace_field(line, 12, 16, ca_name_field)
        # Coordinates
        line = replace_field(line, 30, 38, f"{ca_x:8.3f}")
        line = replace_field(line, 38, 46, f"{ca_y:8.3f}")
        line = replace_field(line, 46, 54, f"{ca_z:8.3f}")
        # Element "C "
        line = replace_field(line, 76, 78, "C ")

        new_lines[h_idx] = line
        print(f"[INFO] add_CA_to_labeled_frag: inserted CA for {resname} {chain}{resseq}{icode} using former H at index {h_idx} (min heavy-atom distance ~ {min_d:.2f} Å).")

    return new_lines

###############################################################################
# MAIN
###############################################################################
def main():
    args = parse_args()

    # Basic sanity on mutually exclusive / conflicting options
    if args.protect_ncAA_from_ligandization and args.leave_ncAA_as_ATOM:
        print("[ERROR] --protect_ncAA_from_ligandization and --leave_ncAA_as_ATOM are mutually exclusive.", file=sys.stderr)
        sys.exit(1)
    if args.frag_ncAA_into_cAA_plus_lig is not None and (args.protect_ncAA_from_ligandization or args.leave_ncAA_as_ATOM):
        print("[ERROR] --frag_ncAA_into_cAA_plus_lig cannot be combined with ncAA protection/leave flags in this wrapper.", file=sys.stderr)
        sys.exit(1)

    # Make sure output folder exists
    out_dir = os.path.dirname(args.output_pdb_path)
    if out_dir: os.makedirs(out_dir, exist_ok=True)

    # Pre-clean input PDB -> precleaned input file (leaves original untouched)
    base_out = os.path.splitext(os.path.basename(args.output_pdb_path))[0]
    precleaned_input = os.path.join(out_dir if out_dir else ".", f"{base_out}__precleaned_INPUT.pdb")
    precleaned_input, protected_atoms = preclean_pdb(args.input_pdb, precleaned_input, args)

    # Step 1: cleanup + REMARK666
    cmd1 = [
        sys.executable, STEP1_SCRIPT,
        '--input_pdb', precleaned_input,
        '--output_pdb_path', args.output_pdb_path,
        '--ligand_complex_3_letter_name', args.ligand_complex_3_letter_name
    ]
    run_script(cmd1, 'Step1 clean-and-add-REMARK666')

    # Step 2: reorder REMARK666
    cmd2 = [sys.executable, STEP2_SCRIPT, '--input_pdb', args.output_pdb_path]
    if args.remark666_residue_front_order:
        cmd2 += ['--remark666_residue_front_order'] + args.remark666_residue_front_order
    if args.remark666_residue_back_order:
        cmd2 += ['--remark666_residue_back_order'] + args.remark666_residue_back_order
    run_script(cmd2, 'Step2 reorder-REMARK666')

    # Post-processing: revert protected ncAA atoms back to HETATM if requested
    if args.protect_ncAA_from_ligandization and protected_atoms:
        print("[INFO] Reverting protected ncAA atoms back to HETATM in final output.")
        with open(args.output_pdb_path, 'r') as fh: out_lines = fh.readlines()
        out_lines = revert_protected_ncaa_atom_to_hetatm(out_lines, protected_atoms)
        with open(args.output_pdb_path, 'w') as fh: fh.writelines(out_lines)

    # Post-processing: revert protected sidechain polar H names Q* -> H*
    if args.protect_sidechain_polarH is not None:
        print("[INFO] Reverting protected sidechain polar hydrogens Q* -> H* in final output.")
        with open(args.output_pdb_path, 'r') as fh: out_lines = fh.readlines()
        out_lines = revert_protected_sidechain_polarH(out_lines)
        with open(args.output_pdb_path, 'w') as fh: fh.writelines(out_lines)

    # Cleanup: delete precleaned PDB unless user wants to keep it for debugging
    if not args.keep_precleaned_pdb and os.path.exists(precleaned_input):
        try:
            os.remove(precleaned_input)
            print(f"[INFO] Deleted intermediate precleaned PDB: {precleaned_input}")
        except OSError as e:
            print(f"[WARN] Could not delete precleaned PDB '{precleaned_input}': {e}", file=sys.stderr)

    print("[INFO] All steps complete.")

if __name__ == '__main__':
    main()
