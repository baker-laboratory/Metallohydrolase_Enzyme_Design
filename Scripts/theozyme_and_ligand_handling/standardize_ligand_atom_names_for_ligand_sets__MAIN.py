#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author:
    Seth M. Woodbury, David Baker Lab, University of Washington

Email:
    woodbuse@uw.edu

Date:
    2025-05-xx

Script: standardize_ligand_atom_names_for_ligand_sets__MAIN.py

Purpose:
    Use a reference ligand in a reference PDB to standardize ligand atom names
    across a set of PDBs. The script:

      1) Identifies a ligand in the reference PDB (optionally via --ref_ligand).
      2) Uses a user-specified anchor atom mapping (REFNAME:TARGETNAME pairs)
         to rigid-body align the target ligand to the reference ligand.
      3) Renames target ligand atoms to match reference ligand atom names,
         preserving element types and using nearest-neighbor matching in the
         aligned frame for all non-anchor atoms.
      4) Leaves any "extra" target ligand atoms (not present in the reference)
         unchanged (deterministic and stable; extendable later).

    Only ligand atom *names* are changed; all other PDB fields are kept identical.
    Output can be written in-place or to a new directory.

Usage:
    python standardize_ligand_atom_names_for_ligand_sets__MAIN.py \
        --ref_pdb ref.pdb \
        --anchor_map P1:P1 C1:C2 N1:N1 \
        --pdbs_to_standardize_ligs "/path/to/dir/*.pdb" \
        [--ref_ligand YYE] \
        [--target_ligand YYE] \
        [--output_dir /path/to/output] \
        [--dry_run]

Notes:
    - The reference ligand is assumed to be a subset of (and similarly
      conformational to) the ligands in the target PDBs.
    - If multiple ligand resnames are found and --ref_ligand/--target_ligand
      is not specified, the script will abort and print a helpful message.
    - Anchor pairs must map atoms of the same element (e.g. O1:O3, N1:N1).
"""

import os, sys, glob, argparse, math
from collections import defaultdict

try:
    import numpy as np
except ImportError:
    print("[ERROR] This script requires numpy. Please install it or use a container with numpy available.", file=sys.stderr)
    sys.exit(1)

###############################################################################
# ARGPARSE
###############################################################################

def parse_args():
    p = argparse.ArgumentParser(description="Standardize ligand atom names across PDBs using a reference ligand.")
    p.add_argument('--ref_pdb', required=True, help='Reference PDB containing the template ligand.')
    p.add_argument('--ref_ligand', help='3-letter resname of reference ligand (needed if multiple ligands present).')
    p.add_argument('--pdbs_to_standardize_ligs', required=True,
                   help='PDB path or glob pattern for PDBs whose ligand atom names should be standardized.')
    p.add_argument('--target_ligand', help='3-letter resname of target ligand in the PDBs (if not provided, inferred).')
    p.add_argument('--anchor_map', nargs='+', required=True,
                   help='List of anchor mappings REFNAME:TARGETNAME (e.g. P1:P1 C1:C2 N1:N1).')
    p.add_argument('--output_dir', help='If set, write edited PDBs here and keep originals untouched.')
    p.add_argument('--dry_run', action='store_true', help='If set, do not write any files; just print mappings.')
    p.add_argument('--verbose', action='store_true', help='If set, print more detailed debug info.')
    return p.parse_args()

###############################################################################
# BASIC PDB UTILITIES
###############################################################################

def is_atom_line(line):  return line.startswith("ATOM  ") or line.startswith("HETATM")
def get_resname(line):   return line[17:20].strip()
def get_chain(line):     return line[21]
def get_resseq(line):    return line[22:26].strip()
def get_icode(line):     return (line[26].strip() or " ")
def get_atom_name(line): return line[12:16].strip()
def get_element(line, atom_name=None):
    elem = line[76:78].strip() if len(line) >= 78 else ""
    if not elem and atom_name:
        elem = atom_name[0].upper()
    return elem

def get_coords(line):
    return np.array([
        float(line[30:38]),
        float(line[38:46]),
        float(line[46:54]),
    ], dtype=float)

def replace_field(line, start, end, new):
    if len(new) != (end - start):
        raise ValueError(f"replace_field length mismatch: wanted {end-start}, got {len(new)}")
    return line[:start] + new + line[end:]

###############################################################################
# LIGAND EXTRACTION
###############################################################################

def find_ligand_residues(lines, lig_resname=None, label="(unspecified)", verbose=False):
    """
    Return:
      lig_key, atoms

    lig_key  = (resname, chain, resseq, icode) of *single* ligand residue
    atoms    = list of (idx, line) for that residue

    If lig_resname is None:
      - Collect all unique ligand resnames among HETATM (excluding HOH/WAT).
      - If exactly one, use that; else raise with message.
    """
    # Collect all HETATM ligand residues
    lig_residues = defaultdict(list)  # key: (resname, chain, resseq, icode) -> list[(idx,line)]
    het_resnames = set()
    for idx, line in enumerate(lines):
        if not line.startswith("HETATM"): continue
        resname = get_resname(line)
        if resname in ("HOH", "WAT"): continue
        chain, resseq, icode = get_chain(line), get_resseq(line), get_icode(line)
        key = (resname, chain, resseq, icode)
        lig_residues[key].append((idx, line))
        het_resnames.add(resname)

    if not lig_residues:
        raise ValueError(f"No HETATM ligand residues found in {label} PDB.")

    if lig_resname is None:
        if len(het_resnames) == 1:
            lig_resname = next(iter(het_resnames))
            if verbose:
                print(f"[INFO] {label}: inferred ligand resname = {lig_resname}")
        else:
            raise ValueError(f"{label}: Multiple ligand resnames found ({sorted(het_resnames)}); "
                             f"please specify --{'ref_ligand' if label=='reference' else 'target_ligand'}.")

    # Filter residues by chosen resname
    keys_for_resname = [k for k in lig_residues if k[0] == lig_resname]
    if not keys_for_resname:
        raise ValueError(f"{label}: No residues with resname '{lig_resname}' found.")

    if len(keys_for_resname) > 1:
        raise ValueError(f"{label}: Multiple residues with resname '{lig_resname}' found: {keys_for_resname}. "
                         f"Please edit script or pdb to disambiguate (currently requires one).")

    lig_key = keys_for_resname[0]
    if verbose:
        print(f"[INFO] {label}: using ligand {lig_key}")
    return lig_key, lig_residues[lig_key]

def build_ligand_atom_table(atoms):
    """
    atoms: list of (idx, line) for a single residue
    Return dict:
      name -> { 'idx': idx, 'line': line, 'coord': np.array(3), 'elem': str }
    """
    table = {}
    for idx, line in atoms:
        atom_name = get_atom_name(line)
        coord     = get_coords(line)
        elem      = get_element(line, atom_name=atom_name)
        table[atom_name] = {"idx": idx, "line": line, "coord": coord, "elem": elem}
    return table

###############################################################################
# ALIGNMENT (KABSCH) & ANCHOR HANDLING
###############################################################################

def parse_anchor_map(anchor_specs):
    """
    anchor_specs: list like ["P1:P1", "C1:C2", "N1:N1"]
    Return list of (ref_name, target_name).
    """
    anchors = []
    for spec in anchor_specs:
        if ":" not in spec:
            raise ValueError(f"Invalid anchor mapping '{spec}' (expected REF:TARGET).")
        ref_name, tgt_name = spec.split(":", 1)
        ref_name, tgt_name = ref_name.strip(), tgt_name.strip()
        if not ref_name or not tgt_name:
            raise ValueError(f"Invalid anchor mapping '{spec}' (empty ref or target).")
        anchors.append((ref_name, tgt_name))
    return anchors

def kabsch(P, Q):
    """
    Compute rotation R, translation t that best aligns Q -> P (minimizing RMSD).
    P, Q: (N,3) arrays
    Returns R (3x3), t (3-vector)
    """
    if P.shape != Q.shape:
        raise ValueError("Kabsch: P and Q must have same shape.")
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("Kabsch: P,Q must be (N,3) arrays.")

    Pc = P.mean(axis=0)
    Qc = Q.mean(axis=0)
    P0 = P - Pc
    Q0 = Q - Qc

    C = np.dot(Q0.T, P0)  # note Q->P
    V, S, Wt = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(Wt))
    if d < 0.0:
        V[:, -1] *= -1.0
    R = np.dot(V, Wt)
    t = Pc - np.dot(R, Qc)
    return R, t

def build_full_mapping(ref_table, tgt_table, anchors, verbose=False):
    """
    ref_table, tgt_table: name->info dicts from build_ligand_atom_table
    anchors: list of (ref_name, tgt_name)

    Returns:
      name_map: dict { target_atom_name -> new_target_atom_name }
    """
    # Validate anchors and gather coords
    P_list, Q_list = [], []
    anchor_pairs   = []
    for ref_name, tgt_name in anchors:
        if ref_name not in ref_table:
            raise KeyError(f"Anchor ref atom '{ref_name}' not found in reference ligand.")
        if tgt_name not in tgt_table:
            raise KeyError(f"Anchor target atom '{tgt_name}' not found in target ligand.")

        ref_info = ref_table[ref_name]
        tgt_info = tgt_table[tgt_name]

        if ref_info["elem"] != tgt_info["elem"]:
            raise ValueError(f"Element mismatch for anchor {ref_name}:{tgt_name} "
                             f"({ref_info['elem']} vs {tgt_info['elem']}).")

        P_list.append(ref_info["coord"])
        Q_list.append(tgt_info["coord"])
        anchor_pairs.append((ref_name, tgt_name))

    P = np.vstack(P_list)
    Q = np.vstack(Q_list)
    R, t = kabsch(P, Q)

    # Transform all target ligand coords into ref frame
    tgt_coords_aligned = {}
    for name, info in tgt_table.items():
        tgt_coords_aligned[name] = R.dot(info["coord"]) + t

    # Start mapping with anchors: target_name -> ref_name
    name_map = {}
    used_ref = set()
    used_tgt = set()

    for ref_name, tgt_name in anchor_pairs:
        name_map[tgt_name] = ref_name
        used_ref.add(ref_name)
        used_tgt.add(tgt_name)

    # For each *remaining* reference atom, assign nearest remaining target atom of same element
    for ref_name, ref_info in ref_table.items():
        if ref_name in used_ref: continue
        elem_ref = ref_info["elem"]
        ref_coord = ref_info["coord"]

        best_tgt_name, best_d2 = None, None
        for tgt_name, tgt_info in tgt_table.items():
            if tgt_name in used_tgt: continue
            if tgt_info["elem"] != elem_ref: continue
            coord_aligned = tgt_coords_aligned[tgt_name]
            d2 = float(np.sum((coord_aligned - ref_coord)**2))
            if best_d2 is None or d2 < best_d2:
                best_d2, best_tgt_name = d2, tgt_name

        if best_tgt_name is None:
            print(f"[WARN] No suitable target atom found for reference atom '{ref_name}' (elem {elem_ref}); skipping.", file=sys.stderr)
            continue

        name_map[best_tgt_name] = ref_name
        used_ref.add(ref_name)
        used_tgt.add(best_tgt_name)

    # ----------------------------------------------------------------------
    # 3) Finalize mapping for any "extra" target atoms (not in reference)
    #    Requirement:
    #      - 1:1 final names (no duplicates)
    #      - For each element, continue numbering from the reference set:
    #          e.g., ref has O1..O5 -> extras become O6, O7, ...
    # ----------------------------------------------------------------------
    used_new_names = set(name_map.values())  # names already assigned (anchors + NN)
    next_index = {}  # elem_letter -> next integer to use (e.g., 'O' -> 6)

    def _update_next_index_from_name(atom_name, elem_letter):
        """Update next_index[elem_letter] from an atom name like O1, C10, H5, etc."""
        s = atom_name.strip()
        if not s or s[0].upper() != elem_letter.upper():
            return
        # Grab any digits after the first character
        digits = "".join(ch for ch in s[1:] if ch.isdigit())
        if not digits:
            return
        idx = int(digits)
        current = next_index.get(elem_letter, 1)
        if idx + 1 > current:
            next_index[elem_letter] = idx + 1

    # Seed next_index from reference atom names
    for rname, rinfo in ref_table.items():
        elem = rinfo["elem"] or (rname[0].upper() if rname else "X")
        elem_letter = elem[0].upper()
        _update_next_index_from_name(rname, elem_letter)

    # Also seed from already-assigned new names (in case any are beyond ref)
    for new_name in used_new_names:
        # If this name is itself a reference name, we already handled it above.
        if new_name in ref_table:
            elem = ref_table[new_name]["elem"]
            elem_letter = (elem[0].upper() if elem else (new_name[0].upper() if new_name else "X"))
        else:
            elem_letter = new_name[0].upper() if new_name else "X"
        _update_next_index_from_name(new_name, elem_letter)

    # Now assign names for extra (non-reference) target atoms
    for tgt_name, tgt_info in tgt_table.items():
        if tgt_name in name_map:
            continue  # already assigned via anchors + NN

        elem = tgt_info["elem"] or (tgt_name[0].upper() if tgt_name else "X")
        elem_letter = elem[0].upper()

        # Start counting from whatever we have (or 1 if nothing exists yet)
        idx = next_index.get(elem_letter, 1)

        while True:
            candidate = f"{elem_letter}{idx}"
            # Candidate must not collide with any already-used new name
            if candidate not in used_new_names:
                new_name = candidate
                break
            idx += 1

        # Record next available index for this element
        next_index[elem_letter] = idx + 1

        name_map[tgt_name] = new_name
        used_new_names.add(new_name)

    # Final sanity: enforce 1:1 mapping on final names
    if len(set(name_map.values())) != len(name_map.values()):
        raise RuntimeError("Non-1:1 mapping detected: multiple target atoms share the same final name.")

    if verbose:
        print("[INFO] Final target->reference name mapping:")
        for tgt_name in sorted(name_map.keys()):
            ref_name = name_map[tgt_name]
            mark = "" if tgt_name == ref_name else " (renamed)"
            print(f"       {tgt_name:>4s} -> {ref_name:>4s}{mark}")
    return name_map

###############################################################################
# APPLY MAPPING TO PDB LINES
###############################################################################

def apply_name_mapping_to_pdb(lines, lig_key, name_map, verbose=False):
    """
    lines   : full PDB file lines
    lig_key: (resname, chain, resseq, icode) of target ligand residue
    name_map: dict { old_name -> new_name }

    Returns new_lines with only ligand atom names changed.
    """
    resname, chain, resseq, icode = lig_key
    new_lines = []
    for idx, line in enumerate(lines):
        if is_atom_line(line) and get_resname(line) == resname and \
           get_chain(line) == chain and get_resseq(line) == resseq and get_icode(line) == icode:
            old_name = get_atom_name(line)
            if old_name in name_map:
                new_name = name_map[old_name][:4].rjust(4)
                if new_name != old_name:
                    if verbose:
                        print(f"[DEBUG] Line {idx}: {old_name} -> {new_name}")
                    line = replace_field(line, 12, 16, new_name)
        new_lines.append(line)
    return new_lines

###############################################################################
# MAIN
###############################################################################

def main():
    args = parse_args()

    # Resolve PDB list
    pdb_paths = glob.glob(args.pdbs_to_standardize_ligs)
    if not pdb_paths:
        print(f"[ERROR] No PDBs matched pattern '{args.pdbs_to_standardize_ligs}'.", file=sys.stderr)
        sys.exit(1)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # Parse reference PDB
    with open(args.ref_pdb, 'r') as fh:
        ref_lines = fh.readlines()

    ref_lig_key, ref_atoms = find_ligand_residues(ref_lines, lig_resname=args.ref_ligand,
                                                  label="reference", verbose=args.verbose)
    ref_table = build_ligand_atom_table(ref_atoms)

    # Parse anchor map
    anchors = parse_anchor_map(args.anchor_map)
    if args.verbose:
        print("[INFO] Anchor mappings (REF:TARGET):", ", ".join([f"{r}:{t}" for r, t in anchors]))

    print(f"[INFO] Reference ligand: {ref_lig_key} in {args.ref_pdb}")
    print(f"[INFO] Standardizing {len(pdb_paths)} PDB(s)...\n")

    for pdb_path in sorted(pdb_paths):
        print(f"[INFO] Processing PDB: {pdb_path}")
        with open(pdb_path, 'r') as fh:
            lines = fh.readlines()

        # Identify target ligand residue
        try:
            tgt_lig_key, tgt_atoms = find_ligand_residues(lines, lig_resname=args.target_ligand,
                                                          label="target", verbose=args.verbose)
        except ValueError as e:
            print(f"[WARN] Skipping {pdb_path}: {e}", file=sys.stderr)
            continue

        tgt_table = build_ligand_atom_table(tgt_atoms)

        # Build name mapping for this PDB
        try:
            name_map = build_full_mapping(ref_table, tgt_table, anchors, verbose=args.verbose)
        except Exception as e:
            print(f"[WARN] Failed to build mapping for {pdb_path}: {e}", file=sys.stderr)
            continue

        # Apply mapping
        new_lines = apply_name_mapping_to_pdb(lines, tgt_lig_key, name_map, verbose=args.verbose)

        # Decide output path
        if args.output_dir:
            base = os.path.basename(pdb_path)
            out_path = os.path.join(args.output_dir, base)
        else:
            out_path = pdb_path  # in-place overwrite

        if args.dry_run:
            print(f"[INFO] Dry-run: would write standardized PDB to: {out_path}\n")
        else:
            with open(out_path, 'w') as fh:
                fh.writelines(new_lines)
            print(f"[INFO] Wrote standardized PDB to: {out_path}\n")

    print("[INFO] All done.")

if __name__ == "__main__":
    main()
