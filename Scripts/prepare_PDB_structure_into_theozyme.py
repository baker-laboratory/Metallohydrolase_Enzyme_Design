#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author:
    Seth M. Woodbury, David Baker Lab, University of Washington

Email:
    woodbuse@uw.edu

Date:
    2025-10-19

Script: prepare_PDB_structure_into_theozyme.py  (ALL-IN-ONE, simplified)

Purpose:
    Single entry-point that:
      • Cleans a PDB and generates REMARK 666 (former STEP1 behavior)
      • Optionally reorders REMARK 666 lines (former STEP2 behavior)
      • Supports optional non-combined ligand handling
    (Subset outputs have been removed by request.)

Key defaults (matching your originals):
      • Strip to REMARK/ATOM/HETATM
      • Remove H from ATOM (protein) only
      • Combine all HETATM into one unified ligand residue (unless --no_combine_ligands)
      • Generate REMARK 666 for each protein residue
      • Output order: REMARKs, new REMARK666s, ATOM block, TER, HETATM block, TER
      • Serial numbers: HETATM serials start at 1, ATOM serials start at n_het+1
      • Blank columns 72–75 in ATOM/HETATM

Usage:
    # COMBINED (default): requires ligand code
    python prepare_PDB_structure_into_theozyme.py \
        --input_pdb /path/to/input.pdb \
        --output_pdb_path /path/to/output.pdb \
        --ligand_complex_3_letter_name LIG \
        [--remark666_residue_front_order A244 A199] \
        [--remark666_residue_back_order A207 A143]

    # NON-COMBINED: do NOT pass a ligand code
    python prepare_PDB_structure_into_theozyme.py \
        --input_pdb /path/to/input.pdb \
        --output_pdb_path /path/to/output.pdb \
        --no_combine_ligands \
        [--remark666_residue_front_order A244 A199] \
        [--remark666_residue_back_order A207 A143]
"""

# ### IMPORTS ###
import argparse
import os
import re
import sys
from typing import List, Tuple, Dict, Set

# ### HELPERS ###

def blank_cols_72_75(line: str) -> str:
    """Blank columns 72-75 (0-based index 71-74) for ATOM/HETATM lines."""
    if len(line) >= 75:
        return line[:71] + '    ' + line[75:]
    return line

def is_atom(line: str) -> bool:
    return line.startswith("ATOM")

def is_hetatm(line: str) -> bool:
    return line.startswith("HETATM")

def is_remark(line: str) -> bool:
    return line.startswith("REMARK")

def atom_is_hydrogen(line: str) -> bool:
    # Uses atom name (cols 12-15)
    name = line[12:16].strip()
    return name.startswith('H')

def parse_chain_resnum(line: str) -> Tuple[str, int]:
    chain = line[21]
    num = int(line[22:26])
    return chain, num

def parse_resname(line: str) -> str:
    return line[17:20].strip()

def element_from_line(line: str) -> str:
    # Prefer element column (76:78); fallback to first letter of atom name (without digits)
    elem = line[76:78].strip()
    if elem:
        return elem
    name = re.sub(r"\d", "", line[12:16].strip())
    return name[0] if name else "X"

def build_unified_ligand(het_lines: List[str], unified_name: str,
                         out_chain='Z', out_resseq=999) -> List[str]:
    """Make one ligand residue out of all HETATM lines, assign atom names element+counter,
       renumber HETATM serials starting at 1, set chain/resseq as given."""
    new = []
    elem_counts: Dict[str, int] = {}
    for i, src in enumerate(het_lines, start=1):
        elem = element_from_line(src)
        elem_counts[elem] = elem_counts.get(elem, 0) + 1
        atom_name = f"{elem}{elem_counts[elem]}"
        rest = src[26:]  # keep xyz/occ/bfac/elem/charge
        out = f"HETATM{i:5d}  {atom_name:<4}{unified_name:>3} {out_chain}{out_resseq:>4}{rest}"
        new.append(blank_cols_72_75(out))
    return new

def renumber_atom_block(atom_lines: List[str], start_serial: int) -> List[str]:
    """Renumber ATOM serials, keeping everything after serial column intact."""
    out = []
    for j, l in enumerate(atom_lines, start=start_serial):
        line_out = f"ATOM  {j:5d}" + l[11:]
        out.append(blank_cols_72_75(line_out))
    return out

def collect_protein_residues(atom_lines: List[str]) -> List[Tuple[str, str, int]]:
    """Unique protein residues as (chain, resname, resnum), sorted by chain then resnum."""
    residues: Set[Tuple[str, str, int]] = set()
    for l in atom_lines:
        ch = l[21]
        rn = parse_resname(l)
        num = int(l[22:26])
        residues.add((ch, rn, num))
    return sorted(residues, key=lambda x: (x[0], x[2]))

def make_remark666_for_residues(residues: List[Tuple[str, str, int]],
                                unified_ligand_name: str) -> List[str]:
    """Create REMARK 666 lines for given protein residues."""
    out = []
    for idx, (chain, res, num) in enumerate(residues, start=1):
        line = (f"REMARK 666 MATCH TEMPLATE X {unified_ligand_name:<3}    0 MATCH MOTIF "
                f"{chain} {res:<3} {num:>4}{idx:>4}{1:>3}\n")
        out.append(line)
    return out

def reorder_remark666_block(lines: List[str],
                            front: List[str],
                            back: List[str]) -> List[str]:
    """
    Reorder a contiguous REMARK 666 block according to:
       front (in given order), then middle sorted by (chain, num), then reversed(back).

    Input:
      lines = only the REMARK 666 lines (strings)
      front/back elements are ChainResidue strings like 'A244'
    """
    # Parse current entries
    items = []
    for line in lines:
        toks = line.split()
        # ['REMARK','666','MATCH','TEMPLATE','X',ligand,'0','MATCH','MOTIF',chain,res_name,seq,idx,last]
        ligand = toks[5]
        chain  = toks[9]
        res    = toks[10]
        num    = int(toks[11])
        items.append({'chain': chain, 'res': res, 'num': num, 'ligand': ligand})

    all_keys = {(it['chain'], it['num']) for it in items}
    res_map  = {(it['chain'], it['num']): it['res'] for it in items}
    ligand   = items[0]['ligand'] if items else "LIG"

    def parse_key(s: str) -> Tuple[str, int]:
        return (s[0], int(s[1:]))

    front_keys = [parse_key(s) for s in front]
    back_keys  = [parse_key(s) for s in back]

    # Filter to keys that actually exist; warn if missing
    missing = [k for k in front_keys + back_keys if k not in all_keys]
    for ch, rn in missing:
        print(f"[WARN] REMARK 666 for {ch}{rn} not found; skipping in reorder", file=sys.stderr)
    front_keys = [k for k in front_keys if k in all_keys]
    back_keys  = [k for k in back_keys if k in all_keys]

    middle_keys = sorted([k for k in all_keys if k not in front_keys and k not in back_keys],
                         key=lambda x: (x[0], x[1]))
    final_keys = front_keys + middle_keys + list(reversed(back_keys))

    new_remark = []
    for idx, (chain, num) in enumerate(final_keys, start=1):
        res = res_map[(chain, num)]
        line = (f"REMARK 666 MATCH TEMPLATE X {ligand:<3}    0 MATCH MOTIF "
                f"{chain} {res:<3} {num:>4}{idx:>4}{1:>3}\n")
        new_remark.append(line)
    return new_remark

def split_prefix_remark_suffix(all_lines: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """Split file into prefix (before the first REMARK 666), the REMARK 666 block, and suffix."""
    prefix, remark_block, suffix = [], [], []
    state = "prefix"
    for raw in all_lines:
        if state == "prefix":
            if raw.startswith("REMARK 666"):
                state = "remark"
                remark_block.append(raw)
            else:
                prefix.append(raw)
        elif state == "remark":
            if raw.startswith("REMARK 666"):
                remark_block.append(raw)
            else:
                state = "suffix"
                suffix.append(raw)
        else:
            suffix.append(raw)
    return prefix, remark_block, suffix

def write_cleaned_pdb(out_path: str,
                      orig_remarks: List[str],
                      new_remark666: List[str],
                      atom_block: List[str],
                      het_block: List[str]) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as out:
        for r in orig_remarks:
            out.write(r)
        for r in new_remark666:
            out.write(r)
        for l in atom_block:
            out.write(l)
        out.write("TER\n")
        for h in het_block:
            out.write(h)
        out.write("TER\n")
    print(f"[INFO] Wrote PDB: {out_path}")

# ### CORE PIPELINE ###

def clean_and_prepare(input_pdb: str,
                      output_pdb_path: str,
                      ligand_complex_3_letter_name: str,
                      no_combine_ligands: bool,
                      remark666_front: List[str],
                      remark666_back: List[str]) -> None:
    """
    Implements (former STEP1) + (former STEP2).
    """

    # Read & filter input
    atom_lines: List[str] = []
    het_lines: List[str]  = []
    remark_lines: List[str] = []

    with open(input_pdb, "r") as f:
        for raw in f:
            if is_atom(raw):
                if not atom_is_hydrogen(raw):   # remove H from protein
                    atom_lines.append(raw)
            elif is_hetatm(raw):
                het_lines.append(raw)
            elif is_remark(raw):
                remark_lines.append(raw)
            else:
                # drop everything else (including TER) on purpose
                continue

    # Ligand handling
    if no_combine_ligands:
        # Keep original HETATM residues (but still blank cols 72-75 and renumber serials)
        het_block = []
        for i, src in enumerate(het_lines, start=1):
            out = f"HETATM{i:5d}" + src[11:]
            het_block.append(blank_cols_72_75(out))
        unified_name_for_remarks = "LIG"  # placeholder tag in REMARK text when not unifying
        n_het = len(het_block)
    else:
        # Combine all het into a single 'unified ligand' (requires ligand name)
        het_block = build_unified_ligand(
            het_lines, ligand_complex_3_letter_name, out_chain='Z', out_resseq=999
        )
        unified_name_for_remarks = ligand_complex_3_letter_name
        n_het = len(het_block)

    # REMARK 666 (protein residues only)
    residues = collect_protein_residues(atom_lines)
    remark666 = make_remark666_for_residues(residues, unified_name_for_remarks)

    # Renumber ATOM serials (ATOM start at n_het+1)
    atom_block = renumber_atom_block(atom_lines, start_serial=n_het + 1)

    # Write initial cleaned PDB
    write_cleaned_pdb(output_pdb_path, remark_lines, remark666, atom_block, het_block)

    # Reorder REMARK 666 (former STEP2) on the main output
    with open(output_pdb_path, "r") as f:
        all_lines = f.readlines()

    prefix, remark_block, suffix = split_prefix_remark_suffix(all_lines)
    if remark_block:
        new_remark_block = reorder_remark666_block(
            remark_block, front=remark666_front, back=remark666_back
        )
        with open(output_pdb_path, "w") as out:
            for l in prefix:
                out.write(l)
            for l in new_remark_block:
                out.write(l)
            for l in suffix:
                out.write(l)
        print(f"[INFO] Reordered {len(new_remark_block)} REMARK 666 lines in {output_pdb_path}")
    else:
        print("[WARN] No REMARK 666 lines found to reorder in main output.", file=sys.stderr)

    print("[INFO] All steps complete.")

# ### CLI ###

def parse_args():
    p = argparse.ArgumentParser(
        description="Clean a PDB, generate/reorder REMARK 666, with optional non-combined ligand handling."
    )
    p.add_argument('--input_pdb', required=True, help='Path to the input PDB file')
    p.add_argument('--output_pdb_path', required=True, help='Destination path for the cleaned PDB file')
    p.add_argument('--ligand_complex_3_letter_name', required=False, default=None,
                   help='Three-letter code for the unified ligand when combining HETATM entries')
    p.add_argument('--no_combine_ligands', action='store_true',
                   help='Keep original HETATM residues (do NOT unify ligands). If set, do not pass a ligand code.')
    p.add_argument('--remark666_residue_front_order', nargs='*', default=[],
                   help='ChainResidue tokens to force to the front, e.g., A244 A199')
    p.add_argument('--remark666_residue_back_order', nargs='*', default=[],
                   help='ChainResidue tokens to force to the back, e.g., A207 A143')
    return p.parse_args()

def main():
    args = parse_args()

    # --- Validation of combine vs no_combine + ligand code ---
    if args.no_combine_ligands:
        # User explicitly requested NOT to unify ligands → ligand name should NOT be provided
        if args.ligand_complex_3_letter_name:
            print("[ERROR] --no_combine_ligands was specified, but a ligand code "
                  f"('{args.ligand_complex_3_letter_name}') was also provided. "
                  "Remove --ligand_complex_3_letter_name or omit it.", file=sys.stderr)
            sys.exit(2)
        ligand_name = None
    else:
        # User wants unified ligand behavior → ligand code is REQUIRED
        if not args.ligand_complex_3_letter_name:
            print("[ERROR] --ligand_complex_3_letter_name is required when not using --no_combine_ligands.",
                  file=sys.stderr)
            sys.exit(2)
        ligand_name = args.ligand_complex_3_letter_name

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output_pdb_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    clean_and_prepare(
        input_pdb=args.input_pdb,
        output_pdb_path=args.output_pdb_path,
        ligand_complex_3_letter_name=(ligand_name if ligand_name else ""),
        no_combine_ligands=args.no_combine_ligands,
        remark666_front=args.remark666_residue_front_order,
        remark666_back=args.remark666_residue_back_order
    )

if __name__ == '__main__':
    main()
