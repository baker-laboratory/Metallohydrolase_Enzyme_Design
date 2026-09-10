#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author:
    Seth M. Woodbury, David Baker Lab, University of Washington

Email:
    woodbuse@uw.edu

Date:
    2025-05-14

Script: prepare_PDB_structure_into_theozyme__STEP1__cleanPDB_and_addREMARK666.py

Purpose:
    - Intake --input_pdb, --output_pdb_path, and --ligand_complex_3_letter_name
    - Strip all lines except ATOM, HETATM, REMARK
    - Remove all hydrogen atoms from protein residues (ATOM records)
    - Combine all HETATM entries into a single ligand residue:
        * Residue name = ligand_complex_3_letter_name
    - Assign unique atom names for HETATM entries (element symbol + counter)
    - Generate REMARK 666 lines for each protein residue (not ligand) in the format:
      REMARK 666 MATCH TEMPLATE X LIG    0 MATCH MOTIF C RES   N IDX I 1
    - Output ordering: REMARKs, new REMARK666s, ATOM block, TER, HETATM block, TER
    - Atom numbering: HETATM serials start at 1, ATOM serials start at n_het+1
    - Blank any characters in cols 72-75 of all ATOM/HETATM lines
"""

import argparse
import re


def parse_args():
    p = argparse.ArgumentParser(description="Clean and prepare a PDB for theozyme analysis")
    p.add_argument('--input_pdb',       required=True, help="Path to the input PDB file")
    p.add_argument('--output_pdb_path', required=True, help="Path for the cleaned PDB output")
    p.add_argument('--ligand_complex_3_letter_name', required=True,
                   help="Three-letter code to unify all HETATM entries under")
    return p.parse_args()


def blank_redundant_chain(line):
    # Blank columns 72-75 (0-based 71-74)
    if len(line) >= 75:
        return line[:71] + '    ' + line[75:]
    return line


def main():
    args = parse_args()

    atom_re   = re.compile(r'^ATOM')
    het_re    = re.compile(r'^HETATM')
    remark_re = re.compile(r'^REMARK')

    atom_lines   = []
    het_lines    = []
    remark_lines = []

    # Read and filter
    with open(args.input_pdb) as f:
        for raw in f:
            if atom_re.match(raw):
                name = raw[12:16].strip()
                if not name.startswith('H'):
                    atom_lines.append(raw)
            elif het_re.match(raw):
                het_lines.append(raw)
            elif remark_re.match(raw):
                remark_lines.append(raw)
            else:
                # drop TER and others
                continue

    # Build unified ligand HETATM block
    new_het = []
    elem_counts = {}
    for i, line in enumerate(het_lines, start=1):
        element = line[76:78].strip() or re.sub(r"\d", "", line[12:16].strip())[0]
        c = elem_counts.get(element, 0) + 1
        elem_counts[element] = c
        atom_name = f"{element}{c}"
        # Keep everything from col 27 onwards, then blank segID region
        rest = line[26:]
        out_line = (
            f"HETATM{i:5d}  {atom_name:<4}{args.ligand_complex_3_letter_name:>3} Z {999:>3}" + rest
        )
        out_line = blank_redundant_chain(out_line)
        new_het.append(out_line)
    n_het = len(new_het)

    # Generate REMARK 666 entries for protein residues only
    residues = set()
    for l in atom_lines:
        chain = l[21]
        res   = l[17:20].strip()
        num   = int(l[22:26])
        residues.add((chain, res, num))

    new_remarks = []
    idx = 1
    for chain, res, num in sorted(residues, key=lambda x: (x[0], x[2])):
        remark = (
            f"REMARK 666 MATCH TEMPLATE X {args.ligand_complex_3_letter_name:<3}    0 MATCH MOTIF "
            f"{chain} {res:<3} {num:>4}{idx:>4}{1:>3}\n"
        )
        new_remarks.append(remark)
        idx += 1

    # Write cleaned PDB
    with open(args.output_pdb_path, 'w') as out:
        # Original REMARKs
        for r in remark_lines:
            out.write(r)
        # New REMARK 666s
        for r in new_remarks:
            out.write(r)
        # ATOM block with renumbered serials
        for j, l in enumerate(atom_lines, start=n_het+1):
            line_out = f"ATOM  {j:5d}" + l[11:]
            line_out = blank_redundant_chain(line_out)
            out.write(line_out)
        out.write("TER\n")
        # HETATM block
        for h in new_het:
            out.write(h)
        out.write("TER\n")

    print(f"[INFO] Cleaned PDB written to {args.output_pdb_path}")

if __name__ == '__main__':
    main()
