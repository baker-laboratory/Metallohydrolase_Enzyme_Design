#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author:
    Seth M. Woodbury, David Baker Lab, University of Washington

Email:
    woodbuse@uw.edu

Date:
    2025-05-14

Script: prepare_PDB_structure_into_theozyme__STEP2__reorder_REMARK666_lines.py

Purpose:
    - Intake --input_pdb and optional --remark666_residue_front_order and --remark666_residue_back_order
    - Reorder REMARK 666 lines according to front_order, then sorted middle, then back_order reversed
    - Update their index (second-to-last column)
    - Write changes in place (overwrite input_pdb)

Usage:
    python prepare_PDB_structure_into_theozyme__STEP2__reorder_REMARK666_lines.py \
        --input_pdb path/to/cleaned.pdb \
        [--remark666_residue_front_order A244 A199] \
        [--remark666_residue_back_order A207 A143]
"""

import argparse
import sys


def parse_args():
    p = argparse.ArgumentParser(description="Reorder REMARK 666 lines and update indices")
    p.add_argument('--input_pdb', required=True,
                   help="Path to the PDB file with REMARK 666 lines")
    p.add_argument('--remark666_residue_front_order', nargs='*', default=[],
                   help="Optional list of ChainResidue entries (e.g. A57) to place first in this order")
    p.add_argument('--remark666_residue_back_order', nargs='*', default=[],
                   help="Optional list of ChainResidue entries (e.g. A207) to place last in this order")
    return p.parse_args()


def main():
    args = parse_args()

    # Read lines and segment REMARK 666 block
    with open(args.input_pdb) as f:
        lines = f.readlines()

    prefix = []
    remark_lines = []
    suffix = []
    state = 'prefix'
    for raw in lines:
        if state == 'prefix':
            if raw.startswith('REMARK 666'):
                state = 'remark'
                remark_lines.append(raw)
            else:
                prefix.append(raw)
        elif state == 'remark':
            if raw.startswith('REMARK 666'):
                remark_lines.append(raw)
            else:
                state = 'suffix'
                suffix.append(raw)
        else:
            suffix.append(raw)

    if not remark_lines:
        print("[ERROR] No REMARK 666 lines found in file.", file=sys.stderr)
        sys.exit(1)

    # Parse existing remark entries
    items = []  # list of dicts with chain, res, num, orig_line
    for line in remark_lines:
        tokens = line.split()
        # tokens: ['REMARK','666','MATCH','TEMPLATE','X',ligand,'0','MATCH','MOTIF',chain,res_name,seq,idx,last]
        ligand = tokens[5]
        chain  = tokens[9]
        res    = tokens[10]
        num    = int(tokens[11])
        items.append({'chain':chain, 'res':res, 'num':num, 'ligand':ligand})

    # All unique keys
    all_keys = [(it['chain'], it['num']) for it in items]
    res_map  = { (it['chain'], it['num']): it['res'] for it in items }
    ligand   = items[0]['ligand']

    # Parse front/back orders
    def parse_chain_res(s):
        return (s[0], int(s[1:]))

    front_keys = [parse_chain_res(s) for s in args.remark666_residue_front_order]
    back_keys  = [parse_chain_res(s) for s in args.remark666_residue_back_order]

    # Validate
    for k in front_keys + back_keys:
        if k not in all_keys:
            print(f"[ERROR] REMARK for {k[0]}{k[1]} not found", file=sys.stderr)
            sys.exit(1)

    # Compute middle keys
    middle_keys = sorted(
        [k for k in all_keys if k not in front_keys and k not in back_keys],
        key=lambda x: (x[0], x[1])
    )

    # Final ordering: front_keys, middle_keys, reversed(back_keys)
    final_keys = front_keys + middle_keys + list(reversed(back_keys))

    # Reconstruct remarks with new indices
    new_remarks = []
    for idx, (chain, num) in enumerate(final_keys, start=1):
        res = res_map[(chain, num)]
        line = (
            f"REMARK 666 MATCH TEMPLATE X {ligand:<3}    0 MATCH MOTIF "
            f"{chain} {res:<3} {num:>4}{idx:>4}{1:>3}\n"
        )
        new_remarks.append(line)

    # Write back: prefix, new remarks, suffix
    with open(args.input_pdb, 'w') as out:
        for l in prefix:
            out.write(l)
        for l in new_remarks:
            out.write(l)
        for l in suffix:
            out.write(l)

    print(f"[INFO] Reordered {len(new_remarks)} REMARK 666 lines in {args.input_pdb}")

if __name__ == '__main__':
    main()
