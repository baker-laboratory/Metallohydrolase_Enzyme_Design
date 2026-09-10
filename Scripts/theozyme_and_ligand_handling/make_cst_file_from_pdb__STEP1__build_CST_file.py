#!/usr/bin/env python3
"""
Author: Seth M. Woodbury (University of Washington)
Date: 2025-05-09

Adapted from Anna Lauko

CST File Generator for Rosetta

This script automates the creation of parameterized Rosetta constraint (.cst) files
from a given PDB structure and user-defined residue-pair specifications. It measures
key geometric features between two sets of three atoms (distance, two angles, three
dihederals), formats them into a standardized Rosetta block with optional custom
tolerances, and writes an ordered CST file. Additional features:

1. **Automatic 3-letter code detection**: If the three-letter residue code is omitted,
it is inferred from the PDB coordinates.

2. **REMARK 666 reordering**: Parses `REMARK 666` lines in the PDB to reorder blocks
according to the mapped index (useful for motif/template matching).

3. **Custom tolerances**: Per-constraint `dist_tol` (Å) and `ang_tol` (°) can be provided;
defaults are 0.02 Å and 5.00°.

4. **Enhanced formatting**: Generates each block with headers, `VARIABLE_CST`,
`CST::BEGIN`, `ALGORITHM_INFO` sections, and fixed formatting:
   - Distances to four decimal places
   - Angles/dihedrals to two decimal places
   - Distance constraint weight fixed at 150
   - Optionally includes `SECONDARY_MATCH` for non-primary constraints

5. **Verbose logging**: A `--verbose` flag outputs debug info to stderr, including PDB
residue count, REMARK mapping entries, and per-block tolerance settings.

**Inputs**:
- `--input_pdb`  : Path to the input PDB file.
- `--output_cst` : Desired output `.cst` filename.
- `--specs`      : JSON string or file path specifying a list of constraint specs.

Each specification dict must contain:
json
{
  "residue_1_identifier__ChainResidue": "A92",  # Chain+residue (e.g. A92)
  "residue_1_atoms": ["CA","CB","CG"],      # List of exactly 3 atom names
  "residue_2_identifier__ChainResidue": "B10",
  "residue_2_atoms": ["N","CA","C"],
  "primary": true,                               # Boolean: primary constraint
  "covalent": false                             # Boolean: treat distance as covalent
  // Optional overrides:
  "dist_tol": 0.05,      # Distance tolerance in Å
  "ang_tol": 3.0         # Angle tolerance in degrees
}


**Example command**:
python make_cst_file_from_pdb.py \
  --input_pdb path/to/structure.pdb \
  --output_cst path/to/output.cst \
  --specs '[
    {"residue_1_identifier__ChainResidue":"Z9","residue_1_atoms":["ZN1","O2","C1"],"residue_2_identifier__ChainResidue":"A96","residue_2_atoms":["NE2","CE1","ND1"],"primary":true,"covalent":true},
    {"residue_1_identifier__ChainResidue":"Z9","residue_1_atoms":["ZN1","O2","C1"],"residue_2_identifier__ChainResidue":"A93","residue_2_atoms":["OE2","CD","CG"],"primary":false,"covalent":false,"dist_tol":0.05,"ang_tol":2.5}
  ]' \
  --verbose

Users should ensure:
- The PDB contains all specified atom names.
- JSON in `--specs` is valid and properly quoted/escaped for the shell.
- Any custom tolerances match the required precision.
"""

import sys
import json
import re
import math
import argparse
from textwrap import dedent
import numpy as np

# --- PDB parsing classes ---
class AtomRecord:
    def __init__(self, name, resName, chainID, resSeq, coord):
        self.name = name.strip()
        self.resName = resName.strip()
        self.chainID = (chainID or " ").strip()  # keep blank if absent
        self.resSeq = resSeq
        self.coord = coord

    @classmethod
    def from_str(cls, line):
        name    = line[12:16]
        resName = line[17:20]
        chainID = line[21]              # <— chain column
        resSeq  = int(line[22:26])
        x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
        return cls(name, resName, chainID, resSeq, np.array([x, y, z]))

class Residue:
    def __init__(self, atom_records):
        self.atom_records = atom_records
        self.coords  = [a.coord for a in atom_records]
        self.resName = atom_records[0].resName if atom_records else ''
        self.resSeq  = atom_records[0].resSeq  if atom_records else None
        self.chainID = atom_records[0].chainID if atom_records else ' '  # <—


    @classmethod
    def from_records(cls, records):
        return cls(records)

# --- Geometry measurement ---
def measure_distance(x, y):
    return math.sqrt(((x - y)**2).sum())

def measure_angle(a, b, c):
    ba = a - b
    bc = c - b
    ca = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return math.degrees(math.acos(max(min(ca, 1.0), -1.0)))

def measure_dihedral(p0, p1, p2, p3):
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1; b2 = p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w); y = np.dot(np.cross(b1, v), w)
    return math.degrees(math.atan2(y, x))

# combine into 6D geometry
def measure_geometry(r1_xyz, r2_xyz):
    r1a1, r1a2, r1a3 = r1_xyz
    r2a1, r2a2, r2a3 = r2_xyz
    d   = measure_distance(r1a1, r2a1)
    aA  = measure_angle(r1a2, r1a1, r2a1)
    aB  = measure_angle(r1a1, r2a1, r2a2)
    dA  = measure_dihedral(r1a3, r1a2, r1a1, r2a1)
    dAB = measure_dihedral(r1a2, r1a1, r2a1, r2a2)
    dB  = measure_dihedral(r1a1, r2a1, r2a2, r2a3)
    return d, aA, aB, dA, dAB, dB

# --- Read PDB and optional REMARK mapping ---
def read_models(pdb_path):
    models=[]; curr_model=[]; curr_atoms=[]; last_seq=None; last_chain=None; multi=False
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('REMARK 666'): continue
            if line.startswith('MODEL'):
                multi=True
                if curr_atoms:
                    curr_model.append(Residue.from_records(curr_atoms)); curr_atoms=[]
                if curr_model:
                    models.append(curr_model)
                curr_model=[]; last_seq=None; last_chain=None
                continue
            if line.startswith('ENDMDL'):
                if curr_atoms:
                    curr_model.append(Residue.from_records(curr_atoms)); curr_atoms=[]
                models.append(curr_model)
                curr_model=[]; last_seq=None; last_chain=None
                continue
            if line.startswith('ATOM') or line.startswith('HETATM'):
                atom = AtomRecord.from_str(line)
                if (last_seq is None
                    or atom.resSeq != last_seq
                    or atom.chainID != last_chain):             # <—
                    if curr_atoms:
                        curr_model.append(Residue.from_records(curr_atoms))
                    curr_atoms=[atom]; last_seq=atom.resSeq; last_chain=atom.chainID
                else:
                    curr_atoms.append(atom)
    if curr_atoms:
        curr_model.append(Residue.from_records(curr_atoms))
    if not multi:
        models.append(curr_model)
    return models


def parse_remark_666(pdb_path):
    pat=re.compile(r'MOTIF\s+([A-Za-z0-9])\s+([A-Z]{3})\s+(\d+)\s+(\d+)')
    mapp={}
    with open(pdb_path) as f:
        for L in f:
            if not L.startswith('REMARK 666'): continue
            m=pat.search(L)
            if m:
                chain, _, seq, idx=m.group(1),m.group(2),int(m.group(3)),int(m.group(4))
                mapp[f"{chain}{seq}"]=idx
    return mapp

# --- Extract atom coord ---
def get_xyz(model, chain_seq, atom_name):
    chain = chain_seq[0]
    seq   = int(chain_seq[1:])
    for res in model:
        if res.resSeq == seq and res.chainID == chain:          # <—
            for atom in res.atom_records:
                if atom.name == atom_name:
                    return atom.coord
    sys.exit(f"Atom {atom_name} in {chain_seq} not found (check chain and residue number)")

# --- Block builder ---
def build_block(idx, entry, verbose=False):
    # unpack and optionally log
    name, r1, r2 = entry['name'], entry['r1_atoms'], entry['r2_atoms']
    d_tol, a_tol = entry.get('dist_tol', 0.02), entry.get('ang_tol', 5.0)
    if verbose:
        print(f"[DEBUG] Block {idx}: {name}, dist_tol={d_tol:.4f}, ang_tol={a_tol:.2f}", file=sys.stderr)
    # format geometry
    g6=entry['g6']
    dist_str = f"{g6[0]:.4f}".rjust(9)
    ang_strs = [f"{v:.2f}".rjust(7) for v in g6[1:]]
    dist_tol_str = f"{d_tol:.4f}".rjust(9)
    ang_tol_str  = f"{a_tol:.2f}".rjust(7)
    cov_flag = 1 if entry['covalent'] else 0

    lines=[f"### --- BLOCK {idx} START --- ###",
           "VARIABLE_CST::BEGIN",
           f"# {name}", 
           "CST::BEGIN","",  
           f"  TEMPLATE::   ATOM_MAP: 1 atom_name: {' '.join(r1)}",
           f"  TEMPLATE::   ATOM_MAP: 1 residue3:  {entry['r1_code']}","",
           f"  TEMPLATE::   ATOM_MAP: 2 atom_name: {' '.join(r2)}",
           f"  TEMPLATE::   ATOM_MAP: 2 residue3:  {entry['r2_code']}","",
           f"  CONSTRAINT:: distanceAB: {dist_str} {dist_tol_str}   150     {cov_flag}   1",
           f"  CONSTRAINT::    angle_A: {ang_strs[0]}   {ang_tol_str}      50   360.  1",
           f"  CONSTRAINT::    angle_B: {ang_strs[1]}   {ang_tol_str}      50   360.  1",
           f"  CONSTRAINT::  torsion_A: {ang_strs[2]}   {ang_tol_str}      50   360.  1",
           f"  CONSTRAINT:: torsion_AB: {ang_strs[3]}   {ang_tol_str}      50   360.  1",
           f"  CONSTRAINT::  torsion_B: {ang_strs[4]}   {ang_tol_str}      50   360.  1","",
           "  ALGORITHM_INFO:: match",
           "   MAX_DUNBRACK_ENERGY 50.0"]
    #if not entry['primary']:
     #   lines.append("   SECONDARY_MATCH: UPSTREAM_CST")
    if not entry['primary']:
        ub = entry.get('upstream_block')
        if ub is None:
            sys.exit(f"[ERROR] spec for block {idx} is primary=False but no "
                     f"'cst_block_of_upstream_protein_res' provided.")
        lines.append(f"   SECONDARY_MATCH: UPSTREAM_CST {ub}")
    lines.extend(["  ALGORITHM_INFO::END","CST::END","VARIABLE_CST::END","\n"])
    return "\n".join(lines)

# --- Main ---
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input_pdb', required=True)
    parser.add_argument('--output_cst', required=True)
    parser.add_argument('--specs', required=True, help='JSON string of constraint specs')
    parser.add_argument('--verbose', action='store_true', help='Enable debug logging')
    args=parser.parse_args()

    specs=json.loads(args.specs)
    model=read_models(args.input_pdb)[0]
    remark_map=parse_remark_666(args.input_pdb)
    if args.verbose:
        print(f"[INFO] Read {len(model)} residues from model", file=sys.stderr)
        print(f"[INFO] Found remark mapping entries: {len(remark_map)}", file=sys.stderr)

    entries=[]

    # inside __main__ where entries are built
    def _split_chain_seq(s):
        return s[0], int(s[1:])

    for spec in specs:
        id1, id2 = spec['residue_1_identifier__ChainResidue'], spec['residue_2_identifier__ChainResidue']
        ch1, seq1 = _split_chain_seq(id1)
        ch2, seq2 = _split_chain_seq(id2)

        code1 = (spec.get('residue_1_3letter_code')
                 or next((r.resName for r in model if r.resSeq==seq1 and r.chainID==ch1), 'UNK'))  # <—
        code2 = (spec.get('residue_2_3letter_code')
                 or next((r.resName for r in model if r.resSeq==seq2 and r.chainID==ch2), 'UNK'))  # <—

        xyz1 = [get_xyz(model, id1, a) for a in spec['residue_1_atoms']]
        xyz2 = [get_xyz(model, id2, a) for a in spec['residue_2_atoms']]
        g6=measure_geometry(xyz1,xyz2)
        if args.verbose:
            found1 = next((r for r in model if r.resSeq==seq1 and r.chainID==ch1), None)
            found2 = next((r for r in model if r.resSeq==seq2 and r.chainID==ch2), None)
            if not found1:
                print(f"[WARN] Could not find residue {id1} in model", file=sys.stderr)
            if not found2:
                print(f"[WARN] Could not find residue {id2} in model", file=sys.stderr)

        entries.append({
            'name': spec.get('name') or f"{code1}{id1}_{code2}{id2}",
            'r1_code': code1, 'r2_code': code2,
            'r1_atoms': spec['residue_1_atoms'], 'r2_atoms': spec['residue_2_atoms'],
            'primary': spec.get('primary',True), 'covalent': spec.get('covalent',False),
            'order': remark_map.get(id2,999), 'g6': g6,
            'dist_tol': spec.get('dist_tol',0.02), 'ang_tol': spec.get('ang_tol',5.0),
            'upstream_block': spec.get('cst_block_of_upstream_protein_res') #NEW
        })
    entries.sort(key=lambda x:x['order'])

    with open(args.output_cst,'w') as outf:
        for i,e in enumerate(entries,1):
            block=build_block(i,e,verbose=args.verbose)
            outf.write(block)
    print(f"Wrote {len(entries)} constraint blocks to {args.output_cst}")
