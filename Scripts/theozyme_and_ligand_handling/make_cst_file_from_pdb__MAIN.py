#!/usr/bin/env python3

# Sibling helper scripts resolve next to this file, so the whole set can be
# relocated together.
_SCRIPT_DIR = Path(__file__).resolve().parent
"""
Author:      Seth M. Woodbury
Date:        2025-05-23
Version:     1

PURPOSE
-------
This wrapper automates a two‑step workflow for generating and then
post‑processing Rosetta constraint (.cst) files from a PDB:

  1. **Constraint Generation**:  
     Invokes `make_cst_file_from_pdb__STEP1__build_CST_file.py` to
     measure pairwise geometries (distance, angles, dihedrals)
     between user‑specified residue atom triplets and writes an
     ordered, formatted .cst file.

  2. **Post‑Processing**:  
     Reads back the generated .cst, then applies optional
     transformations:
       • **Atom‑type substitution** (`atom_name` → `atom_type`)
         for template mappings on either residue1 or residue2.
       • **HIS protonation logic**, automatically choosing
         HIS_D or HIS_E based on first atom or explicit flags.
       • **Dummy‑block reweighting**:  
         Zeroes or custom‑sets constraint weights for any block
         flagged `"dummy_block": true`.
       • **Reciprocal linkage**:  
         Detects any `UPSTREAM_CST N` directives and injects
         corresponding `SECONDARY_MATCH: DOWNSTREAM_CST M`
         into block N to maintain two‑way references.
       • **Custom tolerances** and **verbose debug** bridging.

KEY FEATURES
------------
• **Automatic residue code detection** (3‑letter code inferred
  from PDB if not provided).  
• **REMARK 666 reordering** (respect mapping index in PDB).  
• **Custom tolerances** per constraint: `dist_tol`, `ang_tol`.  
• **Enhanced formatting** (4‑decimals for distances, 2‑decimals
  for angles/dihedrals, fixed distance weight=150).  
• **Atom‑type mappings** for canonical residues, with special
  HIS_D/HIS_E handling.  
• **Dummy‑block adjustment**: zero or custom weight.  
• **Bi‑directional CST references** (UPSTREAM ↔ DOWNSTREAM).  
• **Verbose logging** to stderr for debugging.

USAGE
-----
Required:
  --input_pdb       Path to your PDB file containing ATOM/HETATM
                    (and optional REMARK 666) records.
  --output_cst      Desired path for the .cst output.
  --specs           Inline JSON string or path to JSON file
                    defining a list of constraint specs.

Optional:
  --script_path_cst_generator
                    Full path to the STEP1 generator script
                    (defaults to embedded production script).
  --keep_multiple_atom_types
                    If set, retains *all* mapped atom_types when
                    substituting; otherwise only the first.
  --verbose         Propagate `--verbose` to the STEP1 script and
                    emit extra debug messages in this wrapper.

SPEC FORMAT
-----------
Each element in the specs list is a dict with keys:
  "residue_1_identifier__ChainResidue" : str, e.g. "Z9"
  "residue_1_atoms"                    : [str,str,str]
  "residue_2_identifier__ChainResidue" : str, e.g. "A96"
  "residue_2_atoms"                    : [str,str,str]
  "primary"                            : bool
  "covalent"                           : bool

  # Optional tolerance overrides:
  "dist_tol"                           : float (Å)
  "ang_tol"                            : float (°)

  # Atom‑type substitution flags:
  "residue1_atom_type"                 : bool (default false)
  "residue2_atom_type"                 : bool (default false)

  # HIS special‑case protonation (overrides auto‑infer):
  "residue1_protonation"               : "HIS_D"|"HIS_E"
  "residue2_protonation"               : "HIS_D"|"HIS_E"

  # Dummy‑block reweighting:
  "dummy_block"                        : bool (default false)
  "dummy_block_custom_weight"          : int (default 0)

EXAMPLE
-------
python wrapper.py \
  --input_pdb /path/to/structure.pdb \
  --output_cst /path/to/output.cst \
  --specs '[
    {
      "residue_1_identifier__ChainResidue":"Z9",
      "residue_1_atoms":["ZN1","O2","C1"],
      "residue_2_identifier__ChainResidue":"A96",
      "residue_2_atoms":["NE2","CE1","ND1"],
      "primary":true,
      "covalent":true,
      "residue2_atom_type":true,
      "residue2_protonation":"HIS_D",
      "dummy_block": true,
      "dummy_block_custom_weight": 10
    }
  ]' \
  --verbose

NOTES
-----
- Ensure all specified atom names exist in the PDB.
- If using REMARK 666 reordering, confirm the PDB contains valid
  `REMARK 666 MATCH … MOTIF … idx` entries.
- The HIS_D/HIS_E logic will infer protonation by first‑atom name,
  but can be overridden via the `residue#_protonation` flag.
- Dummy blocks will have their constraint “weight” column zeroed
  (or set to custom value) across all six constraint lines.
- Downstream CST references are auto‑inserted for symmetry; warn
  is printed if block ordering appears reversed.
"""

import os
import sys
import json
import shlex
import subprocess
import argparse
import re
from pathlib import Path

# Predefined atom_name→atom_type mappings by three-letter residue code
### NOTE: YOU MAY NEED TO EXPAND THIS DICTIONARY I DID NOT MAKE EVERY POSSIBLE MAPPING YET...
### YOU CAN FIND THESE MAPPINGS AT: https://docs.rosettacommons.org/docs/latest/rosetta_basics/Rosetta-AtomTypes 
### THE FORMAT IS `RES`: {'AtomName1': `AtomType1`, 'AtomName2': `AtomType2`}
### YOU SHOULD ONLY NEED TO DO THIS FOR ALL THE UNIQUE ATOMS IN ALL 20 CANONICAL AMINO ACIDS
### NOTE ---> HISTIDINE IS WEIRD, YOU NEED TO ADD STUFF BASED ON HIS_D OR HIS_E
ATOM_TYPE_MAP = {
    # aromatics #
    'TYR': {'OH': 'OH', 'CZ': 'aroC', 'CE2': 'aroC', 'CE1': 'aroC', 'CD2': 'aroC', 'CD1': 'aroC', 'CG': 'aroC', 'CB': 'CH2'},
  
    # positive charged #
    'HIS': {'NE2': 'Nhis', 'ND1': 'Ntrp', 'CE1': 'aroC', 'CD2': 'aroC', 'CG': 'aroC'}, # HIS_D aka hydrogen on delta N.
    #'HIS': {'NE2': 'Ntrp', 'ND1': 'Nhis', 'CE1': 'aroC', 'CD2': 'aroC', 'CG': 'aroC'}, # HIS_E aka hydrogen on delta N. YOU NEED TO FIGURE OUT HOW TO IMPLEMENT THIS
  
    # negative charged #
    'GLU': {'OE2': 'OOC', 'OE1': 'OOC', 'CD': 'COO', 'CG': 'CH2', 'CB': 'CH2'},
    'ASP': {'OD2': 'OOC', 'OD1': 'OOC', 'CG': 'COO', 'CB': 'CH2'},

    # extend with other residues as needed
}

ATOM_TYPE_MAP.update({
    'HIS_D': {'NE2': 'Nhis', 'ND1': 'Ntrp','CE1': 'aroC', 'CD2': 'aroC', 'CG': 'aroC'},
    'HIS_E': {'NE2': 'Ntrp', 'ND1': 'Nhis','CE1': 'aroC', 'CD2': 'aroC', 'CG': 'aroC'},
})

def post_process_cst(path, original_specs, keep_multiple=False):
    """
    For each spec with 'residue1_atom_type'=True or 'residue2_atom_type'=True,
    replace the corresponding TEMPLATE atom_name line with atom_type using
    ATOM_TYPE_MAP.
    """
    text = Path(path).read_text().splitlines(keepends=True)
    # Identify block boundaries
    header_idxs = [i for i, line in enumerate(text) if line.startswith('### --- BLOCK')]
    header_idxs.append(len(text))

    for block_num, spec in enumerate(original_specs, start=1):
        start, end = header_idxs[block_num-1], header_idxs[block_num]
        block = text[start:end]

        # Process both maps
        for map_idx, flag_key in ((1, 'residue1_atom_type'), (2, 'residue2_atom_type')):
            if not spec.get(flag_key):
                continue
            # find residue code
            code = None
            pat = re.compile(rf'ATOM_MAP:\s+{map_idx}\s+residue3:\s+(\w+)')
            for L in block:
                m = pat.search(L)
                if m:
                    code = m.group(1)
                    break
            if not code or code not in ATOM_TYPE_MAP:
                print(f"[WARN] No mapping for residue '{code}' in block {block_num}", file=sys.stderr)
                continue
            # ─── choose HIS_D vs HIS_E or residue override ─────────────────────────────
            mapping_key = code
            if code == 'HIS':
                # 1) explicit override from specs?
                protonation = spec.get(f'residue{map_idx}_protonation')
                if protonation in ('HIS_D', 'HIS_E'):
                    mapping_key = protonation
                else:
                    # 2) infer from the first atom in the atom_name line
                    atom_line = next(
                        (l for l in block
                         if f'ATOM_MAP:' in l and f'{map_idx} atom_name:' in l),
                        None
                    )
                    if atom_line:
                        first_atom = atom_line.split('atom_name:')[1].strip().split()[0]
                        if first_atom == 'NE2':
                            mapping_key = 'HIS_D'
                        elif first_atom == 'ND1':
                            mapping_key = 'HIS_E'
            # finally grab whichever mapping dictionary exists
            mapping = ATOM_TYPE_MAP.get(mapping_key, ATOM_TYPE_MAP.get(code, {}))
            # ───────────────────────────────────────────────────────────────────────────────
            # replace atom_name line
            atom_pat = re.compile(rf'^(\s*TEMPLATE::\s+ATOM_MAP:\s+{map_idx}\s+)atom_name:(.*)$')
            for i, L in enumerate(block):
                m = atom_pat.match(L)
                if m:
                    prefix = m.group(1)
                    atoms = m.group(2).strip().split()
                    types = [mapping.get(a, a) for a in atoms]
                    # unless user requested otherwise, only keep the first type
                    if not keep_multiple and types:
                        types = types[:1]
                    block[i] = f"{prefix}atom_type: {' '.join(types)}\n"
                    break

        text[start:end] = block

    Path(path).write_text(''.join(text))


def main():
    p = argparse.ArgumentParser(description="Build, run, and post-process CST file")
    p.add_argument('--input_pdb',  required=True, help="Input PDB file")
    p.add_argument('--output_cst', required=True, help="Output CST file")
    p.add_argument('--specs', required=True, help="Inline JSON string of specs (must start with '[' or '{'), or path to a JSON file")
    p.add_argument('--script_path_cst_generator', default=str(_SCRIPT_DIR / "make_cst_file_from_pdb__STEP1__build_CST_file.py"), help="Path to the CST-generator script")
    p.add_argument('--keep_multiple_atom_types', action='store_true', help="If set, keep all mapped atom_types; otherwise only the first one is used... I DON'T THINK THIS SHOULD EVER BE USED")    
    p.add_argument('--verbose', action='store_true', help="Enable verbose logging in CST script")
    args = p.parse_args()

    # ─── Create the output folder if needed ───────────────────────────────
    out_dir = os.path.dirname(args.output_cst)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    # ────────────────────────────────────────────────────────────────────────
    specs_input = args.specs.strip()
    if specs_input.startswith('[') or specs_input.startswith('{'):
        specs_json = specs_input
    else:
        try:
            specs_json = Path(specs_input).read_text()
        except Exception as e:
            sys.exit(f"ERROR: could not read specs file '{specs_input}': {e}")

    try:
        original_specs = json.loads(specs_json)
    except json.JSONDecodeError as e:
        sys.exit(f"ERROR: --specs is not valid JSON: {e}")

    # Remove internal-only flags before invoking generator
    stripped = []
    for spec in original_specs:
        c = spec.copy()
        c.pop('residue1_atom_type', None)
        c.pop('residue2_atom_type', None)
        stripped.append(c)
    stripped_json = json.dumps(stripped)

    cmd = [
        sys.executable,
        args.script_path_cst_generator,
        '--input_pdb',  args.input_pdb,
        '--output_cst', args.output_cst,
        '--specs',      stripped_json
    ]
    if args.verbose:
        cmd.append('--verbose')

    cmd_str = shlex.join(cmd)
    print(f"[INFO] Executing:\n{cmd_str}\n", file=sys.stderr)
    res = subprocess.run(cmd, shell=False)
    if res.returncode != 0:
        sys.exit(res.returncode)

    post_process_cst(args.output_cst, original_specs, keep_multiple=args.keep_multiple_atom_types)

    # ─── Add reciprocal DOWNSTREAM_CST entries ────────────────────────────────────
    cst_lines = Path(args.output_cst).read_text().splitlines(keepends=True)

    # 1) Build a map of block_number → (start_idx, end_idx) by header markers
    header_idxs = [i for i, line in enumerate(cst_lines) if line.startswith('### --- BLOCK')]
    header_idxs.append(len(cst_lines))
    block_bounds = {}
    for n in range(len(header_idxs) - 1):
        start = header_idxs[n]
        end   = header_idxs[n + 1]
        # extract the block number from the header line
        bnum = int(re.search(r'### --- BLOCK (\d+) START', cst_lines[start]).group(1))
        block_bounds[bnum] = (start, end)

    # 2) Collect only the *template‑provided* UPSTREAM_CST markers
    ups = {}  # up_block → list of downstream_blocks
    for bnum, (start, end) in block_bounds.items():
        for L in cst_lines[start:end]:
            m = re.search(r'SECONDARY_MATCH:\s+UPSTREAM_CST\s+(\d+)', L)
            if m:
                up = int(m.group(1))
                ups.setdefault(up, []).append(bnum)

    # 3) Insert DOWNSTREAM_CST lines in *descending* block order so earlier blocks don't get shifted
    for up in sorted(ups.keys(), reverse=True):
        downs = ups[up]
        if up not in block_bounds:
            print(f"[WARN] Invalid upstream reference {up}", file=sys.stderr)
            continue
        start, end = block_bounds[up]
        # find the last ALGORITHM_INFO::END in that span
        insert_pos = None
        for idx in range(start, end):
            if cst_lines[idx].strip() == "ALGORITHM_INFO::END":
                insert_pos = idx
        if insert_pos is None:
            print(f"[WARN] No ALGORITHM_INFO::END found in block {up}", file=sys.stderr)
            continue

        offset = 0
        for down in downs:
            if down <= up:
                print(f"[WARN] Block {up} listed as upstream of {down}, but {down} ≤ {up}", file=sys.stderr)
            line = f"   SECONDARY_MATCH: DOWNSTREAM_CST {down}\n"
            cst_lines.insert(insert_pos + offset, line)
            offset += 1

    # ─── Zero or re‑weight any dummy blocks ─────────────────────────────────
    # Build a map of block_idx → new_weight
    dummy_map = {}
    for idx, spec in enumerate(original_specs, start=1):
        if spec.get('dummy_block'):
            dummy_map[idx] = spec.get('dummy_block_custom_weight', 0)

    if dummy_map:
        # reuse hdr_idxs from earlier or rebuild
        hdr_idxs = [i for i, line in enumerate(cst_lines) if line.startswith('### --- BLOCK')]
        hdr_idxs.append(len(cst_lines))
        # regex: capture up to the weight, then the old weight, then the rest
        weight_re = re.compile(r'^(\s*CONSTRAINT::\s+\S+:\s+[+-]?\d+\.\d+\s+[+-]?\d+\.\d+\s+)(\d+)(\s+.*)$')
        for blk_idx, new_wt in dummy_map.items():
            if blk_idx < 1 or blk_idx > len(hdr_idxs)-1:
                print(f"[WARN] dummy_block refers to invalid block {blk_idx}", file=sys.stderr)
                continue
            start, end = hdr_idxs[blk_idx-1], hdr_idxs[blk_idx]
            for i in range(start, end):
                m = weight_re.match(cst_lines[i])
                if m:
                    prefix, old, suffix = m.group(1), m.group(2), m.group(3)
                    cst_lines[i] = f"{prefix}{new_wt}{suffix}\n"
    # ───────────────────────────────────────────────────────────────────────────
    # write the augmented file back out
    Path(args.output_cst).write_text(''.join(cst_lines))
    # ───────────────────────────────────────────────────────────────────────────────

    print(f"[INFO] Post-processed atom_type mappings and wrote {args.output_cst}")

if __name__ == '__main__':
    main()
