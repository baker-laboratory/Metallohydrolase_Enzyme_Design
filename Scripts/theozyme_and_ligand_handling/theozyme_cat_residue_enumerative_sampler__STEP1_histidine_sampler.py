#!/usr/bin/env python3
"""
SCRIPT NAME
    theozyme_cat_residue_enumerative_sampler__STEP1_histidine_sampler.py

DESCRIPTION
    Enumerates histidine ring-flip and tautomer-swap OPERATIONS in a PDB.
    Each code is two characters, <flip><tautomer>, describing what is done
    to the input histidine (operations are RELATIVE to the input; the
    letters do NOT assert an absolute epsilon/delta state):

      char 1  (flip):       U = unflipped (ring left as input)
                            F = flipped   (180° ring flip about the axis
                                            perpendicular to CG–ND1)
      char 2  (tautomer):   O = original  (tautomer left as input)
                            A = alternate (tautomer swapped: NE2↔ND1,
                                            CD2↔CG, HE2↔HD1, realigned)

    Resulting codes:
      UO: no change (identity / pass-through)
      FO: flip only                       (was 1E)
      UA: tautomer swap only              (was 0D)
      FA: flip + tautomer swap            (was 1D)

    Legacy codes 0E/1E/0D/1D are still accepted and auto-mapped
    (0E→UO, 1E→FO, 0D→UA, 1D→FA).

    Outputs one PDB per permutation and reports a summary at the end.

ARGUMENTS
    --input_pdb PATH
        Path to the input PDB file containing your target histidines.
    --output_dir PATH
        Directory to write output PDB variants. Defaults to the input file’s folder.
    --histidine_config JSON_OR_PATH
        JSON string or path to JSON file mapping His identifiers (e.g. "A1") to lists of allowed modes
        ["UO","FO","UA","FA"] (legacy ["0E","1E","0D","1D"] also accepted) or the special token "all".
    --allowed_tautomer_swaps_at_once N1,N2,...
        Comma‑separated integers limiting how many tautomer-swap (A‑type: UA/FA) ops are allowed per model.
    --classic_suffix
        Flag; if set, use classic suffix style (_UO_FA_UO) instead of compact (_UFU_OAO).
    --verbose_pdbs
        Flag; if set, print each written PDB path as the script runs.

EXAMPLE
    python theozyme_cat_residue_enumerative_sampler__STEP1_histidine_sampler.py \
      --input_pdb /path/to/project/.../test.pdb \
      --output_dir /path/to/project/.../test_his_variants \
      --histidine_config '{"A1":["UO","FO","UA"],"B2":"all","D4":"all"}' \
      --allowed_tautomer_swaps_at_once 0,1,2 \
      --verbose_pdbs \
      --classic_suffix
"""

import os
import sys
import argparse
import json
import copy
import itertools
import numpy as np

##############################################
### ATOM PARSING & PDB I/O ###
##############################################

class Atom:
    def __init__(self, line):
        self._orig = line.rstrip('\n')
        self.name = self._orig[12:16].strip()
        self.resname = self._orig[17:20].strip()
        self.chain = self._orig[21]
        self.resnum = int(self._orig[22:26])
        x = float(self._orig[30:38])
        y = float(self._orig[38:46])
        z = float(self._orig[46:54])
        self.coord = np.array([x, y, z], dtype=float)

    def format_line(self):
        # replace atom name and coords in original line
        line = self._orig
        name_field = f"{self.name:>4}"
        x, y, z = self.coord
        return (
            line[:12]
            + name_field
            + line[16:30]
            + f"{x:8.3f}{y:8.3f}{z:8.3f}"
            + line[54:]
            + "\n"
        )

def parse_pdb_lines(pdb_path):
    records = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith(("ATOM  ", "HETATM")):
                records.append(Atom(line))
            else:
                records.append(line)
    return records

def write_pdb_records(records, out_path):
    with open(out_path, 'w') as w:
        for rec in records:
            if isinstance(rec, Atom):
                w.write(rec.format_line())
            else:
                w.write(rec)

##############################################
### GEOMETRIC HELPERS ###
##############################################

def rotate_around_axis_180(pt, p1, p2):
    v = pt - p1
    axis = p2 - p1
    axis = axis / np.linalg.norm(axis)
    # rotation by pi: v_rot = -v + 2*(axis·v)*axis
    return p1 + (-v + 2 * np.dot(axis, v) * axis)

def swap_coords(a1, a2):
    c1 = a1.coord.copy()
    c2 = a2.coord.copy()
    a1.coord[:] = c2
    a2.coord[:] = c1

##############################################
### RIGID ALIGNMENTS ###
##############################################
def rigid_realign_1E(res_atoms, orig_coords, pivot=None):
    """
    Pivot about the anchor nitrogen so that:
      • new CD2 → old CE1
      • new CE1 → old CD2
    The anchor (NE2 by default, or ND1) remains exactly fixed.
    """
    # pivot point (default: original NE2 coord)
    if pivot is None:
        pivot = orig_coords['NE2']

    # build vector pairs from pivot
    P = np.stack([
        res_atoms['CD2'].coord - pivot,   # post-rotation CD2 vector
        res_atoms['CE1'].coord - pivot    # post-rotation CE1 vector
    ])
    Q = np.stack([
        orig_coords['CE1'] - pivot,       # target CE1 vector
        orig_coords['CD2'] - pivot        # target CD2 vector
    ])

    # Kabsch for rotation only
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T

    # apply rotation about the pivot
    for atom in res_atoms.values():
        atom.coord[:] = pivot + R @ (atom.coord - pivot)

def rigid_realign_0D(res_atoms, orig_coords, pivot=None,
                     src_names=None, tgt_coords=None):
    """
    Pivot about the locked position. Default (NE2-anchored) mapping:
      • new CG  → old CD2
      • new CE1 → old CE1
    For ND-anchored mirror, the caller passes the symmetric pair:
      • new CD2 → old CG
      • new CE1 → old CE1
    HE2/HD1 are excluded from the rotation (just renamed).
    """
    if pivot is None:
        pivot = res_atoms['ND1'].coord
    if src_names is None:
        src_names = ('CG', 'CE1')
    if tgt_coords is None:
        tgt_coords = (orig_coords['CD2'], orig_coords['CE1'])
    exclude = {'HE2', 'HD1'}

    # build vectors from pivot
    P = np.stack([res_atoms[n].coord - pivot for n in src_names])
    Q = np.stack([t - pivot for t in tgt_coords])

    # Kabsch
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T

    # apply rotation, skipping H’s
    for atom in res_atoms.values():
        if atom.name in exclude:
            continue
        atom.coord[:] = pivot + R @ (atom.coord - pivot)

def rigid_realign_1D(res_atoms, orig_coords, pivot=None,
                     src_names=None, tgt_coords=None):
    """
    Pivot about the locked position. Default (NE2-anchored) mapping:
      • new CG   → old CE1
      • new CE1  → old CG
    For ND-anchored mirror, the caller passes the symmetric pair:
      • new CD2  → old CE1
      • new CE1  → old CD2
    The locked point remains exactly fixed.
    """
    # 1) pivot point (default: post-0D ND1)
    if pivot is None:
        pivot = orig_coords['ND1']
    if src_names is None:
        src_names = ('CG', 'CE1')
    if tgt_coords is None:
        tgt_coords = (orig_coords['CE1'], orig_coords['CG'])

    # 2) build vector pairs (post-rotate → target) from pivot
    P = np.stack([res_atoms[n].coord - pivot for n in src_names])
    Q = np.stack([t - pivot for t in tgt_coords])

    # 3) compute optimal rotation via Kabsch
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T

    # 4) apply rotation about the pivot
    for atom in res_atoms.values():
        atom.coord[:] = pivot + R @ (atom.coord - pivot)

##############################################
### TRANSFORMATIONS ###
##############################################

def transform_0E(res_atoms):
    # no-op
    return

def transform_1E_perp_axis(res_atoms, anchor='NE2'):
    """
    Ring flip, original tautomer (code 'FO').

    1) Rotate the His 180° around the axis perpendicular to the
       CG–<other-N> bond, dropping a perpendicular from the ANCHOR
       nitrogen onto that bond.
    2) Rigid‑body realign pivoting about the ANCHOR so that:
         • new CD2 → original CE1
         • new CE1 → original CD2
       The ANCHOR nitrogen remains exactly fixed at its input position.

    anchor='NE2' (default) reproduces the original NE2‑locked behavior;
    anchor='ND1' locks the ND1 position instead.
    """
    other = 'ND1' if anchor == 'NE2' else 'NE2'

    # 0) Save the original "tip" coordinates for later realignment
    orig_coords = {
        anchor: res_atoms[anchor].coord.copy(),
        'CE1': res_atoms['CE1'].coord.copy(),
        'CD2': res_atoms['CD2'].coord.copy()
    }

    # 1) Key positions
    cg     = res_atoms['CG'].coord            # CG position
    a_pos  = res_atoms[anchor].coord.copy()   # anchor nitrogen (stays fixed)
    o_pos  = res_atoms[other].coord           # the other nitrogen

    # 2) Bond line CG → other-N
    origin = cg
    v      = o_pos - cg

    # 3) Foot of perpendicular from the ANCHOR onto that line
    t    = np.dot(a_pos - origin, v) / np.dot(v, v)
    foot = origin + t * v

    # 4) Axis endpoints: ANCHOR → foot
    p1, p2 = a_pos, foot

    # 5) Rotate every atom 180° about the p1→p2 axis
    for atom in res_atoms.values():
        atom.coord[:] = rotate_around_axis_180(atom.coord, p1, p2)

    # 6) Realign with the ANCHOR pinned at its original position
    rigid_realign_1E(res_atoms, orig_coords, pivot=orig_coords[anchor])

def transform_0D_perp_axis(res_atoms, anchor='NE2'):
    """
    Tautomer swap, no flip (code 'UA').

    1) Save originals of NE2/ND1/CE1/CD2.
    2) Rename HE2↔HD1 (coordinates untouched).
    3) Rotate 180° about the axis from CE1→foot (perp to CD2–CG).
    4) Translate so ND1 → the LOCKED position (original NE2 if
       anchor='NE2' [default], or original ND1 if anchor='ND1'); skip H’s.
    5) Rigid‑body realign about the locked point to match
       CG→orig CD2 & CE1→orig CE1.

    The coordinating nitrogen ends up exactly at the locked position.
    """
    # 1) Save originals (NE2, ND1, CG, CE1, CD2 — both anchors need them)
    orig_coords = {
        'NE2': res_atoms['NE2'].coord.copy(),
        'ND1': res_atoms['ND1'].coord.copy(),
        'CG':  res_atoms['CG'].coord.copy(),
        'CE1': res_atoms['CE1'].coord.copy(),
        'CD2': res_atoms['CD2'].coord.copy()
    }
    exclude = {'HE2','HD1'}

    # 2) Rename hydrogens only
    if 'HE2' in res_atoms and 'HD1' in res_atoms:
        res_atoms['HE2'].name, res_atoms['HD1'].name = 'HD1', 'HE2'
    elif 'HE2' in res_atoms:
        atom = res_atoms.pop('HE2'); atom.name = 'HD1'; res_atoms['HD1'] = atom
    elif 'HD1' in res_atoms:
        atom = res_atoms.pop('HD1'); atom.name = 'HE2'; res_atoms['HE2'] = atom

    # 3) Perp‑axis rotation about CE1→foot on CD2–CG
    ce1 = res_atoms['CE1'].coord
    cd2 = res_atoms['CD2'].coord
    cg  = res_atoms['CG'].coord

    origin = cd2
    v      = cg - cd2
    t      = np.dot(ce1 - origin, v) / np.dot(v, v)
    foot   = origin + t * v
    p1, p2 = ce1, foot

    for atom in res_atoms.values():
        if atom.name in exclude:
            continue
        atom.coord[:] = rotate_around_axis_180(atom.coord, p1, p2)

    # 4) Anchor-aware translate + realign (symmetric imidazole mirror).
    #   NE2 anchor (legacy):  pin atom ND1 onto origNE2; realign
    #       (new CG, new CE1) -> (origCD2, origCE1) about origNE2.
    #   ND1 anchor (mirror through ND1): pin atom NE2 onto origND1; realign
    #       (new CD2, new CE1) -> (origCG,  origCE1) about origND1.
    # The ND1 path keeps the catalytic-N tip at origND1 while properly
    # swapping the NE2/ND1 atoms in space (real tautomer mirror), and the
    # translate delta is ~0 so the backbone barely moves.
    if anchor == 'NE2':
        lock      = orig_coords['NE2']
        move_atom = 'ND1'
        rsrc      = ('CG', 'CE1')
        rtgt      = (orig_coords['CD2'], orig_coords['CE1'])
    else:
        lock      = orig_coords['ND1']
        move_atom = 'NE2'
        rsrc      = ('CD2', 'CE1')
        rtgt      = (orig_coords['CG'], orig_coords['CE1'])

    delta = lock - res_atoms[move_atom].coord
    for atom in res_atoms.values():
        if atom.name in exclude:
            continue
        atom.coord[:] += delta

    rigid_realign_0D(res_atoms, orig_coords, pivot=lock,
                     src_names=rsrc, tgt_coords=rtgt)

def transform_1D_perp_axis(res_atoms, anchor='NE2'):
    """
    Flip + tautomer swap (code 'FA'). The 0D stage already moves the
    coordinating nitrogen onto the locked position (NE2- or ND1-anchored),
    then a second 180° flip is applied and realigned about that locked
    point (which is exactly fixed).

    1) Perform the 0D tautomer swap.
    2) Rotate the His 180° around the axis that:
       • passes through ND1 and its foot‑of‑perpendicular onto the NE2–CD2 bond
       • i.e. is perpendicular to the NE2–CD2 bond
    3) Then rigid‑body realign (pivoting about ND1) so that:
         • new CG  → original CE1
         • new CE1 → original CG
       (ND1 stays fixed).
    """
    # 1) do the 0D transformation first (anchor-aware lock)
    transform_0D_perp_axis(res_atoms, anchor=anchor)

    # Anchor-aware second-stage parameters (mirror through the locked atom).
    #   NE2 anchor (legacy): pivot = atom ND1 (now at origNE2 = lock).
    #       Flip axis: perp to NE2-CD2 line through ND1.
    #       Realign: (new CG, new CE1) -> (origCE1, origCG).
    #   ND1 anchor (mirror): pivot = atom NE2 (now at origND1 = lock).
    #       Flip axis: perp to ND1-CG line through NE2.
    #       Realign: (new CD2, new CE1) -> (origCE1, origCD2).
    if anchor == 'NE2':
        pivot_name = 'ND1'
        axis_a, axis_b = 'NE2', 'CD2'
        sname_a, sname_b = 'CG', 'CE1'
    else:
        pivot_name = 'NE2'
        axis_a, axis_b = 'ND1', 'CG'
        sname_a, sname_b = 'CD2', 'CE1'

    # 2) save post‑0D original coords for realignment
    orig_coords = {
        pivot_name: res_atoms[pivot_name].coord.copy(),
        sname_a:    res_atoms[sname_a].coord.copy(),
        sname_b:    res_atoms[sname_b].coord.copy(),
    }

    # 3) axis construction: perp to (axis_a → axis_b) through pivot atom
    a_pos  = res_atoms[axis_a].coord
    b_pos  = res_atoms[axis_b].coord
    p_pos  = res_atoms[pivot_name].coord.copy()
    origin = a_pos
    v      = b_pos - a_pos
    t      = np.dot(p_pos - origin, v) / np.dot(v, v)
    foot   = origin + t * v
    p1, p2 = p_pos, foot

    # 4) rotate every atom 180° about that axis
    for atom in res_atoms.values():
        atom.coord[:] = rotate_around_axis_180(atom.coord, p1, p2)

    # 5) rigid‑body realign about the locked pivot, mapping src ↔ tgt
    pivot_coord = orig_coords[pivot_name]
    tgt_coords  = (orig_coords[sname_b], orig_coords[sname_a])   # swap a↔b
    rigid_realign_1D(res_atoms, orig_coords, pivot=pivot_coord,
                     src_names=(sname_a, sname_b),
                     tgt_coords=tgt_coords)

##############################################
### MAIN LOGIC ###
##############################################

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input_pdb', required=True,
                   help='Path to input PDB')
    p.add_argument('--output_dir', default=None,
                   help='Directory for output PDB variants')
    p.add_argument('--histidine_config', required=True,
                   help='JSON string or path to file mapping HIS identifiers to lists of '
                        '["UO","FO","UA","FA"] (legacy "0E","1E","0D","1D" also accepted) or "all"')
    p.add_argument('--allowed_tautomer_swaps_at_once', default=None,
                   help='Comma-separated ints, e.g. "0,1,2" to cap number of tautomer-swap (A-type) ops')
    p.add_argument(
        '--classic_suffix',
        action='store_true',
        default=False,
        help='Use classic suffix style (_UO_FA_UO) instead of compact (UFU_OAO)'
    )
    p.add_argument(
        '--verbose_pdbs',
        action='store_true',
        default=False,
        help='If set, print out each newly made PDB path'
    )

    args = p.parse_args()

    # load config
    cfg_raw = args.histidine_config
    if os.path.isfile(cfg_raw):
        cfg = json.load(open(cfg_raw))
    else:
        cfg = json.loads(cfg_raw)

    # parse allowed swaps
    if args.allowed_tautomer_swaps_at_once:
        allowed_D = set(int(x) for x in args.allowed_tautomer_swaps_at_once.split(','))
    else:
        allowed_D = None

    # normalize config
    #   canonical codes: <flip:U|F><tautomer:O|A>
    #   UO = identity, FO = flip only, UA = tautomer swap only, FA = both
    CANONICAL = ['UO', 'FO', 'UA', 'FA']
    LEGACY_MAP = {'0E': 'UO', '1E': 'FO', '0D': 'UA', '1D': 'FA'}

    #   Config keys may carry an optional pivot suffix:
    #     "A201"     -> pivot NE2 (default; identical to legacy behavior)
    #     "A201:ND"  -> lock the ND1 position
    #     "A201:NE"  -> explicit default
    #   The pivot is the nitrogen whose INPUT position is held fixed in
    #   space for every variant of that residue; the "coordinating"
    #   nitrogen ends up exactly there (via ring flip / tautomer switch).
    PIVOT_ALIASES = {'': 'NE2', 'NE': 'NE2', 'NE2': 'NE2',
                     'ND': 'ND1', 'ND1': 'ND1'}

    allowed_map = {}                 # resid -> [codes]
    pivot_map = {}                   # resid -> 'NE2' | 'ND1'
    legacy_seen = False
    for raw_key, val in cfg.items():
        parts = str(raw_key).split(':')
        resid = parts[0].strip()
        piv_tok = (parts[1].strip().upper() if len(parts) > 1 else '')
        if len(parts) > 2:
            sys.exit(f"Error: malformed key '{raw_key}' (use 'A201' or 'A201:ND').")
        if piv_tok not in PIVOT_ALIASES:
            sys.exit(f"Error: invalid pivot '{piv_tok}' in '{raw_key}'. "
                     f"Use NE/NE2 (default) or ND/ND1.")
        anchor = PIVOT_ALIASES[piv_tok]
        if resid in allowed_map:
            sys.exit(f"Error: residue '{resid}' specified more than once "
                     f"(conflicting keys, e.g. with/without pivot).")

        if val == 'all':
            allowed = list(CANONICAL)
        else:
            allowed = []
            for x in list(val):
                tok = str(x).strip().upper()
                if tok in LEGACY_MAP:        # accept legacy 0E/1E/0D/1D
                    legacy_seen = True
                    tok = LEGACY_MAP[tok]
                if tok not in CANONICAL:
                    sys.exit(
                        f"Error: invalid choice '{x}' for {raw_key}. "
                        f"Valid: {CANONICAL} (or legacy {list(LEGACY_MAP)})")
                if tok not in allowed:       # de-dupe, preserve order
                    allowed.append(tok)
        allowed_map[resid] = allowed
        pivot_map[resid] = anchor

    if legacy_seen:
        print("[INFO] Legacy histidine codes (0E/1E/0D/1D) detected and "
              "auto-mapped to UO/FO/UA/FA. Update configs when convenient.")

    # parse PDB
    records = parse_pdb_lines(args.input_pdb)

    # find HIS indices
    idx_map = {}  # resid -> list of record indices
    for i, rec in enumerate(records):
        if isinstance(rec, Atom) and rec.resname == 'HIS':
            for resid in allowed_map:
                chain, resn = resid[0], int(resid[1:])
                if rec.chain == chain and rec.resnum == resn:
                    idx_map.setdefault(resid, []).append(i)
    missing = set(allowed_map.keys()) - idx_map.keys()
    if missing:
        print(f"### CHECK THAT THIS RESIDUE {missing} IS ACTUALLY A HISTDINE ###")
        sys.exit(f"Error: residues not found in PDB: {missing}")

    # ---- configuration / provenance printout ----
    print("\n######### HISTIDINE SAMPLER — CONFIGURATION #########")
    print(f"Input PDB        : {args.input_pdb}")
    print(f"Output dir       : {args.output_dir or os.path.dirname(args.input_pdb)}")
    print(f"Suffix style     : {'classic' if args.classic_suffix else 'compact'}")
    print(f"Allowed A-counts : "
          f"{sorted(allowed_D) if allowed_D is not None else 'all (no cap)'}")
    print("Per-residue plan :")
    for resid in allowed_map:
        anc = pivot_map[resid]
        nice = 'ND1 (delta)' if anc == 'ND1' else 'NE2 (epsilon)'
        print(f"  {resid:<6s} pivot={anc:<3s} [{nice} locked]  "
              f"codes={allowed_map[resid]}")
    print("####################################################\n")

    # prepare output dir
    out_dir = args.output_dir or os.path.dirname(args.input_pdb)
    os.makedirs(out_dir, exist_ok=True)

    # generate permutations
    keys = list(allowed_map.keys())
    all_perms = itertools.product(*(allowed_map[k] for k in keys))
    # Counters for reporting
    total_perms = 0
    generated_perms = 0
    filtered_perms = 0
    code_counts = {c: 0 for c in CANONICAL}

    for perm in all_perms:
        total_perms += 1
        # filter by allowed D-count
        Dcount = sum(1 for c in perm if c[1] == 'A')  # tautomer-swap (A) ops
        if allowed_D is not None and Dcount not in allowed_D:
            filtered_perms += 1
            continue
        for c in perm:
            code_counts[c] += 1

        # deep-copy records and apply transforms
        recs_copy = copy.deepcopy(records)
        for key, conf in zip(keys, perm):
            anchor = pivot_map[key]
            # map atoms of this residue
            subidx = idx_map[key]
            sub_atoms = {a.name: a for a in (recs_copy[i] for i in subidx)}
            if conf == 'UO':            # unflipped, original tautomer (no-op)
                transform_0E(sub_atoms)
            elif conf == 'FO':          # flipped, original tautomer
                transform_1E_perp_axis(sub_atoms, anchor=anchor)
            elif conf == 'UA':          # unflipped, alternate (tautomer swap)
                transform_0D_perp_axis(sub_atoms, anchor=anchor)
            elif conf == 'FA':          # flipped + alternate
                transform_1D_perp_axis(sub_atoms, anchor=anchor)

        # write out variant
        base = os.path.splitext(os.path.basename(args.input_pdb))[0]

        if args.classic_suffix:
            # old style: _0E_1D_0E
            suffix = "_".join(perm)
        else:
            # new default: _010_EDE
            flip_bits    = "".join(p[0] for p in perm)
            taut_letters = "".join(p[1] for p in perm)
            suffix       = f"{flip_bits}_{taut_letters}"

        out_path = os.path.join(out_dir, f"{base}_{suffix}.pdb")
        write_pdb_records(recs_copy, out_path)
        if args.verbose_pdbs:
            mapping = ", ".join(
                f"{k}={c}({pivot_map[k]})" for k, c in zip(keys, perm))
            print(f"Wrote {os.path.basename(out_path)}  |  {mapping}")
        generated_perms += 1

    # summary
    pivots_used = sorted(set(pivot_map.values()))
    nd_residues = [r for r in allowed_map if pivot_map[r] == 'ND1']
    print(f"\n######### HISTIDINE SAMPLER SUMMARY #########")
    print(f"Input PDB                    : {args.input_pdb}")
    print(f"Files wrote to               : {out_dir}")
    print(f"Residues sampled             : {list(allowed_map.keys())}")
    print(f"Pivots used                  : {pivots_used}"
          + (f"  (ND1-locked: {nd_residues})" if nd_residues else ""))
    print(f"A-count cap                  : "
          f"{sorted(allowed_D) if allowed_D is not None else 'none'}")
    print(f"Total permutations identified: {total_perms}")
    print(f"Filtered out (A-count cap)   : {filtered_perms}")
    print(f"Permutations written         : {generated_perms}")
    print(f"Per-code usage (written)     : "
          + ", ".join(f"{c}:{code_counts[c]}" for c in CANONICAL))
    print(f"#############################################")


if __name__ == "__main__":
    main()
