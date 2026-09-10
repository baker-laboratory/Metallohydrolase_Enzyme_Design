#!/usr/bin/env python3
"""
Author: Seth M. Woodbury
SCRIPT NAME
    theozyme_cat_residue_enumerative_sampler__STEP2_glu_asp_sampler.py

DESCRIPTION
    Take a single PDB and enumerate all combinations of glutamate (E) vs.
    aspartate (D) at specified positions.  For each target residue (must
    start as GLU or ASP), you can request:

      E  → keep or convert to GLU (no change if already GLU)
      D  → convert GLU → ASP (drop one CH₂, rename side‑chain atoms,
           realign backbone, restore geometry)

    One output PDB is written per permutation, with a suffix like “_EDE”.

ARGUMENTS
    --input_pdb PATH
        Path to the input PDB file containing GLU/ASP residues to modify.
    --out_dir PATH
        Directory to write the variant PDBs.  Defaults to the input file’s folder.
    --gluE_aspD_json JSON_OR_PATH
        JSON string or path to JSON file mapping residue identifiers (e.g. "A1")
        to allowed states:
            "E"  = only GLU
            "D"  = only ASP
            "ED" = both GLU and ASP
        Example: '{"A1":"ED","B2":"D","C3":"E"}'
    --verbose
        If set, print each output filename as it is written.

EXAMPLE
    python Scripts/theozyme_and_ligand_handling/\
theozyme_cat_residue_enumerative_sampler__STEP2_glu_asp_sampler.py \
      --input_pdb /path/to/project/organophosphatase/paraoxon/\
theozymes/combined_theozymes/theozyme_ORI/test/\
group1_pte_pxon_ChenJACS_exact_v0__lig_YYE_ORI_01.pdb \
      --out_dir /path/to/project/organophosphatase/paraoxon/\
theozymes/combined_theozymes/theozyme_ORI/test_gluasp_variants \
      --gluE_aspD_json '{"F6":"ED","A1":"D"}' \
      --verbose

POTENTIAL INPUTS
    --gluE_aspD_json '{"F6":"ED"}'
        Permute residue F6 between GLU and ASP.
    --gluE_aspD_json '{"B2":"E"}'
        Force residue B2 to remain or convert to GLU only.
    --gluE_aspD_json '/path/to/config.json'
        Load the same mapping from a JSON file instead of inline.

"""
import re
import os
import sys
import argparse
import json
import copy
import itertools
import numpy as np

##############################################
### REMARK 666 GLU/ASP TOKEN SWAPPER ###
##############################################

def _swap_glu_asp_token_anywhere(line, segment, chain, resno, new3, verbose=False):
    """
    Swap only the GLU/ASP token within 'MATCH <segment> <chain> <GLU|ASP> <resno>'
    anywhere in the line (not just at the start).
    Returns (new_line, num_subs).
    """
    # e.g. "... MATCH MOTIF F GLU  6  6  1"
    # or   "... MATCH TEMPLATE F ASP  6 MATCH MOTIF E HIS  5  5  1"
    pat = rf'(MATCH\s+{segment}\s+{re.escape(chain)}\s+)(GLU|ASP)(\s+{resno}\b)'
    new_line, nsubs = re.subn(pat, rf'\1{new3}\3', line)
    if verbose and nsubs:
        which = f"{segment} {chain}{resno}"
        print(f"[REMARK666] token swap in-line ({which}): {nsubs} occurrence(s)")
    return new_line, nsubs

def update_remark666_swap_tokens(records, changes_map, verbose=False):
    """
    records: list from parse_pdb_lines (strings or Atom objects)
    changes_map: dict like {'F6':'ASP', 'B2':'GLU'} for residues that changed.
    Returns number of REMARK 666 lines modified (per-line count).
    """
    if not changes_map:
        return 0

    updated_lines = 0
    for i, rec in enumerate(records):
        if not (isinstance(rec, str) and rec.startswith("REMARK 666")):
            continue

        before = rec
        after  = rec
        changed_any = False

        for key, new3 in changes_map.items():
            ch, rn = key[0], int(key[1:])

            # Swap within TEMPLATE clause(s) anywhere in the line
            after, n1 = _swap_glu_asp_token_anywhere(after, "TEMPLATE", ch, rn, new3, verbose=verbose)
            # Swap within MOTIF clause(s) anywhere in the line
            after, n2 = _swap_glu_asp_token_anywhere(after, "MOTIF",    ch, rn, new3, verbose=verbose)

            if (n1 + n2) > 0:
                changed_any = True
                if verbose:
                    print(f"[REMARK666] {ch}{rn}: GLU/ASP → {new3} on this line ({n1} TEMPLATE, {n2} MOTIF)")

        if changed_any and after != before:
            records[i] = after
            updated_lines += 1
            if verbose:
                print("[REMARK666] UPDATED LINE")
                print("  before:", before.rstrip("\n"))
                print("  after :", after.rstrip("\n"))

    if verbose and updated_lines == 0:
        print("[REMARK666] No REMARK 666 lines required updates for this variant.")
    return updated_lines

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
        x = float(self._orig[30:38]); y = float(self._orig[38:46]); z = float(self._orig[46:54])
        self.coord = np.array([x, y, z], dtype=float)

    def format_line(self):
        line = self._orig
        # 1) atom name (cols 13–16)
        name_field = f"{self.name:>4}"
        # 2) altLoc (col 17)
        altloc = line[16]
        # 3) residue name (cols 18–20), chain & resnum (21–30)
        resname_field = f"{self.resname:>3}"
        rest = line[20:30]
        # 4) updated coords (cols 31–54)
        x, y, z = self.coord
        coord_fields = f"{x:8.3f}{y:8.3f}{z:8.3f}"
        # 5) remainder (occupancy, etc.)
        tail = line[54:]
        return (
            line[:12]
            + name_field
            + altloc
            + resname_field
            + rest
            + coord_fields
            + tail
            + "\n"
        )


def parse_pdb_lines(pdb_path):
    records = []
    with open(pdb_path) as f:
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


def parse_residue_key(key):
    try:
        return key[0], int(key[1:])
    except (IndexError, ValueError):
        sys.exit(f"Invalid residue key '{key}'. Use chain+resnum, e.g. A244.")


def collect_residue(records, key, input_pdb):
    chain, resnum = parse_residue_key(key)
    idxs = [
        i for i, rec in enumerate(records)
        if isinstance(rec, Atom) and rec.chain == chain and rec.resnum == resnum
    ]
    if not idxs:
        sys.exit(f"{input_pdb}: residue {key} was not found.")

    atoms = [records[i] for i in idxs]
    resnames = sorted({atom.resname for atom in atoms})
    if any(resname not in ('GLU', 'ASP') for resname in resnames):
        found = ", ".join(resnames)
        sys.exit(f"{input_pdb}: residue {key} must be GLU or ASP; found {found}.")
    if len(resnames) != 1:
        found = ", ".join(resnames)
        sys.exit(f"{input_pdb}: residue {key} has mixed residue names: {found}.")

    atom_map = {atom.name: atom for atom in atoms}
    return idxs, atom_map, resnames[0]


def require_atoms(atom_map, required, input_pdb, key, resname):
    missing = sorted(set(required) - set(atom_map))
    if missing:
        miss = ", ".join(missing)
        sys.exit(
            f"{input_pdb}: residue {key} ({resname}) is missing required atom(s): {miss}."
        )

##############################################
### GEOMETRIC HELPERS ###
##############################################

def kabsch_rotation(P, Q):
    """Return rotation matrix that best aligns P->Q via Kabsch."""
    # P, Q shape: (N,3)
    C = P.T @ Q
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(Wt.T @ V.T))
    D = np.diag([1,1,d])
    return Wt.T @ D @ V.T

def compute_dihedral(a, b, c, d):
    # returns dihedral angle (degrees)
    b0 = b - a; b1 = c - b; b2 = d - c
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1)*b1
    w = b2 - np.dot(b2, b1)*b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))


def place_atom(a, b, c, bond, angle_deg, dihedral_deg):
    """NeRF placement: return point D such that
        |C-D| == bond,  angle(B,C,D) == angle_deg,
        dihedral(A,B,C,D) == dihedral_deg.
    """
    theta = np.radians(angle_deg)
    chi = np.radians(dihedral_deg)
    bc = c - b
    bc /= np.linalg.norm(bc)
    n = np.cross(b - a, bc)
    n /= np.linalg.norm(n)
    nbc = np.cross(n, bc)
    d = np.array([
        -bond * np.cos(theta),
        bond * np.sin(theta) * np.cos(chi),
        bond * np.sin(theta) * np.sin(chi),
    ])
    M = np.column_stack([bc, nbc, n])
    return c + M.dot(d)


def rotation_between(a, b):
    """Minimal rotation matrix taking unit-ish vector `a` onto `b`."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-8:
        if c > 0:
            return np.eye(3)                       # already aligned
        # antiparallel: 180 deg about any axis perpendicular to a
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(ref, a)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, ref)
        axis /= np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return np.eye(3) + 2.0 * (K @ K)            # Rodrigues, theta = pi
    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


##############################################
### SIDECHAIN TRANSFORMS ###
##############################################

def transform_glu_to_asp(res_atoms, orig_coords, records, residue_indices):
    """
    Convert GLU->ASP in place.

    res_atoms: dict from atom name to Atom() for this residue
    orig_coords: dict of original CB, CA, N, C, O coords (np.array)
    records:    the full list of parsed records (Atom or raw line)
    residue_indices: list of indices into `records` for this residue
    """
    src_CB = res_atoms['CB'].coord.copy()

    # 1) rename sidechain atoms
    if 'OE2' in res_atoms:
        res_atoms['OE2'].name = 'OD2'
    if 'OE1' in res_atoms:
        res_atoms['OE1'].name = 'OD1'
    if 'CD' in res_atoms:
        res_atoms['CD'].name = 'CG'
    if 'HE2' in res_atoms:
        res_atoms['HE2'].name = 'HD2'
    if 'HE1' in res_atoms:
        res_atoms['HE1'].name = 'HD1'

    # The original GLU CG becomes the ASP CB, so its hydrogens follow it.
    if 'HG2' in res_atoms:
        res_atoms['HG2'].name = 'HB2'
    if 'HG3' in res_atoms:
        res_atoms['HG3'].name = 'HB3'
    if 'HG1' in res_atoms:
        res_atoms['HG1'].name = 'HB1'

    # 2) delete the original CB and its hydrogens from the records list
    #    and from our dict mapping
    delete_atoms = []
    for name in ('CB', 'HB1', 'HB2', 'HB3'):
        if name in res_atoms:
            delete_atoms.append(res_atoms.pop(name))
    if delete_atoms:
        delete_ids = {id(atom) for atom in delete_atoms}
        for idx in sorted(residue_indices, reverse=True):
            if id(records[idx]) in delete_ids:
                del records[idx]

    # 3) rigid‐body align so that
    #    new CA → old CB  and  new CG → old CG
    # (assumes orig_coords now also has 'CG')

    # source anchors (pre‐move)
    src_CA = res_atoms['CA'].coord.copy()
    # target anchors (from original)
    tgt_CB = orig_coords['CB']
    tgt_CG = orig_coords['CG']

    # build 2×3 point‐sets and center them
    P = np.vstack([src_CA, src_CB])
    Q = np.vstack([tgt_CB, tgt_CG])
    P_cent = P.mean(axis=0)
    Q_cent = Q.mean(axis=0)
    P0 = P - P_cent
    Q0 = Q - Q_cent

    # compute rotation and translation
    R = kabsch_rotation(P0, Q0)
    t = Q_cent - R.dot(P_cent)

    # apply the rigid‐body (R,t) to backbone atoms
    for name in ('CA','C','N','O'):
        res_atoms[name].coord = R.dot(res_atoms[name].coord) + t

    # 4) rename the CG → new CB
    if 'CG' in res_atoms:
        cg = res_atoms.pop('CG')
        cg.name = 'CB'
        res_atoms['CB'] = cg

    # 5) set residue name on all remaining atoms
    for atom in res_atoms.values():
        atom.resname = 'ASP'

    ### MAYBE GOOD ENOUGH WITHOUT THIS ###
    # 6) restore the original CG–CB–CA–C dihedral by rotating C,N,O about the CB→CA axis

    # original dihedral from the GLU
    orig_phi = compute_dihedral(
        orig_coords['CG'],
        orig_coords['CB'],
        orig_coords['CA'],
        orig_coords['C']
    )
    # current dihedral after our rigid‐body move (we still use orig CG as “a”)
    curr_phi = compute_dihedral(
        orig_coords['CG'],
        res_atoms['CB'].coord,
        res_atoms['CA'].coord,
        res_atoms['C'].coord
    )
    # angle we need to rotate by
    dphi = orig_phi - curr_phi

    # build a Rodrigues rotation about the CB→CA axis
    axis = res_atoms['CA'].coord - res_atoms['CB'].coord
    axis = axis / np.linalg.norm(axis)
    θ = np.radians(dphi)
    ux, uy, uz = axis
    K = np.array([
        [   0, -uz,  uy],
        [  uz,   0, -ux],
        [ -uy,  ux,   0]
    ])
    R_dih = np.eye(3) + np.sin(θ)*K + (1 - np.cos(θ))*(K @ K)

    # apply it to the remaining backbone atoms
    ca = res_atoms['CA'].coord
    for name in ('C','N','O'):
        v = res_atoms[name].coord - ca
        res_atoms[name].coord = ca + R_dih.dot(v)
    ####################################################

def transform_asp_to_glu(res_atoms, orig_coords, records, residue_indices):
    """
    Convert ASP -> GLU in place (inserts one methylene; inverse of
    transform_glu_to_asp).

    Construction (per user spec):
      * ASP carboxylate is kept FIXED in space:
          ASP CG  -> GLU CD ,  ASP OD1/OD2 -> GLU OE1/OE2
      * ASP CB    -> GLU CG  (kept fixed; its HB2/HB3 -> HG2/HG3)
      * NEW GLU CB is placed at the OLD ASP CA position.
      * The backbone (N, CA, C, O + their H's) is rigidly TRANSLATED
        (conformation unchanged) one C-C bond beyond the new CB, along
        the GLU(CG -> CB) axis, so CA sits ~1.52 A past the new CB.
      * Two new HB2/HB3 hydrogens are built on the new CB.

    res_atoms        : dict atom-name -> Atom() for this residue
    orig_coords      : dict of original backbone/CG coords (unused here;
                       kept for signature parity with transform_glu_to_asp)
    records          : full parsed record list (Atom or raw line)
    residue_indices  : indices into `records` for this residue (parity arg)
    """
    CC_BOND = 1.52   # C-C single bond (A)
    CH_BOND = 1.09   # C-H bond (A)

    for need in ('CA', 'CB', 'CG'):
        if need not in res_atoms:
            print(f"[ASP->GLU] residue missing {need}; skipping transform.")
            return

    # --- capture pre-change coordinates ---
    asp_CA = res_atoms['CA'].coord.copy()
    asp_CB = res_atoms['CB'].coord.copy()      # becomes GLU CG (fixed)
    asp_CG = res_atoms['CG'].coord.copy()      # becomes GLU CD (fixed)

    # --- 1) rename side-chain heavy atoms (carboxylate kept fixed) ---
    if 'OD2' in res_atoms:
        res_atoms['OD2'].name = 'OE2'
    if 'OD1' in res_atoms:
        res_atoms['OD1'].name = 'OE1'
    if 'CG' in res_atoms:
        res_atoms['CG'].name = 'CD'            # ASP carboxyl C -> GLU CD
    res_atoms['CB'].name = 'CG'                 # ASP CB -> GLU CG (fixed)

    # ASP CB hydrogens now belong to GLU CG
    if 'HB2' in res_atoms:
        res_atoms['HB2'].name = 'HG2'
    if 'HB3' in res_atoms:
        res_atoms['HB3'].name = 'HG3'
    if 'HB1' in res_atoms:
        res_atoms['HB1'].name = 'HG1'
    # carboxyl proton (protonated ASP) -> GLU naming
    if 'HD2' in res_atoms:
        res_atoms['HD2'].name = 'HE2'
    if 'HD1' in res_atoms:
        res_atoms['HD1'].name = 'HE1'

    # --- 2) new GLU CB stays put at the old ASP CA position (NOT moved).
    #        Reposition CA (and the backbone, rigidly) so that the new CB is
    #        TETRAHEDRAL: angle (GLU CG)-(GLU CB)-(CA) ~= 109.5 deg, with the
    #        side chain in an extended (anti) chi, off the FIXED CG/CD. ---
    new_CB_coord = asp_CA.copy()               # GLU CB == old ASP CA (fixed)
    glu_CG = asp_CB                            # GLU CG == old ASP CB (fixed)
    glu_CD = asp_CG                            # GLU CD == old ASP CG (fixed)
    new_CA_coord = place_atom(
        glu_CD, glu_CG, new_CB_coord,
        bond=CC_BOND, angle_deg=109.5, dihedral_deg=180.0)
    if not np.all(np.isfinite(new_CA_coord)):
        # geometric degeneracy fallback: straight extension (old behavior)
        axis = asp_CA - asp_CB
        axis /= np.linalg.norm(axis)
        new_CA_coord = new_CB_coord + CC_BOND * axis
    # The original ASP CA is already a valid sp3 center (N, C, HA, CB_asp).
    # Rigidly move the whole backbone block so CA -> new_CA_coord AND its
    # original CA->CB direction is rotated onto the new CA->CB direction.
    # Because the block is rigid and internally tetrahedral, this makes
    # N-CA-CB, C-CA-CB and HA-CA-CB all correct automatically.
    old_dir = asp_CB - asp_CA                   # original ASP CA -> CB
    new_dir = new_CB_coord - new_CA_coord       # desired GLU  CA -> CB
    R_bb = rotation_between(old_dir, new_dir)

    backbone_names = ('N', 'CA', 'C', 'O', 'OXT',
                      'H', 'H1', 'H2', 'H3', 'HA', 'HA2', 'HA3', 'HXT')
    for nm in backbone_names:
        if nm in res_atoms:
            res_atoms[nm].coord = new_CA_coord + R_bb.dot(
                res_atoms[nm].coord - asp_CA)

    # --- 3) build two HB2/HB3 hydrogens on the new CB ---
    # CB neighbors: GLU CG (= asp_CB, fixed) and CA (= new_CA_coord).
    b1 = asp_CB - new_CB_coord                 # CB -> CG
    b2 = new_CA_coord - new_CB_coord           # CB -> CA
    b1 /= np.linalg.norm(b1)
    b2 /= np.linalg.norm(b2)
    perp = np.cross(b1, b2)
    if np.linalg.norm(perp) < 1e-6:
        # CA, CB, CG colinear (this construction makes them ~linear):
        # pick any axis-perpendicular direction.
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(ref, b1)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        perp = np.cross(b1, ref)
    perp /= np.linalg.norm(perp)
    bis = b1 + b2
    bn = np.linalg.norm(bis)
    if bn < 1e-6:                               # antiparallel -> use perp only
        in_plane = np.cross(perp, b1)
        in_plane /= np.linalg.norm(in_plane)
        half = np.radians(54.75)               # ~half tetrahedral
        h1 = new_CB_coord + CH_BOND * (np.cos(half) * perp
                                       + np.sin(half) * in_plane)
        h2 = new_CB_coord + CH_BOND * (np.cos(half) * (-perp)
                                       + np.sin(half) * in_plane)
    else:
        bis /= bn
        half = np.radians(54.75)
        h1 = new_CB_coord + CH_BOND * (-np.cos(half) * bis
                                       + np.sin(half) * perp)
        h2 = new_CB_coord + CH_BOND * (-np.cos(half) * bis
                                       - np.sin(half) * perp)

    # --- 4) create the new Atom objects (templates carry valid columns) ---
    c_template = res_atoms['CA']               # carbon record template
    h_template = res_atoms.get('HA') or next(
        (a for a in res_atoms.values() if a.name.startswith('H')), None)

    new_CB = copy.deepcopy(c_template)
    new_CB.name = 'CB'
    new_CB.coord = new_CB_coord

    new_atoms = [new_CB]
    if h_template is not None:
        hb2 = copy.deepcopy(h_template); hb2.name = 'HB2'; hb2.coord = h1
        hb3 = copy.deepcopy(h_template); hb3.name = 'HB3'; hb3.coord = h2
        new_atoms += [hb2, hb3]

    # --- 5) set residue name GLU on everything (existing + new) ---
    for atom in res_atoms.values():
        atom.resname = 'GLU'
    for atom in new_atoms:
        atom.resname = 'GLU'

    # --- 6) splice new atoms into `records` right after GLU CG (old CB),
    #        located by object identity so prior edits can't misalign it ---
    cg_atom = res_atoms['CB']                   # object whose .name is now 'CG'
    try:
        insert_at = records.index(cg_atom) + 1
    except ValueError:
        insert_at = (max(residue_indices) + 1) if residue_indices else len(records)
    for off, atom in enumerate(new_atoms):
        records.insert(insert_at + off, atom)

##############################################
### MAIN ###
##############################################

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input_pdb', required=True,
                   help='Path to input PDB')
    p.add_argument('--out_dir', default=None,
                   help='Directory to write output PDBs')
    p.add_argument('--gluE_aspD_json', required=True,
                   help='JSON or string mapping HIS ids to E/D lists, e.g. "{\"A1\":\"ED\"}"')
    p.add_argument('--verbose', action='store_true', default=False,
                   help='Print progress')
    args = p.parse_args()

    # load config
    try:
        cfg = json.loads(args.gluE_aspD_json)
    except ValueError:
        cfg = json.load(open(args.gluE_aspD_json))

    # normalize mapping: each key -> list of 'E' or 'D'
    allowed = {}
    for key, val in cfg.items():
        if isinstance(val, str):
            allowed[key] = list(val)
        else:
            allowed[key] = val

    # parse PDB
    records = parse_pdb_lines(args.input_pdb)

    # output dir
    out_dir = args.out_dir or os.path.dirname(args.input_pdb)
    os.makedirs(out_dir, exist_ok=True)

    # original backbone coords per residue
    orig_coords_map = {}
    base_required = ('N', 'CA', 'C', 'O', 'CB', 'CG')
    for key in allowed:
        # collect coords for CB, CA, N, C, O
        _, atom_map, resname = collect_residue(records, key, args.input_pdb)
        require_atoms(atom_map, base_required, args.input_pdb, key, resname)
        orig_coords_map[key] = {
            'CB': atom_map['CB'].coord.copy(),
            'CA': atom_map['CA'].coord.copy(),
            'N' : atom_map['N'].coord.copy(),
            'C' : atom_map['C'].coord.copy(),
            'O' : atom_map['O'].coord.copy(),
            'CG' : atom_map['CG'].coord.copy()
        }

    # generate permutations
    keys = list(allowed.keys())
    all_perms = itertools.product(*(allowed[k] for k in keys))
    count=0
    # track total vs. generated
    total_perms = 0
    generated_perms = 0
    for perm in all_perms:
        total_perms += 1
        count+=1
        recs = copy.deepcopy(records)

        # Track only residues that truly flipped identity in this variant
        changed_to = {}

        for key, state in zip(keys, perm):
            idxs, sub_atoms, cur_name = collect_residue(recs, key, args.input_pdb)
            require_atoms(sub_atoms, base_required, args.input_pdb, key, cur_name)
            orig = orig_coords_map[key]

            if state=='E':
                # only flip ASP -> GLU
                if cur_name == 'ASP':
                    require_atoms(
                        sub_atoms, ('OD1', 'OD2'), args.input_pdb, key, cur_name
                    )
                    transform_asp_to_glu(sub_atoms, orig, recs, idxs)
                    changed_to[key] = 'GLU' 
            else:  # state == 'D'
                # only flip GLU -> ASP
                if cur_name == 'GLU':
                    require_atoms(
                        sub_atoms, ('CD', 'OE1', 'OE2'), args.input_pdb, key, cur_name
                    )
                    transform_glu_to_asp(sub_atoms, orig, recs, idxs)
                    changed_to[key] = 'ASP'

        # ---- ONLY swap GLU/ASP tokens for implicated REMARK 666 lines ----
        if changed_to:
            nupd = update_remark666_swap_tokens(recs, changed_to, verbose=args.verbose)
            if args.verbose:
                print(f"[REMARK666] Lines modified (token swaps): {nupd}")

        # build suffix
        suffix = ''.join(perm)
        base = os.path.splitext(os.path.basename(args.input_pdb))[0]
        outp = os.path.join(out_dir, f"{base}_{suffix}.pdb")
        write_pdb_records(recs, outp)
        if args.verbose:
            print(f"Wrote {outp}")
        generated_perms += 1

    print(f"\n######### GLU/ASP SAMPLER SUMMARY #########")
    print(f"Files wrote to: {out_dir}")
    print(f"\nTotal permutations considered : {total_perms}")
    print(f"Permutations written        : {generated_perms}")

if __name__=='__main__':
    main()
