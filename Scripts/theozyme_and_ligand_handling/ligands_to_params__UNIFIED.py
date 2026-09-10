#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ligands_to_params__UNIFIED.py
=============================

Unified, centralized ligand -> Rosetta params pipeline.

Extract ligands from one (or more) PDB files and produce Rosetta ``.params``
(+ PDB) files, with full control over HOW ligands are grouped:

  * SEPARATE  (default)  - every distinct ligand instance, keyed by
                           (resname, chainID, resSeq), becomes its own params.
  * NETWORK   group      - merge several instances/resnames into ONE Rosetta
                           residue, drawing pseudobonds (nearest non-H, then
                           fragment stitching) so the merged set is a single
                           connected graph.  e.g. the whole water box -> WAT.
  * CONFORMER group      - several instances of the SAME ligand (within one
                           PDB and/or across --extra_conformer_pdbs) become
                           conformers of each other.  No inter-instance bonds.
                           All instances must share identical composition.

This script ABSORBS (and generalizes) the logic of:
  - extract_ligands_from_SINGLE_pdb_and_create_PARAMS__MODERN.py
  - mol2_with_confs_to_params.py
  - make_FullyBonded_mol2_file_from_singleXYZ_ThatCanHave_multipleXYZinside.py
  - combine_atomlist_from_multipleXYZ_into_singleXYZ_lig.py
  - give_unique_atom_names_to_pdb_lig_v2.py
  - check_if_ligand_3string_code_exists_in_rosetta.py

All of those scripts remain on disk, untouched, as fallbacks.

Backward compatibility
----------------------
All MODERN flags are preserved.  ``--desired_ligand_3letter_code CODE`` with
no ``--group`` / ``--grouping_json`` is internally translated to one wildcard
group ``CODE=*:network`` -> identical lump-all-selected-HETATM behavior.

See docs/superpowers/specs/2026-05-17-unified-ligand-params-design.md.
"""

import argparse
import json
import os
import re
import string
import subprocess
import sys
from dataclasses import dataclass, field
from math import sqrt, inf
from typing import Dict, List, Optional, Sequence, Tuple

# --- locate repo root + shared external paths ---
import sys as _sys
from pathlib import Path as _Path
for _anc in _Path(__file__).resolve().parents:
    if (_anc / "repo_paths.py").is_file():
        _sys.path.insert(0, str(_anc)); break
import repo_paths

###############################################################################
# PATH CONSTANTS (EDIT HERE IF NEEDED)
###############################################################################

OPENBABEL_BIN = repo_paths.OBABEL
MOLFILE_TO_PARAMS_SCRIPT = repo_paths.MOLFILE_TO_PARAMS
DEFAULT_ROSETTA_RESIDUE_TYPES = repo_paths.ROSETTA_RESIDUE_TYPES

VALID_MODES = ("network", "conformer")

###############################################################################
# DATA MODEL
###############################################################################


@dataclass
class PdbAtom:
    name: str
    resname: str
    chain: str
    resseq: int
    icode: str
    x: float
    y: float
    z: float
    element: str
    raw: str
    order: int
    source: str


@dataclass
class LigandInstance:
    resname: str
    chain: str
    resseq: int
    icode: str
    source: str
    atoms: List[PdbAtom] = field(default_factory=list)

    @property
    def key(self) -> Tuple[str, str, int, str, str]:
        return (self.resname, self.chain, self.resseq, self.icode, self.source)

    @property
    def slug(self) -> str:
        ic = self.icode.strip()
        ch = self.chain.strip() or "_"
        return f"{self.resname}_{ch}{self.resseq}{ic}"


@dataclass
class GroupSpec:
    code: str
    resnames: List[str]
    mode: str = "network"

    @property
    def is_wildcard(self) -> bool:
        return self.resnames == ["*"]


@dataclass
class OutputUnit:
    code: str
    mode: str
    instances: List[LigandInstance] = field(default_factory=list)


class PlanError(Exception):
    """Raised for any planning/validation failure that should abort cleanly."""


###############################################################################
# GROUP / TOKEN PARSING
###############################################################################


def parse_group_spec(spec: str) -> GroupSpec:
    """Parse 'CODE=RES[,RES...][:mode]'.  mode defaults to 'network'."""
    if "=" not in spec:
        raise ValueError(f"--group spec must contain '=': {spec!r}")
    code, rhs = spec.split("=", 1)
    code = code.strip()
    mode = "network"
    if ":" in rhs:
        rhs, mode = rhs.rsplit(":", 1)
        mode = mode.strip().lower()
    if mode not in VALID_MODES:
        raise ValueError(
            f"--group {spec!r}: mode must be one of {VALID_MODES}, got {mode!r}"
        )
    resnames = [r.strip() for r in rhs.split(",") if r.strip()]
    if not code:
        raise ValueError(f"--group {spec!r}: empty output code")
    if not resnames:
        raise ValueError(f"--group {spec!r}: no resnames given")
    return GroupSpec(code=code, resnames=resnames, mode=mode)


def parse_residue_token(token: str) -> Tuple[str, int, str]:
    """Parse '<chain><resSeq><iCode?>' e.g. 'B12' -> ('B', 12, '').

    chain: a single letter/digit, or '_' for a blank chain id.
    resSeq: integer, optionally signed.
    iCode: optional trailing single letter (insertion code).
    """
    m = re.match(r"^([A-Za-z0-9_])?(-?\d+)([A-Za-z])?$", token.strip())
    if not m:
        raise ValueError(
            f"--ignore_residues token {token!r} must be <chain><resSeq>[<iCode>], "
            f"e.g. B12, Z999, _15 (blank chain), A-3, B12A"
        )
    chain = m.group(1) or ""
    if chain == "_":
        chain = ""
    return chain, int(m.group(2)), (m.group(3) or "")


###############################################################################
# ARGPARSE
###############################################################################


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified ligand -> MOL2 -> Rosetta params pipeline "
        "(separate / network-merged / conformer grouping)."
    )

    # --- Preserved MODERN flags (identical legacy behavior) ---
    p.add_argument("--input_single_pdb", required=True,
                   help="Primary PDB (scanned for HETATM ligands).")
    p.add_argument("--ligands_to_extract_via_3letter_code", nargs="*", default=None,
                   help="Optional list of 3-letter codes to extract. "
                        "If omitted, ALL HETATMs are considered.")
    p.add_argument("--output_dir_for_params_stuff", required=True,
                   help="Directory for intermediate and final params/PDB files.")
    p.add_argument("--desired_ligand_3letter_code", default=None,
                   help="LEGACY: with no --group/--grouping_json this becomes "
                        "a single wildcard group CODE=*:network "
                        "(lump-all-selected-HETATM, identical to MODERN.py).")
    p.add_argument("--preserve_pdb_ligand_atom_order", action="store_true",
                   help="Use PDB->MOL2 (retain atom names/order) instead of "
                        "legacy PDB->XYZ->MOL2.")
    p.add_argument("--stop_after_XYZ_is_made", action="store_true",
                   help="Stop after writing per-unit XYZ; print downstream commands.")
    p.add_argument("--stop_after_MOL2_is_made", action="store_true",
                   help="Stop after generating (and bond-fixing) all MOL2 files.")
    p.add_argument("--skip_bond_fix", action="store_true",
                   help="Skip the per-unit bond-fix / connectivity step.")
    p.add_argument("--verbose", action="store_true", help="Extra debug prints.")

    # --- New grouping / conformer / ignore flags ---
    p.add_argument("--group", action="append", default=[], dest="group",
                   help="Repeatable. 'CODE=RES[,RES...][:mode]'. "
                        "mode in {network,conformer}, default network. "
                        "RES may be '*' (all otherwise-ungrouped selected).")
    p.add_argument("--grouping_json", default=None,
                   help="JSON file: {CODE: {\"resnames\": [...], "
                        "\"mode\": \"network|conformer\"}}.")
    p.add_argument("--extra_conformer_pdbs", nargs="*", default=[],
                   help="Extra PDBs scanned only for resnames in conformer groups.")
    p.add_argument("--ignore_codes", nargs="*", default=[],
                   help="Drop all HETATM with these 3-letter codes.")
    p.add_argument("--ignore_residues", nargs="*", default=[],
                   help="Drop specific residues; tokens <ChainLetter><resSeq>, e.g. B12 Z999.")
    p.add_argument("--auto_codes", action="store_true",
                   help="Override the ungrouped-multi-instance hard-stop: "
                        "auto-generate unique 3-char codes.")
    p.add_argument("--conformer_topology", choices=("reference", "strict"),
                   default="reference",
                   help="conformer groups: 'reference' (default) perceives "
                        "bonds ONCE from conformer 1 and reuses that exact "
                        "topology for every conformer, varying only "
                        "coordinates (Rosetta-correct: same molecule, "
                        "different geometry). 'strict' bond-fixes each "
                        "conformer independently and hard-fails if the "
                        "perceived topologies diverge.")
    p.add_argument("--allow_rosetta_code_collision", action="store_true",
                   help="Downgrade a Rosetta residue_types.txt collision to a warning.")
    p.add_argument("--rosetta_residue_types", default=DEFAULT_ROSETTA_RESIDUE_TYPES,
                   help="Path to Rosetta residue_types.txt for the collision check. "
                        "Collision check is skipped if the file does not exist.")
    p.add_argument("--molfile_to_params_script", default=MOLFILE_TO_PARAMS_SCRIPT,
                   help=f"Path to Rosetta's molfile_to_params.py "
                        f"(default: {MOLFILE_TO_PARAMS_SCRIPT}).")

    return p.parse_args(argv)


def resolve_groups(args: argparse.Namespace) -> List[GroupSpec]:
    """Merge --group + --grouping_json; apply legacy translation."""
    groups: List[GroupSpec] = []
    seen_codes: Dict[str, str] = {}

    def _add(g: GroupSpec, origin: str) -> None:
        if g.code in seen_codes:
            raise PlanError(
                f"Output code {g.code!r} defined more than once "
                f"({seen_codes[g.code]} and {origin})."
            )
        seen_codes[g.code] = origin
        groups.append(g)

    for spec in args.group:
        _add(parse_group_spec(spec), "--group")

    if args.grouping_json:
        with open(args.grouping_json) as fh:
            data = json.load(fh)
        for code, cfg in data.items():
            resnames = list(cfg["resnames"])
            mode = cfg.get("mode", "network").lower()
            if mode not in VALID_MODES:
                raise ValueError(
                    f"grouping_json {code!r}: mode must be one of {VALID_MODES}"
                )
            _add(GroupSpec(code=code, resnames=resnames, mode=mode), "--grouping_json")

    if not groups and args.desired_ligand_3letter_code:
        groups.append(
            GroupSpec(code=args.desired_ligand_3letter_code,
                      resnames=["*"], mode="network")
        )

    wildcards = [g for g in groups if g.is_wildcard]
    if len(wildcards) > 1:
        raise PlanError("At most one wildcard ('*') group is allowed.")
    return groups


###############################################################################
# PDB PARSING / SELECTION / IGNORE
###############################################################################


# Two-letter elements commonly seen in HETATM ligands/metals/ions.
_TWO_LETTER_ELEMENTS = {
    "CL", "BR", "ZN", "FE", "MG", "MN", "NA", "CA", "CO", "CU", "NI",
    "SE", "SI", "LI", "AL", "AG", "AU", "PT", "PD", "HG", "CD", "MO",
    "CR", "SN", "PB", "BA", "SR", "CS", "RB", "KR", "AR", "NE", "HE",
}


def _element_from_line(line: str) -> str:
    elem = line[76:78].strip()
    if elem:
        # strip charge punctuation/digits, keep letters
        elem = re.sub(r"[^A-Za-z]", "", elem)
    if elem:
        return elem
    name = line[12:16].strip().upper()
    letters = re.match(r"([A-Za-z]+)", name)
    letters = letters.group(1) if letters else ""
    if len(letters) >= 2 and letters[:2] in _TWO_LETTER_ELEMENTS:
        return letters[:2].capitalize()
    return letters[0] if letters else "X"


def read_pdb_hetatms(pdb_path: str, source: Optional[str] = None) -> List[PdbAtom]:
    if source is None:
        source = os.path.basename(pdb_path)
    atoms: List[PdbAtom] = []
    order = 0
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("HETATM"):
                continue
            raw = line.rstrip("\n")
            padded = raw.ljust(80)
            atoms.append(PdbAtom(
                name=padded[12:16].strip(),
                resname=padded[17:20].strip(),
                chain=padded[21:22].strip(),
                resseq=int(padded[22:26]),
                icode=padded[26:27].strip(),
                x=float(padded[30:38]),
                y=float(padded[38:46]),
                z=float(padded[46:54]),
                element=_element_from_line(padded),
                raw=raw,
                order=order,
                source=source,
            ))
            order += 1
    return atoms


def select_atoms(atoms: List[PdbAtom],
                 codes: Optional[Sequence[str]]) -> List[PdbAtom]:
    if not codes:
        return atoms
    wanted = set(codes)
    return [a for a in atoms if a.resname in wanted]


def apply_ignore(atoms: List[PdbAtom],
                 ignore_codes: Sequence[str],
                 ignore_residues: Sequence[Tuple[str, int]]) -> List[PdbAtom]:
    bad_codes = set(ignore_codes)
    # Normalize tokens to (chain, resseq, icode); accept legacy 2-tuples.
    bad_res = set()
    for entry in ignore_residues:
        if len(entry) == 2:
            bad_res.add((entry[0], entry[1], ""))
        else:
            bad_res.add((entry[0], entry[1], entry[2]))
    return [
        a for a in atoms
        if a.resname not in bad_codes
        and (a.chain, a.resseq, a.icode) not in bad_res
    ]


###############################################################################
# INSTANCE BUILDER + PLANNER
###############################################################################


def build_instances(atoms: List[PdbAtom]) -> List[LigandInstance]:
    order: List[Tuple] = []
    by_key: Dict[Tuple, LigandInstance] = {}
    for a in atoms:
        key = (a.resname, a.chain, a.resseq, a.icode, a.source)
        if key not in by_key:
            by_key[key] = LigandInstance(
                a.resname, a.chain, a.resseq, a.icode, a.source)
            order.append(key)
        by_key[key].atoms.append(a)
    return [by_key[k] for k in order]


_AUTO_ALPHABET = string.digits + string.ascii_uppercase


def _auto_code(resname: str, idx: int, used: set) -> str:
    base = (resname[:2].upper() + "X")[:2]
    suffix_pool = _AUTO_ALPHABET
    i = idx
    while True:
        suffix = suffix_pool[i % len(suffix_pool)]
        cand = (base + suffix)[:3].upper()
        cand = cand.ljust(3, "X")
        if cand not in used and re.match(r"^[A-Za-z0-9]{3}$", cand):
            used.add(cand)
            return cand
        i += 1


def _multi_instance_message(resname: str, n: int) -> str:
    return (
        f"Resname {resname!r} has {n} ungrouped instances, but Rosetta needs a "
        f"unique 3-char code per residue. Choose one:\n"
        f"  --group \"{resname[:3].upper()}={resname}:network\"     "
        f"(merge all into ONE residue, pseudobonds drawn within it)\n"
        f"  --group \"{resname[:3].upper()}={resname}:conformer\"   "
        f"(treat the instances as conformers of each other)\n"
        f"  --ignore_codes {resname}                          (drop them entirely)\n"
        f"  --auto_codes                                       "
        f"(force unique auto-generated 3-char codes)"
    )


def plan_output_units(instances: List[LigandInstance],
                      groups: List[GroupSpec],
                      auto_codes: bool) -> List[OutputUnit]:
    """Map ligand instances to the Rosetta residues that will be produced."""
    non_wild = [g for g in groups if not g.is_wildcard]
    wild = next((g for g in groups if g.is_wildcard), None)

    # resname -> group, rejecting overlaps
    resname_to_group: Dict[str, GroupSpec] = {}
    for g in non_wild:
        for rn in g.resnames:
            if rn in resname_to_group:
                raise PlanError(
                    f"Resname {rn!r} assigned to multiple groups "
                    f"({resname_to_group[rn].code} and {g.code})."
                )
            resname_to_group[rn] = g

    units: List[OutputUnit] = []
    used_codes: set = set()

    # 1) explicit non-wildcard groups (deterministic order = declaration order)
    for g in non_wild:
        members = [i for i in instances if i.resname in g.resnames]
        if not members:
            continue
        used_codes.add(g.code)
        units.append(OutputUnit(code=g.code, mode=g.mode, instances=members))

    # 2) everything not consumed by an explicit group
    remaining = [i for i in instances if i.resname not in resname_to_group]

    if wild is not None:
        if remaining:
            used_codes.add(wild.code)
            units.append(
                OutputUnit(code=wild.code, mode=wild.mode, instances=list(remaining))
            )
        return units

    # 3) no wildcard: ungrouped instances stay SEPARATE (default)
    by_resname: Dict[str, List[LigandInstance]] = {}
    res_order: List[str] = []
    for inst in remaining:
        if inst.resname not in by_resname:
            by_resname[inst.resname] = []
            res_order.append(inst.resname)
        by_resname[inst.resname].append(inst)

    for rn in res_order:
        insts = by_resname[rn]
        if len(insts) == 1:
            code = rn[:3].upper().ljust(3, "X")
            used_codes.add(code)
            units.append(OutputUnit(code=code, mode="network", instances=insts))
        else:
            if not auto_codes:
                raise PlanError(_multi_instance_message(rn, len(insts)))
            for idx, inst in enumerate(insts):
                code = _auto_code(rn, idx, used_codes)
                units.append(
                    OutputUnit(code=code, mode="network", instances=[inst])
                )
    return units


###############################################################################
# CODE VALIDATION / ATOM-NAME UNIQUIFICATION
###############################################################################


def validate_codes(codes: Sequence[str],
                    rosetta_txt: Optional[str],
                    allow_collision: bool) -> None:
    seen = set()
    for c in codes:
        if not re.match(r"^[A-Za-z0-9]{3}$", c):
            raise PlanError(
                f"Ligand code {c!r} must be exactly 3 alphanumeric characters."
            )
        if c in seen:
            raise PlanError(f"Duplicate output ligand code {c!r} within this run.")
        seen.add(c)

    if not rosetta_txt or not os.path.isfile(rosetta_txt):
        return

    with open(rosetta_txt) as fh:
        text = fh.read()
    for c in codes:
        hit = (re.search(rf"{re.escape(c)}\.params\b", text)
               or re.search(rf"(?<![A-Za-z0-9]){re.escape(c)}(?![A-Za-z0-9])", text))
        if hit:
            msg = (f"Ligand code {c!r} already exists in Rosetta "
                   f"({rosetta_txt}). Pick a different code.")
            if allow_collision:
                print(f"[WARNING] {msg} (continuing: --allow_rosetta_code_collision)")
            else:
                raise PlanError(msg)


def uniquify_atom_names(names: Sequence[str],
                        elements: Sequence[str]
                        ) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Make names unique within a residue.  No-op if already unique."""
    if len(set(names)) == len(names):
        return list(names), [(n, n) for n in names]

    def _b36(n: int) -> str:
        if n < len(_AUTO_ALPHABET):
            return _AUTO_ALPHABET[n]
        s = ""
        while n:
            s = _AUTO_ALPHABET[n % 36] + s
            n //= 36
        return s

    counters: Dict[str, int] = {}
    new_names: List[str] = []
    for el in elements:
        el = (el or "X").upper()[:2]
        counters[el] = counters.get(el, 0) + 1
        n = counters[el]
        cand = f"{el}{n}"
        if len(cand) > 4:                      # PDB/Rosetta atom name is 4 cols
            cand = f"{el}{_b36(n - 1)}"
        if len(cand) > 4:
            raise PlanError(
                f"Cannot generate a <=4-char unique atom name for element "
                f"{el!r} (index {n}); too many atoms of this element to "
                f"uniquify within one residue."
            )
        new_names.append(cand)
    return new_names, list(zip(names, new_names))


###############################################################################
# INPUT WRITERS
###############################################################################


def _render_pdb_line(atom: PdbAtom, new_name: Optional[str] = None) -> str:
    raw = atom.raw.ljust(80)
    if new_name and new_name != atom.name:
        raw = raw[:12] + new_name.ljust(4) + raw[16:]
    return raw.rstrip()


def write_xyz(atoms: Sequence[PdbAtom], path: str,
              names: Optional[Sequence[str]] = None) -> None:
    with open(path, "w") as out:
        out.write(f"{len(atoms)}\n")
        out.write("Extracted ligand XYZ\n")
        for a in atoms:
            out.write(f"{a.element:2s} {a.x:12.6f} {a.y:12.6f} {a.z:12.6f}\n")


def write_ligand_pdb(atoms: Sequence[PdbAtom], path: str,
                     names: Optional[Sequence[str]] = None) -> None:
    with open(path, "w") as out:
        out.write("REMARK  Generated ligand-only PDB (ligands_to_params__UNIFIED)\n")
        for idx, a in enumerate(atoms):
            nm = names[idx] if names is not None else None
            out.write(_render_pdb_line(a, nm) + "\n")
        out.write("END\n")


def write_name_map(mapping: Sequence[Tuple[str, str]], path: str) -> None:
    with open(path, "w") as out:
        out.write("# old_atom_name  new_atom_name\n")
        for old, new in mapping:
            out.write(f"{old}\t{new}\n")


###############################################################################
# MOL2 PARSING / BOND-FIX  (ported from MODERN; scoped to a single unit file)
###############################################################################


def standardize_residue_labels(mol2_file, resid=1, resname="UNL1"):
    with open(mol2_file) as f:
        lines = f.readlines()
    out = []
    in_atom = False
    for line in lines:
        if line.startswith("@<TRIPOS>ATOM"):
            in_atom = True
            out.append(line)
            continue
        if line.startswith("@<TRIPOS>"):
            in_atom = False
            out.append(line)
            continue
        if in_atom and line.strip():
            fields = line.split()
            if len(fields) < 9:
                out.append(line)
                continue
            fields[6] = str(resid)
            fields[7] = resname
            out.append(
                f"{int(fields[0]):>7d} "
                f"{fields[1]:<8s}"
                f"{float(fields[2]):>10.4f}"
                f"{float(fields[3]):>10.4f}"
                f"{float(fields[4]):>10.4f} "
                f"{fields[5]:<6s} "
                f"{int(fields[6]):>3d} "
                f"{fields[7]:<6s} "
                f"{float(fields[8]):>10.4f}\n"
            )
        else:
            out.append(line)
    with open(mol2_file, "w") as f:
        f.writelines(out)


def parse_mol2(mol2_file):
    molecules = []
    current = {"atoms": {}, "bonds": []}
    with open(mol2_file) as file:
        in_atom = in_bond = False
        for line in file:
            if line.startswith("@<TRIPOS>MOLECULE"):
                if current["atoms"]:
                    molecules.append(current)
                current = {"atoms": {}, "bonds": []}
                in_atom = in_bond = False
            elif line.startswith("@<TRIPOS>ATOM"):
                in_atom, in_bond = True, False
            elif line.startswith("@<TRIPOS>BOND"):
                in_atom, in_bond = False, True
            elif line.startswith("@<TRIPOS>"):
                in_atom = in_bond = False
            elif in_atom:
                parts = line.split()
                if len(parts) >= 6:
                    aid = int(parts[0])
                    current["atoms"][aid] = {
                        "name": parts[1],
                        "coords": tuple(map(float, parts[2:5])),
                        "element": parts[5],
                        "bonds": [],
                    }
            elif in_bond:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        a1, a2 = int(parts[1]), int(parts[2])
                    except ValueError:
                        continue
                    if a1 in current["atoms"] and a2 in current["atoms"]:
                        current["bonds"].append((a1, a2))
                        current["atoms"][a1]["bonds"].append(a2)
                        current["atoms"][a2]["bonds"].append(a1)
        if current["atoms"]:
            molecules.append(current)
    return molecules


def find_nearest_heteroatom(target_atom, atoms):
    min_dist = inf
    nearest = None
    tx, ty, tz = target_atom["coords"]
    for oid, other in atoms.items():
        if other is target_atom or other["element"].upper() == "H":
            continue
        ox, oy, oz = other["coords"]
        d = sqrt((tx - ox) ** 2 + (ty - oy) ** 2 + (tz - oz) ** 2)
        if d < min_dist:
            min_dist, nearest = d, oid
    return nearest


def gather_new_bonds(molecules):
    new_bonds_dict = {}
    for i, mol in enumerate(molecules, start=1):
        new_bonds = []
        atoms = mol["atoms"]
        for aid, atom in atoms.items():
            if not atom["bonds"]:
                nid = find_nearest_heteroatom(atom, atoms)
                if nid:
                    atom["bonds"].append(nid)
                    atoms[nid]["bonds"].append(aid)
                    mol["bonds"].append((aid, nid))
                    new_bonds.append((aid, nid))
                    print(f"  [bond-fix] atom {aid} ({atom['name']}) -> {nid}")
        new_bonds_dict[i] = new_bonds
    return new_bonds_dict


def connect_fragments(molecules, new_bonds_dict):
    for idx, mol in enumerate(molecules, start=1):
        atoms = mol["atoms"]
        visited = set()
        components = []
        for a0 in atoms:
            if a0 in visited:
                continue
            stack, comp = [a0], set()
            while stack:
                a = stack.pop()
                if a in visited:
                    continue
                visited.add(a)
                comp.add(a)
                for nbr in atoms[a]["bonds"]:
                    if nbr not in visited:
                        stack.append(nbr)
            components.append(comp)
        if len(components) > 1:
            for compA, compB in zip(components, components[1:]):
                best_pair, best_d2 = None, inf
                for a in compA:
                    if atoms[a]["element"].upper() == "H":
                        continue
                    x1, y1, z1 = atoms[a]["coords"]
                    for b in compB:
                        if atoms[b]["element"].upper() == "H":
                            continue
                        x2, y2, z2 = atoms[b]["coords"]
                        d2 = (x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2
                        if d2 < best_d2:
                            best_d2, best_pair = d2, (a, b)
                if best_pair:
                    a, b = best_pair
                    atoms[a]["bonds"].append(b)
                    atoms[b]["bonds"].append(a)
                    mol["bonds"].append((a, b))
                    new_bonds_dict[idx].append((a, b))
                    print(f"  [bond-fix] connected fragment of mol {idx}: {a}-{b}")


def partial_update_mol2(mol2_file, new_bonds_dict):
    with open(mol2_file) as f:
        lines = f.readlines()
    updated = []
    i, total, mol_idx = 0, len(lines), 0
    bond_re = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(.*)")
    while i < total:
        line = lines[i]
        s = line.strip()
        if s.startswith("@<TRIPOS>MOLECULE"):
            mol_idx += 1
            updated.append(line)
            i += 1
            if i < total:
                updated.append(lines[i])
                i += 1
            else:
                break
            if i < total:
                count_line = lines[i]
                parts = count_line.strip().split()
                if mol_idx in new_bonds_dict and new_bonds_dict[mol_idx]:
                    try:
                        parts[1] = str(int(parts[1]) + len(new_bonds_dict[mol_idx]))
                        updated.append(" " + " ".join(parts) + "\n")
                    except (IndexError, ValueError):
                        updated.append(count_line)
                else:
                    updated.append(count_line)
                i += 1
            else:
                break
            continue
        elif s.startswith("@<TRIPOS>BOND"):
            updated.append(line)
            i += 1
            bond_lines, last_id = [], 0
            while i < total and not lines[i].strip().startswith("@<TRIPOS>"):
                bond_lines.append(lines[i])
                m = bond_re.match(lines[i])
                if m:
                    last_id = max(last_id, int(m.group(1)))
                i += 1
            updated.extend(bond_lines)
            for (a1, a2) in new_bonds_dict.get(mol_idx, []):
                last_id += 1
                updated.append(f" {last_id:>5} {a1:>5} {a2:>5} {1:>4}\n")
            continue
        updated.append(line)
        i += 1
    with open(mol2_file, "w") as f:
        f.writelines(updated)
    print(f"[bond-fix] wrote {mol2_file}")


def bond_fix_mol2(mol2_file: str) -> None:
    """Single-unit MOL2 connectivity fix.  Scope == this file only, so
    pseudobonds are never drawn across distinct OutputUnits."""
    molecules = parse_mol2(mol2_file)
    print(f"[bond-fix] parsed {len(molecules)} molecule block(s) from {mol2_file}")
    new_bonds_dict = gather_new_bonds(molecules)
    connect_fragments(molecules, new_bonds_dict)
    partial_update_mol2(mol2_file, new_bonds_dict)
    standardize_residue_labels(mol2_file)


###############################################################################
# CONFORMER COMPOSITION VALIDATION + ASSEMBLY
###############################################################################


def _mol_signature(elements: Sequence[str],
                    names: Sequence[str]) -> Tuple:
    return (len(elements), tuple(elements), tuple(names))


def assert_conformers_compatible(items: List[Tuple[str, Tuple]]) -> None:
    """items = [(label, signature), ...].  Hard-fail on first mismatch."""
    if not items:
        raise PlanError("Conformer set has no instances.")
    ref_label, ref = items[0]
    for label, sig in items[1:]:
        if sig[0] != ref[0]:
            raise PlanError(
                f"Conformer set mismatch: {ref_label} has {ref[0]} atoms, "
                f"{label} has {sig[0]} atoms."
            )
        if sig[1] != ref[1]:
            raise PlanError(
                f"Conformer set mismatch: element sequence differs between "
                f"{ref_label} and {label}."
            )
        if sig[2] != ref[2]:
            raise PlanError(
                f"Conformer set mismatch: atom names/order differ between "
                f"{ref_label} and {label}."
            )


def mol2_molecule_block_count(mol2_file: str) -> int:
    n = 0
    with open(mol2_file) as fh:
        for line in fh:
            if line.startswith("@<TRIPOS>MOLECULE"):
                n += 1
    return n


def assert_single_block(mol2_file: str, context: str) -> None:
    n = mol2_molecule_block_count(mol2_file)
    if n != 1:
        raise PlanError(
            f"{context}: Open Babel produced {n} @<TRIPOS>MOLECULE blocks in "
            f"{mol2_file} (expected exactly 1). The selection likely contains "
            f"multiple disconnected structures that cannot be one Rosetta "
            f"residue. Split it, or group it explicitly."
        )


def _match_by_coords(mol2_xyz: Tuple[float, float, float],
                      coords: Sequence[Tuple[float, float, float]],
                      used: set, tol2: float = 2.5e-3) -> Optional[int]:
    """Return the index of the (still unused) input coordinate closest to a
    MOL2 atom's coordinate, or None if nothing is within tolerance.
    Open Babel preserves coordinates exactly (it may reorder atoms), so a
    near-zero-distance match is a reliable atom identity."""
    mx, my, mz = mol2_xyz
    best_i, best_d2 = None, inf
    for i, (x, y, z) in enumerate(coords):
        if i in used:
            continue
        d2 = (mx - x) ** 2 + (my - y) ** 2 + (mz - z) ** 2
        if d2 < best_d2:
            best_d2, best_i = d2, i
    if best_i is None or best_d2 > tol2:
        return None
    return best_i


def apply_atom_names_to_mol2(
        mol2_file: str,
        names: Sequence[str],
        coords: Optional[Sequence[Tuple[float, float, float]]] = None) -> None:
    """Overwrite ATOM-record atom names (col 2) with `names`.

    Run AFTER Open Babel so planned names actually reach Rosetta (with
    molfile_to_params --keep-names). Requires exactly one MOLECULE block.

    If `coords` is given (input atom coordinates aligned to `names`), each
    MOL2 atom is matched back to its input atom BY COORDINATE, so an Open
    Babel atom reorder cannot silently misassign names/conformer geometry.
    Without `coords` (unit tests only) names are applied in file order."""
    assert_single_block(mol2_file, "apply_atom_names_to_mol2")
    with open(mol2_file) as fh:
        lines = fh.readlines()
    out, in_atom, idx, atom_count = [], False, 0, 0
    used: set = set()
    for line in lines:
        if line.startswith("@<TRIPOS>ATOM"):
            in_atom = True
            out.append(line)
            continue
        if line.startswith("@<TRIPOS>"):
            in_atom = False
            out.append(line)
            continue
        if in_atom and line.strip():
            atom_count += 1
            parts = line.split()
            if len(parts) < 6:
                raise PlanError(
                    f"{mol2_file}: malformed MOL2 ATOM line "
                    f"({len(parts)} fields): {line.strip()!r}")
            if coords is not None:
                mxyz = (float(parts[2]), float(parts[3]), float(parts[4]))
                m = _match_by_coords(mxyz, coords, used)
                if m is None:
                    raise PlanError(
                        f"{mol2_file}: MOL2 atom at {mxyz} has no matching "
                        f"input atom within tolerance; cannot map names "
                        f"safely (Open Babel may have moved atoms).")
                used.add(m)
                parts[1] = names[m]
            else:
                if idx >= len(names):
                    raise PlanError(
                        f"{mol2_file}: more MOL2 atoms than names ({len(names)}).")
                parts[1] = names[idx]
            idx += 1
            subst_id = parts[6] if len(parts) > 6 else "1"
            subst_nm = parts[7] if len(parts) > 7 else "UNL1"
            charge = parts[8] if len(parts) > 8 else "0.0"
            out.append(
                f"{int(parts[0]):>7d} "
                f"{parts[1]:<8s}"
                f"{float(parts[2]):>10.4f}"
                f"{float(parts[3]):>10.4f}"
                f"{float(parts[4]):>10.4f} "
                f"{parts[5]:<6s} "
                f"{int(subst_id):>3d} "
                f"{subst_nm:<6s} "
                f"{float(charge):>10.4f}\n"
            )
        else:
            out.append(line)
    if atom_count != len(names) or idx != len(names):
        raise PlanError(
            f"{mol2_file}: Open Babel emitted {atom_count} atoms but "
            f"{len(names)} were expected; cannot safely preserve atom names."
        )
    with open(mol2_file, "w") as fh:
        fh.writelines(out)


def mol2_topology_signature(mol2_file: str) -> Tuple:
    """Final-MOL2 signature for conformer compatibility: atom count,
    element sequence, atom-name sequence, and undirected bond set."""
    mols = parse_mol2(mol2_file)
    if len(mols) != 1:
        raise PlanError(
            f"{mol2_file}: expected 1 molecule block, got {len(mols)}.")
    m = mols[0]
    aids = sorted(m["atoms"])
    elems = tuple(m["atoms"][i]["element"] for i in aids)
    nm = tuple(m["atoms"][i]["name"] for i in aids)
    bonds = tuple(sorted(tuple(sorted(b)) for b in m["bonds"]))
    return (len(aids), elems, nm, bonds)


def assert_final_conformers_compatible(items: List[Tuple[str, Tuple]]) -> None:
    """items = [(label, mol2_topology_signature), ...]. Validated on the
    FINAL MOL2 (post Open Babel + bond-fix), because Rosetta builds the
    residue from conformer 1's topology and only swaps coordinates."""
    if not items:
        raise PlanError("Conformer set has no instances.")
    ref_label, ref = items[0]
    for label, sig in items[1:]:
        if sig[0] != ref[0]:
            raise PlanError(
                f"Conformer set mismatch (final MOL2): {ref_label} has "
                f"{ref[0]} atoms, {label} has {sig[0]} atoms.")
        if sig[1] != ref[1]:
            raise PlanError(
                f"Conformer set mismatch (final MOL2): element order differs "
                f"between {ref_label} and {label} after Open Babel.")
        if sig[2] != ref[2]:
            raise PlanError(
                f"Conformer set mismatch (final MOL2): atom names/order differ "
                f"between {ref_label} and {label}.")
        if sig[3] != ref[3]:
            raise PlanError(
                f"Conformer set mismatch (final MOL2): bond topology differs "
                f"between {ref_label} and {label} after bond-fix; Rosetta would "
                f"build invalid conformers.")


def build_conformer_from_reference(
        ref_mol2: str,
        coords_by_name: Dict[str, Tuple[float, float, float]],
        out_path: str) -> None:
    """Write a new single-block MOL2 = the reference MOL2 (same atom order,
    names, and BOND topology) but with each atom's X/Y/Z replaced by this
    conformer's coordinates, matched by atom name.

    This is the Rosetta-correct conformer construction: one perceived
    topology (conformer 1), geometry-only variation for the rest."""
    assert_single_block(ref_mol2, "build_conformer_from_reference")
    with open(ref_mol2) as fh:
        lines = fh.readlines()
    out, in_atom, used = [], False, set()
    for line in lines:
        if line.startswith("@<TRIPOS>ATOM"):
            in_atom = True
            out.append(line)
            continue
        if line.startswith("@<TRIPOS>"):
            in_atom = False
            out.append(line)
            continue
        if in_atom and line.strip():
            parts = line.split()
            if len(parts) < 6:
                raise PlanError(
                    f"{ref_mol2}: malformed ATOM line: {line.strip()!r}")
            name = parts[1]
            if name not in coords_by_name:
                raise PlanError(
                    f"{ref_mol2}: reference atom {name!r} has no coordinate "
                    f"in this conformer; conformer atom sets differ.")
            if name in used:
                raise PlanError(
                    f"{ref_mol2}: duplicate atom name {name!r} in reference; "
                    f"cannot map conformer coordinates unambiguously.")
            used.add(name)
            x, y, z = coords_by_name[name]
            subst_id = parts[6] if len(parts) > 6 else "1"
            subst_nm = parts[7] if len(parts) > 7 else "UNL1"
            charge = parts[8] if len(parts) > 8 else "0.0"
            out.append(
                f"{int(parts[0]):>7d} "
                f"{parts[1]:<8s}"
                f"{float(x):>10.4f}"
                f"{float(y):>10.4f}"
                f"{float(z):>10.4f} "
                f"{parts[5]:<6s} "
                f"{int(subst_id):>3d} "
                f"{subst_nm:<6s} "
                f"{float(charge):>10.4f}\n"
            )
        else:
            out.append(line)
    missing = set(coords_by_name) - used
    if missing:
        raise PlanError(
            f"{ref_mol2}: conformer has atoms not in reference: "
            f"{sorted(missing)}")
    with open(out_path, "w") as fh:
        fh.writelines(out)


def concat_mol2_blocks(per_conformer_mol2: List[str], out_path: str) -> None:
    """Concatenate one @<TRIPOS>MOLECULE block per conformer into one MOL2."""
    with open(out_path, "w") as out:
        for mol2 in per_conformer_mol2:
            assert_single_block(mol2, "concat_mol2_blocks")
            with open(mol2) as fh:
                text = fh.read()
            if not text.endswith("\n"):
                text += "\n"
            out.write(text)


###############################################################################
# OPEN BABEL / ROSETTA WRAPPERS
###############################################################################


def run_obabel(input_path: str, input_format: str, out_mol2: str) -> str:
    cmd = [OPENBABEL_BIN, f"-i{input_format}", input_path, "-omol2", "-O", out_mol2]
    print("[obabel]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_mol2


def run_molfile_to_params(code: str, mol2_basename: str, out_dir: str,
                          script_path: str = MOLFILE_TO_PARAMS_SCRIPT) -> None:
    if not os.path.isfile(script_path):
        raise PlanError(
            f"molfile_to_params.py not found at {script_path}. "
            f"Pass --molfile_to_params_script /path/to/molfile_to_params.py "
            f"or update MOLFILE_TO_PARAMS_SCRIPT at the top of this script.")
    cmd = ["python", script_path, "--name", code,
           "--conformers-in-one-file", mol2_basename, "--root_atom=1",
           "--keep-names", "--clobber"]
    print("[rosetta]", " ".join(cmd), "  (cwd=%s)" % out_dir)
    cwd = os.getcwd()
    os.chdir(out_dir)
    try:
        subprocess.run(cmd, check=True)
        main_pdb = f"{code}.pdb"
        conf_pdb = f"{code}_conformers.pdb"
        if os.path.isfile(main_pdb) and os.path.isfile(conf_pdb):
            with open(main_pdb) as f:
                main_lines = f.readlines()
            with open(conf_pdb) as f:
                conf_lines = f.readlines()
            with open(conf_pdb, "w") as f:
                f.writelines(main_lines)
                f.writelines(conf_lines)
            print(f"[rosetta] merged {main_pdb} on top of {conf_pdb}")
        else:
            print(f"[rosetta] WARNING: {main_pdb} or {conf_pdb} missing; no merge")
    finally:
        os.chdir(cwd)


###############################################################################
# MAIN ORCHESTRATION
###############################################################################


def _unit_atoms_in_order(unit: OutputUnit) -> List[PdbAtom]:
    atoms: List[PdbAtom] = []
    for inst in unit.instances:
        atoms.extend(inst.atoms)
    return atoms


def _stop_after_input_stage(args: argparse.Namespace, in_fmt: str) -> bool:
    """stop_after_XYZ means 'stop after the per-unit input file is written'.
    In --preserve_pdb_ligand_atom_order mode there is no XYZ stage, so it
    means stop-after-ligand-PDB (spec §7)."""
    return bool(args.stop_after_XYZ_is_made)


def run_pipeline(args: argparse.Namespace) -> int:
    try:
        groups = resolve_groups(args)
        ignore_res = [parse_residue_token(t) for t in args.ignore_residues]

        primary_path = os.path.abspath(args.input_single_pdb)
        primary = read_pdb_hetatms(primary_path, source=primary_path)
        primary = select_atoms(primary, args.ligands_to_extract_via_3letter_code)
        primary = apply_ignore(primary, args.ignore_codes, ignore_res)
        if not primary:
            raise PlanError(
                "No HETATM atoms passed selection/ignore. "
                f"input={args.input_single_pdb} "
                f"codes={args.ligands_to_extract_via_3letter_code} "
                f"ignore_codes={args.ignore_codes} "
                f"ignore_residues={args.ignore_residues}"
            )

        instances = build_instances(primary)
        units = plan_output_units(instances, groups, args.auto_codes)

        # Conformer units also collect instances from --extra_conformer_pdbs,
        # de-duplicated by full instance key (incl. source path) and never
        # re-reading the primary PDB.
        conf_resnames = {
            inst.resname for u in units if u.mode == "conformer"
            for inst in u.instances
        }
        seen_keys = {inst.key for u in units if u.mode == "conformer"
                     for inst in u.instances}
        if args.extra_conformer_pdbs and conf_resnames:
            for extra in args.extra_conformer_pdbs:
                extra_path = os.path.abspath(extra)
                if extra_path == primary_path:
                    print(f"[WARNING] --extra_conformer_pdbs entry {extra!r} is "
                          f"the primary PDB; skipping to avoid double-counting.")
                    continue
                ex = read_pdb_hetatms(extra_path, source=extra_path)
                ex = select_atoms(ex, list(conf_resnames))
                ex = apply_ignore(ex, args.ignore_codes, ignore_res)
                ex_instances = build_instances(ex)
                for u in units:
                    if u.mode != "conformer":
                        continue
                    u_resnames = {i.resname for i in u.instances}
                    for ei in ex_instances:
                        if ei.resname in u_resnames and ei.key not in seen_keys:
                            seen_keys.add(ei.key)
                            u.instances.append(ei)

        validate_codes([u.code for u in units], args.rosetta_residue_types,
                        args.allow_rosetta_code_collision)

        out_dir = os.path.abspath(args.output_dir_for_params_stuff)
        os.makedirs(out_dir, exist_ok=True)
        in_fmt = "pdb" if args.preserve_pdb_ligand_atom_order else "xyz"

        print("\n[PLAN] %d output unit(s):" % len(units))
        for u in units:
            print(f"  {u.code}  mode={u.mode}  "
                  f"instances={[i.slug for i in u.instances]}")

        for u in units:
            if not u.instances:
                continue

            if u.mode == "conformer":
                # Pre-check (fast) on raw atoms; final check is post-MOL2.
                pre = [(f"{i.source}:{i.slug}",
                        _mol_signature([a.element for a in i.atoms],
                                       [a.name for a in i.atoms]))
                       for i in u.instances]
                assert_conformers_compatible(pre)

                # One canonical, unique atom-name list for every conformer.
                ref = u.instances[0].atoms
                canon_names, mapping = uniquify_atom_names(
                    [a.name for a in ref], [a.element for a in ref])
                if any(o != n for o, n in mapping):
                    write_name_map(mapping, os.path.join(
                        out_dir, f"{u.code}__atom_name_map.txt"))

                # Write every conformer's input file (provenance + stop flag).
                conf_srcs = []
                for ci, inst in enumerate(u.instances):
                    base = os.path.join(out_dir,
                                        f"{u.code}__conf{ci}__{inst.slug}")
                    src = base + (".pdb" if in_fmt == "pdb" else ".xyz")
                    (write_ligand_pdb if in_fmt == "pdb" else write_xyz)(
                        inst.atoms, src, canon_names)
                    conf_srcs.append((base, src, inst))

                if _stop_after_input_stage(args, in_fmt):
                    print(f"[stop_after_XYZ] {u.code}: wrote "
                          f"{len(conf_srcs)} {in_fmt} conformer input(s); "
                          f"stopping.")
                    continue

                # Conformer 0 defines names + (perceived) bond topology.
                base0, src0, inst0 = conf_srcs[0]
                mol20 = run_obabel(src0, in_fmt, base0 + ".mol2")
                assert_single_block(mol20, f"conformer {u.code} #0")
                apply_atom_names_to_mol2(
                    mol20, canon_names, [(a.x, a.y, a.z) for a in inst0.atoms])
                if not args.skip_bond_fix:
                    bond_fix_mol2(mol20)
                per_conf_mol2 = [mol20]

                if args.conformer_topology == "reference":
                    print(f"[INFO] {u.code}: conformer topology = REFERENCE "
                          f"(conformer-1 bonds reused; coords vary).")
                    for ci, (base, src, inst) in enumerate(conf_srcs[1:],
                                                           start=1):
                        coords_by_name = dict(zip(
                            canon_names,
                            [(a.x, a.y, a.z) for a in inst.atoms]))
                        outp = base + ".mol2"
                        build_conformer_from_reference(
                            mol20, coords_by_name, outp)
                        per_conf_mol2.append(outp)
                else:  # strict
                    print(f"[INFO] {u.code}: conformer topology = STRICT "
                          f"(independent perception; must match).")
                    for ci, (base, src, inst) in enumerate(conf_srcs[1:],
                                                           start=1):
                        mol2 = run_obabel(src, in_fmt, base + ".mol2")
                        assert_single_block(mol2, f"conformer {u.code} #{ci}")
                        apply_atom_names_to_mol2(
                            mol2, canon_names,
                            [(a.x, a.y, a.z) for a in inst.atoms])
                        if not args.skip_bond_fix:
                            bond_fix_mol2(mol2)
                        per_conf_mol2.append(mol2)
                    final_sigs = [
                        (os.path.basename(m), mol2_topology_signature(m))
                        for m in per_conf_mol2]
                    assert_final_conformers_compatible(final_sigs)

                combined = os.path.join(out_dir, f"{u.code}.mol2")
                concat_mol2_blocks(per_conf_mol2, combined)
                if args.stop_after_MOL2_is_made:
                    print(f"[stop_after_MOL2] {u.code}: {combined}")
                    continue
                run_molfile_to_params(u.code, os.path.basename(combined), out_dir,
                                       script_path=args.molfile_to_params_script)

            else:  # network / single-instance-separate / legacy wildcard
                atoms = _unit_atoms_in_order(u)
                if args.preserve_pdb_ligand_atom_order:
                    # honor original global HETATM order across merged instances
                    atoms = sorted(atoms, key=lambda a: a.order)
                names = [a.name for a in atoms]
                elements = [a.element for a in atoms]
                new_names, mapping = uniquify_atom_names(names, elements)
                if any(o != n for o, n in mapping):
                    write_name_map(mapping, os.path.join(
                        out_dir, f"{u.code}__atom_name_map.txt"))

                if len(atoms) == 1:
                    print(f"[WARNING] Output unit {u.code!r} is a SINGLE atom "
                          f"({atoms[0].element}). molfile_to_params.py is "
                          f"usually inappropriate for lone metal ions; consider "
                          f"native Rosetta metal handling or grouping it.")

                base = os.path.join(out_dir, f"{u.code}__{u.instances[0].slug}")
                src = base + (".pdb" if in_fmt == "pdb" else ".xyz")
                (write_ligand_pdb if in_fmt == "pdb" else write_xyz)(
                    atoms, src, new_names)

                if _stop_after_input_stage(args, in_fmt):
                    print(f"[stop_after_XYZ] {u.code}: {src}")
                    print(f"  next: obabel -i{in_fmt} {src} -omol2 -O "
                          f"{u.code}.mol2 ; then molfile_to_params.py "
                          f"--name {u.code} --keep-names ...")
                    continue

                mol2 = run_obabel(src, in_fmt,
                                  os.path.join(out_dir, f"{u.code}.mol2"))
                assert_single_block(mol2, f"network unit {u.code}")
                apply_atom_names_to_mol2(
                    mol2, new_names, [(a.x, a.y, a.z) for a in atoms])
                if not args.skip_bond_fix:
                    bond_fix_mol2(mol2)
                if args.stop_after_MOL2_is_made:
                    print(f"[stop_after_MOL2] {u.code}: {mol2}")
                    continue
                run_molfile_to_params(u.code, os.path.basename(mol2), out_dir,
                                       script_path=args.molfile_to_params_script)

        print("\n[DONE] Outputs in:", out_dir)
        return 0

    except PlanError as e:
        print("\n[ERROR] " + str(e), file=sys.stderr)
        return 1


def main() -> None:
    sys.exit(run_pipeline(parse_args()))


if __name__ == "__main__":
    main()
