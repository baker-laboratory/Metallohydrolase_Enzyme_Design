#!/usr/bin/env python3
"""Find designable side chains that hydrogen-bond the active site.

A theozyme pins the catalytic residues, but the residues *around* them often do
real work too -- a serine holding a catalytic histidine in its productive
rotamer, a tyrosine donating to the ligand. Those are designable positions, so
MPNN is free to replace them, and frequently does. Conserving a fraction of them
keeps the second shell that the input structure had, while still letting the
rest of the sequence be redesigned.

The detector is heavy-atom only, because that is what the input structures are.
Rather than requiring explicit hydrogens it gates on the antecedent-donor-
acceptor angle: the antecedent atom sits roughly opposite the donor's hydrogen,
so a real H-bond has the antecedent pointing *away* from the acceptor. Bond
strength is a Gaussian on the heavy-atom distance (d0 = 2.9 A, sigma = 0.35),
which is what the strength bins below are cut from.

Usage is two calls -- find candidates, then choose among them:

    records = find_conservable_sidechain_hbonds(pdb, designable_resnos=...)
    candidates, excluded = select_conservable_resnos(records)
    keep = roll_conserved(candidates, prob=0.8, rng=random.Random(seed))
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

LOGGER = logging.getLogger("mpnn.hbond_conservation")

__all__ = [
    "ConservableHbond",
    "find_conservable_sidechain_hbonds",
    "select_conservable_resnos",
    "normalize_probability",
    "roll_conserved",
    "DEFAULT_MAX_DIST",
    "DEFAULT_MAX_ANGLE_DEG",
]

BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}

DEFAULT_MAX_DIST = 3.9
DEFAULT_MAX_ANGLE_DEG = 90.0   # permissive; a strict detector uses 70
DEFAULT_CLASH_DIST = 1.8

_HBOND_D0 = 2.9
_HBOND_SIGMA = 0.35

# Histidine variants normalized to HIS so both ring nitrogens count as donor and
# acceptor -- the tautomer of a heavy-atom model is unknowable, so assume either.
_HIS_VARIANTS = {"HID", "HIE", "HIP", "HIS_D", "HIS_E", "HISD", "HISE", "HSD", "HSE", "HSP"}

_STRENGTH_BINS: tuple[tuple[float, str], ...] = (
    (0.85, "super_strong"),
    (0.65, "strong"),
    (0.45, "moderate"),
    (0.25, "weak"),
    (0.0, "super_weak"),
)

# Backbone N donates on every residue except proline; backbone O always accepts.
_DONORS_BY_RES: dict[str, set[str]] = {
    "ARG": {"NE", "NH1", "NH2"},
    "ASN": {"ND2"},
    "GLN": {"NE2"},
    "HIS": {"ND1", "NE2"},
    "LYS": {"NZ"},
    "SER": {"OG"},
    "THR": {"OG1"},
    "TYR": {"OH"},
    "TRP": {"NE1"},
    "CYS": {"SG"},
}
_ACCEPTORS_BY_RES: dict[str, set[str]] = {
    "ASN": {"OD1"}, "ASP": {"OD1", "OD2"},
    "GLN": {"OE1"}, "GLU": {"OE1", "OE2"},
    "HIS": {"ND1", "NE2"},
    "SER": {"OG"}, "THR": {"OG1"}, "TYR": {"OH"},
    "MET": {"SD"},
}
_ANTECEDENT: dict[tuple[str, str], str] = {
    ("ARG", "NE"): "CD", ("ARG", "NH1"): "CZ", ("ARG", "NH2"): "CZ",
    ("ASN", "ND2"): "CG", ("GLN", "NE2"): "CD",
    ("LYS", "NZ"): "CE",
    ("SER", "OG"): "CB", ("THR", "OG1"): "CB", ("TYR", "OH"): "CZ",
    ("TRP", "NE1"): "CD1",
    ("CYS", "SG"): "CB",
    ("HIS", "ND1"): "CG", ("HIS", "NE2"): "CD2",
}


@dataclass
class ConservableHbond:
    """One designable-side-chain-to-anchor hydrogen bond."""

    resno: int
    resname: str
    sidechain_atom: str
    partner_kind: str          # "ligand" | "catalytic" | "user_fixed"
    partner_resno: int         # -1 for a ligand partner
    partner_resname: str
    partner_atom: str
    distance: float
    strength: float
    strength_bin: str
    clashes: bool = False
    clash_with: str = ""


def strength_bin(strength: float) -> str:
    for threshold, label in _STRENGTH_BINS:
        if strength >= threshold:
            return label
    return "super_weak"


def _parse_pdb_atoms(pdb_path: str | Path) -> list[dict]:
    """Read ATOM/HETATM records. Hydrogens are dropped -- this is a heavy-atom
    detector, and input models frequently have none anyway."""
    atoms: list[dict] = []
    with open(pdb_path, "r") as fh:
        for line in fh:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            name = line[12:16].strip()
            element = (line[76:78].strip() or name[:1]).upper()
            if element == "H" or name.startswith("H"):
                continue
            try:
                res_seq = int(line[22:26])
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except ValueError:
                continue
            res_name = line[17:20].strip()
            atoms.append({
                "record": "HETATM" if line.startswith("HETATM") else "ATOM",
                "atom_name": name,
                "res_name": "HIS" if res_name in _HIS_VARIANTS else res_name,
                "chain_id": line[21],
                "res_seq": res_seq,
                "element": element,
                "x": x, "y": y, "z": z,
            })
    return atoms


def _atoms_by_res(atoms: list[dict]) -> dict[tuple, dict[str, dict]]:
    out: dict[tuple, dict[str, dict]] = {}
    for a in atoms:
        out.setdefault((a["chain_id"], a["res_seq"]), {})[a["atom_name"]] = a
    return out


def _distance(a: dict, b: dict) -> float:
    dx, dy, dz = a["x"] - b["x"], a["y"] - b["y"], a["z"] - b["z"]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _angle_deg(p_ant: dict, p_donor: dict, p_acc: dict) -> float:
    """Angle at the donor between (donor->antecedent) and (donor->acceptor)."""
    v1 = (p_ant["x"] - p_donor["x"], p_ant["y"] - p_donor["y"], p_ant["z"] - p_donor["z"])
    v2 = (p_acc["x"] - p_donor["x"], p_acc["y"] - p_donor["y"], p_acc["z"] - p_donor["z"])
    n1 = math.sqrt(sum(c * c for c in v1))
    n2 = math.sqrt(sum(c * c for c in v2))
    if n1 == 0 or n2 == 0:
        return 180.0
    cos = sum(a * b for a, b in zip(v1, v2)) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _donors(atoms: list[dict]) -> list[dict]:
    out = []
    for a in atoms:
        name, res = a["atom_name"], a["res_name"]
        if name == "N" and res != "PRO":
            out.append(a)
            continue
        if name in _DONORS_BY_RES.get(res, ()):
            out.append(a)
        elif a["record"] == "HETATM" and a["element"] in ("N", "O"):
            # No SMILES here, so treat every ligand N/O as a possible donor.
            out.append(a)
    return out


def _acceptors(atoms: list[dict]) -> list[dict]:
    out = []
    for a in atoms:
        name, res = a["atom_name"], a["res_name"]
        if name == "O":
            out.append(a)
            continue
        if name in _ACCEPTORS_BY_RES.get(res, ()):
            out.append(a)
        elif a["record"] == "HETATM" and a["element"] in ("N", "O"):
            out.append(a)
    return out


def _antecedent_of(donor: dict, by_res: dict) -> Optional[dict]:
    """The atom the donor's hydrogen points away from."""
    key = (donor["chain_id"], donor["res_seq"])
    name = _ANTECEDENT.get((donor["res_name"], donor["atom_name"]))
    if name is None and donor["atom_name"] == "N":
        name = "CA"
    residue = by_res.get(key, {})
    if name and name in residue:
        return residue[name]
    if donor["record"] == "HETATM":
        # Ligand: stand in the nearest bonded neighbor for the antecedent.
        best, best_d = None, 2.0
        for other_name, other in residue.items():
            if other_name == donor["atom_name"]:
                continue
            d = _distance(other, donor)
            if d < best_d:
                best, best_d = other, d
        return best
    return None


def _detect_hbonds(des: list[dict], anc: list[dict], max_dist: float,
                   max_angle_deg: float) -> list[tuple[dict, dict, float, float]]:
    """Return (donor, acceptor, distance, strength) for every plausible bond
    between the two atom selections."""
    by_res = _atoms_by_res(des + anc)
    pairs: list[tuple[dict, dict, float, float]] = []
    seen: set = set()
    for donors, acceptors in ((_donors(des), _acceptors(anc)),
                              (_donors(anc), _acceptors(des))):
        for d in donors:
            for a in acceptors:
                if d["chain_id"] == a["chain_id"] and d["res_seq"] == a["res_seq"]:
                    continue                        # same residue: covalent
                r = _distance(d, a)
                if r > max_dist or r < 1.5:
                    continue
                ant = _antecedent_of(d, by_res)
                if ant is not None:
                    # A real bond puts the H toward the acceptor, so the
                    # antecedent sits far from it. Small angle = H points away.
                    if _angle_deg(ant, d, a) < (180.0 - max_angle_deg):
                        continue
                key = (d["chain_id"], d["res_seq"], d["atom_name"],
                       a["chain_id"], a["res_seq"], a["atom_name"])
                rkey = key[3:] + key[:3]
                if key in seen or rkey in seen:
                    continue
                seen.add(key)
                strength = math.exp(-((r - _HBOND_D0) ** 2) / (2 * _HBOND_SIGMA ** 2))
                pairs.append((d, a, round(r, 3), round(strength, 3)))
    return pairs


def _annotate_clashes(records: list[ConservableHbond], atoms: list[dict],
                      designable: set[int], chain: str, clash_dist: float) -> None:
    """Flag candidates whose side chain overlaps something that will not move.

    Everything except other designable side chains counts: backbone, ligand and
    fixed side chains all stay put, so a candidate clashing into them is one the
    input never really had, and pinning it would fight the packer.
    """
    if not records:
        return
    candidates = {r.resno for r in records}

    static = [
        a for a in atoms
        if a["record"] == "HETATM" or a["chain_id"] != chain
        or not (a["res_seq"] in designable and a["atom_name"] not in BACKBONE_ATOMS)
    ]
    cand_sidechains: dict[int, list[dict]] = {}
    for a in atoms:
        if (a["record"] == "ATOM" and a["chain_id"] == chain
                and a["res_seq"] in candidates
                and a["atom_name"] not in BACKBONE_ATOMS):
            cand_sidechains.setdefault(a["res_seq"], []).append(a)

    cutoff2 = clash_dist * clash_dist
    clash_with: dict[int, str] = {}
    for resno, sidechain in cand_sidechains.items():
        for s in sidechain:
            for p in static:
                if (p["record"] == "ATOM" and p["chain_id"] == chain
                        and abs(p["res_seq"] - resno) <= 1):
                    continue                    # self and covalent neighbors
                dx, dy, dz = s["x"] - p["x"], s["y"] - p["y"], s["z"] - p["z"]
                if dx * dx + dy * dy + dz * dz < cutoff2:
                    clash_with[resno] = (
                        f"LIG/{p['atom_name']}" if p["record"] == "HETATM"
                        else f"{p['res_name']}{p['res_seq']}/{p['atom_name']}")
                    break
            if resno in clash_with:
                break

    for r in records:
        if r.resno in clash_with:
            r.clashes = True
            r.clash_with = clash_with[r.resno]


def find_conservable_sidechain_hbonds(
    pdb_path: str | Path,
    *,
    designable_resnos: Iterable[int],
    catalytic_resnos: Iterable[int] = (),
    user_fixed_resnos: Iterable[int] = (),
    include_ligand: bool = True,
    chain: str = "A",
    max_dist: float = DEFAULT_MAX_DIST,
    max_angle_deg: float = DEFAULT_MAX_ANGLE_DEG,
    clash_dist: float = DEFAULT_CLASH_DIST,
) -> list[ConservableHbond]:
    """Per-bond records for designable side chains bonding to the active site.

    One residue can produce several records; collapse with
    :func:`select_conservable_resnos`.
    """
    atoms = _parse_pdb_atoms(pdb_path)
    designable = {int(r) for r in designable_resnos}
    catalytic = {int(r) for r in catalytic_resnos}
    user_fixed = {int(r) for r in user_fixed_resnos}
    anchors = catalytic | user_fixed

    des_atoms = [a for a in atoms if a["record"] == "ATOM"
                 and a["chain_id"] == chain and a["res_seq"] in designable]
    anc_atoms = [a for a in atoms if a["record"] == "ATOM"
                 and a["chain_id"] == chain and a["res_seq"] in anchors]
    ligand_keys = {(a["chain_id"], a["res_seq"]) for a in atoms if a["record"] == "HETATM"}
    if include_ligand:
        anc_atoms += [a for a in atoms if a["record"] == "HETATM"]
    if not des_atoms or not anc_atoms:
        return []

    records: list[ConservableHbond] = []
    for donor, acceptor, dist, strength in _detect_hbonds(
            des_atoms, anc_atoms, max_dist, max_angle_deg):
        d_is_des = donor["chain_id"] == chain and donor["res_seq"] in designable
        a_is_des = acceptor["chain_id"] == chain and acceptor["res_seq"] in designable
        if d_is_des == a_is_des:
            continue                        # exactly one side must be designable
        des, partner = (donor, acceptor) if d_is_des else (acceptor, donor)
        if des["atom_name"] in BACKBONE_ATOMS:
            continue                        # the designable side must use its side chain

        pkey = (partner["chain_id"], partner["res_seq"])
        if pkey in ligand_keys:
            kind, presno = "ligand", -1
        elif partner["res_seq"] in catalytic:
            kind, presno = "catalytic", partner["res_seq"]
        elif partner["res_seq"] in user_fixed:
            kind, presno = "user_fixed", partner["res_seq"]
        else:
            continue

        records.append(ConservableHbond(
            resno=des["res_seq"], resname=des["res_name"],
            sidechain_atom=des["atom_name"],
            partner_kind=kind, partner_resno=presno,
            partner_resname=partner["res_name"], partner_atom=partner["atom_name"],
            distance=dist, strength=strength, strength_bin=strength_bin(strength),
        ))

    _annotate_clashes(records, atoms, designable, chain, clash_dist)
    return records


def select_conservable_resnos(
    records: Iterable[ConservableHbond], *, keep_clashing: bool = False,
) -> tuple[list[int], list[tuple[int, str]]]:
    """Collapse per-bond records to candidate residue numbers.

    Returns ``(candidates, excluded)``, where excluded is ``[(resno, reason)]``
    for the clashing ones so the caller can report what it dropped.
    """
    by_res: dict[int, list[ConservableHbond]] = {}
    for r in records:
        by_res.setdefault(r.resno, []).append(r)
    candidates: list[int] = []
    excluded: list[tuple[int, str]] = []
    for resno, rs in sorted(by_res.items()):
        if rs[0].clashes and not keep_clashing:
            excluded.append((resno, rs[0].clash_with))
            continue
        candidates.append(resno)
    return candidates, excluded


def normalize_probability(value: float) -> float:
    """Accept a 0-1 probability, or a percentage in (1, 100]."""
    v = float(value)
    if v < 0:
        raise ValueError(f"must be >= 0 (got {v})")
    if v <= 1:
        return v
    if v <= 100:
        LOGGER.warning("probability %g > 1; reading it as a percentage -> %g", v, v / 100.0)
        return v / 100.0
    raise ValueError(f"must be 0-1 or a percentage <= 100 (got {v})")


def roll_conserved(items: Iterable, prob: float, rng) -> set:
    """Keep each item independently with probability ``prob``.

    Items are rolled in iteration order, so pass an ordered sequence and a
    seeded ``random.Random`` when you want the choice to be reproducible.
    """
    return {x for x in items if rng.random() < prob}
