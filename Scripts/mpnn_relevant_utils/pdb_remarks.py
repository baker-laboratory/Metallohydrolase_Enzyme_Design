#!/usr/bin/env python3
"""REMARK 666 / 667 / 668 handling for designed PDBs.

Enzyme design carries its catalytic-site definition in ``REMARK 666`` lines
written by the Rosetta matcher and by this repository's theozyme tooling:

    REMARK 666 MATCH TEMPLATE X YYE    0 MATCH MOTIF A HIS   93   1  1
                            ^target      ^chain ^name ^resno ^cst

LigandMPNN does not carry remarks through, so a design comes back without any
record of which residues were catalytic or what protonation state they were in.
Everything downstream -- constraint files, catalytic RMSD scoring, AlphaFold3
PTM annotation -- needs that record back, so this module rebuilds it:

    parse_remark_666         read the catalytic-site block
    transfer_remarks         copy an input's remarks onto a design
    build_remark_668_block   describe each catalytic residue's protonation state
    replace_remark_block     swap a 667/668 block in place

REMARK 668 is our own convention, not a PDB standard. It pairs by index with
REMARK 666 and records the tautomer or modification each catalytic residue is
in, so a reader (or PyRosetta, on the way back in) can restore the state the
theozyme was designed around rather than defaulting to the common tautomer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

__all__ = [
    "CatalyticResidue",
    "parse_remark_666",
    "read_remarks",
    "transfer_remarks",
    "transfer_remarks_to_dir",
    "build_remark_668_block",
    "replace_remark_block",
    "add_design_path_remark",
    "detect_protonation_state",
]

# Hydrogens that distinguish the histidine tautomers. A heavy-atom-only model
# (which is what LigandMPNN emits) has neither, hence "unknown".
_HIS_HD1 = {"HD1", "1HD", "HD11"}
_HIS_HE2 = {"HE2", "2HE", "HE21"}

_REMARK_666_RE = re.compile(
    r"^REMARK\s+666\s+MATCH\s+TEMPLATE\s+(?P<tchain>\S+)\s+(?P<tname>\S+)\s+"
    r"(?P<tresno>-?\d+)\s+MATCH\s+MOTIF\s+(?P<chain>\S+)\s+(?P<name>\S+)\s+"
    r"(?P<resno>-?\d+)\s+(?P<cst>\d+)"
)


@dataclass(frozen=True)
class CatalyticResidue:
    """One REMARK 666 MATCH MOTIF entry."""

    cst_no: int
    chain: str
    name3: str
    resno: int
    target_chain: str
    target_name3: str
    target_resno: int

    @property
    def label(self) -> str:
        """The ``A93``-style token LigandMPNN uses for fixed residues."""
        return f"{self.chain}{self.resno}"


def parse_remark_666(pdb_path: str | Path) -> list[CatalyticResidue]:
    """Return the catalytic residues declared in a PDB's REMARK 666 block.

    Returns an empty list when the file has no such block -- callers decide
    whether that means "full redesign" or "this is a mistake".
    """
    out: list[CatalyticResidue] = []
    try:
        with open(pdb_path, "r") as fh:
            for line in fh:
                if not line.startswith("REMARK 666"):
                    if line.startswith(("ATOM", "HETATM")):
                        break            # remarks are done; stop reading
                    continue
                m = _REMARK_666_RE.match(line)
                if not m:
                    continue
                out.append(CatalyticResidue(
                    cst_no=int(m.group("cst")),
                    chain=m.group("chain"),
                    name3=m.group("name"),
                    resno=int(m.group("resno")),
                    target_chain=m.group("tchain"),
                    target_name3=m.group("tname"),
                    target_resno=int(m.group("tresno")),
                ))
    except OSError:
        return []
    return out


def read_remarks(pdb_path: str | Path, keep: Iterable[str] = ("REMARK",)) -> list[str]:
    """All leading REMARK lines of a PDB, in order, newline-terminated."""
    prefixes = tuple(keep)
    out: list[str] = []
    try:
        with open(pdb_path, "r") as fh:
            for line in fh:
                if line.startswith(("ATOM", "HETATM", "MODEL")):
                    break
                if line.startswith(prefixes):
                    out.append(line if line.endswith("\n") else line + "\n")
    except OSError:
        return []
    return out


def _body_without_remarks(pdb_path: str | Path, drop: tuple[str, ...]) -> list[str]:
    """Every line of a PDB except the remark kinds named in ``drop``."""
    lines: list[str] = []
    with open(pdb_path, "r") as fh:
        for line in fh:
            if line.startswith(drop):
                continue
            lines.append(line)
    return lines


def transfer_remarks(
    design_pdb: str | Path,
    source_pdb: str | Path,
    *,
    drop_existing: tuple[str, ...] = ("REMARK 666", "REMARK 667", "REMARK 668"),
) -> int:
    """Copy the source's catalytic remarks onto a design, in place.

    The design's own copies of those remark kinds are dropped first, so this is
    idempotent -- running it twice does not stack duplicate blocks.
    """
    keep = tuple(k for k in drop_existing)
    donor = [ln for ln in read_remarks(source_pdb) if ln.startswith(keep)]
    if not donor:
        return 0
    body = _body_without_remarks(design_pdb, keep)
    Path(design_pdb).write_text("".join(donor + body))
    return len(donor)


def transfer_remarks_to_dir(
    directory: str | Path, source_pdb: str | Path, *, skip: Iterable[str] = ()
) -> int:
    """Apply :func:`transfer_remarks` to every ``*.pdb`` in a flat directory."""
    skip_names = set(skip)
    n = 0
    for pdb in sorted(Path(directory).glob("*.pdb")):
        if pdb.name in skip_names:
            continue
        if transfer_remarks(pdb, source_pdb):
            n += 1
    return n


def detect_protonation_state(pdb_path: str | Path, chain: str, resno: int) -> str:
    """Name the protonation state of one residue from the atoms present.

    Histidine is the case that matters: HD1 alone is the delta tautomer, HE2
    alone is epsilon, both is the protonated cation, neither means the model is
    heavy-atom only and the state is genuinely unknown -- which is exactly what
    comes back from LigandMPNN, and why the state has to be restored from the
    input rather than inferred from the design.
    """
    name3 = ""
    atoms: set[str] = set()
    try:
        with open(pdb_path, "r") as fh:
            for line in fh:
                if not line.startswith(("ATOM", "HETATM")):
                    continue
                if line[21] != chain:
                    continue
                try:
                    if int(line[22:26]) != resno:
                        continue
                except ValueError:
                    continue
                name3 = line[17:20].strip()
                atoms.add(line[12:16].strip())
    except OSError:
        return "unknown"

    if not name3:
        return "unknown"
    if name3 not in {"HIS", "HID", "HIE", "HIP", "HIS_D", "HIS_E"}:
        return name3            # nothing tautomeric to say
    hd1 = bool(atoms & _HIS_HD1)
    he2 = bool(atoms & _HIS_HE2)
    if hd1 and he2:
        return "HIP"            # doubly protonated, +1
    if hd1:
        return "HID"            # delta tautomer  (Rosetta HIS_D)
    if he2:
        return "HIE"            # epsilon tautomer (Rosetta HIS)
    return "unknown"            # heavy atoms only


def _parse_ptm_spec(spec: Optional[str]) -> dict[tuple[str, int], str]:
    """Parse ``CHAIN/RESN/MOTIF_IDX:CODE`` PTM annotations.

    Motif index is 1-based into the REMARK 666 block, matching how the matcher
    numbers its constraints -- e.g. ``A/LYS/3:KCX`` marks the third catalytic
    residue, a lysine on chain A, as carboxylated.
    """
    out: dict[tuple[str, int], str] = {}
    for item in (spec or "").split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        loc, code = item.rsplit(":", 1)
        parts = loc.split("/")
        if len(parts) != 3:
            continue
        chain, _expect_resn, idx = parts
        try:
            out[(chain.strip(), int(idx))] = code.strip()
        except ValueError:
            continue
    return out


def build_remark_668_block(
    pdb_path: str | Path,
    state_source_pdb: str | Path | None = None,
    *,
    ptm_map: Optional[str] = None,
) -> list[str]:
    """Build the REMARK 667/668 protonation-state block for a PDB.

    States are read from ``state_source_pdb`` (default: the file itself), so a
    heavy-atom design can be annotated with the states of the input it came
    from. ``ptm_map`` overrides a slot by motif index, for modifications that
    have no hydrogens to detect -- a carboxylated lysine, say.
    """
    catres = parse_remark_666(pdb_path)
    if not catres:
        return []
    source = Path(state_source_pdb) if state_source_pdb else Path(pdb_path)
    overrides = _parse_ptm_spec(ptm_map)

    lines = [
        "REMARK 667 PROTONATION STATE BLOCK; ONE ENTRY PER REMARK 666 MOTIF, IN ORDER\n",
        "REMARK 667 FIELDS: CST CHAIN RESNAME RESNO STATE\n",
    ]
    for i, cr in enumerate(catres, start=1):
        state = overrides.get((cr.chain, i)) or detect_protonation_state(
            source, cr.chain, cr.resno)
        lines.append(
            f"REMARK 668 MATCH MOTIF {cr.cst_no:>3} {cr.chain:>2} "
            f"{cr.name3:>4} {cr.resno:>5}  {state}\n"
        )
    return lines


def replace_remark_block(
    pdb_path: str | Path,
    block: list[str],
    *,
    kinds: tuple[str, ...] = ("REMARK 667", "REMARK 668"),
) -> bool:
    """Replace a PDB's 667/668 block with ``block``, in place.

    The new block is written directly after the REMARK 666 lines so the two stay
    adjacent and readable. Returns False when there is nothing to write.
    """
    if not block:
        return False
    path = Path(pdb_path)
    try:
        original = path.read_text().splitlines(keepends=True)
    except OSError:
        return False

    out: list[str] = []
    inserted = False
    for line in original:
        if line.startswith(kinds):
            continue                              # drop any previous block
        if not inserted and not line.startswith("REMARK 666") and out and \
                any(l.startswith("REMARK 666") for l in out):
            out.extend(block)                     # just past the 666 block
            inserted = True
        out.append(line)
    if not inserted:
        out = block + out                         # no 666 block; put it on top
    path.write_text("".join(out))
    return True


def add_design_path_remark(pdb_path: str | Path, source: str) -> bool:
    """Record where a design came from, as a free-text REMARK.

    Designs get flattened into one directory and renamed downstream; this is the
    breadcrumb back to the run that produced them.
    """
    path = Path(pdb_path)
    try:
        lines = path.read_text().splitlines(keepends=True)
    except OSError:
        return False
    lines = [l for l in lines if not l.startswith("REMARK DESIGN_PATH")]
    out: list[str] = []
    placed = False
    for line in lines:
        if not placed and line.startswith(("ATOM", "HETATM")):
            out.append(f"REMARK DESIGN_PATH {source}\n")
            placed = True
        out.append(line)
    if not placed:
        out.append(f"REMARK DESIGN_PATH {source}\n")
    path.write_text("".join(out))
    return True
