#!/usr/bin/env python3
"""Add hydrogens to a directory of designed PDBs, in place.

LigandMPNN emits heavy atoms only. Verified on its own packed output: of ~1560
atoms, the only hydrogens present are the ligand's, carried through from the
input -- the protein comes back with none. Anything downstream that reasons
about hydrogen bonding, protonation or tautomers therefore needs this step.

PyRosetta places ideal hydrogens when it builds a pose, so the work here is
mostly about not losing what the design already encoded:

  * **Catalytic tautomers.** A heavy-atom histidine has no tautomer, so
    PyRosetta would give every catalytic HIS its default. The states are read
    from the input structure the design came from and re-applied as Rosetta
    residue types, so the theozyme's geometry survives.
  * **The ligand.** With a ``.params`` the ligand loads and is protonated with
    everything else. Without one, PyRosetta is told to ignore it, the protein
    is protonated apo, and the ligand block is copied back verbatim from the
    input -- so an unparameterized ligand costs you nothing but its own
    hydrogens.
  * **The remarks.** REMARK 666 is restored and a REMARK 668 block describing
    each catalytic residue's final state is written alongside it.

Run it directly, or import ``protonate_directory``:

    python protonate_designs.py --design_dir out/ --seed_pdb input.pdb \\
        --ligand_params LIG.params --ptm A/LYS/3:KCX
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Iterable, Optional

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from pdb_remarks import (  # noqa: E402
    build_remark_668_block,
    detect_protonation_state,
    parse_remark_666,
    replace_remark_block,
    transfer_remarks,
)

LOGGER = logging.getLogger("mpnn.protonate")

__all__ = ["protonate_directory", "protonate_one", "collect_variant_map"]

# Our state labels -> Rosetta residue-type names. Rosetta's default HIS is the
# epsilon tautomer, so HIE maps onto plain HIS and only HID needs HIS_D.
_STATE_TO_ROSETTA = {
    "HID": "HIS_D",
    "HIE": "HIS",
    "HIP": "HIS_P",
    "HIS_D": "HIS_D",
    "HIS_E": "HIS",
}

_INTERMEDIATE_SUFFIXES = (".protonated.pdb", ".rosetta.pdb", ".norm.pdb")

_pyrosetta_inited = False
_pyrosetta_params: tuple[str, ...] = ()


def is_intermediate(name: str) -> bool:
    return name.endswith(_INTERMEDIATE_SUFFIXES)


def collect_variant_map(seed_pdb: str | Path,
                        ptm_spec: Optional[str] = None) -> dict[tuple[str, int], str]:
    """Protonation states of the input's catalytic residues, by (chain, resno).

    Only states worth restoring are returned -- a residue whose state is
    'unknown' (heavy atoms only) or simply its own residue name carries no
    information PyRosetta would not work out for itself.
    """
    catres = parse_remark_666(seed_pdb)
    out: dict[tuple[str, int], str] = {}
    for cr in catres:
        state = detect_protonation_state(seed_pdb, cr.chain, cr.resno)
        if state in _STATE_TO_ROSETTA:
            out[(cr.chain, cr.resno)] = state

    # A PTM has no hydrogens to detect, so it is declared rather than found.
    for item in (ptm_spec or "").split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        loc, code = item.rsplit(":", 1)
        parts = loc.split("/")
        if len(parts) != 3:
            continue
        chain, _resn, idx = parts
        try:
            motif = int(idx)
        except ValueError:
            continue
        if 1 <= motif <= len(catres):
            cr = catres[motif - 1]
            out[(cr.chain, cr.resno)] = code.strip()
    return out


def _init_pyrosetta(ligand_params: tuple[str, ...]) -> None:
    """Initialize PyRosetta once per process, with the ligand params if any."""
    global _pyrosetta_inited, _pyrosetta_params
    if _pyrosetta_inited:
        if ligand_params != _pyrosetta_params:
            LOGGER.warning(
                "PyRosetta was already initialized with %s; the request for %s is "
                "ignored. Run a separate process for a different ligand.",
                _pyrosetta_params or "(no params)", ligand_params or "(no params)")
        return
    import pyrosetta

    flags = ["-mute all", "-out:level 0", "-ignore_unrecognized_res",
             "-load_PDB_components false"]
    if ligand_params:
        flags.append("-extra_res_fa " + " ".join(ligand_params))
    pyrosetta.init(" ".join(flags), silent=True)
    _pyrosetta_inited = True
    _pyrosetta_params = ligand_params


def _hetatm_block(pdb_path: str | Path) -> list[str]:
    """The ligand/metal records of a PDB, verbatim."""
    out: list[str] = []
    with open(pdb_path, "r") as fh:
        for line in fh:
            if line.startswith("HETATM"):
                out.append(line if line.endswith("\n") else line + "\n")
    return out


def _strip_hetatm(src: str | Path, dest: str | Path) -> None:
    """Copy a PDB without its HETATM records (apo protonation input)."""
    with open(src) as fh_in, open(dest, "w") as fh_out:
        for line in fh_in:
            if line.startswith("HETATM"):
                continue
            fh_out.write(line)


def _apply_variants(pose, variant_map: dict[tuple[str, int], str]) -> int:
    """Re-apply catalytic residue types to a freshly built pose."""
    if not variant_map:
        return 0
    from pyrosetta.rosetta.core.pose import (
        replace_pose_residue_copying_existing_coordinates,
    )
    rts = pose.residue_type_set_for_pose()
    applied = 0
    for (chain, resno), state in variant_map.items():
        target = _STATE_TO_ROSETTA.get(state, state)
        try:
            pose_resno = pose.pdb_info().pdb2pose(chain, resno)
            if pose_resno == 0:
                LOGGER.warning("residue %s%d not in pose; skipping", chain, resno)
                continue
            if not rts.has_name(target):
                # HIS_P is absent from some residue-type sets. Fall back to the
                # neutral tautomer rather than dropping the residue entirely.
                if target == "HIS_P" and rts.has_name("HIS"):
                    LOGGER.info("HIS_P unavailable; using HIS for %s%d", chain, resno)
                    target = "HIS"
                else:
                    LOGGER.warning("residue type %r unknown; leaving %s%d default",
                                   target, chain, resno)
                    continue
            replace_pose_residue_copying_existing_coordinates(
                pose, pose_resno, rts.name_map(target))
            applied += 1
        except Exception as exc:            # never let one residue stop the dump
            LOGGER.warning("variant %s for %s%d failed: %s", state, chain, resno, exc)
    return applied


def protonate_one(
    design_pdb: str | Path,
    seed_pdb: str | Path,
    *,
    ligand_params: tuple[str, ...] = (),
    variant_map: Optional[dict[tuple[str, int], str]] = None,
    ptm_spec: Optional[str] = None,
    keep_intermediate: bool = False,
) -> bool:
    """Protonate one design in place. Returns True on success."""
    import pyrosetta

    design = Path(design_pdb)
    apo = not ligand_params
    work = design.with_name(design.name + ".norm.pdb")

    if apo:
        _strip_hetatm(design, work)
    else:
        shutil.copy2(design, work)

    try:
        _init_pyrosetta(ligand_params)
        pose = pyrosetta.pose_from_pdb(str(work))
        _apply_variants(pose, variant_map or {})
        hydrated = design.with_name(design.name + ".rosetta.pdb")
        pose.dump_pdb(str(hydrated))
    except Exception as exc:
        LOGGER.error("protonation failed for %s: %s", design.name, exc)
        if not keep_intermediate:
            work.unlink(missing_ok=True)
        return False

    lines = hydrated.read_text().splitlines(keepends=True)
    if apo:
        # Put the ligand back exactly as it came in, before the terminal record.
        body = [l for l in lines if not l.startswith(("END", "TER"))]
        lines = body + ["TER\n"] + _hetatm_block(seed_pdb) + ["END\n"]

    design.write_text("".join(lines))
    transfer_remarks(design, seed_pdb)
    replace_remark_block(design, build_remark_668_block(design, seed_pdb, ptm_map=ptm_spec))

    if not keep_intermediate:
        work.unlink(missing_ok=True)
        hydrated.unlink(missing_ok=True)
    return True


def protonate_directory(
    design_dir: str | Path,
    seed_pdb: str | Path,
    *,
    ligand_params: Iterable[str | Path] = (),
    ptm_spec: Optional[str] = None,
    keep_intermediate: bool = False,
) -> dict:
    """Protonate every ``*.pdb`` in a flat directory, in place."""
    directory = Path(design_dir)
    if not directory.is_dir():
        return {"skipped": "no_dir", "directory": str(directory)}
    designs = [p for p in sorted(directory.glob("*.pdb")) if not is_intermediate(p.name)]
    if not designs:
        return {"skipped": "empty", "directory": str(directory)}

    params = tuple(str(Path(p).resolve()) for p in ligand_params)
    variant_map = collect_variant_map(seed_pdb, ptm_spec)
    mode = "holo" if params else "apo (ligand copied back from the input)"
    print(f"  protonating {len(designs)} design(s) [{mode}]")
    if variant_map:
        states = ", ".join(f"{c}{r}={s}" for (c, r), s in sorted(variant_map.items()))
        print(f"  restoring catalytic states from the input: {states}")

    ok = 0
    for pdb in designs:
        if protonate_one(pdb, seed_pdb, ligand_params=params, variant_map=variant_map,
                         ptm_spec=ptm_spec, keep_intermediate=keep_intermediate):
            ok += 1
    print(f"  protonated {ok}/{len(designs)} in place")
    return {"protonated": ok, "total": len(designs)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--design_dir", required=True,
                   help="Flat directory of design PDBs; they are overwritten in place.")
    p.add_argument("--seed_pdb", required=True,
                   help="The input structure the designs came from: source of the "
                        "catalytic states, REMARK 666, and the ligand in apo mode.")
    p.add_argument("--ligand_params", nargs="*", default=[],
                   help="Rosetta .params for the ligand. Omit for apo protonation.")
    p.add_argument("--ptm", default=None,
                   help='PTM annotation, e.g. "A/LYS/3:KCX" (CHAIN/RESN/MOTIF_IDX:CODE).')
    p.add_argument("--keep_intermediate", action="store_true",
                   help="Keep the .norm/.rosetta intermediates for inspection.")
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = protonate_directory(
        a.design_dir, a.seed_pdb, ligand_params=a.ligand_params,
        ptm_spec=a.ptm, keep_intermediate=a.keep_intermediate)
    if "skipped" in result:
        print(f"nothing to do: {result['skipped']}")
        return 0
    return 0 if result["protonated"] == result["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
