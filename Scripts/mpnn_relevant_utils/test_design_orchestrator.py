#!/usr/bin/env python3
"""Tests for the LigandMPNN design orchestrator and its helpers.

Runs against real structures from this repository, so it exercises the actual
REMARK formats and active-site geometry rather than synthetic fixtures. No GPU
and no model weights are needed: MPNN itself is exercised through ``--dry_run``,
which builds the commands without running them, and every emitted flag is
checked against LigandMPNN's own argparse.

The protonation test needs PyRosetta. Where it is unavailable the PyRosetta
step is skipped and reported as such -- everything around it still runs.

    python test_design_orchestrator.py
"""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import hbond_conservation as hbc      # noqa: E402
import pdb_remarks as R               # noqa: E402
import protonate_designs             # noqa: E402

REPO = _HERE.parents[1]
LIGANDMPNN_RUN = REPO / "Software/fastmpnndesign/lib/LigandMPNN/run.py"

# A designed zinc hydrolase: full protein, REMARK 666 catalytic triad, bound
# ligand. Its active site is the reference for the H-bond expectations below.
DESIGN_PDB = REPO / (
    "Design_Pipelines/Metalloesterase_RFdiffusion2/outputs/af2_out/predesigned_RFD2/"
    "filtered_structures/alignment/"
    "zn_hydrolase_PSZ_H_H_H_1_63_0_1-atomized-bb-False_0_2_model_4_ptm_seed_0_unrelaxed_PSZ.pdb")

# A phosphotriesterase theozyme: six catalytic residues, histidines carrying
# explicit HD1, and a lysine that is carboxylated in the real system.
THEOZYME_PDB = REPO / (
    "Design_Pipelines/Phosphotriesterase_RFdiffusion3/inputs/theozymes/"
    "step2_theozyme_prep_ori/ZAPP_p1D1_active_site_NEB_TS_rotP_0_ORI_01.pdb")

_passed: list[str] = []
_failed: list[str] = []
_skipped: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    (_passed if condition else _failed).append(name)
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {name}" + (f"  --  {detail}" if detail else ""))
    return condition


def skip(name: str, why: str) -> None:
    _skipped.append(name)
    print(f"  [SKIP] {name}  --  {why}")


# ---------------------------------------------------------------------------
def test_remark_666_parsing() -> None:
    print("\n### REMARK 666 / 668")
    catres = R.parse_remark_666(THEOZYME_PDB)
    check("parses every catalytic residue", len(catres) == 6, f"{len(catres)} found")
    check("labels are LigandMPNN tokens",
          [c.label for c in catres] == ["A93", "A89", "A16", "A170", "A133", "A92"],
          ", ".join(c.label for c in catres))
    check("cst numbering is preserved",
          [c.cst_no for c in catres] == [1, 2, 3, 4, 5, 6])

    # Histidines here carry HD1, so they are the delta tautomer. Getting this
    # wrong is the whole reason the state is restored rather than re-derived.
    states = {c.label: R.detect_protonation_state(THEOZYME_PDB, c.chain, c.resno)
              for c in catres}
    check("histidine tautomers are read from the hydrogens",
          all(states[l] == "HID" for l in ("A93", "A89", "A170", "A133")), str(states))

    block = R.build_remark_668_block(THEOZYME_PDB, THEOZYME_PDB, ptm_map="A/LYS/3:KCX")
    check("668 block has one entry per 666 motif",
          sum(1 for l in block if l.startswith("REMARK 668")) == 6)
    check("a declared PTM overrides the detected state",
          any("KCX" in l and " 16" in l for l in block),
          next((l.strip() for l in block if "KCX" in l), "not found"))

    # A design comes back with no remarks at all; they have to survive a
    # round-trip from the input.
    with tempfile.TemporaryDirectory() as tmp:
        stripped = Path(tmp) / "design.pdb"
        stripped.write_text("".join(
            l for l in THEOZYME_PDB.read_text().splitlines(keepends=True)
            if not l.startswith("REMARK")))
        check("a stripped design starts with no catalytic record",
              R.parse_remark_666(stripped) == [])
        R.transfer_remarks(stripped, THEOZYME_PDB)
        check("transfer restores the catalytic record",
              len(R.parse_remark_666(stripped)) == 6)
        R.transfer_remarks(stripped, THEOZYME_PDB)
        check("transfer is idempotent, not cumulative",
              len(R.parse_remark_666(stripped)) == 6,
              f"{len(R.parse_remark_666(stripped))} after a second transfer")
        R.add_design_path_remark(stripped, "/some/run/dir")
        R.add_design_path_remark(stripped, "/some/run/dir")
        n = sum(1 for l in stripped.read_text().splitlines()
                if l.startswith("REMARK DESIGN_PATH"))
        check("DESIGN_PATH is recorded exactly once", n == 1, f"{n} lines")


# ---------------------------------------------------------------------------
def test_hbond_detection() -> None:
    print("\n### H-bond side-chain conservation")
    catres = R.parse_remark_666(DESIGN_PDB)
    chain = catres[0].chain
    cat_resnos = {c.resno for c in catres}
    atoms = hbc._parse_pdb_atoms(DESIGN_PDB)
    protein = sorted({a["res_seq"] for a in atoms
                      if a["record"] == "ATOM" and a["chain_id"] == chain})
    designable = [r for r in protein if r not in cat_resnos]

    records = hbc.find_conservable_sidechain_hbonds(
        DESIGN_PDB, designable_resnos=designable, catalytic_resnos=cat_resnos,
        include_ligand=True, chain=chain)
    found = {r.resno for r in records}

    check("finds the threonine donating to the ligand", 15 in found)
    check("finds the tyrosine donating to the ligand", 55 in found)
    check("finds the second-shell contact to a catalytic histidine", 140 in found,
          next((f"{r.resname}{r.resno} -> {r.partner_resname}{r.partner_resno}"
                for r in records if r.resno == 140), ""))
    check("never proposes a catalytic residue (already fixed)",
          not (found & cat_resnos), f"catalytic={sorted(cat_resnos)}")
    check("never proposes a backbone-only contact",
          all(r.sidechain_atom not in hbc.BACKBONE_ATOMS for r in records))

    candidates, excluded = hbc.select_conservable_resnos(records)
    check("drops a candidate that clashes with the ligand",
          179 in {r for r, _ in excluded} and 179 not in candidates,
          f"excluded={excluded}")
    kept, _ = hbc.select_conservable_resnos(records, keep_clashing=True)
    check("--conserve_keep_clashing keeps it", 179 in kept)

    # Rolling has to be reproducible from a seed, or a design cannot be replayed.
    import random
    a = hbc.roll_conserved(candidates, 0.8, random.Random(7))
    b = hbc.roll_conserved(candidates, 0.8, random.Random(7))
    check("a seeded roll is reproducible", a == b, f"{sorted(a)}")
    check("probability 0 conserves nothing",
          hbc.roll_conserved(candidates, 0.0, random.Random(7)) == set())
    check("probability 1 conserves everything",
          hbc.roll_conserved(candidates, 1.0, random.Random(7)) == set(candidates))
    check("a percentage is accepted as a probability",
          hbc.normalize_probability(80) == 0.8)


# ---------------------------------------------------------------------------
def _ligandmpnn_flags() -> set[str]:
    """Every flag LigandMPNN's run.py actually accepts."""
    flags: set[str] = set()
    tree = ast.parse(LIGANDMPNN_RUN.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) \
                        and arg.value.startswith("-"):
                    flags.add(arg.value)
    return flags


def test_orchestrator_dry_run() -> None:
    print("\n### orchestrator command assembly")
    if not LIGANDMPNN_RUN.is_file():
        skip("emitted flags are accepted by LigandMPNN",
             "LigandMPNN submodule not initialized")
        return

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "designs"
        proc = subprocess.run(
            [sys.executable, str(_HERE / "design_orchestrator.py"),
             "--pdb_path", str(DESIGN_PDB), "--out_folder", str(out),
             "--model_type", "ligand_mpnn", "--pack_side_chains", "1",
             "--repack_everything", "0", "--omit_AA", "CX", "--bias_AA", "K:-0.5",
             "--conserve_hbonds", "--conserve_seed", "42",
             "--run", "number_of_batches=35;temperature=0.1",
             "--run", "number_of_batches=35;temperature=0.3;omit_AA=CXP",
             "--dry_run"],
            capture_output=True, text=True)
        check("dry run exits cleanly", proc.returncode == 0, proc.stderr.strip()[:200])

        commands = [l.strip() for l in proc.stdout.splitlines()
                    if "run_ligandmpnn.py" in l]
        check("one command per sweep combination", len(commands) == 2,
              f"{len(commands)} built")
        if len(commands) != 2:
            return

        accepted = _ligandmpnn_flags()
        used = {t for c in commands for t in c.split() if t.startswith("--")}
        unknown = sorted(used - accepted)
        check("every emitted flag is accepted by LigandMPNN", not unknown,
              f"unknown: {unknown}" if unknown else f"{len(used)} flags")

        check("a universal flag is inherited by a combination that does not set it",
              "--bias_AA K:-0.5" in commands[0] and "--bias_AA K:-0.5" in commands[1])
        check("a combination overrides the universal value",
              "--omit_AA CX " in commands[0] and "--omit_AA CXP" in commands[1])
        check("each combination gets a distinct packed suffix",
              "_t0_1" in commands[0] and "_t0_3" in commands[1])
        check("each combination stages into its own directory",
              "_stage_t0_1" in commands[0] and "_stage_t0_3" in commands[1])

        fixed = json.loads((out / "_pre" / "fixed_residues_t0_1.json").read_text())
        key = next(iter(fixed))
        check("the fixed-residues JSON is keyed by the literal --pdb_path",
              key == str(DESIGN_PDB),
              "a resolved key would not match on symlinked scratch")
        labels = set(fixed[key])
        check("catalytic residues are held fixed", {"A8", "A53", "A91"} <= labels,
              ", ".join(sorted(labels)))
        check("conserved H-bond residues are held fixed too",
              labels - {"A8", "A53", "A91"}, ", ".join(sorted(labels)))

        omit = json.loads((out / "_pre" / "omit_nterm_met.json").read_text())
        check("methionine is omitted at residue 1", omit[key] == {"A1": "M"}, str(omit[key]))


# ---------------------------------------------------------------------------
def test_protonation() -> None:
    print("\n### protonation")
    variants = protonate_designs.collect_variant_map(THEOZYME_PDB, ptm_spec="A/LYS/3:KCX")
    check("catalytic histidine tautomers are collected for restoration",
          all(variants.get(("A", r)) == "HID" for r in (93, 89, 170, 133)), str(variants))
    check("a declared PTM is collected", variants.get(("A", 16)) == "KCX")
    check("residues with nothing to restore are left out",
          ("A", 92) not in variants, "GLU 92 carries no tautomer")

    # Apo protonation strips the ligand, protonates, then puts the ligand back
    # byte-identically -- so an unparameterized ligand loses nothing.
    with tempfile.TemporaryDirectory() as tmp:
        stripped = Path(tmp) / "apo.pdb"
        protonate_designs._strip_hetatm(THEOZYME_PDB, stripped)
        check("apo input has no HETATM records",
              not any(l.startswith("HETATM") for l in stripped.read_text().splitlines()))
        original = protonate_designs._hetatm_block(THEOZYME_PDB)
        check("the ligand block is recoverable verbatim from the input",
              len(original) > 0 and all(l.startswith("HETATM") for l in original),
              f"{len(original)} records")

    try:
        import pyrosetta  # noqa: F401
    except ImportError:
        skip("PyRosetta adds hydrogens and restores tautomers",
             "PyRosetta not installed in this interpreter")
        return

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "designs"
        work.mkdir()
        shutil.copy2(DESIGN_PDB, work / "design.pdb")
        before = sum(1 for l in (work / "design.pdb").read_text().splitlines()
                     if l.startswith("ATOM") and l[76:78].strip() == "H")
        result = protonate_designs.protonate_directory(work, DESIGN_PDB)
        after = sum(1 for l in (work / "design.pdb").read_text().splitlines()
                    if l.startswith("ATOM") and l[76:78].strip() == "H")
        check("every design is protonated",
              result.get("protonated") == result.get("total"), str(result))
        check("the protein gains hydrogens it did not have",
              after > before, f"{before} -> {after} protein H")
        check("the catalytic record survives protonation",
              len(R.parse_remark_666(work / "design.pdb")) == 3)


# ---------------------------------------------------------------------------
def main() -> int:
    print("Testing the LigandMPNN design orchestrator")
    for pdb in (DESIGN_PDB, THEOZYME_PDB):
        if not pdb.is_file():
            print(f"missing test structure: {pdb}")
            return 2

    test_remark_666_parsing()
    test_hbond_detection()
    test_orchestrator_dry_run()
    test_protonation()

    print(f"\n{len(_passed)} passed, {len(_failed)} failed, {len(_skipped)} skipped")
    for name in _failed:
        print(f"  FAILED: {name}")
    for name in _skipped:
        print(f"  skipped: {name}")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
