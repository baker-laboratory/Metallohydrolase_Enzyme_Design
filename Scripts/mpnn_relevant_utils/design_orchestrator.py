#!/usr/bin/env python3
"""Sequence design around a fixed active site: LigandMPNN with the setup and
cleanup that enzyme design needs.

Calling LigandMPNN directly on a theozyme scaffold leaves a fair amount of
manual work either side of it. This driver does that work:

  BEFORE
    * Read REMARK 666 and hold the catalytic residues fixed, so you do not
      hand-maintain a fixed-residues JSON per structure.
    * Omit Met at position 1, which the expression tag supplies anyway.
    * Optionally conserve designable side chains that hydrogen-bond the active
      site, chosen probabilistically so the second shell varies between designs
      without being thrown away entirely.

  DURING
    * Sweep temperature / batch settings in one invocation, each combination
      tagged so its outputs never collide.

  AFTER
    * Flatten every run into one directory of packed PDBs.
    * Protonate them, restoring the input's catalytic tautomers.
    * Put REMARK 666 back, write a REMARK 668 protonation-state block, record
      where each design came from, and copy the input alongside them.

MPNN itself runs through ``run_ligandmpnn.py``, so designs inherit that
wrapper's fixed-residue coordinate preservation: a catalytic residue taken from
a crystal structure or a QM theozyme comes back where it started rather than
rebuilt at idealised geometry. Any flag this script does not recognize is passed
straight through to LigandMPNN, so you can drive it exactly as you would by
hand.

    python design_orchestrator.py \\
        --pdb_path scaffold.pdb --out_folder designs/ \\
        --ligand_params LIG.params --ptm A/LYS/3:KCX \\
        --conserve_hbonds \\
        --run "number_of_batches=35;temperature=0.1" \\
        --run "number_of_batches=35;temperature=0.2"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import hbond_conservation as hbc          # noqa: E402
import pdb_remarks                        # noqa: E402

LOGGER = logging.getLogger("mpnn.orchestrator")

MPNN_RUNNER = _HERE / "run_ligandmpnn.py"

# Flags a sweep sets per combination; they are stripped from the pass-through
# list so a universal value cannot collide with a per-combo one.
_SWEPT = ["--temperature", "--number_of_batches", "--batch_size",
          "--omit_AA", "--bias_AA", "--packed_suffix", "--out_folder"]


# ---------------------------------------------------------------------------
# Pass-through argument handling
# ---------------------------------------------------------------------------
def _flag_value(extras: list[str], name: str) -> Optional[str]:
    for i, tok in enumerate(extras):
        if tok == name and i + 1 < len(extras):
            return extras[i + 1]
        if tok.startswith(name + "="):
            return tok.split("=", 1)[1]
    return None


def _strip_flags(extras: list[str], names: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(extras):
        tok = extras[i]
        if tok in names:
            i += 2                                  # drop the flag and its value
            continue
        if any(tok.startswith(n + "=") for n in names):
            i += 1
            continue
        out.append(tok)
        i += 1
    return out


@dataclass
class RunSpec:
    """One point in the sweep."""

    tag: str
    out_folder: str
    packed_suffix: str
    overrides: dict = field(default_factory=dict)


def _temp_tag(value: Optional[str]) -> str:
    """`0.15` -> `t0_15`, so a suffix is filename-safe and readable."""
    if value is None:
        return ""
    return "t" + str(value).replace(".", "_")


def _parse_run_string(spec: str) -> dict:
    """Parse ``k=v;k=v``. Semicolons separate fields so a bias like
    ``K:-0.5,R:-0.75`` keeps its commas."""
    out: dict = {}
    for field_str in spec.split(";"):
        field_str = field_str.strip()
        if not field_str or "=" not in field_str:
            continue
        key, value = field_str.split("=", 1)
        out[key.strip()] = value.strip()
    return out


# Swept flags a combination may override. The rest of _SWEPT is set by us.
_SWEEPABLE_KEYS = ("temperature", "number_of_batches", "batch_size",
                   "omit_AA", "bias_AA")


def build_runs(args: argparse.Namespace, base_out: str,
               extras: list[str]) -> tuple[list[RunSpec], bool]:
    """Turn ``--run`` specs into staging directories and suffixes.

    A combination overrides only the keys it names; everything else inherits the
    universal value the caller passed. Those universal flags get stripped from
    the pass-through list (a sweep sets them per run), so they are captured here
    first -- otherwise a universal --omit_AA would silently vanish from every
    run that did not happen to override it.
    """
    if not args.run:
        return [RunSpec(tag="", out_folder=base_out, packed_suffix="")], False

    universal = {}
    for key in _SWEEPABLE_KEYS:
        value = _flag_value(extras, f"--{key}")
        if value is not None:
            universal[key] = value

    runs: list[RunSpec] = []
    seen: set[str] = set()
    for i, spec in enumerate(args.run, start=1):
        combo = _parse_run_string(spec)
        tag = combo.pop("tag", "") or _temp_tag(combo.get("temperature")) or f"c{i}"
        while tag in seen:                          # keep suffixes unique
            tag = f"{tag}_{i}"
        seen.add(tag)
        runs.append(RunSpec(
            tag=tag,
            out_folder=str(Path(base_out) / f"_stage_{tag}"),
            packed_suffix=f"_{tag}",
            overrides={**universal, **combo},
        ))
    return runs, True


def build_mpnn_argv(extras: list[str], run: RunSpec, injected: list[str],
                    sweeping: bool) -> list[str]:
    argv = _strip_flags(extras, _SWEPT) if sweeping else list(extras)
    if sweeping:
        for key, value in run.overrides.items():
            if value not in (None, ""):
                argv += [f"--{key}", str(value)]
        argv += ["--packed_suffix", run.packed_suffix, "--out_folder", run.out_folder]
    return argv + injected


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------
def write_fixed_residues_json(pdb_path: str, labels: list[str], out_json: Path) -> Path:
    """Write LigandMPNN's ``--fixed_residues_multi`` payload.

    Keyed by the literal ``--pdb_path`` string, deliberately not a resolved one:
    LigandMPNN looks the key up verbatim, so on symlinked scratch a resolved
    path would not match and every residue would be redesigned.
    """
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({str(pdb_path): sorted(set(labels))}, indent=2))
    return out_json


def write_omit_nterm_met_json(pdb_path: str, label: str, out_json: Path) -> Path:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({str(pdb_path): {label: "M"}}, indent=2))
    return out_json


def first_protein_residue(pdb_path: str) -> Optional[tuple[str, int]]:
    try:
        with open(pdb_path) as fh:
            for line in fh:
                if line.startswith("ATOM"):
                    try:
                        return (line[21].strip() or "A", int(line[22:26]))
                    except ValueError:
                        return None
    except OSError:
        return None
    return None


def user_fixed_labels(extras: list[str]) -> set[str]:
    """Labels the caller already pinned, so we do not double-fix them."""
    labels: set[str] = set()
    single = _flag_value(extras, "--fixed_residues")
    if single:
        labels.update(single.split())
    multi = _flag_value(extras, "--fixed_residues_multi")
    if multi and Path(multi).is_file():
        try:
            for value in json.loads(Path(multi).read_text()).values():
                labels.update(value if isinstance(value, list) else str(value).split())
        except (OSError, ValueError):
            pass
    return labels


def protein_resnos(pdb_path: str, chain: str) -> set[int]:
    out: set[int] = set()
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("ATOM  ") and line[21] == chain:
                try:
                    out.add(int(line[22:26]))
                except ValueError:
                    pass
    return out


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
def flatten_outputs(base_out: str, runs: list[RunSpec], subdir: str,
                    keep_intermediates: bool) -> int:
    """Move every run's packed PDBs up into one flat directory."""
    base = Path(base_out)
    moved = 0
    for run in runs:
        stage = Path(run.out_folder)
        src = stage / subdir
        if src.is_dir():
            for pdb in sorted(src.glob("*.pdb")):
                os.replace(pdb, base / pdb.name)
                moved += 1
        if keep_intermediates:
            seqs = stage / "seqs"
            if seqs.is_dir():
                for fasta in sorted(seqs.glob("*")):
                    if fasta.is_file():
                        stem, ext = os.path.splitext(fasta.name)
                        suffix = f"__{run.tag}" if run.tag else ""
                        os.replace(fasta, base / f"{stem}{suffix}{ext}")
        if stage.resolve() != base.resolve():
            shutil.rmtree(stage, ignore_errors=True)
        else:
            for d in ("seqs", "backbones", "packed", "stats"):
                shutil.rmtree(base / d, ignore_errors=True)
    return moved


def run_streaming(argv: list[str], label: str, dry_run: bool) -> int:
    if dry_run:
        print(f"  [{label}] would run:\n    {shlex.join(argv)}")
        return 0
    print(f"  [{label}] {shlex.join(argv)}")
    return subprocess.run(argv).returncode


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Unrecognized arguments are passed straight through to LigandMPNN.")

    p.add_argument("--run", action="append", default=[], metavar="'k=v;k=v'",
                   help="One sweep combination, e.g. 'number_of_batches=35;temperature=0.2'. "
                        "Repeat for more. Semicolons separate fields so bias values keep "
                        "their commas. Recognized keys: temperature, number_of_batches, "
                        "batch_size, omit_AA, bias_AA, tag.")

    g = p.add_argument_group("pre-processing")
    g.add_argument("--no_fix_remark666_catres", action="store_true",
                   help="Do not auto-fix the REMARK 666 catalytic residues.")
    g.add_argument("--no_omit_nterm_met", action="store_true",
                   help="Allow Met at residue 1 (it is normally supplied by the tag).")

    g = p.add_argument_group("H-bond side-chain conservation")
    g.add_argument("--conserve_hbonds", action="store_true",
                   help="Pin designable side chains that H-bond the active site.")
    g.add_argument("--conserve_hbond_prob", type=float, default=0.8,
                   help="Per-residue probability of pinning a candidate (default 0.8).")
    g.add_argument("--conserve_anchors", default="ligand,catalytic,user_fixed",
                   help="Which partners count as the active site (comma separated).")
    g.add_argument("--conserve_hbond_max_dist", type=float, default=hbc.DEFAULT_MAX_DIST)
    g.add_argument("--conserve_hbond_max_angle", type=float, default=hbc.DEFAULT_MAX_ANGLE_DEG,
                   help="Antecedent-donor-acceptor gate; larger is more permissive.")
    g.add_argument("--conserve_hbond_all_or_none", action="store_true",
                   help="Roll once and apply to every combination, instead of per combo.")
    g.add_argument("--conserve_keep_clashing", action="store_true",
                   help="Keep candidates whose side chain clashes with fixed atoms.")
    g.add_argument("--conserve_seed", type=int, default=None,
                   help="Seed for the rolls. One is generated and printed if omitted.")

    g = p.add_argument_group("post-processing")
    g.add_argument("--no_protonate", action="store_true",
                   help="Leave designs heavy-atom only.")
    g.add_argument("--ligand_params", nargs="*", default=[],
                   help="Rosetta .params for the ligand. Omitted means apo protonation "
                        "with the ligand copied back from the input.")
    g.add_argument("--ptm", default=None,
                   help='PTM annotation, e.g. "A/LYS/3:KCX".')
    g.add_argument("--protonate_subdir", default=None,
                   help="Which LigandMPNN output dir to keep (default: packed, or "
                        "backbones when --pack_side_chains 0).")
    g.add_argument("--no_copy_input_structure", action="store_true")
    g.add_argument("--no_transfer_remarks", action="store_true")
    g.add_argument("--no_design_path_remark", action="store_true")
    g.add_argument("--keep_intermediates", action="store_true",
                   help="Keep sequence FASTAs and the generated helper JSONs.")

    p.add_argument("--dry_run", action="store_true",
                   help="Print the commands that would run and stop.")
    p.add_argument("--quiet", action="store_true")
    return p.parse_known_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args, extras = parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(levelname)s %(message)s")

    pdb_path = _flag_value(extras, "--pdb_path")
    base_out = _flag_value(extras, "--out_folder")
    if not pdb_path or not base_out:
        sys.exit("--pdb_path and --out_folder are required (they pass through to LigandMPNN).")
    if not Path(pdb_path).is_file():
        sys.exit(f"input PDB not found: {pdb_path}")
    if not MPNN_RUNNER.is_file():
        sys.exit(f"LigandMPNN runner not found: {MPNN_RUNNER}")
    Path(base_out).mkdir(parents=True, exist_ok=True)
    helpers = Path(base_out) / "_pre"

    catres = pdb_remarks.parse_remark_666(pdb_path)
    catres_labels = [cr.label for cr in catres]
    chain = catres[0].chain if catres else "A"
    already_fixed = user_fixed_labels(extras)

    print(f"### {Path(pdb_path).name}")
    if catres:
        print(f"  REMARK 666: {len(catres)} catalytic residue(s) -> {', '.join(catres_labels)}")
    elif not args.no_fix_remark666_catres:
        LOGGER.warning("no REMARK 666 block in %s -- running a FULL redesign; "
                       "nothing will be held fixed.", pdb_path)

    # ---- fixed residues -----------------------------------------------------
    injected: list[str] = []
    shared_fixed_json: Optional[Path] = None
    if catres_labels and not args.no_fix_remark666_catres and not already_fixed:
        shared_fixed_json = write_fixed_residues_json(
            pdb_path, catres_labels, helpers / "fixed_residues.json")

    # ---- N-terminal methionine ---------------------------------------------
    if not args.no_omit_nterm_met:
        first = first_protein_residue(pdb_path)
        if first:
            label = f"{first[0]}{first[1]}"
            if label not in set(catres_labels) | already_fixed:
                omit_json = write_omit_nterm_met_json(
                    pdb_path, label, helpers / "omit_nterm_met.json")
                injected += ["--omit_AA_per_residue_multi", str(omit_json)]
                print(f"  omitting Met at the N-terminus ({label})")

    # ---- H-bond conservation ------------------------------------------------
    conserve_candidates: list[str] = []
    conserve_base: set[str] = set()
    conserve_prob = 0.8
    seed = args.conserve_seed
    if args.conserve_hbonds:
        try:
            conserve_prob = hbc.normalize_probability(args.conserve_hbond_prob)
        except ValueError as exc:
            sys.exit(f"--conserve_hbond_prob {exc}")
        anchors = {a.strip() for a in (args.conserve_anchors or "").split(",") if a.strip()}
        cat_resnos = {cr.resno for cr in catres}
        fixed_resnos = {int(l[1:]) for l in already_fixed if l[1:].isdigit()}
        designable = sorted(protein_resnos(pdb_path, chain) - cat_resnos - fixed_resnos)

        records = hbc.find_conservable_sidechain_hbonds(
            pdb_path,
            designable_resnos=designable,
            catalytic_resnos=cat_resnos if "catalytic" in anchors else set(),
            user_fixed_resnos=fixed_resnos if "user_fixed" in anchors else set(),
            include_ligand="ligand" in anchors,
            chain=chain,
            max_dist=args.conserve_hbond_max_dist,
            max_angle_deg=args.conserve_hbond_max_angle,
        )
        print(f"  H-bond conservation: {len(records)} bond(s) from designable side chains")
        for r in sorted(records, key=lambda r: (r.resno, r.distance)):
            partner = f"{r.partner_resname}{r.partner_resno if r.partner_resno > 0 else ''}"
            flag = f"  [CLASH {r.clash_with}]" if r.clashes else ""
            print(f"    {chain}{r.resno} {r.resname} {r.sidechain_atom} <-> "
                  f"{r.partner_kind} {partner} {r.partner_atom}  "
                  f"d={r.distance} {r.strength_bin}{flag}")
        candidates, excluded = hbc.select_conservable_resnos(
            records, keep_clashing=args.conserve_keep_clashing)
        for resno, why in excluded:
            print(f"    excluding {chain}{resno}: clashes with {why} "
                  f"(--conserve_keep_clashing to keep)")
        conserve_candidates = [f"{chain}{r}" for r in candidates]
        conserve_base = set(catres_labels) | already_fixed
        # The per-combo JSON becomes the only fixed-residue source while
        # conserving, so drop the shared one and any caller-supplied flags --
        # they are folded into conserve_base above.
        shared_fixed_json = None
        extras = _strip_flags(extras, ["--fixed_residues", "--fixed_residues_multi"])
        if seed is None:
            seed = random.randrange(2 ** 31)
            print(f"    roll seed = {seed} (pass --conserve_seed {seed} to replay)")

    if shared_fixed_json is not None:
        injected += ["--fixed_residues_multi", str(shared_fixed_json)]

    # ---- runs ---------------------------------------------------------------
    runs, sweeping = build_runs(args, base_out, extras)
    once: Optional[set[str]] = None
    failures = 0
    for i, run in enumerate(runs):
        Path(run.out_folder).mkdir(parents=True, exist_ok=True)
        per_run = list(injected)
        if args.conserve_hbonds:
            rng = random.Random(seed if args.conserve_hbond_all_or_none else seed + i)
            if args.conserve_hbond_all_or_none:
                if once is None:
                    once = hbc.roll_conserved(conserve_candidates, conserve_prob, rng)
                keep = once
            else:
                keep = hbc.roll_conserved(conserve_candidates, conserve_prob, rng)
            labels = sorted(conserve_base | keep)
            combo_json = write_fixed_residues_json(
                pdb_path, labels, helpers / f"fixed_residues{run.packed_suffix or ''}.json")
            per_run += ["--fixed_residues_multi", str(combo_json)]
            if keep:
                print(f"  [{run.tag or 'run'}] conserving {len(keep)}/"
                      f"{len(conserve_candidates)}: {', '.join(sorted(keep))}")

        argv_mpnn = [sys.executable, str(MPNN_RUNNER)] + build_mpnn_argv(
            extras, run, per_run, sweeping)
        rc = run_streaming(argv_mpnn, run.tag or "mpnn", args.dry_run)
        if rc != 0:
            LOGGER.error("LigandMPNN failed for %s (exit %d)", run.tag or "run", rc)
            failures += 1

    if args.dry_run:
        return 0
    if failures == len(runs):
        LOGGER.error("every run failed; nothing to post-process")
        return 1

    # ---- post-processing ----------------------------------------------------
    pack = _flag_value(extras, "--pack_side_chains")
    subdir = args.protonate_subdir or ("backbones" if str(pack) == "0" else "packed")
    moved = flatten_outputs(base_out, runs, subdir, args.keep_intermediates)
    print(f"  flattened {moved} design(s) into {base_out}")

    if not args.no_protonate:
        import protonate_designs
        protonate_designs.protonate_directory(
            base_out, pdb_path, ligand_params=args.ligand_params,
            ptm_spec=args.ptm, keep_intermediate=args.keep_intermediates)
    elif not args.no_transfer_remarks:
        n = pdb_remarks.transfer_remarks_to_dir(base_out, pdb_path)
        print(f"  transferred REMARK 666 onto {n} design(s)")

    if not args.no_design_path_remark:
        for pdb in sorted(Path(base_out).glob("*.pdb")):
            pdb_remarks.add_design_path_remark(pdb, str(Path(base_out).resolve()))

    if not args.no_copy_input_structure:
        dest = Path(base_out) / Path(pdb_path).name
        if dest.resolve() != Path(pdb_path).resolve():
            shutil.copy2(pdb_path, dest)
            block = pdb_remarks.build_remark_668_block(dest, dest, ptm_map=args.ptm)
            pdb_remarks.replace_remark_block(dest, block)
            print(f"  copied the input alongside the designs ({dest.name})")

    if not args.keep_intermediates:
        shutil.rmtree(helpers, ignore_errors=True)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
