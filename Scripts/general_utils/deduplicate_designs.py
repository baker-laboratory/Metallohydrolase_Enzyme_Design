#!/usr/bin/env python3
"""
Deduplicate designed PDBs by protein sequence within scaffold_family groups.

Sequence design produces many variants per input scaffold, and after several
rounds the same sequence often reappears from different branches. Grouping by
scaffold family first means only genuinely redundant designs are collapsed --
two scaffolds that happen to converge on a sequence are left alone.

A scaffold family is the input stem with the per-stage suffixes removed; the
tokens to split on are configurable with --family_tokens.

Duplicates are moved into a subdirectory inside their original parent folder,
so the original folder layout is preserved and the duplicate PDBs are backed up
instead of deleted.

For each duplicate sequence group, the representative kept in place is the PDB
with the shortest basename when basename lengths differ. If all basenames have
the same length, the representative is chosen randomly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import shutil
import sys
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    # Common modified residues mapped to the parent residue when appropriate.
    "ASH": "D",
    "CYM": "C",
    "CYX": "C",
    "GLH": "E",
    "HID": "H",
    "HIE": "H",
    "HIP": "H",
    "KCX": "K",
    "MSE": "M",
    "SEC": "U",
    "SEP": "S",
    "TPO": "T",
    "PTR": "Y",
}


@dataclass(frozen=True)
class PdbEntry:
    path: str
    subdir: str
    family: str
    status: str
    sequence: str = ""
    sequence_hash: str = ""
    sequence_length: int = 0
    error: str = ""


@dataclass(frozen=True)
class DuplicatePlan:
    duplicate: PdbEntry
    representative: PdbEntry
    destination: str


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{timestamp()}] {message}", flush=True)


DEFAULT_FAMILY_TOKENS = ("_inp", "_FS")


def scaffold_family_from_name(name: str, tokens: Tuple[str, ...] = DEFAULT_FAMILY_TOKENS) -> str:
    """Reduce a design or scaffold name to the family it belongs to."""
    family = name
    for token in tokens:
        family = family.split(token, 1)[0]
    return family


def list_pdbs_for_subdir(subdir: Path, pdb_glob: str) -> Tuple[str, List[str]]:
    pdbs = sorted(
        str(p)
        for p in subdir.glob(pdb_glob)
        if p.is_file() and p.suffix.lower() == ".pdb"
    )
    return str(subdir), pdbs


def discover_pdbs(design_dir: Path, pdb_glob: str, workers: int,
                  family_tokens: Tuple[str, ...] = DEFAULT_FAMILY_TOKENS,
                  ) -> List[Tuple[str, str, str]]:
    subdirs = sorted(
        [p for p in design_dir.iterdir() if p.is_dir()],
        key=lambda p: p.name.lower(),
    )
    log(f"Found {len(subdirs)} immediate subdirectories in {design_dir}")
    jobs: List[Tuple[str, str, str]] = []
    if not subdirs:
        return jobs

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(list_pdbs_for_subdir, subdir, pdb_glob) for subdir in subdirs]
        for future in as_completed(futures):
            subdir, pdbs = future.result()
            family = scaffold_family_from_name(Path(subdir).name, family_tokens)
            jobs.extend((pdb, subdir, family) for pdb in pdbs)

    jobs.sort(key=lambda x: (x[2].lower(), Path(x[1]).name.lower(), Path(x[0]).name.lower()))
    log(f"Found {len(jobs)} top-level PDB files across those subdirectories")
    return jobs


def normalize_chain_filter(protein_chains: Optional[Sequence[str]]) -> Optional[set]:
    if not protein_chains:
        return None
    chains = {chain.strip() for chain in protein_chains if chain.strip()}
    return chains or None


def extract_sequence_from_pdb(pdb_path: str, protein_chains: Optional[Sequence[str]]) -> str:
    chain_filter = normalize_chain_filter(protein_chains)
    residues_by_chain: "OrderedDict[str, OrderedDict[Tuple[str, str, str], str]]" = OrderedDict()

    with open(pdb_path, "r", errors="replace") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            if len(line) < 27:
                continue
            chain_id = line[21].strip() or "_"
            if chain_filter is not None and chain_id not in chain_filter:
                continue
            resname = line[17:20].strip().upper()
            resseq = line[22:26].strip()
            icode = line[26].strip()
            residue_key = (chain_id, resseq, icode)
            if chain_id not in residues_by_chain:
                residues_by_chain[chain_id] = OrderedDict()
            if residue_key in residues_by_chain[chain_id]:
                continue
            residues_by_chain[chain_id][residue_key] = AA3_TO_1.get(resname, "X")

    chain_sequences = []
    for chain_id, residues in residues_by_chain.items():
        sequence = "".join(residues.values())
        if sequence:
            chain_sequences.append(f"{chain_id}:{sequence}")
    return "|".join(chain_sequences)


def sequence_length(sequence_key: str) -> int:
    total = 0
    for chain_seq in sequence_key.split("|"):
        if not chain_seq:
            continue
        total += len(chain_seq.split(":", 1)[-1])
    return total


def parse_pdb_entry(job: Tuple[str, str, str], protein_chains: Optional[Sequence[str]]) -> PdbEntry:
    pdb_path, subdir, family = job
    try:
        sequence = extract_sequence_from_pdb(pdb_path, protein_chains)
        if not sequence:
            return PdbEntry(
                path=pdb_path,
                subdir=subdir,
                family=family,
                status="no_sequence",
                error="No ATOM protein residues found for selected chain(s)",
            )
        digest = hashlib.sha1(sequence.encode("utf-8")).hexdigest()
        return PdbEntry(
            path=pdb_path,
            subdir=subdir,
            family=family,
            status="ok",
            sequence=sequence,
            sequence_hash=digest,
            sequence_length=sequence_length(sequence),
        )
    except Exception as exc:
        return PdbEntry(
            path=pdb_path,
            subdir=subdir,
            family=family,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
        )


def parse_all_sequences(
    jobs: List[Tuple[str, str, str]],
    protein_chains: Optional[Sequence[str]],
    workers: int,
    progress_every: int,
) -> List[PdbEntry]:
    entries: List[PdbEntry] = []
    total = len(jobs)
    start = time.time()
    if not jobs:
        return entries

    log(f"Parsing sequences with {workers} worker(s)")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(parse_pdb_entry, job, protein_chains) for job in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            entries.append(future.result())
            if progress_every and (i == total or i % progress_every == 0):
                elapsed = time.time() - start
                rate = i / elapsed if elapsed > 0 else 0.0
                log(f"Parsed {i}/{total} PDBs ({rate:.1f} PDB/s)")

    entries.sort(key=lambda e: (e.family.lower(), Path(e.subdir).name.lower(), Path(e.path).name.lower()))
    return entries


def unique_destination(src: Path, duplicates_subdir: str) -> Path:
    dest_dir = src.parent / duplicates_subdir
    candidate = dest_dir / src.name
    if not candidate.exists():
        return candidate

    for idx in range(1, 100000):
        renamed = dest_dir / f"{src.stem}__dedup_duplicate_{idx}{src.suffix}"
        if not renamed.exists():
            return renamed
    raise RuntimeError(f"Could not find a free duplicate filename for {src}")


def basename_length(entry: PdbEntry) -> int:
    return len(Path(entry.path).name)


def choose_representative(entries: List[PdbEntry], rng: random.Random) -> PdbEntry:
    lengths = [basename_length(entry) for entry in entries]
    unique_lengths = set(lengths)
    if len(unique_lengths) > 1:
        min_len = min(unique_lengths)
        shortest_entries = [entry for entry in entries if basename_length(entry) == min_len]
        return rng.choice(shortest_entries)
    return rng.choice(entries)


def build_duplicate_plan(
    entries: Iterable[PdbEntry],
    duplicates_subdir: str,
    verbose: bool,
    rng: random.Random,
) -> Tuple[List[DuplicatePlan], Dict[str, str], List[Tuple[str, int, int, int]]]:
    ok_entries = [entry for entry in entries if entry.status == "ok"]
    by_family: Dict[str, List[PdbEntry]] = defaultdict(list)
    for entry in ok_entries:
        by_family[entry.family].append(entry)

    duplicate_plans: List[DuplicatePlan] = []
    representative_by_path: Dict[str, str] = {}
    family_summaries: List[Tuple[str, int, int, int]] = []

    log(f"Grouping {len(ok_entries)} sequence-bearing PDBs into {len(by_family)} scaffold_family group(s)")
    for family in sorted(by_family, key=str.lower):
        group = sorted(by_family[family], key=lambda e: (Path(e.subdir).name.lower(), Path(e.path).name.lower()))
        by_sequence: Dict[str, List[PdbEntry]] = defaultdict(list)
        for entry in group:
            by_sequence[entry.sequence_hash].append(entry)

        duplicate_count = 0
        for sequence_hash in sorted(by_sequence):
            sequence_entries = by_sequence[sequence_hash]
            representative = choose_representative(sequence_entries, rng)
            representative_by_path[representative.path] = representative.path
            duplicates = [entry for entry in sequence_entries if entry.path != representative.path]
            duplicate_count += len(duplicates)
            for entry in duplicates:
                representative_by_path[entry.path] = representative.path
                duplicate_plans.append(
                    DuplicatePlan(
                        duplicate=entry,
                        representative=representative,
                        destination=str(unique_destination(Path(entry.path), duplicates_subdir)),
                    )
                )

        family_summaries.append((family, len(group), len(by_sequence), duplicate_count))
        if verbose and duplicate_count:
            log(
                f"Family {family}: {len(group)} PDBs, "
                f"{len(by_sequence)} unique sequences, {duplicate_count} duplicate(s)"
            )

    return duplicate_plans, representative_by_path, family_summaries


def move_duplicates(plans: List[DuplicatePlan], dry_run: bool, print_moves: bool, move_preview_limit: int) -> Tuple[int, int]:
    moved = 0
    errors = 0
    if not plans:
        log("No duplicate PDBs found")
        return moved, errors

    action = "Would move" if dry_run else "Moving"
    log(f"{action} {len(plans)} duplicate PDB(s) into their parent duplicate subdirectories")

    for i, plan in enumerate(plans, 1):
        src = Path(plan.duplicate.path)
        dest = Path(plan.destination)
        should_print = print_moves or i <= move_preview_limit
        if should_print:
            prefix = "[DRY RUN]" if dry_run else "[MOVE]"
            print(f"{prefix} {src} -> {dest} | keep {plan.representative.path}", flush=True)
        elif i == move_preview_limit + 1:
            remaining = len(plans) - move_preview_limit
            print(f"... suppressing {remaining} additional duplicate move line(s)", flush=True)

        if dry_run:
            continue

        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
            moved += 1
        except Exception as exc:
            errors += 1
            print(f"[ERROR] Failed to move {src} -> {dest}: {exc}", file=sys.stderr, flush=True)

    return moved, errors


def write_report(
    report_csv: str,
    entries: List[PdbEntry],
    plans: List[DuplicatePlan],
    representative_by_path: Dict[str, str],
    dry_run: bool,
) -> None:
    plan_by_path = {plan.duplicate.path: plan for plan in plans}
    report_path = Path(report_csv)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "path",
        "subdir",
        "scaffold_family",
        "status",
        "sequence_hash",
        "sequence_length",
        "is_duplicate",
        "representative_path",
        "duplicate_destination",
        "action",
        "error",
    ]
    with report_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            plan = plan_by_path.get(entry.path)
            is_duplicate = plan is not None
            action = ""
            if is_duplicate:
                action = "would_move" if dry_run else "moved"
            elif entry.status == "ok":
                action = "kept"
            writer.writerow(
                {
                    "path": entry.path,
                    "subdir": entry.subdir,
                    "scaffold_family": entry.family,
                    "status": entry.status,
                    "sequence_hash": entry.sequence_hash,
                    "sequence_length": entry.sequence_length,
                    "is_duplicate": int(is_duplicate),
                    "representative_path": representative_by_path.get(entry.path, ""),
                    "duplicate_destination": plan.destination if plan else "",
                    "action": action,
                    "error": entry.error,
                }
            )
    log(f"Wrote report CSV: {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deduplicate PDBs by protein sequence within scaffold_family groups."
    )
    parser.add_argument("--design_dir", "--design-dir", required=True,
                        help="Directory of per-input subdirectories holding designs")
    parser.add_argument("--family_tokens", "--family-tokens", nargs="*",
                        default=list(DEFAULT_FAMILY_TOKENS),
                        help="Name fragments marking where a scaffold family name ends")
    parser.add_argument("--pdb_glob", "--pdb-glob", default="*.pdb", help="Top-level PDB glob inside each subdirectory")
    parser.add_argument(
        "--duplicates_subdir",
        "--duplicates-subdir",
        default="duplicates",
        help="Subdirectory created inside each source folder for duplicate PDBs",
    )
    parser.add_argument("--protein_chains", "--protein-chains", nargs="*", default=None, help="Protein chain IDs to use")
    parser.add_argument("--workers", type=int, default=max(1, min(32, os.cpu_count() or 1)), help="Parallel worker count")
    parser.add_argument("--progress_every", "--progress-every", type=int, default=250, help="Progress print interval")
    parser.add_argument("--report_csv", "--report-csv", default=None, help="Optional CSV report path")
    parser.add_argument("--dry_run", "--dry-run", action="store_true", help="Preview moves without changing files")
    parser.add_argument("--verbose", action="store_true", help="Print per-family duplicate summaries")
    parser.add_argument("--print_moves", "--print-moves", action="store_true", help="Print every duplicate move")
    parser.add_argument(
        "--random_seed",
        "--random-seed",
        type=int,
        default=None,
        help="Optional seed for random representative tie-breaking",
    )
    parser.add_argument(
        "--move_preview_limit",
        "--move-preview-limit",
        type=int,
        default=20,
        help="Number of duplicate moves to print when --print_moves is not set",
    )
    parser.add_argument(
        "--max_pdbs",
        "--max-pdbs",
        type=int,
        default=None,
        help="Optional testing limit after discovery. Not intended for production deduplication.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    design_dir = Path(args.design_dir).resolve()
    if not design_dir.is_dir():
        raise FileNotFoundError(f"design_dir does not exist or is not a directory: {design_dir}")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    log("Starting scaffold_family sequence deduplication")
    log(f"design_dir:        {design_dir}")
    log(f"pdb_glob:          {args.pdb_glob}")
    log(f"duplicates_subdir: {args.duplicates_subdir}")
    log(f"protein_chains:    {args.protein_chains or 'all ATOM chains'}")
    log(f"workers:           {args.workers}")
    log(f"dry_run:           {args.dry_run}")
    log(f"random_seed:       {args.random_seed}")

    jobs = discover_pdbs(design_dir, args.pdb_glob, args.workers,
                         tuple(args.family_tokens))
    if args.max_pdbs is not None:
        jobs = jobs[: args.max_pdbs]
        log(f"Testing limit active: processing first {len(jobs)} discovered PDB(s)")

    entries = parse_all_sequences(jobs, args.protein_chains, args.workers, args.progress_every)
    status_counts = defaultdict(int)
    for entry in entries:
        status_counts[entry.status] += 1
    log("Parse status counts: " + ", ".join(f"{k}={v}" for k, v in sorted(status_counts.items())))

    plans, representative_by_path, family_summaries = build_duplicate_plan(
        entries=entries,
        duplicates_subdir=args.duplicates_subdir,
        verbose=args.verbose,
        rng=random.Random(args.random_seed),
    )
    families_with_duplicates = sum(1 for _, _, _, dup_count in family_summaries if dup_count)
    unique_sequences = sum(unique_count for _, _, unique_count, _ in family_summaries)

    moved, errors = move_duplicates(
        plans=plans,
        dry_run=args.dry_run,
        print_moves=args.print_moves,
        move_preview_limit=args.move_preview_limit,
    )

    if args.report_csv:
        write_report(args.report_csv, entries, plans, representative_by_path, args.dry_run)

    log("=" * 60)
    log("DEDUPLICATION SUMMARY")
    log(f"Total PDBs discovered:        {len(jobs)}")
    log(f"Sequence-bearing PDBs:        {status_counts.get('ok', 0)}")
    log(f"Scaffold families:            {len(family_summaries)}")
    log(f"Families with duplicates:     {families_with_duplicates}")
    log(f"Unique sequences kept:        {unique_sequences}")
    log(f"Duplicate PDBs identified:    {len(plans)}")
    if args.dry_run:
        log("Duplicate PDBs moved:         0 (dry run)")
    else:
        log(f"Duplicate PDBs moved:         {moved}")
    log(f"Move errors:                  {errors}")
    log("=" * 60)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
