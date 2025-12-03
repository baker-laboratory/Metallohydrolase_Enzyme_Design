#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
###########################################################################
### PDB RESIDUE FILTER (KEEP/REMOVE) + REMARK 666 RENUMBERING (Robust)  ###
###########################################################################
Goal
----
Filter a PDB by residues (Chain + ResidueNumber), while:
  • Preserving or removing ATOM/HETATM records per your KEEP/REMOVE list
  • Optionally auto-keeping ligands (HETATM) unless explicitly disabled
  • Filtering/renumbering "REMARK 666 ... MATCH MOTIF ..." lines to match kept residues
  • Producing a clear, verbose summary of what was kept/removed

Token format for residues:
  • "A92" (Chain A, residue 92) — insertion codes are ignored for matching
  • Mixed case is accepted; leading zeros ignored (e.g., A092 → A92)

Examples
--------
# KEEP a set of protein residues; auto-keep ligands (default), write to out.pdb
python pdb_filter_residues.py \
  --input_pdb_to_split in.pdb \
  --residue_list_to_keep A92 A93 A96 A102 \
  --output_pdb out.pdb \
  --verbose

# REMOVE specific residues; still auto-keep ligands (unless disabled)
python pdb_filter_residues.py \
  --input_pdb_to_split in.pdb \
  --residue_list_to_remove A147 A149 \
  --output_pdb out.pdb

# KEEP only what you list (including ligands); disable auto-keep ligands
python pdb_filter_residues.py \
  --input_pdb_to_split in.pdb \
  --residue_list_to_keep A92 A93 A96 A102 B2 \
  --do_not_automatically_keep_ligands__SPECIFY_THEM_IN_LIST \
  --output_pdb out.pdb

Notes
-----
• By default, HETATM waters (HOH/WAT/DOD) are NOT auto-kept. Use
  --auto_keep_hetatm_include_water to include waters in auto-keep mode.
• TER lines are pruned if they would be orphaned (no preceding kept coords).
"""

import argparse
import re
import sys
from pathlib import Path

############################
# CONSTANTS / CONFIG
############################

WATER_NAMES = {"HOH", "WAT", "DOD"}  # common water/reservoir names

# REMARK 666 "MATCH MOTIF" regex (captures chain/residue and the constraint index)
REMARK_MOTIF_RE = re.compile(
    r'^(REMARK 666\s+MATCH TEMPLATE\s+.*?MATCH MOTIF\s+)'   # 1: preamble up to 'MATCH MOTIF '
    r'([A-Za-z0-9])\s+'                                     # 2: chain
    r'([A-Z]{3})\s+'                                        # 3: resname
    r'(\d+)\s+'                                             # 4: resseq
    r'(\d+)'                                                # 5: constraint index (to be renumbered)
    r'(\s+.*)$'                                             # 6: tail (keep as-is)
)

ATOM_RECS = ("ATOM  ", "HETATM")

############################
# HELPER FUNCTIONS
############################

def vprint(flag, *args, **kwargs):
    """Verbose print helper."""
    if flag:
        print(*args, **kwargs)

def normalize_token(tok: str) -> str:
    """
    Normalize a residue token like 'A092' → 'A92', 'a 92' → 'A92'.
    Insertion codes are ignored (A92A treated as A92) for matching purposes.
    """
    t = tok.strip().replace(" ", "")
    m = re.fullmatch(r'([A-Za-z])0*([0-9]+)[A-Za-z]?$', t)  # optional trailing insertion letter ignored
    if not m:
        raise ValueError(f"Bad residue token: '{tok}'. Use like 'A92' or 'A 92'.")
    chain = m.group(1).upper()
    resi  = int(m.group(2))
    return f"{chain}{resi}"

def parse_atom_hetatm_key(line: str):
    """
    Parse chain/resi key from ATOM/HETATM line (PDB fixed columns).
    Returns (key, resname, is_hetatm) where key is like 'A92'.
    Returns (None, None, None) if parsing fails (line likely malformed).
    """
    try:
        rec = line[0:6]
        if rec not in ATOM_RECS:
            return (None, None, None)
        chain = line[21].strip()          # column 22 (1-char)
        resseq_str = line[22:26].strip()  # columns 23-26
        resname = line[17:20].strip().upper()
        if not chain or not resseq_str:
            return (None, None, None)
        resi = int(resseq_str)
        key  = f"{chain}{resi}"
        is_het = (rec == "HETATM")
        return (key, resname, is_het)
    except Exception:
        return (None, None, None)

def parse_remark_motif_key(line: str):
    """
    If line is a REMARK 666 MATCH MOTIF, return (key, matchobj).
    Else return (None, None).
    """
    m = REMARK_MOTIF_RE.match(line)
    if not m:
        return (None, None)
    chain = m.group(2)
    resi  = m.group(4)
    key = f"{chain}{int(resi)}"
    return (key, m)

def rebuild_remark_with_new_index(m: re.Match, new_index: int, index_width: int = 3) -> str:
    pre  = m.group(1)
    ch   = m.group(2)
    resn = m.group(3)
    resi = m.group(4)
    tail = m.group(6)
    return f"{pre}{ch} {resn:>3} {int(resi):>4} {int(new_index):>{index_width}}{tail}"

def build_target_set(keep_list, remove_list):
    """
    Validate exclusivity and build target set + mode.
    Returns (targets: set[str], keep_mode: bool)
    """
    if (keep_list and remove_list) or (not keep_list and not remove_list):
        raise ValueError("Specify exactly one of --residue_list_to_keep or --residue_list_to_remove.")
    raw = keep_list if keep_list else remove_list
    targets = {normalize_token(t) for t in raw}
    keep_mode = bool(keep_list)
    return targets, keep_mode

############################
# CORE FILTERING
############################

def should_keep_coord_line(
    line: str,
    targets: set,
    keep_mode: bool,
    auto_keep_hetatm: bool,
    auto_keep_hetatm_include_water: bool,
):
    """
    Decide whether to keep an ATOM/HETATM line given the mode and flags.
    """
    key, resname, is_het = parse_atom_hetatm_key(line)
    if key is None:
        # Malformed coord line: keep to avoid unintended loss
        return True

    # Auto-keep ligands (HETATM), unless disabled
    if is_het and auto_keep_hetatm:
        if (not auto_keep_hetatm_include_water) and resname in WATER_NAMES:
            # treat as normal coord under keep/remove logic
            pass
        else:
            return True

    # Normal residue-based logic
    if keep_mode:
        return key in targets
    else:
        return key not in targets

def filter_lines(
    lines,
    targets: set,
    keep_mode: bool,
    auto_keep_hetatm: bool,
    auto_keep_hetatm_include_water: bool,
    prune_orphan_TER: bool,
    verbose: bool,
):
    """
    Apply filtering to all lines:
      • ATOM/HETATM per residue logic + auto-keep(hetatm)
      • REMARK 666 MATCH MOTIF lines filtered by residue logic and renumbered
      • Other lines pass through
      • Optionally prune orphan TER lines
    Returns (kept_lines: list[str], stats: dict)
    """
    kept = []
    motif_slots = []  # (index_in_kept_placeholder, matchobj)
    stats = {
        "input_lines": 0,
        "kept_lines": 0,
        "kept_atom": 0,
        "kept_hetatm": 0,
        "kept_other": 0,
        "kept_motif": 0,
        "removed_atom": 0,
        "removed_hetatm": 0,
        "removed_motif": 0,
        "removed_other": 0,  # (rare) only if we prune TER
        "orphan_TER_pruned": 0,
    }

    # First, keep/remove lines (insert placeholders for motif we keep)
    for line in lines:
        stats["input_lines"] += 1

        if line.startswith(ATOM_RECS):
            keepit = should_keep_coord_line(
                line, targets, keep_mode,
                auto_keep_hetatm, auto_keep_hetatm_include_water
            )
            if keepit:
                kept.append(line)
                # tally atom vs hetatm
                if line.startswith("ATOM  "):
                    stats["kept_atom"] += 1
                else:
                    stats["kept_hetatm"] += 1
            else:
                if line.startswith("ATOM  "):
                    stats["removed_atom"] += 1
                else:
                    stats["removed_hetatm"] += 1

        elif line.startswith("REMARK 666"):
            key, m = parse_remark_motif_key(line)
            if m:
                # Apply same residue selection to the motif remark
                keepit = (key in targets) if keep_mode else (key not in targets)
                if keepit:
                    kept.append(None)  # placeholder; renumber later
                    motif_slots.append((len(kept) - 1, m))
                else:
                    stats["removed_motif"] += 1
            else:
                kept.append(line)
                stats["kept_other"] += 1

        elif line.startswith("TER"):
            # Temporarily keep; may prune as orphan later
            kept.append(line)
            stats["kept_other"] += 1

        else:
            kept.append(line)
            stats["kept_other"] += 1

    # Renumber motif REMARKs 1..N in kept order
    for new_idx, (slot, m) in enumerate(motif_slots, start=1):
        rebuilt = rebuild_remark_with_new_index(m, new_idx)
        kept[slot] = rebuilt + ("" if rebuilt.endswith("\n") else "\n")
        stats["kept_motif"] += 1

    # Optionally prune orphan TER (i.e., TER with no preceding kept coords since last TER)
    if prune_orphan_TER:
        pruned = []
        have_coords_since_last_TER = False
        for ln in kept:
            if ln is None:
                # shouldn't happen; but ignore
                continue
            if ln.startswith(ATOM_RECS):
                have_coords_since_last_TER = True
                pruned.append(ln)
            elif ln.startswith("TER"):
                if have_coords_since_last_TER:
                    pruned.append(ln)
                else:
                    stats["orphan_TER_pruned"] += 1
                    stats["removed_other"] += 1
                have_coords_since_last_TER = False
            else:
                # non-coordinate, non-TER
                pruned.append(ln)
        kept = pruned

    stats["kept_lines"] = len(kept)
    vprint(verbose, f"[filter] Kept {stats['kept_lines']} / {stats['input_lines']} lines.")
    return kept, stats

def write_lines(path: Path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ln in lines:
            if ln is None:
                continue
            f.write(ln if ln.endswith("\n") else ln + "\n")

def print_summary(stats: dict, keep_mode: bool, auto_keep_hetatm: bool, include_waters: bool, verbose: bool):
    mode = "KEEP-LIST MODE" if keep_mode else "REMOVE-LIST MODE"
    auto = "ON" if auto_keep_hetatm else "OFF"
    water = "INCLUDED" if include_waters else "EXCLUDED"
    print("\n===== PDB FILTER SUMMARY =====")
    print(f"Mode: {mode}")
    print(f"Auto-keep HETATM ligands: {auto} (waters: {water})")
    print(f"Input lines:  {stats['input_lines']}")
    print(f"Kept lines:   {stats['kept_lines']}")
    print(f"  ATOM kept:      {stats['kept_atom']}")
    print(f"  HETATM kept:    {stats['kept_hetatm']}")
    print(f"  REMARK kept:    {stats['kept_motif']}")
    print(f"  Other kept:     {stats['kept_other']}")
    print(f"Removed:")
    print(f"  ATOM removed:   {stats['removed_atom']}")
    print(f"  HETATM removed: {stats['removed_hetatm']}")
    print(f"  REMARK removed: {stats['removed_motif']}")
    print(f"  Other removed:  {stats['removed_other']}")
    print(f"Orphan TER pruned: {stats['orphan_TER_pruned']}")
    if verbose:
        print("==============================\n")

############################
# CLI
############################

def main():
    ap = argparse.ArgumentParser(
        description="Filter PDB by residues and renumber REMARK 666 motif indices; optionally auto-keep ligands."
    )
    ap.add_argument("--input_pdb_to_split", required=True, help="Input PDB file path.")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--residue_list_to_keep", nargs="+", help="Residues to KEEP (e.g., A92 A93).")
    group.add_argument("--residue_list_to_remove", nargs="+", help="Residues to REMOVE (e.g., A147 A149).")
    ap.add_argument("--output_pdb", required=True, help="Output PDB path (will be overwritten).")

    # Ligand handling
    ap.add_argument(
        "--do_not_automatically_keep_ligands__SPECIFY_THEM_IN_LIST",
        action="store_true",
        help="Disable auto-keep for HETATM. If set, HETATMs follow the same keep/remove rules as ATOM. "
             "If you want to KEEP a ligand in this mode, you must list its residue token (e.g., B2)."
    )
    ap.add_argument(
        "--auto_keep_hetatm_include_water",
        action="store_true",
        help="When auto-keeping ligands, also keep water HETATMs (HOH/WAT/DOD)."
    )

    # Other behaviors
    ap.add_argument("--no_prune_orphan_TER", action="store_true", help="Do not prune orphan TER lines.")
    ap.add_argument("--dry_run", action="store_true", help="Read/process/summary only; do not write output.")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging.")

    args = ap.parse_args()

    # Resolve mode and targets
    targets, keep_mode = build_target_set(args.residue_list_to_keep, args.residue_list_to_remove)

    # Resolve ligand policy
    auto_keep_hetatm = not args.do_not_automatically_keep_ligands__SPECIFY_THEM_IN_LIST
    include_waters = args.auto_keep_hetatm_include_water
    prune_orphan_TER = not args.no_prune_orphan_TER

    # IO
    in_path = Path(args.input_pdb_to_split)
    out_path = Path(args.output_pdb)

    if not in_path.exists():
        raise FileNotFoundError(f"Input PDB not found: {in_path}")

    # Read all lines
    with open(in_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    vprint(args.verbose, f"[load] Read {len(lines)} lines from {in_path}")

    # Filter
    kept_lines, stats = filter_lines(
        lines=lines,
        targets=targets,
        keep_mode=keep_mode,
        auto_keep_hetatm=auto_keep_hetatm,
        auto_keep_hetatm_include_water=include_waters,
        prune_orphan_TER=prune_orphan_TER,
        verbose=args.verbose,
    )

    # Summary
    print_summary(stats, keep_mode, auto_keep_hetatm, include_waters, args.verbose)

    # Dry run?
    if args.dry_run:
        print("[dry-run] Skipping write.")
        return

    # Write output
    write_lines(out_path, kept_lines)
    print(f"[write] Wrote filtered PDB to: {out_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
