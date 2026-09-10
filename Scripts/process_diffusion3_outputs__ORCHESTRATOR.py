#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestrator around process_diffusion3_outputs__REORG.py.

When the --pdb list exceeds --chunk_threshold files, this script discovers a
natural split in the filenames (by underscore-separated token position) and
runs the underlying REORG script on each chunk sequentially, then merges the
per-chunk Rosetta-style scorefiles into the requested --scorefile_out.

Usage is drop-in: pass exactly the same args you would pass to the child
script; extra orchestrator-only flags are listed below.

Orchestrator-only flags:
    --chunk_threshold INT   Max files per chunk (default: 4000).
    --child_script PATH     Path to the child script to invoke.
    --keep_chunk_scorefiles Keep per-chunk intermediate .sc files.
    --pdb_list FILE         Read --pdb entries from FILE (one path per line).
                            Useful when the full glob would exceed ARG_MAX.
    --dry_run               Print planned chunking and exit.

Glob handling:
    --pdb entries are passed through Python's glob.glob so that quoted
    patterns like --pdb '/dir/*ZAPP*.cif.gz' are expanded by the orchestrator
    itself. Quote the pattern when the shell-expanded form would exceed
    ARG_MAX (~2 MB of total argv); the quoted form is one token regardless
    of how many files the pattern matches.
"""
import argparse
import glob as _glob
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from collections import defaultdict

# The child script needs only PyRosetta + numpy/pandas, all of which the
# `zinc_hydro` conda environment provides -- no container is required. Set
# ZINC_HYDRO_SIF (or pass --sbatch_apptainer) only if you want SLURM jobs to run
# inside one. The previous default pointed at a Baker lab container that does
# not exist outside our filesystem.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from env_config import resolve_runner
    DEFAULT_APPTAINER = os.environ.get("ZINC_HYDRO_SIF") or " ".join(resolve_runner(None))
except Exception:  # env_config unavailable -- fall back to this interpreter
    DEFAULT_APPTAINER = os.environ.get("ZINC_HYDRO_SIF") or sys.executable

# The child ships alongside this file in Scripts/.
DEFAULT_CHILD = str(Path(__file__).resolve().parent / "process_diffusion3_outputs.py")
DEFAULT_THRESHOLD = 4000
STRIP_EXTS = (".cif.gz", ".pdb.gz", ".cif", ".pdb")


def strip_ext(name: str) -> str:
    for ext in STRIP_EXTS:
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def tokenize(path: str):
    base = strip_ext(os.path.basename(path))
    return base.split("_")


def group_by_position(files, pos):
    groups = defaultdict(list)
    for f in files:
        toks = tokenize(f)
        key = toks[pos] if pos < len(toks) else "__missing__"
        groups[key].append(f)
    return dict(groups)


def pick_split_position(files):
    """
    Pick the token position that minimizes the largest resulting group.
    Returns None if no position produces more than one non-trivial group.
    """
    if len(files) < 2:
        return None
    max_len = max(len(tokenize(f)) for f in files)
    best_pos = None
    best_max = len(files) + 1
    best_ngroups = 0
    for pos in range(max_len):
        groups = group_by_position(files, pos)
        if len(groups) <= 1:
            continue
        largest = max(len(g) for g in groups.values())
        # prefer the split with smallest max-chunk; tiebreak on more groups
        if largest < best_max or (largest == best_max and len(groups) > best_ngroups):
            best_max = largest
            best_pos = pos
            best_ngroups = len(groups)
    # Require actual progress: the best split must shrink the largest chunk
    if best_pos is None or best_max >= len(files):
        return None
    return best_pos


def chunk_files(files, threshold):
    """Recursively split files into chunks of size <= threshold when possible."""
    files = sorted(files)
    if len(files) <= threshold:
        return [files]
    pos = pick_split_position(files)
    if pos is None:
        # No token-based split reduces the set further; slice arbitrarily.
        return [files[i : i + threshold] for i in range(0, len(files), threshold)]
    groups = group_by_position(files, pos)
    out = []
    for key in sorted(groups):
        out.extend(chunk_files(groups[key], threshold))
    return out


def merge_scorefiles(sc_paths, out_path):
    """
    Merge Rosetta-style scorefiles. Keep the title line from the first file
    that has one; append data rows from all files. Warn on title mismatch.
    """
    canonical_title = None
    with open(out_path, "w") as out:
        for sc in sc_paths:
            if not os.path.exists(sc):
                print(f"[orchestrator] merge: missing {sc} (skipped)", file=sys.stderr)
                continue
            with open(sc) as fh:
                lines = fh.readlines()
            if not lines:
                continue
            title, data = lines[0], lines[1:]
            if canonical_title is None:
                canonical_title = title
                out.write(title)
            elif title.split() != canonical_title.split():
                print(
                    f"[orchestrator] merge: title mismatch in {sc}; appending rows anyway",
                    file=sys.stderr,
                )
            out.writelines(data)


def build_parser():
    p = argparse.ArgumentParser(
        description="Chunked orchestrator around process_diffusion3_outputs__REORG.py.",
        add_help=False,
    )
    p.add_argument("--pdb", nargs="+", default=[],
                   help="Input CIF.GZ files (shell-glob expanded).")
    p.add_argument("--pdb_list", type=str, default=None,
                   help="File containing one --pdb path per line (avoids ARG_MAX).")
    p.add_argument("--scorefile_out", type=str, default=None,
                   help="Explicit scorefile path. If omitted, a deterministic "
                        "name is built: '<prefix>_<label>.sc' (see --scorefile_prefix).")
    p.add_argument("--scorefile_prefix", type=str, default="rfdiffusion3_analysis",
                   help="Prefix for the auto-named scorefile when --scorefile_out "
                        "is not set (default: rfdiffusion3_analysis).")
    p.add_argument("--outdir", type=str, default="filtered_structures")
    p.add_argument("--chunk_threshold", type=int, default=DEFAULT_THRESHOLD)
    p.add_argument("--child_script", type=str, default=DEFAULT_CHILD)
    p.add_argument("--keep_chunk_scorefiles", action="store_true", default=False)
    p.add_argument("--dry_run", action="store_true", default=False)

    # --- sbatch / SLURM array mode -------------------------------------------
    p.add_argument("--sbatch", action="store_true", default=False,
                   help="Submit chunks as a single SLURM job array (one array "
                        "task per chunk, unless --sbatch_cmds_per_job >1). "
                        "Blocks via `sbatch --wait`, then merges scorefiles.")
    p.add_argument("--sbatch_queue", default="cpu")
    p.add_argument("--sbatch_cores", type=int, default=16,
                   help="CPU cores per array task (also passed as --nproc to "
                        "the child unless --nproc is already in passthrough). "
                        "Default 16: matches a sensible PyRosetta multiprocessing "
                        "pool size per chunk.")
    p.add_argument("--sbatch_mem", default="32G",
                   help="Memory per array task. Default 32G: PyRosetta holds "
                        "~1-2 GB per worker; 16 workers * ~1.5 GB + overhead "
                        "fits in 32 GB.")
    p.add_argument("--sbatch_time", default="01:00:00",
                   help="Wallclock limit per array task. Default 1h: each "
                        "chunk is <=4000 files; at 16 cores that finishes "
                        "well inside an hour for typical scoring.")
    p.add_argument("--sbatch_job_name", default=None,
                   help="Job name (default: rfd3_<label>).")
    p.add_argument("--sbatch_logs_dir", default=None,
                   help="Directory for stdout/stderr (default: ./logs_<label>).")
    p.add_argument("--sbatch_workdir", default=None,
                   help="Staging dir for chunk lists + scorefiles "
                        "(default: ./.rfd3_orch_<label>).")
    p.add_argument("--sbatch_apptainer", default=DEFAULT_APPTAINER,
                   help="Apptainer/Singularity image invoked per array task.")
    p.add_argument("--sbatch_cmds_per_job", type=int, default=1,
                   help="Chunks per array task (increase if you have very "
                        "many small chunks).")
    p.add_argument("--sbatch_no_wait", action="store_true", default=False,
                   help="Submit the array but return immediately; skip the "
                        "merge step. Use for fire-and-forget submissions.")

    p.add_argument("-h", "--help", action="store_true", default=False,
                   help="Show this help, then the child script's help.")
    return p


def print_help(parser, child_script):
    parser.print_help()
    print("\n--- downstream (passthrough) args from child script ---\n")
    try:
        subprocess.run([sys.executable, child_script, "--help"])
    except Exception as e:
        print(f"(could not invoke child help: {e})")


def _dedupe_preserve_order(seq):
    seen = set()
    out = []
    for item in seq:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _label_from_patterns(pdb_args):
    """
    Build a label from --pdb glob patterns by taking the literal (non-wildcard)
    fragments of each pattern's basename.

    Example:  '*ZAPP*model_0*.cif.gz'  ->  'ZAPP_model_0'
              '*ZAPP*rotP_0*ORI_11*.cif.gz' -> 'ZAPP_rotP_0_ORI_11'
    Returns None if no --pdb arg contains glob metacharacters.
    """
    frags = []
    found_pattern = False
    for entry in pdb_args:
        if not any(c in entry for c in "*?["):
            continue
        found_pattern = True
        base = strip_ext(os.path.basename(entry))
        parts = re.split(r"[*?\[\]]+", base)
        parts = [p.strip("_") for p in parts if p.strip("_")]
        frags.extend(parts)
    if not found_pattern:
        return None
    frags = _dedupe_preserve_order(frags)
    return "_".join(frags) if frags else None


def _label_from_filenames(files, max_tokens=8):
    """
    Fallback label: longest leading run of underscore-tokens that are constant
    across every resolved filename (capped at max_tokens).
    """
    if not files:
        return None
    toks = [tokenize(f) for f in files]
    shortest = min(len(t) for t in toks)
    prefix = []
    for i in range(min(shortest, max_tokens)):
        vals = {t[i] for t in toks}
        if len(vals) != 1:
            break
        prefix.append(next(iter(vals)))
    return "_".join(prefix) if prefix else None


def derive_label(pdb_args, files):
    """Prefer pattern-based label; fall back to filename-based."""
    label = _label_from_patterns(pdb_args)
    if label:
        return label
    label = _label_from_filenames(files)
    return label or "diffusion_analysis"


def describe_chunk(files):
    """Return a short filename-pattern summary for logging."""
    if not files:
        return "<empty>"
    if len(files) == 1:
        return os.path.basename(files[0])
    toks = [tokenize(f) for f in files]
    max_len = max(len(t) for t in toks)
    parts = []
    for i in range(max_len):
        vals = {t[i] for t in toks if i < len(t)}
        parts.append(next(iter(vals)) if len(vals) == 1 else "*")
    return "_".join(parts)


def run_chunks_local(chunks, args, passthrough, sc_dir, sc_stem):
    """Run each chunk sequentially via subprocess. Returns (chunk_scs, failed)."""
    chunk_scorefiles = []
    failed = []
    for i, chunk in enumerate(chunks):
        chunk_sc = os.path.join(sc_dir, f".{sc_stem}.chunk_{i:04d}.sc")
        chunk_scorefiles.append(chunk_sc)
        cmd = [
            sys.executable,
            args.child_script,
            "--pdb", *chunk,
            "--scorefile_out", chunk_sc,
            "--outdir", args.outdir,
            *passthrough,
        ]
        print(
            f"[orchestrator] running chunk {i+1}/{len(chunks)} "
            f"({len(chunk)} files) -> {chunk_sc}"
        )
        sys.stdout.flush()
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(
                f"[orchestrator] chunk {i+1} exited non-zero ({ret.returncode}); continuing",
                file=sys.stderr,
            )
            failed.append(i)
    return chunk_scorefiles, failed


SBATCH_TEMPLATE = """\
#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -p {queue}
#SBATCH -c {cores}
#SBATCH --mem={memory}
#SBATCH -t {time}
#SBATCH -o {logs_dir}/{job_name}_%a.stdout
#SBATCH -e {logs_dir}/{job_name}_%a.stderr
#SBATCH -a 1-{num_array_tasks}

PER_TASK={cmds_per_job}
START_NUM=$(( ($SLURM_ARRAY_TASK_ID - 1) * $PER_TASK + 1 ))
END_NUM=$(( $SLURM_ARRAY_TASK_ID * $PER_TASK ))
echo "[JOB $SLURM_ARRAY_TASK_ID] Runs $START_NUM to $END_NUM"
for (( run=$START_NUM; run<=END_NUM; run++ )); do
  CMD=$(sed -n "${{run}}p" {commands_file})
  if [ -z "$CMD" ]; then continue; fi
  echo "[START] Run $run | $(date '+%H:%M:%S')"
  echo "${{CMD}}" | bash
  EXIT_CODE=$?
  echo "[DONE]  Run $run | exit=$EXIT_CODE | $(date '+%H:%M:%S')"
done
echo "[JOB $SLURM_ARRAY_TASK_ID] All runs finished"
"""


def run_chunks_sbatch(chunks, args, passthrough, sc_dir, sc_stem, scorefile_out):
    """
    Submit chunks as a SLURM job array. Each array task invokes the apptainer
    image and the child script with one chunk's pdb list. Returns
    (chunk_scorefiles, failed). If --sbatch_no_wait, returns (None, []).
    """
    label = sc_stem
    workdir = args.sbatch_workdir or os.path.abspath(f"./.rfd3_orch_{label}")
    logs_dir = args.sbatch_logs_dir or os.path.abspath(f"./logs_{label}")
    job_name = args.sbatch_job_name or f"rfd3_{label}"
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # nproc: inject only if user didn't already pass --nproc
    user_set_nproc = any(a == "--nproc" for a in passthrough)

    # Write one pdb-list per chunk + build the commands file
    commands_path = os.path.join(workdir, "commands.txt")
    chunk_scorefiles = []
    with open(commands_path, "w") as cmdfh:
        for i, chunk in enumerate(chunks):
            list_path = os.path.join(workdir, f"chunk_{i:04d}.list")
            with open(list_path, "w") as lf:
                lf.write("\n".join(chunk) + "\n")
            chunk_sc = os.path.join(sc_dir, f".{sc_stem}.chunk_{i:04d}.sc")
            chunk_scorefiles.append(chunk_sc)

            # The child script has no --pdb_list flag. Build a shell line that
            # streams the chunk's file list into --pdb via xargs so we never
            # put thousands of paths on a single command line in commands.txt.
            child_cmd = [
                args.sbatch_apptainer,
                args.child_script,
                "--scorefile_out", chunk_sc,
                "--outdir", args.outdir,
            ]
            if not user_set_nproc:
                child_cmd += ["--nproc", str(args.sbatch_cores)]
            child_cmd += passthrough
            # Build a single shell line: xargs feeds the files from the list
            # to the child as --pdb arguments, avoiding any per-line argv limit.
            quoted_child = " ".join(shlex.quote(x) for x in child_cmd)
            line = (
                f"xargs -a {shlex.quote(list_path)} -d '\\n' "
                f"{quoted_child} --pdb"
            )
            cmdfh.write(line + "\n")

    num_cmds = len(chunks)
    per_task = max(1, args.sbatch_cmds_per_job)
    num_array_tasks = math.ceil(num_cmds / per_task)

    sbatch_path = os.path.join(workdir, "submit_array.sbatch")
    with open(sbatch_path, "w") as fh:
        fh.write(SBATCH_TEMPLATE.format(
            job_name=job_name,
            queue=args.sbatch_queue,
            cores=args.sbatch_cores,
            memory=args.sbatch_mem,
            time=args.sbatch_time,
            logs_dir=logs_dir,
            num_array_tasks=num_array_tasks,
            cmds_per_job=per_task,
            commands_file=commands_path,
        ))

    print(f"[orchestrator] sbatch workdir:    {workdir}")
    print(f"[orchestrator] sbatch logs_dir:   {logs_dir}")
    print(f"[orchestrator] sbatch commands:   {commands_path}  ({num_cmds} lines)")
    print(f"[orchestrator] sbatch script:     {sbatch_path}")
    print(f"[orchestrator] array tasks:       {num_array_tasks} x {per_task} cmd(s) each")

    if args.dry_run:
        print("[orchestrator] --dry_run: wrote sbatch files but not submitting")
        return None, []

    if shutil.which("sbatch") is None:
        print(
            "[orchestrator] ERROR: `sbatch` not found on PATH. If you ran the "
            "orchestrator inside the apptainer image, rerun it from the host "
            "(plain python3) — sbatch is typically not bundled in the SIF.",
            file=sys.stderr,
        )
        return [], list(range(num_cmds))

    if args.sbatch_no_wait:
        ret = subprocess.run(["sbatch", sbatch_path])
        if ret.returncode != 0:
            print(f"[orchestrator] sbatch submission failed (exit {ret.returncode})",
                  file=sys.stderr)
            return [], list(range(num_cmds))
        print("[orchestrator] submitted (no-wait). Skipping merge; rerun "
              f"with --scorefile_out {scorefile_out} and no --sbatch to merge "
              "after the array completes.")
        return None, []

    print("[orchestrator] submitting with `sbatch --wait` (blocks until array finishes)")
    sys.stdout.flush()
    ret = subprocess.run(["sbatch", "--wait", sbatch_path])
    if ret.returncode != 0:
        print(f"[orchestrator] sbatch --wait returned non-zero ({ret.returncode}); "
              "some array tasks likely failed. Inspecting per-chunk scorefiles.",
              file=sys.stderr)

    # Mark as failed any chunk whose scorefile didn't materialize
    failed = [i for i, sc in enumerate(chunk_scorefiles) if not os.path.exists(sc)]
    return chunk_scorefiles, failed


def main():
    parser = build_parser()
    args, passthrough = parser.parse_known_args()

    if args.help:
        print_help(parser, args.child_script)
        return 0

    raw_entries = list(args.pdb)
    if args.pdb_list:
        with open(args.pdb_list) as fh:
            raw_entries.extend(line.strip() for line in fh if line.strip())

    files = []
    glob_chars = set("*?[")
    for entry in raw_entries:
        if glob_chars & set(entry):
            matches = _glob.glob(entry)
            if not matches:
                print(f"[orchestrator] warning: pattern matched nothing: {entry}",
                      file=sys.stderr)
            else:
                print(f"[orchestrator] expanded pattern '{entry}' -> {len(matches)} files")
            files.extend(matches)
        else:
            files.append(entry)

    files = sorted(set(files))
    if not files:
        print("[orchestrator] no input files found via --pdb or --pdb_list",
              file=sys.stderr)
        return 2

    chunks = chunk_files(files, args.chunk_threshold)
    print(
        f"[orchestrator] {len(files)} files -> {len(chunks)} chunk(s) "
        f"(threshold={args.chunk_threshold}); sizes={[len(c) for c in chunks]}"
    )
    for i, c in enumerate(chunks):
        print(f"[orchestrator]   chunk {i+1}: n={len(c):>5}  pattern={describe_chunk(c)}")

    if args.scorefile_out is None:
        label = derive_label(args.pdb, files)
        scorefile_out = f"{args.scorefile_prefix}_{label}.sc"
        print(f"[orchestrator] auto scorefile name: {scorefile_out}")
    else:
        scorefile_out = args.scorefile_out

    if args.dry_run and not args.sbatch:
        return 0

    sc_out_abs = os.path.abspath(scorefile_out)
    sc_dir = os.path.dirname(sc_out_abs) or "."
    os.makedirs(sc_dir, exist_ok=True)
    sc_base = os.path.basename(sc_out_abs)
    sc_stem = sc_base[:-3] if sc_base.endswith(".sc") else sc_base

    if args.sbatch:
        chunk_scorefiles, failed = run_chunks_sbatch(
            chunks, args, passthrough, sc_dir, sc_stem, scorefile_out,
        )
        if chunk_scorefiles is None:
            # fire-and-forget path
            return 0
    else:
        chunk_scorefiles, failed = run_chunks_local(
            chunks, args, passthrough, sc_dir, sc_stem,
        )

    print(f"[orchestrator] merging {len(chunk_scorefiles)} scorefile(s) -> {sc_out_abs}")
    merge_scorefiles(chunk_scorefiles, sc_out_abs)

    if not args.keep_chunk_scorefiles:
        for sc in chunk_scorefiles:
            try:
                os.remove(sc)
            except OSError:
                pass

    if failed:
        print(f"[orchestrator] done with {len(failed)} failed chunk(s): {failed}",
              file=sys.stderr)
        return 1
    print("[orchestrator] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
