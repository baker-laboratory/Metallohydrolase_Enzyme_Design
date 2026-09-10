#!/usr/bin/env python3
"""
Cluster protein structure files with Foldseek.

Use cases
---------
- Cluster one directory of PDB/mmCIF files by structural similarity.
- Cluster multiple directories together without manually copying files first.
- Keep the original files untouched and only write a cluster-membership table.
- Optionally create cluster-labeled copies such as ``*_FS001.pdb``.
- Optionally rename files in place to append or replace a stable cluster label
  suffix.

Mandatory inputs
----------------
- ``inputs``: one or more directories and/or individual structure files.

Optional but commonly useful
----------------------------
- ``--output-prefix``: prefix used for retained output files. If omitted, the
  script derives a default prefix inside the first input directory (or the
  parent directory of the first input file).

Recommended / commonly changed options
--------------------------------------
- ``--strictness``: convenient preset controlling default coverage and TM-score
  thresholds. Default is ``strict``.
- ``--alignment-type``: Foldseek alignment engine. Default is ``1`` (TM-align).
- ``--recursive``: recurse into subdirectories of each requested input
  directory. Default is off, so only the exact directories you pass are
  scanned.
- ``-c`` / ``--coverage``: explicit coverage threshold override.
- ``--tmscore-threshold``: explicit TM-score threshold override.
- ``--cov-mode``: how Foldseek interprets coverage.
- ``--cluster-mode``: how Foldseek turns pairwise hits into clusters.
- ``--single-step-clustering`` or ``--no-single-step-clustering``.
- ``--clustered-files-mode``: ``none``, ``copy``, or ``rename-in-place``.
- ``--threads``: CPU count passed to Foldseek.

Less common but exposed options
-------------------------------
- ``-e`` / ``--evalue``
- ``--min-seq-id``
- ``--lddt-threshold``
- ``--exact-tmscore``
- ``--tmalign-hit-order``
- ``--tmalign-fast``
- ``--sensitivity``
- ``--max-seqs``
- ``--split-memory-limit``
- ``--cluster-reassign``
- ``--chain-name-mode``
- ``--model-name-mode``
- ``--foldseek-execution-mode``
- ``--foldseek-arg`` for raw pass-through flags not exposed directly

Accepted input types
--------------------
- Directory paths. By default, only files directly inside each requested
  directory are scanned. Add ``--recursive`` to descend into subdirectories.
- Individual structure files.
- Supported suffixes: ``.pdb``, ``.cif``, ``.mmcif`` and their ``.gz`` forms.

High-level logic
----------------
1. Collect supported structure files from every requested input path.
2. Deduplicate files by real path so nested or repeated inputs do not double
   count the same structure.
3. Stage symlinks into a single temporary directory because Foldseek
   ``easy-cluster`` accepts one directory input cleanly.
4. Build and run ``foldseek easy-cluster`` through Apptainer.
5. Parse Foldseek's representative/member table back onto the original source
   paths.
6. Assign stable labels such as ``FS001`` to each final cluster.
7. Optionally copy or rename files using those labels without modifying file
   contents. In rename-in-place mode, existing trailing labels such as
   ``_FS001`` are replaced instead of stacked.
8. Write a single main cluster-membership TSV plus one metadata JSON by
   default. Raw Foldseek outputs can still be kept with ``--keep-raw-foldseek-outputs``.

Main retained outputs
---------------------
- ``<output-prefix>_clusters.tsv``:
  one row per input structure, including cluster label, representative/member
  names, source paths, and optional labeled-file paths.
- ``<output-prefix>_run_metadata.json``:
  records settings, resolved defaults, timing, success status, and retained
  output paths.

Example commands
----------------
Minimal:

    python3 cluster_pdb_structures_with_foldseek.py \\
      /path/to/pdb_dir1 /path/to/pdb_dir2

Explicit output prefix:

    python3 cluster_pdb_structures_with_foldseek.py \\
      --output-prefix /path/to/run/foldseek_run \\
      /path/to/pdb_dir1 /path/to/pdb_dir2

Recursive directory scan:

    python3 cluster_pdb_structures_with_foldseek.py \\
      --output-prefix /path/to/run/foldseek_run \\
      --recursive \\
      /path/to/parent_dir

Validated strict TM-align style run:

    python3 cluster_pdb_structures_with_foldseek.py \\
      --output-prefix /path/to/run/foldseek_run \\
      --alignment-type 1 \\
      --tmscore-threshold 0.8 \\
      -c 0.8 \\
      --cov-mode 0 \\
      --cluster-mode 1 \\
      --single-step-clustering \\
      /path/to/pdb_dir1 /path/to/pdb_dir2

Create labeled copies:

    python3 cluster_pdb_structures_with_foldseek.py \\
      --output-prefix /path/to/run/foldseek_run \\
      --clustered-files-mode copy \\
      /path/to/pdb_dir1 /path/to/pdb_dir2

Rename files in place:

    python3 cluster_pdb_structures_with_foldseek.py \\
      --output-prefix /path/to/run/foldseek_run \\
      --clustered-files-mode rename-in-place \\
      /path/to/pdb_dir1 /path/to/pdb_dir2

Advanced Foldseek pass-through:

    python3 cluster_pdb_structures_with_foldseek.py \\
      --output-prefix /path/to/run/foldseek_run \\
      --foldseek-arg "--max-accept 123 --max-rejected 456" \\
      /path/to/pdb_dir1 /path/to/pdb_dir2

Outer-Apptainer execution of this script itself:

    apptainer exec --bind /net:/net --bind /home:/home \\
      $APPTAINER_SIF \\
      python3 cluster_pdb_structures_with_foldseek.py \\
      --foldseek-execution-mode direct \\
      --output-prefix /path/to/run/foldseek_run \\
      /path/to/pdb_dir1 /path/to/pdb_dir2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

def _default_foldseek_root():
    """Foldseek checkout: $FOLDSEEK_ROOT, else the copy bundled with this repository."""
    import os as _os
    from pathlib import Path as _Path
    env = _os.environ.get("FOLDSEEK_ROOT")
    if env:
        return env
    for _anc in _Path(__file__).resolve().parents:
        cand = _anc / "Software" / "foldseek"
        if cand.is_dir():
            return str(cand)
    return "foldseek"



SUPPORTED_SUFFIXES = (
    ".mmcif.gz",
    ".pdb.gz",
    ".cif.gz",
    ".mmcif",
    ".pdb",
    ".cif",
)

STRICTNESS_PRESETS = {
    "very-loose": {"coverage": 0.50, "tmscore_threshold": 0.50},
    "loose": {"coverage": 0.60, "tmscore_threshold": 0.60},
    "medium": {"coverage": 0.70, "tmscore_threshold": 0.70},
    "strict": {"coverage": 0.80, "tmscore_threshold": 0.80},
    "very-strict": {"coverage": 0.90, "tmscore_threshold": 0.90},
}

FOLDSEEK_PHASE_PATTERNS = (
    ("createdb ", "createdb"),
    ("kmermatcher ", "kmermatcher"),
    ("structurerescorediagonal ", "structurerescorediagonal"),
    ("tmalign ", "tmalign"),
    ("clust ", "clust"),
    ("mergeclusters ", "mergeclusters"),
    ("createtsv ", "createtsv"),
    ("result2repseq ", "result2repseq"),
    ("createseqfiledb ", "createseqfiledb"),
    ("result2flat ", "result2flat"),
)


class HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Combine readable multi-line help text with explicit default reporting."""


def parse_args() -> argparse.Namespace:
    """Define the CLI.

    The argument surface is split into:
    - minimal/common knobs that most users will touch frequently
    - advanced Foldseek options that are still exposed directly
    - a raw ``--foldseek-arg`` escape hatch for anything rarer
    """
    parser = argparse.ArgumentParser(
        formatter_class=HelpFormatter,
        description=(
            "Cluster PDB/mmCIF structures with Foldseek easy-cluster via "
            "Apptainer.\n\n"
            "Required inputs:\n"
            "  one or more positional input paths.\n\n"
            "Common workflow:\n"
            "  start with --strictness strict and only change\n"
            "  --coverage / --tmscore-threshold / --cov-mode / --cluster-mode\n"
            "  if you need different clustering behavior.\n\n"
            "Default retained outputs:\n"
            "  <output-prefix>_clusters.tsv\n"
            "  <output-prefix>_run_metadata.json\n"
            "  If --output-prefix is omitted, a default prefix is created\n"
            "  inside the first input directory."
        ),
        epilog=(
            "Coverage mode reference:\n"
            "  0 = require coverage threshold on both query and target\n"
            "  1 = require coverage threshold on target only\n"
            "  2 = require coverage threshold on query only\n"
            "  3 = target length must be at least x percent of query length\n"
            "  4 = query length must be at least x percent of target length\n"
            "  5 = shorter sequence must be at least x percent of the other\n\n"
            "Alignment type reference:\n"
            "  0 = 3Di local alignment\n"
            "  1 = TM-align global alignment\n"
            "  2 = 3Di + amino-acid local alignment\n\n"
            "Cluster mode reference:\n"
            "  0 = set-cover greedy clustering\n"
            "  1 = connected-component clustering\n"
            "  2 = greedy clustering by sequence length\n"
            "  3 = alternate greedy clustering by sequence length"
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help=(
            "One or more input directories and/or individual structure files. "
            "By default, directories are scanned non-recursively, so only "
            "files directly inside each requested directory are considered. "
            "Use --recursive to descend into subdirectories. Supported "
            "extensions are .pdb, .cif, .mmcif, and their .gz forms. "
            "Duplicate real paths are skipped automatically."
        ),
    )
    parser.set_defaults(recursive=False)
    parser.add_argument(
        "--recursive",
        dest="recursive",
        action="store_true",
        help=(
            "Scan each input directory recursively. If omitted, the script "
            "only considers files directly inside each requested directory."
        ),
    )
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help=(
            "Do not recurse into subdirectories when scanning input "
            "directories. This is the default."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        help=(
            "Prefix for retained outputs, for example "
            "/path/run/foldseek_clustered. The script writes "
            "<output-prefix>_clusters.tsv and "
            "<output-prefix>_run_metadata.json. If omitted, the script "
            "creates a default prefix inside the first input "
            "directory, or inside the parent directory of the first input "
            "file."
        ),
    )
    parser.add_argument(
        "--tmp-dir",
        help=(
            "Foldseek working directory. This contains the transient Foldseek "
            "database and workflow intermediates. It is removed after success "
            "unless --keep-tmp is used."
        ),
    )
    parser.add_argument(
        "--staging-dir",
        help=(
            "Directory used for staged symlink inputs. The script symlinks all "
            "collected structures into one directory because Foldseek "
            "easy-cluster handles a single directory input cleanly. Removed "
            "after success unless --keep-staging is used."
        ),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=os.cpu_count() or 1,
        help=(
            "Threads passed to Foldseek. Increase for more CPU parallelism; "
            "lower it if you want to reduce resource use."
        ),
    )
    parser.add_argument(
        "--apptainer-image",
        default=os.environ.get("ZINC_HYDRO_SIF"),
        help=(
            "Apptainer image used to run Foldseek. The host Foldseek checkout "
            "is bind-mounted into this container and executed there."
        ),
    )
    parser.add_argument(
        "--foldseek-root",
        default=_default_foldseek_root(),
        help=(
            "Host path to the Foldseek checkout containing build/src/foldseek. "
            "The script bind-mounts this location as /opt/foldseek inside the "
            "Apptainer container."
        ),
    )
    parser.add_argument(
        "--foldseek-execution-mode",
        choices=("auto", "apptainer", "direct"),
        default="auto",
        help=(
            "How Foldseek itself is launched. "
            "auto = use apptainer if the apptainer executable is available, "
            "otherwise run the Foldseek binary directly. "
            "apptainer = always launch Foldseek through apptainer exec. "
            "direct = call <foldseek-root>/build/src/foldseek directly from "
            "the current environment. Use direct when this Python script is "
            "already being run inside an outer Apptainer container."
        ),
    )
    parser.add_argument(
        "--alignment-type",
        type=int,
        choices=(0, 1, 2),
        default=1,
        help=(
            "Foldseek alignment algorithm. "
            "0 = 3Di local alignment, "
            "1 = TM-align global alignment, "
            "2 = 3Di + amino-acid local alignment. "
            "TM-align (1) is the default because it is the most interpretable "
            "strict structural similarity mode for these workflows."
        ),
    )
    parser.add_argument(
        "--strictness",
        choices=tuple(STRICTNESS_PRESETS),
        default="strict",
        help=(
            "User-facing clustering preset. This only sets the default values "
            "for coverage and TM-score threshold. "
            "very-loose = 0.50/0.50, "
            "loose = 0.60/0.60, "
            "medium = 0.70/0.70, "
            "strict = 0.80/0.80, "
            "very-strict = 0.90/0.90. "
            "Explicit --coverage and --tmscore-threshold values override this."
        ),
    )
    parser.add_argument(
        "-c",
        "--coverage",
        type=float,
        help=(
            "Coverage threshold passed to Foldseek. Higher values are stricter "
            "and push the clustering toward more global similarity. The exact "
            "meaning depends on --cov-mode. If omitted, the value comes from "
            "--strictness."
        ),
    )
    parser.add_argument(
        "--cov-mode",
        type=int,
        choices=(0, 1, 2, 3, 4, 5),
        default=0,
        help=(
            "How Foldseek interprets -c/--coverage. "
            "0 = both query and target must satisfy coverage. "
            "1 = target only. "
            "2 = query only. "
            "3 = target length must be at least x percent of query length. "
            "4 = query length must be at least x percent of target length. "
            "5 = shorter sequence must be at least x percent of the other. "
            "Mode 0 is the symmetric default."
        ),
    )
    parser.add_argument(
        "--tmscore-threshold",
        type=float,
        help=(
            "Minimum TM-score required when --alignment-type 1 is used. "
            "Higher values are stricter. If omitted, the default comes from "
            "--strictness. Ignored when --alignment-type is not 1."
        ),
    )
    parser.add_argument(
        "--tmscore-threshold-mode",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help=(
            "How Foldseek normalizes the TM-score threshold. "
            "0 = normalize by alignment length, "
            "1 = normalize by representative/query length, "
            "2 = normalize by member/target length."
        ),
    )
    parser.add_argument(
        "-e",
        "--evalue",
        type=float,
        help=(
            "Optional Foldseek E-value threshold. Smaller values are stricter. "
            "Leave unset to accept Foldseek's workflow default."
        ),
    )
    parser.add_argument(
        "--min-seq-id",
        type=float,
        help=(
            "Optional Foldseek minimum sequence identity threshold. This is "
            "usually not the main knob for structural clustering, but it can "
            "be useful when you want to prevent low-sequence-identity merges."
        ),
    )
    parser.add_argument(
        "--lddt-threshold",
        type=float,
        help=(
            "Optional Foldseek LDDT threshold. Higher values are stricter. "
            "Leave unset unless you specifically want an LDDT filter."
        ),
    )
    parser.add_argument(
        "--exact-tmscore",
        action="store_true",
        help=(
            "Ask Foldseek to compute exact TM-score instead of the faster "
            "approximation. This can be slower."
        ),
    )
    parser.add_argument(
        "--tmalign-hit-order",
        type=int,
        choices=(0, 1, 2, 3, 4),
        help=(
            "TM-align hit ordering mode. "
            "0 = average(qTM, tTM), "
            "1 = qTM only, "
            "2 = tTM only, "
            "3 = min(qTM, tTM), "
            "4 = max(qTM, tTM)."
        ),
    )
    parser.add_argument(
        "--tmalign-fast",
        type=int,
        choices=(0, 1),
        help=(
            "TM-align fast-mode override. "
            "1 keeps Foldseek's fast search path, 0 disables it."
        ),
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        help=(
            "Foldseek sensitivity (-s). Higher values usually increase search "
            "sensitivity at the cost of runtime. Leave unset to let Foldseek "
            "choose automatically."
        ),
    )
    parser.add_argument(
        "--max-seqs",
        type=int,
        help=(
            "Optional Foldseek max-seqs override controlling how many results "
            "per query survive early filtering. Larger values can increase "
            "sensitivity but also cost more time and memory."
        ),
    )
    parser.add_argument(
        "--split-memory-limit",
        help=(
            "Optional Foldseek split memory limit such as 8G. Use this if you "
            "need to cap Foldseek's memory use and allow splitting."
        ),
    )
    parser.add_argument(
        "--cluster-mode",
        type=int,
        choices=(0, 1, 2, 3),
        default=1,
        help=(
            "How Foldseek converts pairwise structure hits into clusters. "
            "0 = set-cover greedy clustering. "
            "1 = connected-component clustering. "
            "2 = greedy clustering by sequence length. "
            "3 = alternate greedy clustering by sequence length. "
            "Default 1 matched the validated RF backbone grouping in this use "
            "case."
        ),
    )
    parser.set_defaults(single_step_clustering=True)
    parser.add_argument(
        "--single-step-clustering",
        dest="single_step_clustering",
        action="store_true",
        help=(
            "Use Foldseek single-step clustering. This is enabled by default "
            "in this script because it paired well with connected-component "
            "clustering in validation."
        ),
    )
    parser.add_argument(
        "--no-single-step-clustering",
        dest="single_step_clustering",
        action="store_false",
        help="Disable Foldseek single-step clustering and allow cascaded behavior.",
    )
    parser.add_argument(
        "--cluster-reassign",
        action="store_true",
        help=(
            "Enable Foldseek cluster reassignment. This can correct some "
            "cascaded-clustering assignment errors, but it is not usually "
            "needed for the default single-step workflow."
        ),
    )
    parser.add_argument(
        "--chain-name-mode",
        type=int,
        choices=(0, 1),
        default=1,
        help=(
            "Foldseek chain-name mode. "
            "0 = auto, 1 = always append chain to the parsed structure name."
        ),
    )
    parser.add_argument(
        "--model-name-mode",
        type=int,
        choices=(0, 1),
        default=0,
        help=(
            "Foldseek model-name mode. "
            "0 = auto, 1 = always append model name to the parsed structure name."
        ),
    )
    parser.add_argument(
        "--clustered-files-mode",
        choices=("none", "copy", "rename-in-place"),
        default="none",
        help=(
            "Optional file-labeling mode after clustering. "
            "none = only write the cluster table. "
            "copy = create cluster-labeled copies in a new directory. "
            "rename-in-place = rename the original files to append the "
            "cluster label. If a filename already ends with one or more "
            "trailing labels matching --cluster-label-prefix plus digits, "
            "those old labels are replaced instead of stacked. No mode ever "
            "edits file contents."
        ),
    )
    parser.add_argument(
        "--clustered-files-dir",
        help=(
            "Destination directory for --clustered-files-mode copy. "
            "Ignored unless --clustered-files-mode copy is used."
        ),
    )
    parser.add_argument(
        "--cluster-label-prefix",
        default="FS",
        help=(
            "Prefix used for cluster labels in the main table and optional "
            "copied/renamed filenames, for example FS001."
        ),
    )
    parser.add_argument(
        "--status-interval-seconds",
        type=int,
        default=60,
        help=(
            "Emit lightweight runtime status messages every N seconds while "
            "Foldseek is running. These messages report elapsed time and the "
            "current inferred Foldseek phase. Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--extra-bind",
        action="append",
        default=[],
        help=(
            "Additional Apptainer bind mount(s), repeatable. Use this only if "
            "Foldseek needs to see extra host paths not already covered by the "
            "input, staging, tmp, output, and Foldseek root mounts."
        ),
    )
    parser.add_argument(
        "--foldseek-arg",
        action="append",
        default=[],
        help=(
            "Extra raw argument string appended to the Foldseek command after "
            "the script's built-in defaults. Repeatable. This is the escape "
            "hatch for advanced Foldseek flags not exposed directly. Example: "
            "--foldseek-arg \"--max-accept 123 --max-rejected 456\"."
        ),
    )
    parser.add_argument(
        "--keep-raw-foldseek-outputs",
        action="store_true",
        help=(
            "Retain Foldseek's raw _cluster.tsv, _rep_seq.fasta, and "
            "_all_seqs.fasta outputs in addition to the main combined table. "
            "These are usually unnecessary unless you want the raw Foldseek "
            "artifacts for debugging or downstream reuse."
        ),
    )
    parser.add_argument(
        "--keep-staging-map",
        action="store_true",
        help=(
            "Retain the staged-input mapping TSV used internally to map staged "
            "symlink names back to the original source paths."
        ),
    )
    parser.add_argument(
        "--keep-staging",
        action="store_true",
        help="Retain the staged symlink directory after a successful run.",
    )
    parser.add_argument(
        "--keep-tmp",
        action="store_true",
        help=(
            "Retain Foldseek temporary files and databases instead of cleaning "
            "them up after success."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite known outputs and recreate staging/tmp/cluster-copy "
            "directories if they already exist."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Prepare inputs and print the final Foldseek command without "
            "running Foldseek. Useful for sanity-checking the resolved CLI."
        ),
    )
    return parser.parse_args()


def abs_path(raw_path: str) -> Path:
    """Resolve ``~`` and relative paths into absolute ``Path`` objects."""
    return Path(os.path.abspath(os.path.expanduser(raw_path)))


def derive_default_output_prefix(input_paths: list[Path]) -> Path:
    """Choose a stable default output prefix beside the first input."""
    first_input = input_paths[0]
    if first_input.is_dir():
        output_dir = first_input
        base_name = first_input.name or "input_dir"
    else:
        output_dir = first_input.parent
        base_name, _suffix = split_structure_suffix(first_input.name)
    return output_dir / f"foldseek_cluster_{base_name}"


def eprint(message: str) -> None:
    """Write a flushed progress message to stderr."""
    print(message, file=sys.stderr, flush=True)


def iso_now() -> str:
    """Return a timestamp for metadata records."""
    return datetime.now().isoformat(timespec="seconds")


def format_seconds(seconds: float) -> str:
    """Render a floating-point duration as HH:MM:SS."""
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def is_supported_structure(path: Path) -> bool:
    """Return True if the file suffix looks like a supported structure file."""
    lower_name = path.name.lower()
    return any(lower_name.endswith(suffix) for suffix in SUPPORTED_SUFFIXES)


def split_structure_suffix(filename: str) -> tuple[str, str]:
    """Split a structure filename into stem and full recognized suffix."""
    lower_name = filename.lower()
    for suffix in SUPPORTED_SUFFIXES:
        if lower_name.endswith(suffix):
            return filename[: -len(suffix)], filename[-len(suffix) :]
    return Path(filename).stem, Path(filename).suffix


def output_path(output_prefix: Path, suffix: str) -> Path:
    """Build a path by appending a fixed suffix to the output prefix."""
    return Path(f"{output_prefix}{suffix}")


def main_cluster_table_path(output_prefix: Path) -> Path:
    return output_path(output_prefix, "_clusters.tsv")


def metadata_path(output_prefix: Path) -> Path:
    return output_path(output_prefix, "_run_metadata.json")


def staging_map_path(output_prefix: Path) -> Path:
    return output_path(output_prefix, "_staged_input_map.tsv")


def raw_foldseek_paths(output_prefix: Path) -> list[Path]:
    return [
        output_path(output_prefix, "_cluster.tsv"),
        output_path(output_prefix, "_rep_seq.fasta"),
        output_path(output_prefix, "_all_seqs.fasta"),
    ]


def legacy_output_paths(output_prefix: Path) -> list[Path]:
    return [
        output_path(output_prefix, "_cluster_with_paths.tsv"),
        output_path(output_prefix, "_cluster_sizes.tsv"),
    ]


def remove_existing_path(path: Path) -> None:
    """Remove an existing file or directory in a force-overwrite-safe way."""
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def collect_structure_files(
    input_paths: list[Path],
    recursive: bool,
) -> tuple[list[Path], list[str]]:
    """Collect supported files from the requested inputs.

    Directories are scanned non-recursively by default so users have explicit
    control over which directories are included. Real paths are deduplicated so
    overlapping directories or repeated explicit file paths do not double count
    work. ``recursive=True`` opts into descending through subdirectories.
    """
    collected: list[Path] = []
    seen_realpaths: set[str] = set()
    duplicate_paths: list[str] = []

    for input_path in input_paths:
        eprint(f"Scanning input: {input_path}")
        if input_path.is_dir():
            if recursive:
                iterator = sorted(
                    path for path in input_path.rglob("*") if path.is_file()
                )
            else:
                iterator = sorted(
                    path for path in input_path.iterdir() if path.is_file()
                )
        elif input_path.is_file():
            iterator = [input_path]
        else:
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

        for path in iterator:
            if not is_supported_structure(path):
                continue
            real_path = str(path.resolve())
            if real_path in seen_realpaths:
                duplicate_paths.append(real_path)
                continue
            seen_realpaths.add(real_path)
            collected.append(path.resolve())

    collected.sort()
    return collected, duplicate_paths


def ensure_parent_dir(path: Path) -> None:
    """Create the parent directory for an output path if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)


def prepare_output_paths(
    output_prefix: Path,
    staging_dir: Path,
    tmp_dir: Path,
    clustered_files_dir: Path | None,
    force: bool,
) -> None:
    """Validate and clear output locations before the run begins."""
    ensure_parent_dir(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    for path in (
        [
            main_cluster_table_path(output_prefix),
            metadata_path(output_prefix),
            staging_map_path(output_prefix),
        ]
        + raw_foldseek_paths(output_prefix)
        + legacy_output_paths(output_prefix)
    ):
        if path.exists():
            if force:
                remove_existing_path(path)
            else:
                raise FileExistsError(
                    f"Output already exists: {path}. Use --force to overwrite."
                )

    for work_dir in (staging_dir, tmp_dir, clustered_files_dir):
        if work_dir is None:
            continue
        if work_dir.exists():
            if force:
                remove_existing_path(work_dir)
            else:
                raise FileExistsError(
                    f"Working directory already exists: {work_dir}. "
                    "Use --force to recreate it."
                )


def make_unique_stem(stem: str, used_stems: set[str]) -> str:
    """Return a unique staged stem while preserving the original base name."""
    candidate = stem
    counter = 2
    while candidate in used_stems:
        candidate = f"{stem}__dup{counter}"
        counter += 1
    used_stems.add(candidate)
    return candidate


def stage_structure_files(
    structure_files: list[Path],
    staging_dir: Path,
    staging_map_output_path: Path | None,
) -> dict[str, dict[str, str]]:
    """Create one staged symlink directory for Foldseek input.

    Foldseek ``easy-cluster`` accepts one directory cleanly, so this step
    provides a single merged input view while still preserving a mapping back
    to the original source paths.
    """
    staging_dir.mkdir(parents=True, exist_ok=False)
    mapping: dict[str, dict[str, str]] = {}
    used_stems: set[str] = set()

    mapping_handle = None
    mapping_writer = None
    if staging_map_output_path is not None:
        mapping_handle = staging_map_output_path.open("w", newline="")
        mapping_writer = csv.writer(mapping_handle, delimiter="\t")
        mapping_writer.writerow(["staged_stem", "staged_filename", "source_path"])

    try:
        for index, source_path in enumerate(structure_files, start=1):
            source_stem, source_suffix = split_structure_suffix(source_path.name)
            staged_stem = make_unique_stem(source_stem, used_stems)
            staged_filename = f"{staged_stem}{source_suffix}"
            staged_path = staging_dir / staged_filename
            os.symlink(source_path, staged_path)
            mapping[staged_stem] = {
                "staged_filename": staged_filename,
                "source_path": str(source_path),
            }
            if mapping_writer is not None:
                mapping_writer.writerow([staged_stem, staged_filename, str(source_path)])
            if index == len(structure_files) or index % 1000 == 0:
                eprint(
                    f"Staged {index}/{len(structure_files)} structure files..."
                )
    finally:
        if mapping_handle is not None:
            mapping_handle.close()

    return mapping


def build_apptainer_binds(
    user_inputs: list[Path],
    foldseek_root: Path,
    staging_dir: Path,
    tmp_dir: Path,
    output_prefix: Path,
    clustered_files_dir: Path | None,
    extra_binds: list[str],
) -> list[str]:
    """Build the Apptainer bind list needed for the Foldseek run."""
    binds: set[str] = set()

    for input_path in user_inputs:
        binds.add(str(input_path if input_path.is_dir() else input_path.parent))
    binds.add(str(staging_dir.parent))
    binds.add(str(tmp_dir.parent))
    binds.add(str(output_prefix.parent))
    if clustered_files_dir is not None:
        binds.add(str(clustered_files_dir.parent))
    binds.add(f"{foldseek_root}:/opt/foldseek")

    for bind in extra_binds:
        binds.add(bind)

    return sorted(binds)


def resolve_effective_parameters(args: argparse.Namespace) -> dict[str, object]:
    """Resolve convenience presets into concrete Foldseek thresholds."""
    preset = STRICTNESS_PRESETS[args.strictness]
    coverage = args.coverage if args.coverage is not None else preset["coverage"]
    tmscore_threshold = None
    if args.alignment_type == 1:
        default_tm = preset["tmscore_threshold"]
        tmscore_threshold = (
            args.tmscore_threshold
            if args.tmscore_threshold is not None
            else default_tm
        )
    elif args.tmscore_threshold is not None:
        eprint(
            "Ignoring --tmscore-threshold because --alignment-type is not 1 "
            "(TM-align)."
        )

    return {
        "strictness": args.strictness,
        "coverage": coverage,
        "tmscore_threshold": tmscore_threshold,
        "coverage_overridden": args.coverage is not None,
        "tmscore_threshold_overridden": args.tmscore_threshold is not None,
    }


def resolve_foldseek_execution_mode(args: argparse.Namespace) -> str:
    """Choose whether Foldseek should run via Apptainer or directly.

    This supports both of the intended usage styles:
    - Run this Python script on the host and let it launch Foldseek through
      Apptainer internally.
    - Run this Python script inside an outer Apptainer container and let it
      call the Foldseek binary directly, avoiding nested Apptainer execution.
    """
    apptainer_available = shutil.which("apptainer") is not None

    if args.foldseek_execution_mode == "auto":
        return "apptainer" if apptainer_available else "direct"

    if args.foldseek_execution_mode == "apptainer" and not apptainer_available:
        raise RuntimeError(
            "--foldseek-execution-mode apptainer was requested, but the "
            "'apptainer' executable is not available in the current "
            "environment."
        )

    return args.foldseek_execution_mode


def build_foldseek_command(
    args: argparse.Namespace,
    effective_parameters: dict[str, object],
    execution_mode: str,
    foldseek_bin: Path,
    staging_dir: Path,
    tmp_dir: Path,
    output_prefix: Path,
    binds: list[str],
) -> list[str]:
    """Translate script-level settings into the final Foldseek command."""
    if execution_mode == "apptainer":
        command = ["apptainer", "exec"]
        for bind in binds:
            command.extend(["--bind", bind])
        command.extend(
            [
                str(abs_path(args.apptainer_image)),
                "/opt/foldseek/build/src/foldseek",
                "easy-cluster",
            ]
        )
    else:
        command = [str(foldseek_bin), "easy-cluster"]

    command.extend(
        [
            str(staging_dir),
            str(output_prefix),
            str(tmp_dir),
            "--alignment-type",
            str(args.alignment_type),
            "-c",
            str(effective_parameters["coverage"]),
            "--cov-mode",
            str(args.cov_mode),
            "--cluster-mode",
            str(args.cluster_mode),
            "--threads",
            str(args.threads),
            "--chain-name-mode",
            str(args.chain_name_mode),
            "--model-name-mode",
            str(args.model_name_mode),
            "--file-include",
            ".*",
            "--file-exclude",
            "^$",
            "--remove-tmp-files",
            "0" if args.keep_tmp else "1",
        ]
    )

    if (
        args.alignment_type == 1
        and effective_parameters["tmscore_threshold"] is not None
    ):
        command.extend(
            ["--tmscore-threshold", str(effective_parameters["tmscore_threshold"])]
        )
        command.extend(
            ["--tmscore-threshold-mode", str(args.tmscore_threshold_mode)]
        )
    if args.evalue is not None:
        command.extend(["-e", str(args.evalue)])
    if args.min_seq_id is not None:
        command.extend(["--min-seq-id", str(args.min_seq_id)])
    if args.lddt_threshold is not None:
        command.extend(["--lddt-threshold", str(args.lddt_threshold)])
    if args.exact_tmscore:
        command.extend(["--exact-tmscore", "1"])
    if args.tmalign_hit_order is not None:
        command.extend(["--tmalign-hit-order", str(args.tmalign_hit_order)])
    if args.tmalign_fast is not None:
        command.extend(["--tmalign-fast", str(args.tmalign_fast)])
    if args.sensitivity is not None:
        command.extend(["-s", str(args.sensitivity)])
    if args.max_seqs is not None:
        command.extend(["--max-seqs", str(args.max_seqs)])
    if args.split_memory_limit is not None:
        command.extend(["--split-memory-limit", str(args.split_memory_limit)])
    if args.single_step_clustering:
        command.extend(["--single-step-clustering", "1"])
    if args.cluster_reassign:
        command.extend(["--cluster-reassign", "1"])
    for raw_extra_arg in args.foldseek_arg:
        command.extend(shlex.split(raw_extra_arg))

    return command


def detect_foldseek_phase(line: str, current_phase: str) -> str:
    """Infer a coarse Foldseek phase from a log line for status reporting."""
    for needle, phase_name in FOLDSEEK_PHASE_PATTERNS:
        if line.startswith(needle):
            return phase_name
    return current_phase


def run_foldseek(
    command: list[str],
    total_structures: int,
    status_interval_seconds: int,
) -> dict[str, object]:
    """Run Foldseek while streaming logs and printing periodic status lines."""
    start_time = time.monotonic()
    phase_state = {"phase": "starting"}
    stop_event = threading.Event()

    def status_worker() -> None:
        while not stop_event.wait(status_interval_seconds):
            elapsed_seconds = time.monotonic() - start_time
            eprint(
                "[status] "
                f"structures={total_structures} "
                f"phase={phase_state['phase']} "
                f"elapsed={format_seconds(elapsed_seconds)}"
            )

    status_thread = None
    if status_interval_seconds > 0:
        status_thread = threading.Thread(target=status_worker, daemon=True)
        status_thread.start()

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )

    if process.stdout is None:
        raise RuntimeError("Failed to capture Foldseek output stream.")

    try:
        for raw_line in process.stdout:
            line = raw_line.rstrip("\n")
            phase_state["phase"] = detect_foldseek_phase(line, phase_state["phase"])
            print(line, file=sys.stderr, flush=True)
    finally:
        process.stdout.close()
        return_code = process.wait()
        stop_event.set()
        if status_thread is not None:
            status_thread.join()

    elapsed_seconds = time.monotonic() - start_time
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)

    return {
        "elapsed_seconds": elapsed_seconds,
        "last_phase": phase_state["phase"],
    }


def resolve_cluster_name(
    foldseek_name: str,
    staged_mapping: dict[str, dict[str, str]],
) -> tuple[str | None, str]:
    """Map a Foldseek-emitted name back to the staged input stem.

    Foldseek may append chain/model suffixes such as ``_A``. This helper strips
    trailing underscore-delimited suffix fragments until the staged name is
    recovered, then returns both the staged stem and the stripped suffix.
    """
    candidate = foldseek_name
    suffix_tokens: list[str] = []

    while True:
        if candidate in staged_mapping:
            suffix = "_".join(reversed(suffix_tokens))
            return candidate, suffix
        if "_" not in candidate:
            return None, ""
        candidate, tail = candidate.rsplit("_", 1)
        suffix_tokens.append(tail)


def load_clusters(
    output_prefix: Path,
    staged_mapping: dict[str, dict[str, str]],
) -> tuple[list[dict[str, object]], int]:
    """Load Foldseek's representative/member table and restore source paths."""
    raw_cluster_path = output_path(output_prefix, "_cluster.tsv")
    clusters_by_representative: dict[str, dict[str, object]] = {}
    unresolved_rows = 0

    with raw_cluster_path.open() as source_handle:
        reader = csv.reader(source_handle, delimiter="\t")
        for row in reader:
            if len(row) != 2:
                continue

            representative_name, member_name = row
            rep_stem, _rep_suffix = resolve_cluster_name(
                representative_name, staged_mapping
            )
            mem_stem, _mem_suffix = resolve_cluster_name(member_name, staged_mapping)

            representative_source_path = staged_mapping.get(rep_stem, {}).get(
                "source_path", ""
            )
            member_source_path = staged_mapping.get(mem_stem, {}).get("source_path", "")
            if rep_stem is None or mem_stem is None:
                unresolved_rows += 1

            cluster = clusters_by_representative.setdefault(
                representative_name,
                {
                    "representative_name": representative_name,
                    "representative_source_path": representative_source_path,
                    "members": [],
                },
            )

            cluster["members"].append(
                {
                    "member_name": member_name,
                    "member_source_path": member_source_path,
                }
            )

    clusters = sorted(
        clusters_by_representative.values(),
        key=lambda cluster: (
            str(cluster["representative_source_path"] or ""),
            str(cluster["representative_name"]),
        ),
    )

    width = max(3, len(str(len(clusters))))
    for cluster_index, cluster in enumerate(clusters, start=1):
        cluster["cluster_id"] = cluster_index
        cluster["cluster_size"] = len(cluster["members"])
        cluster["members"].sort(
            key=lambda member: (
                0
                if member["member_source_path"] == cluster["representative_source_path"]
                else 1,
                str(member["member_source_path"]),
                str(member["member_name"]),
            )
        )
        cluster["cluster_label_width"] = width

    return clusters, unresolved_rows


def assign_cluster_labels(
    clusters: list[dict[str, object]],
    cluster_label_prefix: str,
) -> None:
    """Assign compact stable labels like FS001 to every final cluster."""
    width = max(3, len(str(len(clusters))))
    for cluster_index, cluster in enumerate(clusters, start=1):
        cluster["cluster_id"] = cluster_index
        cluster["cluster_label"] = f"{cluster_label_prefix}{cluster_index:0{width}d}"
        cluster["cluster_size"] = len(cluster["members"])
        for member_index, member in enumerate(cluster["members"], start=1):
            member["member_index"] = member_index
            member["is_representative"] = (
                member["member_source_path"] == cluster["representative_source_path"]
                and member["member_name"] == cluster["representative_name"]
            )


def make_unique_copy_target(
    target_path: Path,
    reserved_targets: set[str],
) -> Path:
    """Choose a non-colliding copy target if a labeled filename already exists."""
    target_stem, target_suffix = split_structure_suffix(target_path.name)
    candidate = target_path
    counter = 2
    while str(candidate) in reserved_targets or candidate.exists():
        candidate = target_path.with_name(f"{target_stem}__dup{counter}{target_suffix}")
        counter += 1
    reserved_targets.add(str(candidate))
    return candidate


def strip_existing_cluster_label_suffix(
    source_stem: str,
    cluster_label_prefix: str,
) -> str:
    """Remove one or more trailing labels like ``_FS001`` from a file stem."""
    if not cluster_label_prefix:
        return source_stem
    return re.sub(
        rf"(?:_{re.escape(cluster_label_prefix)}\d+)+$",
        "",
        source_stem,
    )


def plan_file_labeling(
    clusters: list[dict[str, object]],
    mode: str,
    clustered_files_dir: Path | None,
    cluster_label_prefix: str,
) -> dict[str, dict[str, str]]:
    """Plan optional copy or rename operations after clustering is complete."""
    if mode == "none":
        return {}

    if mode == "copy":
        if clustered_files_dir is None:
            raise ValueError("clustered_files_dir is required for copy mode.")
        clustered_files_dir.mkdir(parents=True, exist_ok=False)

    plan: dict[str, dict[str, str]] = {}
    reserved_targets: set[str] = set()

    for cluster in clusters:
        cluster_label = str(cluster["cluster_label"])
        for member in cluster["members"]:
            source_path = Path(str(member["member_source_path"]))
            source_stem, source_suffix = split_structure_suffix(source_path.name)

            if mode == "copy":
                labeled_filename = f"{source_stem}_{cluster_label}{source_suffix}"
                target_path = make_unique_copy_target(
                    clustered_files_dir / labeled_filename, reserved_targets
                )
                action = "copied"
            else:
                normalized_stem = strip_existing_cluster_label_suffix(
                    source_stem,
                    cluster_label_prefix,
                )
                labeled_filename = f"{normalized_stem}_{cluster_label}{source_suffix}"
                target_path = source_path.with_name(labeled_filename)
                action = "renamed-in-place"
                if str(target_path) in reserved_targets:
                    raise RuntimeError(
                        f"Two files would be renamed to the same target path: "
                        f"{target_path}"
                    )
                reserved_targets.add(str(target_path))
                if target_path.exists() and target_path != source_path:
                    raise FileExistsError(
                        "Refusing to rename in place because the destination "
                        f"already exists: {target_path}"
                    )

            plan[str(source_path)] = {
                "target_path": str(target_path),
                "action": action,
            }

    return plan


def execute_file_labeling(file_labeling_plan: dict[str, dict[str, str]]) -> None:
    """Apply the planned copy or rename operations."""
    if not file_labeling_plan:
        return

    total_files = len(file_labeling_plan)
    action_name = next(iter(file_labeling_plan.values()))["action"]
    eprint(f"Applying file-labeling mode: {action_name} on {total_files} files...")

    for index, source_path in enumerate(sorted(file_labeling_plan), start=1):
        target_path = file_labeling_plan[source_path]["target_path"]
        action = file_labeling_plan[source_path]["action"]
        if action == "copied":
            shutil.copy2(source_path, target_path)
        elif action == "renamed-in-place":
            Path(source_path).rename(target_path)
        else:
            raise ValueError(f"Unsupported file action: {action}")

        if index == total_files or index % 1000 == 0:
            eprint(f"Labeled {index}/{total_files} files...")


def write_main_cluster_table(
    output_prefix: Path,
    clusters: list[dict[str, object]],
    file_labeling_plan: dict[str, dict[str, str]],
) -> int:
    """Write the single main cluster-membership TSV retained by default."""
    main_table_path = main_cluster_table_path(output_prefix)
    row_count = 0

    with main_table_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "cluster_id",
                "cluster_label",
                "cluster_size",
                "member_index",
                "is_representative",
                "representative_name",
                "representative_source_path",
                "representative_labeled_path",
                "member_name",
                "member_source_path",
                "member_labeled_path",
                "file_action",
            ]
        )

        for cluster in clusters:
            representative_source_path = str(cluster["representative_source_path"])
            representative_labeled_path = file_labeling_plan.get(
                representative_source_path, {}
            ).get("target_path", "")
            for member in cluster["members"]:
                member_source_path = str(member["member_source_path"])
                member_labeled_path = file_labeling_plan.get(
                    member_source_path, {}
                ).get("target_path", "")
                file_action = file_labeling_plan.get(member_source_path, {}).get(
                    "action", "none"
                )

                writer.writerow(
                    [
                        cluster["cluster_id"],
                        cluster["cluster_label"],
                        cluster["cluster_size"],
                        member["member_index"],
                        1 if member["is_representative"] else 0,
                        cluster["representative_name"],
                        representative_source_path,
                        representative_labeled_path,
                        member["member_name"],
                        member_source_path,
                        member_labeled_path,
                        file_action,
                    ]
                )
                row_count += 1

    return row_count


def cleanup_outputs(
    output_prefix: Path,
    keep_raw_foldseek_outputs: bool,
    keep_staging_map: bool,
) -> None:
    """Remove intermediate outputs unless the user asked to retain them."""
    if not keep_raw_foldseek_outputs:
        for path in raw_foldseek_paths(output_prefix):
            if path.exists():
                remove_existing_path(path)

    if not keep_staging_map:
        path = staging_map_path(output_prefix)
        if path.exists():
            remove_existing_path(path)


def retained_output_paths(
    output_prefix: Path,
    clustered_files_dir: Path | None,
    keep_raw_foldseek_outputs: bool,
    keep_staging_map: bool,
) -> list[str]:
    """List the output paths that should remain after cleanup."""
    retained = [
        str(main_cluster_table_path(output_prefix)),
        str(metadata_path(output_prefix)),
    ]
    if clustered_files_dir is not None and clustered_files_dir.exists():
        retained.append(str(clustered_files_dir))
    if keep_raw_foldseek_outputs:
        retained.extend(str(path) for path in raw_foldseek_paths(output_prefix))
    if keep_staging_map:
        path = staging_map_path(output_prefix)
        if path.exists():
            retained.append(str(path))
    return retained


def write_metadata(
    output_prefix: Path,
    payload: dict[str, object],
) -> None:
    """Write a JSON record of parameters, timing, and retained outputs."""
    with metadata_path(output_prefix).open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def main() -> int:
    """Execute the full workflow.

    Stages:
    1. Parse CLI and normalize paths.
    2. Collect and stage input structures.
    3. Resolve script defaults into explicit Foldseek parameters.
    4. Resolve whether Foldseek itself should run via Apptainer or directly.
    5. Run Foldseek.
    6. Parse clusters and assign stable cluster labels.
    7. Optionally copy or rename files using those labels.
    8. Write the main TSV and metadata JSON.
    9. Clean up temporary artifacts unless retention flags were requested.
    """
    args = parse_args()

    user_inputs = [abs_path(path) for path in args.inputs]
    output_prefix_was_default = args.output_prefix is None
    output_prefix = (
        derive_default_output_prefix(user_inputs)
        if output_prefix_was_default
        else abs_path(args.output_prefix)
    )
    tmp_dir = abs_path(args.tmp_dir or f"{output_prefix}_tmp")
    staging_dir = abs_path(args.staging_dir or f"{output_prefix}_staged_inputs")
    apptainer_image = abs_path(args.apptainer_image)
    foldseek_root = abs_path(args.foldseek_root)
    foldseek_bin = foldseek_root / "build" / "src" / "foldseek"

    if output_prefix_was_default:
        eprint(f"No --output-prefix supplied; using derived prefix: {output_prefix}")

    clustered_files_dir = None
    if args.clustered_files_mode == "copy":
        clustered_files_dir = abs_path(
            args.clustered_files_dir or f"{output_prefix}_clustered_files"
        )
    elif args.clustered_files_dir is not None:
        eprint(
            "Ignoring --clustered-files-dir because --clustered-files-mode is not "
            "'copy'."
        )

    if not foldseek_bin.exists():
        raise FileNotFoundError(f"Foldseek binary not found: {foldseek_bin}")
    resolved_execution_mode = resolve_foldseek_execution_mode(args)
    if resolved_execution_mode == "apptainer" and not apptainer_image.exists():
        raise FileNotFoundError(f"Apptainer image not found: {apptainer_image}")

    prepare_output_paths(
        output_prefix=output_prefix,
        staging_dir=staging_dir,
        tmp_dir=tmp_dir,
        clustered_files_dir=clustered_files_dir,
        force=args.force,
    )

    # Track end-to-end timing so the metadata file can describe the full run,
    # not only the Foldseek subprocess time.
    run_started_at = iso_now()
    overall_start_time = time.monotonic()

    # Stage 1: collect and deduplicate supported structure files.
    collect_start_time = time.monotonic()
    eprint(
        "Collecting structure files "
        f"({'recursive' if args.recursive else 'non-recursive'} directory scan)..."
    )
    structure_files, duplicate_paths = collect_structure_files(
        user_inputs,
        recursive=args.recursive,
    )
    collect_elapsed_seconds = time.monotonic() - collect_start_time
    if not structure_files:
        raise RuntimeError("No supported PDB/mmCIF files were found in the inputs.")

    eprint(
        f"Found {len(structure_files)} structure files "
        f"({len(duplicate_paths)} duplicates skipped)."
    )

    # Stage 2: create one staged symlink directory so Foldseek sees a single
    # clean directory input, while optionally recording a reverse map.
    staging_map_output_path = (
        staging_map_path(output_prefix) if args.keep_staging_map else None
    )
    staging_start_time = time.monotonic()
    eprint("Staging symlink inputs...")
    staged_mapping = stage_structure_files(
        structure_files=structure_files,
        staging_dir=staging_dir,
        staging_map_output_path=staging_map_output_path,
    )
    tmp_dir.mkdir(parents=True, exist_ok=False)
    staging_elapsed_seconds = time.monotonic() - staging_start_time

    # Stage 3: convert convenience presets into concrete Foldseek thresholds.
    effective_parameters = resolve_effective_parameters(args)
    eprint(
        "Effective clustering thresholds: "
        f"strictness={effective_parameters['strictness']} "
        f"coverage={effective_parameters['coverage']:.2f} "
        + (
            f"tmscore={effective_parameters['tmscore_threshold']:.2f}"
            if effective_parameters["tmscore_threshold"] is not None
            else "tmscore=n/a"
        )
    )

    # Stage 4: resolve whether Foldseek should be launched via Apptainer or
    # directly in the current environment, then build the exact command.
    binds: list[str] = []
    if resolved_execution_mode == "apptainer":
        binds = build_apptainer_binds(
            user_inputs=user_inputs,
            foldseek_root=foldseek_root,
            staging_dir=staging_dir,
            tmp_dir=tmp_dir,
            output_prefix=output_prefix,
            clustered_files_dir=clustered_files_dir,
            extra_binds=args.extra_bind,
        )
    command = build_foldseek_command(
        args=args,
        effective_parameters=effective_parameters,
        execution_mode=resolved_execution_mode,
        foldseek_bin=foldseek_bin,
        staging_dir=staging_dir,
        tmp_dir=tmp_dir,
        output_prefix=output_prefix,
        binds=binds,
    )

    eprint("Foldseek command:")
    eprint(shlex.join(command))

    succeeded = False
    clusters: list[dict[str, object]] = []
    unresolved_rows = 0
    cluster_table_rows = 0
    file_labeling_plan: dict[str, dict[str, str]] = {}
    foldseek_run_info: dict[str, object] = {"elapsed_seconds": 0.0, "last_phase": ""}
    annotation_elapsed_seconds = 0.0
    file_labeling_elapsed_seconds = 0.0

    try:
        if args.dry_run:
            eprint("Dry run requested. Foldseek was not executed.")
            succeeded = True
            return 0

        # Stage 5: execute Foldseek and stream its logs live.
        foldseek_run_info = run_foldseek(
            command=command,
            total_structures=len(structure_files),
            status_interval_seconds=args.status_interval_seconds,
        )

        # Stage 6: parse Foldseek cluster output back onto the original files.
        annotation_start_time = time.monotonic()
        clusters, unresolved_rows = load_clusters(output_prefix, staged_mapping)
        assign_cluster_labels(clusters, args.cluster_label_prefix)
        annotation_elapsed_seconds = time.monotonic() - annotation_start_time

        if args.clustered_files_mode != "none":
            # Stage 7: optionally create copies or rename originals to append
            # stable cluster labels such as _FS001.
            file_labeling_start_time = time.monotonic()
            file_labeling_plan = plan_file_labeling(
                clusters=clusters,
                mode=args.clustered_files_mode,
                clustered_files_dir=clustered_files_dir,
                cluster_label_prefix=args.cluster_label_prefix,
            )
            execute_file_labeling(file_labeling_plan)
            file_labeling_elapsed_seconds = (
                time.monotonic() - file_labeling_start_time
            )

        # Stage 8: write the default retained artifacts, then remove raw
        # Foldseek outputs unless the user asked to keep them.
        cluster_table_rows = write_main_cluster_table(
            output_prefix=output_prefix,
            clusters=clusters,
            file_labeling_plan=file_labeling_plan,
        )

        cleanup_outputs(
            output_prefix=output_prefix,
            keep_raw_foldseek_outputs=args.keep_raw_foldseek_outputs,
            keep_staging_map=args.keep_staging_map,
        )

        succeeded = True
        return 0
    finally:
        run_finished_at = iso_now()
        total_elapsed_seconds = time.monotonic() - overall_start_time
        cluster_count = len(clusters)

        metadata_payload = {
            "started_at": run_started_at,
            "finished_at": run_finished_at,
            "elapsed_seconds": total_elapsed_seconds,
            "elapsed_hms": format_seconds(total_elapsed_seconds),
            "script": str(Path(__file__).resolve()),
            "output_prefix": str(output_prefix),
            "output_prefix_was_default": output_prefix_was_default,
            "user_inputs": [str(path) for path in user_inputs],
            "structure_count": len(structure_files) if "structure_files" in locals() else 0,
            "duplicate_realpaths_skipped": duplicate_paths
            if "duplicate_paths" in locals()
            else [],
            "cluster_count": cluster_count,
            "cluster_table_rows": cluster_table_rows,
            "unresolved_cluster_rows": unresolved_rows,
            "apptainer_image": str(apptainer_image),
            "foldseek_root": str(foldseek_root),
            "command": command if "command" in locals() else [],
            "command_shell": shlex.join(command) if "command" in locals() else "",
            "status_interval_seconds": args.status_interval_seconds,
            "success": succeeded,
            "durations_seconds": {
                "collect": collect_elapsed_seconds
                if "collect_elapsed_seconds" in locals()
                else 0.0,
                "stage": staging_elapsed_seconds
                if "staging_elapsed_seconds" in locals()
                else 0.0,
                "foldseek": foldseek_run_info["elapsed_seconds"],
                "annotation": annotation_elapsed_seconds,
                "file_labeling": file_labeling_elapsed_seconds,
                "total": total_elapsed_seconds,
            },
            "effective_parameters": {
                "foldseek_execution_mode_requested": args.foldseek_execution_mode,
                "foldseek_execution_mode_resolved": resolved_execution_mode
                if "resolved_execution_mode" in locals()
                else args.foldseek_execution_mode,
                "recursive": args.recursive,
                "alignment_type": args.alignment_type,
                "strictness": effective_parameters["strictness"]
                if "effective_parameters" in locals()
                else args.strictness,
                "coverage": effective_parameters["coverage"]
                if "effective_parameters" in locals()
                else None,
                "coverage_overridden": effective_parameters["coverage_overridden"]
                if "effective_parameters" in locals()
                else False,
                "cov_mode": args.cov_mode,
                "tmscore_threshold": (
                    effective_parameters["tmscore_threshold"]
                    if "effective_parameters" in locals()
                    else None
                ),
                "tmscore_threshold_overridden": (
                    effective_parameters["tmscore_threshold_overridden"]
                    if "effective_parameters" in locals()
                    else False
                ),
                "tmscore_threshold_mode": args.tmscore_threshold_mode,
                "evalue": args.evalue,
                "min_seq_id": args.min_seq_id,
                "lddt_threshold": args.lddt_threshold,
                "exact_tmscore": args.exact_tmscore,
                "tmalign_hit_order": args.tmalign_hit_order,
                "tmalign_fast": args.tmalign_fast,
                "sensitivity": args.sensitivity,
                "max_seqs": args.max_seqs,
                "split_memory_limit": args.split_memory_limit,
                "cluster_mode": args.cluster_mode,
                "single_step_clustering": args.single_step_clustering,
                "cluster_reassign": args.cluster_reassign,
                "chain_name_mode": args.chain_name_mode,
                "model_name_mode": args.model_name_mode,
                "raw_foldseek_args": args.foldseek_arg,
            },
            "file_labeling": {
                "mode": args.clustered_files_mode,
                "cluster_label_prefix": args.cluster_label_prefix,
                "planned_files": len(file_labeling_plan),
                "clustered_files_dir": (
                    str(clustered_files_dir) if clustered_files_dir is not None else None
                ),
            },
            "foldseek_last_phase": foldseek_run_info["last_phase"],
            "retained_outputs": retained_output_paths(
                output_prefix=output_prefix,
                clustered_files_dir=clustered_files_dir,
                keep_raw_foldseek_outputs=args.keep_raw_foldseek_outputs,
                keep_staging_map=args.keep_staging_map,
            ),
            "tmp_dir": str(tmp_dir),
            "staging_dir": str(staging_dir),
            "keep_tmp": args.keep_tmp,
            "keep_staging": args.keep_staging,
            "keep_raw_foldseek_outputs": args.keep_raw_foldseek_outputs,
            "keep_staging_map": args.keep_staging_map,
        }

        if not args.dry_run:
            write_metadata(output_prefix, metadata_payload)
            if succeeded:
                eprint(
                    f"Clustering finished in {format_seconds(total_elapsed_seconds)}. "
                    f"Processed {metadata_payload['structure_count']} structures into "
                    f"{cluster_count} clusters."
                )
                eprint(
                    f"Retained outputs: {', '.join(metadata_payload['retained_outputs'])}"
                )

        if args.dry_run:
            if not args.keep_staging and staging_dir.exists():
                remove_existing_path(staging_dir)
            if not args.keep_tmp and tmp_dir.exists():
                remove_existing_path(tmp_dir)
            if (
                not args.keep_staging_map
                and staging_map_path(output_prefix).exists()
            ):
                remove_existing_path(staging_map_path(output_prefix))
        elif succeeded:
            if not args.keep_staging and staging_dir.exists():
                remove_existing_path(staging_dir)
            if not args.keep_tmp and tmp_dir.exists():
                remove_existing_path(tmp_dir)


if __name__ == "__main__":
    raise SystemExit(main())
