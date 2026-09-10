#!/usr/bin/env python3
"""Normalize AlphaFold 3 outputs and convert per-model CIF files to PDB.

Per AF3 job directory:
  1. Read the canonical job name from `*_data.json` and rename the directory.
  2. Flatten per-sample files up to the job root as
     `<job>_idx_<N>_{model.cif, confidences.json, summary_confidences.json}`.
  3. Convert each `*_idx_<N>_model.cif` to `*_idx_<N>_model.pdb` atomically.
  4. Delete originals only after the PDB is structurally valid on disk.

Both AF3 layouts are handled:
  - Legacy: per-model subdirs `..._seed-<S>_sample-<N>/` containing
    `model.cif`, `confidences.json`, `summary_confidences.json`.
  - New:    a `samples/` dir holding
    `..._seed-<X>_sample-<Y>_{model.cif, confidences.json, summary_confidences.json}`.

The script is fully idempotent:
  - Re-running on a fully processed dir is a fast no-op (single iterdir).
  - Re-running on a partially processed dir resumes from a manifest.
  - Re-running mid-convert resumes by re-running just the missing conversions.
  - Backwards compatible with dirs processed (or left partial) by the previous
    version of this script, which did not write a manifest.

Crash safety: every destructive operation is either (a) atomic via temp+rename,
or (b) checkpointed in `.af3_manifest.json` before the next destructive step.
A per-job lockfile prevents two processes from racing on the same directory.

Intended use: one invocation per job via `--af3_subdir <dir>`. The `--af3_dir`
mode is kept for convenience but is not the primary parallelism mechanism —
you're expected to run N single-job commands in parallel (e.g., via sbatch).
"""
import argparse
import errno
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import multiprocessing
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CONVERTER = os.environ.get("MAXIT_SIF", "")   # CIF->PDB converter; see the campaign README

MANIFEST_NAME    = ".af3_manifest.json"
LOCK_NAME        = ".af3_lock"
PARTIAL_SUFFIX   = ".partial"   # used as <name>.pdb.partial during atomic writes
MANIFEST_VERSION = 1

# Legacy AF3 layout: subdirs "...seed-<S>_sample-<N>" (any seed, any sample).
RE_LEGACY_DIR = re.compile(r"seed-(\d+)_sample-(\d+)$")

# New AF3 layout: files inside "samples/" tagged "_seed-<X>_sample-<Y>_".
RE_NEW_TOKEN = re.compile(r"_seed-(\d+)_sample-(\d+)_")
RE_NEW_CONF  = re.compile(r"_seed-\d+_sample-\d+_confidences\.json$",         re.I)
RE_NEW_MODEL = re.compile(r"_seed-\d+_sample-\d+_model\.cif$",                re.I)
RE_NEW_SUM   = re.compile(r"_seed-\d+_sample-\d+_summary_confidences\.json$", re.I)

# Flattened triplets at the job root: "<job>_idx_<N>_{...}".
RE_IDX_MODEL_CIF = re.compile(r"_idx_(\d+)_model\.cif$",                re.I)
RE_IDX_CONF      = re.compile(r"_idx_(\d+)_confidences\.json$",         re.I)
RE_IDX_SUM       = re.compile(r"_idx_(\d+)_summary_confidences\.json$", re.I)
RE_IDX_PDB       = re.compile(r"_idx_(\d+)_model\.pdb$",                re.I)
RE_IDX_PARTIAL   = re.compile(r"_idx_(\d+)_model\.pdb\.partial$",       re.I)

# File kinds tracked per entry.
KIND_MODEL_CIF = "model_cif"
KIND_CONF      = "confidences"
KIND_SUM       = "summary_confidences"
KIND_PDB       = "model_pdb"
ALL_FLATTENED_KINDS = (KIND_MODEL_CIF, KIND_CONF, KIND_SUM)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg: str, level: str = "INFO"):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [{level}] {msg}", flush=True)

def vlog(verbose: bool, msg: str):
    if verbose:
        log(msg, level="DEBUG")


# ---------------------------------------------------------------------------
# Host/process identity for the lock file
# ---------------------------------------------------------------------------

def get_boot_id() -> str:
    """Linux boot ID; unique per boot. Empty string if unavailable."""
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except Exception:
        return ""

def is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but owned by another user. Treat as alive.
        return True
    except OSError:
        return False
    return True


# ---------------------------------------------------------------------------
# Lock
# ---------------------------------------------------------------------------

class LockError(RuntimeError):
    pass

def _lock_payload() -> dict:
    return {
        "pid":        os.getpid(),
        "hostname":   socket.gethostname(),
        "boot_id":    get_boot_id(),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }

def _read_lock(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}

def _can_break_lock(existing: dict) -> bool:
    """True only if we can prove the existing lock is stale from THIS node.

    Empty/corrupt lock content is treated as BUSY (refuse), not stale. A
    prior version returned True here, which combined with the racy
    create-then-write lock open could let a slow process's empty lockfile be
    stolen by a faster competitor before the payload was written. The user
    can manually delete a genuinely abandoned but unparseable lockfile.
    """
    if not existing:
        return False
    if existing.get("hostname") != socket.gethostname():
        return False  # different node — can't verify liveness; refuse
    boot_now = get_boot_id()
    if not boot_now or not existing.get("boot_id"):
        return False  # can't compare reliably; refuse
    if existing["boot_id"] != boot_now:
        return True  # machine rebooted since lock was taken: definitely stale
    pid = existing.get("pid")
    if not isinstance(pid, int):
        return False  # malformed but on same boot: don't risk stealing
    return not is_pid_alive(pid)

@contextmanager
def acquire_job_lock(job_root: Path, verbose: bool = False, max_attempts: int = 5):
    """Acquire an exclusive per-job lock.

    Publication: write payload to a unique tmp file, then `os.link` it to the
    lock path. Atomic: either the lockfile appears with content already in it,
    or someone else owns it.

    Stale-lock break: read the existing lock's exact bytes, decide if it's
    breakable, then atomically `os.rename` it to a unique sidecar name. If the
    rename succeeds and the sidecar's content still matches what we read, we
    won the breakage race. Otherwise (concurrent breaker swapped in a fresh
    lock between our read and rename) we abort cleanly to avoid stealing a
    live lock — closing the TOCTOU window earlier versions had.
    """
    lock_path = job_root / LOCK_NAME
    payload   = json.dumps(_lock_payload()).encode("utf-8")
    tmp_path  = lock_path.with_name(f"{LOCK_NAME}.{os.getpid()}.{id(payload)}.tmp")

    tmp_path.write_bytes(payload)
    try:
        for attempt in range(max_attempts):
            try:
                os.link(str(tmp_path), str(lock_path))
                break
            except FileExistsError:
                try:
                    existing_bytes = lock_path.read_bytes()
                except FileNotFoundError:
                    continue   # gone between FileExistsError and read; retry
                try:
                    existing = json.loads(existing_bytes)
                except (ValueError, json.JSONDecodeError):
                    existing = {}

                if not _can_break_lock(existing):
                    raise LockError(
                        f"Job is locked by another process: pid={existing.get('pid')} "
                        f"host={existing.get('hostname')} "
                        f"started={existing.get('started_at')}"
                    )

                # Atomically claim breakage by renaming to a unique sidecar name.
                broken_path = lock_path.with_name(
                    f"{LOCK_NAME}.broken.{os.getpid()}.{time.time_ns()}.{attempt}"
                )
                try:
                    os.rename(str(lock_path), str(broken_path))
                except FileNotFoundError:
                    continue   # someone else broke it first; retry

                try:
                    moved_bytes = broken_path.read_bytes()
                except FileNotFoundError:
                    moved_bytes = b""
                try:
                    broken_path.unlink()
                except FileNotFoundError:
                    pass

                if moved_bytes != existing_bytes:
                    # The lockfile was replaced between our read and rename:
                    # we just unlinked someone's *fresh* lock. Refuse to
                    # proceed; the user can re-run.
                    raise LockError(
                        "Lock state changed during breakage attempt (race detected). "
                        "Re-run the command."
                    )

                vlog(verbose, f"Broke stale lock at {lock_path} "
                              f"(owner pid={existing.get('pid')}, "
                              f"host={existing.get('hostname')})")
                # Loop back and retry the link.
        else:
            raise LockError(f"Could not acquire lock after {max_attempts} attempts.")
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass

    try:
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# Atomic file primitives
# ---------------------------------------------------------------------------

def atomic_write_json(path: Path, data: dict):
    """Write JSON atomically via tmp + os.replace."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=False))
    os.replace(str(tmp), str(path))

def move_exclusive(src: Path, dst: Path, verbose: bool = False):
    """Move `src` → `dst`, refusing if `dst` already exists.

    Unlike `shutil.move`, this never silently overwrites. The caller is
    responsible for knowing `dst` is free (manifest + per-slot reconciliation).
    """
    if dst.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    vlog(verbose, f"MOVE: {src} → {dst}")
    shutil.move(str(src), str(dst))


# ---------------------------------------------------------------------------
# PDB validation
# ---------------------------------------------------------------------------

def is_valid_pdb(path: Path) -> bool:
    """Structural validity: file exists, non-empty, has ATOM/HETATM, ends with END.

    Reads the file sequentially once. Cheap relative to the conversion itself.
    """
    if not path.is_file():
        return False
    try:
        if path.stat().st_size == 0:
            return False
    except OSError:
        return False

    has_coord = False
    last_record = b""
    try:
        with path.open("rb") as f:
            for line in f:
                # Record type is columns 1–6 per PDB spec.
                rec = line[:6].rstrip()
                if rec in (b"ATOM", b"HETATM"):
                    has_coord = True
                stripped = line.strip()
                if stripped:
                    last_record = stripped
    except OSError:
        return False

    if not has_coord:
        return False
    # Strict equality: "ENDMDL" (multi-model marker) must NOT pass — a
    # truncated multi-model file could end on ENDMDL and look valid.
    return last_record == b"END"


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

@dataclass
class Inventory:
    job_root:           Path
    data_json:          Optional[Path]
    samples_dir:        Optional[Path]          # only set if a real dir (not a symlink)
    legacy_model_dirs:  list                    # list[Path]
    new_tokens:         dict                    # token -> {kind: Path}
    root_idx_groups:    dict                    # idx:int -> {kind: Path}; kinds in ALL_FLATTENED_KINDS + KIND_PDB
    stale_partials:     list                    # list[Path] of *.pdb.partial
    has_manifest:       bool
    cruft:              list                    # list[Path] of known AF3 cruft + top-model duplicates

def inventory_dir(job_root: Path) -> Inventory:
    """Single pass over the job dir (and its samples/ if present).

    Designed for the fast-path check: everything callers need to classify
    the job state comes from this one call.
    """
    inv = Inventory(
        job_root=job_root,
        data_json=None,
        samples_dir=None,
        legacy_model_dirs=[],
        new_tokens={},
        root_idx_groups={},
        stale_partials=[],
        has_manifest=False,
        cruft=[],
    )

    for entry in job_root.iterdir():
        name = entry.name

        if entry.is_file():
            if name == MANIFEST_NAME:
                inv.has_manifest = True
                continue
            if name == LOCK_NAME:
                continue
            if name.endswith("_data.json"):
                inv.data_json = entry
                continue
            # Indexed triplet classifiers MUST run before the cruft detector
            # below, because cruft suffixes (_model.cif, _confidences.json,
            # _summary_confidences.json) are themselves a substring of the
            # indexed names.
            m = RE_IDX_MODEL_CIF.search(name)
            if m:
                inv.root_idx_groups.setdefault(int(m.group(1)), {})[KIND_MODEL_CIF] = entry
                continue
            m = RE_IDX_CONF.search(name)
            if m:
                inv.root_idx_groups.setdefault(int(m.group(1)), {})[KIND_CONF] = entry
                continue
            m = RE_IDX_SUM.search(name)
            if m:
                inv.root_idx_groups.setdefault(int(m.group(1)), {})[KIND_SUM] = entry
                continue
            m = RE_IDX_PDB.search(name)
            if m:
                inv.root_idx_groups.setdefault(int(m.group(1)), {})[KIND_PDB] = entry
                continue
            m = RE_IDX_PARTIAL.search(name)
            if m:
                inv.stale_partials.append(entry)
                continue
            if name in AF3_FIXED_CRUFT:
                inv.cruft.append(entry)
                continue
            if any(name.endswith(s) for s in AF3_TOP_MODEL_SUFFIXES):
                inv.cruft.append(entry)
                continue
            # Unknown top-level file — leave alone.
            continue

        if entry.is_dir():
            if name == "samples" and not entry.is_symlink():
                inv.samples_dir = entry
                for p in entry.iterdir():
                    if not p.is_file():
                        continue
                    m = RE_NEW_TOKEN.search(p.name)
                    if not m:
                        continue
                    token = m.group(0)
                    group = inv.new_tokens.setdefault(token, {})
                    if RE_NEW_MODEL.search(p.name):
                        group[KIND_MODEL_CIF] = p
                    elif RE_NEW_CONF.search(p.name):
                        group[KIND_CONF] = p
                    elif RE_NEW_SUM.search(p.name):
                        group[KIND_SUM] = p
                continue
            if RE_LEGACY_DIR.search(name) and not entry.is_symlink():
                inv.legacy_model_dirs.append(entry)
                continue
            # Unknown subdir or symlinked legacy match — leave alone (data-loss guard).

    return inv


# ---------------------------------------------------------------------------
# Fast-path: is the job already done?
# ---------------------------------------------------------------------------

def is_already_processed(inv: Inventory) -> bool:
    """True if the dir is in the canonical "indexed-only" final shape.

    Specifically:
      - No manifest, lock-not-checked (handled separately)
      - No sources: no samples/ files, no legacy model dirs
      - No stale .pdb.partial files
      - No `_data.json` (would still need cleanup)
      - No AF3 cruft/top-model duplicates (would still need cleanup)
      - No empty `samples/` dir lingering (would still need cleanup)
      - At least one idx group exists, every group has valid PDB and no CIF
    """
    if inv.has_manifest:
        return False
    if inv.samples_dir is not None:   # even an empty samples/ needs removal
        return False
    if inv.new_tokens:
        return False
    if inv.legacy_model_dirs:
        return False
    if inv.stale_partials:
        return False
    if inv.data_json is not None:
        return False
    if inv.cruft:
        return False
    if not inv.root_idx_groups:
        return False
    for idx, files in inv.root_idx_groups.items():
        pdb = files.get(KIND_PDB)
        if pdb is None or not is_valid_pdb(pdb):
            return False
        if KIND_MODEL_CIF in files:
            return False
    return True


# ---------------------------------------------------------------------------
# Job name resolution
# ---------------------------------------------------------------------------

def read_job_name_from_data_json(data_json: Path) -> str:
    with data_json.open("r") as f:
        data = json.load(f)
    name = data.get("name")
    if not name:
        raise KeyError(f"'name' missing/empty in {data_json}")
    return name

def infer_job_name_from_idx_files(inv: Inventory) -> Optional[str]:
    """Recover the job name from an existing `<job>_idx_<N>_...` file's prefix."""
    for files in inv.root_idx_groups.values():
        for kind, p in files.items():
            name = p.name
            if kind == KIND_MODEL_CIF:
                m = RE_IDX_MODEL_CIF.search(name); suffix = m.group(0) if m else None
            elif kind == KIND_CONF:
                m = RE_IDX_CONF.search(name);      suffix = m.group(0) if m else None
            elif kind == KIND_SUM:
                m = RE_IDX_SUM.search(name);       suffix = m.group(0) if m else None
            elif kind == KIND_PDB:
                m = RE_IDX_PDB.search(name);       suffix = m.group(0) if m else None
            else:
                continue
            if not suffix:
                continue
            prefix = name[: name.rfind(suffix)]
            # The suffix regex includes the leading "_idx", so strip the remainder.
            # Actually: suffix starts with "_idx_"; prefix is "<job>".
            if prefix:
                return prefix
    return None

def resolve_job_name(job_dir: Path, inv: Inventory, manifest_name: Optional[str]) -> str:
    """Determine canonical job name from (in order): manifest, data.json, existing idx file, dirname."""
    if manifest_name:
        return manifest_name
    if inv.data_json is not None:
        return read_job_name_from_data_json(inv.data_json)
    inferred = infer_job_name_from_idx_files(inv)
    if inferred:
        return inferred
    return job_dir.name


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

@dataclass
class Entry:
    idx:           int
    provenance:    str                # e.g. "_seed-1_sample-3_" or "seed-1_sample-3" or "root_only"
    src_layout:    str                # "new" | "legacy" | "root_only"
    src_dir:       str                # relative to job_root: "samples", "<dirname>", or ""
    src_files:     dict = field(default_factory=dict)  # kind -> filename (within src_dir)
    dst_files:     dict = field(default_factory=dict)  # kind -> filename (within job_root)
    flattened:     bool = False
    converted:    bool = False
    src_deleted:   bool = False

@dataclass
class Manifest:
    version:       int
    job_name:      str
    source_layout: str                # "new" | "legacy" | "mixed"
    created_at:    str
    entries:       list               # list[Entry]

    def to_dict(self) -> dict:
        return {
            "version":       self.version,
            "job_name":      self.job_name,
            "source_layout": self.source_layout,
            "created_at":    self.created_at,
            "entries":       [asdict(e) for e in self.entries],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Manifest":
        if d.get("version") != MANIFEST_VERSION:
            raise RuntimeError(f"Unsupported manifest version: {d.get('version')}")
        return cls(
            version       = d["version"],
            job_name      = d["job_name"],
            source_layout = d["source_layout"],
            created_at    = d["created_at"],
            entries       = [Entry(**e) for e in d["entries"]],
        )

def load_manifest(job_root: Path) -> Optional[Manifest]:
    path = job_root / MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        return Manifest.from_dict(json.loads(path.read_text()))
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Corrupt manifest at {path}: {e}")

def save_manifest(job_root: Path, manifest: Manifest):
    atomic_write_json(job_root / MANIFEST_NAME, manifest.to_dict())

def delete_manifest(job_root: Path):
    try:
        (job_root / MANIFEST_NAME).unlink()
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Manifest construction from disk
# ---------------------------------------------------------------------------

class AmbiguousResumeError(RuntimeError):
    pass

class NothingToDoError(RuntimeError):
    pass

def _dst_names(job_name: str, idx: int) -> dict:
    return {
        KIND_MODEL_CIF: f"{job_name}_idx_{idx}_model.cif",
        KIND_CONF:      f"{job_name}_idx_{idx}_confidences.json",
        KIND_SUM:       f"{job_name}_idx_{idx}_summary_confidences.json",
    }

def _new_token_sort_key_numeric(token: str) -> tuple:
    """Sort new-layout tokens by numeric (seed, sample), stable across digit counts."""
    m = RE_NEW_TOKEN.search(token)
    if not m:
        return (10**9, 10**9, token)
    return (int(m.group(1)), int(m.group(2)), token)

def _build_legacy(inv: Inventory, job_name: str) -> Manifest:
    """Fresh OR partial legacy: each model dir's (seed, sample) deterministically maps to idx.

    For backwards compat with single-seed jobs, idx = sample number directly (old behavior).
    For multi-seed, idx is the 0-based rank of (seed, sample) in sorted order.
    """
    seeds = {int(RE_LEGACY_DIR.search(d.name).group(1)) for d in inv.legacy_model_dirs}
    single_seed = len(seeds) == 1

    # Sort dirs deterministically by (seed, sample).
    dirs_sorted = sorted(
        inv.legacy_model_dirs,
        key=lambda d: tuple(int(x) for x in RE_LEGACY_DIR.search(d.name).groups()),
    )

    used_idx: set = set()
    entries: list = []
    for rank, d in enumerate(dirs_sorted):
        seed, sample = (int(x) for x in RE_LEGACY_DIR.search(d.name).groups())
        idx = sample if single_seed else rank
        if idx in used_idx:
            raise AmbiguousResumeError(
                f"Duplicate idx {idx} inferred from legacy dirs (multi-seed collision)."
            )
        used_idx.add(idx)

        # Cross-reference any matching root-idx group so we don't try to
        # convert a CIF that the previous run already converted+deleted while
        # leaving an empty model dir behind.
        root_files     = inv.root_idx_groups.get(idx, {})
        src_files_left = {
            kind: fname
            for kind, fname in (
                (KIND_MODEL_CIF, "model.cif"),
                (KIND_CONF,      "confidences.json"),
                (KIND_SUM,       "summary_confidences.json"),
            )
            if (d / fname).is_file()
        }
        has_valid_pdb = KIND_PDB in root_files and is_valid_pdb(root_files[KIND_PDB])
        flattened     = (len(src_files_left) == 0)
        converted     = has_valid_pdb
        # Only consider the dst CIF deleted if all flattening is finished AND
        # there is no CIF anywhere (root or dir). Otherwise reconcile_flatten
        # would later produce a root CIF that this flag would prevent us from
        # deleting at the end of the run.
        src_deleted   = (
            converted
            and flattened
            and KIND_MODEL_CIF not in root_files
        )

        entries.append(Entry(
            idx=idx,
            provenance=d.name,
            src_layout="legacy",
            src_dir=d.name,
            src_files={
                KIND_MODEL_CIF: "model.cif",
                KIND_CONF:      "confidences.json",
                KIND_SUM:       "summary_confidences.json",
            },
            dst_files=_dst_names(job_name, idx),
            flattened=flattened,
            converted=converted,
            src_deleted=src_deleted,
        ))

    # Orphan root idx groups (model dir already gone from a prior partial run).
    for idx, files in sorted(inv.root_idx_groups.items()):
        if idx in used_idx:
            continue
        used_idx.add(idx)
        entries.append(Entry(
            idx=idx,
            provenance=f"root_idx_{idx}",
            src_layout="root_only",
            src_dir="",
            src_files={},
            dst_files=_dst_names(job_name, idx),
            flattened=True,
            converted=(KIND_PDB in files and is_valid_pdb(files[KIND_PDB])),
            src_deleted=(KIND_MODEL_CIF not in files),
        ))

    entries.sort(key=lambda e: e.idx)
    return Manifest(MANIFEST_VERSION, job_name, "legacy",
                    datetime.now().isoformat(timespec="seconds"), entries)

def _build_new(inv: Inventory, job_name: str) -> Manifest:
    """Fresh new-layout OR partial with strict safety checks.

    Fresh: enumerate samples/ tokens in numeric (seed, sample) order; idx 0..N-1.
    Partial resume without manifest:
      - Every existing root idx group must have the FULL triplet (model.cif +
        confidences + summary). Otherwise we can't tell which token was in
        flight.
      - Root idx values must be a contiguous prefix 0..K. Otherwise ambiguous.
      - Remaining tokens in samples/ get idx values K+1..K+M in LEXICAL order
        (to match the previous version's behavior, which used string sort).
    """
    tokens = sorted(inv.new_tokens.keys(), key=_new_token_sort_key_numeric)

    if not inv.root_idx_groups:
        entries = []
        for idx, token in enumerate(tokens):
            srcs = inv.new_tokens[token]
            entries.append(Entry(
                idx=idx,
                provenance=token,
                src_layout="new",
                src_dir="samples",
                src_files={k: srcs[k].name for k in ALL_FLATTENED_KINDS if k in srcs},
                dst_files=_dst_names(job_name, idx),
            ))
        return Manifest(MANIFEST_VERSION, job_name, "new",
                        datetime.now().isoformat(timespec="seconds"), entries)

    # Partial new-layout without manifest. Apply strict safety checks.
    root_idx_sorted = sorted(inv.root_idx_groups.keys())
    if root_idx_sorted != list(range(len(root_idx_sorted))):
        raise AmbiguousResumeError(
            f"Non-contiguous idx set {root_idx_sorted} at root with sources still present. "
            f"Cannot safely resume without manifest."
        )
    for idx, files in inv.root_idx_groups.items():
        if not all(k in files for k in ALL_FLATTENED_KINDS):
            raise AmbiguousResumeError(
                f"idx {idx} at root is missing one of model.cif/confidences/summary_confidences. "
                f"Cannot safely resume without manifest (provenance lost)."
            )
    # Samples tokens must each have the full triplet — otherwise some kind was
    # moved out mid-token and we have no way to map it back.
    for token, srcs in inv.new_tokens.items():
        if not all(k in srcs for k in ALL_FLATTENED_KINDS):
            raise AmbiguousResumeError(
                f"samples/ token {token!r} is missing one of model.cif/confidences/summary_confidences. "
                f"Cannot safely resume without manifest (mid-token partial move)."
            )

    # Previous version used lexical string sort on tokens; emulate that here
    # so we assign the remaining tokens in a way consistent with the first
    # partial run. Numeric (seed, sample) sort would misalign.
    remaining_tokens = sorted(inv.new_tokens.keys())
    next_idx = len(root_idx_sorted)
    entries = []
    for idx in root_idx_sorted:
        files = inv.root_idx_groups[idx]
        entries.append(Entry(
            idx=idx,
            provenance=f"root_idx_{idx}",
            src_layout="root_only",
            src_dir="",
            src_files={},
            dst_files=_dst_names(job_name, idx),
            flattened=True,
            converted=(KIND_PDB in files and is_valid_pdb(files[KIND_PDB])),
            src_deleted=(KIND_MODEL_CIF not in files),
        ))
    for token in remaining_tokens:
        srcs = inv.new_tokens[token]
        entries.append(Entry(
            idx=next_idx,
            provenance=token,
            src_layout="new",
            src_dir="samples",
            src_files={k: srcs[k].name for k in ALL_FLATTENED_KINDS if k in srcs},
            dst_files=_dst_names(job_name, next_idx),
        ))
        next_idx += 1

    entries.sort(key=lambda e: e.idx)
    return Manifest(MANIFEST_VERSION, job_name, "new",
                    datetime.now().isoformat(timespec="seconds"), entries)

def _build_root_only(inv: Inventory, job_name: str) -> Manifest:
    """No sources, only idx groups at root (possibly missing PDBs)."""
    entries = []
    for idx, files in sorted(inv.root_idx_groups.items()):
        entries.append(Entry(
            idx=idx,
            provenance=f"root_idx_{idx}",
            src_layout="root_only",
            src_dir="",
            src_files={},
            dst_files=_dst_names(job_name, idx),
            flattened=True,
            converted=(KIND_PDB in files and is_valid_pdb(files[KIND_PDB])),
            src_deleted=(KIND_MODEL_CIF not in files),
        ))
    return Manifest(MANIFEST_VERSION, job_name, "new",
                    datetime.now().isoformat(timespec="seconds"), entries)

def build_manifest(inv: Inventory, job_name: str) -> Manifest:
    if inv.legacy_model_dirs:
        return _build_legacy(inv, job_name)
    if inv.samples_dir is not None and inv.new_tokens:
        return _build_new(inv, job_name)
    if inv.root_idx_groups:
        return _build_root_only(inv, job_name)
    raise NothingToDoError("No AF3 sources and no flattened outputs in job dir.")


# ---------------------------------------------------------------------------
# Per-entry reconciliation (flatten → convert → delete source)
# ---------------------------------------------------------------------------

def _src_path(job_root: Path, entry: Entry, kind: str) -> Optional[Path]:
    """Resolve current disk path for entry's source file of given kind, or None."""
    if not entry.src_dir:
        return None
    name = entry.src_files.get(kind)
    if not name:
        return None
    return job_root / entry.src_dir / name

def _dst_path(job_root: Path, entry: Entry, kind: str) -> Path:
    return job_root / entry.dst_files[kind]

def reconcile_flatten(job_root: Path, entry: Entry, verbose: bool = False) -> bool:
    """Move any un-moved source files for this entry to their destinations.

    Per-kind reconciliation handles all four src/dst presence combinations:
      - src+dst: anomaly — both copies exist. Stop with a clear error.
      - src only: move now.
      - dst only: already moved; noop.
      - neither: missing; noop (entry will be marked flattened anyway if
        provenance tracks it that way).

    Returns True if `entry.flattened` changed from False to True.
    """
    if entry.flattened:
        return False

    for kind in ALL_FLATTENED_KINDS:
        src = _src_path(job_root, entry, kind)
        dst = _dst_path(job_root, entry, kind)

        src_exists = src is not None and src.is_file()
        dst_exists = dst.is_file()

        if src_exists and dst_exists:
            raise RuntimeError(
                f"Anomaly at idx {entry.idx} kind {kind}: both src ({src}) and dst ({dst}) exist."
            )
        if src_exists:
            move_exclusive(src, dst, verbose=verbose)
        # else: dst-only or missing — nothing to do here.

    entry.flattened = True
    return True

def convert_atomic(job_root: Path, entry: Entry, converter: str, verbose: bool = False) -> bool:
    """Convert entry's CIF → PDB atomically. Returns True if `converted` changed.

    Writes to `<final>.partial`, verifies structural validity, then atomically
    renames to the final name. Never touches the source CIF.
    """
    if entry.converted:
        # Double-check on disk: if a previous run crashed between setting the
        # flag and fsyncing, we might still be OK. But we don't flip it back.
        return False

    cif = _dst_path(job_root, entry, KIND_MODEL_CIF)
    pdb = job_root / f"{entry.dst_files[KIND_MODEL_CIF][:-4]}.pdb"
    partial = pdb.with_name(pdb.name + PARTIAL_SUFFIX)

    if not cif.is_file():
        raise RuntimeError(f"idx {entry.idx}: cannot convert — CIF missing at {cif}")

    # If a valid PDB already exists, we can short-circuit.
    if is_valid_pdb(pdb):
        entry.converted = True
        return True

    # Clear any stale partial from a previous crash.
    if partial.exists():
        vlog(verbose, f"Removing stale partial: {partial}")
        partial.unlink()

    vlog(verbose, f"CONVERT: {cif} → {pdb}")
    args = [converter, "-input", str(cif), "-output", str(partial), "-o", "2"]
    subprocess.run(args, check=True)

    if not is_valid_pdb(partial):
        # Keep the partial for inspection; do NOT clobber the CIF.
        raise RuntimeError(
            f"idx {entry.idx}: converter produced an invalid PDB at {partial}. "
            f"CIF left in place for re-conversion."
        )

    os.replace(str(partial), str(pdb))
    entry.converted = True
    return True

def delete_src_if_converted(job_root: Path, entry: Entry, verbose: bool = False) -> bool:
    """Delete the source CIF once the PDB is verified present. Returns True if flag changed."""
    if entry.src_deleted or not entry.converted:
        return False
    pdb = job_root / f"{entry.dst_files[KIND_MODEL_CIF][:-4]}.pdb"
    if not is_valid_pdb(pdb):
        # Defensive: don't delete if the PDB went missing/invalid somehow.
        raise RuntimeError(f"idx {entry.idx}: refusing to delete CIF; PDB not valid at {pdb}")
    cif = _dst_path(job_root, entry, KIND_MODEL_CIF)
    if cif.is_file():
        vlog(verbose, f"Removing CIF (PDB verified): {cif.name}")
        cif.unlink()
    entry.src_deleted = True
    return True


# ---------------------------------------------------------------------------
# Final cleanup
# ---------------------------------------------------------------------------

# AF3 emits a "top-ranked" copy of the best sample at the job root, plus a
# license file and a ranking summary. These are duplicates of the per-sample
# data we already preserve under the indexed names, so they are safe to remove
# at the end of a successful run. Listed by exact basename (job_name-suffixed
# entries are produced at finalize time).
AF3_FIXED_CRUFT = ("ranking_scores.csv", "TERMS_OF_USE.md")
AF3_TOP_MODEL_SUFFIXES = (
    "_model.cif",
    "_confidences.json",
    "_summary_confidences.json",
)

def remove_empty_dir(path: Path, verbose: bool = False):
    """Remove `path` only if it is a real empty directory (not a symlink)."""
    if not path.exists() or path.is_symlink() or not path.is_dir():
        return
    try:
        next(path.iterdir())
    except StopIteration:
        vlog(verbose, f"Removing empty dir: {path}")
        path.rmdir()

def remove_known_cruft(job_root: Path, job_name: str, verbose: bool = False):
    """Remove AF3 outputs we know are duplicates/license boilerplate.

    Positive allowlist only — anything not on this list is left alone, so
    user-added files (notes, custom outputs) are never lost. Deleting these
    is what brings a finalized job dir into the canonical "indexed-only"
    shape.
    """
    for name in AF3_FIXED_CRUFT:
        p = job_root / name
        if p.is_file() and not p.is_symlink():
            vlog(verbose, f"Removing AF3 cruft: {name}")
            p.unlink()
    for suffix in AF3_TOP_MODEL_SUFFIXES:
        p = job_root / f"{job_name}{suffix}"
        if p.is_file() and not p.is_symlink():
            vlog(verbose, f"Removing AF3 top-model duplicate: {p.name}")
            p.unlink()

def finalize_job(job_root: Path, job_name: str, verbose: bool = False):
    """All entries complete: delete data.json + AF3 cruft, remove empty source dirs, delete manifest.

    This is the ONLY path that deletes the manifest. Nothing else touches it.
    Order matters: cruft and source dirs first, manifest last (its presence
    is how a resumable partial run is detected).
    """
    for p in job_root.glob("*_data.json"):
        vlog(verbose, f"Removing {p.name}")
        p.unlink()

    remove_known_cruft(job_root, job_name, verbose=verbose)

    samples = job_root / "samples"
    if samples.is_dir() and not samples.is_symlink():
        remove_empty_dir(samples, verbose=verbose)

    for d in job_root.iterdir():
        if d.is_dir() and not d.is_symlink() and RE_LEGACY_DIR.search(d.name):
            remove_empty_dir(d, verbose=verbose)

    for p in job_root.glob(f"{MANIFEST_NAME}.tmp"):
        vlog(verbose, f"Removing stray manifest tmp: {p.name}")
        try:
            p.unlink()
        except FileNotFoundError:
            pass

    delete_manifest(job_root)


# ---------------------------------------------------------------------------
# Directory rename (uses canonical job name from data.json)
# ---------------------------------------------------------------------------

def rename_job_dir_if_needed(job_dir: Path, canonical_name: str, verbose: bool = False) -> Path:
    """Rename `job_dir` to `canonical_name` (as sibling) if not already there.

    Returns the (possibly new) path. Never renames across mounts — POSIX rename
    on the same filesystem is atomic. The lock/manifest move with the dir.
    """
    target = job_dir.parent / canonical_name
    if job_dir.resolve() == target.resolve():
        return job_dir
    if target.exists():
        raise FileExistsError(f"Target directory already exists: {target}")
    vlog(verbose, f"RENAME DIR: {job_dir.name} → {target.name}")
    os.rename(str(job_dir), str(target))
    return target


# ---------------------------------------------------------------------------
# Orchestration: process a single AF3 job directory
# ---------------------------------------------------------------------------

def process_one(subdir: Path,
                converter: str,
                verbose: bool = False,
                disable_deletions: bool = False) -> str:
    """Process one AF3 job dir to a fully normalized state.

    Returns one of: "already_processed", "completed", "nothing_to_do".
    Raises for anything the script can't safely resolve (ambiguous resume,
    lock contention, converter failure, etc.).
    """
    # 1) Cheap inventory + fast-path.
    pre_inv = inventory_dir(subdir)
    if is_already_processed(pre_inv):
        vlog(verbose, f"Already processed (fast-path): {subdir.name}")
        return "already_processed"

    # 2) Resolve canonical name (prefer existing manifest on disk, then data.json,
    #    then inferred-from-filename). Rename dir if needed.
    existing_manifest = load_manifest(subdir)
    manifest_name = existing_manifest.job_name if existing_manifest else None
    canonical = resolve_job_name(subdir, pre_inv, manifest_name)
    job_root = rename_job_dir_if_needed(subdir, canonical, verbose=verbose)

    # 3) Acquire per-dir lock.
    with acquire_job_lock(job_root, verbose=verbose):
        # 4) Re-inventory inside the lock (state may have changed during rename).
        inv = inventory_dir(job_root)

        # Remove any stale .pdb.partial files from prior crashes.
        for p in inv.stale_partials:
            vlog(verbose, f"Removing stale partial: {p.name}")
            p.unlink()

        # 5) Load or build manifest.
        manifest = load_manifest(job_root)
        if manifest is None:
            try:
                manifest = build_manifest(inv, canonical)
            except NothingToDoError:
                log(f"Nothing to do in {job_root.name} (no sources, no outputs).", level="WARN")
                return "nothing_to_do"
            save_manifest(job_root, manifest)
        else:
            # If manifest exists but disk already has valid PDBs, reflect that.
            for e in manifest.entries:
                if not e.converted:
                    pdb = job_root / f"{e.dst_files[KIND_MODEL_CIF][:-4]}.pdb"
                    if is_valid_pdb(pdb):
                        e.converted = True

        if disable_deletions:
            log(f"DEBUG: --disable-deletions set. Inventory complete, no destructive action.",
                level="WARN")
            return "nothing_to_do"

        # 6) Per-entry reconcile: flatten → convert → delete source.
        #    Checkpoint after each destructive operation.
        for entry in manifest.entries:
            if reconcile_flatten(job_root, entry, verbose=verbose):
                save_manifest(job_root, manifest)

            if not entry.converted:
                convert_atomic(job_root, entry, converter, verbose=verbose)
                save_manifest(job_root, manifest)

            if delete_src_if_converted(job_root, entry, verbose=verbose):
                save_manifest(job_root, manifest)

        # 7) Finalize (only when all entries fully complete).
        if all(e.flattened and e.converted and e.src_deleted for e in manifest.entries):
            finalize_job(job_root, manifest.job_name, verbose=verbose)
            log(f"DONE: {job_root.name} ({len(manifest.entries)} models)")
            return "completed"
        else:
            incomplete = [e.idx for e in manifest.entries
                          if not (e.flattened and e.converted and e.src_deleted)]
            log(f"PARTIAL: {job_root.name} — incomplete idx {incomplete}", level="WARN")
            return "completed"   # still “we did work” — caller can re-run to finish


# ---------------------------------------------------------------------------
# Multi-job driver (secondary path; user typically sbatches single-job commands)
# ---------------------------------------------------------------------------

def _worker_main(queue, counter, lock, total, converter, verbose, disable_deletions):
    while True:
        item = queue.get()
        if item is None:
            return
        idx, subdir = item
        try:
            process_one(Path(subdir), converter,
                        verbose=verbose, disable_deletions=disable_deletions)
        except Exception as e:
            with lock:
                counter.value += 1
            log(f"ERROR processing {subdir}: {e}", level="ERROR")
        # Progress every ~5%.
        step = max(1, total // 20)
        if (idx + 1) % step == 0 or (idx + 1) == total:
            log(f"PROGRESS: {idx + 1}/{total}")

def run_many(af3_root: Path,
             converter: str,
             max_procs: int,
             verbose: bool,
             disable_deletions: bool) -> int:
    """Process every subdir under `af3_root`. Returns the number of failed jobs."""
    subdirs = sorted([p for p in af3_root.iterdir() if p.is_dir()])
    total = len(subdirs)
    log(f"Discovered {total} subdirectories under {af3_root}")
    if total == 0:
        return 0

    nwork = max(1, min(max_procs, total))
    log(f"Starting pool with {nwork} workers (deletions {'DISABLED' if disable_deletions else 'ENABLED'})")

    manager = multiprocessing.Manager()
    q       = manager.Queue()
    counter = manager.Value("i", 0)     # count of failed jobs
    cblock  = manager.Lock()

    for i, d in enumerate(subdirs):
        q.put((i, str(d)))
    for _ in range(nwork):
        q.put(None)

    pool = []
    for _ in range(nwork):
        p = multiprocessing.Process(
            target=_worker_main,
            args=(q, counter, cblock, total, converter, verbose, disable_deletions),
        )
        p.start()
        pool.append(p)
    for p in pool:
        p.join()

    failed = counter.value
    if failed:
        log(f"{failed}/{total} jobs failed. See ERROR lines above.", level="ERROR")
    else:
        log(f"All {total} jobs complete.")
    return failed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize AF3 outputs: flatten samples, convert CIF→PDB (atomic), clean up. Idempotent and resumable."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--af3_dir",
                     help="Directory containing many AF3 job subdirs.")
    src.add_argument("--af3_subdir",
                     help="Single AF3 job directory (preferred: run N of these in parallel via sbatch).")
    parser.add_argument("--cif_sif",  default=DEFAULT_CONVERTER,
                        help=f"Path to CIF→PDB converter (default: {DEFAULT_CONVERTER}).")
    parser.add_argument("--max_procs", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                        help="Worker processes for --af3_dir mode only.")
    parser.add_argument("--disable-deletions", action="store_true",
                        help="Inventory and log only; do not flatten, convert, or delete.")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose logging.")
    # Historic flag; silently accepted (state-based classifier makes it a no-op).
    parser.add_argument("--robust-mode", action="store_true",
                        help=argparse.SUPPRESS)
    # Historic flag; non-indexed CIFs are now left alone by design, not deleted.
    parser.add_argument("--include-nonindexed-cifs", action="store_true",
                        help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.robust_mode:
        log("--robust-mode is deprecated and ignored (behavior is now state-based).", level="WARN")

    if args.af3_subdir:
        try:
            process_one(Path(args.af3_subdir), args.cif_sif,
                        verbose=args.verbose,
                        disable_deletions=args.disable_deletions)
            return 0
        except Exception as e:
            log(f"ERROR: {e}", level="ERROR")
            return 1

    failed = run_many(Path(args.af3_dir), args.cif_sif,
                      max_procs=args.max_procs,
                      verbose=args.verbose,
                      disable_deletions=args.disable_deletions)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
