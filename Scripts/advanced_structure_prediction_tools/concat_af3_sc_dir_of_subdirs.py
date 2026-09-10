#!/usr/bin/env python3
"""
SCRIPT: concat_af3_sc_dir_of_subdirs.py

PURPOSE (CSV-only):
  Given --af3_dir_of_subdirs (a directory containing many subdirectories),
  go 1-level into each subdirectory to find its .sc file (CSV: header + data),
  and concatenate all of them into a single CSV.

  Priority per subdir "<d>":
    1) <d>/<d>.sc
    2) <d>/*.sc (first match)
    3) If none, skip with a note.

DESIGNED FOR MASSIVE SCALE (~100k+ subdirs):
  - Bounded threaded discovery with ETA.
  - Default single-read parse: each remote .sc is opened once.
  - Header union is built while raw parsed rows are written to temporary shards.
  - Final remap reads temporary raw shards, writes union-schema CSV shards, then
    streams them into the final output.
  - Final streaming concat (no big RAM use).
  - Keeps union of all columns; fills missing values with empty string ("").
  - Verbose logging: counts, elapsed, ETA, memory, avg/file, shard details.

DEFAULT PIPELINE (since 2026-06): single-read + bounded threaded parse.
  - Remote .sc files are opened once, not once for header union and again for
    data parsing. This is much faster on NFS-backed scratch trees.
  - Thread pools use a bounded in-flight window so peak RAM is
    O(max_inflight + chunk_rows), flat regardless of file count.
  - Output is identical (same rows + columns); row order is completion-order.
  - Use --low_memory for the legacy single-threaded Pass 2 (minimal RAM, slower)
    on extreme-scale or very memory-limited nodes.
  - Use --nfs_friendly for conservative defaults on shared network filesystems.
  - --fast is now a deprecated no-op (single-read threaded is the default).

USAGE:
  python concat_af3_sc_dir_of_subdirs.py \
      --af3_dir_of_subdirs /path/to/AF3/iteration_dir \
      [--optional_path_for_summary_stats /path/to/out.csv] \
      [--chunk_rows 10000] \
      [--workers 8] \
      [--max_inflight 64] \
      [--nfs_friendly] \
      [--low_memory] \
      [--strict_name_match | --assume_preferred_sc_name]

OUTPUT:
  - If --optional_path_for_summary_stats is provided: write there.
  - Else: write zzzzz_af3_analysis_csv_zzzzz.csv inside --af3_dir_of_subdirs.
"""

import os
import csv
import glob
import json
import time
import shlex
import shutil
import argparse
import tempfile
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from functools import partial

# -------------------------
# Utils: timing + memory
# -------------------------
def fmt_secs(s: float) -> str:
    if s < 60: return f"{s:.1f}s"
    m, sec = divmod(int(s), 60)
    if m < 60: return f"{m}m {sec}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {sec}s"

def get_mem_used_mb() -> float:
    """Best-effort memory usage in MB (Linux). Falls back gracefully."""
    try:
        import psutil  # optional
        return psutil.Process().memory_info().rss / (1024**2)
    except Exception:
        try:
            with open("/proc/self/statm", "r") as f:
                parts = f.read().strip().split()
                rss_pages = int(parts[1])
                page_size = os.sysconf("SC_PAGE_SIZE")
                return (rss_pages * page_size) / (1024**2)
        except Exception:
            return float("nan")

def get_cpu_count() -> int:
    try:
        import multiprocessing
        return multiprocessing.cpu_count()
    except Exception:
        return 1

def default_worker_count(nfs_friendly: bool = False) -> int:
    cpu = get_cpu_count()
    if nfs_friendly:
        return min(8, max(1, 2 * cpu))
    return min(64, max(4, 2 * cpu))

def default_max_inflight(workers: int, nfs_friendly: bool = False) -> int:
    if nfs_friendly:
        return 64
    return max(2 * workers, 64)

# -----------------------------------------------------------
# Bounded-window threaded map (constant memory at any scale)
# -----------------------------------------------------------
def bounded_imap_unordered(fn, items, workers, max_inflight):
    """
    Apply fn to each item on a thread pool, keeping at most `max_inflight`
    futures live at once: a new task is submitted each time one completes.
    Yields (item, future) pairs in completion order; the future is already
    done, so caller can call future.result() without blocking.

    Memory is O(max_inflight) regardless of len(items). This is the key
    difference from submitting every future up front (which makes the
    in-flight set, and thus RAM, grow with the total file count).
    """
    it = iter(items)
    future_item = {}
    pending = set()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        def _submit_next():
            try:
                item = next(it)
            except StopIteration:
                return False
            fut = ex.submit(fn, item)
            future_item[fut] = item
            pending.add(fut)
            return True
        # Prime the window.
        for _ in range(max(1, max_inflight)):
            if not _submit_next():
                break
        # Drain + refill: keep at most max_inflight tasks outstanding.
        while pending:
            done, keep = wait(pending, return_when=FIRST_COMPLETED)
            pending = keep
            for fut in done:
                item = future_item.pop(fut)
                _submit_next()  # refill to keep the window full
                yield item, fut

# -----------------------------------
# Fast discovery (threads + progress)
# -----------------------------------
def _find_sc_for_subdir(root_dir: str, entry: os.DirEntry, strict: bool = False) -> Optional[Tuple[str, str]]:
    """
    Prefer <subdir>/<subdir>.sc; if strict=False, fallback to first *.sc in subdir.
    """
    dname = entry.name
    dpath = entry.path
    preferred = os.path.join(dpath, f"{dname}.sc")
    if os.path.exists(preferred):
        return (dname, preferred)
    if strict:
        return None
    try:
        with os.scandir(dpath) as it:
            for e in it:
                if e.is_file() and e.name.endswith(".sc"):
                    return (dname, e.path)
    except Exception:
        pass
    return None

def discover_sc_files_threaded(root_dir: str, workers: Optional[int] = None,
                               progress_every: int = 10000, strict: bool = False,
                               max_inflight: Optional[int] = None
                               ) -> Tuple[List[Tuple[str, str]], int, List[str]]:
    """
    Bounded threaded discovery over immediate subdirectories with ETA prints.
    Returns: (list_of_pairs, total_subdirs, missing_subdir_paths)
    """
    t0 = time.time()
    subdir_entries: List[os.DirEntry] = []
    with os.scandir(root_dir) as it:
        for e in it:
            if e.is_dir():
                subdir_entries.append(e)
    total = len(subdir_entries)

    if workers is None:
        workers = default_worker_count()
    if max_inflight is None:
        max_inflight = default_max_inflight(workers)

    print(f"[Discovery] Scanning {total} subdirectories with {workers} workers "
          f"(max in-flight {max_inflight})…")
    found: List[Tuple[str, str]] = []
    missing: List[str] = []
    milestones = {1, 10, 100, 1000, 10000}
    parsed = 0

    fn = partial(_find_sc_for_subdir, root_dir, strict=strict)
    for entry, fut in bounded_imap_unordered(fn, subdir_entries, workers, max_inflight):
        res = fut.result()
        if res is not None:
            found.append(res)
        else:
            missing.append(entry.path)
        parsed += 1
        if parsed in milestones or (parsed >= 10000 and parsed % progress_every == 0):
            elapsed = time.time() - t0
            rate = parsed / elapsed if elapsed > 0 else 0.0
            remaining = total - parsed
            eta = remaining / rate if rate > 0 else float("inf")
            mem_mb = get_mem_used_mb()
            print(f"  [Discovery] {parsed}/{total} | elapsed {fmt_secs(elapsed)} "
                  f"| ~{rate:.1f} subdirs/s | ETA {fmt_secs(eta)} | RSS ~{mem_mb:.1f} MB")

    elapsed = time.time() - t0
    print(f"[Discovery] Done: {len(found)} .sc files found, {len(missing)} missing "
          f"(from {total} subdirs) in {fmt_secs(elapsed)}.")
    return found, total, missing

def _extract_outscr_fast(line: str) -> Optional[str]:
    """
    Fast path for notebook-generated commands, where --outscr is an unquoted
    path followed by another option. Avoid parsing very large JSON arguments
    with shlex on every line.
    """
    marker = " --outscr "
    idx = line.find(marker)
    if idx >= 0:
        rest = line[idx + len(marker):].lstrip()
        if not rest or rest[0] in {"'", '"'}:
            return None
        end = rest.find(" --")
        return (rest if end < 0 else rest[:end]).strip() or None

    marker = " --outscr="
    idx = line.find(marker)
    if idx >= 0:
        rest = line[idx + len(marker):].lstrip()
        if not rest or rest[0] in {"'", '"'}:
            return None
        end = rest.find(" --")
        return (rest if end < 0 else rest[:end]).strip() or None

    return None

def _extract_outscr_shlex(line: str) -> Optional[str]:
    parts = shlex.split(line)
    for i, part in enumerate(parts):
        if part == "--outscr":
            if i + 1 < len(parts):
                return parts[i + 1]
            return None
        if part.startswith("--outscr="):
            return part.split("=", 1)[1]
    return None

def sc_files_from_process_cmds(commands_path: str) -> Tuple[List[Tuple[str, str]], int, List[Tuple[int, str]]]:
    """
    Read an AF3 PDB-process commands file and extract the expected .sc output
    path from each --outscr argument.

    This is much faster than discovering .sc files by scanning an AF3 output
    directory containing tens of thousands of subdirectories. It intentionally
    does not os.path.exists() every output path; the normal read pass opens each
    .sc once and handles missing/bad files there.
    """
    sc_files: List[Tuple[str, str]] = []
    bad_lines: List[Tuple[int, str]] = []
    total_lines = 0

    with open(commands_path, "r", encoding="utf-8", errors="replace") as fh:
        for line_no, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line:
                continue
            total_lines += 1
            outscr = _extract_outscr_fast(line)
            if outscr is None:
                try:
                    outscr = _extract_outscr_shlex(line)
                except ValueError:
                    bad_lines.append((line_no, "could not parse with shlex fallback"))
                    continue

            if not outscr:
                bad_lines.append((line_no, "missing --outscr"))
                continue

            subdir = os.path.basename(os.path.dirname(outscr.rstrip("/")))
            if not subdir:
                bad_lines.append((line_no, f"could not derive subdir from --outscr={outscr!r}"))
                continue
            sc_files.append((subdir, outscr))

    return sc_files, total_lines, bad_lines

# ------------------------------------------------
# CSV helpers (each file: header row + data row)
# ------------------------------------------------
def read_csv_header(sc_path: str) -> List[str]:
    with open(sc_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and any(cell.strip() for cell in row):
                return [c.strip() for c in row]
    return []

def read_csv_data_row(sc_path: str) -> Optional[List[str]]:
    """
    Return the FIRST non-empty row AFTER the header row; ignore extras.
    """
    with open(sc_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        header_seen = False
        for row in reader:
            if not row or not any(cell.strip() for cell in row):
                continue
            if not header_seen:
                header_seen = True
                continue
            return [c.strip() for c in row]
    return None

def read_header_and_data(sc_path: str) -> Tuple[List[str], Optional[List[str]]]:
    """Read header and first data row in a single file open (used by --fast pipeline)."""
    with open(sc_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        header = None
        for row in reader:
            if not row or not any(cell.strip() for cell in row):
                continue
            if header is None:
                header = [c.strip() for c in row]
                continue
            return header, [c.strip() for c in row]
    return header or [], None

# ---------------------------------------
# Column ordering helper
# ---------------------------------------
def compute_union_from_headers(colset: set, first_col: str = "af3_models_dir") -> List[str]:
    """Order columns: first_col (or legacy 'description') first, sorted middle, subdir+sc_path last."""
    colset = set(colset)  # copy to avoid mutating caller's set
    colset.update(["subdir", "sc_path"])
    ordered: List[str] = []
    if first_col in colset:
        ordered.append(first_col)
        colset.remove(first_col)
    elif "description" in colset:
        ordered.append("description")
        colset.remove("description")
    colset.discard("subdir")
    colset.discard("sc_path")
    ordered += sorted(colset)
    ordered += ["subdir", "sc_path"]
    return ordered

# ---------------------------------------
# Union-of-columns (threaded + progress)
# ---------------------------------------
def union_columns_threaded(sc_files: List[Tuple[str, str]], workers: int,
                           first_col: str = "af3_models_dir",
                           max_inflight: Optional[int] = None) -> List[str]:
    """
    Build global union of columns by reading the header of each CSV in parallel.
    Uses a bounded in-flight window so memory stays flat regardless of how many
    files there are (instead of holding one future per file in RAM at once).
    """
    t0 = time.time()
    colset = set()
    milestones = {1, 10, 100, 1000, 10000}
    done = 0
    total = len(sc_files)
    if total == 0:
        return ["subdir", "sc_path"]
    if max_inflight is None:
        max_inflight = default_max_inflight(workers)

    print(f"[Header-Union] Parsing headers from {total} files with {workers} workers "
          f"(max in-flight {max_inflight})…")
    for (_sub, _scp), fut in bounded_imap_unordered(
            lambda pair: read_csv_header(pair[1]), sc_files, workers, max_inflight):
        try:
            cols = fut.result()
        except Exception:
            cols = []
        colset.update(cols)
        done += 1
        if done in milestones or (done >= 10000 and done % 10000 == 0):
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (total - done) / rate if rate > 0 else float("inf")
            mem_mb = get_mem_used_mb()
            print(f"  [Header-Union] {done}/{total} | elapsed {fmt_secs(elapsed)} "
                  f"| ~{rate:.1f} files/s | ETA {fmt_secs(eta)} | RSS ~{mem_mb:.1f} MB")

    ordered = compute_union_from_headers(colset, first_col=first_col)
    print(f"[Header-Union] Done in {fmt_secs(time.time()-t0)}. Global union columns = {len(ordered)}.")
    return ordered

# -----------------------
# Shard writing / concat
# -----------------------
def remap_row_to_union(vals: List[str], header: List[str], union_cols: List[str],
                       subdir: str, sc_path: str,
                       union_col_set: Optional[set] = None) -> Dict[str, str]:
    """
    Map a CSV data row to the global union schema; fill missing with "".
    """
    if union_col_set is None:
        union_col_set = set(union_cols)
    row: Dict[str, str] = {col: "" for col in union_cols}
    # Map overlapping header -> values
    for i, col in enumerate(header):
        if i < len(vals) and col in union_col_set:
            row[col] = vals[i]
    # Provenance
    row["subdir"] = subdir
    row["sc_path"] = sc_path
    return row

def shard_write(rows: List[Dict[str, str]], union_cols: List[str], shard_idx: int, tmpdir: str) -> str:
    shard_path = os.path.join(tmpdir, f"shard_{shard_idx:06d}.csv")
    with open(shard_path, "w", newline="") as wf:
        writer = csv.DictWriter(wf, fieldnames=union_cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [Shard] Wrote {len(rows):>6} rows → {shard_path}")
    return shard_path

RAW_FIELDNAMES = ["header_id", "subdir", "sc_path", "values_json"]

def raw_shard_write(rows: List[Dict[str, str]], shard_idx: int, tmpdir: str) -> str:
    raw_path = os.path.join(tmpdir, f"raw_{shard_idx:06d}.csv")
    with open(raw_path, "w", newline="") as wf:
        writer = csv.DictWriter(wf, fieldnames=RAW_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [Raw-Shard] Wrote {len(rows):>6} rows → {raw_path}")
    return raw_path

def shard_write_values(rows: List[List[str]], union_cols: List[str], shard_idx: int, tmpdir: str) -> str:
    shard_path = os.path.join(tmpdir, f"shard_{shard_idx:06d}.csv")
    with open(shard_path, "w", newline="") as wf:
        writer = csv.writer(wf)
        writer.writerow(union_cols)
        writer.writerows(rows)
    print(f"  [Shard] Wrote {len(rows):>6} rows → {shard_path}")
    return shard_path

def stream_concat_csvs(shard_paths: List[str], out_csv: str):
    if not shard_paths:
        open(out_csv, "w").close()
        return
    with open(out_csv, "w", newline="") as out_f:
        with open(shard_paths[0], "r", newline="") as first:
            shutil.copyfileobj(first, out_f)  # header + data
        for p in shard_paths[1:]:
            with open(p, "r", newline="") as f:
                next(f, None)  # drop header
                shutil.copyfileobj(f, out_f)

def remap_values_to_union(vals: List[str], header_id: int, headers_by_id: List[List[str]],
                          union_cols: List[str], union_index: Dict[str, int],
                          mapping_cache: Dict[int, List[Tuple[int, int]]],
                          subdir: str, sc_path: str) -> List[str]:
    """
    Return a row list already ordered like union_cols.

    Header-to-union index mappings are cached because AF3 .sc headers repeat
    heavily across files, and list rows are faster for csv.writer than dict
    rows for wide tables.
    """
    mapping = mapping_cache.get(header_id)
    if mapping is None:
        header = headers_by_id[header_id]
        mapping = [(src_i, union_index[col]) for src_i, col in enumerate(header) if col in union_index]
        mapping_cache[header_id] = mapping

    out = [""] * len(union_cols)
    for src_i, dst_i in mapping:
        if src_i < len(vals):
            out[dst_i] = vals[src_i]
    out[union_index["subdir"]] = subdir
    out[union_index["sc_path"]] = sc_path
    return out

def remap_raw_shards_to_union(raw_shard_paths: List[str], headers_by_id: List[List[str]],
                              union_cols: List[str], chunk_rows: int, tmpdir: str,
                              expected_rows: int) -> Tuple[List[str], int]:
    print(f"\n[Pass 2] Remapping {len(raw_shard_paths)} raw shard(s) to final union columns…")
    t0 = time.time()
    milestones = {1, 10, 100, 1000, 10000}
    union_index = {col: i for i, col in enumerate(union_cols)}
    mapping_cache: Dict[int, List[Tuple[int, int]]] = {}
    rows_buffer: List[List[str]] = []
    final_shard_paths: List[str] = []
    shard_idx = 0
    done = 0

    for raw_path in raw_shard_paths:
        with open(raw_path, "r", newline="") as rf:
            reader = csv.DictReader(rf)
            for raw in reader:
                header_id = int(raw["header_id"])
                vals = json.loads(raw["values_json"])
                rows_buffer.append(remap_values_to_union(
                    vals=vals,
                    header_id=header_id,
                    headers_by_id=headers_by_id,
                    union_cols=union_cols,
                    union_index=union_index,
                    mapping_cache=mapping_cache,
                    subdir=raw["subdir"],
                    sc_path=raw["sc_path"],
                ))

                if len(rows_buffer) >= chunk_rows:
                    final_shard_paths.append(shard_write_values(rows_buffer, union_cols, shard_idx, tmpdir))
                    shard_idx += 1
                    rows_buffer.clear()

                done += 1
                if done in milestones or (done >= 10000 and done % 10000 == 0):
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0.0
                    eta = (expected_rows - done) / rate if rate > 0 else float("inf")
                    mem_mb = get_mem_used_mb()
                    print(f"  [Pass 2] {done}/{expected_rows} | elapsed {fmt_secs(elapsed)} "
                          f"| ~{rate:.1f} rows/s | ETA {fmt_secs(eta)} "
                          f"| Shards: {shard_idx} | RSS ~{mem_mb:.1f} MB")

    if rows_buffer:
        final_shard_paths.append(shard_write_values(rows_buffer, union_cols, shard_idx, tmpdir))
        shard_idx += 1
        rows_buffer.clear()

    elapsed = time.time() - t0
    print(f"[Pass 2] Done in {fmt_secs(elapsed)}: "
          f"{shard_idx} final shard(s), {done} rows, {len(mapping_cache)} unique header mapping(s).")
    return final_shard_paths, done

# -----------------------------------------------------------------
# Default pipeline: single remote read + bounded threaded parse
# -----------------------------------------------------------------
def run_single_read_pipeline(sc_files: List[Tuple[str, str]], workers: int,
                             chunk_rows: int, first_col: str, tmpdir: str,
                             max_inflight: Optional[int] = None
                             ) -> Tuple[List[str], int, int]:
    """
    Read each remote .sc exactly once, writing compact raw shards while building
    the global column union. A second pass over the temporary raw shards remaps
    rows to the final union schema.

    This trades cheap sequential temp I/O for avoiding a second full pass of
    NFS opens against tens of thousands of tiny .sc files.
    """
    total = len(sc_files)
    milestones = {1, 10, 100, 1000, 10000}
    if max_inflight is None:
        max_inflight = default_max_inflight(workers)

    print(f"\n[Pass 1] Reading headers+data once from {total} files with {workers} workers "
          f"(max in-flight {max_inflight})…")
    t0 = time.time()
    colset = set()
    header_id_by_tuple: Dict[Tuple[str, ...], int] = {}
    headers_by_id: List[List[str]] = []
    raw_rows: List[Dict[str, str]] = []
    raw_shard_paths: List[str] = []
    raw_shard_idx = 0
    parsed_files = 0
    total_rows = 0
    skipped = 0

    for (sub, scp), fut in bounded_imap_unordered(
            lambda pair: read_header_and_data(pair[1]), sc_files, workers, max_inflight):
        try:
            header, data_row = fut.result()
        except Exception as exc:
            print(f"  [Error] {scp}: {exc}")
            skipped += 1
            parsed_files += 1
            continue

        if not header or data_row is None:
            print(f"  [Skip] No usable data in: {scp}")
            skipped += 1
            parsed_files += 1
            continue

        if len(data_row) < len(header):
            data_row = data_row + [""] * (len(header) - len(data_row))

        header_tuple = tuple(header)
        header_id = header_id_by_tuple.get(header_tuple)
        if header_id is None:
            header_id = len(headers_by_id)
            header_id_by_tuple[header_tuple] = header_id
            headers_by_id.append(header)
            colset.update(header)

        raw_rows.append({
            "header_id": header_id,
            "subdir": sub,
            "sc_path": scp,
            "values_json": json.dumps(data_row, separators=(",", ":")),
        })
        total_rows += 1

        if len(raw_rows) >= chunk_rows:
            raw_shard_paths.append(raw_shard_write(raw_rows, raw_shard_idx, tmpdir))
            raw_shard_idx += 1
            raw_rows.clear()

        parsed_files += 1
        if parsed_files in milestones or (parsed_files >= 10000 and parsed_files % 10000 == 0):
            elapsed = time.time() - t0
            rate = parsed_files / elapsed if elapsed > 0 else 0.0
            eta = (total - parsed_files) / rate if rate > 0 else float("inf")
            mem_mb = get_mem_used_mb()
            print(f"  [Pass 1] {parsed_files}/{total} | elapsed {fmt_secs(elapsed)} "
                  f"| ~{rate:.1f} files/s | ETA {fmt_secs(eta)} "
                  f"| Raw shards: {raw_shard_idx} | Rows: {total_rows} "
                  f"| Unique headers: {len(headers_by_id)} | RSS ~{mem_mb:.1f} MB")

    if raw_rows:
        raw_shard_paths.append(raw_shard_write(raw_rows, raw_shard_idx, tmpdir))
        raw_shard_idx += 1
        raw_rows.clear()

    union_cols = compute_union_from_headers(colset, first_col=first_col)
    elapsed = time.time() - t0
    print(f"[Pass 1] Done in {fmt_secs(elapsed)}: "
          f"{raw_shard_idx} raw shard(s), {total_rows} rows, {skipped} skipped, "
          f"{len(headers_by_id)} unique header(s).")
    print(f"[Pass 1] Global union columns: {len(union_cols)}")
    if union_cols:
        preview = ", ".join(union_cols[:min(12, len(union_cols))])
        print(f"[Pass 1] Column preview: {preview}{' …' if len(union_cols) > 12 else ''}")

    _final_shard_paths, remapped_rows = remap_raw_shards_to_union(
        raw_shard_paths=raw_shard_paths,
        headers_by_id=headers_by_id,
        union_cols=union_cols,
        chunk_rows=chunk_rows,
        tmpdir=tmpdir,
        expected_rows=total_rows,
    )
    if remapped_rows != total_rows:
        print(f"  [Warning] Raw remap row count mismatch: parsed {total_rows}, remapped {remapped_rows}")

    return union_cols, parsed_files, total_rows

# -----------------------------------------------------------------
# Legacy two-pass pipeline: threaded + bounded
# -----------------------------------------------------------------
def run_bounded_pipeline(sc_files: List[Tuple[str, str]], workers: int,
                         chunk_rows: int, first_col: str, tmpdir: str,
                         max_inflight: Optional[int] = None
                         ) -> Tuple[List[str], int, int]:
    """
    Legacy pipeline: threaded header union (Pass 1) + threaded data read
    (Pass 2), BOTH using a bounded in-flight window. Peak memory is
    O(max_inflight + chunk_rows) — flat regardless of file count — while
    keeping the speedup of threading.

    Output is identical to the --low_memory pipeline: same rows, same columns.
    Row order is completion-order (not stable run-to-run), exactly as the
    legacy threaded path behaved; the data set is unchanged.

    Returns (union_cols, parsed_files, total_rows).
    """
    total = len(sc_files)
    milestones = {1, 10, 100, 1000, 10000}
    if max_inflight is None:
        max_inflight = default_max_inflight(workers)

    # Pass 1: header scan → column union (threaded, bounded)
    print(f"\n[Pass 1] Building global column union from {total} headers "
          f"with {workers} workers (max in-flight {max_inflight})…")
    union_cols = union_columns_threaded(sc_files, workers=workers,
                                        first_col=first_col, max_inflight=max_inflight)
    union_col_set = set(union_cols)
    print(f"[Pass 1] Global union columns: {len(union_cols)}")
    if union_cols:
        preview = ", ".join(union_cols[:min(12, len(union_cols))])
        print(f"[Pass 1] Column preview: {preview}{' …' if len(union_cols) > 12 else ''}")

    # Pass 2: data read (header+data in one open) → streaming shard writes,
    # with a bounded in-flight window so only ~max_inflight results are held.
    print(f"\n[Pass 2] Reading data from {total} files with {workers} workers "
          f"(threaded, max in-flight {max_inflight})…")
    t0 = time.time()
    rows_buffer: List[Dict[str, str]] = []
    shard_idx = 0
    total_rows = 0
    parsed_files = 0
    skipped = 0

    for (sub, scp), fut in bounded_imap_unordered(
            lambda pair: read_header_and_data(pair[1]), sc_files, workers, max_inflight):
        try:
            header, data_row = fut.result()
        except Exception as exc:
            print(f"  [Error] {scp}: {exc}")
            skipped += 1
            parsed_files += 1
            continue

        if not header or data_row is None:
            print(f"  [Skip] No usable data in: {scp}")
            skipped += 1
            parsed_files += 1
            continue

        # Pad if data shorter than header
        if len(data_row) < len(header):
            data_row = data_row + [""] * (len(header) - len(data_row))

        # Remap immediately (no buffering of raw header/data)
        row = remap_row_to_union(data_row, header, union_cols, sub, scp, union_col_set=union_col_set)
        rows_buffer.append(row)

        if len(rows_buffer) >= chunk_rows:
            shard_write(rows_buffer, union_cols, shard_idx, tmpdir)
            shard_idx += 1
            total_rows += len(rows_buffer)
            rows_buffer.clear()

        parsed_files += 1
        if parsed_files in milestones or (parsed_files >= 10000 and parsed_files % 10000 == 0):
            elapsed = time.time() - t0
            rate = parsed_files / elapsed if elapsed > 0 else 0.0
            eta = (total - parsed_files) / rate if rate > 0 else float("inf")
            mem_mb = get_mem_used_mb()
            print(f"  [Pass 2] {parsed_files}/{total} | elapsed {fmt_secs(elapsed)} "
                  f"| ~{rate:.1f} files/s | ETA {fmt_secs(eta)} "
                  f"| Shards: {shard_idx} | Rows: {total_rows} | RSS ~{mem_mb:.1f} MB")

    # Flush remaining
    if rows_buffer:
        shard_write(rows_buffer, union_cols, shard_idx, tmpdir)
        shard_idx += 1
        total_rows += len(rows_buffer)
        rows_buffer.clear()

    elapsed = time.time() - t0
    print(f"[Pass 2] Done in {fmt_secs(elapsed)}: "
          f"{shard_idx} shard(s), {total_rows} rows, {skipped} skipped.")

    return union_cols, parsed_files, total_rows

# -------------
# Main driver
# -------------
def main():
    parser = argparse.ArgumentParser(description="Concatenate AF3 .sc CSV files (one per subdir) into a single CSV.")
    parser.add_argument("--af3_dir_of_subdirs", required=True,
                        help="Directory whose immediate subdirectories each contain a .sc (CSV: header + data).")
    parser.add_argument("--optional_path_for_summary_stats", default=None,
                        help="Optional final output CSV path; otherwise writes zzzzz_af3_analysis_csv_zzzzz.csv in the root dir.")
    parser.add_argument("--chunk_rows", type=int, default=None,
                        help="Rows per temporary shard CSV (default 10,000; 20,000 with --nfs_friendly).")
    parser.add_argument("--workers", type=int, default=None,
                        help="Threads for discovery and parsing (default ≈ min(64, 2*CPU); capped at 8 with --nfs_friendly).")
    parser.add_argument("--strict_name_match", action="store_true",
                        help="Only accept <subdir>/<subdir>.sc; skip fallback *.sc scan for speed.")
    parser.add_argument("--assume_preferred_sc_name", action="store_true",
                        help="Alias for --strict_name_match; use when every subdir should contain <subdir>/<subdir>.sc.")
    parser.add_argument("--nfs_friendly", action="store_true",
                        help="Use conservative defaults for shared NFS trees: workers<=8, max_inflight=64, "
                             "chunk_rows=20000, and preferred-name-only .sc discovery.")
    parser.add_argument("--find_subdirs_without_viable_sc", action="store_true", help="Only perform discovery and print subdirectories lacking a usable .sc, then exit.")
    parser.add_argument("--first_col", type=str, default="af3_models_dir",
                        help="Column name to place first in output (default: af3_models_dir; falls back to 'description' for legacy .sc files).")
    parser.add_argument("--sc_paths_from_process_cmds", default=None,
                        help="AF3 PDB-process commands file to use as a manifest of expected .sc paths "
                             "(extracts --outscr from each command and skips directory discovery).")
    parser.add_argument("--tmpdir_base", default=None,
                        help="Directory under which temporary concat shards should be created. "
                             "Use node-local scratch for Slurm runs. Default: af3_dir_of_subdirs.")
    parser.add_argument("--preserve_input_order", action="store_true",
                        help="Keep manifest/discovery order. By default, .sc paths are sorted before parsing "
                             "to avoid latency-biased discovery order.")
    parser.add_argument("--low_memory", action="store_true",
                        help="Use the legacy single-threaded streaming Pass 2 (minimal, near-constant "
                             "RAM, but slower). Use for extreme scale or very memory-limited nodes. "
                             "The default is now the single-read threaded pipeline.")
    parser.add_argument("--max_inflight", type=int, default=None,
                        help="Max discovery/parse tasks in flight (bounds RAM and NFS pressure). "
                             "Default=max(2*workers, 64), or 64 with --nfs_friendly.")
    parser.add_argument("--fast", action="store_true",
                        help="DEPRECATED no-op: the single-read threaded pipeline is now the default. "
                             "Accepted for backward compatibility; ignored.")
    args = parser.parse_args()

    root = os.path.abspath(args.af3_dir_of_subdirs.rstrip("/"))
    out_csv = args.optional_path_for_summary_stats or os.path.join(root, "zzzzz_af3_analysis_csv_zzzzz.csv")
    workers = args.workers if args.workers is not None else default_worker_count(args.nfs_friendly)
    max_inflight = args.max_inflight if args.max_inflight is not None else default_max_inflight(workers, args.nfs_friendly)
    chunk_rows = max(1, args.chunk_rows if args.chunk_rows is not None else (20000 if args.nfs_friendly else 10000))
    preferred_name_only = args.strict_name_match or args.assume_preferred_sc_name or args.nfs_friendly

    print("############################################")
    print("### AF3 .sc CONCAT — MASSIVE SCALE MODE  ###")
    print("############################################")
    print(f"Root directory        : {root}")
    print(f"Output path           : {out_csv}")
    print(f"CPU cores (available) : {get_cpu_count()}")
    print(f"NFS-friendly preset   : {args.nfs_friendly}")
    print(f"Workers (resolved)    : {workers}")
    print(f"Preferred-name only   : {preferred_name_only}")
    print(f"Rows per shard        : {chunk_rows}")
    print(f"Pipeline              : {'low-memory (legacy serial Pass 2)' if args.low_memory else 'single-read threaded + bounded (default)'}")
    print(f"Max in-flight         : {max_inflight}")
    if args.sc_paths_from_process_cmds:
        print(f"SC manifest           : {os.path.abspath(args.sc_paths_from_process_cmds)}")
    if args.tmpdir_base:
        print(f"Temp base             : {os.path.abspath(args.tmpdir_base)}")
    print("--------------------------------------------")

    t0 = time.time()

    # PASS 0: Resolve candidate .sc files.
    if args.sc_paths_from_process_cmds:
        manifest_path = os.path.abspath(args.sc_paths_from_process_cmds)
        sc_files, total_subdirs, bad_lines = sc_files_from_process_cmds(manifest_path)
        print(f"[Manifest] Loaded {len(sc_files)} expected .sc path(s) from {manifest_path}")
        print(f"[Manifest] Parsed {total_subdirs} non-empty command line(s).")
        if bad_lines:
            print(f"[Manifest] Skipped {len(bad_lines)} malformed command line(s); first 25:")
            for line_no, reason in bad_lines[:25]:
                print(f"  [Malformed] line {line_no}: {reason}")
        missing = []
    else:
        # Directory discovery is intentionally still available for older
        # workflows that do not have an AF3 process command manifest.
        sc_files, total_subdirs, missing = discover_sc_files_threaded(
            root, workers=workers, strict=preferred_name_only, max_inflight=max_inflight)
        print(f"Discovered {total_subdirs} subdirectories.")
        if preferred_name_only:
            print(f"Found {len(sc_files)} usable .sc files (preferred '<d>.sc' only).")
        else:
            print(f"Found {len(sc_files)} usable .sc files (preferred '<d>.sc' else first '*.sc').")
        print(f"Missing .sc in {len(missing)} subdir(s). Listing them below:")
        for p in missing:
            print(f"  [Missing] {p}")

    # If the user only wants the missing-list, exit early.
    if args.find_subdirs_without_viable_sc:
        print("\n[Exit-by-flag] Completed discovery-only run (--find_subdirs_without_viable_sc).")
        return

    if not sc_files:
        print("[Exit] No .sc files found. Nothing to do.")
        return

    if not args.preserve_input_order:
        sc_files.sort(key=lambda pair: (pair[0].lower(), pair[1].lower()))
        print(f"[Order] Sorted {len(sc_files)} .sc path(s) before parsing.")
    else:
        print(f"[Order] Preserving input order for {len(sc_files)} .sc path(s).")

    # Prepare temp shard dir
    tmpdir_parent = os.path.abspath(args.tmpdir_base) if args.tmpdir_base else root
    os.makedirs(tmpdir_parent, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix=".af3_concat_", dir=tmpdir_parent)
    print(f"\n[Temp] Will write shard CSVs under: {tmpdir}")

    if args.fast:
        print("[Note] --fast is deprecated and ignored: the single-read threaded pipeline is now the default.")

    if not args.low_memory:
        # =============================================
        # DEFAULT PIPELINE: single remote read + bounded parse
        # =============================================
        union_cols, parsed_files, total_rows = run_single_read_pipeline(
            sc_files, workers=workers, chunk_rows=chunk_rows,
            first_col=args.first_col, tmpdir=tmpdir, max_inflight=max_inflight)
    else:
        # =============================================
        # LEGACY LOW-MEMORY PIPELINE: threaded header union + SERIAL Pass 2
        # =============================================
        # PASS 1: Compute union-of-columns (header-only, parallel + ETA)
        print("\n[Pass 1] Building global column union from headers…")
        union_cols = union_columns_threaded(sc_files, workers=workers,
                                             first_col=args.first_col, max_inflight=max_inflight)
        union_col_set = set(union_cols)
        print(f"[Pass 1] Global union columns: {len(union_cols)}")
        if union_cols:
            preview = ", ".join(union_cols[:min(12, len(union_cols))])
            print(f"[Pass 1] Column preview: {preview}{' …' if len(union_cols) > 12 else ''}")

        # PASS 2: Parse all files → write shard CSVs
        milestones = {1, 10, 100, 1000, 10000}
        def is_milestone(n: int) -> bool:
            return (n in milestones) or (n >= 10000 and n % 10000 == 0)

        parsed_files = 0
        total_rows  = 0
        shard_idx   = 0
        rows_buffer: List[Dict[str, str]] = []

        print("\n[Pass 2] Parsing data rows and writing shard CSVs…")
        t2 = time.time()
        for subdir, scp in sc_files:
            header = read_csv_header(scp)
            data_row = read_csv_data_row(scp)
            if not header or data_row is None:
                # Malformed or empty; skip but log
                print(f"  [Skip] No usable data in: {scp}")
                parsed_files += 1
                continue

            # Normalize length mismatch (pad with empty if data shorter)
            if len(data_row) < len(header):
                data_row = data_row + [""] * (len(header) - len(data_row))

            row = remap_row_to_union(data_row, header, union_cols, subdir, scp, union_col_set=union_col_set)
            rows_buffer.append(row)

            if len(rows_buffer) >= chunk_rows:
                shard_write(rows_buffer, union_cols, shard_idx, tmpdir)
                shard_idx += 1
                total_rows += len(rows_buffer)
                rows_buffer.clear()

            parsed_files += 1
            if is_milestone(parsed_files):
                elapsed = time.time() - t2
                avg_per_file = elapsed / max(1, parsed_files)
                mem_mb = get_mem_used_mb()
                print(f"  [Pass 2] Parsed {parsed_files} files | Elapsed: {fmt_secs(elapsed)} "
                      f"| Avg/file: {avg_per_file:.4f}s | Shards: {shard_idx} "
                      f"| Rows (written so far): {total_rows} | RSS ~{mem_mb:.1f} MB")

        # Flush any remaining rows
        if rows_buffer:
            shard_write(rows_buffer, union_cols, shard_idx, tmpdir)
            shard_idx += 1
            total_rows += len(rows_buffer)
            rows_buffer.clear()

    # FINAL: Concatenate shards
    print("\n[Final] Concatenating shard CSVs into final output…")
    shard_paths = sorted(glob.glob(os.path.join(tmpdir, "shard_*.csv")))
    print(f"[Final] {len(shard_paths)} shard(s) to merge.")
    t_concat0 = time.time()
    stream_concat_csvs(shard_paths, out_csv)
    print(f"[Final] Concatenation done in {fmt_secs(time.time() - t_concat0)}.")

    # Clean up shards
    try:
        shutil.rmtree(tmpdir)
        print(f"[Temp] Removed temporary shard directory: {tmpdir}")
    except Exception as e:
        print(f"[Temp] Could not remove {tmpdir}: {e}")

    # SUMMARY
    elapsed = time.time() - t0
    # Count final rows/cols quickly by reading header line + counting remaining lines
    n_cols = 0
    n_rows = 0
    try:
        with open(out_csv, "r", newline="") as f:
            header_line = f.readline()
            n_cols = header_line.count(",") + 1 if header_line.strip() else 0
            n_rows = sum(1 for line in f if line.strip())
    except Exception:
        n_rows = total_rows
        n_cols = len(union_cols)

    print("\n============================================")
    print("                  SUMMARY")
    print("============================================")
    print(f"Total subdirectories     : {total_subdirs}")
    print(f"Total .sc files found    : {len(sc_files)}")
    print(f"Total .sc files parsed   : {parsed_files}")
    print(f"Total rows in output     : {n_rows}")
    print(f"Total columns in output  : {n_cols}")
    print(f"Output CSV               : {out_csv}")
    print(f"Total elapsed            : {fmt_secs(elapsed)}")
    print(f"CPU cores (available)    : {get_cpu_count()}")
    print("============================================")

if __name__ == "__main__":
    main()
