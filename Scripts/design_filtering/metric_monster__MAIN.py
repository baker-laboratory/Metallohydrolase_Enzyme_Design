#!/usr/bin/env python3
"""
metric_monster__MAIN.py

Master wrapper to orchestrate multiple independent metric pipelines on one or more PDB files.

This script sequentially runs three specialized metric-generation tools:
  1) contact_counter__MAIN.py         → protein–ligand contact counts and summaries
  2) protein_size_shape_metrics__MAIN.py → core protein size/shape descriptors (Rg, anisotropy, etc.)
  3) execute_fpocket_on_holo_structure__MAIN.py → fpocket pocket detection and aggregation

Key Features:
  • **Modular Sub-Script Integration**: Pass custom flags to each tool via `--cc`, `--pss`, and `--fpock`.
  • **Optional Cleanup**: Keep or remove intermediate CSVs with `--keep-temp-csvs`.
  • **Automatic Merging**: After all pipelines complete, collate all `<base>*_*.csv` into `<base>_combined_metrics.csv`.
  • **Prefixing & Ordering**: Each pipeline’s columns are automatically prefixed (e.g. `contact_counter__`, `protein_SizeShape__`, `fpocket__`), then sorted alphabetically with `pdb_path` first.
  • **Parallel Execution**: Use `--run_parallel_in_terminal_on_pdb_dir <DIR>` and `--nproc N` to distribute jobs across multiple processes, with live progress and ETA.
  • **Quiet Mode**: Suppress sub-script stdout via `--quiet`, while still surfacing errors.

Usage Examples:
  # Single PDB run:
  python metric_monster__MAIN.py /path/to/structure.pdb \
      --cc --ligands LIG1 --pss --sphere-samples 1000 --fpock --min_spheres_per_pocket 20

  # Parallel over directory:
  python metric_monster__MAIN.py --run_parallel_in_terminal_on_pdb_dir /my/pdbs \
      --nproc 8 --keep-temp-csvs --quiet

Adding New Metric Pipelines:
  1) **Create** an independent `*_MAIN.py` that reads a PDB and outputs a single-row CSV named `<base>_<unique_suffix>.csv`.
  2) **Add** its path to this master script under the constants section:
       ```python
       NEW_MAIN = "/path/to/new_metrics__MAIN.py"
       ```
  3) **Extend** the `run_one()` function:
       - Insert a new step between (or after) existing calls, building and running the command:
         ```python
         new_cmd = [sys.executable, NEW_MAIN, pdb_file] + args.new_args
         subprocess.run(new_cmd, check=True, stdout=stdout_opt)
         ```
  4) **Map** its output filename to a prefix in `prefix_map`:
       ```python
       prefix_map['new_suffix'] = 'newPipeline__'
       ```
  5) **Expose** any extra flags through `argparse` (e.g. `--new-args`).

With these steps, `metric_monster__MAIN.py` will automatically include your new metrics in the merged output without altering its core logic.
"""
from pathlib import Path
import os
import sys
import glob
import argparse
import subprocess
import shlex
import multiprocessing as mp
import time
import pandas as pd

# Sibling helper scripts resolve next to this file, so the whole set can be
# relocated together.
_SCRIPT_DIR = Path(__file__).resolve().parent


# paths to main scripts
CONTACT_MAIN = str(_SCRIPT_DIR / "contact_counter__MAIN.py")
PS_MAIN      = str(_SCRIPT_DIR / "protein_size_shape_metrics__MAIN.py")
PF_MAIN      = str(_SCRIPT_DIR / "execute_fpocket_on_holo_structure__MAIN.py")


def run_one(pdb_file, args):
    base = os.path.splitext(os.path.basename(pdb_file))[0]
    print(f"[MASTER] Processing {pdb_file} ...")
    # prepare stdout suppression
    stdout_opt = subprocess.DEVNULL if args.quiet else None
    print()

    # 1) contact_counter
    print("############# --- [STARTING NEW METRIC STEP] --- #############")
    cc_cmd = [sys.executable, CONTACT_MAIN, pdb_file] + args.cc
    print(f"[MASTER] Running contact_counter: {' '.join(shlex.quote(x) for x in cc_cmd)}")
    try:
        subprocess.run(cc_cmd, check=True, stdout=stdout_opt)
    except subprocess.CalledProcessError as e:
        print(f"[MASTER] Warning: contact_counter failed on {base} ({e.returncode}), continuing.")
    print()

    # 2) protein size/shape metrics
    print("############# --- [STARTING NEW METRIC STEP] --- #############")
    ps_cmd = [sys.executable, PS_MAIN, pdb_file] + args.pss
    print(f"[MASTER] Running protein_size_shape_metrics: {' '.join(shlex.quote(x) for x in ps_cmd)}")
    try:
        subprocess.run(ps_cmd, check=True, stdout=stdout_opt)
    except subprocess.CalledProcessError as e:
        print(f"[MASTER] Warning: protein_size_shape_metrics failed on {base} ({e.returncode}), continuing.")
    print()

    # 3) fpocket
    print("############# --- [STARTING NEW METRIC STEP] --- #############")
    pf_cmd = [sys.executable, PF_MAIN, "--input_pdb", pdb_file] + args.fpock
    print(f"[MASTER] Running fpocket pipeline: {' '.join(shlex.quote(x) for x in pf_cmd)}")
    try:
        subprocess.run(pf_cmd, check=True, stdout=stdout_opt)
    except subprocess.CalledProcessError as e:
        print(f"[MASTER] Warning: fpocket pipeline failed on {base} ({e.returncode}), continuing.")
    # --- optional raw‐fpocket cleanup in the case of failure --- #
    if not args.keep_fpocket_raw:
        for ext in ('_info.csv', '_info.txt', '_out.pdb'):
            fp = f"{base}{ext}"
            if os.path.exists(fp):
                try:
                    os.remove(fp)
                    print(f"[MASTER] Deleted raw fpocket file {fp}")
                except Exception as e:
                    print(f"[MASTER] Warning: could not delete {fp}: {e}")
    print()

    # 4) combine CSVs with prefixes and pdb_path
    print("############# COMBINING CSVs NOW #############")
    temp_csvs = sorted(glob.glob(f"{base}*_*.csv"))
    combined_name = f"{base}_combined_metrics.csv"
    temp_csvs = [c for c in temp_csvs if os.path.basename(c) != os.path.basename(combined_name)]
    print(f"[MASTER] Combining {len(temp_csvs)} CSVs: {temp_csvs}")
    print()

    # mapping from a unique identifier to its prefix
    prefix_map = {
        'contacts_summary':      'contact_counter__',
        'protein_shapeSIZE':     'protein_SizeShape__',
        'info_aggregated':       'fpocket__',
    }

    dfs = []
    for c in temp_csvs:
        df = pd.read_csv(c)
        fname = os.path.basename(c)

        # try to find any key as substring of the filename
        chosen = None
        for key, pre in prefix_map.items():
            if key in fname:
                chosen = pre
                break

        if chosen is None:
            # fallback: warn and use no prefix
            print(f"[MASTER] Warning: no prefix found for {fname}, leaving columns unprefixed")
        else:
            df = df.add_prefix(chosen)

        dfs.append(df)

    df_all = pd.concat(dfs, axis=1)
    # add pdb_path column first
    df_all.insert(0, 'pdb_path', pdb_file)
    # reorder columns alphabetically, with pdb_path first
    cols = df_all.columns.tolist()
    rest = [col for col in cols if col != 'pdb_path']
    df_all = df_all[['pdb_path'] + sorted(rest)]

    df_all.to_csv(combined_name, index=False)
    print(f"[MASTER] Wrote combined metrics to {combined_name}")

    # 5) cleanup
    if not args.keep_temp_csvs:
        for c in temp_csvs:
            try:
                os.remove(c)
                print(f"[MASTER] Deleted temp CSV {c}")
            except Exception:
                print(f"[MASTER] Warning: failed to delete {c}")

def main():
    parser = argparse.ArgumentParser(__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('pdb', nargs='?', help='Single PDB file to process.')
    group.add_argument('--run_parallel_in_terminal_on_pdb_dir', metavar='DIR',
                       help='Directory of PDBs to process in parallel.')
    parser.add_argument('--nproc', type=int, default=mp.cpu_count(),
                        help='Number of parallel jobs when using --run_parallel (defaults to all available CPUs)')
    parser.add_argument('--keep-temp-csvs', action='store_true',
                        help='Do not delete temporary per-script CSVs.')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress sub-script STDOUT outputs (errors still shown).')
    parser.add_argument('--cc', nargs='*', default=[],
                        help='Extra flags to pass to contact_counter__MAIN.')
    parser.add_argument('--pss', nargs='*', default=[],
                        help='Extra flags to pass to protein_size_shape_metrics__MAIN.')
    parser.add_argument('--fpock', nargs='*', default=[],
                        help='Extra flags to pass to execute_fpocket_on_holo_structure__MAIN.')
    parser.add_argument('--keep-fpocket-raw', action='store_true',
                        help='Do NOT delete raw fpocket outputs (info.csv, info.txt, _out.pdb).')
    args = parser.parse_args()

    if args.run_parallel_in_terminal_on_pdb_dir:
        pdb_dir = args.run_parallel_in_terminal_on_pdb_dir
        pdb_files = sorted(glob.glob(os.path.join(pdb_dir, '*.pdb')))
        total = len(pdb_files)
        print(f"[MASTER] Parallel mode: found {total} PDBs in {pdb_dir}")

        processed = mp.Value('i', 0)
        lock = mp.Lock()
        start = time.time()

        def update_status(_):
            with lock:
                processed.value += 1
                n = processed.value
                # only print at 1, 10, 50, 100, then every 100 after 100
                if n in (1, 10, 50, 100) or (n > 100 and n % 100 == 0):
                    elapsed = time.time() - start
                    avg = elapsed / n
                    rem = avg * (total - n)
                    print()
                    print("#################################################################################################")
                    print("################################# !!! --- [TIME REPORT] --- !!! #################################")
                    print("#################################################################################################")
                    print(f"[MASTER] Progress: {n}/{total} | elapsed={elapsed:.1f}s | "
                          f"avg/file={avg:.1f}s | ETA={rem:.1f}s")
                    print()
                    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
                    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

        pool = mp.Pool(args.nproc)
        for pdb_file in pdb_files:
            pool.apply_async(
                run_one,
                args=(pdb_file, args),
                callback=update_status,
                error_callback=lambda e: print(f"[MASTER] Error: {e}")
            )
        pool.close()
        pool.join()
        # === MASTER MERGE OF ALL PER-PDB COMBINED_METRICS.CSV ===
        master_pattern = os.path.join(pdb_dir, '*_combined_metrics.csv')
        combined_files = sorted(glob.glob(master_pattern))
        master_name = 'Zzz___METRIC_MONSTER_DF___zzZ.csv'
        if combined_files:
            print(f"[MASTER] Merging {len(combined_files)} per-PDB combined metrics into {master_name}…")
            # read all, concat rows even if columns differ
            df_list = [pd.read_csv(f) for f in combined_files]
            df_master = pd.concat(df_list, axis=0, sort=False, ignore_index=True)
            df_master.to_csv(master_name, index=False)
            print(f"[MASTER] Wrote master metrics DataFrame to {master_name}")
            # cleanup per-pdb combined files
            if not args.keep_temp_csvs:
                for f in combined_files:
                    try:
                        os.remove(f)
                        #print(f"[MASTER] Deleted combined file {f}")
                    except Exception as e:
                        print(f"[MASTER] Warning: could not delete {f}: {e}")
        else:
            print("[MASTER] No per-PDB combined_metrics.csv files found to merge.")

        # report final tally
        print()
        print(f"[MASTER] Finished processing {processed.value}/{total} PDB files.")
        print("[MASTER] All parallel jobs complete.")
    else:
        run_one(args.pdb, args)

if __name__=='__main__':
    main()
