#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OPTIMIZED VERSION - Created 2026-01-14 by Seth Woodbury (woodbuse@uw.edu)

Optimized for processing 100k+ PDB files with:
- Multiprocessing (uses all CPU cores) - disable with --single_thread
- Pre-cached reference data (load each ref once)
- Progress tracking with ETA
- Milestone reporting (10, 100, 500, then every 1000)
- Debug mode (--debug) for detailed output like original script
- Chunked work distribution for better load balancing

Usage:
  # Multi-threaded - process all refs in directory (default)
  python add_remark666_lines_AND_rosetta_hydrogens_to_pdb_OPTIMIZED.py \
    --ref_pdb_dir /path/to/references \
    --output_pdb_dir /path/to/outputs \
    --final_output_dir /path/to/final_outputs \
    --find_suffixes_from_random_sample 10 \
    --n_workers 32 \
    --clobber

  # With manual suffix specification (skips auto-detection - FASTER for 100k+ files)
  python add_remark666_lines_AND_rosetta_hydrogens_to_pdb_OPTIMIZED.py \
    --ref_pdb_dir /path/to/references \
    --output_pdb_dir /path/to/outputs \
    --suffixes "" "_eV1_T0_10__1_1" "_eV1_T0_15__1_1" "_packed" \
    --clobber

  # Single ref PDB mode (for SLURM parallelization)
  python add_remark666_lines_AND_rosetta_hydrogens_to_pdb_OPTIMIZED.py \
    --single_ref_pdb /path/to/references/specific_ref.pdb \
    --output_pdb_dir /path/to/outputs \
    --suffixes "" "_eV1_T0_10__1_1" "_eV1_T0_15__1_1" \
    --clobber

  # Single-threaded with debug
  python add_remark666_lines_AND_rosetta_hydrogens_to_pdb_OPTIMIZED.py \
    --ref_pdb_dir /path/to/references \
    --output_pdb_dir /path/to/outputs \
    --single_thread --debug --clobber
"""

import os
import sys
import glob
import argparse
import time
import random
import multiprocessing as mp
from pathlib import Path
import tempfile
import shutil
import numpy as np
from datetime import timedelta

try:
    import pyrosetta
    import pyrosetta.rosetta
    import pyrosetta.distributed.io
except ImportError:
    print("ERROR: PyRosetta is not installed or not in PYTHONPATH")
    sys.exit(1)

# Try to import design_utils
try:
    # design_utils ships with this repository under Scripts/enzyme_design/utils.
    # $ENZYME_DESIGN_UTILS still overrides, for an external checkout.
    _utils = os.environ.get("ENZYME_DESIGN_UTILS") or str(
        Path(__file__).resolve().parent.parent / "enzyme_design" / "utils")
    sys.path.append(_utils)
    import design_utils
    HAS_DESIGN_UTILS = True
except ImportError:
    HAS_DESIGN_UTILS = False


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def check_pdb_has_remark666(pdb_path):
    with open(pdb_path, 'r') as f:
        for line in f:
            if "REMARK 666" in line:
                return True
    return False


def check_pdb_has_hydrogens(pdb_path):
    hydrogen_patterns = [' H ', ' 1H', ' 2H', ' 3H', 'HH', 'HG', 'HD', 'HE', 'HZ', 'HA', 'HB']
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                atom_name = line[12:16]
                if any(pattern in atom_name for pattern in hydrogen_patterns):
                    return True
    return False


def verify_pdb_is_ready(pdb_path):
    return check_pdb_has_remark666(pdb_path) and check_pdb_has_hydrogens(pdb_path)


def separate_protein_and_hetatm(pdb_content):
    protein_lines = []
    hetatm_lines = []
    for line in pdb_content:
        if line.startswith("HETATM"):
            hetatm_lines.append(line)
        elif line.startswith(("ATOM", "TER")):
            protein_lines.append(line)
    return protein_lines, hetatm_lines


def extract_non_hydrogen_coords_with_ids(pdb_lines):
    coords_dict = {}
    for line in pdb_lines:
        if line.startswith(("ATOM", "HETATM")):
            try:
                atom_name = line[12:16].strip()
                chain = line[21].strip()
                resno = int(line[22:26].strip())
                if atom_name.startswith('H') or (len(atom_name) > 0 and atom_name[0].isdigit() and 'H' in atom_name):
                    continue
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                key = (chain, resno, atom_name)
                coords_dict[key] = np.array([x, y, z])
            except (ValueError, IndexError):
                continue
    return coords_dict


def calculate_rmsd_from_dicts(coords_dict1, coords_dict2):
    common_keys = set(coords_dict1.keys()) & set(coords_dict2.keys())
    if len(common_keys) == 0:
        return 0.0, 0
    coords1_list = []
    coords2_list = []
    for key in sorted(common_keys):
        coords1_list.append(coords_dict1[key])
        coords2_list.append(coords_dict2[key])
    coords1_arr = np.array(coords1_list)
    coords2_arr = np.array(coords2_list)
    diff = coords1_arr - coords2_arr
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    return rmsd, len(common_keys)


def build_his_tautomer_map_from_raw_pdb(pdb_path, debug=False):
    his_map = {}
    his_atoms = {}
    with open(pdb_path, 'r') as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            try:
                atom_name = line[12:16].strip()
                resn = line[17:20].strip()
                chain = line[21].strip()
                resno = int(line[22:26].strip())
            except (ValueError, IndexError):
                continue
            if resn != "HIS":
                continue
            key = (chain, resno)
            if key not in his_atoms:
                his_atoms[key] = set()
            his_atoms[key].add(atom_name)

    for key, atoms in his_atoms.items():
        has_hd1 = "HD1" in atoms
        has_he2 = "HE2" in atoms
        if has_hd1 and not has_he2:
            his_map[key] = "HIS_D"
            if debug:
                print(f"[DEBUG] HIS {key[0]}{key[1]}: found HD1 only -> HIS_D")
        elif has_he2 and not has_hd1:
            his_map[key] = "HIS"
            if debug:
                print(f"[DEBUG] HIS {key[0]}{key[1]}: found HE2 only -> HIS")
        else:
            his_map[key] = "HIS"
            if debug:
                status = "both HD1 and HE2" if (has_hd1 and has_he2) else "neither HD1 nor HE2"
                print(f"[DEBUG] HIS {key[0]}{key[1]}: {status} -> defaulting to HIS")
    return his_map


def renumber_pdb_atoms(pdb_lines, start_number=1):
    renumbered = []
    atom_num = start_number
    for line in pdb_lines:
        if line.startswith(("ATOM", "HETATM")):
            new_line = line[:6] + f"{atom_num:5d}" + line[11:]
            renumbered.append(new_line)
            atom_num += 1
        else:
            renumbered.append(line)
    return renumbered


def extract_remark666_and_headers(ref_pdb_path):
    headers_to_add = []
    seen_headers = set()
    with open(ref_pdb_path, 'r') as f:
        for line in f:
            if line.startswith(("HEADER", "REMARK", "HETNAM", "LINK")):
                if line not in seen_headers:
                    headers_to_add.append(line)
                    seen_headers.add(line)
    return headers_to_add


def get_matcher_residues_from_remark666(ref_pdb_path):
    matcher_residues = []
    with open(ref_pdb_path, 'r') as f:
        for line in f:
            if "REMARK 666" in line and "MATCH TEMPLATE" in line:
                parts = line.split()
                if len(parts) >= 7:
                    try:
                        resnum = int(parts[6])
                        if resnum not in matcher_residues:
                            matcher_residues.append(resnum)
                    except (ValueError, IndexError):
                        continue
    return matcher_residues


# ============================================================================
# PRE-LOAD REFERENCE DATA (done once in main process)
# ============================================================================

def preload_reference_data(ref_pdb_path, debug=False):
    """
    Pre-load all reference data for a given reference PDB.
    This is called once per reference PDB in the main process.

    Returns dict with all needed data for workers.
    """
    if debug:
        print(f"[DEBUG] Pre-loading reference data: {os.path.basename(ref_pdb_path)}")

    ref_data = {
        'ref_pdb_path': ref_pdb_path,
        'headers': extract_remark666_and_headers(ref_pdb_path),
        'his_map': build_his_tautomer_map_from_raw_pdb(ref_pdb_path, debug),
    }

    # Get matcher residues
    if HAS_DESIGN_UTILS:
        try:
            ref_data['matched_residues'] = design_utils.get_matcher_residues(ref_pdb_path)
        except:
            ref_data['matched_residues'] = get_matcher_residues_from_remark666(ref_pdb_path)
    else:
        ref_data['matched_residues'] = get_matcher_residues_from_remark666(ref_pdb_path)

    if debug and ref_data['matched_residues']:
        print(f"[DEBUG] Found {len(ref_data['matched_residues'])} catalytic residues in REMARK 666")

    # Pre-load and cache reference protein-only PDB content (as lines)
    with open(ref_pdb_path, 'r') as f:
        ref_content = f.readlines()
    ref_protein_lines, _ = separate_protein_and_hetatm(ref_content)
    ref_data['ref_protein_lines'] = ref_protein_lines

    return ref_data


# ============================================================================
# WORKER PROCESS FUNCTION
# ============================================================================

# Global variable for PyRosetta initialization (per worker)
_PYROSETTA_INITIALIZED = False

def _init_worker():
    """Initialize PyRosetta once per worker process."""
    global _PYROSETTA_INITIALIZED
    if not _PYROSETTA_INITIALIZED:
        pyrosetta.init("-mute all -run:preserve_header")
        _PYROSETTA_INITIALIZED = True


def process_single_pdb_worker(args):
    """
    Worker function to process a single output PDB.

    Args:
        args: tuple of (output_pdb_path, ref_data, final_output_path, clobber, debug)

    Returns:
        tuple: (status, output_pdb_path, message)
    """
    output_pdb_path, ref_data, final_output_path, clobber, debug = args

    try:
        # Ensure PyRosetta is initialized (should already be done by _init_worker)
        global _PYROSETTA_INITIALIZED
        if not _PYROSETTA_INITIALIZED:
            _init_worker()

        output_basename = os.path.basename(output_pdb_path)
        ref_basename = os.path.basename(ref_data['ref_pdb_path'])

        if debug:
            print(f"\n[DEBUG] Processing: {output_basename}")
            print(f"[DEBUG]   Reference: {ref_basename}")
            print(f"[DEBUG]   Output: {final_output_path}")

        # Check if this is a copied reference file that's already ready
        if output_basename == ref_basename:
            if verify_pdb_is_ready(output_pdb_path):
                if debug:
                    print(f"[DEBUG] SKIPPING: {output_basename} (identical to reference, already has REMARK 666 and hydrogens)")
                # Copy to final output if needed
                if os.path.abspath(final_output_path) != os.path.abspath(output_pdb_path):
                    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
                    shutil.copy2(output_pdb_path, final_output_path)
                    if debug:
                        print(f"[DEBUG] Copied to: {final_output_path}")
                return ("skipped_ready", output_pdb_path, "Already has REMARK 666 and hydrogens")

        # Check if final output exists and clobber not set
        if os.path.exists(final_output_path) and not clobber:
            return ("failed", output_pdb_path, "Output exists, --clobber not set")

        # Read output PDB
        with open(output_pdb_path, 'r') as f:
            output_content = f.readlines()

        protein_lines, hetatm_lines = separate_protein_and_hetatm(output_content)
        original_coords_dict = extract_non_hydrogen_coords_with_ids(output_content)

        if debug:
            print(f"[DEBUG] Separated {len(protein_lines)} protein lines and {len(hetatm_lines)} HETATM lines")
            print(f"[DEBUG] Extracted {len(original_coords_dict)} non-hydrogen atoms from original structure")

        # Create temporary protein-only PDB
        temp_protein_pdb = tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False)
        temp_protein_pdb.writelines(protein_lines)
        temp_protein_pdb.close()

        try:
            # Load protein in PyRosetta
            if debug:
                print(f"[DEBUG] Loading protein in PyRosetta (hydrogens will be auto-added)...")

            pose = pyrosetta.pose_from_file(temp_protein_pdb.name)

            # Fix catalytic residues if needed
            if ref_data['matched_residues']:
                if debug:
                    print(f"[DEBUG] Loading reference PDB to fix {len(ref_data['matched_residues'])} catalytic residues...")

                # Create temp ref protein PDB
                temp_ref_protein_pdb = tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False)
                temp_ref_protein_pdb.writelines(ref_data['ref_protein_lines'])
                temp_ref_protein_pdb.close()

                try:
                    ref_pose = pyrosetta.pose_from_file(temp_ref_protein_pdb.name)

                    for catres_seqpos in ref_data['matched_residues']:
                        if catres_seqpos > pose.size() or catres_seqpos > ref_pose.size():
                            if debug:
                                print(f"[DEBUG] WARNING: Residue {catres_seqpos} out of range, skipping")
                            continue

                        ref_rsd = ref_pose.residue(catres_seqpos)
                        catres_AA = ref_rsd.name()
                        catres_AA3 = ref_rsd.name3()

                        # For HIS, override with raw PDB tautomer
                        if catres_AA3 == "HIS":
                            ref_chain = ref_pose.pdb_info().chain(catres_seqpos)
                            ref_pdbno = ref_pose.pdb_info().number(catres_seqpos)
                            key = (ref_chain, ref_pdbno)
                            if key in ref_data['his_map']:
                                raw_his_type = ref_data['his_map'][key]
                                if debug:
                                    print(f"[DEBUG] Catalytic HIS at {ref_chain}{ref_pdbno} (seqpos {catres_seqpos}): using {raw_his_type} from raw PDB")
                                catres_AA = raw_his_type
                            elif ":" in catres_AA:
                                if debug:
                                    print(f"[DEBUG] WARNING: Catalytic HIS at {ref_chain}{ref_pdbno} has patched type {catres_AA} but not in raw his_map - defaulting to HIS")
                                catres_AA = "HIS"

                        if debug:
                            print(f"[DEBUG] Fixing catalytic residue {catres_AA3}{catres_seqpos} with reference type {catres_AA}")

                        mutres = pyrosetta.rosetta.protocols.simple_moves.MutateResidue()
                        mutres.set_res_name(catres_AA)
                        mutres.set_target(catres_seqpos)
                        mutres.set_preserve_atom_coords(True)
                        mutres.apply(pose)

                finally:
                    if os.path.exists(temp_ref_protein_pdb.name):
                        os.unlink(temp_ref_protein_pdb.name)

            # Convert to PDB string
            pdb_string = pyrosetta.distributed.io.to_pdbstring(pose)
            protein_with_h_lines = pdb_string.split('\n')

            # Calculate RMSD
            rosetta_output_lines = [line + '\n' for line in protein_with_h_lines if line.startswith(("ATOM", "HETATM"))]
            rosetta_coords_dict = extract_non_hydrogen_coords_with_ids(rosetta_output_lines)
            rmsd, n_matched = calculate_rmsd_from_dicts(original_coords_dict, rosetta_coords_dict)

            if debug:
                print(f"[DEBUG] Non-hydrogen atom RMSD: {rmsd:.4f} Å ({n_matched} atoms matched)")
                if rmsd > 0.01:
                    print(f"[DEBUG] WARNING: Non-hydrogen RMSD = {rmsd:.4f} Å (coordinates may have changed!)")

            # Build final output
            final_lines = []
            final_lines.extend(ref_data['headers'])
            for line in protein_with_h_lines:
                if line.startswith(("ATOM", "TER")):
                    final_lines.append(line + '\n')
            final_lines.extend(hetatm_lines)
            if not any(line.startswith("END") for line in final_lines):
                final_lines.append("END\n")

            # Renumber atoms
            if debug:
                print(f"[DEBUG] Renumbering all atoms sequentially...")
            final_lines = renumber_pdb_atoms(final_lines, start_number=1)

            # Write output
            os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
            with open(final_output_path, 'w') as f:
                f.writelines(final_lines)

            if debug:
                print(f"[DEBUG] SUCCESS: Wrote {final_output_path}")

            msg = f"RMSD={rmsd:.4f}Å ({n_matched} atoms)"
            if rmsd > 0.01:
                msg += " WARNING: RMSD>0.01"

            return ("processed", output_pdb_path, msg)

        finally:
            if os.path.exists(temp_protein_pdb.name):
                os.unlink(temp_protein_pdb.name)

    except Exception as e:
        if debug:
            import traceback
            print(f"[DEBUG] ERROR processing {output_pdb_path}:")
            traceback.print_exc()
        return ("failed", output_pdb_path, f"Error: {str(e)}")


# ============================================================================
# PROGRESS TRACKING
# ============================================================================

class ProgressTracker:
    def __init__(self, total, milestones=[10, 100, 500]):
        self.total = total
        self.milestones = set(milestones)
        self.processed = 0
        self.start_time = time.time()
        self.last_report = 0
        self.last_quick_update = 0

    def update(self, n=1):
        self.processed += n

        # Quick updates every 100 files (lightweight)
        if self.processed - self.last_quick_update >= 100:
            self.quick_update()
            self.last_quick_update = self.processed

        # Full report at milestones
        if self.processed in self.milestones:
            self.report()
        # After 500, full report every 1000
        elif self.processed > 500 and (self.processed % 1000 == 0 or self.processed == self.total):
            self.report()

    def quick_update(self):
        """Lightweight progress indicator"""
        elapsed = time.time() - self.start_time
        rate = self.processed / elapsed if elapsed > 0 else 0
        pct = 100.0 * self.processed / self.total if self.total > 0 else 0
        print(f"  [{self.processed}/{self.total}] {pct:.1f}% complete - {rate:.1f} PDBs/sec", flush=True)

    def report(self):
        elapsed = time.time() - self.start_time
        rate = self.processed / elapsed if elapsed > 0 else 0
        remaining = self.total - self.processed
        eta_seconds = remaining / rate if rate > 0 else 0

        eta_str = str(timedelta(seconds=int(eta_seconds)))
        elapsed_str = str(timedelta(seconds=int(elapsed)))

        pct = 100.0 * self.processed / self.total if self.total > 0 else 0

        print(f"\n{'='*80}", flush=True)
        print(f"PROGRESS: {self.processed}/{self.total} ({pct:.1f}%)", flush=True)
        print(f"  Elapsed: {elapsed_str}", flush=True)
        print(f"  Rate: {rate:.1f} PDBs/sec", flush=True)
        print(f"  ETA: {eta_str}", flush=True)
        print(f"{'='*80}\n", flush=True)
        sys.stdout.flush()


# ============================================================================
# OPTIMIZED DIRECTORY MODE
# ============================================================================

def process_directory_mode_optimized(
    ref_pdb_dir,
    output_pdb_dir,
    final_output_dir,
    clobber=False,
    debug=False,
    sample_size=None,
    n_workers=None,
    single_thread=False,
    manual_suffixes=None,
    single_ref_pdb=None
):
    """
    Optimized directory processing with multiprocessing (or single-threaded if requested).
    """
    mode_str = "SINGLE-THREADED" if single_thread else "OPTIMIZED DIRECTORY MODE with Multiprocessing"
    print("\n" + "="*80)
    print(mode_str)
    print("="*80)

    if final_output_dir is None:
        final_output_dir = output_pdb_dir
        print("No separate output directory specified - will overwrite input files")

    # Determine number of workers
    if single_thread:
        n_workers = 1
        print("Running in single-threaded mode")
    else:
        if n_workers is None:
            n_workers = mp.cpu_count()
        print(f"Using {n_workers} worker processes")

    # Get all PDB files
    if single_ref_pdb:
        # Process only the specified single reference PDB
        ref_files = [single_ref_pdb]
        print(f"\nSINGLE REF PDB MODE: Processing only {os.path.basename(single_ref_pdb)}")
    else:
        # Process all reference PDBs in directory
        ref_files = sorted(glob.glob(os.path.join(ref_pdb_dir, '*.pdb')))
        print(f"\nFound {len(ref_files)} reference PDB files")

    output_files = sorted(glob.glob(os.path.join(output_pdb_dir, '*.pdb')))
    print(f"Found {len(output_files)} output PDB files")

    if not ref_files or not output_files:
        print("ERROR: No PDB files found!")
        return

    # Identify suffixes
    print("\nIdentifying suffix mappings...")
    start_time = time.time()

    suffix_map = {}
    cumulative_suffixes = set()

    if manual_suffixes is not None:
        # Use manually specified suffixes
        cumulative_suffixes = set(manual_suffixes)
        print(f"Using {len(manual_suffixes)} manually specified suffixes: {manual_suffixes}")

        # Map all refs to these suffixes
        for ref_file in ref_files:
            ref_base = Path(ref_file).stem
            suffix_map[ref_base] = list(cumulative_suffixes)
    else:
        # Auto-detect suffixes
        if sample_size is not None and sample_size < len(ref_files):
            sampled_files = random.sample(ref_files, sample_size)
            if debug:
                print(f"[DEBUG] Using a random sample of {sample_size} reference files for suffix identification")
        else:
            sampled_files = ref_files

        for ref_file in sampled_files:
            ref_base = Path(ref_file).stem
            suffixes_found = []
            for output_file in output_files:
                output_base = Path(output_file).stem
                if output_base.startswith(ref_base):
                    suffix = output_base[len(ref_base):]
                    suffixes_found.append(suffix)
                    cumulative_suffixes.add(suffix)
            suffix_map[ref_base] = suffixes_found
            if debug:
                print(f"[DEBUG] Reference '{ref_base}' matched {len(suffixes_found)} output files")

        # Map all refs to cumulative suffixes
        for ref_file in ref_files:
            ref_base = Path(ref_file).stem
            suffix_map[ref_base] = list(cumulative_suffixes)

        print(f"Suffix identification completed in {time.time() - start_time:.2f} seconds")
        print(f"Found {len(cumulative_suffixes)} unique suffixes: {sorted(cumulative_suffixes)}")

    # PRE-LOAD ALL REFERENCE DATA (in main process)
    print("\nPre-loading reference data...")
    start_time = time.time()
    ref_data_cache = {}
    for ref_file in ref_files:
        ref_data_cache[ref_file] = preload_reference_data(ref_file, debug=debug)
    print(f"Pre-loaded {len(ref_data_cache)} reference PDB files in {time.time() - start_time:.2f} seconds")

    # Build work queue
    print("\nBuilding work queue...")
    work_items = []
    for ref_file in ref_files:
        ref_base = Path(ref_file).stem
        suffixes = suffix_map.get(ref_base, [])
        ref_data = ref_data_cache[ref_file]

        for suffix in suffixes:
            output_base = ref_base + suffix
            output_file = os.path.join(output_pdb_dir, output_base + '.pdb')
            final_output_file = os.path.join(final_output_dir, output_base + '.pdb')

            if os.path.exists(output_file):
                work_items.append((output_file, ref_data, final_output_file, clobber, debug))
            elif debug:
                print(f"[DEBUG] WARNING: Expected output file not found: {output_file}")

    total_work = len(work_items)
    print(f"Created work queue with {total_work} tasks")

    # Process with multiprocessing or single-threaded
    print("\n" + "-"*80)
    print("Processing PDB files...")
    print("-"*80)

    progress = ProgressTracker(total_work)

    total_processed = 0
    total_skipped = 0
    total_already_ready = 0

    overall_start = time.time()

    if single_thread:
        # Single-threaded mode (for debugging)
        _init_worker()  # Initialize PyRosetta
        for work_item in work_items:
            result = process_single_pdb_worker(work_item)
            status, output_path, message = result

            if status == "processed":
                total_processed += 1
            elif status == "skipped_ready":
                total_already_ready += 1
            else:  # failed
                total_skipped += 1
                print(f"FAILED: {os.path.basename(output_path)} - {message}")

            progress.update(1)
    else:
        # Multi-threaded mode
        # Optimize chunk size: smaller chunks = better load balancing but more overhead
        # For large jobs, use adaptive chunking
        if total_work < 100:
            chunk_size = 1  # Process one at a time for small jobs
        elif total_work < 1000:
            chunk_size = max(1, total_work // (n_workers * 8))  # ~8 chunks per worker
        else:
            chunk_size = max(1, min(10, total_work // (n_workers * 10)))  # ~10 chunks per worker, max chunk size 10

        print(f"Using chunk size: {chunk_size} (optimized for {total_work} tasks with {n_workers} workers)")

        with mp.Pool(processes=n_workers, initializer=_init_worker) as pool:
            for result in pool.imap_unordered(process_single_pdb_worker, work_items, chunksize=chunk_size):
                status, output_path, message = result

                if status == "processed":
                    total_processed += 1
                elif status == "skipped_ready":
                    total_already_ready += 1
                else:  # failed
                    total_skipped += 1
                    if not debug:  # Only print in non-debug mode (debug already printed)
                        print(f"FAILED: {os.path.basename(output_path)} - {message}")

                progress.update(1)

    total_time = time.time() - overall_start

    print("\n" + "="*80)
    print("FINAL SUMMARY:")
    print(f"  Processed: {total_processed} files")
    print(f"  Already ready (skipped): {total_already_ready} files")
    print(f"  Failed: {total_skipped} files")
    print(f"  Total: {total_processed + total_already_ready + total_skipped} files")
    print(f"\n  Unique suffixes: {sorted(cumulative_suffixes)}")
    print(f"  Number of suffixes: {len(cumulative_suffixes)}")
    print(f"\n  Total time: {str(timedelta(seconds=int(total_time)))}")
    if total_time > 0:
        print(f"  Average rate: {total_work / total_time:.1f} PDBs/sec")
    print("="*80)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="OPTIMIZED: Add REMARK 666 lines and Rosetta hydrogens to PDB files",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--ref_pdb_dir', type=str, help='Directory with reference PDBs')
    parser.add_argument('--single_ref_pdb', type=str, default=None, help='Process only this single reference PDB (must be in ref_pdb_dir). Use for SLURM parallelization.')
    parser.add_argument('--output_pdb_dir', type=str, help='Directory with output PDBs')
    parser.add_argument('--final_output_dir', type=str, default=None, help='Final output directory')
    parser.add_argument('--clobber', action='store_true', default=False, help='Overwrite existing files')
    parser.add_argument('--verbose', action='store_true', default=False, help='Verbose output (deprecated, use --debug)')
    parser.add_argument('--debug', action='store_true', default=False, help='Enable debug output with detailed prints')
    parser.add_argument('--find_suffixes_from_random_sample', type=int, default=None, help='Sample size for suffix identification')
    parser.add_argument('--suffixes', nargs='+', default=None, help='Manually specify list of suffixes (e.g., --suffixes "_eV1_T0_10__1_1" "_eV1_T0_15__1_1"). If provided, auto-detection is skipped.')
    parser.add_argument('--n_workers', type=int, default=None, help='Number of worker processes (default: all CPUs)')
    parser.add_argument('--single_thread', action='store_true', default=False, help='Run in single-threaded mode (useful for debugging)')

    args = parser.parse_args()

    # Handle verbose as alias for debug
    if args.verbose:
        args.debug = True

    # Validation
    if not args.output_pdb_dir:
        parser.error("Must specify --output_pdb_dir")

    if args.single_ref_pdb:
        # Single ref PDB mode - validate file exists
        if not os.path.exists(args.single_ref_pdb):
            parser.error(f"Specified --single_ref_pdb does not exist: {args.single_ref_pdb}")
        # Set ref_pdb_dir to the directory containing this file if not specified
        if not args.ref_pdb_dir:
            args.ref_pdb_dir = os.path.dirname(args.single_ref_pdb)
            print(f"Auto-detected ref_pdb_dir from single_ref_pdb: {args.ref_pdb_dir}")
    else:
        # Directory mode - require ref_pdb_dir
        if not args.ref_pdb_dir:
            parser.error("Must specify --ref_pdb_dir (or use --single_ref_pdb)")

    print("\n" + "="*80)
    print("Initializing PyRosetta in main process...")
    print("="*80)
    start_time = time.time()
    pyrosetta.init("-mute all -run:preserve_header")
    print(f"PyRosetta initialized in {time.time() - start_time:.2f} seconds")

    overall_start = time.time()

    process_directory_mode_optimized(
        ref_pdb_dir=args.ref_pdb_dir,
        output_pdb_dir=args.output_pdb_dir,
        final_output_dir=args.final_output_dir,
        clobber=args.clobber,
        debug=args.debug,
        sample_size=args.find_suffixes_from_random_sample,
        n_workers=args.n_workers,
        single_thread=args.single_thread,
        manual_suffixes=args.suffixes,
        single_ref_pdb=args.single_ref_pdb
    )

    total_time = time.time() - overall_start
    print(f"\nOverall execution time: {str(timedelta(seconds=int(total_time)))}")


if __name__ == "__main__":
    main()
