#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Original filtering logic: Indrek Kalvet (ikalvet@uw.edu)
Reworked / refactored: Seth M. Woodbury

Post-process and filter aa_diffusion / HAL designs using PyRosetta.

High-level:
    - Takes diffusion outputs as CIF(.gz) plus companion JSON (design_info-like) files
    - Loads the corresponding reference PDB(s) used for diffusion
    - Builds a mapping between reference and design residues from diffused_index_map
    - Applies a set of geometric / environment filters to reject bad backbones
    - Optionally dumps only the “passing” designs as PDBs with annotations
    - Writes a Rosetta-style scorefile summarizing all per-design metrics

Main filters and metrics:
    - Backbone sanity:
        • max CA–CA distance across sequential residues (chainbreak)
        • min non-adjacent CA–CA distance (rCA_nonadj)
    - Ligand environment:
        • min backbone–ligand heavy-atom distance (lig_dist)
        • ligand SASA and relative burial vs. free ligand (SASA, SASA_rel)
        • optional SASA for specific ligand atoms (SASA_exposed_atoms)
        • distance from N- and C-termini to ligand (term_mindist)
    - Motif geometry (when tip-atom diffusion was used):
        • sidechain bond-length deviations vs. ideal (bondlen_dev)
        • cart_bonded and fa_dun scores at motif residues (cart_bonded_avg, fa_dun_avg)
    - Global scaffold quality:
        • loop fraction (loop_frac)
        • longest helix length (longest_helix)
        • radius of gyration (rog)
        • optional penalty if catalytic residues sit between loops (loop_at_motif)

Reference / design mapping:
    - Uses design_info["diffused_index_map"] to map reference residues (e.g. A94) to
      HAL numbering (e.g. A101), in both full and partial diffusion modes
    - Can restrict mapping to user-specified --ref_catres (e.g. “A94-96 B101”)
    - Optionally infers catalytic residues from REMARK 666 lines in the reference
      if --ref_catres is not provided

REMARK 666 handling:
    - Reads REMARK 666 MATCH TEMPLATE / MATCH MOTIF lines from the reference PDB
    - Remaps motif and template residues into HAL numbering using diffused_index_map
    - Re-inserts adjusted REMARK 666 lines into each passing design PDB
    - With --fix_unmatched_remark_lines_to_lig, attempts to repair “unmapped”
      MATCH TEMPLATE targets by assigning them to a unique residue (protein or
      ligand) in the design with the same residue name3, avoiding ambiguous cases

Other behavior:
    - Supports both full-protein diffusion and partial diffusion onto a fixed
      reference scaffold (with optional alignment + ligand grafting)
    - Uses multiprocessing to process many designs in parallel
    - Writes a single Rosetta-style scorefile (diffusion_analysis.sc by default)
      containing all metrics and pass/fail status for each backbone
"""


import os
import sys
import glob
import time
import json
import copy
import gzip
import itertools
import tempfile
import multiprocessing
import argparse
import shutil
from pathlib import Path
from shutil import copy2  # kept in case future logic relies on it

import numpy as np
import pandas as pd

import pyrosetta as pyr
import pyrosetta.rosetta
import pyrosetta.distributed.io
from pyrosetta.rosetta.core.scoring import score_type_from_name

# =============================================================================
# GLOBAL CONSTANTS & SIMPLE UTILITIES
# =============================================================================

aa3to1 = {
    "ALA": 'A', "ARG": 'R', "ASN": 'N', "ASP": 'D', "CYS": 'C',
    "GLN": 'Q', "GLU": 'E', "GLY": 'G', "HIS": 'H', "ILE": 'I',
    "LEU": 'L', "LYS": 'K', "MET": 'M', "PHE": 'F', "PRO": 'P',
    "SER": 'S', "THR": 'T', "TRP": 'W', "TYR": 'Y', "VAL": 'V'
}
aa1to3 = {val: k for k, val in aa3to1.items()}

# Toggle this manually if you want extra debug prints
DEBUG = True


def debug_print(msg: str):
    """Print debug information if DEBUG is enabled."""
    if DEBUG:
        print(f"[DEBUG] {msg}")


# =============================================================================
# I/O & POSE-LEVEL UTILITIES
# =============================================================================

def load_cif_to_pose(cif_file: str) -> pyr.Pose:
    """
    Load a CIF (or CIF.GZ) into a PyRosetta Pose, preserving fold tree.

    Steps:
      - Reads the CIF or CIF.GZ
      - Appends an empty citation title (to avoid certain CIF issues)
      - Uses Rosetta's CIF importer
      - Rebuilds a canonical pose chain-by-chain
    """
    debug_print(f"Loading CIF file into pose: {cif_file}")

    if cif_file.endswith(".cif"):
        with open(cif_file, "r") as fh:
            lines = fh.readlines()
    elif cif_file.endswith(".cif.gz"):
        with gzip.open(cif_file, "rt") as fh:
            lines = fh.readlines()
    else:
        raise ValueError(f"Unsupported file extension for CIF input: {cif_file}")

    lines.append('_citation.title  ""\n')

    tempcif = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + ".cif")
    with open(tempcif, "w") as fh:
        fh.write("".join(lines))

    pose = pyrosetta.rosetta.core.import_pose.pose_from_file(
        tempcif,
        read_fold_tree=True,
        type=pyrosetta.rosetta.core.import_pose.FileType.CIF_file
    )

    pose2 = pyrosetta.rosetta.core.pose.Pose()
    for chain in range(1, pose.num_chains() + 1):
        pyrosetta.rosetta.core.pose.append_subpose_to_pose(
            pose2, pose, pose.chain_begin(chain), pose.chain_end(chain), True
        )

    pdb_string = pyrosetta.distributed.io.to_pdbstring(pose2)
    pose3 = pyr.Pose()
    pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(pose3, pdb_string)

    os.remove(tempcif)
    debug_print(f"Finished loading CIF into pose: {cif_file}")
    return pose3


def add_matcher_line_to_pose(pose, ref_pose, tgt_residues, ref_residues):
    """
    Adjust and insert REMARK 666 lines into the given pose.

    Parameters
    ----------
    pose : pyrosetta.Pose
        Target pose where REMARK 666 lines should be inserted.
    ref_pose : pyrosetta.Pose
        Reference pose (unused here but kept for interface consistency).
    tgt_residues : dict
        Dictionary keyed by residue number in the target pose; each value is
        the REMARK 666 info for that residue (chain, name3, target info, etc.).
    ref_residues : dict
        Original REMARK 666 dictionary from the reference PDB.
    """
    if len(tgt_residues) == 0:
        debug_print("No target residues provided for matcher remark insertion; returning original pose.")
        return pose

    ligand_name = pose.residue(pose.size()).name3()
    debug_print(f"Inserting REMARK 666 lines into pose for ligand name3={ligand_name}, n_lines={len(tgt_residues)}")

    new_remarks = []
    for resno, info in tgt_residues.items():
        new_remarks.append(
            f"REMARK 666 MATCH TEMPLATE {info['target_chain']} {info['target_name']}  {info['target_resno']:>3} MATCH MOTIF {info['chain']} {info['name3']}  {resno:>3}  {info['cst_no']}  {info['cst_no_var']}               \n"
        )

    pdb_str = pyrosetta.distributed.io.to_pdbstring(pose)
    pdb_lines = pdb_str.split("\n")

    new_pdb_lines = []
    if "ATOM" in pdb_lines[0]:
        for lr in new_remarks:
            new_pdb_lines.append(lr)
        new_pdb_lines.extend(pdb_lines)
    else:
        for line in pdb_lines:
            if "HEADER" in line:
                new_pdb_lines.append(line)
                for lr in new_remarks:
                    new_pdb_lines.append(lr)
            elif "REMARK 666" in line:
                continue
            else:
                new_pdb_lines.append(line)

    pose2 = pyr.Pose()
    pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(pose2, "\n".join(new_pdb_lines))
    debug_print("Finished inserting REMARK 666 lines into pose.")
    return pose2


def get_matcher_residues(filename: str) -> dict:
    """
    Parse REMARK 666 matcher lines from a PDB and return a dictionary:

    {
      resno: {
        'target_name': ...,
        'target_chain': ...,
        'target_resno': ...,
        'chain': ...,
        'name3': ...,
        'cst_no': ...,
        'cst_no_var': ...
      },
      ...
    }
    """
    matches = {}
    with open(filename, "r") as fh:
        for line in fh:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                break
            # Strict: only true matcher records, not commentary lines like
            # "REMARK 665 fmt: REMARK 666 MATCH TEMPLATE ..." that just *mention* REMARK 666.
            if not line.startswith("REMARK 666 MATCH"):
                continue
            lspl = line.split()
            resno = int(lspl[11])
            matches[resno] = {
                "target_name": lspl[5],
                "target_chain": lspl[4],
                "target_resno": int(lspl[6]),
                "chain": lspl[9],
                "name3": lspl[10],
                "cst_no": int(lspl[12]),
                "cst_no_var": int(lspl[13]),
            }
    debug_print(f"Parsed {len(matches)} matcher residues from {filename}")
    return matches


def get_extra_remark_lines(filename: str) -> list:
    """
    Return all REMARK lines from `filename` that are NOT actual
    `REMARK 666 MATCH ...` matcher records.

    PyRosetta only round-trips REMARK types it understands, so commentary /
    metadata REMARKs (e.g. `REMARK QCB ...`, `REMARK 665 ...`) on the reference
    PDB would otherwise be dropped when the design pose is dumped. This lets
    us re-inject them into the output PDB so downstream analysis can read them.

    The remapped `REMARK 666 MATCH` lines are added separately via
    add_matcher_line_to_pose(), so they're excluded here to avoid duplicates.
    """
    extras = []
    try:
        with open(filename, "r") as fh:
            for line in fh:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    break
                if line.startswith("REMARK") and not line.startswith("REMARK 666 MATCH"):
                    extras.append(line.rstrip("\n"))
    except OSError as e:
        print(f"[WARNING] Could not read extra REMARK lines from {filename}: {e}")
    return extras


def build_design_path_remark_lines(args, design_info, pdbfile, jsonfile) -> list:
    """
    Build REMARK lines describing a design's provenance, from the rfd3 JSON
    metadata, for injection into the output PDB header.

    Lines (in emit order):
      * REMARK DESIGN_PATH rfd3 input <design_info["specification"]["input"]>
      * REMARK DESIGN_PATH rfd3 output <pdbfile>            # the pre-filter CIF/PDB
      * REMARK DESIGN_PATH rfd3 output_json <jsonfile>      # opt-in
      * REMARK rfd3_property "select_fixed_atoms": <dict>   # opt-in, one-line JSON
      * REMARK rfd3_property "diffused_index_map": <dict>   # opt-in, one-line JSON

    Toggled by:
      --no_design_path_remarks            disables the two default lines
      --inject_output_json_remark         enables the output_json line
      --inject_select_fixed_atoms_remark  enables the select_fixed_atoms line
      --inject_diffused_index_map_remark  enables the diffused_index_map line

    Only the caller knows whether we're in analyze mode; this function does
    not check args.analyze itself.
    """
    if not isinstance(design_info, dict):
        return []

    spec = design_info.get("specification", {}) or {}

    def _get(key):
        # Look top-level first, then under 'specification' (rfd3 JSON nests
        # select_fixed_atoms / input there, but diffused_index_map sits at top).
        if key in design_info:
            return design_info[key]
        return spec.get(key)

    lines = []

    if not args.no_design_path_remarks:
        input_path = _get("input")
        if input_path:
            lines.append(f"REMARK DESIGN_PATH rfd3 input {input_path}")
        if pdbfile:
            lines.append(f"REMARK DESIGN_PATH rfd3 output {pdbfile}")

    if args.inject_output_json_remark and jsonfile:
        lines.append(f"REMARK DESIGN_PATH rfd3 output_json {jsonfile}")

    if args.inject_select_fixed_atoms_remark:
        sfa = _get("select_fixed_atoms")
        if sfa is not None:
            lines.append(
                f'REMARK rfd3_property "select_fixed_atoms": {json.dumps(sfa)}'
            )

    if args.inject_diffused_index_map_remark:
        dim = _get("diffused_index_map")
        if dim is not None:
            lines.append(
                f'REMARK rfd3_property "diffused_index_map": {json.dumps(dim)}'
            )

    return lines


def inject_remark_lines_into_pdb(pdb_file: str, remark_lines: list) -> None:
    """
    Inject `remark_lines` into `pdb_file` just before the first ATOM/HETATM record.
    No-op if the list is empty.
    """
    if not remark_lines:
        return
    with open(pdb_file, "r") as fh:
        lines = fh.readlines()
    out_lines = []
    inserted = False
    for line in lines:
        if not inserted and (line.startswith("ATOM") or line.startswith("HETATM")):
            for r in remark_lines:
                out_lines.append(r.rstrip("\n") + "\n")
            inserted = True
        out_lines.append(line)
    if not inserted:
        for r in remark_lines:
            out_lines.append(r.rstrip("\n") + "\n")
    with open(pdb_file, "w") as fh:
        fh.writelines(out_lines)


# =============================================================================
# GEOMETRY / METRIC UTILITIES
# =============================================================================

def getSASA(pose, resno=None, SASA_atoms=None, ignore_sc=False):
    """
    Calculate SASA for a pose, a residue, or selected atoms in a residue.

    Returns either a surf_vol object (if resno is None) or a float.
    """
    atoms = pyr.rosetta.core.id.AtomID_Map_bool_t()
    atoms.resize(pose.size())

    for i, res in enumerate(pose.residues):
        if res.is_ligand():
            atoms.resize(i + 1, res.natoms(), True)
        else:
            atoms.resize(i + 1, res.natoms(), not ignore_sc)
            if ignore_sc:
                for n in range(1, res.natoms() + 1):
                    if res.atom_is_backbone(n) and not res.atom_is_hydrogen(n):
                        atoms[i + 1][n] = True

    surf_vol = pyr.rosetta.core.scoring.packing.get_surf_vol(pose, atoms, 1.4)

    if resno is not None:
        if isinstance(resno, int):
            res_surf = 0.0
            for i in range(1, pose.residue(resno).natoms() + 1):
                if SASA_atoms is not None and i not in SASA_atoms:
                    continue
                res_surf += surf_vol.surf(resno, i)
            return res_surf

        elif isinstance(resno, list):
            res_surf = 0.0
            for rn in resno:
                for i in range(1, pose.residue(rn).natoms() + 1):
                    if SASA_atoms is not None and i not in SASA_atoms:
                        continue
                    res_surf += surf_vol.surf(rn, i)
            return res_surf

    return surf_vol


def get_ROG(pose) -> float:
    """Compute a simple radius of gyration measure based on protein CA atoms only."""
    ca_coords = [res.xyz("CA") for res in pose.residues if res.is_protein()]
    if not ca_coords:
        return 0.0
    centroid = np.array([np.mean([c.__getattribute__(axis) for c in ca_coords]) for axis in "xyz"])
    ROG = max(np.linalg.norm(centroid - np.array([c.x, c.y, c.z])) for c in ca_coords)
    return float(ROG)


def sidechain_connectivity(res):
    """
    Evaluate the physical correctness of sidechain bond lengths of a residue.

    Compares distances against a reference A-X-A pose (same residue type) and
    returns the maximum absolute deviation of bond lengths.
    """
    ref_res_pose = pyr.pose_from_sequence("A" + res.name1() + "A")
    ref_res = ref_res_pose.residue(2)
    bondlen_deviations = []

    for an in range(1, res.natoms() + 1):
        if res.atom_type(an).element() == "H":
            continue
        for nn in res.bonded_neighbor(an):
            if res.atom_type(nn).element() == "H":
                continue
            bondlen_deviations.append(abs((res.xyz(an) - res.xyz(nn)).norm() - (ref_res.xyz(an) - ref_res.xyz(nn)).norm()))

    return max(bondlen_deviations) if bondlen_deviations else 0.0


def thread_seq_to_pose(pose, sequence, skip_resnos=None):
    """
    Simple sequence threading onto a pose backbone, skipping ligand residues
    and any positions specified in skip_resnos.
    """
    if skip_resnos is None:
        skip_resnos = []

    pose2 = pose.clone()
    for i, aa in enumerate(sequence):
        seqpos = i + 1
        if seqpos in skip_resnos:
            continue
        if pose.residue(seqpos).is_ligand():
            continue

        mutres = pyrosetta.rosetta.protocols.simple_moves.MutateResidue()
        mutres.set_target(seqpos)
        mutres.set_res_name(aa1to3[aa])
        mutres.apply(pose2)

    return pose2

def get_rosetta_scores(pose, sfx, sfx_cart, catres_list):
    """
    Get Rosetta per-residue sidechain quality scores (cart_bonded and fa_dun)
    for a set of residues in catres_list.
    """
    sfx(pose)
    sfx_cart(pose)

    scoredict = {}
    for term in ["cart_bonded", "fa_dun"]:
        scoredict[term] = {
            res.seqpos(): pose.energies().residue_total_energies(res.seqpos()).get(score_type_from_name(term))
            for res in pose.residues
            if res.seqpos() in catres_list
        }

    averages = {term: np.average(list(scores.values())) if scores else 0.0 for term, scores in scoredict.items()}
    return averages, scoredict


# =============================================================================
# SCOREFILE I/O
# =============================================================================

def dump_scorefile(df: pd.DataFrame, filename: str):
    """
    Dump a Rosetta-style scorefile from a DataFrame.
    """
    widths = {}
    for k in df.keys():
        if k in ["SCORE:", "design_path", "name"]:
            widths[k] = 0
        elif len(k) >= 12:
            widths[k] = len(k) + 1
        else:
            widths[k] = 12

    with open(filename, "w") as fh:
        title = ""
        for k in df.keys():
            if k == "SCORE:":
                title += k
            elif k in ["design_path", "name"]:
                title += f" {k}"
            else:
                title += f"{k:>{widths[k]}}"
        if all([t not in df.keys() for t in ["design_path", "name"]]):
            title += " design_path"
        fh.write(title + "\n")

        for index, row in df.iterrows():
            line = ""
            for k in df.keys():
                if isinstance(row[k], (float, np.floating)):
                    val = f"{row[k]:.3f}"
                else:
                    val = row[k]
                if k == "SCORE:":
                    line += val
                elif k in ["design_path", "name"]:
                    line += f" {val}"
                else:
                    line += f"{val:>{widths[k]}}"
            if all([t not in df.keys() for t in ["design_path", "name"]]):
                line += f" {index}"
            fh.write(line + "\n")


# =============================================================================
# MAIN PIPELINE HELPERS
# =============================================================================

def parse_ref_catres(ref_catres_args):
    """
    Parse reference catalytic residue identifiers.

    Accepts a list like: ["A94-96", "B101"] and returns a flat list:
    ["A94", "A95", "A96", "B101"]
    """
    ref_catres = []
    if ref_catres_args is None:
        return ref_catres

    for r in ref_catres_args:
        if "-" in r:
            start_token, end_token = r.split("-")[0], r.split("-")[1]
            ch = r[0]
            for n in range(int(start_token[1:]), int(end_token) + 1):
                ref_catres.append(f"{ch}{n}")
        else:
            ref_catres.append(r)
    return ref_catres


def resolve_pdbfiles(args):
    """
    Resolve the list of design CIF/CIF.GZ files from --pdb or --pdbpath.
    """
    assert any([x is not None for x in [args.pdb, args.pdbpath]]), "Need to provide either --pdb or --pdbpath"

    if args.pdb is not None:
        pdbfiles = args.pdb
    else:
        pdbfiles = []
        for pth in args.pdbpath:
            pdbfiles.extend(glob.glob(os.path.join(pth, "*.cif.gz")))
    pdbfiles = sorted(pdbfiles)
    debug_print(f"Resolved {len(pdbfiles)} design CIF files.")
    return pdbfiles


def resolve_ref_pdbs(args):
    """
    Resolve reference PDBs either from:
      - --ref
      - --ref_path
      - or defer to JSON ("specification.input") if both are None.
    """
    if args.ref is not None:
        debug_print(f"Using explicitly provided reference PDBs (--ref), n={len(args.ref)}")
        return args.ref
    elif args.ref_path is not None:
        ref_pdbs = sorted(glob.glob(os.path.join(args.ref_path, "*.pdb")))
        debug_print(f"Found {len(ref_pdbs)} reference PDBs in --ref_path.")
        return ref_pdbs
    else:
        debug_print("No --ref or --ref_path given; will use design_info 'specification.input'.")
        return None


def ensure_output_dir(path):
    """Create output directory if it doesn't exist (ignore permission errors)."""
    try:
        if not os.path.exists(path):
            os.mkdir(path)
            debug_print(f"Created output directory: {path}")
    except PermissionError:
        print(f"WARNING: Could not create output directory (permission denied): {path}")


def init_pyrosetta_with_params(params):
    """
    Initialize PyRosetta with provided params and DAlphaBall path.
    """
    extra_res_fa = ""
    if params is not None:
        extra_res_fa = "-extra_res_fa"
        for p in params:
            extra_res_fa += f" {p}"

    # DAlphaBall ships with this repository, next to the enzyme_design utilities.
    # $DALPHABALL and then $PATH override it, for a locally compiled build.
    DAB = os.environ.get("DALPHABALL") or str(
        Path(__file__).resolve().parent / "enzyme_design" / "DAlphaBall.gcc")
    if not os.path.exists(DAB):
        DAB = shutil.which("DAlphaBall.gcc")
    if not DAB or not os.path.exists(DAB):
        DAB = None

    assert DAB is not None, (
        "Please compile DAlphaBall.gcc and manually provide a path to it in this script under the variable `DAB`.\n"
        "For more info on DAlphaBall, visit: "
        "https://www.rosettacommons.org/docs/latest/scripting_documentation/RosettaScripts/Filters/HolesFilter"
    )

    init_flags = f"{extra_res_fa} -mute all -dalphaball {DAB} -run:preserve_header -in:fast_restyping true" # set -in:fast_restyping false sometimes? true = faster
    print(f"Initializing PyRosetta with flags: {init_flags}")
    pyr.init(init_flags)


def compute_datacolumns(args):
    """
    Build the list of data columns for the final DataFrame.
    """
    columns = ["chainbreak", "rCA_nonadj", "lig_dist", "bondlen_dev"]
    if args.cart_bonded is not None:
        columns.append("cart_bonded_avg")
    if args.fa_dun is not None:
        columns.append("fa_dun_avg")

    columns.extend(["loop_frac", "longest_helix", "rog"])

    if args.loop_catres is True:
        columns.append("loop_at_motif")

    columns.append("term_mindist")
    columns.extend(["SASA", "SASA_rel"])

    if args.ligand_exposed_atoms is not None:
        columns.append("SASA_exposed_atoms")

    debug_print(f"Initial datacolumns: {columns}")
    return columns


def determine_num_processes(args):
    """
    Determine number of CPU cores to use, preserving the original search order.
    """
    if args.nproc is not None:
        return args.nproc
    if "SLURM_CPUS_ON_NODE" in os.environ:
        return int(os.environ["SLURM_CPUS_ON_NODE"])
    if "OMP_NUM_THREADS" in os.environ:
        return int(os.environ["OMP_NUM_THREADS"])
    return os.cpu_count()


def load_design_info(jsonfile, design_name):
    """Safe JSON load with diagnostics."""
    try:
        with open(jsonfile, "r") as fh:
            design_info = json.load(fh)
        debug_print(f"Loaded design_info JSON for {design_name}: {jsonfile}")
        return design_info
    except Exception as e:
        print(f"[ERROR] Failed to load design_info JSON for {design_name} at {jsonfile}: {e}")
        return None


def select_reference_pdb(design_info, ref_pdbs_local, pdbfile, ref_poses_local):
    """
    Decide which reference PDB to use for a given design and cache the Pose.
    """
    if ref_pdbs_local is not None:
        matching_refs = [
            r for r in ref_pdbs_local
            if os.path.basename(r).replace(".cif.gz", "_") in os.path.basename(pdbfile)
        ]
        if len(matching_refs) != 1:
            print(f"[ERROR] Bad number of reference PDBs found for {pdbfile}: matches={matching_refs}")
            return None, None
        ref_pdb = matching_refs[0]
    else:
        ref_pdb = design_info["specification"]["input"]

    if ref_pdb not in ref_poses_local.keys():
        debug_print(f"Loading reference PDB into pose cache: {ref_pdb}")
        with open(ref_pdb, "r") as fh:
            pdb_lines = [l for l in fh if "ORI" not in l]
        _pose = pyrosetta.rosetta.core.pose.Pose()
        pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(_pose, "".join(pdb_lines))
        if len(_pose.sequence()) == 0:
            print(f"[ERROR] Reference pose appears to be empty for {ref_pdb}")
            return None, None
        ref_poses_local[ref_pdb] = _pose.clone()

    return ref_pdb, ref_poses_local[ref_pdb].clone()


def build_fixed_positions(design_info, args, ref_catres, matched_residues, ref_pose):
    """
    Build mapping between reference residues and HAL (design) residues.

    Returns
    -------
    fixed_positions_from_JSON : dict[str, str]
        Mapping from ref (e.g. 'A106') -> hal (e.g. 'A42')
    fixed_pos_in_hal : list[int]
        HAL residue indices (1-based, no chain prefix).
    fixed_pos_in_ref : list[tuple[str,int]]
        Reference (chain, resno) pairs.
    _ref_catres : list[str]
        Effective catalytic residues used for mapping.
    """
    if args.partial:
        # ------------------------ PARTIAL DIFFUSION -------------------------
        debug_print("Building fixed position mapping in PARTIAL diffusion mode.")
        if "diffused_index_map" not in design_info:
            fixed_positions_from_JSON = {}
            contig = design_info["specification"]["contig"]
            debug_print(f"No diffused_index_map found; building from contig={contig}")
            for x in contig.split(","):
                if not x or not x[0].isalpha():
                    continue
                if "-" in x:
                    start_token, end_token = x.split("-")[0], x.split("-")[1]
                    ch = x[0]
                    for n in range(int(start_token[1:]), int(end_token) + 1):
                        fixed_positions_from_JSON[f"{ch}{n}"] = f"{ch}{n}"
                else:
                    fixed_positions_from_JSON[x] = x
        else:
            fixed_positions_from_JSON = design_info["diffused_index_map"]

        fixed_pos_in_hal = [int(x[1:]) for x in fixed_positions_from_JSON.values()]
        fixed_pos_in_ref = [(x[0], int(x[1:])) for x in fixed_positions_from_JSON.keys()]
        _ref_catres = [f"{ch}{rn}" for ch, rn in fixed_pos_in_ref]
        print(f"### REFERENCE CATALYTIC RESIDUES (PARTIAL, V2) ###\n{_ref_catres}\n")

    else:
        # ------------------------- FULL DIFFUSION ---------------------------
        debug_print("Building fixed position mapping in FULL diffusion mode.")
        full_map = design_info["diffused_index_map"]  # all diffused residues from JSON

        if args.ref_catres is not None:
            # Enforce that --ref_catres is a subset of diffused_index_map keys
            requested = list(args.ref_catres)
            available_keys = set(full_map.keys())
            missing = [r for r in requested if r not in available_keys]

            if missing:
                msg = (
                    "ERROR: The following --ref_catres entries are NOT present in "
                    "diffused_index_map (design_info['diffused_index_map']) and this is "
                    "not allowed:\n"
                    f"    {', '.join(sorted(missing))}\n"
                    "Every residue passed in --ref_catres must appear as a key in "
                    "diffused_index_map. Please fix your input or the design_info and rerun."
                )
                print(msg)
                sys.exit(1)

            # Use ONLY the subset of diffused_index_map that was explicitly requested
            fixed_positions_from_JSON = {
                k: v for k, v in full_map.items() if k in requested
            }

            # Preserve user-specified order for catalytic residues
            _ref_catres = requested.copy()
            print("### REFERENCE CATALYTIC RESIDUES (FULL, from --ref_catres) ###")
            print(f"{_ref_catres}\n")

        else:
            # No explicit --ref_catres: keep original behavior
            fixed_positions_from_JSON = full_map
            fixed_pos_in_hal_tmp = [int(x[1:]) for x in fixed_positions_from_JSON.values()]
            fixed_pos_in_ref_tmp = [(x[0], int(x[1:])) for x in fixed_positions_from_JSON.keys()]

            _ref_catres = ref_catres.copy()
            print(f"### REFERENCE CATALYTIC RESIDUES (FULL, V3 - initial) ###\n{_ref_catres}\n")

            # Optional: infer from REMARK 666 if user did not pass --ref_catres
            if len(_ref_catres) == 0 and len(matched_residues) > 0:
                inferred = []
                pdbinfo = ref_pose.pdb_info()

                for rn, d in matched_residues.items():
                    chain = d["chain"]  # e.g. 'A'

                    # Map (chain, PDB residue number) -> Rosetta seqpos
                    seqpos = None
                    for i in range(1, ref_pose.size() + 1):
                        if pdbinfo.chain(i) == chain and pdbinfo.number(i) == rn:
                            seqpos = i
                            break

                    if seqpos is None:
                        debug_print(
                            f"[build_fixed_positions] WARNING: "
                            f"could not map REMARK 666 residue {chain}{rn} into ref_pose; skipping."
                        )
                        continue

                    if ref_pose.residue(seqpos).is_protein():
                        inferred.append(f"{chain}{rn}")

                if inferred:
                    _ref_catres = inferred
                    print("### REFERENCE CATALYTIC RESIDUES (FULL, V4 - inferred from REMARK 666) ###")
                    print(f"{_ref_catres}\n")
                else:
                    print("### WARNING: No protein REMARK 666 residues found to infer catalytic residues; using empty list. ###")
                    _ref_catres = []

        # Now that fixed_positions_from_JSON is finalized (either subset or full),
        # compute the numeric lists.
        fixed_pos_in_hal = [int(x[1:]) for x in fixed_positions_from_JSON.values()]
        fixed_pos_in_ref = [(x[0], int(x[1:])) for x in fixed_positions_from_JSON.keys()]

    debug_print(f"Fixed positions from JSON (n={len(fixed_positions_from_JSON)}): {fixed_positions_from_JSON}")
    return fixed_positions_from_JSON, fixed_pos_in_hal, fixed_pos_in_ref, _ref_catres


def compute_backbone_metrics(pose, scores_entry, args, pdbfile, pose_label="pose"):
    """
    Compute chainbreak and non-adjacent CA-CA distances, update scores_entry.

    Returns True if passes filters, False if should be filtered out.
    """
    # Chainbreak
    dists = []
    for n in range(1, pose.size()):
        if pose.residue(n).is_ligand():
            continue
        if pose.residue(n + 1).is_ligand():
            continue
        if pose.chain(n) != pose.chain(n + 1):
            continue
        dists.append((pose.residue(n).xyz("CA") - pose.residue(n + 1).xyz("CA")).norm())
    if not dists:
        print(f"[WARNING] No chainbreak distances computed for {pdbfile} ({pose_label}); skipping chainbreak filter.")
        scores_entry["chainbreak"] = np.nan
    else:
        scores_entry["chainbreak"] = max(dists)
        # pass: chainbreak <= 4.5  (inclusive upper bound, hardcoded)
        if not args.analyze and scores_entry["chainbreak"] > 4.5:
            print(f"{pdbfile}: chainbreak found! max_CA_CA={scores_entry['chainbreak']:.2f}")
            return False

    # Non-adjacent CA-CA
    nonadjacentCAs = []
    for (r1, r2) in itertools.combinations(pose.residues, 2):
        if r1.is_ligand() or r2.is_ligand():
            continue
        if r1.is_virtual_residue() or r2.is_virtual_residue():
            continue
        if not r1.is_protein() or not r2.is_protein():
            continue
        if abs(r1.seqpos() - r2.seqpos()) == 1:
            continue
        nonadjacentCAs.append((r1.xyz("CA") - r2.xyz("CA")).norm())
    if not nonadjacentCAs:
        print(f"[WARNING] No non-adjacent CA-CA distances computed for {pdbfile} ({pose_label}); skipping rCA_nonadj filter.")
        scores_entry["rCA_nonadj"] = np.nan
    else:
        scores_entry["rCA_nonadj"] = min(nonadjacentCAs)
        # pass: rCA_nonadj >= 3.0  (inclusive lower bound, hardcoded)
        if not args.analyze and scores_entry["rCA_nonadj"] < 3.0:
            print(f"{pdbfile}: some residues are too close to each other: min_CA_CA={scores_entry['rCA_nonadj']:.2f}")
            return False

    return True

# =============================================================================
# ROSETTA HELPERS
# =============================================================================

def build_raw_his_tautomer_map_from_pdb(
    pdb_path: str,
    *,
    nd1_ne2_H_dist_cut: float = 1.35,
    allow_fallback_geometry: bool = True,
):
    """
    Build a dict mapping raw-PDB HIS residues to desired Rosetta histidine types.

    Returns
    -------
    his_map : dict
        (chain:str, resno:int) -> "HIS_D" or "HIS"

        Logic priority:
          1) If exactly one of HD1 / HE2 is present in raw PDB atoms:
             - HD1 only => "HIS_D"
             - HE2 only => "HIS"
          2) Else (missing or both), optionally fall back to geometry:
             - Find hydrogens whose nearest heavy atom is ND1 or NE2 (using distance).
             - If H near ND1 only => "HIS_D"
             - If H near NE2 only => "HIS"
             - If both => ambiguous_both
             - If neither => ambiguous_none

    diagnostics : dict
        A structured record of ambiguous cases and how each residue was decided.
        Keys include:
          - "ambiguous_both": {(chain,resno): {...}}
          - "ambiguous_none": {(chain,resno): {...}}
          - "used_name_override": {(chain,resno): {...}}
          - "used_geom_fallback": {(chain,resno): {...}}

    Notes
    -----
    - This reads ONLY ATOM/HETATM lines.
    - Requires ND1 and NE2 coordinates to do the fallback.
    - Geometry fallback is robust to weird H naming (e.g., 1H/2H/3H at N-terminus),
      because it uses distance-to-N rather than atom names.
    - If you want it to *error* on ambiguous cases, handle that using the diagnostics.
    """
    # ----------------------------
    # Helpers
    # ----------------------------
    def _parse_atom_line(line: str):
        # PDB fixed columns
        atom = line[12:16].strip()
        resn = line[17:20].strip()
        ch = line[21].strip()
        resi_raw = line[22:26]
        try:
            resi = int(resi_raw)
        except Exception:
            return None

        # element is optional, but if present it helps filter H faster
        elem = line[76:78].strip().upper() if len(line) >= 78 else ""
        # coords
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except Exception:
            return None

        return atom, resn, ch, resi, (x, y, z), elem

    def _dist(a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5

    # ----------------------------
    # Pass 1: collect atoms for each HIS residue
    # ----------------------------
    # (chain,resno) -> {"atoms": set(str), "coords": {atomname: (x,y,z)}, "elem": {atomname: elem}}
    seen = {}

    with open(pdb_path, "r") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            parsed = _parse_atom_line(line)
            if parsed is None:
                continue
            atom, resn, ch, resi, xyz, elem = parsed

            if resn != "HIS":
                continue

            key = (ch, resi)
            if key not in seen:
                seen[key] = {"atoms": set(), "coords": {}, "elem": {}}

            seen[key]["atoms"].add(atom)
            # If duplicates exist (altloc), last one wins; for this use-case that’s OK
            seen[key]["coords"][atom] = xyz
            seen[key]["elem"][atom] = elem

    # ----------------------------
    # Pass 2: decide tautomer
    # ----------------------------
    his_map = {}
    diagnostics = {
        "used_name_override": {},
        "used_geom_fallback": {},
        "ambiguous_both": {},
        "ambiguous_none": {},
        "missing_nd1_ne2": {},
    }

    for key, rec in seen.items():
        atoms = rec["atoms"]
        coords = rec["coords"]
        elem_map = rec["elem"]

        has_hd1 = ("HD1" in atoms)
        has_he2 = ("HE2" in atoms)

        # ---- name-based decision (preferred when unambiguous) ----
        if has_hd1 and not has_he2:
            his_map[key] = "HIS_D"
            diagnostics["used_name_override"][key] = {"mode": "names", "has_hd1": True, "has_he2": False}
            continue
        if has_he2 and not has_hd1:
            his_map[key] = "HIS"
            diagnostics["used_name_override"][key] = {"mode": "names", "has_hd1": False, "has_he2": True}
            continue

        # If both present or neither present, try geometry fallback (optional)
        if not allow_fallback_geometry:
            # Don't assign; mark ambiguous
            bucket = "ambiguous_both" if (has_hd1 and has_he2) else "ambiguous_none"
            diagnostics[bucket][key] = {
                "mode": "names_only",
                "has_hd1": has_hd1,
                "has_he2": has_he2,
                "atoms_present": sorted(list(atoms)),
            }
            continue

        # ---- geometry fallback ----
        # Need ND1/NE2 coords
        if ("ND1" not in coords) or ("NE2" not in coords):
            diagnostics["missing_nd1_ne2"][key] = {
                "mode": "geom",
                "has_hd1": has_hd1,
                "has_he2": has_he2,
                "atoms_present": sorted(list(atoms)),
            }
            continue

        nd1_xyz = coords["ND1"]
        ne2_xyz = coords["NE2"]

        # Collect candidate H atoms in this residue
        # Prefer element column if available; otherwise use atom name heuristic
        H_atoms = []
        for aname in atoms:
            elem = elem_map.get(aname, "")
            if elem == "H":
                H_atoms.append(aname)
            elif not elem:
                # element missing; fall back to name-based guess
                if aname.startswith("H") or aname.endswith("H") or aname[0].isdigit():
                    # includes "1H", "2H", "3H"
                    H_atoms.append(aname)

        # Find if any H is within cutoff of ND1/NE2
        nd1_hits = []
        ne2_hits = []
        for haname in H_atoms:
            hxyz = coords.get(haname)
            if hxyz is None:
                continue
            d_nd1 = _dist(hxyz, nd1_xyz)
            d_ne2 = _dist(hxyz, ne2_xyz)
            if d_nd1 <= nd1_ne2_H_dist_cut:
                nd1_hits.append((haname, d_nd1))
            if d_ne2 <= nd1_ne2_H_dist_cut:
                ne2_hits.append((haname, d_ne2))

        has_h_on_nd1 = len(nd1_hits) > 0
        has_h_on_ne2 = len(ne2_hits) > 0

        diagnostics["used_geom_fallback"][key] = {
            "mode": "geom",
            "has_hd1": has_hd1,
            "has_he2": has_he2,
            "nd1_hits": sorted(nd1_hits, key=lambda x: x[1]),
            "ne2_hits": sorted(ne2_hits, key=lambda x: x[1]),
            "cut": nd1_ne2_H_dist_cut,
        }

        if has_h_on_nd1 and not has_h_on_ne2:
            his_map[key] = "HIS_D"
        elif has_h_on_ne2 and not has_h_on_nd1:
            his_map[key] = "HIS"
        elif has_h_on_nd1 and has_h_on_ne2:
            diagnostics["ambiguous_both"][key] = diagnostics["used_geom_fallback"][key]
        else:
            diagnostics["ambiguous_none"][key] = diagnostics["used_geom_fallback"][key]

    return his_map, diagnostics

def _is_chain_nterm(pose, seqpos: int) -> bool:
    ch = pose.chain(seqpos)
    return seqpos == pose.chain_begin(ch)

def _is_chain_cterm(pose, seqpos: int) -> bool:
    ch = pose.chain(seqpos)
    return seqpos == pose.chain_end(ch)

def _allow_terminal_patch_on_target(args, pose, seqpos: int, suffix: str) -> bool:
    # Only allow Nterm/Cterm patches if the residue is actually terminal in the target pose
    if suffix == "NtermProteinFull":
        return args.preserve_N_terminal_catres_rosetta_understanding and _is_chain_nterm(pose, seqpos)
    if suffix == "CtermProteinFull":
        return args.preserve_C_terminal_catres_rosetta_understanding and _is_chain_cterm(pose, seqpos)
    return True  # non-terminal patches: keep prior behavior

def _resname_exists_in_pose(pose: pyr.Pose, resname: str) -> bool:
    """Return True if `resname` exists in this pose's ResidueTypeSet."""
    rts = pose.residue_type_set_for_pose()
    try:
        _ = rts.name_map(resname)  # throws if not present
        return True
    except Exception:
        return False


def _is_nz_bonded_to_heteroatom_elsewhere(
    pose: pyr.Pose,
    lys_seqpos: int,
    *,
    cutoff: float = 1.9,
) -> tuple[bool, dict]:
    """
    Check if LYS NZ is within `cutoff` Å of any heteroatom (O/N/S)
    on a different residue (protein or ligand).

    Returns (is_bonded, details_dict).
    """
    rsd = pose.residue(lys_seqpos)
    if rsd.name3() != "LYS" or (not rsd.has("NZ")):
        return False, {"reason": "not_LYS_or_missing_NZ"}

    nz = rsd.xyz("NZ")

    hetero_elems = {"O", "N", "S", "C"}
    metal_elems  = {"ZN", "MG", "FE", "CU", "MN", "CO", "NI", "CA"}

    best = None  # (dist, other_seqpos, other_name3, atomname, elem)

    for j in range(1, pose.size() + 1):
        if j == lys_seqpos:
            continue
        other = pose.residue(j)
        if other.is_virtual_residue():
            continue

        # quick-ish coarse filter
        if (other.nbr_atom_xyz() - nz).norm() > 8.0:
            continue

        for a in range(1, other.natoms() + 1):
            if other.atom_is_hydrogen(a):
                continue
            elem = other.atom_type(a).element().strip().upper()

            # only heteroatoms; exclude metals
            if elem in metal_elems:
                continue
            if elem not in hetero_elems:
                continue

            d = (other.xyz(a) - nz).norm()
            if d <= cutoff:
                cand = (float(d), j, other.name3(), other.atom_name(a).strip(), elem)
                if (best is None) or (cand[0] < best[0]):
                    best = cand

    if best is None:
        return False, {"reason": "no_heteroatom_within_cutoff", "cutoff": cutoff}

    d, j, name3, aname, elem = best
    return True, {
        "reason": "found_contact",
        "cutoff": cutoff,
        "dist": d,
        "partner_seqpos": j,
        "partner_name3": name3,
        "partner_atom": aname,
        "partner_elem": elem,
    }

### THIS DID NOT WORK IN PRACTICE!!! ###
def _choose_neutral_lys_resname_for_pose(pose2: pyr.Pose) -> str | None:
    """
    Return the neutral lysine residue type name to use in this Rosetta build,
    or None if not available.

    Most commonly it's 'LYN'. We probe a few plausible names.
    """
    candidates = [
        "LYN",  # common neutral lysine in many Rosetta setups
        "LYS:neutral",  # sometimes seen in some patched/type setups
    ]
    for c in candidates:
        if _resname_exists_in_pose(pose2, c):
            return c
    return None


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main(args):
    """
    Main entry point.

    Steps:
      1) Resolve input design CIF files and reference PDBs
      2) Initialize PyRosetta & scoring functions
      3) Build multiprocessing queue & shared data structures
      4) Spawn workers that parse and filter each design (via Pool initializer loop)
      5) Aggregate scores into a DataFrame and write a scorefile
    """
    # ----- Filter convention -----------------------------------------------
    # All metric bounds in this script are INCLUSIVE: a design passes when the
    # observed metric is <= the upper bound or >= the lower bound (equality
    # passes). Each fail check below is written as the negation of that pass
    # condition (e.g. `loop_frac > loop_limit_max` fires the failure branch,
    # which is mathematically the same as `not (loop_frac <= loop_limit_max)`).
    # Bounds that are None on `args` are skipped entirely.
    # -----------------------------------------------------------------------
    pdbfiles = resolve_pdbfiles(args)
    params = args.params

    ref_pdbs = resolve_ref_pdbs(args)
    ref_catres = parse_ref_catres(args.ref_catres)

    exposed_sasa_bounded = (args.exposed_atom_SASA_min is not None
                            or args.exposed_atom_SASA_max is not None)
    if args.ligand_exposed_atoms is not None and not exposed_sasa_bounded:
        sys.exit("Defined --ligand_exposed_atoms but neither --exposed_atom_SASA_min nor --exposed_atom_SASA_max (legacy --exposed_atom_SASA).")
    if exposed_sasa_bounded and args.ligand_exposed_atoms is None:
        sys.exit("Defined --exposed_atom_SASA_(min|max) but not --ligand_exposed_atoms.")

    filtered_dir = args.outdir
    ensure_output_dir(filtered_dir)

    init_pyrosetta_with_params(params)

    # ImportPose options (currently unused but kept for future flexibility)
    opts = pyrosetta.rosetta.core.import_pose.ImportPoseOptions()
    opts.set_fast_restyping(False)

    sfx = pyr.get_fa_scorefxn()
    sfx_cart = sfx.clone()
    sfx_cart.set_weight(score_type_from_name("cart_bonded"), 0.5)
    sfx_cart.set_weight(score_type_from_name("pro_close"), 0.0)

    start = time.time()

    the_queue = multiprocessing.Queue()
    manager = multiprocessing.Manager()

    ref_poses = manager.dict()   # reference pose cache across workers
    scores = manager.dict()      # per-design scores across workers

    print(f"{len(pdbfiles)} designs to analyze.")
    print("Building multiprocessing queue of designs.")

    for idx, pdbfile in enumerate(pdbfiles):
        scores[idx] = manager.dict()
        the_queue.put((idx, pdbfile))

    datacolumns = manager.list()
    datacolumns += compute_datacolumns(args)

    # -------------------------------------------------------------------------
    # INNER WORKER FUNCTION
    # -------------------------------------------------------------------------

    def process_worker(q, ref_pdbs_local, ref_poses_local):
        """
        Worker loop that processes designs until it receives a sentinel (None).

        NOTE: This function is used as the Pool initializer and never returns
        until it consumes a sentinel from the shared queue.
        """

        raw_his_cache = {}         # key: ref_pdb_path -> (raw_his_map, raw_his_diag) |  Cache raw HIS tautomer maps per reference PDB path

        while True:
            try:
                job = q.get(block=True)
            except Exception as err:
                print(f"[ERROR] Worker encountered queue error: {err}")
                return

            if job is None:
                debug_print("Worker received sentinel None; exiting.")
                return

            i, pdbfile = job
            design_name = os.path.basename(pdbfile)
            scores[i]["design_path"] = pdbfile
            scores[i]["passed"] = False if not args.analyze else np.nan  # will only be True for passing designs

            if args.analyze:
                print(f"\n===== PROCESSING DESIGN [{i}] {design_name} =====")

            jsonfile = pdbfile.replace(".cif.gz", ".json")
            debug_print(f"[{i}] JSON metadata path: {jsonfile}")

            design_info = load_design_info(jsonfile, design_name)
            if design_info is None:
                scores[i]["error"] = "failed_to_load_design_info"
                continue

            ref_pdb, ref_pose = select_reference_pdb(design_info, ref_pdbs_local, pdbfile, ref_poses_local)
            if ref_pdb is None or ref_pose is None:
                scores[i]["error"] = "failed_to_load_ref_pose"
                continue

            # Build raw HIS tautomer map for THIS reference PDB (cache per ref_pdb)
            if ref_pdb not in raw_his_cache:
                raw_his_cache[ref_pdb] = build_raw_his_tautomer_map_from_pdb(ref_pdb)
            raw_his_map, raw_his_diag = raw_his_cache[ref_pdb]

            matched_residues = get_matcher_residues(ref_pdb)
            print(f"### MATCHED RESIDUES FOR {design_name} ###\n{matched_residues}\n")

            fixed_positions_from_JSON, fixed_pos_in_hal, fixed_pos_in_ref, _ref_catres = build_fixed_positions(
                design_info, args, ref_catres, matched_residues, ref_pose
            )

            # Load design pose from CIF
            print(f"[{design_name}] Loading CIF into Pose...")
            try:
                pose = load_cif_to_pose(pdbfile)
            except Exception as e:
                print(f"[ERROR] Error loading CIF for {pdbfile}: {e}")
                scores[i]["error"] = "failed_to_load_cif"
                continue

            poses_to_parse = [pose]  # kept for future extension with multiple states/trajectories

            for traj_idx, pose in enumerate(poses_to_parse):
                pose2 = pose.clone()
                debug_print(f"[{design_name}] Processing trajectory index {traj_idx}.")

                # Partial diffusion: align protein and append ligand from reference if needed
                if args.partial and any(res.is_ligand() for res in ref_pose.residues) and not any(res.is_ligand() for res in pose2.residues):
                    print(f"[{design_name}] Aligning partial diffusion output to reference and appending ligand(s).")
                    align_map = pyrosetta.rosetta.std.map_core_id_AtomID_core_id_AtomID()
                    aln_atoms = ['N', 'CA', 'C', 'O']

                    for template_i, target_i in zip(design_info["con_ref_idx0"], design_info["con_hal_idx0"]):
                        res_template_i = ref_pose.residue(template_i + 1)
                        res_target_i = pose2.residue(target_i + 1)
                        for aname in aln_atoms:
                            template_atom_idx = res_template_i.atom_index(aname)
                            target_atom_idx = res_target_i.atom_index(aname)
                            atom_id_template = pyrosetta.rosetta.core.id.AtomID(template_atom_idx, template_i + 1)
                            atom_id_target = pyrosetta.rosetta.core.id.AtomID(target_atom_idx, target_i + 1)
                            align_map[atom_id_target] = atom_id_template

                    rmsd = pyrosetta.rosetta.core.scoring.superimpose_pose(pose2, ref_pose, align_map)
                    print(f"[{design_name}] Alignment RMSD (partial diffusion) = {rmsd:.3f}")

                    ligands_ref = [res for res in ref_pose.residues if res.is_ligand()]
                    for lig in ligands_ref:
                        pyrosetta.rosetta.core.pose.append_subpose_to_pose(pose2, ref_pose, lig.seqpos(), lig.seqpos(), True)

                # --------------------------------------------------------------
                # 1) BACKBONE CHAINBREAKS & NON-ADJ CA-CA
                # --------------------------------------------------------------
                passed_backbone = compute_backbone_metrics(pose, scores[i], args, pdbfile, pose_label=f"traj{traj_idx}")
                if not passed_backbone:
                    scores[i]["failed_stage"] = "backbone"
                    break

                # --------------------------------------------------------------
                # 2) LIGAND CLASH CHECK
                # --------------------------------------------------------------
                ligands = [res for res in pose2.residues if res.is_ligand()]
                if ligands:
                    lig_dists = []
                    for lig in ligands:
                        ligand_HAs = [n for n in range(1, lig.natoms() + 1) if not lig.atom_is_hydrogen(n)]
                        for res in pose2.residues:
                            if (res.nbr_atom_xyz() - lig.nbr_atom_xyz()).norm() > 15.0:
                                continue
                            if res.is_ligand():
                                continue
                            for lha in ligand_HAs:
                                if args.exclude_clash_atoms is not None and lig.atom_name(lha).strip() in args.exclude_clash_atoms:
                                    continue
                                for n in range(1, min(5, res.natoms() + 1)):
                                    lig_dists.append((res.xyz(n) - lig.xyz(lha)).norm())
                    if not lig_dists:
                        lig_dists = [9.9]
                        print(f"[WARNING] {pdbfile} no clashcheck-valid residues around ligand found; defaulting lig_dist=9.9.")
                    scores[i]["lig_dist"] = min(lig_dists)
                    # pass: lig_dist >= args.lig_dist  (inclusive lower bound)
                    if not args.analyze and scores[i]["lig_dist"] < args.lig_dist:
                        print(f"{pdbfile}: ligand is too close to the backbone (min ligand-backbone dist={scores[i]['lig_dist']:.2f})")
                        scores[i]["failed_stage"] = "ligand_clash"
                        break
                else:
                    scores[i]["lig_dist"] = np.nan
                    debug_print(f"[{design_name}] No ligands found in pose; skipping ligand clash check.")

                # --------------------------------------------------------------
                # 3) TIP-ATOM SIDECHAIN CONNECTIVITY & CART_BONDED / FA_DUN
                # --------------------------------------------------------------
                if "select_fixed_atoms" in design_info.get("specification", {}):
                    motif_res_bond_deviations = []
                    for res in fixed_positions_from_JSON.values():
                        resno = int(res[1:])
                        motif_res_bond_deviations.append(sidechain_connectivity(pose2.residue(resno + 1)))
                    scores[i]["bondlen_dev"] = max(motif_res_bond_deviations) if motif_res_bond_deviations else 0.0
                    # pass: bondlen_dev <= args.bondlen_dev  (inclusive upper bound)
                    if (not args.analyze and args.bondlen_dev is not None
                            and scores[i]["bondlen_dev"] > args.bondlen_dev):
                        print(f"{pdbfile}: motif residue geometry too distorted (bondlen_dev={scores[i]['bondlen_dev']:.2f})")
                        scores[i]["failed_stage"] = "bondlen_dev"
                        break

                    if args.cart_bonded is not None or args.fa_dun is not None:
                        catres_seqpos = [int(r[1:]) for r in fixed_positions_from_JSON.values()]
                        averages, scoredict = get_rosetta_scores(pose, sfx, sfx_cart, catres_seqpos)
                        for term in averages:
                            scores[i][f"{term}_avg"] = averages[term]

                        # pass: cart_bonded_avg <= args.cart_bonded  (inclusive upper bound)
                        if not args.analyze and args.cart_bonded is not None and scores[i]["cart_bonded_avg"] > args.cart_bonded:
                            print(f"{pdbfile}: motif sidechain geometry too distorted (cart_bonded_avg={scores[i]['cart_bonded_avg']:.2f})")
                            scores[i]["failed_stage"] = "cart_bonded"
                            break

                        # pass: fa_dun_avg <= args.fa_dun  (inclusive upper bound)
                        if not args.analyze and args.fa_dun is not None and scores[i]["fa_dun_avg"] > args.fa_dun:
                            print(f"{pdbfile}: motif rotamers suboptimal (fa_dun_avg={scores[i]['fa_dun_avg']:.2f})")
                            scores[i]["failed_stage"] = "fa_dun"
                            break
                else:
                    scores[i]["bondlen_dev"] = np.nan
                    debug_print(f"[{design_name}] No 'select_fixed_atoms' in specification; skipping motif connectivity and rotamer checks.")

                # --------------------------------------------------------------
                # 4) SUBJECTIVE SCAFFOLD QUALITY (Loops, helices, ROG, loops@motif)
                # --------------------------------------------------------------
                dssp = pyrosetta.rosetta.core.scoring.dssp.Dssp(pose2)
                secstruct = dssp.get_dssp_secstruct()

                loop_frac = secstruct.count("L") / max(1, pose2.size())
                scores[i]["loop_frac"] = loop_frac
                # pass: loop_frac <= args.loop_limit_max  (inclusive upper bound)
                if (not args.analyze and args.loop_limit_max is not None
                        and loop_frac > args.loop_limit_max):
                    if loop_frac > 0.9 and traj_idx > 0:
                        print(f"{pdbfile}: unusual trajectory loopiness? loop_frac={loop_frac:.3f}")
                    else:
                        print(f"{pdbfile}: protein too loopy (loop_frac={loop_frac:.3f})")
                    scores[i]["failed_stage"] = "loop_frac"
                    break
                # pass: loop_frac >= args.loop_limit_min  (inclusive lower bound)
                if (not args.analyze and args.loop_limit_min is not None
                        and loop_frac < args.loop_limit_min):
                    print(f"{pdbfile}: protein not loopy enough (loop_frac={loop_frac:.3f})")
                    scores[i]["failed_stage"] = "loop_frac_low"
                    break

                if "H" in secstruct:
                    longest_helix = max(len(x.replace("E", "")) for x in secstruct.split("L") if "H" in x)
                else:
                    longest_helix = 0
                scores[i]["longest_helix"] = longest_helix
                # pass: longest_helix <= args.longest_helix  (inclusive upper bound)
                if (not args.analyze and args.longest_helix is not None
                        and longest_helix > args.longest_helix):
                    print(f"{pdbfile}: longest helix too long (longest_helix={longest_helix})")
                    scores[i]["failed_stage"] = "longest_helix"
                    break

                if "metrics" in design_info and "radius_of_gyration" in design_info["metrics"]:
                    scores[i]["rog"] = design_info["metrics"]["radius_of_gyration"]
                    debug_print(f"[{design_name}] Using ROG from design_info: {scores[i]['rog']:.3f}")
                else:
                    scores[i]["rog"] = get_ROG(pose2)
                    debug_print(f"[{design_name}] Using computed ROG: {scores[i]['rog']:.3f}")

                # pass: rog <= args.rog_max  (inclusive upper bound)
                if (not args.analyze and args.rog_max is not None
                        and scores[i]["rog"] > args.rog_max):
                    print(f"{pdbfile}: radius of gyration too high (rog={scores[i]['rog']:.1f})")
                    scores[i]["failed_stage"] = "rog"
                    break
                # pass: rog >= args.rog_min  (inclusive lower bound)
                if (not args.analyze and args.rog_min is not None
                        and scores[i]["rog"] < args.rog_min):
                    print(f"{pdbfile}: radius of gyration too low (rog={scores[i]['rog']:.1f})")
                    scores[i]["failed_stage"] = "rog_low"
                    break

                if args.loop_catres:
                    loops_next_to_catres = False
                    for r_hal, r_ref in zip(fixed_pos_in_hal, fixed_pos_in_ref):
                        if f"{r_ref[0]}{r_ref[1]}" not in _ref_catres:
                            continue
                        left = secstruct[max(0, r_hal - 3):max(0, r_hal - 1)]
                        right = secstruct[r_hal:r_hal + 2]
                        if left == "LL" and right == "LL":
                            loops_next_to_catres = True
                            break
                    scores[i]["loop_at_motif"] = int(loops_next_to_catres)
                    if not args.analyze and loops_next_to_catres:
                        print(f"{pdbfile}: catalytic residue sits between loops (loop_at_motif=1)")
                        scores[i]["failed_stage"] = "loop_at_motif"
                        break
                else:
                    scores[i]["loop_at_motif"] = np.nan

                # --------------------------------------------------------------
                # 5) TERMINI DISTANCE TO LIGAND
                # --------------------------------------------------------------
                if ligands:
                    ligands_design = [res for res in pose2.residues if res.is_ligand()]
                    term_mindists = []
                    for lig in ligands_design:
                        lig_HAs = [n + 1 for n in range(lig.natoms()) if lig.atom_type(n + 1).element() != "H"]
                        d_Nt_lig = min((pose2.residue(1).xyz("CA") - lig.xyz(a)).norm() for a in lig_HAs)
                        d_Ct_lig = min((pose2.residue(pose2.size() - len(ligands_design)).xyz("CA") - lig.xyz(a)).norm() for a in lig_HAs)
                        term_mindists.append(min(d_Nt_lig, d_Ct_lig))
                    scores[i]["term_mindist"] = min(term_mindists) if term_mindists else np.nan
                    # pass: term_mindist >= args.term_limit  (inclusive lower bound)
                    if (not args.analyze and args.term_limit is not None
                            and scores[i]["term_mindist"] < args.term_limit):
                        print(f"{pdbfile}: terminus too close to ligand (term_mindist={scores[i]['term_mindist']:.2f})")
                        scores[i]["failed_stage"] = "term_mindist"
                        break
                else:
                    scores[i]["term_mindist"] = np.nan

                # --------------------------------------------------------------
                # 6) MAP REFERENCE RESIDUES TO HAL NUMBERING (Mutate ALL diffused)
                # --------------------------------------------------------------
                ref_catres_nos = []
                hal_catres_nos = []
                mapping_failure = False

                # Always mutate ALL residues in diffused_index_map, regardless of --ref_catres
                mutate_keys = list(fixed_positions_from_JSON.keys())  # e.g. ["A94", "A96", "A106", "A119", ...]

                for r in mutate_keys:
                    ch = r[0]
                    ref_resno = int(r[1:])

                    if (ch, ref_resno) not in fixed_pos_in_ref:
                        print(f"[ERROR] Cannot find reference residue {r} in diffused_index_map keys {fixed_pos_in_ref}")
                        mapping_failure = True
                        break

                    # Find this residue in the reference pose by chain + PDB number
                    ref_resno_in_pose = None
                    for res in ref_pose.residues:
                        chain = ref_pose.pdb_info().chain(res.seqpos())
                        pdb_no = ref_pose.pdb_info().number(res.seqpos())
                        if chain == ch and pdb_no == ref_resno:
                            ref_resno_in_pose = (chain, res.seqpos())
                            break

                    if ref_resno_in_pose is None:
                        print(f"[ERROR] Could not determine ref_pose residue number for reference residue {r}")
                        mapping_failure = True
                        break

                    ref_catres_nos.append(ref_resno_in_pose)

                    # Map from (chain, ref_resno) → HAL seqpos using fixed_pos_in_ref / fixed_pos_in_hal
                    hal_catres_nos.append(fixed_pos_in_hal[fixed_pos_in_ref.index((ch, ref_resno))])

                if mapping_failure:
                    scores[i]["error"] = "catres_mapping_failed"
                    break
                
                for j, ref_res in enumerate(ref_catres_nos):
                    ref_catres_no = ref_res[1]
                    catres_seqpos = hal_catres_nos[j]
                    catres_AA = ref_pose.residue(ref_catres_no).name().split(":")[0]
                    tgt_name = pose2.residue(catres_seqpos).name()
                    if ("ProteinFull" in tgt_name) and (":" in tgt_name) and (":" not in catres_AA):
                        suffix = tgt_name.split(":", 1)[1]

                        # Do NOT propagate terminal patches unless explicitly allowed AND actually terminal in target
                        if suffix in ("NtermProteinFull", "CtermProteinFull"):
                            if _allow_terminal_patch_on_target(args, pose2, catres_seqpos, suffix):
                                catres_AA = catres_AA + ":" + suffix
                                print(f"  [TERM preserve] keeping :{suffix} at target seqpos={catres_seqpos}")
                            else:
                                print(f"  [TERM strip] NOT keeping :{suffix} (target not terminal or preserve flag not set)")
                        else:
                            # Non-terminal ProteinFull variants: keep old behavior
                            catres_AA = catres_AA + ":" + suffix
                    
                    pdbi = pose2.pdb_info()
                    seq = catres_seqpos

                    print("\n=== HIS TAUTOMER DEBUG ===")
                    print("INTENDED catres_AA:", catres_AA)

                    print("BEFORE:",
                        "seq", seq, pdbi.chain(seq), pdbi.number(seq),
                        pose2.residue(seq).name(),
                        "HD1", pose2.residue(seq).has("HD1"),
                        "HE2", pose2.residue(seq).has("HE2"))

                    # ================= PATCHED HIS FALLBACK OVERRIDE =================
                    ref_rsd   = ref_pose.residue(ref_catres_no)
                    ref_name  = ref_rsd.name()     # e.g. "HIS:NtermProteinFull", "HIS_D:NtermProteinFull", ...
                    ref_name3 = ref_rsd.name3()    # should be "HIS" for any histidine variant

                    # Trigger fallback only for HIS variants that Rosetta "patches" (usually indicated by ':')
                    # i.e. anything that is not exactly "HIS" or "HIS_D", OR more robustly: anything containing ':'
                    is_his_variant = (ref_name3 == "HIS") and (":" in ref_name)

                    if is_his_variant:
                        ref_chain = ref_pose.pdb_info().chain(ref_catres_no)
                        ref_pdbno = ref_pose.pdb_info().number(ref_catres_no)
                        key = (ref_chain, ref_pdbno)

                        if key in raw_his_map:
                            catres_AA = raw_his_map[key]   # "HIS" or "HIS_D"
                            print(f"  [HIS fallback] ref is patched ({ref_name}); raw PDB says {catres_AA} at {ref_chain}{ref_pdbno}")
                        else:
                            # Give a useful debug message from diagnostics if available
                            msg = f"[ERROR] Raw PDB HIS tautomer ambiguous/missing for {ref_chain}{ref_pdbno} (ref_name={ref_name})."
                            if raw_his_diag.get("ambiguous_both", {}).get(key):
                                msg += " Reason: ambiguous_both (H near BOTH ND1 and NE2, or both HD1/HE2 present)."
                            elif raw_his_diag.get("ambiguous_none", {}).get(key):
                                msg += " Reason: ambiguous_none (no H assigned near ND1/NE2, or neither HD1/HE2 found)."
                            elif raw_his_diag.get("missing_nd1_ne2", {}).get(key):
                                msg += " Reason: missing_nd1_ne2 (ND1/NE2 coords not found in raw PDB for geometry fallback)."
                            else:
                                msg += " Reason: residue not found as HIS in raw PDB (check chain/resno match)."

                            print(msg)
                            scores[i]["error"] = "raw_ref_his_tautomer_ambiguous"
                            break

                        # Preserve ProteinFull suffix from TARGET if needed
                        tgt_name = pose2.residue(catres_seqpos).name()
                        if ("ProteinFull" in tgt_name) and (":" in tgt_name) and (":" not in catres_AA):
                            suffix = tgt_name.split(":", 1)[1]

                            # By default, do NOT propagate terminal patches (Nterm/Cterm) onto mutations
                            if suffix in ("NtermProteinFull", "CtermProteinFull"):
                                if _allow_terminal_patch_on_target(args, pose2, catres_seqpos, suffix):
                                    catres_AA = catres_AA + ":" + suffix
                                    print(f"  [TERM preserve] keeping :{suffix} at target seqpos={catres_seqpos}")
                                else:
                                    print(f"  [TERM strip] NOT keeping :{suffix} (target not terminal or preserve flag not set)")
                            else:
                                # Non-terminal ProteinFull variants: keep old behavior
                                catres_AA = catres_AA + ":" + suffix
                    # ================= END OVERRIDE =================

                    # ================= LYS NEUTRALIZATION OVERRIDE =================
                    if args.neutralize_charge_for_noncanonical_lysine_represented_as_lys:
                        ref_rsd = ref_pose.residue(ref_catres_no)

                        # Only consider catalytic lysines from the reference (including patched terminal variants)
                        if ref_rsd.name3() == "LYS":
                            is_bonded, details = _is_nz_bonded_to_heteroatom_elsewhere(
                                ref_pose,
                                ref_catres_no,
                                cutoff=args.lys_nz_bond_cutoff,
                            )

                            if is_bonded:
                                neutral_name = _choose_neutral_lys_resname_for_pose(pose2)
                                if neutral_name is None:
                                    print(
                                        f"[LYS neutralize] Wanted neutral LYS at ref seqpos={ref_catres_no} "
                                        f"but no neutral residue type (e.g. LYN) exists in this build; leaving as {catres_AA}. "
                                        f"Contact: {details}"
                                    )
                                else:
                                    # Preserve any non-terminal ProteinFull suffix if you want (same philosophy as your terminal logic)
                                    tgt_name = pose2.residue(catres_seqpos).name()
                                    if (":" in tgt_name) and ("ProteinFull" in tgt_name):
                                        suffix = tgt_name.split(":", 1)[1]
                                        # Like your terminal policy: do NOT propagate Nterm/Cterm unless explicitly preserved
                                        if suffix not in ("NtermProteinFull", "CtermProteinFull"):
                                            neutral_name = neutral_name + ":" + suffix

                                    print(
                                        f"[LYS neutralize] ref LYS NZ contact -> mutating target to {neutral_name}. "
                                        f"ref_seqpos={ref_catres_no} tgt_seqpos={catres_seqpos} details={details}"
                                    )
                                    catres_AA = neutral_name
                    # ================= END LYS OVERRIDE =================


                    mutres = pyrosetta.rosetta.protocols.simple_moves.MutateResidue()
                    mutres.set_res_name(catres_AA)
                    mutres.set_target(catres_seqpos)
                    mutres.set_preserve_atom_coords(True)
                    mutres.apply(pose2)

                    print("AFTER: ",
                        "seq", seq, pdbi.chain(seq), pdbi.number(seq),
                        pose2.residue(seq).name(),
                        "HD1", pose2.residue(seq).has("HD1"),
                        "HE2", pose2.residue(seq).has("HE2"))

                    print("REF:",
                        "ref_seqpos", ref_catres_no,
                        ref_pose.pdb_info().chain(ref_catres_no),
                        ref_pose.pdb_info().number(ref_catres_no),
                        ref_pose.residue(ref_catres_no).name())
                    print("==========================\n")
                    
                # --------------------------------------------------------------
                # 7) LIGAND SASA & EXPOSURE
                # --------------------------------------------------------------
                if ligands:
                    free_ligands = {}
                    for lig in ligands:
                        tmp_pose = pyrosetta.rosetta.core.pose.Pose()
                        tmp_pose.append_residue_by_jump(lig, 0)
                        free_ligands[lig.name3()] = tmp_pose.clone()

                    free_ligand_SASA = sum(getSASA(p, resno=1) for p in free_ligands.values())
                    scores[i]["SASA"] = getSASA(pose2, resno=[lig.seqpos() for lig in ligands]) if ligands else 0.0
                    scores[i]["SASA_rel"] = scores[i]["SASA"] / free_ligand_SASA if free_ligand_SASA > 0 else np.nan

                    if not args.analyze and not np.isnan(scores[i]["SASA_rel"]):
                        # pass: SASA_rel <= args.SASA_limit_max  (inclusive upper bound)
                        if (args.SASA_limit_max is not None
                                and scores[i]["SASA_rel"] > args.SASA_limit_max):
                            print(f"{pdbfile}: ligand too exposed (SASA_rel={scores[i]['SASA_rel']:.3f})")
                            scores[i]["failed_stage"] = "SASA_rel_high"
                            break
                        # pass: SASA_rel >= args.SASA_limit_min  (inclusive lower bound)
                        if (args.SASA_limit_min is not None
                                and scores[i]["SASA_rel"] < args.SASA_limit_min):
                            print(f"{pdbfile}: ligand too buried (SASA_rel={scores[i]['SASA_rel']:.3f})")
                            scores[i]["failed_stage"] = "SASA_rel_low"
                            break

                    _exposed_sasa_bounded = (args.exposed_atom_SASA_min is not None
                                             or args.exposed_atom_SASA_max is not None)
                    if args.ligand_exposed_atoms is not None and _exposed_sasa_bounded:
                        target_ligand = [res for res in pose2.residues if all(res.has(a) for a in args.ligand_exposed_atoms)]
                        if not target_ligand:
                            print(f"[WARNING] Cannot find a ligand containing atoms {args.ligand_exposed_atoms} in {design_name}; skipping exposed_atom_SASA check.")
                            scores[i]["SASA_exposed_atoms"] = np.nan
                        else:
                            indexes = [target_ligand[0].atom_index(x) for x in args.ligand_exposed_atoms]
                            surf_vol_nosc = getSASA(pose2, ignore_sc=True)
                            scores[i]["SASA_exposed_atoms"] = sum(surf_vol_nosc.surf(pose2.size(), idx) for idx in indexes)

                            # pass: SASA_exposed_atoms >= args.exposed_atom_SASA_min  (inclusive lower bound)
                            if (not args.analyze and args.exposed_atom_SASA_min is not None
                                    and scores[i]["SASA_exposed_atoms"] < args.exposed_atom_SASA_min):
                                print(f"{pdbfile}: ligand atoms {args.ligand_exposed_atoms} too buried (SASA_exposed_atoms={scores[i]['SASA_exposed_atoms']:.3f})")
                                scores[i]["failed_stage"] = "SASA_exposed_atoms"
                                break
                            # pass: SASA_exposed_atoms <= args.exposed_atom_SASA_max  (inclusive upper bound)
                            if (not args.analyze and args.exposed_atom_SASA_max is not None
                                    and scores[i]["SASA_exposed_atoms"] > args.exposed_atom_SASA_max):
                                print(f"{pdbfile}: ligand atoms {args.ligand_exposed_atoms} too exposed (SASA_exposed_atoms={scores[i]['SASA_exposed_atoms']:.3f})")
                                scores[i]["failed_stage"] = "SASA_exposed_atoms_high"
                                break
                else:
                    scores[i]["SASA"] = np.nan
                    scores[i]["SASA_rel"] = np.nan
                    scores[i]["SASA_exposed_atoms"] = np.nan

                # --------------------------------------------------------------
                # 8) MARK AS PASSED (IF NOT ANALYZE-ONLY)
                # --------------------------------------------------------------
                if not args.analyze:
                    scores[i]["passed"] = True
                    print(f"{pdbfile}: GOOD design (passed all filters).")

                # --------------------------------------------------------------
                # 9) ADJUST REMARK 666 MATCHER LINES IN OUTPUT PDB
                # --------------------------------------------------------------
                if not args.analyze and len(matched_residues) != 0:
                    matched_residues_in_design = {}

                    # fixed_positions_from_JSON: ref 'A94' -> hal 'A101' (for example)
                    ref2hal = fixed_positions_from_JSON

                    for ref_resno, info in matched_residues.items():
                        # MATCH MOTIF residue in reference (PDB numbering)
                        ref_motif_key = f"{info['chain']}{ref_resno}"
                        if ref_motif_key not in ref2hal:
                            # motif residue not in diffused region; skip remapping for this line
                            continue

                        hal_motif = ref2hal[ref_motif_key]  # e.g. 'A101'
                        hal_chain, hal_resno = hal_motif[0], int(hal_motif[1:])

                        new_info = copy.deepcopy(info)
                        new_info["chain"] = hal_chain
                        # Track whether we *successfully* remapped the MATCH TEMPLATE target
                        new_info["_target_mapped"] = False

                        tgt_resno_orig = info["target_resno"]
                        tgt_chain_orig = info["target_chain"]

                        if tgt_resno_orig != 0:
                            # Map MATCH TEMPLATE residue (chain + PDB number) to ref_pose seqpos
                            tgt_seqpos = None
                            for res in ref_pose.residues:
                                chain = ref_pose.pdb_info().chain(res.seqpos())
                                pdb_no = ref_pose.pdb_info().number(res.seqpos())
                                if chain == tgt_chain_orig and pdb_no == tgt_resno_orig:
                                    tgt_seqpos = res.seqpos()
                                    break

                            if tgt_seqpos is None:
                                print(f"[WARNING] Could not map MATCH TEMPLATE residue {tgt_chain_orig}{tgt_resno_orig} into ref_pose; "
                                    "leaving target_* fields unchanged in REMARK 666.")
                                matched_residues_in_design[hal_resno] = new_info
                                continue

                            # If MATCH TEMPLATE residue is a protein, try to remap using diffused_index_map
                            if ref_pose.residue(tgt_seqpos).is_protein():
                                tgt_key = f"{tgt_chain_orig}{tgt_resno_orig}"  # uses PDB numbering
                                if tgt_key in ref2hal:
                                    hal_tgt = ref2hal[tgt_key]  # e.g. 'A56'
                                    new_info["target_chain"] = hal_tgt[0]
                                    new_info["target_resno"] = int(hal_tgt[1:])
                                    new_info["_target_mapped"] = True
                                # else: no mapping in diffused_index_map; keep original target_* as-is

                            # If MATCH TEMPLATE residue is a ligand, point it at the matching ligand in the design
                            elif ref_pose.residue(tgt_seqpos).is_ligand():
                                ligands2_in_design = [
                                    res2 for res2 in pose2.residues
                                    if res2.name3() == info["target_name"]
                                ]
                                if len(ligands2_in_design) == 1:
                                    lig_seqpos = ligands2_in_design[0].seqpos()
                                    new_info["target_chain"] = pose2.pdb_info().chain(lig_seqpos)
                                    new_info["target_resno"] = pose2.pdb_info().number(lig_seqpos)
                                    new_info["_target_mapped"] = True
                                else:
                                    print("[WARNING] Multiple ligands with same name in system; REMARK 666 target remap may be incorrect; leaving "
                                        "target_* unchanged.")
                                    matched_residues_in_design[hal_resno] = new_info
                                    continue

                        # If tgt_resno_orig == 0, we *never* tried to map it above,
                        # so _target_mapped stays False; those are "unmatched" and
                        # may be fixed later if unique by residue name.
                        matched_residues_in_design[hal_resno] = new_info

                    # At this point matched_residues_in_design contains REMARK 666 entries
                    # for the design, including some where target_chain/resno might still
                    # be dummy values (e.g. X / 0) OR that we consciously left unmapped.
                    # We marked "mapped" ones with _target_mapped=True.

                    # Optional: try to fix unresolved MATCH TEMPLATE targets where the
                    # target_name corresponds to a unique residue (protein or ligand)
                    # in the design. If multiple residues share the same name3, we
                    # leave the REMARK line unchanged to avoid ambiguity.
                    if args.fix_unmatched_remark_lines_to_lig:
                        # Build mapping: residue name3 -> list of (chain, resno) in pose2
                        res_pos_by_name = {}
                        pdbinfo_design = pose2.pdb_info()
                        for res in pose2.residues:
                            if res.is_virtual_residue():
                                continue
                            name3 = res.name3()
                            chain = pdbinfo_design.chain(res.seqpos())
                            resno = pdbinfo_design.number(res.seqpos())
                            res_pos_by_name.setdefault(name3, []).append((chain, resno))

                        n_fixed = 0
                        for hal_resno, info in matched_residues_in_design.items():
                            # Only attempt to fix entries that were NOT successfully mapped earlier
                            if info.get("_target_mapped", False):
                                continue

                            tname = info.get("target_name", "").strip()
                            if not tname:
                                continue

                            candidates = res_pos_by_name.get(tname, [])

                            # Only fix if there is exactly one residue instance for tname
                            # (protein or ligand). If there are multiple (e.g. many HIS),
                            # we cannot safely disambiguate, so we skip.
                            if len(candidates) == 1:
                                chain, resno = candidates[0]
                                info["target_chain"] = chain
                                info["target_resno"] = resno
                                info["_target_mapped"] = True
                                n_fixed += 1

                        if n_fixed > 0:
                            print(f"[{design_name}] fix_unmatched_remark_lines_to_lig: fixed {n_fixed} REMARK 666 MATCH TEMPLATE targets "
                                f"based on unique residue positions.")

                    print(f"\n[{design_name}] REMARK 666: {len(matched_residues)} in ref, {len(matched_residues_in_design)} mapped into design.")
                    pose2 = add_matcher_line_to_pose(pose2, ref_pose, matched_residues_in_design, matched_residues)
                
                # Only annotate/dump PDBs if not in analyze-only mode
                if not args.analyze:
                    # Add 'motif' label to motif residues
                    for rn in fixed_positions_from_JSON.values():
                        pose2.pdb_info().add_reslabel(int(rn[1:]), "motif")

                    # Decide output file path (PDB instead of CIF)
                    outfile = os.path.join(filtered_dir, os.path.basename(pdbfile))
                    if outfile.endswith(".cif.gz"):
                        outfile = outfile.replace(".cif.gz", ".pdb")
                    elif outfile.endswith(".cif"):
                        outfile = outfile.replace(".cif", ".pdb")

                    debug_print(f"Dumping PDB to: {outfile}")
                    pose2.dump_pdb(outfile)

                    # Preserve all other REMARK lines from the reference PDB
                    # (e.g. REMARK QCB TOTAL_CHARGE, REMARK 665 commentary) since
                    # PyRosetta drops REMARK types it doesn't understand on dump.
                    inject_remark_lines_into_pdb(outfile, get_extra_remark_lines(ref_pdb))

                    # Append DESIGN_PATH / rfd3_property provenance REMARKs from
                    # the design JSON. Goes after the ref_pdb extras because each
                    # injection inserts just before the first ATOM line.
                    inject_remark_lines_into_pdb(
                        outfile,
                        build_design_path_remark_lines(args, design_info, pdbfile, jsonfile),
                    )
                else:
                    debug_print(f"[ANALYZE ONLY] Skipping PDB dump for {pdbfile}")

                # Only one trajectory per file currently; break out of loop
                break

    # -------------------------------------------------------------------------
    # MULTIPROCESSING: START WORKERS
    # -------------------------------------------------------------------------
    N_PROCESSES = determine_num_processes(args)
    print(f"Using {N_PROCESSES} processes for analysis/filtering.")

    pool = multiprocessing.Pool(
        processes=N_PROCESSES,
        initializer=process_worker,
        initargs=(the_queue, ref_pdbs, ref_poses),
    )

    for _ in range(N_PROCESSES):
        the_queue.put(None)

    the_queue.close()
    the_queue.join_thread()
    pool.close()
    pool.join()

    end = time.time()
    print(f"Analyzing diffusion outputs took {end - start:.3f} seconds.")

    # -------------------------------------------------------------------------
    # BUILD DATAFRAME FROM SCORES
    # -------------------------------------------------------------------------
    df = pd.DataFrame()

    if not scores:
        print("[WARNING] No scores collected; nothing to write.")
        return

    keylens = {i: len(scores[i]) for i in scores.keys()}
    all_keys_idx = max(keylens, key=keylens.get)

    for k in scores[all_keys_idx].keys():
        if k == "design_path":
            continue
        df[k] = float
    df["design_path"] = str

    for i in scores.keys():
        for k in scores[i].keys():
            if k == "design_path":
                continue
            df.at[i, k] = scores[i][k]
        df.at[i, "design_path"] = scores[i]["design_path"]

    for k in datacolumns:
        if k not in df.keys():
            df[k] = np.nan

    for k in df.keys():
        if k == "design_path":
            continue
        df = df.sort_values(k)

    if len(df) < 200:
        print(df)
    else:
        print("Too many structures, only printing rows with non-NaN metrics:")
        print(df.dropna(how="all", subset=[c for c in df.columns if c not in ["design_path"]]))

    if args.analyze is False and "passed" in df.columns:
        n_passed = int(df.loc[df.passed == 1.0].shape[0])
        print(f"##### {n_passed}/{len(df)} backbones passed all filters  #######")
        print(f"Backbones and design_infos that pass filters have been copied to `{args.outdir}`")

    print(f"Saving analysis scores of each backbone into `{args.scorefile_out}`")
    dump_scorefile(df, args.scorefile_out)


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pdb", nargs="+", type=str, help="Input PDB/CIFs from aa_diffusion (CIF.GZ expected).")
    parser.add_argument("--pdbpath", nargs="+", type=str, help="Directories where input CIF.GZ files can be found.")
    parser.add_argument("--ref", nargs="+", type=str, help="Reference PDBs used as input for diffusion.")
    parser.add_argument("--ref_path", type=str, help="Path where reference PDBs can be found.")
    parser.add_argument("--analyze", action="store_true", default=False, help="Analyze only; do not filter/move files.")

    parser.add_argument("--params", nargs="+", type=str, help="Params files of ligands and noncanonicals.")

    parser.add_argument("--lig_dist", default=2.5, type=float,
                        help="Inclusive lower bound on backbone-to-ligand heavy-atom distance — design passes when lig_dist >= value.")

    # All bound-style filters below are INCLUSIVE: a design passes when the
    # metric is <= the upper bound or >= the lower bound (equality passes).
    # Filters are skipped when their bound is not given on the CLI.
    parser.add_argument("--SASA_limit_max", "--SASA_limit", default=None, type=float, dest="SASA_limit_max",
                        help="Inclusive upper bound on ligand relative SASA — passes when SASA_rel <= value (recommended: 0.20). Legacy alias: --SASA_limit.")
    parser.add_argument("--SASA_limit_min", default=None, type=float,
                        help="Inclusive lower bound on ligand relative SASA — passes when SASA_rel >= value.")

    parser.add_argument("--loop_limit_max", "--loop_limit", default=None, type=float, dest="loop_limit_max",
                        help="Inclusive upper bound on loop fraction — passes when loop_frac <= value (recommended: 0.30). Legacy alias: --loop_limit.")
    parser.add_argument("--loop_limit_min", default=None, type=float,
                        help="Inclusive lower bound on loop fraction — passes when loop_frac >= value.")

    parser.add_argument("--longest_helix", default=None, type=int,
                        help="Inclusive upper bound on longest helix length — passes when longest_helix <= value (recommended: 30).")

    parser.add_argument("--rog_max", "--rog", default=None, type=float, dest="rog_max",
                        help="Inclusive upper bound on radius of gyration — passes when rog <= value (recommended: 30.0). Legacy alias: --rog.")
    parser.add_argument("--rog_min", default=None, type=float,
                        help="Inclusive lower bound on radius of gyration — passes when rog >= value.")

    parser.add_argument("--term_limit", default=None, type=float,
                        help="Inclusive lower bound on terminus-to-ligand distance — passes when term_mindist >= value (recommended: 15.0).")
    parser.add_argument("--bondlen_dev", default=None, type=float,
                        help="Inclusive upper bound on sidechain bondlength deviation at motif residues — passes when bondlen_dev <= value (recommended: 0.1).")

    parser.add_argument("--exclude_clash_atoms", type=str, nargs="+", help="Ligand atom names to exclude from ligand clash checking.")
    parser.add_argument("--ligand_exposed_atoms", type=str, nargs="+", help="Ligand atoms used with --exposed_atom_SASA_min / --exposed_atom_SASA_max.")
    parser.add_argument("--exposed_atom_SASA_min", "--exposed_atom_SASA", default=None, type=float, dest="exposed_atom_SASA_min",
                        help="Inclusive lower bound on per-atom SASA for --ligand_exposed_atoms — passes when exposed SASA >= value. Legacy alias: --exposed_atom_SASA.")
    parser.add_argument("--exposed_atom_SASA_max", default=None, type=float,
                        help="Inclusive upper bound on per-atom SASA for --ligand_exposed_atoms — passes when exposed SASA <= value.")

    parser.add_argument("--cart_bonded", type=float, help="Inclusive upper bound on cart_bonded_avg at motif residues — passes when cart_bonded_avg <= value.")
    parser.add_argument("--fa_dun", type=float, help="Inclusive upper bound on fa_dun_avg at motif residues — passes when fa_dun_avg <= value.")
    parser.add_argument("--ref_catres", type=str, nargs="+", help="Catalytic residue positions in reference structure (e.g. A94-96).")
    parser.add_argument("--loop_catres", action="store_false", default=True, help="If True, filter out designs with catalytic residues between loops.")

    parser.add_argument("--scorefile_out", type=str, default="diffusion_analysis.sc", help="Output Rosetta-style scorefile.")
    parser.add_argument("--outdir", type=str, default="filtered_structures", help="Directory for filtered output PDBs.")
    parser.add_argument("--partial", action="store_true", default=False, help="Set if running on partial diffusion output.")
    parser.add_argument("--nproc", type=int, help="# of CPU cores used.")
    parser.add_argument("--fix_unmatched_remark_lines_to_lig", action="store_true", default=False, help="If set, try to fix REMARK 666 lines whose MATCH TEMPLATE target could not be mapped earlier by assigning them to a unique residue (protein or ligand) in the design with the same residue name3.")

    # ---- REMARK provenance lines injected into each filtered PDB --------------
    parser.add_argument("--no_design_path_remarks", action="store_true", default=False,
                        help="Disable the default REMARK DESIGN_PATH rfd3 input / output lines that are otherwise injected into each filtered output PDB.")
    parser.add_argument("--inject_output_json_remark", action="store_true", default=False,
                        help="Also inject a 'REMARK DESIGN_PATH rfd3 output_json <jsonfile>' line pointing at the pre-filter JSON.")
    parser.add_argument("--inject_select_fixed_atoms_remark", action="store_true", default=False,
                        help="Inject a single-line 'REMARK rfd3_property \"select_fixed_atoms\": {...}' carrying the dict from the rfd3 JSON.")
    parser.add_argument("--inject_diffused_index_map_remark", action="store_true", default=False,
                        help="Inject a single-line 'REMARK rfd3_property \"diffused_index_map\": {...}' carrying the dict from the rfd3 JSON.")

    parser.add_argument("--preserve_N_terminal_catres_rosetta_understanding", action="store_true", default=False, help="If set, preserve :NtermProteinFull on catalytic residues IF that residue is also a chain N-terminus in the TARGET pose.")
    parser.add_argument("--preserve_C_terminal_catres_rosetta_understanding", action="store_true", default=False, help="If set, preserve :CtermProteinFull on catalytic residues IF that residue is also a chain C-terminus in the TARGET pose.")

    parser.add_argument("--neutralize_charge_for_noncanonical_lysine_represented_as_lys", action="store_true", default=False, help="If set: for catalytic LYS in the reference, if NZ is within bonding distance " "to a heteroatom (O/N/S) on another residue or ligand, mutate the target residue " "to a neutral lysine residue type (e.g. LYN) instead of canonical charged LYS.")
    parser.add_argument("--lys_nz_bond_cutoff", type=float, default=1.9, help="Distance cutoff (Å) for deciding NZ is 'bonded' to another heteroatom.")

    args = parser.parse_args()
    main(args)
