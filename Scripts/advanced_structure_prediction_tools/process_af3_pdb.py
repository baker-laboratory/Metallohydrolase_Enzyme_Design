#!/usr/bin/env python3
"""
================================================================================
AlphaFold3 PDB Processing Script
================================================================================

Processes a directory of AlphaFold3 (AF3) predictions against a single reference
PDB structure and writes a one-row CSV of per-prediction structural metrics.

--------------------------------------------------------------------------------
MODES OF OPERATION
--------------------------------------------------------------------------------

HOLO mode  (default when --ligand_groups_json is supplied)
    Computes protein metrics + explicit per-ligand RMSD and ligand-group pLDDT
    using the atom mapping you provide in --ligand_groups_json.

APO  mode  (active when --ligand_groups_json is OMITTED)
    Skips ligand-matching RMSD/pLDDT computation but still extracts every
    ligand-independent metric that AF3 emits: ipTM, pTM, per-chain pLDDT (for
    ALL chains including any ligand chains), chain-pair PAE min/mean, CA RMSD,
    TM-score, catalytic-residue RMSD / pLDDT / lDDT, and ncAA metadata.

    Use APO mode when:
      - The input has no ligand at all (pure protein prediction).
      - The input HAS a ligand, but you only want iptm / ptm / PAE / per-chain
        pLDDT and don't want to bother writing a ligand atom mapping.

    iptm / ptm / chain_pair_pae_min are pulled via .get() with NaN fallback,
    so monomer predictions (where these may be absent) don't crash.

--------------------------------------------------------------------------------
METRICS COMPUTED (per AF3 prediction, suffixed with _idx_{N})
--------------------------------------------------------------------------------

Global confidence (from *_summary_confidences.json):
  iptm, ptm

Per-chain pLDDT (from *_confidences.json, for every chain in the AF3 PDB):
  chain{X}_plddt

Chain-pair PAE (computed from raw NxN PAE matrix by default, or from AF3
summary if --use_af3_summary_pae_min):
  chain{X}_chain{Y}_pair_pae_min               (asymmetric, all pairs)
  chain{X}_chain{Y}_pair_pae_min_avg           (symmetric average, off-diag)
  chain{X}_chain{Y}_pair_pae_min_min           (symmetric minimum, off-diag)
  chain{X}_chain{Y}_pair_pae_min_af3summary    (AF3's pre-computed values)
  chain{X}_chain{Y}_pair_pae_mean              (only if --calculate_avg_pae...)

Protein-structure metrics (Kabsch-aligned on CA atoms of reference):
  ca_rmsd                                      Kabsch CA RMSD
  ca_rmsd_TMalign                              TM-align-style superposition
  tm_score                                     length-independent fold similarity
  terminal_ignore_N / terminal_ignore_C        actual ignore counts used

Catalytic-residue metrics (one row per REMARK 666 entry):
  {AA}{i}_rmsd, {AA}{i}_rmsd_TMalign, {AA}{i}_plddt
  catres_rmsd, catres_rmsd_TMalign, catres_plddt, catres_lddt
  catres_count, catres_subset_count
  catres_subset_rmsd, catres_subset_rmsd_TMalign,
  catres_subset_plddt, catres_subset_lddt     (subset defined by --catres_subset)

Non-canonical amino acid (ncAA) metadata (emitted only when AF3 residue has
extra atoms vs reference):
  {AA}{i}_extra_atoms_count
  {AA}{i}_extra_atoms_plddt
  {AA}{i}_common_atoms_used

HOLO-only metrics (one per ligand group in --ligand_groups_json):
  {label}_rmsd, {label}_rmsd_TMalign, {label}_plddt

--------------------------------------------------------------------------------
EXPECTED INPUT LAYOUT
--------------------------------------------------------------------------------

--af3_dir should contain, for each prediction:
  *_model.pdb                     AF3 model coordinates (pLDDT in B-factor column)
  *_summary_confidences.json      ipTM, pTM, chain_pair_pae_min
  *_confidences.json              atom_plddts, atom_chain_ids, token-level PAE

--ref_pdb is a ground-truth / designed reference PDB that may contain:
  - REMARK 666 lines  (Rosetta enzyme-design convention; used as catalytic
    residues) in the format:
      REMARK 666 MATCH TEMPLATE X XXX 0 MATCH MOTIF <chain> <name3> <resnum> \\
                                                   <cst_block> <cst_idx>
  - Ligand atoms (referenced by --ligand_groups_json in HOLO mode)

--------------------------------------------------------------------------------
TERMINAL IGNORING  (--N_terminus_tag_length_to_ignore / --C_...)
--------------------------------------------------------------------------------
When set, the first N and/or last C PROTEIN residues (standard + modified AAs)
of each AF3 prediction are excluded from:
  - CA alignment
  - catalytic residue RMSD / pLDDT / lDDT lookups
  - TM-score / TM-align superposition
Ligand / HETATM residues are NEVER ignored. Residue numbering becomes
"effective" (residue 1 = first non-ignored protein residue), matching the
reference PDB's numbering. If the supplied counts cause a CA count mismatch
with the reference, the script automatically falls back through (N,C)
-> (0,0) -> (N,0) -> (0,C) and records which pair actually worked in
`terminal_ignore_N` / `terminal_ignore_C`.

Terminal ignoring is applied PER-CHAIN, and only to chains listed in
--ignore_n_term_chain / --ignore_c_term_chain (default: 'A' for both). The
pipeline assumes chain A is the designed protein and all other chains are
peptide substrates or ligands — those chains are never trimmed. If the
reference has additional protein chains (true multimer), the script logs a
prominent warning at startup naming the unconfigured chains and suggesting
a --ignore_{n,c}_term_chain override (e.g. 'A,B,D').

--------------------------------------------------------------------------------
SYMMETRIC ATOMS
--------------------------------------------------------------------------------
For residues with chemically equivalent atoms (PHE / TYR / ASP / GLU / ARG /
LEU / VAL), the RMSD / lDDT / TM-align code aligns on non-symmetric atoms,
compares both naming conventions, and keeps the lower-RMSD assignment.

For ligands, pass a `symmetric_atom_groups` block inside the ligand entry to
enumerate permutations — see examples below.

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
    python process_af3_pdb.py --af3_dir <path> --ref_pdb <path> --outscr <path> \\
        [--ligand_groups_json '<json>']                    # HOLO; omit for APO
        [--verbose] [--verbose_pae_matrix]
        [--calculate_avg_pae_in_addition_to_pair]
        [--N_terminus_tag_length_to_ignore N]
        [--C_terminus_tag_length_to_ignore C]
        [--ignore_n_term_chain A,B,D]              # default 'A'
        [--ignore_c_term_chain A,B,D]              # default 'A'
        [--catres_subset 1,3,5]
        [--use_af3_summary_pae_min] [--validate_pae_min]
        [--rapid_mode_skip_sc_if_found] [--no_strip_conect_master]

    See --help for per-flag documentation and worked examples.

Author: Woodbuse Lab
"""

# ==============================================================================
# SECTION 1: IMPORTS AND CONSTANTS
# ==============================================================================

from Bio import PDB
from datetime import datetime
from itertools import combinations, product
from numpy.linalg import norm
from typing import Dict, List, Tuple, Optional, Any
import argparse
import glob
import json
import numpy as np
import os
import pandas as pd
import sys

# Biotite imports for TM-score and lDDT calculations
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile

# PDB format column positions (0-indexed)
PDB_RECORD_TYPE_COLS = (0, 6)
PDB_ATOM_NAME_COLS = (12, 16)
PDB_RESIDUE_NAME_COLS = (17, 21)
PDB_CHAIN_ID_COLS = (21, 23)
PDB_RESIDUE_NUM_COLS = (23, 27)
PDB_PLDDT_COLS = (61, 67)

# Canonical backbone atom ordering for consistent RMSD comparisons
CANONICAL_BACKBONE_ORDER = ["N", "CA", "C", "O"]

# Standard amino acids for protein vs ligand distinction (used for terminal ignoring)
STANDARD_AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"
}

# Modified/non-canonical amino acids that should be treated as protein residues
# These have backbone atoms (CA) and participate in the protein chain
# Note: These may appear with hetero flag 'H_XXX' in some PDB files (e.g., AlphaFold3 output)
MODIFIED_AMINO_ACIDS = {
    "KCX",  # Carboxylated lysine (lysino-carboxylic acid)
    "MSE",  # Selenomethionine
    "SEC",  # Selenocysteine
    "PYL",  # Pyrrolysine
}

# All amino acids to consider as part of protein chain
PROTEIN_AMINO_ACIDS = STANDARD_AMINO_ACIDS | MODIFIED_AMINO_ACIDS

# Symmetric atom pairs for lDDT calculations
# These atoms are chemically equivalent due to rotational symmetry:
# - PHE/TYR: 180° ring flip exchanges CD1↔CD2 and CE1↔CE2
# - ASP/GLU: carboxylate oxygen swap
# - ARG: guanidinium terminal nitrogens
# - LEU/VAL: branched aliphatic carbons
SYMMETRIC_ATOM_PAIRS = {
    "PHE": [("CD1", "CD2"), ("CE1", "CE2")],
    "TYR": [("CD1", "CD2"), ("CE1", "CE2")],
    "ASP": [("OD1", "OD2")],
    "GLU": [("OE1", "OE2")],
    "ARG": [("NH1", "NH2")],
    "LEU": [("CD1", "CD2")],
    "VAL": [("CG1", "CG2")],
}


# ==============================================================================
# SECTION 2: LOGGING UTILITIES
# ==============================================================================

def log(msg: str, level: str = "INFO") -> None:
    """
    Print a timestamped log message to stdout.

    Args:
        msg: The message to log
        level: Log level (INFO, WARN, ERROR, DEBUG)
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] [{level}] {msg}", flush=True)


def vlog(verbose: bool, msg: str, level: str = "DEBUG") -> None:
    """
    Conditional verbose logging - only prints if verbose is True.

    Args:
        verbose: Whether to print the message
        msg: The message to log
        level: Log level (default: DEBUG)
    """
    if verbose:
        log(msg, level)


# ==============================================================================
# SECTION 3: PDB FILE I/O
# ==============================================================================

def load_pdb(file_path: str) -> PDB.Structure.Structure:
    """
    Load a PDB file using Biopython's PDBParser.

    Args:
        file_path: Path to the PDB file

    Returns:
        Biopython Structure object
    """
    parser = PDB.PDBParser(QUIET=True)
    structure_id = os.path.basename(file_path)
    return parser.get_structure(structure_id, file_path)


def save_pdb(structure: PDB.Structure.Structure, output_file: str) -> None:
    """
    Save a Biopython structure to a PDB file.

    Args:
        structure: Biopython Structure object
        output_file: Output file path
    """
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_file)


def rapid_skip_if_sc_present(af3_dir: str, enabled: bool) -> None:
    """
    Early exit if output score files already exist in the directory.

    This function enables a "rapid mode" where processing is skipped if
    any .sc (score) files are already present, avoiding redundant computation.

    Args:
        af3_dir: Directory containing AF3 predictions
        enabled: Whether rapid skip mode is enabled
    """
    if not enabled:
        return
    if not af3_dir or not os.path.isdir(af3_dir):
        return

    sc_hits = glob.glob(os.path.join(af3_dir, "*.sc"))
    if sc_hits:
        log(f"[RAPID] Found existing .sc file(s) in {af3_dir}:", "INFO")
        for p in sc_hits[:5]:
            print(f"        - {os.path.basename(p)}")
        if len(sc_hits) > 5:
            print(f"        ... (+{len(sc_hits) - 5} more)")
        log("[RAPID] Skipping processing and exiting cleanly (--rapid_mode_skip_sc_if_found).", "INFO")
        sys.exit(0)


def strip_conect_master_lines(pdb_paths: List[str], verbose: bool = False) -> int:
    """
    Remove CONECT and MASTER lines from PDB files in-place.

    These records are sometimes present in AF3 PDB files converted from CIF
    and are not needed for structural analysis. Only rewrites files that
    actually contain these lines.

    Args:
        pdb_paths: List of PDB file paths to clean
        verbose: Whether to log per-file details

    Returns:
        Number of files that were modified
    """
    modified = 0
    for pdb_path in pdb_paths:
        with open(pdb_path, 'r') as f:
            lines = f.readlines()
        cleaned = [l for l in lines if not l.startswith(("CONECT", "MASTER"))]
        if len(cleaned) < len(lines):
            with open(pdb_path, 'w') as f:
                f.writelines(cleaned)
            modified += 1
            vlog(verbose, f"Stripped {len(lines) - len(cleaned)} CONECT/MASTER lines from {os.path.basename(pdb_path)}")
    return modified


# ==============================================================================
# SECTION 4: COORDINATE EXTRACTION
# ==============================================================================

def is_protein_residue(residue) -> bool:
    """
    Check if a Biopython residue is a protein residue (standard or modified amino acid).

    This handles both standard amino acids (hetero flag ' ') and modified amino acids
    like KCX (which may have hetero flag 'H_KCX' in AlphaFold3 output).

    Args:
        residue: Biopython Residue object

    Returns:
        True if residue is a protein residue, False otherwise
    """
    resname = residue.get_resname()
    hetero_flag = residue.get_id()[0]

    # Standard amino acids should have hetero flag ' '
    if hetero_flag == ' ' and resname in STANDARD_AMINO_ACIDS:
        return True

    # Modified amino acids may have 'H_XXX' hetero flag but are still protein residues
    if resname in MODIFIED_AMINO_ACIDS:
        return True

    return False


# Default chain set that receives terminal-ignore trimming. The pipeline assumes
# chain A is the designed/predicted protein and any other chain is a peptide
# substrate or ligand that must NOT have its termini trimmed.
DEFAULT_TRIM_CHAINS = frozenset({"A"})


def _effective_ignore(
    chain_id: str,
    n_ignore: int,
    c_ignore: int,
    n_trim_chains: Optional[set] = None,
    c_trim_chains: Optional[set] = None
) -> Tuple[int, int]:
    """
    Return (effective_n_ignore, effective_c_ignore) for a specific chain.

    A chain is trimmed at its N-terminus only if it appears in n_trim_chains,
    and at its C-terminus only if it appears in c_trim_chains. This lets
    callers apply tag-stripping to the designed protein (chain A by default)
    without touching peptide substrates or ligand chains that share the model.

    Args:
        chain_id: Chain identifier (e.g., 'A', 'B')
        n_ignore: Global N-terminal ignore count
        c_ignore: Global C-terminal ignore count
        n_trim_chains: Chains where N-terminal ignoring applies (default: {'A'})
        c_trim_chains: Chains where C-terminal ignoring applies (default: {'A'})

    Returns:
        Tuple (effective_n_ignore, effective_c_ignore) where either is 0 if
        this chain is excluded from the corresponding trim set.
    """
    n_set = DEFAULT_TRIM_CHAINS if n_trim_chains is None else n_trim_chains
    c_set = DEFAULT_TRIM_CHAINS if c_trim_chains is None else c_trim_chains
    eff_n = n_ignore if chain_id in n_set else 0
    eff_c = c_ignore if chain_id in c_set else 0
    return eff_n, eff_c


def extract_ca_coords(
    structure: PDB.Structure.Structure,
    n_ignore: int = 0,
    c_ignore: int = 0,
    n_trim_chains: Optional[set] = None,
    c_trim_chains: Optional[set] = None
) -> np.ndarray:
    """
    Extract coordinates of all Calpha (CA) atoms for structural alignment.

    Terminal ignoring is applied per-chain: the first n_ignore and last c_ignore
    protein residues are skipped ONLY on chains listed in n_trim_chains /
    c_trim_chains (default: {'A'}). Non-listed chains (peptide substrates,
    additional protein chains, HETATM) are never trimmed.

    Args:
        structure: Biopython Structure object
        n_ignore: Number of protein residues at N-terminus to ignore (default: 0)
        c_ignore: Number of protein residues at C-terminus to ignore (default: 0)
        n_trim_chains: Chains where N-terminal ignoring applies (default: {'A'})
        c_trim_chains: Chains where C-terminal ignoring applies (default: {'A'})

    Returns:
        NumPy array of shape (N, 3) containing CA coordinates
    """
    ca_coords = []
    for model in structure:
        for chain in model:
            # Get effective ignore counts for THIS chain only
            eff_n, eff_c = _effective_ignore(
                chain.id, n_ignore, c_ignore, n_trim_chains, c_trim_chains
            )
            # Get protein residues (standard and modified amino acids)
            protein_residues = [r for r in chain.get_residues() if is_protein_residue(r)]
            # Apply terminal ignoring
            if eff_c > 0:
                effective_residues = protein_residues[eff_n:len(protein_residues) - eff_c]
            else:
                effective_residues = protein_residues[eff_n:]

            for residue in effective_residues:
                if 'CA' in residue:
                    ca_coords.append(residue['CA'].get_coord())
    return np.array(ca_coords)


def extract_coords_atom(
    structure: PDB.Structure.Structure,
    chain_id: str,
    residue_name: str,
    atom_name: str
) -> np.ndarray:
    """
    Extract coordinates of a specific atom within residues of a given name.

    Args:
        structure: Biopython Structure object
        chain_id: Chain identifier (e.g., 'A', 'B')
        residue_name: 3-letter residue name (e.g., 'ALA', 'YYE')
        atom_name: Atom name (e.g., 'CA', 'P1')

    Returns:
        NumPy array of matching atom coordinates
    """
    coords = []
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            for residue in chain:
                if residue.get_resname() == residue_name:
                    for atom in residue:
                        if atom.get_name() == atom_name:
                            coords.append(atom.get_coord())
    return np.array(coords)


def extract_coords_res(
    structure: PDB.Structure.Structure,
    chain_id: str,
    residue_num: int,
    n_ignore: int = 0,
    c_ignore: int = 0,
    n_trim_chains: Optional[set] = None,
    c_trim_chains: Optional[set] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract heavy-atom coordinates and names for a specific residue.

    When n_ignore/c_ignore are specified AND chain_id is in the corresponding
    trim set, residue_num is treated as an "effective" index (1-based) after
    skipping terminal residues. On chains NOT in the trim set, residue_num is
    an effective index into the untrimmed protein sequence of that chain.

    Excludes hydrogen atoms for more robust RMSD comparisons.

    Args:
        structure: Biopython Structure object
        chain_id: Chain identifier
        residue_num: Effective residue index (1-based) when using terminal ignoring
        n_ignore: Number of protein residues at N-terminus to ignore (default: 0)
        c_ignore: Number of protein residues at C-terminus to ignore (default: 0)
        n_trim_chains: Chains where N-terminal ignoring applies (default: {'A'})
        c_trim_chains: Chains where C-terminal ignoring applies (default: {'A'})

    Returns:
        Tuple of (coordinates array, list of atom names)
    """
    # Apply trim only if this chain is in the configured trim set
    eff_n, eff_c = _effective_ignore(
        chain_id, n_ignore, c_ignore, n_trim_chains, c_trim_chains
    )

    coords, names = [], []
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue

            # Get protein residues and apply terminal ignoring
            protein_residues = [r for r in chain.get_residues() if is_protein_residue(r)]
            if eff_c > 0:
                effective_residues = protein_residues[eff_n:len(protein_residues) - eff_c]
            else:
                effective_residues = protein_residues[eff_n:]

            # Find residue at effective index (1-based)
            for i, residue in enumerate(effective_residues, start=1):
                if i == residue_num:
                    for atom in residue:
                        if atom.element == "H":
                            continue
                        coords.append(atom.get_coord())
                        names.append(atom.get_name())
                    break

    return np.array(coords), names


def get_catalytic_residues(pdb_file: str, verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Parse catalytic residue information from REMARK 666 lines in a PDB file.

    REMARK 666 is a Rosetta/enzyme design convention for marking catalytic residues.
    Format: REMARK 666 MATCH TEMPLATE X XXX 0 MATCH MOTIF <chain> <resname> <resnum> <cst_block> <cst_idx>

    Args:
        pdb_file: Path to the PDB file
        verbose: Whether to print extracted residues

    Returns:
        List of dicts with keys: 'res_num', 'name3', 'chain', 'cst_block'
    """
    catalytic_residues = []

    with open(pdb_file, 'r') as file:
        for line in file:
            if line.startswith("ATOM"):
                break  # REMARK lines come before ATOM records
            if line.startswith("REMARK 666"):
                parts = line.split()
                chain = parts[9]
                residue_name = parts[10]
                residue_number = int(parts[11])
                # Constraint block number is parts[12]
                cst_block = int(parts[12]) if len(parts) > 12 else 0
                catalytic_residues.append({
                    'res_num': residue_number,
                    'name3': residue_name,
                    'chain': chain,
                    'cst_block': cst_block
                })

    if verbose or catalytic_residues:
        vlog(True, f"Catalytic residues from {os.path.basename(pdb_file)}: {catalytic_residues}", "INFO")

    return catalytic_residues


# ==============================================================================
# SECTION 5: STRUCTURAL ALIGNMENT (KABSCH ALGORITHM)
# ==============================================================================

def kabsch_alignment(
    coords1: np.ndarray,
    coords2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute optimal rotation matrix to align coords1 onto coords2 using Kabsch algorithm.

    The Kabsch algorithm finds the rotation matrix that minimizes RMSD between
    two paired sets of points. It uses Singular Value Decomposition (SVD) to
    find the optimal rotation.

    Args:
        coords1: Source coordinates, shape (N, 3)
        coords2: Target coordinates, shape (N, 3)

    Returns:
        Tuple of (rotation_matrix, centroid1, centroid2)
        - rotation_matrix: 3x3 rotation matrix
        - centroid1: Center of mass of coords1
        - centroid2: Center of mass of coords2
    """
    # Center the coordinates by subtracting the mean
    centroid1 = np.mean(coords1, axis=0)
    centroid2 = np.mean(coords2, axis=0)

    coords1_centered = coords1 - centroid1
    coords2_centered = coords2 - centroid2

    # Compute the covariance matrix
    H = np.dot(coords1_centered.T, coords2_centered)

    # Perform Singular Value Decomposition
    V, S, W = np.linalg.svd(H)

    # Ensure a proper rotation matrix (determinant should be +1, not -1)
    d = np.sign(np.linalg.det(V @ W))
    V[:, -1] *= d
    rotation_matrix = V @ W

    return rotation_matrix, centroid1, centroid2


def apply_rotation(
    coords: np.ndarray,
    rotation_matrix: np.ndarray,
    centroid1: np.ndarray,
    centroid2: np.ndarray
) -> np.ndarray:
    """
    Apply rotation matrix to coordinates, transforming from frame 1 to frame 2.

    Args:
        coords: Coordinates to transform, shape (N, 3)
        rotation_matrix: 3x3 rotation matrix from kabsch_alignment
        centroid1: Original centroid (subtracted before rotation)
        centroid2: Target centroid (added after rotation)

    Returns:
        Transformed coordinates
    """
    coords_centered = coords - centroid1
    coords_rotated = np.dot(coords_centered, rotation_matrix) + centroid2
    return coords_rotated


def calculate_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """
    Calculate Root Mean Square Deviation between two coordinate sets.

    RMSD = sqrt(mean(sum((coords1 - coords2)^2)))

    Args:
        coords1: First coordinate set, shape (N, 3)
        coords2: Second coordinate set, shape (N, 3)

    Returns:
        RMSD value in Angstroms
    """
    diff = coords1 - coords2
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))


# ==============================================================================
# SECTION 6: NON-CANONICAL AMINO ACID (ncAA) SUPPORT
# ==============================================================================

def order_common_atoms(names1: List[str], names2: List[str]) -> List[str]:
    """
    Return a canonical ordering of atoms present in both residues.

    Prioritizes backbone atoms (N, CA, C, O) first, then remaining atoms
    alphabetically. This ensures consistent RMSD calculations when comparing
    residues with different atom compositions (e.g., KCX vs LYS).

    Args:
        names1: Atom names from structure 1
        names2: Atom names from structure 2

    Returns:
        Ordered list of common atom names
    """
    s1, s2 = set(names1), set(names2)
    inter = list(s1 & s2)
    if not inter:
        return []

    # Backbone atoms first (if present), then the rest sorted alphabetically
    back = [a for a in CANONICAL_BACKBONE_ORDER if a in inter]
    rest = sorted([a for a in inter if a not in CANONICAL_BACKBONE_ORDER])
    return back + rest


def rmsd_on_common_atoms(
    coords1: np.ndarray,
    names1: List[str],
    coords2: np.ndarray,
    names2: List[str]
) -> Tuple[float, List[str], List[str], List[str]]:
    """
    Compute RMSD using only atoms present in both structures.

    Useful for comparing ncAAs or modified residues where atom sets differ.

    Args:
        coords1: Coordinates from structure 1
        names1: Atom names from structure 1
        coords2: Coordinates from structure 2
        names2: Atom names from structure 2

    Returns:
        Tuple of:
        - rmsd_value: RMSD computed on common atoms (or NaN if no overlap)
        - af3_only: Atoms only in structure 1
        - ref_only: Atoms only in structure 2
        - common: List of common atom names used
    """
    common = order_common_atoms(names1, names2)
    s1, s2 = set(names1), set(names2)
    af3_only = sorted(list(s1 - s2))
    ref_only = sorted(list(s2 - s1))

    if not common:
        return (float("nan"), af3_only, ref_only, [])

    idx1 = [names1.index(n) for n in common]
    idx2 = [names2.index(n) for n in common]
    return (calculate_rmsd(coords1[idx1], coords2[idx2]), af3_only, ref_only, common)


def rmsd_on_common_atoms_with_symmetry(
    coords1: np.ndarray,
    names1: List[str],
    coords2: np.ndarray,
    names2: List[str],
    resname: str
) -> Tuple[float, List[str], List[str], List[str], bool]:
    """
    Compute RMSD using only atoms present in both structures, with symmetric atom handling.

    For residues with chemically equivalent symmetric atoms (PHE, TYR, ASP, GLU, ARG,
    LEU, VAL), tries both atom name assignments and uses the one that gives lower RMSD.
    Uses per-residue alignment on non-symmetric atoms to determine the best assignment.

    Args:
        coords1: Coordinates from structure 1 (AF3)
        names1: Atom names from structure 1
        coords2: Coordinates from structure 2 (reference)
        names2: Atom names from structure 2
        resname: 3-letter residue name

    Returns:
        Tuple of:
        - rmsd_value: RMSD computed on common atoms (or NaN if no overlap)
        - af3_only: Atoms only in structure 1
        - ref_only: Atoms only in structure 2
        - common: List of common atom names used
        - swapped: Whether symmetric atoms were swapped
    """
    # If residue doesn't have symmetric atoms, use standard calculation
    if resname not in SYMMETRIC_ATOM_PAIRS:
        rmsd_val, af3_only, ref_only, common = rmsd_on_common_atoms(
            coords1, names1, coords2, names2
        )
        return rmsd_val, af3_only, ref_only, common, False

    # Get symmetric atom names for this residue type
    symmetric_atoms = set()
    for atom1, atom2 in SYMMETRIC_ATOM_PAIRS[resname]:
        symmetric_atoms.add(atom1)
        symmetric_atoms.add(atom2)

    # Build name-to-coord mappings
    name_to_coord1 = {names1[i]: coords1[i] for i in range(len(names1))}
    name_to_coord2 = {names2[i]: coords2[i] for i in range(len(names2))}

    # Find common atoms
    s1, s2 = set(names1), set(names2)
    common_all = s1 & s2
    af3_only = sorted(list(s1 - s2))
    ref_only = sorted(list(s2 - s1))

    if not common_all:
        return float("nan"), af3_only, ref_only, [], False

    # Separate into non-symmetric and symmetric atoms
    common_non_symmetric = [n for n in common_all if n not in symmetric_atoms]
    common_symmetric = [n for n in common_all if n in symmetric_atoms]

    # If not enough non-symmetric atoms for alignment, fall back to standard
    if len(common_non_symmetric) < 3 or not common_symmetric:
        rmsd_val, af3_only, ref_only, common = rmsd_on_common_atoms(
            coords1, names1, coords2, names2
        )
        return rmsd_val, af3_only, ref_only, common, False

    # Align using non-symmetric atoms (Kabsch)
    align_coords1 = np.array([name_to_coord1[n] for n in sorted(common_non_symmetric)])
    align_coords2 = np.array([name_to_coord2[n] for n in sorted(common_non_symmetric)])

    center1 = np.mean(align_coords1, axis=0)
    center2 = np.mean(align_coords2, axis=0)
    centered1 = align_coords1 - center1
    centered2 = align_coords2 - center2

    H = centered1.T @ centered2
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    def transform(coord):
        return R @ (coord - center1) + center2

    # Compare symmetric atoms RMSD with original vs swapped naming
    def swap_name(name):
        for a1, a2 in SYMMETRIC_ATOM_PAIRS[resname]:
            if name == a1:
                return a2
            elif name == a2:
                return a1
        return name

    # Original assignment: compare AF3 atom to ref atom with same name
    rmsd_sym_orig = 0.0
    for name in common_symmetric:
        af3_transformed = transform(name_to_coord1[name])
        rmsd_sym_orig += np.sum((af3_transformed - name_to_coord2[name]) ** 2)

    # Swapped assignment: compare AF3 atom to ref atom with swapped name
    rmsd_sym_swap = 0.0
    for name in common_symmetric:
        swapped = swap_name(name)
        if swapped in name_to_coord2:
            af3_transformed = transform(name_to_coord1[name])
            rmsd_sym_swap += np.sum((af3_transformed - name_to_coord2[swapped]) ** 2)
        else:
            rmsd_sym_swap += np.sum((transform(name_to_coord1[name]) - name_to_coord2[name]) ** 2)

    use_swap = rmsd_sym_swap < rmsd_sym_orig

    # Compute final RMSD with chosen assignment
    common_ordered = order_common_atoms(names1, names2)

    if use_swap:
        # Build swapped name mapping for coords1
        swapped_names1 = [swap_name(n) if n in symmetric_atoms else n for n in names1]
        common_ordered = order_common_atoms(swapped_names1, names2)
        idx1 = [swapped_names1.index(n) for n in common_ordered]
    else:
        idx1 = [names1.index(n) for n in common_ordered]

    idx2 = [names2.index(n) for n in common_ordered]
    rmsd_val = calculate_rmsd(coords1[idx1], coords2[idx2])

    return rmsd_val, af3_only, ref_only, common_ordered, use_swap


def get_resname_by_chain_resnum(
    structure: PDB.Structure.Structure,
    chain_id: str,
    resnum: int,
    n_ignore: int = 0,
    c_ignore: int = 0,
    n_trim_chains: Optional[set] = None,
    c_trim_chains: Optional[set] = None
) -> Optional[str]:
    """
    Look up the 3-letter residue name at a specific effective position.

    Terminal ignoring is only applied if chain_id is in the configured trim
    set (default: {'A'}); otherwise the raw protein sequence for that chain
    is used.

    Args:
        structure: Biopython Structure object
        chain_id: Chain identifier
        resnum: Effective residue index (1-based)
        n_ignore: Number of protein residues at N-terminus to ignore (default: 0)
        c_ignore: Number of protein residues at C-terminus to ignore (default: 0)
        n_trim_chains: Chains where N-terminal ignoring applies (default: {'A'})
        c_trim_chains: Chains where C-terminal ignoring applies (default: {'A'})

    Returns:
        3-letter residue name or None if not found
    """
    eff_n, eff_c = _effective_ignore(
        chain_id, n_ignore, c_ignore, n_trim_chains, c_trim_chains
    )

    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue

            # Get protein residues and apply terminal ignoring
            protein_residues = [r for r in chain.get_residues() if is_protein_residue(r)]
            if eff_c > 0:
                effective_residues = protein_residues[eff_n:len(protein_residues) - eff_c]
            else:
                effective_residues = protein_residues[eff_n:]

            # Find residue at effective index (1-based)
            if 1 <= resnum <= len(effective_residues):
                return effective_residues[resnum - 1].get_resname()

    return None


def get_raw_resseq_for_effective_index(
    structure: PDB.Structure.Structure,
    chain_id: str,
    eff_index: int,
    n_ignore: int = 0,
    c_ignore: int = 0,
    n_trim_chains: Optional[set] = None,
    c_trim_chains: Optional[set] = None
) -> Optional[int]:
    """
    Map an effective index back to the raw PDB residue sequence number.

    Terminal ignoring is only applied if chain_id is in the configured trim
    set (default: {'A'}).

    Args:
        structure: Biopython Structure object
        chain_id: Chain identifier
        eff_index: Effective residue index (1-based)
        n_ignore: Number of protein residues at N-terminus to ignore (default: 0)
        c_ignore: Number of protein residues at C-terminus to ignore (default: 0)
        n_trim_chains: Chains where N-terminal ignoring applies (default: {'A'})
        c_trim_chains: Chains where C-terminal ignoring applies (default: {'A'})

    Returns:
        Raw residue sequence number or None if not found
    """
    eff_n, eff_c = _effective_ignore(
        chain_id, n_ignore, c_ignore, n_trim_chains, c_trim_chains
    )

    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue

            protein_residues = [r for r in chain.get_residues() if is_protein_residue(r)]
            if eff_c > 0:
                effective_residues = protein_residues[eff_n:len(protein_residues) - eff_c]
            else:
                effective_residues = protein_residues[eff_n:]

            if 1 <= eff_index <= len(effective_residues):
                return effective_residues[eff_index - 1].get_id()[1]

    return None


# ==============================================================================
# SECTION 6B: TM-SCORE AND lDDT CALCULATIONS (using Biotite)
# ==============================================================================

def load_biotite_structure(pdb_path: str) -> Optional[struc.AtomArray]:
    """
    Load a PDB file as a Biotite AtomArray.

    Args:
        pdb_path: Path to PDB file

    Returns:
        Biotite AtomArray or None if loading fails
    """
    try:
        pdb_file = PDBFile.read(pdb_path)
        return pdb_file.get_structure(model=1)
    except Exception:
        return None


def _build_chain_residue_order(atoms: struc.AtomArray) -> Dict[str, List[int]]:
    """
    Build {chain_id: [res_id, ...]} with first-appearance order preserved.

    Residue IDs are de-duplicated WITHIN each chain; chains are kept separate
    so that overlapping residue numbers (chain A 1-100 and chain B 1-50)
    don't collide.
    """
    result: Dict[str, List[int]] = {}
    seen: Dict[str, set] = {}
    for ch, ri in zip(atoms.chain_id, atoms.res_id):
        ch_s = str(ch)
        ri_i = int(ri)
        if ch_s not in seen:
            seen[ch_s] = set()
            result[ch_s] = []
        if ri_i not in seen[ch_s]:
            seen[ch_s].add(ri_i)
            result[ch_s].append(ri_i)
    return result


def filter_biotite_protein_residues(
    atoms: struc.AtomArray,
    n_ignore: int = 0,
    c_ignore: int = 0,
    n_trim_chains: Optional[set] = None,
    c_trim_chains: Optional[set] = None
) -> struc.AtomArray:
    """
    Filter Biotite AtomArray to protein residues with per-chain terminal ignoring.

    Terminal ignoring is applied INDEPENDENTLY per chain, and ONLY to chains
    listed in n_trim_chains / c_trim_chains (default: {'A'}). Other chains
    (peptide substrates, additional protein chains, ligand chains) are kept
    in full.

    Residue IDs are de-duplicated within each chain (by first appearance), so
    overlapping numbering across chains (chain A 1-100 and chain B 1-50)
    doesn't collide.

    Args:
        atoms: Biotite AtomArray
        n_ignore: Global N-terminal ignore count
        c_ignore: Global C-terminal ignore count
        n_trim_chains: Chains where N-terminal ignoring applies (default: {'A'})
        c_trim_chains: Chains where C-terminal ignoring applies (default: {'A'})

    Returns:
        Filtered AtomArray (protein atoms only, with terminals trimmed on the
        configured chains).
    """
    # Filter to protein amino acids (standard + modified like KCX)
    aa_mask = np.isin(atoms.res_name, list(PROTEIN_AMINO_ACIDS))
    protein_atoms = atoms[aa_mask]

    if len(protein_atoms) == 0:
        return protein_atoms

    if n_ignore == 0 and c_ignore == 0:
        return protein_atoms

    # Build {chain_id: [res_id, ...]} in first-appearance order
    chain_res_order = _build_chain_residue_order(protein_atoms)

    # Per chain, apply terminal ignoring only if this chain is in the trim sets
    keep_by_chain: Dict[str, set] = {}
    for ch_s, res_list in chain_res_order.items():
        eff_n, eff_c = _effective_ignore(
            ch_s, n_ignore, c_ignore, n_trim_chains, c_trim_chains
        )
        if eff_c > 0:
            kept = res_list[eff_n:len(res_list) - eff_c]
        else:
            kept = res_list[eff_n:]
        keep_by_chain[ch_s] = set(kept)

    # Build boolean mask chain-by-chain (vectorized)
    keep_mask = np.zeros(len(protein_atoms), dtype=bool)
    for ch_s, kept_ids in keep_by_chain.items():
        if not kept_ids:
            continue
        chain_mask = (protein_atoms.chain_id == ch_s)
        res_mask = np.isin(protein_atoms.res_id, list(kept_ids))
        keep_mask |= (chain_mask & res_mask)

    return protein_atoms[keep_mask]


def calculate_tm_score(
    af3_pdb_path: str,
    ref_pdb_path: str,
    n_ignore: int = 0,
    c_ignore: int = 0,
    verbose: bool = False,
    n_trim_chains: Optional[set] = None,
    c_trim_chains: Optional[set] = None
) -> float:
    """
    Calculate TM-score between AF3 prediction and reference structure.

    TM-score is a length-independent metric ranging from 0 to 1, where:
    - TM-score > 0.5 generally indicates same fold
    - TM-score > 0.17 is better than random

    Uses biotite.structure.tm_score() for the calculation.

    Args:
        af3_pdb_path: Path to AF3 PDB file
        ref_pdb_path: Path to reference PDB file
        n_ignore: Number of protein residues at N-terminus to ignore in AF3
        c_ignore: Number of protein residues at C-terminus to ignore in AF3
        verbose: Whether to print debug information
        n_trim_chains: Chains where N-terminal ignoring applies (default: {'A'})
        c_trim_chains: Chains where C-terminal ignoring applies (default: {'A'})

    Returns:
        TM-score value (0-1)
    """
    # Load structures with biotite
    af3_atoms = load_biotite_structure(af3_pdb_path)
    ref_atoms = load_biotite_structure(ref_pdb_path)

    if af3_atoms is None or ref_atoms is None:
        vlog(verbose, "TM-score: Failed to load structures with biotite", "WARN")
        return float("nan")

    # Filter to protein with per-chain terminal ignoring
    af3_protein = filter_biotite_protein_residues(
        af3_atoms, n_ignore, c_ignore, n_trim_chains, c_trim_chains
    )
    ref_protein = filter_biotite_protein_residues(ref_atoms, 0, 0)

    # Filter to CA atoms only
    af3_ca = af3_protein[af3_protein.atom_name == "CA"]
    ref_ca = ref_protein[ref_protein.atom_name == "CA"]

    n_af3 = len(af3_ca)
    n_ref = len(ref_ca)

    if n_af3 == 0 or n_ref == 0:
        vlog(verbose, "TM-score: No CA atoms found", "WARN")
        return float("nan")

    # Ensure same number of atoms (truncate to minimum)
    n_atoms = min(n_af3, n_ref)
    if n_af3 != n_ref:
        vlog(verbose, f"TM-score: Truncating to {n_atoms} CA atoms (AF3={n_af3}, ref={n_ref})", "WARN")

    af3_ca = af3_ca[:n_atoms]
    ref_ca = ref_ca[:n_atoms]

    # Create index arrays (paired 1:1)
    indices = np.arange(n_atoms)

    try:
        # Superimpose AF3 onto reference (Kabsch algorithm minimizing RMSD)
        # This is required because biotite's tm_score expects pre-aligned structures
        af3_ca_superimposed, _ = struc.superimpose(ref_ca, af3_ca)

        # Calculate TM-score using biotite on superimposed structures
        tm_val = struc.tm_score(
            ref_ca,
            af3_ca_superimposed,
            indices,
            indices,
            reference_length=n_ref  # Normalize by reference length
        )
        vlog(verbose, f"TM-score: {tm_val:.4f} (n_CA={n_atoms})")
        return float(tm_val)
    except Exception as e:
        vlog(verbose, f"TM-score calculation failed: {e}", "WARN")
        return float("nan")


def _get_symmetric_atom_names(resname: str) -> set:
    """Get set of all symmetric atom names for a residue type."""
    if resname not in SYMMETRIC_ATOM_PAIRS:
        return set()
    symmetric = set()
    for atom1, atom2 in SYMMETRIC_ATOM_PAIRS[resname]:
        symmetric.add(atom1)
        symmetric.add(atom2)
    return symmetric


def _apply_swap_to_name(name: str, resname: str) -> str:
    """Apply symmetric swap to an atom name."""
    if resname not in SYMMETRIC_ATOM_PAIRS:
        return name
    for atom1, atom2 in SYMMETRIC_ATOM_PAIRS[resname]:
        if name == atom1:
            return atom2
        elif name == atom2:
            return atom1
    return name


def _determine_best_swap(
    af3_res_atoms,
    ref_res_atoms,
    resname: str,
    verbose: bool = False
) -> bool:
    """
    Determine whether to swap symmetric atoms for best correspondence.

    Aligns AF3 residue to reference using NON-symmetric atoms only,
    then compares RMSD of symmetric atoms under both naming conventions.

    Args:
        af3_res_atoms: Biotite AtomArray for AF3 residue
        ref_res_atoms: Biotite AtomArray for reference residue
        resname: 3-letter residue name
        verbose: Whether to print debug information

    Returns:
        True if swapped naming gives better correspondence, False otherwise
    """
    if resname not in SYMMETRIC_ATOM_PAIRS:
        return False

    symmetric_atoms = _get_symmetric_atom_names(resname)

    # Build name-to-coord mappings
    af3_name_to_coord = {af3_res_atoms.atom_name[i]: af3_res_atoms.coord[i]
                         for i in range(len(af3_res_atoms))}
    ref_name_to_coord = {ref_res_atoms.atom_name[i]: ref_res_atoms.coord[i]
                         for i in range(len(ref_res_atoms))}

    # Find common non-symmetric atoms for alignment
    common_names = set(af3_name_to_coord.keys()) & set(ref_name_to_coord.keys())
    non_symmetric_common = [n for n in common_names if n not in symmetric_atoms]

    if len(non_symmetric_common) < 3:
        # Not enough atoms for alignment, fall back to no swap
        return False

    # Collect coords for non-symmetric atoms (for alignment)
    af3_align_coords = np.array([af3_name_to_coord[n] for n in sorted(non_symmetric_common)])
    ref_align_coords = np.array([ref_name_to_coord[n] for n in sorted(non_symmetric_common)])

    # Kabsch alignment: find rotation that minimizes RMSD
    # Center both coordinate sets
    af3_center = np.mean(af3_align_coords, axis=0)
    ref_center = np.mean(ref_align_coords, axis=0)
    af3_centered = af3_align_coords - af3_center
    ref_centered = ref_align_coords - ref_center

    # Compute optimal rotation using SVD
    H = af3_centered.T @ ref_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Apply transformation to all AF3 coords
    def transform(coord):
        return R @ (coord - af3_center) + ref_center

    # Find common symmetric atoms
    symmetric_common = [n for n in common_names if n in symmetric_atoms]
    if not symmetric_common:
        return False

    # Compute RMSD for symmetric atoms with original naming
    rmsd_orig = 0.0
    for name in symmetric_common:
        af3_transformed = transform(af3_name_to_coord[name])
        ref_coord = ref_name_to_coord[name]
        rmsd_orig += np.sum((af3_transformed - ref_coord) ** 2)
    rmsd_orig = np.sqrt(rmsd_orig / len(symmetric_common))

    # Compute RMSD for symmetric atoms with swapped naming
    rmsd_swap = 0.0
    for name in symmetric_common:
        swapped_name = _apply_swap_to_name(name, resname)
        if swapped_name in ref_name_to_coord:
            af3_transformed = transform(af3_name_to_coord[name])
            ref_coord = ref_name_to_coord[swapped_name]
            rmsd_swap += np.sum((af3_transformed - ref_coord) ** 2)
        else:
            # Swapped name not in reference, can't compare
            rmsd_swap += np.sum((transform(af3_name_to_coord[name]) - ref_name_to_coord[name]) ** 2)
    rmsd_swap = np.sqrt(rmsd_swap / len(symmetric_common))

    return rmsd_swap < rmsd_orig


def _get_residue_coords_with_swap(
    af3_res_atoms,
    ref_res_atoms,
    resname: str,
    swap: bool = False
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """
    Get matched coordinates between AF3 and reference residue atoms.

    For residues with symmetric atoms (PHE, TYR, ASP, GLU, ARG, LEU, VAL),
    optionally swap the symmetric atom pairs in the AF3 structure.

    Args:
        af3_res_atoms: Biotite AtomArray for AF3 residue
        ref_res_atoms: Biotite AtomArray for reference residue
        resname: 3-letter residue name
        swap: If True, swap symmetric atom pairs in AF3 coords

    Returns:
        Tuple of (af3_coords, ref_coords, atom_names) for common atoms
    """
    # Build name-to-coord mapping for AF3 (with optional swapping)
    af3_name_to_coord = {}
    for i, name in enumerate(af3_res_atoms.atom_name):
        af3_name_to_coord[name] = af3_res_atoms.coord[i]

    # If swapping, create swapped mapping
    if swap and resname in SYMMETRIC_ATOM_PAIRS:
        swapped_mapping = {}
        for name, coord in af3_name_to_coord.items():
            swapped_name = _apply_swap_to_name(name, resname)
            swapped_mapping[swapped_name] = coord
        af3_name_to_coord = swapped_mapping

    # Build name-to-coord mapping for reference
    ref_name_to_coord = {}
    for i, name in enumerate(ref_res_atoms.atom_name):
        ref_name_to_coord[name] = ref_res_atoms.coord[i]

    # Find common atom names and collect coords
    common_names = set(af3_name_to_coord.keys()) & set(ref_name_to_coord.keys())

    af3_coords = []
    ref_coords = []
    atom_names = []

    for name in sorted(common_names):
        af3_coords.append(af3_name_to_coord[name])
        ref_coords.append(ref_name_to_coord[name])
        atom_names.append(name)

    return af3_coords, ref_coords, atom_names


def calculate_catres_lddt(
    af3_pdb_path: str,
    ref_pdb_path: str,
    catres_list: List[Dict],
    n_ignore: int = 0,
    c_ignore: int = 0,
    verbose: bool = False,
    n_trim_chains: Optional[set] = None,
    c_trim_chains: Optional[set] = None
) -> float:
    """
    Calculate lDDT for catalytic residues as a group.

    lDDT (local Distance Difference Test) measures the fraction of interatomic
    distances that are preserved within certain thresholds (0.5, 1, 2, 4 Å).

    This computes lDDT using all heavy atoms from ALL catalytic residues together,
    measuring inter-residue distances between catalytic residues. This reflects
    how well the overall active site geometry is preserved.

    For residues with chemically equivalent symmetric atoms (PHE, TYR, ASP, GLU,
    ARG, LEU, VAL), the function determines the best atom name assignment by:
    1. Aligning each residue using non-symmetric atoms only (Kabsch alignment)
    2. Comparing RMSD of symmetric atoms under both naming conventions
    3. Using the assignment that gives lower RMSD for the symmetric atoms

    Args:
        af3_pdb_path: Path to AF3 PDB file
        ref_pdb_path: Path to reference PDB file
        catres_list: List of catalytic residue definitions
        n_ignore: Number of protein residues at N-terminus to ignore in AF3
        c_ignore: Number of protein residues at C-terminus to ignore in AF3
        verbose: Whether to print debug information
        n_trim_chains: Chains where N-terminal ignoring applies (default: {'A'})
        c_trim_chains: Chains where C-terminal ignoring applies (default: {'A'})

    Returns:
        Single lDDT value (0-1) for the entire catalytic residue set
    """
    if not catres_list:
        return float("nan")

    # Load structures with biotite
    af3_atoms = load_biotite_structure(af3_pdb_path)
    ref_atoms = load_biotite_structure(ref_pdb_path)

    if af3_atoms is None or ref_atoms is None:
        vlog(verbose, "lDDT: Failed to load structures with biotite", "WARN")
        return float("nan")

    # Filter to protein with per-chain terminal ignoring
    af3_protein = filter_biotite_protein_residues(
        af3_atoms, n_ignore, c_ignore, n_trim_chains, c_trim_chains
    )
    ref_protein = filter_biotite_protein_residues(ref_atoms, 0, 0)

    # Chain-scoped effective residue maps: {chain_id: [res_id, ...]}
    # (ordered by first appearance so eff_idx -> res_id[eff_idx-1] works per chain)
    af3_chain_res = _build_chain_residue_order(af3_protein)
    ref_chain_res = _build_chain_residue_order(ref_protein)

    # Collect all heavy atoms from all catalytic residues
    all_af3_coords = []
    all_ref_coords = []
    atom_labels = []  # For debugging

    for catres in catres_list:
        eff_idx = catres["res_num"]  # 1-based effective index
        resname = catres["name3"]
        ch_id = catres["chain"]

        af3_res_list = af3_chain_res.get(ch_id, [])
        ref_res_list = ref_chain_res.get(ch_id, [])

        # Map effective index to actual residue ID on this specific chain
        if eff_idx < 1 or eff_idx > len(af3_res_list) or eff_idx > len(ref_res_list):
            vlog(verbose, f"  Catres {ch_id}:{eff_idx}: Out of range for chain "
                          f"(af3 n={len(af3_res_list)}, ref n={len(ref_res_list)}), skipping", "WARN")
            continue

        af3_target_res_id = af3_res_list[eff_idx - 1]
        ref_target_res_id = ref_res_list[eff_idx - 1]

        # Get atoms for this residue on THIS CHAIN (heavy atoms only)
        af3_res_mask = ((af3_protein.chain_id == ch_id) &
                        (af3_protein.res_id == af3_target_res_id) &
                        (af3_protein.element != "H"))
        ref_res_mask = ((ref_protein.chain_id == ch_id) &
                        (ref_protein.res_id == ref_target_res_id) &
                        (ref_protein.element != "H"))

        af3_res_atoms = af3_protein[af3_res_mask]
        ref_res_atoms = ref_protein[ref_res_mask]

        if len(af3_res_atoms) == 0 or len(ref_res_atoms) == 0:
            vlog(verbose, f"  Catres {ch_id}:{eff_idx}: No atoms found, skipping", "WARN")
            continue

        # For residues with symmetric atoms, determine best assignment using
        # per-residue alignment (align non-symmetric atoms, compare symmetric atoms)
        use_swapped = False
        if resname in SYMMETRIC_ATOM_PAIRS:
            use_swapped = _determine_best_swap(af3_res_atoms, ref_res_atoms, resname, verbose)
            if use_swapped:
                vlog(verbose, f"  Catres {resname}{eff_idx}: Using swapped symmetric atoms (per-residue alignment)")

        # Get coords with the chosen assignment
        af3_coords, ref_coords, names = _get_residue_coords_with_swap(
            af3_res_atoms, ref_res_atoms, resname, swap=use_swapped
        )

        # Add to overall lists
        for af3_c, ref_c, name in zip(af3_coords, ref_coords, names):
            all_af3_coords.append(af3_c)
            all_ref_coords.append(ref_c)
            atom_labels.append(f"{resname}{eff_idx}:{name}")

    if len(all_af3_coords) < 2:
        vlog(verbose, "lDDT: Not enough atoms across all catalytic residues", "WARN")
        return float("nan")

    all_af3_coords = np.array(all_af3_coords)
    all_ref_coords = np.array(all_ref_coords)

    # Compute lDDT on all catalytic residue atoms together
    try:
        lddt_val = _compute_simple_lddt(all_ref_coords, all_af3_coords)
        vlog(verbose, f"Catres lDDT: {lddt_val:.4f} ({len(all_af3_coords)} atoms from {len(catres_list)} residues)")
        return lddt_val
    except Exception as e:
        vlog(verbose, f"Catres lDDT calculation failed: {e}", "WARN")
        return float("nan")


def _compute_simple_lddt(
    ref_coords: np.ndarray,
    pred_coords: np.ndarray,
    inclusion_radius: float = 15.0,
    thresholds: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
) -> float:
    """
    Compute simplified lDDT between reference and predicted coordinates.

    lDDT measures the fraction of pairwise distances that are preserved
    within certain thresholds.

    Args:
        ref_coords: Reference coordinates (N x 3)
        pred_coords: Predicted coordinates (N x 3)
        inclusion_radius: Only consider atom pairs within this distance in reference
        thresholds: Distance deviation thresholds for scoring

    Returns:
        lDDT score (0-1)
    """
    n_atoms = len(ref_coords)
    if n_atoms < 2:
        return float("nan")

    # Compute pairwise distances
    ref_dists = np.zeros((n_atoms, n_atoms))
    pred_dists = np.zeros((n_atoms, n_atoms))

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            ref_dists[i, j] = np.linalg.norm(ref_coords[i] - ref_coords[j])
            ref_dists[j, i] = ref_dists[i, j]
            pred_dists[i, j] = np.linalg.norm(pred_coords[i] - pred_coords[j])
            pred_dists[j, i] = pred_dists[i, j]

    # Find pairs within inclusion radius in reference
    mask = (ref_dists > 0) & (ref_dists < inclusion_radius)
    n_pairs = np.sum(mask) // 2  # Each pair counted twice

    if n_pairs == 0:
        return float("nan")

    # Compute distance deviations
    deviations = np.abs(ref_dists - pred_dists)

    # Score: fraction of distances preserved within each threshold
    scores = []
    for thresh in thresholds:
        preserved = np.sum((deviations < thresh) & mask) // 2
        scores.append(preserved / n_pairs)

    # lDDT is the average across thresholds
    return float(np.mean(scores))


# ==============================================================================
# SECTION 7: SYMMETRIC ATOM HANDLING
# ==============================================================================

def permute_symmetric_atoms(
    coords1: np.ndarray,
    coords2: np.ndarray,
    atom_names1: List[str],
    atom_names2: List[str],
    symmetric_groups: List[Dict]
) -> float:
    """
    Calculate minimum RMSD considering symmetric atom permutations.

    Many ligands have chemically equivalent atoms that can be permuted without
    changing the molecule's identity (e.g., the two oxygens in a carboxylate).
    This function tries all valid permutations and returns the minimum RMSD.

    Args:
        coords1: Coordinates from structure 1
        coords2: Coordinates from structure 2
        atom_names1: Atom names from structure 1
        atom_names2: Atom names from structure 2
        symmetric_groups: List of dicts defining symmetric atom groups:
            {
                "lig1": [["atom1", "atom2"], ...],  # Subgroups in ligand 1
                "lig2": [[["perm1_atom1", "perm1_atom2"], ["perm2_atom1", "perm2_atom2"]], ...]
            }

    Returns:
        Minimum RMSD across all valid permutations
    """
    name_to_idx1 = {name: i for i, name in enumerate(atom_names1)}
    name_to_idx2 = {name: i for i, name in enumerate(atom_names2)}
    min_rmsd = float("inf")

    for group in symmetric_groups:
        atoms1_subgroups = group["lig1"]
        atoms2_permutations_per_subgroup = group["lig2"]

        if len(atoms1_subgroups) != len(atoms2_permutations_per_subgroup):
            raise ValueError("Mismatch in number of lig1 subgroups and lig2 permutation sets")

        # Create all combinations of one permutation per subgroup
        subgroup_perm_combos = product(*atoms2_permutations_per_subgroup)

        for atoms2_perm_combo in subgroup_perm_combos:
            coords1_concat = []
            coords2_concat = []

            for atoms1_subgroup, atoms2_perm in zip(atoms1_subgroups, atoms2_perm_combo):
                idxs1 = [name_to_idx1[a] for a in atoms1_subgroup]
                idxs2 = [name_to_idx2[a] for a in atoms2_perm]
                coords1_concat.extend(coords1[idxs1])
                coords2_concat.extend(coords2[idxs2])

            coords1_array = np.array(coords1_concat)
            coords2_array = np.array(coords2_concat)

            if coords1_array.shape != coords2_array.shape:
                raise ValueError("Mismatch in coords shape during symmetric RMSD calculation")

            rmsd = calculate_rmsd(coords1_array, coords2_array)
            min_rmsd = min(min_rmsd, rmsd)

    return min_rmsd


# ==============================================================================
# SECTION 8: pLDDT CONFIDENCE METRICS
# ==============================================================================

def get_plddt_per_residue_dic(
    af3_pdb: str,
    n_ignore: int = 0,
    c_ignore: int = 0,
    n_trim_chains: Optional[set] = None,
    c_trim_chains: Optional[set] = None
) -> Dict[Tuple[str, int], float]:
    """
    Parse per-residue average pLDDT from an AF3 PDB file.

    pLDDT (predicted Local Distance Difference Test) is stored in the B-factor
    column of AF3 PDB files. Values range from 0-100:
    - >90: Very high confidence
    - 70-90: High confidence
    - 50-70: Low confidence
    - <50: Very low confidence (often disordered regions)

    Terminal ignoring is applied per-chain and only to chains listed in
    n_trim_chains / c_trim_chains (default: {'A'}). Non-listed chains use
    their full protein sequence; ligand/HETATM residues always use raw
    residue numbers.

    Args:
        af3_pdb: Path to AF3 PDB file
        n_ignore: Global N-terminal ignore count
        c_ignore: Global C-terminal ignore count
        n_trim_chains: Chains where N-terminal ignoring applies (default: {'A'})
        c_trim_chains: Chains where C-terminal ignoring applies (default: {'A'})

    Returns:
        Dict mapping (chain_id, effective_res_num) to average pLDDT for proteins,
        or (chain_id, raw_res_num) for ligands
    """
    # First pass: identify protein residue sequences per chain
    chain_protein_resids: Dict[str, List[int]] = {}

    with open(af3_pdb, 'r') as f:
        for line in f:
            if len(line.strip()) < PDB_PLDDT_COLS[1]:
                continue
            record_type = line[PDB_RECORD_TYPE_COLS[0]:PDB_RECORD_TYPE_COLS[1]]
            res_name = line[PDB_RESIDUE_NAME_COLS[0]:PDB_RESIDUE_NAME_COLS[1]].strip()

            # Include ATOM records with standard amino acids
            # AND HETATM records with modified amino acids (like KCX)
            is_protein = False
            if record_type == "ATOM  " and res_name in STANDARD_AMINO_ACIDS:
                is_protein = True
            elif record_type == "HETATM" and res_name in MODIFIED_AMINO_ACIDS:
                is_protein = True

            if not is_protein:
                continue

            ch_id = line[PDB_CHAIN_ID_COLS[0]:PDB_CHAIN_ID_COLS[1]].strip()
            res_num = int(line[PDB_RESIDUE_NUM_COLS[0]:PDB_RESIDUE_NUM_COLS[1]].strip())
            chain_protein_resids.setdefault(ch_id, []).append(res_num)

    # Convert to sorted unique lists
    for ch_id in chain_protein_resids:
        chain_protein_resids[ch_id] = sorted(set(chain_protein_resids[ch_id]))

    # Build effective index map: (chain, raw_resseq) -> effective_index
    # Terminal ignoring applies only to chains in the configured trim sets.
    eff_num_map: Dict[Tuple[str, int], int] = {}
    for ch_id, res_list in chain_protein_resids.items():
        eff_n, eff_c = _effective_ignore(
            ch_id, n_ignore, c_ignore, n_trim_chains, c_trim_chains
        )
        if eff_c > 0:
            real_subset = res_list[eff_n:len(res_list) - eff_c]
        else:
            real_subset = res_list[eff_n:]
        for i, raw_res_seq in enumerate(real_subset, start=1):
            eff_num_map[(ch_id, raw_res_seq)] = i

    # Second pass: collect pLDDT values
    plddt_dic: Dict[Tuple[str, int], Dict[str, float]] = {}

    with open(af3_pdb, 'r') as f:
        for line in f:
            record_type = line[PDB_RECORD_TYPE_COLS[0]:PDB_RECORD_TYPE_COLS[1]]
            if record_type not in ["ATOM  ", "HETATM"]:
                continue
            if len(line.strip()) < PDB_PLDDT_COLS[1]:
                continue

            ch_id = line[PDB_CHAIN_ID_COLS[0]:PDB_CHAIN_ID_COLS[1]].strip()
            raw_res_num = int(line[PDB_RESIDUE_NUM_COLS[0]:PDB_RESIDUE_NUM_COLS[1]].strip())
            res_name = line[PDB_RESIDUE_NAME_COLS[0]:PDB_RESIDUE_NAME_COLS[1]].strip()
            atom_name = line[PDB_ATOM_NAME_COLS[0]:PDB_ATOM_NAME_COLS[1]].strip()
            plddt = float(line[PDB_PLDDT_COLS[0]:PDB_PLDDT_COLS[1]])

            # Determine key based on protein vs ligand
            # Protein: ATOM + standard AA, or HETATM + modified AA (like KCX)
            is_protein = (record_type == "ATOM  " and res_name in STANDARD_AMINO_ACIDS) or \
                         (record_type == "HETATM" and res_name in MODIFIED_AMINO_ACIDS)

            if is_protein:
                # Protein: use effective index, skip if outside effective range
                if (ch_id, raw_res_num) not in eff_num_map:
                    continue
                key = (ch_id, eff_num_map[(ch_id, raw_res_num)])
            else:
                # Ligand/HETATM (non-modified): use raw residue number
                key = (ch_id, raw_res_num)

            if key not in plddt_dic:
                plddt_dic[key] = {}
            plddt_dic[key][atom_name] = plddt

    # Average pLDDT across all atoms in each residue
    plddt_per_residue = {key: np.average(list(plddt_dic[key].values())) for key in plddt_dic}
    return plddt_per_residue


def get_avr_plddt(af3_pdb: str, ligand_groups_json: List[Dict]) -> List[float]:
    """
    Calculate average pLDDT for each ligand group.

    Args:
        af3_pdb: Path to AF3 PDB file
        ligand_groups_json: List of ligand group definitions

    Returns:
        List of average pLDDT values, one per ligand group
    """
    avr_plddt_list = []

    with open(af3_pdb, 'r') as f:
        pdb_lines = f.readlines()

    for ligand in ligand_groups_json:
        plddt_list = []
        for line in pdb_lines:
            record_type = line[PDB_RECORD_TYPE_COLS[0]:PDB_RECORD_TYPE_COLS[1]]
            if record_type not in ["ATOM  ", "HETATM"]:
                continue
            if len(line.strip()) < PDB_PLDDT_COLS[1]:
                continue

            atom = line[PDB_ATOM_NAME_COLS[0]:PDB_ATOM_NAME_COLS[1]].strip()
            name3 = line[PDB_RESIDUE_NAME_COLS[0]:PDB_RESIDUE_NAME_COLS[1]].strip()
            chain = line[PDB_CHAIN_ID_COLS[0]:PDB_CHAIN_ID_COLS[1]].strip()
            plddt = float(line[PDB_PLDDT_COLS[0]:PDB_PLDDT_COLS[1]])

            if (name3 == ligand["name3"][0] and
                atom in ligand["atoms"][0] and
                chain == ligand["chain"][0]):
                plddt_list.append(plddt)

        avr_plddt_list.append(np.mean(plddt_list) if plddt_list else float("nan"))

    return avr_plddt_list


def get_plddt_for_specific_atoms(
    pdb_path: str,
    chain_id: str,
    resnum: int,
    atom_names: List[str]
) -> float:
    """
    Get average pLDDT for specific atoms in a residue.

    Args:
        pdb_path: Path to PDB file
        chain_id: Chain identifier
        resnum: Residue number
        atom_names: List of atom names to average

    Returns:
        Average pLDDT or NaN if no atoms found
    """
    if not atom_names:
        return float("nan")

    plddts = []
    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            if len(line.strip()) < PDB_PLDDT_COLS[1]:
                continue

            ch = line[PDB_CHAIN_ID_COLS[0]:PDB_CHAIN_ID_COLS[1]].strip()
            try:
                rn = int(line[PDB_RESIDUE_NUM_COLS[0]:PDB_RESIDUE_NUM_COLS[1]].strip())
            except ValueError:
                continue

            if ch != chain_id or rn != resnum:
                continue

            atom = line[PDB_ATOM_NAME_COLS[0]:PDB_ATOM_NAME_COLS[1]].strip()
            if atom in atom_names:
                try:
                    plddts.append(float(line[PDB_PLDDT_COLS[0]:PDB_PLDDT_COLS[1]]))
                except ValueError:
                    pass

    return float(np.mean(plddts)) if plddts else float("nan")


# ==============================================================================
# SECTION 9: PAE (PREDICTED ALIGNED ERROR) ANALYSIS
# ==============================================================================

def compute_interchain_pae_means_from_conf(
    conf_json_path: str,
    valid_chains: Optional[List[str]] = None,
    verbose: bool = False,
    verbose_matrix: bool = False,
    label: Optional[str] = None
) -> Dict[Tuple[str, str], float]:
    """
    Compute mean interchain PAE from AF3's token-level PAE matrix.

    This function reads the NxN PAE matrix from *_confidences.json where N is
    the total number of tokens (residues/atoms) across all chains. For each
    unique chain pair (A, B), it computes the mean PAE across all token pairs
    where one token belongs to chain A and the other to chain B.

    PAE Interpretation:
    - PAE[i][j] = predicted error in position of token j when aligned using
      the frame of token i (in Angstroms)
    - Lower values indicate higher confidence in relative positioning
    - Typically: PAE < 5A suggests confident interaction
                 PAE > 20A suggests no interaction

    Algorithm:
    1. Load the NxN PAE matrix and token_chain_ids list
    2. For each unique chain pair (A, B) where A < B alphabetically:
       - Create boolean mask selecting A->B and B->A entries
       - Extract PAE values at masked positions
       - Compute mean (ignoring NaN values)

    Args:
        conf_json_path: Path to AF3's *_confidences.json file
        valid_chains: Optional list of chain IDs to include (default: all chains)
        verbose: If True, print compact PAE summaries per chain pair
        verbose_matrix: If True, print full NxN PAE matrices (very large output)
        label: Optional label for verbose output

    Returns:
        Dict mapping (chain_a, chain_b) tuples to mean PAE values,
        where chain_a < chain_b alphabetically. Empty dict on error.
    """
    try:
        with open(conf_json_path, "r") as f:
            data = json.load(f)
    except Exception:
        return {}

    pae_list = data.get("pae")
    token_chains_list = data.get("token_chain_ids")
    if pae_list is None or token_chains_list is None:
        return {}

    # Build arrays (None -> NaN for safety)
    pae = np.array(
        [[(float(x) if x is not None else np.nan) for x in row] for row in pae_list],
        dtype=float,
    )
    token_chains = np.array(token_chains_list, dtype=str)

    # Validate shapes
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1] or len(token_chains) != pae.shape[0]:
        return {}

    # Determine chain IDs to consider
    chain_set = sorted(set(token_chains.tolist()))
    if valid_chains is not None:
        valid = set(map(str, valid_chains))
        chain_list = [c for c in chain_set if c in valid]
    else:
        chain_list = chain_set

    # Labels for verbose output
    labels = [f"{i}:{c}" for i, c in enumerate(token_chains)]

    out = {}
    with np.errstate(all="ignore"):
        for a, b in combinations(chain_list, 2):
            # Create boolean masks for chain pair selection
            # rows_a: True where token belongs to chain A (Nx1 for broadcasting)
            # cols_b: True where token belongs to chain B (1xN for broadcasting)
            rows_a = (token_chains == a)[:, None]  # Nx1
            cols_b = (token_chains == b)[None, :]  # 1xN
            rows_b = (token_chains == b)[:, None]
            cols_a = (token_chains == a)[None, :]

            # Mask selects both A->B and B->A entries (symmetric)
            mask = (rows_a & cols_b) | (rows_b & cols_a)
            vals = pae[mask]
            finite = np.isfinite(vals)
            count = int(np.count_nonzero(finite))
            sum_val = float(np.nansum(vals)) if vals.size > 0 else float("nan")
            mean_val = float(np.nanmean(vals)) if count > 0 else float("nan")

            out[(a, b)] = mean_val

            if verbose or verbose_matrix:
                _print_verbose_pae_analysis(
                    pae, token_chains, labels, mask, a, b,
                    sum_val, count, mean_val, label, conf_json_path,
                    print_full_matrix=verbose_matrix
                )

    return out


def _print_verbose_pae_analysis(
    pae: np.ndarray,
    token_chains: np.ndarray,
    labels: List[str],
    mask: np.ndarray,
    chain_a: str,
    chain_b: str,
    sum_val: float,
    count: int,
    mean_val: float,
    label: Optional[str],
    conf_json_path: str,
    print_full_matrix: bool = False
) -> None:
    """
    Helper function to print verbose PAE analysis output.

    Args:
        print_full_matrix: If True, prints the full NxN matrices (can be huge).
                          If False, prints only a compact summary.
    """
    if print_full_matrix:
        # Full matrix output (only when explicitly requested)
        try:
            with pd.option_context(
                "display.max_rows", None,
                "display.max_columns", None,
                "display.width", 500,
                "display.float_format", lambda x: f"{x:6.2f}"
            ):
                print("\n" + "=" * 80)
                print(f"[MATRIX] Token-level PAE  |  Pair {chain_a}-{chain_b}  |  "
                      f"{label or os.path.basename(conf_json_path)}")
                print("- token_chain_ids:")
                print(" ".join(token_chains.tolist()))

                df_pae = pd.DataFrame(pae, index=labels, columns=labels)
                print("\n- PAE matrix:")
                print(df_pae)

                df_mask = pd.DataFrame(mask.astype(int), index=labels, columns=labels)
                print("\n- Boolean mask (1==selected AB/BA):")
                print(df_mask)

                sliced = np.full_like(pae, np.nan, dtype=float)
                sliced[mask] = pae[mask]
                df_sliced = pd.DataFrame(sliced, index=labels, columns=labels).fillna("")
                print("\n- Sliced matrix (only AB/BA shown; others blank):")
                print(df_sliced)

                print(f"\n- Sum over mask: {sum_val:.6g}  |  Count: {count}  |  Mean: {mean_val:.6g}")
                print("=" * 80)
        except Exception:
            print(f"[MATRIX] Pair {chain_a}-{chain_b}: Sum={sum_val:.4f}, Count={count}, Mean={mean_val:.4f}")
    else:
        # Compact summary output (default for --verbose)
        vals = pae[mask]
        min_val = float(np.nanmin(vals)) if vals.size > 0 else float("nan")
        max_val = float(np.nanmax(vals)) if vals.size > 0 else float("nan")
        vlog(True, f"    {chain_a}<->{chain_b}: mean={mean_val:.2f}, min={min_val:.2f}, max={max_val:.2f}, n={count}", "DEBUG")


def compute_interchain_pae_min_from_conf(
    conf_json_path: str,
    valid_chains: Optional[List[str]] = None,
    verbose: bool = False,
    label: Optional[str] = None
) -> Dict[Tuple[str, str], float]:
    """
    Compute minimum interchain PAE from AF3's token-level PAE matrix.

    This function computes PAE min directly from the raw NxN PAE matrix,
    which can be used to validate AF3's pre-computed chain_pair_pae_min values.

    PAE Min Definition:
    - For chain pair (A, B): min(PAE[i][j]) for all tokens i in chain A, j in chain B
    - This is ASYMMETRIC: PAE_min(A→B) may differ from PAE_min(B→A)

    Args:
        conf_json_path: Path to AF3's *_confidences.json file
        valid_chains: Optional list of chain IDs to include (default: all chains)
        verbose: If True, print detailed PAE min calculations
        label: Optional label for verbose output

    Returns:
        Dict mapping (chain_a, chain_b) tuples to min PAE values.
        Note: Returns BOTH directions, so (A,B) and (B,A) are separate keys.
        Empty dict on error.
    """
    try:
        with open(conf_json_path, "r") as f:
            data = json.load(f)
    except Exception:
        return {}

    pae_list = data.get("pae")
    token_chains_list = data.get("token_chain_ids")
    if pae_list is None or token_chains_list is None:
        return {}

    # Build arrays (None -> NaN for safety)
    pae = np.array(
        [[(float(x) if x is not None else np.nan) for x in row] for row in pae_list],
        dtype=float,
    )
    token_chains = np.array(token_chains_list, dtype=str)

    # Validate shapes
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1] or len(token_chains) != pae.shape[0]:
        return {}

    # Determine chain IDs to consider
    chain_set = sorted(set(token_chains.tolist()))
    if valid_chains is not None:
        valid = set(map(str, valid_chains))
        chain_list = [c for c in chain_set if c in valid]
    else:
        chain_list = chain_set

    out = {}
    with np.errstate(all="ignore"):
        # Compute for ALL ordered pairs (including A→A diagonal)
        for a in chain_list:
            for b in chain_list:
                # Create mask for A→B: rows from chain A, columns from chain B
                rows_a = (token_chains == a)[:, None]  # Nx1
                cols_b = (token_chains == b)[None, :]  # 1xN
                mask = rows_a & cols_b

                vals = pae[mask]
                if vals.size > 0:
                    min_val = float(np.nanmin(vals))
                else:
                    min_val = float("nan")

                out[(a, b)] = min_val

                if verbose:
                    count = int(np.count_nonzero(np.isfinite(vals)))
                    vlog(True, f"  PAE min {a}→{b}: {min_val:.4f} (from {count} token pairs)", "DEBUG")

    return out


def validate_pae_min_against_summary(
    conf_json_path: str,
    summary_conf_path: str,
    chain_ids: List[str],
    tolerance: float = 0.01,
    verbose: bool = False
) -> Tuple[bool, List[str]]:
    """
    Validate AF3's pre-computed chain_pair_pae_min against raw matrix calculation.

    This function computes PAE min from the raw NxN matrix and compares it
    to the pre-computed values in summary_confidences.json.

    Args:
        conf_json_path: Path to *_confidences.json (contains raw PAE matrix)
        summary_conf_path: Path to *_summary_confidences.json (contains pre-computed PAE min)
        chain_ids: List of chain IDs in order matching the summary matrix
        tolerance: Maximum allowed difference (default: 0.01 Angstroms)
        verbose: If True, print comparison details

    Returns:
        Tuple of (all_match: bool, discrepancies: List[str])
        - all_match: True if all values match within tolerance
        - discrepancies: List of human-readable discrepancy descriptions
    """
    discrepancies = []

    # Load summary confidences
    try:
        with open(summary_conf_path, "r") as f:
            summary_data = json.load(f)
        summary_pae_min = summary_data.get("chain_pair_pae_min", [])
    except Exception as e:
        return False, [f"Failed to load summary confidences: {e}"]

    # Compute PAE min from raw matrix
    computed_pae_min = compute_interchain_pae_min_from_conf(
        conf_json_path, valid_chains=chain_ids, verbose=False
    )

    if not computed_pae_min:
        return False, ["Failed to compute PAE min from raw matrix"]

    if verbose:
        log("Validating PAE min: computed from raw matrix vs AF3 summary", "INFO")

    # Compare each chain pair
    all_match = True
    for i, ch_i in enumerate(chain_ids):
        for j, ch_j in enumerate(chain_ids):
            # Get AF3's pre-computed value
            try:
                af3_val = summary_pae_min[i][j]
                if af3_val is None:
                    af3_val = float("nan")
            except (IndexError, TypeError):
                af3_val = float("nan")

            # Get our computed value
            computed_val = computed_pae_min.get((ch_i, ch_j), float("nan"))

            # Compare
            if np.isnan(af3_val) and np.isnan(computed_val):
                match = True
            elif np.isnan(af3_val) or np.isnan(computed_val):
                match = False
            else:
                match = abs(af3_val - computed_val) <= tolerance

            if verbose:
                status = "OK" if match else "MISMATCH"
                vlog(True, f"  {ch_i}→{ch_j}: AF3={af3_val:.4f}, computed={computed_val:.4f} [{status}]", "DEBUG")

            if not match:
                all_match = False
                discrepancies.append(
                    f"{ch_i}→{ch_j}: AF3={af3_val:.4f} vs computed={computed_val:.4f} "
                    f"(diff={abs(af3_val - computed_val):.4f})"
                )

    if verbose:
        if all_match:
            log("PAE min validation PASSED: All values match within tolerance", "INFO")
        else:
            log(f"PAE min validation FAILED: {len(discrepancies)} discrepancies found", "WARN")

    return all_match, discrepancies


# ==============================================================================
# SECTION 10: MAIN RMSD CALCULATION PIPELINE
# ==============================================================================

def calculate_rmsd_after_alignment(
    structure1: PDB.Structure.Structure,
    structure2: PDB.Structure.Structure,
    catres_list: List[Dict],
    ligand_atom_mapping_list: List[Dict],
    verbose: bool = False,
    n_ignore: int = 0,
    c_ignore: int = 0,
    n_trim_chains: Optional[set] = None,
    c_trim_chains: Optional[set] = None
) -> Tuple[float, List[float], List[float], List[Dict]]:
    """
    Align two structures and calculate various RMSD metrics.

    This is the main RMSD calculation pipeline that:
    1. Aligns structure1 onto structure2 using Kabsch algorithm on CA atoms
    2. Calculates CA RMSD between aligned structures
    3. Calculates RMSD for each catalytic residue (ncAA-aware)
    4. Calculates RMSD for each ligand group (with symmetric atom handling)

    Terminal ignoring is applied per-chain according to n_trim_chains /
    c_trim_chains (default: {'A'}) — other chains are never trimmed.

    Args:
        structure1: AF3 predicted structure (will be modified in-place during alignment)
        structure2: Reference structure
        catres_list: List of catalytic residue definitions
        ligand_atom_mapping_list: List of ligand group definitions
        verbose: Whether to print detailed progress
        n_ignore: Global N-terminal ignore count
        c_ignore: Global C-terminal ignore count
        n_trim_chains: Chains where N-terminal ignoring applies (default: {'A'})
        c_trim_chains: Chains where C-terminal ignoring applies (default: {'A'})

    Returns:
        Tuple of:
        - ca_rmsd: CA RMSD value
        - catres_rmsds: List of RMSD values for each catalytic residue
        - ligand_rmsds: List of RMSD values for each ligand group
        - ncaa_meta: List of metadata dicts for ncAA handling
    """
    vlog(verbose, "Performing Kabsch alignment on CA atoms...")
    if n_ignore > 0 or c_ignore > 0:
        vlog(verbose, f"  Terminal ignoring: N={n_ignore}, C={c_ignore} "
                      f"(N chains={sorted(n_trim_chains or DEFAULT_TRIM_CHAINS)}, "
                      f"C chains={sorted(c_trim_chains or DEFAULT_TRIM_CHAINS)})")

    # Extract CA coordinates for alignment (with per-chain terminal ignoring for structure1)
    ca_coords1 = extract_ca_coords(structure1, n_ignore, c_ignore, n_trim_chains, c_trim_chains)
    ca_coords2 = extract_ca_coords(structure2, 0, 0)  # Reference uses full structure

    # Perform Kabsch alignment of CA atoms
    rotation_matrix, centroid1, centroid2 = kabsch_alignment(ca_coords1, ca_coords2)

    # Apply rotation to all atoms in structure1
    for model in structure1:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    new_coord = apply_rotation(
                        np.array([atom.get_coord()]),
                        rotation_matrix, centroid1, centroid2
                    )[0]
                    atom.set_coord(new_coord)

    # Calculate CA RMSD after alignment
    ca_coords1 = extract_ca_coords(structure1, n_ignore, c_ignore, n_trim_chains, c_trim_chains)
    ca_rmsd_value = calculate_rmsd(ca_coords1, ca_coords2)
    vlog(verbose, f"CA RMSD after alignment: {ca_rmsd_value:.4f} A")

    # Calculate RMSD for catalytic residues (ncAA-aware)
    vlog(verbose, f"Calculating RMSD for {len(catres_list)} catalytic residues...")
    catres_rmsds = []
    ncaa_meta = []

    for catres in catres_list:
        eff_idx = catres["res_num"]
        ch_id = catres["chain"]
        resname = catres["name3"]

        # AF3 structure uses effective indexing (per-chain trim); reference uses full structure
        catres_coords1, catres_atom_name1 = extract_coords_res(
            structure1, ch_id, eff_idx, n_ignore, c_ignore, n_trim_chains, c_trim_chains
        )
        catres_coords2, catres_atom_name2 = extract_coords_res(
            structure2, ch_id, eff_idx, 0, 0
        )

        # Use symmetric-aware RMSD calculation for residues with equivalent atoms
        rmsd_val, af3_only_atoms, ref_only_atoms, common_used, swapped = rmsd_on_common_atoms_with_symmetry(
            catres_coords1, catres_atom_name1, catres_coords2, catres_atom_name2, resname
        )

        if not common_used:
            rmsd_val = float("nan")

        if swapped:
            vlog(verbose, f"  {resname}{eff_idx}: Using swapped symmetric atoms for RMSD")

        catres_rmsds.append(rmsd_val)

        # Capture ncAA / overlap metadata per residue
        af3_resname = get_resname_by_chain_resnum(
            structure1, ch_id, eff_idx, n_ignore, c_ignore, n_trim_chains, c_trim_chains
        )
        ref_resname = get_resname_by_chain_resnum(structure2, ch_id, eff_idx, 0, 0)

        # Get raw residue sequence number for pLDDT lookups later
        af3_raw_resseq = get_raw_resseq_for_effective_index(
            structure1, ch_id, eff_idx, n_ignore, c_ignore, n_trim_chains, c_trim_chains
        )

        ncaa_meta.append({
            "chain": ch_id,
            "resnum": eff_idx,
            "ref_resname": ref_resname,
            "af3_resname": af3_resname,
            "af3_only_atoms": af3_only_atoms,
            "ref_only_atoms": ref_only_atoms,
            "common_atoms_used": common_used,
            "af3_raw_resseq": af3_raw_resseq
        })

    # Calculate RMSD for ligand atoms
    vlog(verbose, f"Calculating RMSD for {len(ligand_atom_mapping_list)} ligand groups...")
    ligand_rmsd_values = []

    for ligand_atom_mapping in ligand_atom_mapping_list:
        label = ligand_atom_mapping["label"]
        chain = ligand_atom_mapping["chain"]
        name3 = ligand_atom_mapping["name3"]
        atoms = ligand_atom_mapping["atoms"]

        # Extract coordinates for specified atoms
        ligand_coords1, ligand_coords2 = [], []
        for atom1, atom2 in zip(atoms[0], atoms[1]):
            res1, res2 = name3[0], name3[1]
            ch1, ch2 = chain[0], chain[1]
            ligand_coords1.append(extract_coords_atom(structure1, ch1, res1, atom1))
            ligand_coords2.append(extract_coords_atom(structure2, ch2, res2, atom2))

        # Validate coordinates BEFORE concatenating. extract_coords_atom() returns an
        # empty array for any atom it cannot find. The previous guard only caught the
        # PARTIAL-missing case (a mix of found + absent atoms makes np.concatenate raise
        # a shape ValueError); when the ENTIRE residue/chain is absent every piece is a
        # uniform empty array that concatenates cleanly into a size-0 array, slips past
        # the guard, and crashes later in permute_symmetric_atoms() with an opaque
        # "IndexError: index 0 is out of bounds for axis 0 with size 0". Check each atom
        # explicitly so a wrong ligand name / chain ID is reported in plain language.
        missing_af3 = [atoms[0][i] for i, el in enumerate(ligand_coords1) if len(el) == 0]
        missing_ref = [atoms[1][i] for i, el in enumerate(ligand_coords2) if len(el) == 0]
        if missing_af3 or missing_ref:
            def _resnames_on_chain(structure, chain_id):
                names = sorted({res.get_resname()
                                for model in structure for ch in model
                                if ch.id == chain_id for res in ch})
                return names if names else f"<no chain '{chain_id}' present in this structure>"
            problems = []
            if missing_af3:
                problems.append(
                    f"AF3 prediction: chain '{chain[0]}' residue '{name3[0]}' is missing "
                    f"atom(s) {missing_af3} (requested {atoms[0]}). "
                    f"Residue names actually present on AF3 chain '{chain[0]}': "
                    f"{_resnames_on_chain(structure1, chain[0])}."
                )
            if missing_ref:
                problems.append(
                    f"REF structure: chain '{chain[1]}' residue '{name3[1]}' is missing "
                    f"atom(s) {missing_ref} (requested {atoms[1]}). "
                    f"Residue names actually present on REF chain '{chain[1]}': "
                    f"{_resnames_on_chain(structure2, chain[1])}."
                )
            detail = "\n  ".join(problems)
            msg = (
                f"[ligand group '{label}'] Cannot compute ligand RMSD: the atom mapping in "
                f"--ligand_groups_json does not match the structures.\n  {detail}\n"
                f"  REMINDER: in --ligand_groups_json each of name3/chain/atoms is ordered "
                f"[AF3, REF] -- here name3={name3}, chain={chain}. A wholly-absent residue/"
                f"chain (usually a wrong ligand name or chain ID) is the typical cause; fix "
                f"the mapping to match the actual residue names/chains listed above."
            )
            log(msg, "ERROR")
            raise ValueError(msg)

        # All requested atoms present on both sides -- safe to concatenate.
        ligand_coords1 = np.concatenate(ligand_coords1)
        ligand_coords2 = np.concatenate(ligand_coords2)

        # Calculate RMSD (with symmetric handling if specified)
        if "symmetric_atom_groups" in ligand_atom_mapping:
            min_rmsd = permute_symmetric_atoms(
                ligand_coords1, ligand_coords2,
                atoms[0], atoms[1],
                ligand_atom_mapping["symmetric_atom_groups"]
            )
            ligand_rmsd_values.append(min_rmsd)
            vlog(verbose, f"  {label}: {min_rmsd:.4f} A (symmetric)")
        else:
            rmsd = calculate_rmsd(ligand_coords1, ligand_coords2)
            ligand_rmsd_values.append(rmsd)
            vlog(verbose, f"  {label}: {rmsd:.4f} A")

    return ca_rmsd_value, catres_rmsds, ligand_rmsd_values, ncaa_meta


def calculate_rmsd_after_tmalign(
    af3_pdb_path: str,
    ref_pdb_path: str,
    catres_list: List[Dict],
    ligand_groups_json: List[Dict],
    verbose: bool = False,
    n_ignore: int = 0,
    c_ignore: int = 0,
    n_trim_chains: Optional[set] = None,
    c_trim_chains: Optional[set] = None
) -> Tuple[float, List[float], List[float]]:
    """
    Calculate RMSDs after TM-align style structural superposition.

    Uses biotite's superimpose_structural_homologs() which is inspired by
    the TM-align algorithm. This method is more robust for structures with
    insertions/deletions or domain movements compared to simple Kabsch CA alignment.

    Args:
        af3_pdb_path: Path to AF3 PDB file
        ref_pdb_path: Path to reference PDB file
        catres_list: List of catalytic residue definitions
        ligand_groups_json: List of ligand group definitions
        verbose: Whether to print detailed progress
        n_ignore: Global N-terminal ignore count
        c_ignore: Global C-terminal ignore count
        n_trim_chains: Chains where N-terminal ignoring applies (default: {'A'})
        c_trim_chains: Chains where C-terminal ignoring applies (default: {'A'})

    Returns:
        Tuple of:
        - ca_rmsd_tmalign: CA RMSD after TM-align superposition
        - catres_rmsds_tmalign: List of RMSD values for each catalytic residue
        - ligand_rmsds_tmalign: List of RMSD values for each ligand group
    """
    vlog(verbose, "Performing TM-align style superposition...")

    # Load structures with biotite
    af3_atoms = load_biotite_structure(af3_pdb_path)
    ref_atoms = load_biotite_structure(ref_pdb_path)

    if af3_atoms is None or ref_atoms is None:
        vlog(verbose, "TM-align: Failed to load structures", "WARN")
        nan_catres = [float("nan")] * len(catres_list)
        nan_ligands = [float("nan")] * len(ligand_groups_json)
        return float("nan"), nan_catres, nan_ligands

    # Filter to protein residues only for alignment (TM-align requires peptide chains)
    af3_protein = filter_biotite_protein_residues(
        af3_atoms, n_ignore, c_ignore, n_trim_chains, c_trim_chains
    )
    ref_protein = filter_biotite_protein_residues(ref_atoms, 0, 0)

    if len(af3_protein) == 0 or len(ref_protein) == 0:
        vlog(verbose, "TM-align: No protein atoms after filtering", "WARN")
        nan_catres = [float("nan")] * len(catres_list)
        nan_ligands = [float("nan")] * len(ligand_groups_json)
        return float("nan"), nan_catres, nan_ligands

    try:
        # Perform TM-align style superposition
        # ref is fixed, af3 is mobile
        aligned_af3, transform, fix_indices, mob_indices = struc.superimpose_structural_homologs(
            ref_protein, af3_protein, max_iterations=10
        )

        vlog(verbose, f"  TM-align used {len(fix_indices)} corresponding CA atoms")

        # Calculate CA RMSD on aligned corresponding atoms
        ca_rmsd_tmalign = struc.rmsd(ref_protein[fix_indices], aligned_af3[mob_indices])
        vlog(verbose, f"  CA RMSD (TM-align): {ca_rmsd_tmalign:.4f} A")

        # Apply the same transformation to the FULL AF3 structure (including ligands)
        af3_full_aligned = transform.apply(af3_atoms)

        # Chain-scoped effective residue maps: {chain_id: [res_id, ...]}
        af3_chain_res = _build_chain_residue_order(aligned_af3)
        ref_chain_res = _build_chain_residue_order(ref_protein)

        # Calculate catres RMSDs on TM-aligned structure
        vlog(verbose, f"  Calculating catres RMSD (TM-align) for {len(catres_list)} residues...")
        catres_rmsds_tmalign = []

        for catres in catres_list:
            eff_idx = catres["res_num"]
            resname = catres["name3"]
            ch_id = catres["chain"]

            af3_res_list = af3_chain_res.get(ch_id, [])
            ref_res_list = ref_chain_res.get(ch_id, [])

            if eff_idx < 1 or eff_idx > len(af3_res_list) or eff_idx > len(ref_res_list):
                catres_rmsds_tmalign.append(float("nan"))
                continue

            af3_target_res_id = af3_res_list[eff_idx - 1]
            ref_target_res_id = ref_res_list[eff_idx - 1]

            # Get heavy atoms for this residue on THIS CHAIN from aligned structures
            af3_res_mask = ((aligned_af3.chain_id == ch_id) &
                            (aligned_af3.res_id == af3_target_res_id) &
                            (aligned_af3.element != "H"))
            ref_res_mask = ((ref_protein.chain_id == ch_id) &
                            (ref_protein.res_id == ref_target_res_id) &
                            (ref_protein.element != "H"))

            af3_res_atoms = aligned_af3[af3_res_mask]
            ref_res_atoms = ref_protein[ref_res_mask]

            if len(af3_res_atoms) == 0 or len(ref_res_atoms) == 0:
                catres_rmsds_tmalign.append(float("nan"))
                continue

            # Find common atoms and calculate RMSD with symmetric handling
            af3_names = list(af3_res_atoms.atom_name)
            ref_names = list(ref_res_atoms.atom_name)
            af3_coords = af3_res_atoms.coord
            ref_coords = ref_res_atoms.coord

            rmsd_val, _, _, _, _ = rmsd_on_common_atoms_with_symmetry(
                af3_coords, af3_names, ref_coords, ref_names, resname
            )
            catres_rmsds_tmalign.append(rmsd_val)

        # Calculate ligand RMSDs on TM-aligned structure
        vlog(verbose, f"  Calculating ligand RMSD (TM-align) for {len(ligand_groups_json)} groups...")
        ligand_rmsds_tmalign = []

        for ligand in ligand_groups_json:
            label = ligand["label"]
            chain = ligand["chain"]
            name3 = ligand["name3"]
            atoms = ligand["atoms"]

            try:
                # Extract ligand coords from TM-aligned AF3 structure
                ligand_coords_af3 = []
                for atom_name in atoms[0]:
                    mask = ((af3_full_aligned.chain_id == chain[0]) &
                            (af3_full_aligned.res_name == name3[0]) &
                            (af3_full_aligned.atom_name == atom_name))
                    matched = af3_full_aligned[mask]
                    if len(matched) > 0:
                        ligand_coords_af3.append(matched.coord[0])
                    else:
                        ligand_coords_af3.append(None)

                # Extract ligand coords from reference
                ligand_coords_ref = []
                for atom_name in atoms[1]:
                    mask = ((ref_atoms.chain_id == chain[1]) &
                            (ref_atoms.res_name == name3[1]) &
                            (ref_atoms.atom_name == atom_name))
                    matched = ref_atoms[mask]
                    if len(matched) > 0:
                        ligand_coords_ref.append(matched.coord[0])
                    else:
                        ligand_coords_ref.append(None)

                # Filter out None values and calculate RMSD
                valid_pairs = [(a, r) for a, r in zip(ligand_coords_af3, ligand_coords_ref)
                               if a is not None and r is not None]

                if len(valid_pairs) > 0:
                    af3_arr = np.array([p[0] for p in valid_pairs])
                    ref_arr = np.array([p[1] for p in valid_pairs])

                    # Handle symmetric atoms if specified
                    if "symmetric_atom_groups" in ligand:
                        # For now, use simple RMSD - symmetric ligand handling would need more work
                        rmsd_val = np.sqrt(np.mean(np.sum((af3_arr - ref_arr) ** 2, axis=1)))
                    else:
                        rmsd_val = np.sqrt(np.mean(np.sum((af3_arr - ref_arr) ** 2, axis=1)))

                    ligand_rmsds_tmalign.append(rmsd_val)
                else:
                    ligand_rmsds_tmalign.append(float("nan"))

            except Exception as e:
                vlog(verbose, f"    {label}: Error - {e}", "WARN")
                ligand_rmsds_tmalign.append(float("nan"))

        return ca_rmsd_tmalign, catres_rmsds_tmalign, ligand_rmsds_tmalign

    except Exception as e:
        vlog(verbose, f"TM-align superposition failed: {e}", "WARN")
        nan_catres = [float("nan")] * len(catres_list)
        nan_ligands = [float("nan")] * len(ligand_groups_json)
        return float("nan"), nan_catres, nan_ligands


# ==============================================================================
# SECTION 11: SINGLE PDB PROCESSING
# ==============================================================================

def process_single_af3_pdb(
    af3_pdb: str,
    af3_pdb_num: int,
    ref_structure: PDB.Structure.Structure,
    ref_pdb_path: str,
    catres_list: List[Dict],
    ligand_groups_json: List[Dict],
    calculate_avg_pae: bool,
    validate_pae_min: bool,
    use_af3_summary_pae_min: bool,
    verbose: bool,
    verbose_pae_matrix: bool = False,
    n_ignore: int = 0,
    c_ignore: int = 0,
    catres_subset_blocks: Optional[List[int]] = None,
    catres_count: int = 0,
    catres_subset_count: int = 0,
    n_trim_chains: Optional[set] = None,
    c_trim_chains: Optional[set] = None
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Process a single AF3 PDB prediction and extract all metrics.

    This function handles all metric extraction for one AF3 prediction:
    1. Loads AF3 structure and confidence files
    2. Extracts global metrics (ipTM, pTM) via .get() so missing keys -> NaN
       (supports monomer / apo predictions where ipTM may not be emitted)
    3. Extracts per-chain pLDDT for EVERY chain in the AF3 output (protein and
       any ligand chain) regardless of apo/holo mode
    4. Extracts chain-pair PAE min (and optional mean) over all chain pairs
       found in the AF3 output
    5. Optionally validates PAE min against raw matrix calculation
    6. Calculates CA + catres RMSD via Kabsch and TM-align superposition
    7. Calculates TM-score and lDDT for catalytic residues
    8. Extracts per-catalytic-residue metrics; if ligand_groups_json is
       non-empty (HOLO mode), also extracts per-ligand RMSD / pLDDT

    APO vs HOLO:
    - Pass ligand_groups_json=[] (the default when --ligand_groups_json is
      omitted from the CLI) to skip ligand atom-matching work. All other
      metrics (ipTM, pTM, PAE, per-chain pLDDT, CA/catres RMSD, TM-score,
      lDDT) are still computed.
    - This means you can run "apo mode" even on a holo AF3 output when you
      only want confidence / PAE metrics and don't want to write an atom map.

    When n_ignore/c_ignore are specified:
    - Protein terminal residues are skipped for alignment and catalytic residue comparison
    - Ligands are never affected by terminal ignoring

    Args:
        af3_pdb: Path to AF3 PDB file
        af3_pdb_num: Index of this prediction (0-based)
        ref_structure: Reference Biopython structure
        ref_pdb_path: Path to reference PDB file (for TM-score/lDDT calculation)
        catres_list: Catalytic residue definitions
        ligand_groups_json: Ligand group definitions
        calculate_avg_pae: Whether to compute mean interchain PAE
        validate_pae_min: Whether to validate PAE min against raw matrix
        use_af3_summary_pae_min: If True, use AF3's pre-computed PAE min (may have NaN gaps);
                                  if False (default), compute from raw NxN matrix
        verbose: Whether to print detailed progress
        verbose_pae_matrix: Whether to print full NxN PAE matrices
        n_ignore: Number of protein residues at N-terminus to ignore (default: 0)
        c_ignore: Number of protein residues at C-terminus to ignore (default: 0)
        catres_subset_blocks: List of constraint block numbers to define catres subset (default: None = use all)
        catres_count: Total number of catalytic residues from REMARK 666 lines
        catres_subset_count: Number of catalytic residues in the subset (or all if no subset specified)

    Returns:
        Tuple of:
        - Dict with all extracted metrics, keyed by metric_name_idx_{af3_pdb_num}
        - List of warning/issue messages for summary
    """
    metrics = {}
    warnings = []  # Track issues for summary
    idx_suffix = f"idx_{af3_pdb_num}"

    vlog(verbose, f"Loading AF3 structure from {os.path.basename(af3_pdb)}")
    af3_structure = load_pdb(af3_pdb)
    af3_ch_ids = [chain.id for model in af3_structure for chain in model]
    vlog(verbose, f"Found {len(af3_ch_ids)} chains: {af3_ch_ids}")

    # Load confidence files
    vlog(verbose, "Loading confidence files...")
    summary_conf_path = af3_pdb.replace("_model.pdb", "_summary_confidences.json")
    conf_path = af3_pdb.replace("_model.pdb", "_confidences.json")

    with open(summary_conf_path, "r") as f:
        summary_confidences = json.load(f)
    with open(conf_path, "r") as f:
        confidences = json.load(f)

    # Extract global metrics (iptm may be absent for single-chain apo predictions)
    iptm = summary_confidences.get("iptm", float("nan"))
    ptm = summary_confidences.get("ptm", float("nan"))
    if iptm is None:
        iptm = float("nan")
    if ptm is None:
        ptm = float("nan")
    metrics[f"iptm_{idx_suffix}"] = iptm
    metrics[f"ptm_{idx_suffix}"] = ptm
    vlog(verbose, f"Global metrics: ipTM={iptm:.4f}, pTM={ptm:.4f}")

    # Per-chain pLDDT
    vlog(verbose, "Computing per-chain pLDDT scores...")
    for ch_id in af3_ch_ids:
        chain_plddts = [
            el[1] for el in zip(confidences["atom_chain_ids"], confidences["atom_plddts"])
            if el[0] == ch_id
        ]
        metrics[f"chain{ch_id}_plddt_{idx_suffix}"] = np.mean(chain_plddts)

    # Safely extract AF3's chain_pair_pae_min matrix. May be absent on apo/monomer
    # predictions, or shorter than len(af3_ch_ids) if only protein chains are scored.
    # Bounds/None checks below fall back to NaN so we never crash on missing entries.
    pae_min_matrix = summary_confidences.get("chain_pair_pae_min", [])

    def _safe_af3_pae_min(i: int, j: int) -> float:
        try:
            v = pae_min_matrix[i][j]
        except (IndexError, TypeError):
            return float("nan")
        return float("nan") if v is None else float(v)

    # Chain-pair PAE minimums
    if use_af3_summary_pae_min:
        # Use AF3's pre-computed values from summary file (may have NaN gaps for single-atom chains)
        vlog(verbose, "Using AF3 summary PAE min values (--use_af3_summary_pae_min)...")
        for i, ch_id in enumerate(af3_ch_ids):
            for j, ch_id2 in enumerate(af3_ch_ids):
                metrics[f"chain{ch_id}_chain{ch_id2}_pair_pae_min_{idx_suffix}"] = _safe_af3_pae_min(i, j)
    else:
        # Default: compute from raw NxN matrix for complete coverage
        # (AF3's summary_confidences.json often has NaN for non-protein chains)
        vlog(verbose, "Computing chain-pair PAE minimums from raw NxN matrix...")
        computed_pae_min = compute_interchain_pae_min_from_conf(
            conf_path, valid_chains=af3_ch_ids, verbose=False
        )

        # Store computed values (complete coverage, no NaN gaps)
        for i, ch_i in enumerate(af3_ch_ids):
            for j, ch_j in enumerate(af3_ch_ids):
                computed_val = computed_pae_min.get((ch_i, ch_j), float("nan"))
                metrics[f"chain{ch_i}_chain{ch_j}_pair_pae_min_{idx_suffix}"] = computed_val

        # Also store AF3's pre-computed values for comparison (may have NaN gaps)
        vlog(verbose, "Storing AF3 summary PAE min values (for reference)...")
        for i, ch_id in enumerate(af3_ch_ids):
            for j, ch_id2 in enumerate(af3_ch_ids):
                metrics[f"chain{ch_id}_chain{ch_id2}_pair_pae_min_af3summary_{idx_suffix}"] = _safe_af3_pae_min(i, j)

    # Symmetric PAE min averages (off-diagonal only)
    vlog(verbose, "Computing symmetric PAE min averages...")
    with np.errstate(all="ignore"):
        for i, ch_i in enumerate(af3_ch_ids):
            for j, ch_j in enumerate(af3_ch_ids):
                if j <= i:
                    continue  # Skip diagonal and reverse duplicates

                key_ij = f"chain{ch_i}_chain{ch_j}_pair_pae_min_{idx_suffix}"
                key_ji = f"chain{ch_j}_chain{ch_i}_pair_pae_min_{idx_suffix}"

                v_ij = metrics.get(key_ij, float("nan"))
                v_ji = metrics.get(key_ji, float("nan"))

                avg_val = np.nanmean([v_ij, v_ji])
                min_val = np.nanmin([v_ij, v_ji])

                # Canonical name: lower/earlier chain ID first (alphabetical)
                a, b = sorted([str(ch_i), str(ch_j)])
                avg_key = f"chain{a}_chain{b}_pair_pae_min_avg_{idx_suffix}"
                min_key = f"chain{a}_chain{b}_pair_pae_min_min_{idx_suffix}"
                metrics[avg_key] = float(avg_val) if np.isfinite(avg_val) else float("nan")
                metrics[min_key] = float(min_val) if np.isfinite(min_val) else float("nan")

    # Validate PAE min against raw matrix calculation (optional)
    if validate_pae_min:
        vlog(verbose, "Validating computed PAE min against AF3 summary...")
        all_match, discrepancies = validate_pae_min_against_summary(
            conf_path, summary_conf_path, af3_ch_ids,
            tolerance=0.1, verbose=verbose  # Increased tolerance for rounding differences
        )
        if not all_match:
            # Filter to show only significant discrepancies (not just rounding or NaN fill-ins)
            significant_discrepancies = [d for d in discrepancies if "AF3=nan" not in d]
            if significant_discrepancies:
                warn_msg = f"idx_{af3_pdb_num}: PAE min has {len(significant_discrepancies)} significant discrepancies (beyond rounding)"
                warnings.append(warn_msg)
                log(warn_msg, "WARN")
                for disc in significant_discrepancies[:3]:
                    log(f"    {disc}", "WARN")
                if len(significant_discrepancies) > 3:
                    log(f"    ... and {len(significant_discrepancies) - 3} more", "WARN")
            else:
                vlog(verbose, f"PAE min validation: all discrepancies are rounding/NaN fill-ins (OK)")
        else:
            vlog(verbose, f"PAE min validation passed for idx {af3_pdb_num}")

    # Mean interchain PAE from token-level PAE matrix (optional)
    if calculate_avg_pae:
        vlog(verbose, "Computing token-level mean PAE (interchain)...")
        try:
            if os.path.isfile(conf_path) and not conf_path.endswith("summary_confidences.json"):
                pair_means = compute_interchain_pae_means_from_conf(
                    conf_path,
                    valid_chains=af3_ch_ids,
                    verbose=verbose,
                    verbose_matrix=verbose_pae_matrix,
                    label=f"idx {af3_pdb_num}"
                )
                if pair_means:
                    for (a, b), mean_val in pair_means.items():
                        x, y = sorted([str(a), str(b)])
                        key = f"chain{x}_chain{y}_pair_pae_mean_{idx_suffix}"
                        metrics[key] = float(mean_val) if np.isfinite(mean_val) else float("nan")
        except Exception as e:
            warn_msg = f"idx_{af3_pdb_num}: Failed to compute token-level PAE - {e}"
            warnings.append(warn_msg)
            log(warn_msg, "WARN")

    # Structural alignment and RMSD calculations (with terminal ignore fallback)
    # Build fallback sequence: try original (N, C), then (0, 0), then single-side ignores
    fallback_pairs = [(n_ignore, c_ignore)]
    if n_ignore > 0 or c_ignore > 0:
        fallback_pairs.append((0, 0))
        if n_ignore > 0 and c_ignore > 0 and n_ignore != c_ignore:
            fallback_pairs.append((n_ignore, 0))
            fallback_pairs.append((0, c_ignore))

    ca_rmsd = catres_rmsds = ligand_rmsds = ncaa_meta = None
    for attempt_n, attempt_c in fallback_pairs:
        try:
            ca_rmsd, catres_rmsds, ligand_rmsds, ncaa_meta = calculate_rmsd_after_alignment(
                af3_structure, ref_structure, catres_list, ligand_groups_json, verbose,
                attempt_n, attempt_c, n_trim_chains, c_trim_chains
            )
            if (attempt_n, attempt_c) != (n_ignore, c_ignore):
                log(f"  Terminal ignore fallback succeeded: N={attempt_n}, C={attempt_c} "
                    f"(original: N={n_ignore}, C={c_ignore})")
                n_ignore, c_ignore = attempt_n, attempt_c
            break
        except ValueError as e:
            if "not aligned" in str(e):
                vlog(verbose, f"  Terminal ignore (N={attempt_n}, C={attempt_c}) "
                     f"shape mismatch: {e}")
                continue
            raise
    else:
        raise ValueError(
            f"CA count mismatch for all terminal ignore combinations: {fallback_pairs}"
        )

    # Track which terminal ignore was actually used
    metrics[f"terminal_ignore_N_{idx_suffix}"] = n_ignore
    metrics[f"terminal_ignore_C_{idx_suffix}"] = c_ignore

    # Ligand pLDDT scores (ligands are never affected by terminal ignoring)
    ligand_plddts = get_avr_plddt(af3_pdb, ligand_groups_json)

    # Store CA RMSD
    metrics[f"ca_rmsd_{idx_suffix}"] = ca_rmsd

    # Store ligand metrics
    vlog(verbose, "Storing ligand RMSD and pLDDT metrics...")
    for i, (ligand_rmsd, ligand_plddt) in enumerate(zip(ligand_rmsds, ligand_plddts)):
        label = ligand_groups_json[i]['label']
        metrics[f"{label}_rmsd_{idx_suffix}"] = ligand_rmsd
        metrics[f"{label}_plddt_{idx_suffix}"] = ligand_plddt

    # Store catalytic residue RMSD metrics
    vlog(verbose, "Storing catalytic residue RMSD metrics...")
    for i, catres in enumerate(catres_list):
        metrics[f"{catres['name3']}{i+1}_rmsd_{idx_suffix}"] = catres_rmsds[i]
    metrics[f"catres_rmsd_{idx_suffix}"] = (
        float(np.nanmean(catres_rmsds)) if catres_rmsds else float("nan")
    )

    # Calculate and store TM-align based RMSD metrics
    vlog(verbose, "Calculating TM-align based RMSDs...")
    ca_rmsd_tmalign, catres_rmsds_tmalign, ligand_rmsds_tmalign = calculate_rmsd_after_tmalign(
        af3_pdb, ref_pdb_path, catres_list, ligand_groups_json, verbose,
        n_ignore, c_ignore, n_trim_chains, c_trim_chains
    )

    metrics[f"ca_rmsd_TMalign_{idx_suffix}"] = ca_rmsd_tmalign

    for i, catres in enumerate(catres_list):
        metrics[f"{catres['name3']}{i+1}_rmsd_TMalign_{idx_suffix}"] = catres_rmsds_tmalign[i]
    metrics[f"catres_rmsd_TMalign_{idx_suffix}"] = np.nanmean(catres_rmsds_tmalign)

    for i, ligand in enumerate(ligand_groups_json):
        label = ligand['label']
        metrics[f"{label}_rmsd_TMalign_{idx_suffix}"] = ligand_rmsds_tmalign[i]

    # Store catalytic residue pLDDT metrics
    vlog(verbose, "Extracting catalytic residue pLDDT scores...")
    plddt_per_residue_dic = get_plddt_per_residue_dic(
        af3_pdb, n_ignore, c_ignore, n_trim_chains, c_trim_chains
    )
    cat_res_plddts = []
    for i, catres in enumerate(catres_list):
        # Use effective index for pLDDT lookup (matches what's in plddt_per_residue_dic)
        key = (catres["chain"], catres["res_num"])
        cat_res_plddt = plddt_per_residue_dic.get(key, float("nan"))
        metrics[f"{catres['name3']}{i+1}_plddt_{idx_suffix}"] = cat_res_plddt
        cat_res_plddts.append(cat_res_plddt)
    metrics[f"catres_plddt_{idx_suffix}"] = np.nanmean(cat_res_plddts)

    # Store ncAA (non-canonical amino acid) metadata
    for i, meta in enumerate(ncaa_meta):
        if not meta["af3_only_atoms"]:
            continue
        label_core = (meta["af3_resname"] or catres_list[i]["name3"]) + str(i + 1)
        metrics[f"{label_core}_extra_atoms_count_{idx_suffix}"] = len(meta["af3_only_atoms"])
        # Use raw residue sequence number for atom-level pLDDT lookup
        raw_resseq = meta.get("af3_raw_resseq", meta["resnum"])
        avg_plddt_ncaa = get_plddt_for_specific_atoms(
            af3_pdb, meta["chain"], raw_resseq, meta["af3_only_atoms"]
        )
        metrics[f"{label_core}_extra_atoms_plddt_{idx_suffix}"] = avg_plddt_ncaa
        metrics[f"{label_core}_common_atoms_used_{idx_suffix}"] = (
            "|".join(meta["common_atoms_used"]) if meta["common_atoms_used"] else ""
        )

    # Calculate TM-score and lDDT (using aligned structure saved to temp file)
    vlog(verbose, "Calculating TM-score and lDDT...")
    import tempfile
    tmp_af3_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp_af3:
            tmp_af3_path = tmp_af3.name
        save_pdb(af3_structure, tmp_af3_path)

        # Calculate TM-score
        tm_score_val = calculate_tm_score(
            tmp_af3_path, ref_pdb_path, n_ignore, c_ignore, verbose,
            n_trim_chains, c_trim_chains
        )

        # Calculate lDDT for all catalytic residues together (inter-residue distances)
        catres_lddt_val = calculate_catres_lddt(
            tmp_af3_path, ref_pdb_path, catres_list, n_ignore, c_ignore, verbose,
            n_trim_chains, c_trim_chains
        )

        # Calculate lDDT for catres_subset if specified
        if catres_subset_blocks is not None:
            subset_catres = [c for c in catres_list if c.get('cst_block', 0) in catres_subset_blocks]
            if subset_catres:
                catres_subset_lddt_val = calculate_catres_lddt(
                    tmp_af3_path, ref_pdb_path, subset_catres, n_ignore, c_ignore, verbose,
                    n_trim_chains, c_trim_chains
                )
            else:
                catres_subset_lddt_val = float("nan")
        else:
            catres_subset_lddt_val = catres_lddt_val  # Same as full set

    except Exception as e:
        vlog(verbose, f"TM-score/lDDT calculation failed: {e}", "WARN")
        tm_score_val = float("nan")
        catres_lddt_val = float("nan")
        catres_subset_lddt_val = float("nan")
    finally:
        if tmp_af3_path and os.path.exists(tmp_af3_path):
            os.unlink(tmp_af3_path)

    metrics[f"tm_score_{idx_suffix}"] = tm_score_val
    metrics[f"catres_lddt_{idx_suffix}"] = catres_lddt_val

    # Record catalytic residue counts
    metrics[f"catres_count_{idx_suffix}"] = catres_count
    metrics[f"catres_subset_count_{idx_suffix}"] = catres_subset_count

    # Calculate catres_subset metrics
    # If catres_subset_blocks is None, use all catres (metrics will be duplicated)
    if catres_subset_blocks is None:
        # Use all catres - just copy the metrics with subset prefix
        vlog(verbose, "Catres subset: using all catalytic residues (no subset specified)")
        metrics[f"catres_subset_rmsd_{idx_suffix}"] = metrics[f"catres_rmsd_{idx_suffix}"]
        metrics[f"catres_subset_rmsd_TMalign_{idx_suffix}"] = metrics[f"catres_rmsd_TMalign_{idx_suffix}"]
        metrics[f"catres_subset_plddt_{idx_suffix}"] = metrics[f"catres_plddt_{idx_suffix}"]
        metrics[f"catres_subset_lddt_{idx_suffix}"] = catres_subset_lddt_val
    else:
        # Filter to subset based on constraint block numbers
        vlog(verbose, f"Catres subset: filtering to constraint blocks {catres_subset_blocks}")
        subset_indices = [
            i for i, catres in enumerate(catres_list)
            if catres.get('cst_block', 0) in catres_subset_blocks
        ]

        if not subset_indices:
            vlog(verbose, "Catres subset: No residues matched the specified constraint blocks", "WARN")
            metrics[f"catres_subset_rmsd_{idx_suffix}"] = float("nan")
            metrics[f"catres_subset_rmsd_TMalign_{idx_suffix}"] = float("nan")
            metrics[f"catres_subset_plddt_{idx_suffix}"] = float("nan")
            metrics[f"catres_subset_lddt_{idx_suffix}"] = float("nan")
        else:
            vlog(verbose, f"Catres subset: {len(subset_indices)} residues in subset")
            # Extract subset metrics for RMSD and pLDDT
            subset_rmsds = [catres_rmsds[i] for i in subset_indices]
            subset_rmsds_tmalign = [catres_rmsds_tmalign[i] for i in subset_indices]
            subset_plddts = [cat_res_plddts[i] for i in subset_indices]

            metrics[f"catres_subset_rmsd_{idx_suffix}"] = np.nanmean(subset_rmsds) if subset_rmsds else float("nan")
            metrics[f"catres_subset_rmsd_TMalign_{idx_suffix}"] = np.nanmean(subset_rmsds_tmalign) if subset_rmsds_tmalign else float("nan")
            metrics[f"catres_subset_plddt_{idx_suffix}"] = np.nanmean(subset_plddts) if subset_plddts else float("nan")
            metrics[f"catres_subset_lddt_{idx_suffix}"] = catres_subset_lddt_val

    return metrics, warnings


# ==============================================================================
# SECTION 12: MAIN ENTRY POINT
# ==============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed argument namespace
    """
    parser = argparse.ArgumentParser(
        description="Process AlphaFold3 PDB predictions and compute structural metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ligand JSON format (for --ligand_groups_json):
  A JSON list of ligand-group objects, each describing how to pair AF3 atoms
  with reference atoms for RMSD computation:

    [
      {
        "label":  "LIG_core",                 # column-name prefix in output
        "chain":  ["B", "B"],                 # [af3_chain, ref_chain]
        "name3":  ["LIG", "LIG"],             # [af3_resname, ref_resname]
        "atoms":  [["C1","N1","O1"],          # af3 atom order
                   ["C1","N1","O1"]]          # ref atom order (paired 1:1)
      },
      {
        "label":  "LIG_carboxylate",
        "chain":  ["B", "B"],
        "name3":  ["LIG", "LIG"],
        "atoms":  [["OE1","OE2"], ["OE1","OE2"]],
        "symmetric_atom_groups": [             # optional: try permutations
          {
            "lig1": [["OE1","OE2"]],
            "lig2": [[["OE1","OE2"], ["OE2","OE1"]]]
          }
        ]
      }
    ]

  - `label` becomes the prefix of output columns: {label}_rmsd_idx_N,
     {label}_rmsd_TMalign_idx_N, {label}_plddt_idx_N.
  - Omit this argument entirely to run in APO mode.

Examples:
  HOLO (basic): match one ligand group, score everything:
    python process_af3_pdb.py --af3_dir ./af3_output --ref_pdb reference.pdb \\
        --outscr scores.sc \\
        --ligand_groups_json '[{"label":"LIG","chain":["B","B"],"name3":["LIG","LIG"],"atoms":[["C1","N1"],["C1","N1"]]}]'

  APO (pure protein, no ligand at all):
    python process_af3_pdb.py --af3_dir ./af3_output --ref_pdb reference.pdb \\
        --outscr scores.sc

  APO on holo output (AF3 predicted a ligand, you only want iptm/PAE/pLDDT):
    python process_af3_pdb.py --af3_dir ./af3_output --ref_pdb reference.pdb \\
        --outscr scores.sc --verbose

  With mean interchain PAE (in addition to min):
    python process_af3_pdb.py --af3_dir ./af3_output --ref_pdb reference.pdb \\
        --outscr scores.sc --ligand_groups_json '[...]' \\
        --calculate_avg_pae_in_addition_to_pair

  Skip a 10-residue N-terminal His-tag (and any C-terminal tag residues):
    python process_af3_pdb.py --af3_dir ./af3_output --ref_pdb reference.pdb \\
        --outscr scores.sc --ligand_groups_json '[...]' \\
        --N_terminus_tag_length_to_ignore 10 --C_terminus_tag_length_to_ignore 0

  Compute extra "subset" catres metrics from a subset of REMARK 666 blocks:
    python process_af3_pdb.py --af3_dir ./af3_output --ref_pdb reference.pdb \\
        --outscr scores.sc --ligand_groups_json '[...]' \\
        --catres_subset 1,3

  True multimer: tag-trim chain A AND chain B (e.g. heterodimer with matching
  N-terminal tags on both chains), leave ligand chain C alone:
    python process_af3_pdb.py --af3_dir ./af3_output --ref_pdb reference.pdb \\
        --outscr scores.sc --ligand_groups_json '[...]' \\
        --N_terminus_tag_length_to_ignore 10 \\
        --ignore_n_term_chain A,B

  Skip if scores already exist (useful in big parallel sweeps):
    python process_af3_pdb.py --af3_dir ./af3_output --ref_pdb reference.pdb \\
        --outscr scores.sc --rapid_mode_skip_sc_if_found

Output:
  A single-row CSV at --outscr. Column names are suffixed with _idx_N where
  N is the (sorted) position of the AF3 PDB in --af3_dir. See the module
  docstring at the top of this file for the full column list.
        """
    )

    parser.add_argument(
        "--ref_pdb",
        required=True,
        help="Path to reference PDB file with REMARK 666 catalytic residue annotations"
    )
    parser.add_argument(
        "--af3_dir",
        required=True,
        help="Directory containing AF3 predictions (*_model.pdb and *_confidences.json files)"
    )
    parser.add_argument(
        "--outscr",
        type=str,
        required=True,
        help="Output score file path (.sc or .csv)"
    )
    parser.add_argument(
        "--ligand_groups_json",
        type=str,
        default=None,
        help="JSON string defining ligand atom groups for per-ligand RMSD/pLDDT. "
             "See module docstring or --help epilog for the expected schema. "
             "OMIT this flag for APO mode: ligand RMSD/pLDDT columns are dropped, "
             "but ipTM, pTM, per-chain pLDDT, and chain-pair PAE are still computed "
             "(useful when you just want confidence metrics without writing an atom map)."
    )
    parser.add_argument(
        "--calculate_avg_pae_in_addition_to_pair",
        action="store_true",
        help="Also compute MEAN interchain PAE from the raw NxN token-level PAE matrix "
             "(in addition to the default PAE MIN). Adds chain{X}_chain{Y}_pair_pae_mean_idx_N columns."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress and debug information"
    )
    parser.add_argument(
        "--rapid_mode_skip_sc_if_found",
        action="store_true",
        help="Exit early without processing if any .sc file already exists in --af3_dir. "
             "Useful for resuming large parallel sweeps without re-scoring completed jobs."
    )
    parser.add_argument(
        "--validate_pae_min",
        action="store_true",
        help="Sanity-check: recompute PAE min from the raw NxN matrix and compare to AF3's "
             "pre-computed chain_pair_pae_min in the summary file. Logs any discrepancies."
    )
    parser.add_argument(
        "--use_af3_summary_pae_min",
        action="store_true",
        help="Use AF3's pre-computed chain_pair_pae_min from *_summary_confidences.json "
             "INSTEAD OF computing from the raw NxN matrix. Default (off) is more complete "
             "because AF3's summary values can be NaN for single-atom / single-token chains."
    )
    parser.add_argument(
        "--verbose_pae_matrix",
        action="store_true",
        help="Also print the full NxN PAE matrix for each prediction. WARNING: very large "
             "output (N can be thousands); implies --verbose; for debugging only."
    )
    parser.add_argument(
        "--N_terminus_tag_length_to_ignore",
        type=int,
        default=0,
        help="Number of protein residues at N-terminus to ignore in AF3 predictions (default: 0). "
             "Only applies to chains listed in --ignore_n_term_chain (default: A). "
             "Ligand residues (HETATM) are never ignored."
    )
    parser.add_argument(
        "--C_terminus_tag_length_to_ignore",
        type=int,
        default=0,
        help="Number of protein residues at C-terminus to ignore in AF3 predictions (default: 0). "
             "Only applies to chains listed in --ignore_c_term_chain (default: A). "
             "Ligand residues (HETATM) are never ignored."
    )
    parser.add_argument(
        "--ignore_n_term_chain",
        type=str,
        default="A",
        help="Comma-separated chain IDs where --N_terminus_tag_length_to_ignore should be "
             "applied (default: 'A'). The pipeline assumes chain A is the designed protein "
             "and other chains are peptide substrates or ligands that must not be trimmed. "
             "Override when multimer proteins are present, e.g. 'A,B,D'."
    )
    parser.add_argument(
        "--ignore_c_term_chain",
        type=str,
        default="A",
        help="Comma-separated chain IDs where --C_terminus_tag_length_to_ignore should be "
             "applied (default: 'A'). See --ignore_n_term_chain."
    )
    parser.add_argument(
        "--catres_subset",
        type=str,
        default=None,
        help="Comma-separated REMARK 666 constraint-block numbers defining a catres subset "
             "(e.g. '1,3,5'). Emits catres_subset_rmsd / catres_subset_rmsd_TMalign / "
             "catres_subset_plddt / catres_subset_lddt computed ONLY over matching residues. "
             "Useful when you care about a functional subset (e.g. only the active-site "
             "triad) separately from auxiliary constrained residues. Invalid block numbers "
             "are logged and dropped. If omitted, subset metrics mirror the full catres metrics."
    )
    parser.add_argument(
        "--no_strip_conect_master",
        action="store_true",
        help="Disable the default removal of CONECT and MASTER lines from AF3 PDB files. "
             "These records sometimes appear in AF3 CIF-to-PDB output and aren't needed "
             "for scoring; stripping prevents downstream tools from choking on them."
    )

    return parser.parse_args()


def main() -> None:
    """
    Main entry point for AF3 PDB processing.

    Workflow:
    1. Parse arguments and check for rapid skip
    2. Load reference structure and extract catalytic residues
    3. Process each AF3 PDB prediction
    4. Aggregate metrics and write output CSV
    5. Print comprehensive summary
    """
    args = parse_arguments()

    # Extract terminal ignore parameters
    n_ignore = args.N_terminus_tag_length_to_ignore
    c_ignore = args.C_terminus_tag_length_to_ignore

    # Parse chain-trim sets from CLI (default: {'A'})
    def _parse_chain_list(s: str) -> set:
        return {c.strip() for c in (s or "").split(",") if c.strip()}

    n_trim_chains = _parse_chain_list(args.ignore_n_term_chain)
    c_trim_chains = _parse_chain_list(args.ignore_c_term_chain)

    # Parse catres_subset parameter
    catres_subset_blocks = None
    if args.catres_subset is not None:
        try:
            catres_subset_blocks = [int(x.strip()) for x in args.catres_subset.split(',')]
        except ValueError:
            log(f"WARNING: Invalid --catres_subset format '{args.catres_subset}'. Expected comma-separated integers.", "WARN")
            catres_subset_blocks = None

    # Log startup information
    log("=" * 70)
    log("AF3 PDB PROCESSING - STARTED")
    log("=" * 70)
    log(f"  Reference PDB: {args.ref_pdb}")
    log(f"  AF3 directory: {args.af3_dir}")
    log(f"  Output score file: {args.outscr}")
    log(f"  Options: verbose={args.verbose}, validate_pae_min={args.validate_pae_min}, "
        f"calculate_avg_pae={args.calculate_avg_pae_in_addition_to_pair}, "
        f"strip_conect_master={not args.no_strip_conect_master}")
    if n_ignore > 0 or c_ignore > 0:
        log(f"  Terminal ignoring: N-term={n_ignore} on chains {sorted(n_trim_chains)}, "
            f"C-term={c_ignore} on chains {sorted(c_trim_chains)}")
    if catres_subset_blocks is not None:
        log(f"  Catres subset: constraint blocks {catres_subset_blocks}")
    else:
        log(f"  Catres subset: using all catalytic residues")

    # Check for rapid skip mode
    rapid_skip_if_sc_present(args.af3_dir, args.rapid_mode_skip_sc_if_found)

    # Parse ligand groups JSON (apo mode if not provided)
    if args.ligand_groups_json is None:
        args.ligand_groups_json = []
        log(f"  Ligand groups defined: 0 (APO mode - no ligand scoring)")
    else:
        args.ligand_groups_json = json.loads(args.ligand_groups_json)
        log(f"  Ligand groups defined: {len(args.ligand_groups_json)}")

    # Initialize score dictionary and tracking
    score_dic = {"af3_models_dir": args.af3_dir, "ref_path": args.ref_pdb}
    all_warnings = []
    per_pdb_summary = []

    # Load reference structure
    vlog(args.verbose, f"Loading reference structure from {args.ref_pdb}")
    ref_structure = load_pdb(args.ref_pdb)

    # Multimer detection: the pipeline assumes chain A is the only protein chain
    # and other chains are peptide substrates or ligands. If the reference has
    # protein residues on any chain outside the configured trim sets, warn the
    # user so they can decide whether to extend --ignore_{n,c}_term_chain.
    protein_chain_counts: Dict[str, int] = {}
    for model in ref_structure:
        for chain in model:
            protein_residues = [r for r in chain.get_residues() if is_protein_residue(r)]
            if protein_residues:
                protein_chain_counts[chain.id] = len(protein_residues)
    configured_trim = n_trim_chains | c_trim_chains
    # Treat chains with fewer than 15 protein residues as peptide substrates,
    # not multimer protein chains — no warning for those.
    PEPTIDE_RES_THRESHOLD = 15
    unconfigured_protein_chains = {
        ch: n for ch, n in protein_chain_counts.items()
        if ch not in configured_trim and n >= PEPTIDE_RES_THRESHOLD
    }
    if len(protein_chain_counts) > 1 and unconfigured_protein_chains:
        print("\n")
        log("!" * 80, "WARN")
        log("!!! MULTIMER DETECTED: protein chains present that are NOT in trim sets !!!", "WARN")
        log("!" * 80, "WARN")
        for ch, n in sorted(protein_chain_counts.items()):
            in_n = ch in n_trim_chains
            in_c = ch in c_trim_chains
            marker = " (TRIMMED)" if (in_n or in_c) else " (not trimmed)"
            log(f"  chain {ch}: {n} protein residues{marker}", "WARN")
        log("  The script assumes chain A is the designed protein; other chains are", "WARN")
        log("  treated as peptide substrates or ligands. If the extra protein chain(s)", "WARN")
        log("  above ARE additional protein chains you want to tag-trim, pass e.g.:", "WARN")
        suggested = ",".join(sorted(configured_trim | set(unconfigured_protein_chains.keys())))
        log(f"    --ignore_n_term_chain {suggested}  --ignore_c_term_chain {suggested}", "WARN")
        log("!" * 80, "WARN")
        print("\n")

    # Extract catalytic residues from reference
    catres_list = get_catalytic_residues(args.ref_pdb, args.verbose)
    log(f"  Catalytic residues: {len(catres_list)}")

    # Validate catres_subset_blocks against available constraint blocks
    available_cst_blocks = set(catres.get('cst_block', 0) for catres in catres_list)
    catres_subset_count = len(catres_list)  # Default: all catres

    if catres_subset_blocks is not None:
        # Check for invalid blocks
        invalid_blocks = [b for b in catres_subset_blocks if b not in available_cst_blocks]
        valid_blocks = [b for b in catres_subset_blocks if b in available_cst_blocks]

        if invalid_blocks:
            print("\n")
            log("!" * 80, "WARN")
            log("!" * 80, "WARN")
            log("!!! WARNING: INVALID CONSTRAINT BLOCK NUMBERS IN --catres_subset !!!", "WARN")
            log("!" * 80, "WARN")
            log(f"  Requested blocks: {catres_subset_blocks}", "WARN")
            log(f"  Available blocks in REMARK 666: {sorted(available_cst_blocks)}", "WARN")
            log(f"  INVALID/MISSING blocks: {invalid_blocks}", "WARN")
            log(f"  Continuing with valid blocks only: {valid_blocks}", "WARN")
            log("!" * 80, "WARN")
            log("!" * 80, "WARN")
            print("\n")

        # Update to only use valid blocks
        catres_subset_blocks = valid_blocks if valid_blocks else None
        if catres_subset_blocks:
            catres_subset_count = sum(1 for catres in catres_list if catres.get('cst_block', 0) in catres_subset_blocks)
            log(f"  Catres subset: {catres_subset_count} residues from blocks {catres_subset_blocks}")
        else:
            log("  Catres subset: No valid blocks remaining, using all catalytic residues", "WARN")
            catres_subset_count = len(catres_list)

    # Find all AF3 PDB files
    af3_pdbs = sorted(glob.glob(os.path.join(args.af3_dir, "*.pdb")))
    total_pdbs = len(af3_pdbs)
    log(f"  AF3 PDB files found: {total_pdbs}")

    # Strip CONECT/MASTER lines from AF3 PDBs (default: on)
    if not args.no_strip_conect_master:
        n_cleaned = strip_conect_master_lines(af3_pdbs, args.verbose)
        if n_cleaned > 0:
            log(f"  Stripped CONECT/MASTER lines from {n_cleaned}/{total_pdbs} PDB files")

    log("-" * 70)

    # Process each AF3 prediction
    for af3_pdb_num, af3_pdb in enumerate(af3_pdbs):
        log(f"Processing [{af3_pdb_num + 1}/{total_pdbs}]: {os.path.basename(af3_pdb)}")

        metrics, warnings = process_single_af3_pdb(
            af3_pdb=af3_pdb,
            af3_pdb_num=af3_pdb_num,
            ref_structure=ref_structure,
            ref_pdb_path=args.ref_pdb,
            catres_list=catres_list,
            ligand_groups_json=args.ligand_groups_json,
            calculate_avg_pae=args.calculate_avg_pae_in_addition_to_pair,
            validate_pae_min=args.validate_pae_min,
            use_af3_summary_pae_min=args.use_af3_summary_pae_min,
            verbose=args.verbose,
            verbose_pae_matrix=args.verbose_pae_matrix,
            n_ignore=n_ignore,
            c_ignore=c_ignore,
            catres_subset_blocks=catres_subset_blocks,
            catres_count=len(catres_list),
            catres_subset_count=catres_subset_count,
            n_trim_chains=n_trim_chains,
            c_trim_chains=c_trim_chains
        )

        # Merge metrics into main score dictionary
        score_dic.update(metrics)
        all_warnings.extend(warnings)

        # Collect per-PDB summary
        idx = f"idx_{af3_pdb_num}"
        pdb_summary = {
            "pdb": os.path.basename(af3_pdb),
            "iptm": metrics.get(f"iptm_{idx}", float("nan")),
            "ptm": metrics.get(f"ptm_{idx}", float("nan")),
            "ca_rmsd": metrics.get(f"ca_rmsd_{idx}", float("nan")),
            "catres_rmsd": metrics.get(f"catres_rmsd_{idx}", float("nan")),
            "tm_score": metrics.get(f"tm_score_{idx}", float("nan")),
            "catres_lddt": metrics.get(f"catres_lddt_{idx}", float("nan")),
            "catres_subset_rmsd": metrics.get(f"catres_subset_rmsd_{idx}", float("nan")),
            "catres_subset_lddt": metrics.get(f"catres_subset_lddt_{idx}", float("nan")),
            "warnings": len(warnings)
        }
        per_pdb_summary.append(pdb_summary)

    # Write output
    log("-" * 70)
    log(f"Writing results to {args.outscr}")
    score_df = pd.DataFrame([score_dic])
    score_df.to_csv(args.outscr, index=False)

    # =========================================================================
    # SUMMARY SECTION
    # =========================================================================
    print("\n")
    log("=" * 70)
    log("PROCESSING SUMMARY")
    log("=" * 70)

    # Per-PDB metrics summary
    print(f"\n{'PDB File':<45} {'ipTM':>7} {'pTM':>7} {'CA RMSD':>9} {'CatRMSD':>8} {'TM-sc':>7} {'lDDT':>7} {'Warn':>5}")
    print("-" * 100)
    for s in per_pdb_summary:
        print(f"{s['pdb']:<45} {s['iptm']:>7.4f} {s['ptm']:>7.4f} {s['ca_rmsd']:>9.4f} {s['catres_rmsd']:>8.4f} {s['tm_score']:>7.4f} {s['catres_lddt']:>7.4f} {s['warnings']:>5}")

    # Overall statistics
    print("\n" + "-" * 100)
    if per_pdb_summary:
        avg_iptm = np.nanmean([s['iptm'] for s in per_pdb_summary])
        avg_ptm = np.nanmean([s['ptm'] for s in per_pdb_summary])
        avg_ca_rmsd = np.nanmean([s['ca_rmsd'] for s in per_pdb_summary])
        avg_catres_rmsd = np.nanmean([s['catres_rmsd'] for s in per_pdb_summary])
        avg_tm_score = np.nanmean([s['tm_score'] for s in per_pdb_summary])
        avg_catres_lddt = np.nanmean([s['catres_lddt'] for s in per_pdb_summary])
        avg_catres_subset_rmsd = np.nanmean([s['catres_subset_rmsd'] for s in per_pdb_summary])
        avg_catres_subset_lddt = np.nanmean([s['catres_subset_lddt'] for s in per_pdb_summary])
        def _safe_best(values, func):
            arr = np.asarray(values, dtype=float)
            if np.all(np.isnan(arr)):
                return None
            return int(func(arr))

        best_iptm_idx = _safe_best([s['iptm'] for s in per_pdb_summary], np.nanargmax)
        best_ca_rmsd_idx = _safe_best([s['ca_rmsd'] for s in per_pdb_summary], np.nanargmin)
        best_tm_score_idx = _safe_best([s['tm_score'] for s in per_pdb_summary], np.nanargmax)

        print(f"AVERAGES:  ipTM={avg_iptm:.4f}  pTM={avg_ptm:.4f}  CA_RMSD={avg_ca_rmsd:.4f}A  CatRes_RMSD={avg_catres_rmsd:.4f}A  TM-score={avg_tm_score:.4f}  lDDT={avg_catres_lddt:.4f}")
        print(f"SUBSET AVERAGES:  CatRes_Subset_RMSD={avg_catres_subset_rmsd:.4f}A  CatRes_Subset_lDDT={avg_catres_subset_lddt:.4f}")
        print(f"CATRES COUNTS:    Total={len(catres_list)}  Subset={catres_subset_count}")
        if best_iptm_idx is not None:
            print(f"BEST ipTM: idx_{best_iptm_idx} ({per_pdb_summary[best_iptm_idx]['pdb']}) = {per_pdb_summary[best_iptm_idx]['iptm']:.4f}")
        else:
            print(f"BEST ipTM: N/A (all NaN - likely apo/monomer prediction)")
        if best_ca_rmsd_idx is not None:
            print(f"BEST CA_RMSD: idx_{best_ca_rmsd_idx} ({per_pdb_summary[best_ca_rmsd_idx]['pdb']}) = {per_pdb_summary[best_ca_rmsd_idx]['ca_rmsd']:.4f}A")
        if best_tm_score_idx is not None:
            print(f"BEST TM-score: idx_{best_tm_score_idx} ({per_pdb_summary[best_tm_score_idx]['pdb']}) = {per_pdb_summary[best_tm_score_idx]['tm_score']:.4f}")

    # Warnings summary
    print("\n" + "-" * 70)
    if all_warnings:
        log(f"WARNINGS: {len(all_warnings)} issue(s) detected", "WARN")
        for w in all_warnings:
            print(f"  - {w}")
    else:
        log("WARNINGS: None - all validations passed", "INFO")

    # Final status
    print("\n" + "=" * 70)
    status = "COMPLETED WITH WARNINGS" if all_warnings else "COMPLETED SUCCESSFULLY"
    log(f"STATUS: {status}")
    log(f"Output: {args.outscr}")
    log(f"Total metrics columns: {len(score_dic)}")
    log("=" * 70)


if __name__ == "__main__":
    main()
