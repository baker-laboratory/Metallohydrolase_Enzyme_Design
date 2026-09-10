#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
align_prediction_to_ref_pdb_and_copy_lig.py

Created: 2026-01-15 by Seth Woodbury (woodbuse@uw.edu)
Updated: 2026-01-15 - Fixed HETATM stripping, improved sidechain optimization

Description:
    Advanced PDB alignment script for realigning AlphaFold3 predictions with designed
    enzyme structures. Uses multiple alignment strategies with iterative outlier removal
    and sidechain dihedral optimization to recapitulate catalytic geometries.

Features:
    - Multiple alignment strategies with iterative outlier removal
    - Per-residue sidechain dihedral optimization using reference-guided search
    - NCAA (non-canonical amino acid) handling for AF3 predictions
    - Proper HETATM handling (ALL stripped from prediction, reference added back)
    - HIS tautomer detection (HIS vs HIS_D)
    - Comprehensive metrics tracking and CSV output
    - All REMARK lines copied from reference
    - Configurable thresholds

Usage:
    python align_prediction_to_ref_pdb_and_copy_lig.py \\
        --ref_pdb reference.pdb \\
        --pdb_for_alignment prediction.pdb \\
        --output_dir ./aligned_outputs \\
        --catres_subset 1,2,3,4,5,6,7,8,9,10 \\
        --ptm_from_remark666 "A/LYS/6:KCX" \\
        --verbose
"""

import os
import sys
import argparse
import tempfile
import shutil
import time
import copy
import warnings
import re
from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd

# BioPython imports
try:
    from Bio.PDB import PDBParser, PDBIO, Superimposer, Structure, Model, Chain
    from Bio.PDB.Atom import Atom
    from Bio.PDB.Residue import Residue
    from Bio.PDB.vectors import Vector, calc_dihedral, rotaxis
except ImportError:
    print("ERROR: BioPython is not installed. Install with: pip install biopython")
    sys.exit(1)

# PyRosetta imports (for hydrogen addition)
try:
    import pyrosetta
    import pyrosetta.rosetta
    import pyrosetta.distributed.io
    HAS_PYROSETTA = True
except ImportError:
    print("WARNING: PyRosetta not available. Hydrogen addition will be skipped.")
    HAS_PYROSETTA = False

# Biotite imports (for TM-align style superposition)
try:
    import biotite.structure as bs
    import biotite.structure.io.pdb as bpdb
    from biotite.structure import superimpose_structural_homologs
    HAS_BIOTITE = True
except ImportError:
    print("WARNING: Biotite not available. TM-align strategies will be skipped.")
    HAS_BIOTITE = False


# ============================================================================
# RESOURCE DETECTION
# ============================================================================

def detect_cpu_count() -> int:
    """Best-effort count of CPUs actually usable by this process.

    Honors a SLURM allocation first (so a 1-CPU SLURM task reports 1 and runs
    serially), then the process CPU affinity mask (respects cgroups / cpusets),
    then os.cpu_count(). Always returns >= 1.
    """
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        val = os.environ.get(var)
        if val and val.isdigit() and int(val) > 0:
            return int(val)
    try:
        n = len(os.sched_getaffinity(0))
        if n > 0:
            return n
    except (AttributeError, OSError):
        pass
    return max(1, os.cpu_count() or 1)


# ============================================================================
# NCAA (NON-CANONICAL AMINO ACID) DEFINITIONS
# ============================================================================

# Hardcoded dictionary of common NCAAs and their atoms to cut
NCAA_ATOMS_TO_CUT = {
    'KCX': ['CX', 'OQ1', 'OQ2'],           # Carboxy-lysine
    'MLY': ['CM', 'CE', 'NZ'],              # N-dimethyl-lysine
    'ALY': ['N1'],                          # N-acetyl-lysine
    'MLZ': ['CM'],                          # N-methyl-lysine
    'M3L': ['CM', 'CN'],                    # N-trimethyl-lysine
    'HYP': ['OD1'],                         # Hydroxyproline
    'SEP': ['P', 'O1P', 'O2P', 'O3P'],     # Phosphoserine
    'TPO': ['P', 'O1P', 'O2P', 'O3P'],     # Phosphothreonine
    'PTR': ['P', 'O1P', 'O2P', 'O3P'],     # Phosphotyrosine
    'CSO': ['OD'],                          # S-hydroxycysteine
    'CSD': ['OD1', 'OD2'],                  # Cysteinesulfinic acid
    'OCS': ['O1', 'O2'],                    # Cysteinesulfonic acid
    'MSE': ['SE'],                          # Selenomethionine
}


# ============================================================================
# SYMMETRIC ATOM PAIRS - Chemically equivalent atoms
# ============================================================================

# Symmetric atom pairs for RMSD and lDDT calculations
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


# ============================================================================
# CHI ANGLE DEFINITIONS - Complete atom connectivity for proper rotation
# ============================================================================

# For each residue type, define:
# - chi_atoms: the 4 atoms defining each chi angle dihedral
# - atoms_to_rotate: atoms that move when rotating around that chi angle
CHI_DEFINITIONS = {
    'ARG': {
        'chi_atoms': [
            ('N', 'CA', 'CB', 'CG'),
            ('CA', 'CB', 'CG', 'CD'),
            ('CB', 'CG', 'CD', 'NE'),
            ('CG', 'CD', 'NE', 'CZ')
        ],
        'downstream_atoms': [
            ['CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2', 'HB2', 'HB3', 'HG2', 'HG3', 'HD2', 'HD3', 'HE', 'HH11', 'HH12', 'HH21', 'HH22'],
            ['CD', 'NE', 'CZ', 'NH1', 'NH2', 'HG2', 'HG3', 'HD2', 'HD3', 'HE', 'HH11', 'HH12', 'HH21', 'HH22'],
            ['NE', 'CZ', 'NH1', 'NH2', 'HD2', 'HD3', 'HE', 'HH11', 'HH12', 'HH21', 'HH22'],
            ['CZ', 'NH1', 'NH2', 'HE', 'HH11', 'HH12', 'HH21', 'HH22']
        ]
    },
    'ASN': {
        'chi_atoms': [
            ('N', 'CA', 'CB', 'CG'),
            ('CA', 'CB', 'CG', 'OD1')
        ],
        'downstream_atoms': [
            ['CG', 'OD1', 'ND2', 'HB2', 'HB3', 'HD21', 'HD22'],
            ['OD1', 'ND2', 'HD21', 'HD22']
        ]
    },
    'ASP': {
        'chi_atoms': [
            ('N', 'CA', 'CB', 'CG'),
            ('CA', 'CB', 'CG', 'OD1')
        ],
        'downstream_atoms': [
            ['CG', 'OD1', 'OD2', 'HB2', 'HB3'],
            ['OD1', 'OD2']
        ]
    },
    'CYS': {
        'chi_atoms': [
            ('N', 'CA', 'CB', 'SG')
        ],
        'downstream_atoms': [
            ['SG', 'HB2', 'HB3', 'HG']
        ]
    },
    'GLN': {
        'chi_atoms': [
            ('N', 'CA', 'CB', 'CG'),
            ('CA', 'CB', 'CG', 'CD'),
            ('CB', 'CG', 'CD', 'OE1')
        ],
        'downstream_atoms': [
            ['CG', 'CD', 'OE1', 'NE2', 'HB2', 'HB3', 'HG2', 'HG3', 'HE21', 'HE22'],
            ['CD', 'OE1', 'NE2', 'HG2', 'HG3', 'HE21', 'HE22'],
            ['OE1', 'NE2', 'HE21', 'HE22']
        ]
    },
    'GLU': {
        'chi_atoms': [
            ('N', 'CA', 'CB', 'CG'),
            ('CA', 'CB', 'CG', 'CD'),
            ('CB', 'CG', 'CD', 'OE1')
        ],
        'downstream_atoms': [
            ['CG', 'CD', 'OE1', 'OE2', 'HB2', 'HB3', 'HG2', 'HG3'],
            ['CD', 'OE1', 'OE2', 'HG2', 'HG3'],
            ['OE1', 'OE2']
        ]
    },
    'HIS': {
        'chi_atoms': [
            ('N', 'CA', 'CB', 'CG'),
            ('CA', 'CB', 'CG', 'ND1')
        ],
        'downstream_atoms': [
            ['CG', 'ND1', 'CD2', 'CE1', 'NE2', 'HB2', 'HB3', 'HD1', 'HD2', 'HE1', 'HE2'],
            ['ND1', 'CD2', 'CE1', 'NE2', 'HD1', 'HD2', 'HE1', 'HE2']
        ]
    },
    'ILE': {
        'chi_atoms': [
            ('N', 'CA', 'CB', 'CG1'),
            ('CA', 'CB', 'CG1', 'CD1')
        ],
        'downstream_atoms': [
            ['CG1', 'CG2', 'CD1', 'HB', 'HG12', 'HG13', 'HG21', 'HG22', 'HG23', 'HD11', 'HD12', 'HD13'],
            ['CD1', 'HG12', 'HG13', 'HD11', 'HD12', 'HD13']
        ]
    },
    'LEU': {
        'chi_atoms': [
            ('N', 'CA', 'CB', 'CG'),
            ('CA', 'CB', 'CG', 'CD1')
        ],
        'downstream_atoms': [
            ['CG', 'CD1', 'CD2', 'HB2', 'HB3', 'HG', 'HD11', 'HD12', 'HD13', 'HD21', 'HD22', 'HD23'],
            ['CD1', 'CD2', 'HG', 'HD11', 'HD12', 'HD13', 'HD21', 'HD22', 'HD23']
        ]
    },
    'LYS': {
        'chi_atoms': [
            ('N', 'CA', 'CB', 'CG'),
            ('CA', 'CB', 'CG', 'CD'),
            ('CB', 'CG', 'CD', 'CE'),
            ('CG', 'CD', 'CE', 'NZ')
        ],
        'downstream_atoms': [
            ['CG', 'CD', 'CE', 'NZ', 'HB2', 'HB3', 'HG2', 'HG3', 'HD2', 'HD3', 'HE2', 'HE3', 'HZ1', 'HZ2', 'HZ3'],
            ['CD', 'CE', 'NZ', 'HG2', 'HG3', 'HD2', 'HD3', 'HE2', 'HE3', 'HZ1', 'HZ2', 'HZ3'],
            ['CE', 'NZ', 'HD2', 'HD3', 'HE2', 'HE3', 'HZ1', 'HZ2', 'HZ3'],
            ['NZ', 'HE2', 'HE3', 'HZ1', 'HZ2', 'HZ3']
        ]
    },
    'MET': {
        'chi_atoms': [
            ('N', 'CA', 'CB', 'CG'),
            ('CA', 'CB', 'CG', 'SD'),
            ('CB', 'CG', 'SD', 'CE')
        ],
        'downstream_atoms': [
            ['CG', 'SD', 'CE', 'HB2', 'HB3', 'HG2', 'HG3', 'HE1', 'HE2', 'HE3'],
            ['SD', 'CE', 'HG2', 'HG3', 'HE1', 'HE2', 'HE3'],
            ['CE', 'HE1', 'HE2', 'HE3']
        ]
    },
    'PHE': {
        'chi_atoms': [
            ('N', 'CA', 'CB', 'CG'),
            ('CA', 'CB', 'CG', 'CD1')
        ],
        'downstream_atoms': [
            ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'HB2', 'HB3', 'HD1', 'HD2', 'HE1', 'HE2', 'HZ'],
            ['CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'HD1', 'HD2', 'HE1', 'HE2', 'HZ']
        ]
    },
    'PRO': {
        'chi_atoms': [
            ('N', 'CA', 'CB', 'CG'),
            ('CA', 'CB', 'CG', 'CD')
        ],
        'downstream_atoms': [
            ['CG', 'CD', 'HB2', 'HB3', 'HG2', 'HG3', 'HD2', 'HD3'],
            ['CD', 'HG2', 'HG3', 'HD2', 'HD3']
        ]
    },
    'SER': {
        'chi_atoms': [
            ('N', 'CA', 'CB', 'OG')
        ],
        'downstream_atoms': [
            ['OG', 'HB2', 'HB3', 'HG']
        ]
    },
    'THR': {
        'chi_atoms': [
            ('N', 'CA', 'CB', 'OG1')
        ],
        'downstream_atoms': [
            ['OG1', 'CG2', 'HB', 'HG1', 'HG21', 'HG22', 'HG23']
        ]
    },
    'TRP': {
        'chi_atoms': [
            ('N', 'CA', 'CB', 'CG'),
            ('CA', 'CB', 'CG', 'CD1')
        ],
        'downstream_atoms': [
            ['CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2', 'HB2', 'HB3', 'HD1', 'HE1', 'HE3', 'HZ2', 'HZ3', 'HH2'],
            ['CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2', 'HD1', 'HE1', 'HE3', 'HZ2', 'HZ3', 'HH2']
        ]
    },
    'TYR': {
        'chi_atoms': [
            ('N', 'CA', 'CB', 'CG'),
            ('CA', 'CB', 'CG', 'CD1')
        ],
        'downstream_atoms': [
            ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH', 'HB2', 'HB3', 'HD1', 'HD2', 'HE1', 'HE2', 'HH'],
            ['CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH', 'HD1', 'HD2', 'HE1', 'HE2', 'HH']
        ]
    },
    'VAL': {
        'chi_atoms': [
            ('N', 'CA', 'CB', 'CG1')
        ],
        'downstream_atoms': [
            ['CG1', 'CG2', 'HB', 'HG11', 'HG12', 'HG13', 'HG21', 'HG22', 'HG23']
        ]
    }
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class AlignmentMetrics:
    """Store metrics for a single alignment strategy."""
    strategy_name: str
    converged_rmsd: float
    n_iterations: int
    n_atoms_final: int
    all_backbone_rmsd: float
    ca_rmsd: float
    catres_backbone_rmsd: float
    catres_ca_rmsd: float
    catres_subset_backbone_rmsd: float
    catres_subset_ca_rmsd: float
    catres_subset_all_atom_rmsd_before_opt: float
    catres_subset_all_atom_rmsd_after_opt: float
    n_sidechain_opt_iterations: int
    sidechain_opt_improvement: float
    # New fields for enhanced metrics
    catres_subset_internal_lddt: float = 0.0  # lDDT between catres_subset atoms only (catres↔catres pairs)
    tm_score: float = float('nan')  # TM-score (NaN for non-TM-align strategies; computed only for global_TMalign)
    # Clash-penalty diagnostics (populated only when --apply_clash_penalty is on)
    n_unexpected_clashes_initial: int = -1   # -1 sentinel = not computed
    n_unexpected_clashes_final: int = -1
    soft_clash_penalty_final: float = float('nan')  # Å² units
    n_overshot_anchors_final: int = -1       # exempt pairs that got too close (d_mob < d_ref - tol_close)
    n_loosened_anchors_final: int = -1       # exempt pairs that got too far (d_mob > d_ref + tol_loose)
    soft_anchor_penalty_final: float = float('nan')  # weighted, Å²-cost-equivalent units


@dataclass
class CatalyticResidue:
    """Information about a catalytic residue from REMARK 666."""
    chain: str
    resname: str
    resnum: int
    catres_index: int


@dataclass
class PTMSpec:
    """Specification for a post-translational modification."""
    chain: str
    canonical_resname: str
    catres_index: int
    ncaa_resname: str
    atoms_to_cut: List[str]


# ============================================================================
# NCAA PARSING
# ============================================================================

def parse_ptm_spec(spec_str: str) -> PTMSpec:
    """Parse PTM specification string."""
    if ':' not in spec_str:
        raise ValueError(f"Invalid PTM spec format: {spec_str}")

    left, right = spec_str.split(':', 1)
    parts = left.split('/')
    if len(parts) != 3:
        raise ValueError(f"Invalid PTM spec: {left}")

    chain = parts[0].strip()
    canonical_resname = parts[1].strip()
    catres_index = int(parts[2].strip())

    if '-' in right:
        ncaa_resname, atoms_str = right.split('-', 1)
        ncaa_resname = ncaa_resname.strip()
        atoms_to_cut = [a.strip() for a in atoms_str.split(',')]
    else:
        ncaa_resname = right.strip()
        atoms_to_cut = NCAA_ATOMS_TO_CUT.get(ncaa_resname, []).copy()
        if not atoms_to_cut:
            print(f"WARNING: No default atoms for NCAA {ncaa_resname}")

    return PTMSpec(chain, canonical_resname, catres_index, ncaa_resname, atoms_to_cut)


def parse_ptm_specs(spec_list: List[str]) -> List[PTMSpec]:
    """Parse multiple PTM specifications."""
    return [parse_ptm_spec(s) for s in spec_list]


# ============================================================================
# NCAA AND HETATM PROCESSING
# ============================================================================

def process_prediction_pdb(
    pdb_path: str,
    output_path: str,
    ptm_specs: List[PTMSpec],
    catalytic_residues: List[CatalyticResidue],
    verbose: bool = False
) -> None:
    """
    Process prediction PDB:
    1. Handle NCAA residues (convert to canonical, cut atoms)
    2. STRIP ALL HETATM from prediction (they will be added from reference later)
    """
    if verbose:
        print(f"\nProcessing prediction PDB...")
        print(f"  PTM specs: {len(ptm_specs)}")

    # Build mapping from catres_index to residue info
    catres_map = {cr.catres_index: cr for cr in catalytic_residues}

    # Build PTM mapping: (chain, resnum) -> PTMSpec
    ptm_map = {}
    for ptm_spec in ptm_specs:
        if ptm_spec.catres_index in catres_map:
            catres = catres_map[ptm_spec.catres_index]
            key = (catres.chain, catres.resnum)
            ptm_map[key] = ptm_spec
            if verbose:
                print(f"  PTM: Catres #{ptm_spec.catres_index} ({catres.chain} {catres.resnum}) "
                      f"{ptm_spec.canonical_resname} -> {ptm_spec.ncaa_resname}")
                print(f"    Atoms to cut: {ptm_spec.atoms_to_cut}")

    # Process PDB file
    with open(pdb_path, 'r') as f:
        lines = f.readlines()

    output_lines = []
    hetatm_removed = 0
    atoms_cut = 0
    ncaa_converted = 0

    for line in lines:
        # STRIP ALL HETATM - they will be added from reference later
        if line.startswith("HETATM"):
            # Check if this is an NCAA that should be converted
            try:
                chain = line[21].strip()
                resnum = int(line[22:26].strip())
                atom_name = line[12:16].strip()
                key = (chain, resnum)

                if key in ptm_map:
                    ptm_spec = ptm_map[key]
                    # Cut specified atoms
                    if atom_name in ptm_spec.atoms_to_cut:
                        atoms_cut += 1
                        if verbose:
                            print(f"    Cutting atom {atom_name} from {chain} {resnum}")
                        continue

                    # Convert HETATM to ATOM AND change residue name from NCAA to canonical
                    # PDB format: columns 0-5 = record type, 17-19 = residue name
                    new_line = "ATOM  " + line[6:17] + f"{ptm_spec.canonical_resname:>3}" + line[20:]
                    output_lines.append(new_line)
                    ncaa_converted += 1
                    if verbose and ncaa_converted == 1:
                        print(f"    Converting {ptm_spec.ncaa_resname} -> {ptm_spec.canonical_resname}")
                    continue
            except (ValueError, IndexError):
                pass

            # All other HETATM are removed
            hetatm_removed += 1
            if verbose and hetatm_removed <= 10:
                resname = line[17:20].strip() if len(line) > 20 else "???"
                print(f"    Removing HETATM: {resname}")
            continue

        # Keep ATOM and other lines
        if line.startswith(("ATOM", "TER", "END")):
            output_lines.append(line)
        elif line.startswith("REMARK") or line.startswith("HEADER"):
            # Keep some header lines but we'll replace with ref later
            pass
        else:
            output_lines.append(line)

    # Write output
    with open(output_path, 'w') as f:
        f.writelines(output_lines)

    if verbose:
        print(f"\nPrediction PDB processing complete:")
        print(f"  HETATM removed: {hetatm_removed}")
        print(f"  NCAA atoms cut: {atoms_cut}")
        print(f"  NCAA atoms converted to ATOM: {ncaa_converted}")


# ============================================================================
# SYMMETRIC ATOM HANDLING
# ============================================================================

def _get_symmetric_atom_names(resname: str) -> Set[str]:
    """Get set of all symmetric atom names for a residue type."""
    if resname not in SYMMETRIC_ATOM_PAIRS:
        return set()
    names = set()
    for pair in SYMMETRIC_ATOM_PAIRS[resname]:
        names.add(pair[0])
        names.add(pair[1])
    return names


def _apply_swap_to_name(name: str, resname: str) -> str:
    """Apply symmetric swap to an atom name. Returns swapped name or original if not symmetric."""
    if resname not in SYMMETRIC_ATOM_PAIRS:
        return name
    for a, b in SYMMETRIC_ATOM_PAIRS[resname]:
        if name == a:
            return b
        elif name == b:
            return a
    return name


def _determine_best_swap_for_residue(
    ref_res: Residue,
    mob_res: Residue,
    verbose: bool = False
) -> bool:
    """
    Determine whether to swap symmetric atoms using per-residue Kabsch alignment.

    Aligns on non-symmetric atoms only, then checks which assignment
    (original vs swapped) gives lower RMSD for the symmetric atoms.

    Args:
        ref_res: Reference residue (Biopython Residue object)
        mob_res: Mobile residue (Biopython Residue object)
        verbose: Print debug info

    Returns:
        True if swapping gives better RMSD, False otherwise
    """
    resname = ref_res.resname
    if resname not in SYMMETRIC_ATOM_PAIRS:
        return False

    symmetric_names = _get_symmetric_atom_names(resname)

    # Get non-symmetric heavy atoms present in both
    ref_atoms = {a.name: a for a in ref_res.get_atoms() if a.element != 'H'}
    mob_atoms = {a.name: a for a in mob_res.get_atoms() if a.element != 'H'}

    common_nonsym = [n for n in ref_atoms if n in mob_atoms and n not in symmetric_names]

    if len(common_nonsym) < 3:
        # Not enough atoms to align, default to no swap
        return False

    # Get coords for non-symmetric atoms
    ref_nonsym_coords = np.array([ref_atoms[n].coord for n in common_nonsym])
    mob_nonsym_coords = np.array([mob_atoms[n].coord for n in common_nonsym])

    # Center coords
    ref_centroid = np.mean(ref_nonsym_coords, axis=0)
    mob_centroid = np.mean(mob_nonsym_coords, axis=0)
    ref_centered = ref_nonsym_coords - ref_centroid
    mob_centered = mob_nonsym_coords - mob_centroid

    # Kabsch rotation
    H = mob_centered.T @ ref_centered
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T

    # Apply rotation to all mobile coords
    def transform_coord(coord):
        return R @ (coord - mob_centroid) + ref_centroid

    # Get symmetric atoms present in both
    common_sym = [n for n in ref_atoms if n in mob_atoms and n in symmetric_names]
    if not common_sym:
        return False

    # Calculate RMSD for original assignment
    rmsd_original = 0.0
    for name in common_sym:
        ref_coord = ref_atoms[name].coord
        mob_coord_transformed = transform_coord(mob_atoms[name].coord)
        rmsd_original += np.sum((ref_coord - mob_coord_transformed) ** 2)

    # Calculate RMSD for swapped assignment
    rmsd_swapped = 0.0
    for name in common_sym:
        swapped_name = _apply_swap_to_name(name, resname)
        if swapped_name in mob_atoms:
            ref_coord = ref_atoms[name].coord
            mob_coord_transformed = transform_coord(mob_atoms[swapped_name].coord)
            rmsd_swapped += np.sum((ref_coord - mob_coord_transformed) ** 2)
        else:
            # Swapped atom not present, use original
            ref_coord = ref_atoms[name].coord
            mob_coord_transformed = transform_coord(mob_atoms[name].coord)
            rmsd_swapped += np.sum((ref_coord - mob_coord_transformed) ** 2)

    should_swap = rmsd_swapped < rmsd_original

    if verbose and should_swap:
        print(f"      {resname} {ref_res.id[1]}: swapping symmetric atoms "
              f"(RMSD orig={np.sqrt(rmsd_original/len(common_sym)):.3f} vs swap={np.sqrt(rmsd_swapped/len(common_sym)):.3f})")

    return should_swap


def resolve_symmetric_atoms(
    ref_structure: Structure.Structure,
    mobile_structure: Structure.Structure,
    catalytic_residues: List['CatalyticResidue'],
    verbose: bool = False
) -> Dict[Tuple[str, int], bool]:
    """
    Determine best symmetric atom assignment for each catalytic residue.

    Called ONCE after alignment/optimization, before any metrics are calculated.
    All subsequent RMSD, lDDT, etc. calculations use this resolved mapping.

    Args:
        ref_structure: Reference BioPython structure
        mobile_structure: Mobile (aligned) BioPython structure
        catalytic_residues: List of CatalyticResidue objects
        verbose: Print debug info

    Returns:
        Dict mapping (chain, resnum) -> should_swap (bool)
    """
    swap_map = {}

    # Build residue lookup for both structures
    ref_res_dict = {}
    mob_res_dict = {}

    for model in ref_structure:
        for chain in model:
            for res in chain:
                if res.id[0] == ' ':  # Standard residue
                    ref_res_dict[(chain.id, res.id[1])] = res

    for model in mobile_structure:
        for chain in model:
            for res in chain:
                if res.id[0] == ' ':  # Standard residue
                    mob_res_dict[(chain.id, res.id[1])] = res

    if verbose:
        print("\n  Resolving symmetric atoms for catalytic residues...")

    for catres in catalytic_residues:
        key = (catres.chain, catres.resnum)
        resname = catres.resname

        # Check if this residue has symmetric atoms
        if resname not in SYMMETRIC_ATOM_PAIRS:
            swap_map[key] = False
            continue

        # Get residues from both structures
        ref_res = ref_res_dict.get(key)
        mob_res = mob_res_dict.get(key)

        if ref_res is None or mob_res is None:
            swap_map[key] = False
            continue

        # Determine best swap
        should_swap = _determine_best_swap_for_residue(ref_res, mob_res, verbose)
        swap_map[key] = should_swap

        if verbose and should_swap:
            print(f"    {resname} {catres.chain}/{catres.resnum}: using swapped assignment")

    return swap_map


# ============================================================================
# PDB UTILITIES
# ============================================================================

def separate_protein_and_hetatm(pdb_content: List[str]) -> Tuple[List[str], List[str]]:
    """Separate protein and HETATM lines."""
    protein_lines = []
    hetatm_lines = []
    for line in pdb_content:
        if line.startswith("HETATM"):
            hetatm_lines.append(line)
        elif line.startswith(("ATOM", "TER")):
            protein_lines.append(line)
    return protein_lines, hetatm_lines


def renumber_pdb_atoms(pdb_lines: List[str], start_number: int = 1) -> List[str]:
    """Renumber all ATOM and HETATM lines sequentially."""
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


def build_his_tautomer_map_from_raw_pdb(pdb_path: str, debug: bool = False) -> Dict[Tuple[str, int], str]:
    """Build HIS tautomer map from raw PDB file."""
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

            if resn not in ["HIS", "HIE", "HID", "HIP"]:
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
        else:
            his_map[key] = "HIS"

    return his_map


def extract_all_remark_lines(pdb_path: str) -> List[str]:
    """Extract all REMARK, HEADER, HETNAM, LINK lines from PDB."""
    headers = []
    seen = set()
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith(("HEADER", "REMARK", "HETNAM", "LINK", "CONECT")):
                if line not in seen:
                    headers.append(line)
                    seen.add(line)
    return headers


def parse_remark666_lines(pdb_path: str) -> Tuple[List[str], List[CatalyticResidue]]:
    """Parse REMARK 666 lines from a PDB file."""
    remark_lines = []
    catalytic_residues = []
    seen_catres = set()

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("REMARK 666"):
                remark_lines.append(line)

                if "MATCH TEMPLATE" in line and "MATCH MOTIF" in line:
                    parts = line.split()
                    try:
                        motif_idx = parts.index("MOTIF")
                        chain = parts[motif_idx + 1]
                        resname = parts[motif_idx + 2]
                        resnum = int(parts[motif_idx + 3])
                        catres_index = int(parts[motif_idx + 4])

                        key = (chain, resnum, catres_index)
                        if key not in seen_catres:
                            catalytic_residues.append(
                                CatalyticResidue(chain, resname, resnum, catres_index)
                            )
                            seen_catres.add(key)
                    except (ValueError, IndexError):
                        print(f"WARNING: Could not parse REMARK 666 line: {line.strip()}")

    catalytic_residues.sort(key=lambda x: x.catres_index)
    return remark_lines, catalytic_residues


def filter_catres_by_subset(
    catalytic_residues: List[CatalyticResidue],
    subset_indices: Optional[List[int]]
) -> List[CatalyticResidue]:
    """Filter catalytic residues to only include those in the subset."""
    if subset_indices is None:
        return catalytic_residues

    max_index = max(cr.catres_index for cr in catalytic_residues) if catalytic_residues else 0
    for idx in subset_indices:
        if idx > max_index or idx < 1:
            print(f"WARNING: catres_subset index {idx} out of range (max={max_index})")

    return [cr for cr in catalytic_residues if cr.catres_index in subset_indices]


def extract_hetatm_lines(pdb_path: str) -> List[str]:
    """Extract all HETATM lines from a PDB file."""
    hetatm_lines = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("HETATM"):
                hetatm_lines.append(line)
    return hetatm_lines


# ============================================================================
# RMSD CALCULATIONS
# ============================================================================

def get_backbone_atoms(residue: Residue) -> List[Atom]:
    """Get backbone atoms (N, CA, C, O) from a residue."""
    atoms = []
    for atom_name in ['N', 'CA', 'C', 'O']:
        if atom_name in residue:
            atoms.append(residue[atom_name])
    return atoms


def get_ca_atom(residue: Residue) -> Optional[Atom]:
    """Get CA atom from a residue."""
    return residue['CA'] if 'CA' in residue else None


def get_all_heavy_atoms(residue: Residue) -> List[Atom]:
    """Get all non-hydrogen atoms from a residue."""
    return [atom for atom in residue.get_atoms() if atom.element != 'H']


def calculate_rmsd(atoms1: List[Atom], atoms2: List[Atom]) -> float:
    """Calculate RMSD between two lists of atoms."""
    if len(atoms1) != len(atoms2) or len(atoms1) == 0:
        return np.inf

    coords1 = np.array([atom.coord for atom in atoms1])
    coords2 = np.array([atom.coord for atom in atoms2])

    diff = coords1 - coords2
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))


def calculate_rmsd_with_residues(
    residues1: List[Residue],
    residues2: List[Residue],
    atom_selector: str = 'ca'
) -> Tuple[float, List[Atom], List[Atom]]:
    """Calculate RMSD between residue lists."""
    if len(residues1) != len(residues2):
        return np.inf, [], []

    atoms1 = []
    atoms2 = []

    for res1, res2 in zip(residues1, residues2):
        if atom_selector == 'ca':
            a1 = get_ca_atom(res1)
            a2 = get_ca_atom(res2)
            if a1 and a2:
                atoms1.append(a1)
                atoms2.append(a2)
        elif atom_selector == 'backbone':
            bb1 = get_backbone_atoms(res1)
            bb2 = get_backbone_atoms(res2)
            if len(bb1) == len(bb2) and len(bb1) > 0:
                atoms1.extend(bb1)
                atoms2.extend(bb2)
        elif atom_selector == 'all_heavy':
            heavy1 = get_all_heavy_atoms(res1)
            heavy2 = get_all_heavy_atoms(res2)
            atom_dict1 = {a.name: a for a in heavy1}
            atom_dict2 = {a.name: a for a in heavy2}
            common_atoms = set(atom_dict1.keys()) & set(atom_dict2.keys())
            for atom_name in sorted(common_atoms):
                atoms1.append(atom_dict1[atom_name])
                atoms2.append(atom_dict2[atom_name])

    return calculate_rmsd(atoms1, atoms2), atoms1, atoms2


def calculate_rmsd_with_symmetry(
    residues1: List[Residue],
    residues2: List[Residue],
    catalytic_residues: List['CatalyticResidue'],
    swap_map: Dict[Tuple[str, int], bool],
    atom_selector: str = 'all_heavy'
) -> Tuple[float, List[Atom], List[Atom]]:
    """
    Calculate RMSD between residue lists with symmetric atom handling.

    Uses pre-resolved swap_map to determine atom naming for symmetric residues.

    Args:
        residues1: Reference residues
        residues2: Mobile residues
        catalytic_residues: List of CatalyticResidue objects for chain/resnum lookup
        swap_map: Pre-computed mapping of (chain, resnum) -> should_swap
        atom_selector: 'ca', 'backbone', or 'all_heavy'

    Returns:
        Tuple of (rmsd, atoms_list1, atoms_list2)
    """
    if len(residues1) != len(residues2):
        return np.inf, [], []

    atoms1 = []
    atoms2 = []

    for res1, res2, catres in zip(residues1, residues2, catalytic_residues):
        key = (catres.chain, catres.resnum)
        should_swap = swap_map.get(key, False)
        resname = res1.resname

        if atom_selector == 'ca':
            a1 = get_ca_atom(res1)
            a2 = get_ca_atom(res2)
            if a1 and a2:
                atoms1.append(a1)
                atoms2.append(a2)

        elif atom_selector == 'backbone':
            bb1 = get_backbone_atoms(res1)
            bb2 = get_backbone_atoms(res2)
            if len(bb1) == len(bb2) and len(bb1) > 0:
                atoms1.extend(bb1)
                atoms2.extend(bb2)

        elif atom_selector == 'all_heavy':
            heavy1 = get_all_heavy_atoms(res1)
            heavy2 = get_all_heavy_atoms(res2)
            atom_dict1 = {a.name: a for a in heavy1}
            atom_dict2 = {a.name: a for a in heavy2}

            # If swapping, remap the atom names for structure2
            if should_swap and resname in SYMMETRIC_ATOM_PAIRS:
                atom_dict2_swapped = {}
                for name, atom in atom_dict2.items():
                    swapped_name = _apply_swap_to_name(name, resname)
                    atom_dict2_swapped[swapped_name] = atom
                atom_dict2 = atom_dict2_swapped

            common_atoms = set(atom_dict1.keys()) & set(atom_dict2.keys())
            for atom_name in sorted(common_atoms):
                atoms1.append(atom_dict1[atom_name])
                atoms2.append(atom_dict2[atom_name])

    return calculate_rmsd(atoms1, atoms2), atoms1, atoms2


def calculate_catres_subset_lddt(
    ref_structure: Structure.Structure,
    mobile_structure: Structure.Structure,
    catres_subset: List['CatalyticResidue'],
    swap_map: Dict[Tuple[str, int], bool],
    thresholds: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
    inclusion_radius: float = 15.0,
    verbose: bool = False
) -> float:
    """
    Calculate lDDT for catalytic residue subset.

    IMPORTANT: Only considers atom pairs BETWEEN residues in catres_subset.
    This measures how well the inter-residue geometry of the catalytic site
    is preserved, ignoring interactions with the rest of the protein.

    Args:
        ref_structure: Reference BioPython structure
        mobile_structure: Mobile (aligned) BioPython structure
        catres_subset: List of CatalyticResidue objects defining the subset
        swap_map: Pre-computed mapping of (chain, resnum) -> should_swap
        thresholds: Distance thresholds for lDDT calculation
        inclusion_radius: Max distance in reference to consider for lDDT
        verbose: Print debug info

    Returns:
        lDDT score (0.0 to 1.0)
    """
    if not catres_subset:
        return float('nan')

    # Build residue lookup for both structures
    ref_res_dict = {}
    mob_res_dict = {}

    for model in ref_structure:
        for chain in model:
            for res in chain:
                if res.id[0] == ' ':
                    ref_res_dict[(chain.id, res.id[1])] = res

    for model in mobile_structure:
        for chain in model:
            for res in chain:
                if res.id[0] == ' ':
                    mob_res_dict[(chain.id, res.id[1])] = res

    # Collect all atoms from catres_subset with their resolved names
    # Each entry: (residue_key, atom_name, ref_coord, mob_coord)
    atom_data = []

    for catres in catres_subset:
        key = (catres.chain, catres.resnum)
        resname = catres.resname
        should_swap = swap_map.get(key, False)

        ref_res = ref_res_dict.get(key)
        mob_res = mob_res_dict.get(key)

        if ref_res is None or mob_res is None:
            continue

        ref_heavy = {a.name: a.coord for a in ref_res.get_atoms() if a.element != 'H'}
        mob_heavy = {a.name: a.coord for a in mob_res.get_atoms() if a.element != 'H'}

        # Apply swap to mobile atom names if needed
        if should_swap and resname in SYMMETRIC_ATOM_PAIRS:
            mob_heavy_swapped = {}
            for name, coord in mob_heavy.items():
                swapped_name = _apply_swap_to_name(name, resname)
                mob_heavy_swapped[swapped_name] = coord
            mob_heavy = mob_heavy_swapped

        # Get common atoms
        common_names = set(ref_heavy.keys()) & set(mob_heavy.keys())
        for atom_name in common_names:
            atom_data.append({
                'res_key': key,
                'atom_name': atom_name,
                'ref_coord': ref_heavy[atom_name],
                'mob_coord': mob_heavy[atom_name]
            })

    if len(atom_data) < 2:
        return float('nan')

    # Calculate all pairwise distances between atoms from DIFFERENT residues
    preserved_count = 0
    total_comparisons = 0

    for i in range(len(atom_data)):
        for j in range(i + 1, len(atom_data)):
            # Only consider pairs from DIFFERENT residues
            if atom_data[i]['res_key'] == atom_data[j]['res_key']:
                continue

            ref_dist = np.linalg.norm(
                np.array(atom_data[i]['ref_coord']) - np.array(atom_data[j]['ref_coord'])
            )

            # Only consider pairs within inclusion radius in reference
            if ref_dist > inclusion_radius:
                continue

            mob_dist = np.linalg.norm(
                np.array(atom_data[i]['mob_coord']) - np.array(atom_data[j]['mob_coord'])
            )

            # Check each threshold
            for thresh in thresholds:
                total_comparisons += 1
                if abs(ref_dist - mob_dist) < thresh:
                    preserved_count += 1

    if total_comparisons == 0:
        return float('nan')

    lddt = preserved_count / total_comparisons

    if verbose:
        print(f"    lDDT: {lddt:.4f} ({preserved_count}/{total_comparisons} preserved, "
              f"{len(atom_data)} atoms from {len(catres_subset)} residues)")

    return lddt


# ============================================================================
# RESIDUE MATCHING
# ============================================================================

def match_residues_by_chain_and_number(
    structure1: Structure.Structure,
    structure2: Structure.Structure
) -> List[Tuple[Residue, Residue]]:
    """Match residues between two structures."""
    res_dict1 = {}
    res_dict2 = {}

    for model in structure1:
        for chain in model:
            for res in chain:
                if res.id[0] == ' ':
                    res_dict1[(chain.id, res.id[1])] = res

    for model in structure2:
        for chain in model:
            for res in chain:
                if res.id[0] == ' ':
                    res_dict2[(chain.id, res.id[1])] = res

    common_keys = set(res_dict1.keys()) & set(res_dict2.keys())
    return [(res_dict1[k], res_dict2[k]) for k in sorted(common_keys)]


def get_catalytic_residues_from_structure(
    structure: Structure.Structure,
    catalytic_residues: List[CatalyticResidue]
) -> List[Residue]:
    """Extract catalytic residues from structure."""
    res_dict = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == ' ':
                    res_dict[(chain.id, residue.id[1])] = residue

    catres_list = []
    for catres in catalytic_residues:
        key = (catres.chain, catres.resnum)
        if key in res_dict:
            catres_list.append(res_dict[key])
        else:
            print(f"WARNING: Catalytic residue {catres.chain} {catres.resnum} not found")

    return catres_list


# ============================================================================
# TM-ALIGN SUPERPOSITION
# ============================================================================

def tmalign_superimpose(
    ref_structure: Structure.Structure,
    mobile_structure: Structure.Structure,
    residue_subset: Optional[List['CatalyticResidue']] = None,
    verbose: bool = False
) -> Tuple[float, int]:
    """
    Perform TM-align style superposition using biotite.

    Args:
        ref_structure: Reference BioPython structure
        mobile_structure: Mobile BioPython structure (will be modified in-place)
        residue_subset: Optional list of CatalyticResidue to restrict alignment
        verbose: Print debug info

    Returns:
        Tuple of (tm_score, n_aligned_residues)
        Returns (0.0, 0) if biotite not available or alignment fails
    """
    if not HAS_BIOTITE:
        if verbose:
            print("  WARNING: Biotite not available, cannot perform TM-align")
        return 0.0, 0

    try:
        # Convert BioPython structures to biotite AtomArrays
        # We need to write to temp files and read back with biotite
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            ref_temp = f.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            mob_temp = f.name

        # Save structures
        io = PDBIO()
        io.set_structure(ref_structure)
        io.save(ref_temp)
        io.set_structure(mobile_structure)
        io.save(mob_temp)

        # Load with biotite
        ref_file = bpdb.PDBFile.read(ref_temp)
        mob_file = bpdb.PDBFile.read(mob_temp)

        ref_array = ref_file.get_structure(model=1)
        mob_array = mob_file.get_structure(model=1)

        # Filter to CA atoms for superposition
        ref_ca_mask = (ref_array.atom_name == "CA") & (ref_array.element != "")
        mob_ca_mask = (mob_array.atom_name == "CA") & (mob_array.element != "")

        ref_ca = ref_array[ref_ca_mask]
        mob_ca = mob_array[mob_ca_mask]

        if residue_subset is not None:
            # Filter to specified residues
            subset_keys = {(cr.chain, cr.resnum) for cr in residue_subset}

            ref_subset_mask = np.array([
                (ref_ca.chain_id[i], ref_ca.res_id[i]) in subset_keys
                for i in range(len(ref_ca))
            ])
            mob_subset_mask = np.array([
                (mob_ca.chain_id[i], mob_ca.res_id[i]) in subset_keys
                for i in range(len(mob_ca))
            ])

            ref_ca = ref_ca[ref_subset_mask]
            mob_ca = mob_ca[mob_subset_mask]

        if len(ref_ca) == 0 or len(mob_ca) == 0:
            if verbose:
                print("  WARNING: No CA atoms found for TM-align")
            os.unlink(ref_temp)
            os.unlink(mob_temp)
            return 0.0, 0

        # Perform TM-align style superposition
        # biotite's superimpose_structural_homologs uses a TM-align inspired algorithm
        # Returns: (fitted, transform, fixed_indices, mobile_indices)
        mob_superimposed, transformation, fixed_indices, mobile_indices = superimpose_structural_homologs(
            ref_ca, mob_ca
        )

        # Number of aligned residues (those within d0 threshold)
        n_aligned = len(fixed_indices)

        # Calculate TM-score
        # TM-score = (1/L) * sum_i(1/(1 + (d_i/d0)^2))
        # where L = length of target, d0 = 1.24 * cuberoot(L - 15) - 1.8
        L = len(ref_ca)
        if L > 15:
            d0 = 1.24 * np.cbrt(L - 15) - 1.8
        else:
            d0 = 0.5  # Minimum d0

        # Calculate distances after superposition for the aligned residues
        if n_aligned > 0:
            aligned_ref_coords = ref_ca.coord[fixed_indices]
            aligned_mob_coords = mob_superimposed.coord[mobile_indices]
            distances = np.sqrt(np.sum((aligned_ref_coords - aligned_mob_coords) ** 2, axis=1))
            tm_score = np.sum(1 / (1 + (distances / d0) ** 2)) / L
        else:
            tm_score = float('nan')  # FIX (Bug C): NaN signals "alignment failed", not "perfect" or "0"

        if verbose:
            print(f"  TM-align: TM-score={tm_score:.4f}, {n_aligned}/{len(ref_ca)} CA atoms aligned, d0={d0:.2f}")

        # Apply transformation to ALL atoms in mobile structure
        # Biotite's AffineTransformation has: rotation (3D), center_translation, target_translation
        # The full transform is: R @ (coord - center_translation) + target_translation
        # We need to extract these and apply to BioPython structure

        # Get transformation components
        rot_matrix = transformation.rotation[0]  # Extract 2D rotation from 3D array
        center_trans = transformation.center_translation[0] if transformation.center_translation.ndim > 1 else transformation.center_translation
        target_trans = transformation.target_translation[0] if transformation.target_translation.ndim > 1 else transformation.target_translation

        # FIX (Bug B): biotite stores center_translation = -mob_centroid (already
        # negated) and AffineTransformation.apply() does `mobile + center_translation`
        # i.e. it ADDS center_trans (not subtract). Previous code did `-` which moved
        # the structure by 2 * mob_centroid in the wrong direction, producing TM-score
        # 0.99 (computed on biotite's internal fit) but BioPython RMSD ~5.9 Å.
        # The mathematically correct form: R @ (coord + center_trans) + target_trans.
        for atom in mobile_structure.get_atoms():
            old_coord = np.array(atom.coord)
            centered = old_coord + center_trans
            rotated = rot_matrix @ centered
            new_coord = rotated + target_trans
            atom.coord = new_coord

        # Cleanup
        os.unlink(ref_temp)
        os.unlink(mob_temp)

        return tm_score, n_aligned

    except Exception as e:
        if verbose:
            print(f"  WARNING: TM-align failed: {e}")
            import traceback
            traceback.print_exc()
        return 0.0, 0


# ============================================================================
# ITERATIVE ALIGNMENT
# ============================================================================

class IterativeAligner:
    """Performs iterative alignment with outlier removal."""

    def __init__(
        self,
        ref_structure: Structure.Structure,
        mobile_structure: Structure.Structure,
        atom_selector: str = 'ca',
        rmsd_threshold: float = 0.5,
        max_iterations: int = 50,
        convergence_threshold: float = 0.001,
        min_atoms: int = 10,
        verbose: bool = True
    ):
        self.ref_structure = ref_structure
        self.mobile_structure = mobile_structure
        self.atom_selector = atom_selector
        self.rmsd_threshold = rmsd_threshold
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.min_atoms = min_atoms
        self.verbose = verbose
        self.superimposer = Superimposer()

    def align_with_outlier_removal(
        self,
        ref_residues: List[Residue],
        mobile_residues: List[Residue]
    ) -> Tuple[float, int, List[int]]:
        """Perform iterative alignment with outlier removal."""
        if len(ref_residues) != len(mobile_residues):
            print(f"ERROR: Residue count mismatch: ref={len(ref_residues)}, mob={len(mobile_residues)}")
            return np.inf, 0, []

        if len(ref_residues) == 0:
            return np.inf, 0, []

        n_residues = len(ref_residues)
        active_indices = list(range(n_residues))
        prev_rmsd = np.inf
        rmsd = np.inf

        for iteration in range(self.max_iterations):
            ref_res_active = [ref_residues[i] for i in active_indices]
            mob_res_active = [mobile_residues[i] for i in active_indices]

            rmsd, ref_atoms, mob_atoms = calculate_rmsd_with_residues(
                ref_res_active, mob_res_active, self.atom_selector
            )

            if len(ref_atoms) == 0:
                return np.inf, iteration + 1, active_indices

            if self.verbose:
                print(f"  Iter {iteration+1}: RMSD={rmsd:.4f}Å ({len(ref_atoms)} atoms, {len(active_indices)} res)")

            if abs(prev_rmsd - rmsd) < self.convergence_threshold:
                if self.verbose:
                    print(f"  Converged after {iteration+1} iterations")
                break

            prev_rmsd = rmsd

            self.superimposer.set_atoms(ref_atoms, mob_atoms)
            self.superimposer.apply(self.mobile_structure.get_atoms())

            # FIX (Bug A): recompute RMSD AFTER applying the Kabsch transform.
            # Otherwise on early-exit paths (esp. small catres-only sets that hit
            # "no outliers" on iter 0) we return the PRE-fit RMSD, which can be huge
            # while the structure is actually well-fitted. The variable `rmsd` now
            # reflects post-Kabsch state for use in convergence check + return.
            rmsd_post, _, _ = calculate_rmsd_with_residues(
                ref_res_active, mob_res_active, self.atom_selector
            )
            rmsd = rmsd_post

            # Calculate per-residue deviations
            per_res_dev = []
            for i, (ref_res, mob_res) in enumerate(zip(ref_res_active, mob_res_active)):
                if self.atom_selector == 'ca':
                    ref_ca = get_ca_atom(ref_res)
                    mob_ca = get_ca_atom(mob_res)
                    if ref_ca and mob_ca:
                        dev = np.linalg.norm(ref_ca.coord - mob_ca.coord)
                    else:
                        dev = 0.0
                else:
                    ref_bb = get_backbone_atoms(ref_res)
                    mob_bb = get_backbone_atoms(mob_res)
                    if len(ref_bb) > 0 and len(mob_bb) > 0:
                        dev = calculate_rmsd(ref_bb, mob_bb)
                    else:
                        dev = 0.0
                per_res_dev.append((active_indices[i], dev))

            outliers = [idx for idx, dev in per_res_dev if dev > self.rmsd_threshold]

            if not outliers:
                if self.verbose:
                    print(f"  No outliers, done")
                break

            worst = max(outliers, key=lambda idx: next(d for i, d in per_res_dev if i == idx))
            active_indices.remove(worst)

            if len(active_indices) < self.min_atoms:
                if self.verbose:
                    print(f"  Stopped at {len(active_indices)} residues (min={self.min_atoms})")
                break

        return rmsd, iteration + 1, active_indices


# ============================================================================
# SIDECHAIN OPTIMIZATION - THOROUGH CONVERGENCE-BASED VERSION
# ============================================================================

# Common rotamer chi angles for each residue type (from Dunbrack library)
# Values in degrees
ROTAMER_LIBRARY = {
    'ARG': [
        (62, 180, 65, 85), (62, 180, 65, -175), (62, 180, 180, 85), (62, 180, 180, 180),
        (-67, 180, 65, 85), (-67, 180, 65, -175), (-67, 180, 180, 85), (-67, 180, 180, 180),
        (-177, 65, 65, 85), (-177, 65, 65, -175), (-177, 180, 65, 85), (-177, 180, 180, 85),
    ],
    'ASN': [
        (62, -10), (62, 30), (-65, -40), (-65, 120), (-174, -20), (-174, 30),
    ],
    'ASP': [
        (62, 10), (62, -70), (-70, -15), (-70, 65), (-177, 0), (-177, 65),
    ],
    'CYS': [
        (62,), (-65,), (-176,),
    ],
    'GLN': [
        (62, 180, 20), (62, 180, -40), (-67, 180, 0), (-67, 180, -40),
        (-174, 65, 20), (-174, 180, 0), (-65, -65, -40), (-65, -65, 100),
    ],
    'GLU': [
        (62, 180, -20), (62, 180, -60), (-67, 180, -20), (-67, -65, -60),
        (-177, 65, -10), (-177, 180, 0), (-65, -65, -40), (-65, -65, -100),
    ],
    'HIS': [
        (62, -75), (62, 80), (-65, -70), (-65, 165), (-177, -75), (-177, 80),
    ],
    'ILE': [
        (62, 170), (62, -60), (-65, 170), (-65, -60), (-57, -60), (-170, 170),
    ],
    'LEU': [
        (62, 80), (-65, 175), (-177, 65), (-65, -65), (-90, 65),
    ],
    'LYS': [
        (62, 180, 68, 180), (62, 180, 180, 65), (-67, 180, 68, 180), (-67, 180, 180, 65),
        (-177, 68, 68, 180), (-62, -68, 180, 65), (-67, 180, 180, 180), (-177, 180, 68, 180),
    ],
    'MET': [
        (62, 180, 75), (62, 180, -75), (-67, 180, 75), (-67, 180, -75),
        (-177, 65, 75), (-177, 180, 75), (-65, -65, 103), (-65, -65, -75),
    ],
    'PHE': [
        (62, 90), (-65, -85), (-177, 80),
    ],
    'PRO': [
        (30, -40), (-30, 30),
    ],
    'SER': [
        (62,), (-65,), (-176,),
    ],
    'THR': [
        (62,), (-65,), (-175,),
    ],
    'TRP': [
        (62, 90), (62, -90), (-65, -90), (-65, 90), (-177, -90), (-177, 90),
    ],
    'TYR': [
        (62, 90), (-65, -85), (-177, 80), (-65, 90),
    ],
    'VAL': [
        (62,), (-60,), (-175,), (175,),
    ],
}


# SP3 carbon atoms that can have bond angle flexibility
SP3_CARBONS = {
    'ARG': ['CB', 'CG', 'CD'],
    'ASN': ['CB'],
    'ASP': ['CB'],
    'CYS': ['CB'],
    'GLN': ['CB', 'CG'],
    'GLU': ['CB', 'CG'],
    'HIS': ['CB'],
    'ILE': ['CB', 'CG1'],
    'LEU': ['CB', 'CG'],
    'LYS': ['CB', 'CG', 'CD', 'CE'],
    'MET': ['CB', 'CG'],
    'PHE': ['CB'],
    'PRO': ['CB', 'CG'],
    'SER': ['CB'],
    'THR': ['CB'],
    'TRP': ['CB'],
    'TYR': ['CB'],
    'VAL': ['CB'],
}


# ============================================================================
# CLASH-PENALTY HELPERS (--apply_clash_penalty)
# ============================================================================
# Two-term physics model:
#   1. UNEXEMPT pairs: pairs NOT close in REF that drift below clash_cutoff in
#      mobile → soft_clash = max(0, cutoff - d)². Weight: clash_weight.
#   2. REF-ANCHOR pairs: pairs that WERE close (< clash_cutoff_ref) in REF →
#      anchored to their REF distance. Penalty kicks in when |d_mob - d_ref|
#      exceeds tolerance, in BOTH directions:
#          close_excess = max(0, d_ref - tol_close - d_mob)
#          loose_excess = max(0, d_mob - d_ref - tol_loose)
#          anchor_penalty = w_close * close_excess² + w_loose * loose_excess²
#      Default tol=0.075 Å and w=7.5 on both sides → strict preservation of
#      REF contact distances (Zn coord, H-bonds, PTM covalent, etc.).
# Symmetric residues (ASP/GLU/PHE/TYR/LEU/VAL/ARG): for each pair_key, we
# store a LIST of allowed d_ref candidates (symmetric-equivalent atom
# orderings can produce multiple). At evaluation time we score against the
# candidate that MINIMIZES the penalty — equivalent to scoring against the
# best swap assignment for this rotamer.

def _atom_key(chain: str, resnum: int, atom_name: str) -> Tuple[str, int, str]:
    return (chain, resnum, atom_name)


def _pair_key(a: Tuple[str, int, str], b: Tuple[str, int, str]) -> Tuple:
    """Unordered atom-pair key (canonical sorted form)."""
    return tuple(sorted((a, b)))


def _symmetric_atom_variants(resname: str, atom_name: str) -> List[str]:
    """Return list of atom names equivalent under residue symmetry (self + swap)."""
    variants = [atom_name]
    if resname in SYMMETRIC_ATOM_PAIRS:
        swapped = _apply_swap_to_name(atom_name, resname)
        if swapped != atom_name:
            variants.append(swapped)
    return variants


def build_ref_anchor_dict(
    ref_structure: Structure.Structure,
    catalytic_residues: List[CatalyticResidue],
    clash_cutoff_ref: float,
    verbose: bool = False,
) -> Dict[Tuple, List[float]]:
    """Scan REF, return dict[pair_key, list of d_ref candidates] for cross-residue
    catres↔ligand AND catres↔other-catres heavy pairs with d_ref < clash_cutoff_ref.

    For symmetric residues we insert all atom-name orderings; multiple ref
    distances per pair_key are kept as a list so the evaluator can score against
    the best matching candidate.
    """
    anchors: Dict[Tuple, List[float]] = {}
    catres_set = {(c.chain, c.resnum) for c in catalytic_residues}
    catres_atoms: List[Tuple[str, int, str, str, np.ndarray]] = []
    ligand_atoms: List[Tuple[str, int, str, str, np.ndarray]] = []
    for model in ref_structure:
        for chain in model:
            for residue in chain:
                hetflag = residue.id[0]
                resnum = residue.id[1]
                resname = residue.resname
                key = (chain.id, resnum)
                is_catres = key in catres_set
                is_hetatm = hetflag != ' ' and hetflag != 'W'
                if not (is_catres or is_hetatm):
                    continue
                for atom in residue.get_atoms():
                    if atom.element == 'H':
                        continue
                    entry = (chain.id, resnum, atom.name, resname, np.array(atom.coord))
                    if is_catres:
                        catres_atoms.append(entry)
                    if is_hetatm and not is_catres:
                        ligand_atoms.append(entry)

    def _add(pk: Tuple, d: float) -> None:
        anchors.setdefault(pk, []).append(d)

    n_cl = n_cc = 0
    cutoff2 = clash_cutoff_ref * clash_cutoff_ref
    # catres ↔ ligand
    for (ca_c, ca_n, ca_a, ca_rn, ca_xyz) in catres_atoms:
        for (lg_c, lg_n, lg_a, lg_rn, lg_xyz) in ligand_atoms:
            d2 = float(np.sum((ca_xyz - lg_xyz) ** 2))
            if d2 >= cutoff2:
                continue
            d = float(np.sqrt(d2))
            for ca_var in _symmetric_atom_variants(ca_rn, ca_a):
                _add(_pair_key(_atom_key(ca_c, ca_n, ca_var),
                               _atom_key(lg_c, lg_n, lg_a)), d)
            n_cl += 1
    # catres ↔ other-catres
    for i in range(len(catres_atoms)):
        for j in range(i + 1, len(catres_atoms)):
            a = catres_atoms[i]; b = catres_atoms[j]
            if (a[0], a[1]) == (b[0], b[1]):
                continue
            d2 = float(np.sum((a[4] - b[4]) ** 2))
            if d2 >= cutoff2:
                continue
            d = float(np.sqrt(d2))
            for av in _symmetric_atom_variants(a[3], a[2]):
                for bv in _symmetric_atom_variants(b[3], b[2]):
                    _add(_pair_key(_atom_key(a[0], a[1], av),
                                   _atom_key(b[0], b[1], bv)), d)
            n_cc += 1
    if verbose:
        print(f"  Ref-anchor pairs: {n_cl} catres↔ligand + {n_cc} catres↔catres "
              f"({len(anchors)} pair-keys, sym-expanded)")
    return anchors


def evaluate_clash_penalty(
    mobile_structure: Structure.Structure,
    catalytic_residues: List[CatalyticResidue],
    ref_anchors: Dict[Tuple, List[float]],
    clash_cutoff_mob: float,
    anchor_tol_close: float = 0.05,
    anchor_tol_loose: float = 0.05,
    anchor_w_close: float = 15.0,
    anchor_w_loose: float = 3.0,
) -> Tuple[int, int, int, float, float]:
    """Evaluate (i) clashes for non-exempt pairs and (ii) anchor drift for
    exempt pairs.

    Returns:
        n_unexpected_clashes   : count of pairs <clash_cutoff_mob that are NOT
                                 in the REF anchor dict
        n_overshot_anchors     : exempt pairs with d_mob < d_ref - tol_close
        n_loosened_anchors     : exempt pairs with d_mob > d_ref + tol_loose
        soft_clash_penalty     : sum of (cutoff - d)² over unexempt clashes
                                 weighted by 1.0 (caller multiplies by clash_weight)
        soft_anchor_penalty    : sum of (w_close * close_excess² +
                                          w_loose * loose_excess²) over exempt pairs
                                 — weights already folded in here (raw cost units)
    """
    catres_set = {(c.chain, c.resnum) for c in catalytic_residues}
    catres_atoms: List[Tuple[str, int, str, np.ndarray]] = []
    ligand_atoms: List[Tuple[str, int, str, np.ndarray]] = []
    for model in mobile_structure:
        for chain in model:
            for residue in chain:
                hetflag = residue.id[0]
                resnum = residue.id[1]
                key = (chain.id, resnum)
                is_catres = key in catres_set
                is_hetatm = hetflag != ' ' and hetflag != 'W'
                if not (is_catres or is_hetatm):
                    continue
                for atom in residue.get_atoms():
                    if atom.element == 'H':
                        continue
                    entry = (chain.id, resnum, atom.name, np.array(atom.coord))
                    if is_catres:
                        catres_atoms.append(entry)
                    if is_hetatm and not is_catres:
                        ligand_atoms.append(entry)

    cutoff2 = clash_cutoff_mob * clash_cutoff_mob
    n_clashes = 0
    n_overshot = 0
    n_loosened = 0
    soft_clash = 0.0
    soft_anchor = 0.0

    def _score_pair(pk: Tuple, d_mob: float) -> Tuple[bool, bool, bool, float, float]:
        """Return (is_anchor, overshot, loosened, anchor_penalty_added, clash_penalty_added)."""
        if pk in ref_anchors:
            # Score against the best-matching ref candidate (handles sym dupes)
            best_pen = None
            best_ov = False
            best_loose = False
            for d_ref in ref_anchors[pk]:
                ce = max(0.0, d_ref - anchor_tol_close - d_mob)
                le = max(0.0, d_mob - d_ref - anchor_tol_loose)
                p = anchor_w_close * ce * ce + anchor_w_loose * le * le
                if best_pen is None or p < best_pen:
                    best_pen = p
                    best_ov = ce > 0
                    best_loose = le > 0
            return True, best_ov, best_loose, best_pen or 0.0, 0.0
        # Not in anchor dict — treat as potential clash
        if d_mob * d_mob < cutoff2:
            return False, False, False, 0.0, (clash_cutoff_mob - d_mob) ** 2
        return False, False, False, 0.0, 0.0

    # catres ↔ ligand
    for (ca_c, ca_n, ca_a, ca_xyz) in catres_atoms:
        for (lg_c, lg_n, lg_a, lg_xyz) in ligand_atoms:
            d2 = float(np.sum((ca_xyz - lg_xyz) ** 2))
            d = float(np.sqrt(d2))
            pk = _pair_key(_atom_key(ca_c, ca_n, ca_a), _atom_key(lg_c, lg_n, lg_a))
            is_anc, ov, lo, ap, cp = _score_pair(pk, d)
            soft_anchor += ap
            soft_clash += cp
            if is_anc:
                n_overshot += int(ov)
                n_loosened += int(lo)
            elif cp > 0:
                n_clashes += 1

    # catres ↔ other-catres
    for i in range(len(catres_atoms)):
        for j in range(i + 1, len(catres_atoms)):
            a = catres_atoms[i]; b = catres_atoms[j]
            if (a[0], a[1]) == (b[0], b[1]):
                continue
            d2 = float(np.sum((a[3] - b[3]) ** 2))
            d = float(np.sqrt(d2))
            pk = _pair_key(_atom_key(a[0], a[1], a[2]), _atom_key(b[0], b[1], b[2]))
            is_anc, ov, lo, ap, cp = _score_pair(pk, d)
            soft_anchor += ap
            soft_clash += cp
            if is_anc:
                n_overshot += int(ov)
                n_loosened += int(lo)
            elif cp > 0:
                n_clashes += 1

    return n_clashes, n_overshot, n_loosened, soft_clash, soft_anchor


class _ClashPlan:
    """Cached, vectorized equivalent of evaluate_clash_penalty().

    The SET of catres + ligand heavy atoms (hence every atom-pair key and its
    REF-anchor membership) is FIXED for a given mobile_structure; only the
    coordinates change as sidechain optimization / rigid docking proceeds. This
    class precomputes everything that does not depend on coordinates ONCE, then
    each evaluate() call:
      - reads live coords from the held Atom references (which are mutated in
        place by set_chi_angle / rigid_dock, so they are always current), and
      - computes the catres↔ligand and catres↔catres distances with vectorized
        numpy instead of a per-pair Python double loop.

    Results are numerically equivalent to evaluate_clash_penalty(): the squared
    distances are computed with the same reduction (sum over the 3 coordinate
    axes), the soft_anchor term is accumulated over the same pairs in the same
    order (bit-identical), and the soft_clash term is summed over the same set
    of clashing pairs (np.sum vs sequential +=, which can differ only at the
    floating-point ULP level — far below the optimizer's selection resolution).
    The returned integer counts are identical.

    It deliberately mirrors evaluate_clash_penalty()'s structure-traversal order
    so the cached atom lists match the original's pair enumeration exactly.
    """

    def __init__(self, mobile_structure, catalytic_residues, ref_anchors,
                 clash_cutoff_mob, anchor_tol_close, anchor_tol_loose,
                 anchor_w_close, anchor_w_loose):
        self.clash_cutoff_mob = clash_cutoff_mob
        self.cutoff2 = clash_cutoff_mob * clash_cutoff_mob
        self.anchor_tol_close = anchor_tol_close
        self.anchor_tol_loose = anchor_tol_loose
        self.anchor_w_close = anchor_w_close
        self.anchor_w_loose = anchor_w_loose

        catres_set = {(c.chain, c.resnum) for c in catalytic_residues}
        catres_atoms = []   # (chain, resnum, name, atom_obj)
        ligand_atoms = []
        for model in mobile_structure:
            for chain in model:
                for residue in chain:
                    hetflag = residue.id[0]
                    resnum = residue.id[1]
                    key = (chain.id, resnum)
                    is_catres = key in catres_set
                    is_hetatm = hetflag != ' ' and hetflag != 'W'
                    if not (is_catres or is_hetatm):
                        continue
                    for atom in residue.get_atoms():
                        if atom.element == 'H':
                            continue
                        entry = (chain.id, resnum, atom.name, atom)
                        if is_catres:
                            catres_atoms.append(entry)
                        if is_hetatm and not is_catres:
                            ligand_atoms.append(entry)

        self.catres_atoms = catres_atoms
        self.ligand_atoms = ligand_atoms
        self.n_c = len(catres_atoms)
        self.n_l = len(ligand_atoms)

        # --- catres↔ligand: precompute anchor pairs + non-anchor clash mask ---
        self.cl_nonanchor_mask = np.ones((self.n_c, self.n_l), dtype=bool)
        self.cl_anchor = []  # list of (i, j, candidates_tuple)
        for i, (ca_c, ca_n, ca_a, _) in enumerate(catres_atoms):
            ak = _atom_key(ca_c, ca_n, ca_a)
            for j, (lg_c, lg_n, lg_a, _) in enumerate(ligand_atoms):
                pk = _pair_key(ak, _atom_key(lg_c, lg_n, lg_a))
                if pk in ref_anchors:
                    self.cl_anchor.append((i, j, tuple(ref_anchors[pk])))
                    self.cl_nonanchor_mask[i, j] = False

        # --- catres↔catres: valid (different-residue) upper-triangle pairs, in
        #     the SAME i<j order the original nested loop visits them ---
        cc_pairs = []  # (i, j)
        for i in range(self.n_c):
            a_c, a_n = catres_atoms[i][0], catres_atoms[i][1]
            for j in range(i + 1, self.n_c):
                if (a_c, a_n) == (catres_atoms[j][0], catres_atoms[j][1]):
                    continue
                cc_pairs.append((i, j))
        if cc_pairs:
            arr = np.asarray(cc_pairs, dtype=int)
            self.cc_I = arr[:, 0]
            self.cc_J = arr[:, 1]
        else:
            self.cc_I = np.zeros(0, dtype=int)
            self.cc_J = np.zeros(0, dtype=int)
        self.cc_nonanchor_mask = np.ones(len(cc_pairs), dtype=bool)
        self.cc_anchor = []  # list of (pos_in_cc_pairs, candidates_tuple)
        for pos, (i, j) in enumerate(cc_pairs):
            pk = _pair_key(_atom_key(*catres_atoms[i][:3]),
                           _atom_key(*catres_atoms[j][:3]))
            if pk in ref_anchors:
                self.cc_anchor.append((pos, tuple(ref_anchors[pk])))
                self.cc_nonanchor_mask[pos] = False

    def _anchor_pen(self, d_mob, cands):
        """Replicates _score_pair's anchor branch: best (min-penalty) candidate."""
        best_pen = None
        best_ov = False
        best_loose = False
        for d_ref in cands:
            ce = max(0.0, d_ref - self.anchor_tol_close - d_mob)
            le = max(0.0, d_mob - d_ref - self.anchor_tol_loose)
            p = self.anchor_w_close * ce * ce + self.anchor_w_loose * le * le
            if best_pen is None or p < best_pen:
                best_pen = p
                best_ov = ce > 0
                best_loose = le > 0
        return (best_pen or 0.0), best_ov, best_loose

    def evaluate(self):
        """Return (n_clashes, n_overshot, n_loosened, soft_clash, soft_anchor)."""
        if self.n_c == 0:
            return 0, 0, 0, 0.0, 0.0

        # Keep the native float32 of Atom.coord: the original evaluate_clash_penalty
        # does np.array(atom.coord) (float32) for the subtraction/square/sum, then
        # float()-widens the per-pair sum to float64. Forcing float64 here would
        # make distances ~1e-7 more precise than the original and perturb the
        # continuous L-BFGS dock trajectory. We replicate the original exactly:
        # float32 sum-of-squares, then .astype(float64) (== float(np.sum(...))).
        C = np.array([e[3].coord for e in self.catres_atoms])
        n_clashes = 0
        n_overshot = 0
        n_loosened = 0
        soft_clash = 0.0
        soft_anchor = 0.0

        # catres ↔ ligand
        if self.n_l > 0:
            L = np.array([e[3].coord for e in self.ligand_atoms])
            diff = C[:, None, :] - L[None, :, :]
            d2 = (diff * diff).sum(axis=2).astype(np.float64)  # float32 sum, then widen
            d = np.sqrt(d2)
            # Non-anchor clashes: accumulate sequentially in the original's nested
            # order (i-catres outer, j-ligand inner = row-major) so the float sum is
            # bit-identical to evaluate_clash_penalty. A vectorized np.sum would
            # reassociate the additions and shift soft_clash by ~1 ULP, which the
            # continuous L-BFGS ligand dock then amplifies to ~1e-9. The clash test
            # is d*d<cutoff2 (re-squaring sqrt) to match _score_pair exactly.
            clash_mask = self.cl_nonanchor_mask & ((d * d) < self.cutoff2)
            if clash_mask.any():
                for i, j in np.argwhere(clash_mask):
                    soft_clash += (self.clash_cutoff_mob - float(d[i, j])) ** 2
                    n_clashes += 1
            for (i, j, cands) in self.cl_anchor:
                ap, ov, lo = self._anchor_pen(float(d[i, j]), cands)
                soft_anchor += ap
                n_overshot += int(ov)
                n_loosened += int(lo)

        # catres ↔ catres
        if self.cc_I.size:
            dd = C[self.cc_I] - C[self.cc_J]
            d2 = (dd * dd).sum(axis=1).astype(np.float64)  # float32 sum, then widen
            d = np.sqrt(d2)
            clash_mask = self.cc_nonanchor_mask & ((d * d) < self.cutoff2)
            if clash_mask.any():
                for pos in np.nonzero(clash_mask)[0]:
                    soft_clash += (self.clash_cutoff_mob - float(d[pos])) ** 2
                    n_clashes += 1
            for (pos, cands) in self.cc_anchor:
                ap, ov, lo = self._anchor_pen(float(d[pos]), cands)
                soft_anchor += ap
                n_overshot += int(ov)
                n_loosened += int(lo)

        return n_clashes, n_overshot, n_loosened, soft_clash, soft_anchor


def inject_ref_ligand_into_mobile(mobile_structure, ref_structure, catalytic_residues, verbose=False):
    """Copy REF HETATM residues (excluding water and any catres-resnum) into mobile_structure.

    The script's preprocessor strips all HETATM from the mobile prediction; this
    helper puts them back AFTER catres alignment so that:
      - evaluate_clash_penalty sees ligand atoms in mobile (anchor scoring works)
      - rigid_dock_ligand_complex has ligand atoms to mutate
      - PDB save writes the (possibly docked) ligand pose, not the raw REF coords

    Returns count of injected residues.
    """
    import copy as _copy
    catres_set = {(c.chain, c.resnum) for c in catalytic_residues}
    # Build map of (chain_id) -> existing chain object in mobile, or create new
    mob_model = next(iter(mobile_structure))
    existing_chain_ids = {c.id for c in mob_model}
    n_injected = 0
    for ref_model in ref_structure:
        for ref_chain in ref_model:
            for ref_residue in ref_chain:
                hetflag = ref_residue.id[0]
                if hetflag == ' ' or hetflag == 'W':
                    continue
                if (ref_chain.id, ref_residue.id[1]) in catres_set:
                    continue
                # Clone and add to mobile under the same chain
                if ref_chain.id not in existing_chain_ids:
                    # Create new chain
                    from Bio.PDB.Chain import Chain as _Chain
                    new_chain = _Chain(ref_chain.id)
                    mob_model.add(new_chain)
                    existing_chain_ids.add(ref_chain.id)
                target_chain = mob_model[ref_chain.id]
                if ref_residue.id in [r.id for r in target_chain]:
                    continue  # already present
                target_chain.add(_copy.deepcopy(ref_residue))
                n_injected += 1
    if verbose and n_injected:
        print(f"  Injected {n_injected} REF HETATM residues into mobile_structure for dock/anchor scoring")
    return n_injected


def _collect_ligand_atoms(mobile_structure, catalytic_residues):
    """Return list of Atom objects that constitute the full ligand complex (HETATM
    non-water that is NOT a catres). For YYE-style ligands this includes the
    substrate plus all bound metals and bound waters that share the same residue
    record. Treated as a single rigid body for docking."""
    catres_set = {(c.chain, c.resnum) for c in catalytic_residues}
    atoms = []
    for model in mobile_structure:
        for chain in model:
            for residue in chain:
                hetflag = residue.id[0]
                if hetflag == ' ' or hetflag == 'W':
                    continue
                if (chain.id, residue.id[1]) in catres_set:
                    continue
                for atom in residue.get_atoms():
                    atoms.append(atom)
    return atoms


def rigid_dock_ligand_complex(
    mobile_structure,
    catalytic_residues,
    ref_anchors,
    clash_cutoff_mob: float,
    anchor_tol_close: float,
    anchor_tol_loose: float,
    anchor_w_close: float,
    anchor_w_loose: float,
    clash_weight: float,
    max_trans: float = 0.5,
    max_rot_deg: float = 5.0,
    verbose: bool = False,
):
    """6-DOF rigid-body optimization of the entire ligand complex (substrate +
    bound metals + bound waters as one rigid unit) to minimize
        clash_weight * soft_clash_penalty + soft_anchor_penalty.

    Bounded by ±max_trans on each translation axis and ±max_rot_deg on each
    rotation axis. Uses scipy L-BFGS-B (analytic gradient by finite-difference).

    Side effect: applies the best transform to the mobile_structure in place.

    Returns dict: translation_A, rotation_deg, penalty_before, penalty_after,
    n_func_evals, n_iterations.
    """
    from scipy.optimize import minimize
    from scipy.spatial.transform import Rotation as _R

    atoms = _collect_ligand_atoms(mobile_structure, catalytic_residues)
    if not atoms:
        if verbose:
            print("  Rigid dock: no ligand atoms found, skipping.")
        return None
    coords0 = np.array([a.coord for a in atoms], dtype=float)
    com0 = coords0.mean(axis=0)
    coords0_centered = coords0 - com0

    # Cached evaluator for the dock objective (~2x faster dock: ~18s vs ~32s; both
    # vs ~678s for the original un-cached evaluator). catres atoms are static during
    # the dock and only the ligand moves (in place), so the plan's live atom
    # references stay current. The cached and original evaluators are numerically
    # equivalent on any static structure; the only observable difference vs legacy
    # is that this continuous L-BFGS-B optimizer amplifies sub-ULP rounding into a
    # sub-milliångström pose shift — the docked PDB still rounds byte-identically,
    # only the logged soft_anchor diagnostic moves at the ~9th significant figure.
    _plan = _ClashPlan(
        mobile_structure, catalytic_residues, ref_anchors, clash_cutoff_mob,
        anchor_tol_close, anchor_tol_loose, anchor_w_close, anchor_w_loose,
    )

    def _apply(params):
        tx, ty, tz, rx, ry, rz = params
        rot = _R.from_rotvec([rx, ry, rz])
        return rot.apply(coords0_centered) + com0 + np.array([tx, ty, tz])

    def _objective(params):
        new_coords = _apply(params)
        for atom, c in zip(atoms, new_coords):
            atom.coord = c
        _, _, _, soft_clash, soft_anchor = _plan.evaluate()
        return float(clash_weight * soft_clash + soft_anchor)

    penalty_before = _objective([0.0] * 6)
    max_rot_rad = float(np.radians(max_rot_deg))
    bounds = [(-max_trans, max_trans)] * 3 + [(-max_rot_rad, max_rot_rad)] * 3
    result = minimize(_objective, x0=np.zeros(6), bounds=bounds,
                      method='L-BFGS-B', options={'ftol': 1e-7, 'gtol': 1e-6, 'maxiter': 200})

    # Make sure final transform is applied (it is from the last objective call,
    # but if scipy backtracked we need to redo)
    new_coords = _apply(result.x)
    for atom, c in zip(atoms, new_coords):
        atom.coord = c
    penalty_after = _objective(result.x)

    tx, ty, tz, rx, ry, rz = result.x
    trans_mag = float(np.sqrt(tx * tx + ty * ty + tz * tz))
    rot_mag_deg = float(np.degrees(np.sqrt(rx * rx + ry * ry + rz * rz)))

    if verbose:
        print(f"  Rigid dock: ΔCOM={trans_mag:.3f} Å, Δrot={rot_mag_deg:.2f}°, "
              f"penalty {penalty_before:.4f} → {penalty_after:.4f} "
              f"({result.nit} iters, {result.nfev} func-evals)")

    return {
        'translation_A': trans_mag,
        'rotation_deg': rot_mag_deg,
        'penalty_before': float(penalty_before),
        'penalty_after': float(penalty_after),
        'n_func_evals': int(result.nfev),
        'n_iterations': int(result.nit),
    }


class SidechainDihedralOptimizer:
    """
    Thorough, convergence-based sidechain optimization.

    Strategy:
    1. Global search phase: Full 360° search or rotamer library sampling
    2. Local refinement: Iteratively refine with decreasing grid sizes
    3. Convergence-based: Continue until no improvement above threshold
    4. Multi-pass: Account for chi angle coupling by multiple passes
    5. Optional SP3 bond angle flexibility (±3° from ideal 109.5°)
    """

    def __init__(
        self,
        ref_structure: Structure.Structure,
        mobile_structure: Structure.Structure,
        catalytic_residues: List[CatalyticResidue],
        verbose: bool = True,
        sp3_angle_flexibility: bool = False,
        sp3_angle_tolerance: float = 3.0,
        apply_clash_penalty: bool = False,
        clash_cutoff_mob: float = 2.0,
        clash_weight: float = 1.0,
        ref_anchors: Optional[Dict[Tuple, List[float]]] = None,
        anchor_tol_close: float = 0.05,
        anchor_tol_loose: float = 0.05,
        anchor_w_close: float = 15.0,
        anchor_w_loose: float = 3.0,
    ):
        self.ref_structure = ref_structure
        self.mobile_structure = mobile_structure
        self.catalytic_residues = catalytic_residues
        self.apply_clash_penalty = apply_clash_penalty
        self.clash_cutoff_mob = clash_cutoff_mob
        self.clash_weight = clash_weight
        self.ref_anchors = ref_anchors if ref_anchors is not None else {}
        self.anchor_tol_close = anchor_tol_close
        self.anchor_tol_loose = anchor_tol_loose
        self.anchor_w_close = anchor_w_close
        self.anchor_w_loose = anchor_w_loose
        self.verbose = verbose
        self.sp3_angle_flexibility = sp3_angle_flexibility
        self.sp3_angle_tolerance = sp3_angle_tolerance

        self.ref_catres = get_catalytic_residues_from_structure(ref_structure, catalytic_residues)
        self.mob_catres = get_catalytic_residues_from_structure(mobile_structure, catalytic_residues)

        # Cached, vectorized clash/anchor evaluator. Built once: the atom set is
        # fixed for this strategy and the plan holds live Atom references, so it
        # reflects sidechain/ligand moves without re-scanning the structure.
        self._clash_plan = None
        if self.apply_clash_penalty:
            self._clash_plan = _ClashPlan(
                self.mobile_structure, self.catalytic_residues, self.ref_anchors,
                self.clash_cutoff_mob, self.anchor_tol_close, self.anchor_tol_loose,
                self.anchor_w_close, self.anchor_w_loose,
            )

    def get_chi_angle(self, residue: Residue, chi_idx: int) -> Optional[float]:
        """Calculate current chi angle for a residue."""
        resname = residue.resname
        if resname not in CHI_DEFINITIONS:
            return None

        chi_atoms_list = CHI_DEFINITIONS[resname]['chi_atoms']
        if chi_idx >= len(chi_atoms_list):
            return None

        atom_names = chi_atoms_list[chi_idx]
        try:
            atoms = [residue[name] for name in atom_names]
            vectors = [Vector(atom.coord) for atom in atoms]
            return calc_dihedral(*vectors)
        except KeyError:
            return None

    def set_chi_angle(self, residue: Residue, chi_idx: int, target_angle: float) -> bool:
        """Set chi angle by rotating downstream atoms."""
        resname = residue.resname
        if resname not in CHI_DEFINITIONS:
            return False

        chi_def = CHI_DEFINITIONS[resname]
        if chi_idx >= len(chi_def['chi_atoms']):
            return False

        atom_names = chi_def['chi_atoms'][chi_idx]
        downstream_names = chi_def['downstream_atoms'][chi_idx]

        try:
            dihedral_atoms = [residue[name] for name in atom_names]
            vectors = [Vector(atom.coord) for atom in dihedral_atoms]
            current_angle = calc_dihedral(*vectors)
            rotation_angle = target_angle - current_angle

            axis_start = np.array(dihedral_atoms[1].coord)
            axis_end = np.array(dihedral_atoms[2].coord)
            axis = axis_end - axis_start
            axis = axis / np.linalg.norm(axis)

            from scipy.spatial.transform import Rotation as R
            rot = R.from_rotvec(rotation_angle * axis)
            rotation_matrix = rot.as_matrix()

            for atom in residue.get_atoms():
                if atom.name in downstream_names:
                    coord = np.array(atom.coord) - axis_start
                    new_coord = rotation_matrix @ coord
                    atom.coord = new_coord + axis_start

            return True
        except KeyError:
            return False

    def get_all_chi_angles(self, residue: Residue) -> List[Optional[float]]:
        """Get all chi angles for a residue."""
        resname = residue.resname
        if resname not in CHI_DEFINITIONS:
            return []
        n_chi = len(CHI_DEFINITIONS[resname]['chi_atoms'])
        return [self.get_chi_angle(residue, i) for i in range(n_chi)]

    def set_all_chi_angles(self, residue: Residue, angles: List[float]) -> None:
        """Set all chi angles for a residue."""
        for chi_idx, angle in enumerate(angles):
            if angle is not None:
                self.set_chi_angle(residue, chi_idx, angle)

    def get_bond_angle(self, residue: Residue, atom1_name: str, atom2_name: str, atom3_name: str) -> Optional[float]:
        """Calculate bond angle between three atoms (atom2 is the central atom)."""
        try:
            atom1 = residue[atom1_name]
            atom2 = residue[atom2_name]
            atom3 = residue[atom3_name]

            v1 = np.array(atom1.coord) - np.array(atom2.coord)
            v2 = np.array(atom3.coord) - np.array(atom2.coord)

            v1_norm = v1 / np.linalg.norm(v1)
            v2_norm = v2 / np.linalg.norm(v2)

            cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
            return np.arccos(cos_angle)
        except (KeyError, ZeroDivisionError):
            return None

    def adjust_sp3_bond_angle(
        self,
        residue: Residue,
        central_atom_name: str,
        target_angle_deviation: float  # In radians, deviation from 109.5°
    ) -> bool:
        """
        Adjust the bond angle at an SP3 carbon by moving downstream atoms.

        This applies a small angular displacement to all atoms downstream of the
        central SP3 carbon to simulate bond angle flexibility.
        """
        resname = residue.resname
        if resname not in SP3_CARBONS or central_atom_name not in SP3_CARBONS[resname]:
            return False

        if resname not in CHI_DEFINITIONS:
            return False

        try:
            central_atom = residue[central_atom_name]
            central_coord = np.array(central_atom.coord)

            # Find upstream atom (toward backbone)
            chi_atoms_list = CHI_DEFINITIONS[resname]['chi_atoms']
            upstream_atom_name = None
            downstream_atoms = set()

            for chi_idx, chi_atoms in enumerate(chi_atoms_list):
                if central_atom_name in chi_atoms:
                    atom_pos = chi_atoms.index(central_atom_name)
                    if atom_pos > 0:
                        upstream_atom_name = chi_atoms[atom_pos - 1]
                    # Get downstream atoms from this chi and all subsequent
                    for ds_idx in range(chi_idx, len(CHI_DEFINITIONS[resname]['downstream_atoms'])):
                        downstream_atoms.update(CHI_DEFINITIONS[resname]['downstream_atoms'][ds_idx])
                    break

            if not upstream_atom_name or not downstream_atoms:
                return False

            upstream_atom = residue[upstream_atom_name]
            upstream_coord = np.array(upstream_atom.coord)

            # Define rotation axis perpendicular to the bond
            bond_vector = central_coord - upstream_coord
            bond_vector = bond_vector / np.linalg.norm(bond_vector)

            # Create perpendicular vector for rotation
            perp = np.array([1, 0, 0])
            if abs(np.dot(bond_vector, perp)) > 0.9:
                perp = np.array([0, 1, 0])
            rotation_axis = np.cross(bond_vector, perp)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

            # Apply rotation to downstream atoms
            from scipy.spatial.transform import Rotation as R
            rot = R.from_rotvec(target_angle_deviation * rotation_axis)
            rotation_matrix = rot.as_matrix()

            for atom in residue.get_atoms():
                if atom.name in downstream_atoms and atom.name != central_atom_name:
                    coord = np.array(atom.coord) - central_coord
                    new_coord = rotation_matrix @ coord
                    atom.coord = new_coord + central_coord

            return True
        except (KeyError, ZeroDivisionError):
            return False

    def optimize_sp3_angles(
        self,
        ref_res: Residue,
        mob_res: Residue
    ) -> Tuple[float, int]:
        """
        Optimize SP3 bond angles within tolerance to minimize RMSD.
        Returns (best_rmsd, n_evaluations).
        """
        if not self.sp3_angle_flexibility:
            rmsd, _ = self.calculate_residue_rmsd(ref_res, mob_res)
            return rmsd, 0

        resname = mob_res.resname
        if resname not in SP3_CARBONS:
            rmsd, _ = self.calculate_residue_rmsd(ref_res, mob_res)
            return rmsd, 0

        best_rmsd, _ = self.calculate_residue_rmsd(ref_res, mob_res)
        n_evals = 1

        # Store original coordinates
        original_coords = {atom.name: atom.coord.copy() for atom in mob_res.get_atoms()}

        # Try small adjustments to each SP3 carbon
        tolerance_rad = np.radians(self.sp3_angle_tolerance)
        angle_steps = np.linspace(-tolerance_rad, tolerance_rad, 5)  # 5 steps including 0

        for sp3_atom in SP3_CARBONS[resname]:
            for angle_adj in angle_steps:
                if abs(angle_adj) < 0.001:  # Skip zero adjustment
                    continue

                # Restore original coords before trying new adjustment
                for atom in mob_res.get_atoms():
                    if atom.name in original_coords:
                        atom.coord = original_coords[atom.name].copy()

                if self.adjust_sp3_bond_angle(mob_res, sp3_atom, angle_adj):
                    rmsd, _ = self.calculate_residue_rmsd(ref_res, mob_res)
                    n_evals += 1

                    if rmsd < best_rmsd:
                        best_rmsd = rmsd
                        # Save as new best
                        original_coords = {atom.name: atom.coord.copy() for atom in mob_res.get_atoms()}

        # Restore best configuration
        for atom in mob_res.get_atoms():
            if atom.name in original_coords:
                atom.coord = original_coords[atom.name].copy()

        return best_rmsd, n_evals

    def calculate_residue_rmsd(self, ref_res: Residue, mob_res: Residue) -> Tuple[float, int]:
        """Calculate all-atom RMSD for a single residue pair."""
        ref_heavy = get_all_heavy_atoms(ref_res)
        mob_heavy = get_all_heavy_atoms(mob_res)

        ref_dict = {a.name: a for a in ref_heavy}
        mob_dict = {a.name: a for a in mob_heavy}

        common = set(ref_dict.keys()) & set(mob_dict.keys())
        if not common:
            return np.inf, 0

        ref_coords = np.array([ref_dict[n].coord for n in sorted(common)])
        mob_coords = np.array([mob_dict[n].coord for n in sorted(common)])

        diff = ref_coords - mob_coords
        rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
        return rmsd, len(common)

    def residue_cost(self, ref_res: Residue, mob_res: Residue) -> Tuple[float, float, int]:
        """Optimization objective for a residue. Returns (cost, rmsd, n_atoms).

        cost = rmsd + clash_weight * soft_clash_penalty
        when --apply_clash_penalty is on; otherwise cost == rmsd.

        Soft clash is computed against the current mobile_structure state (all
        catres + ligand atoms), so it correctly captures cross-residue effects.
        """
        rmsd, n_atoms = self.calculate_residue_rmsd(ref_res, mob_res)
        if not self.apply_clash_penalty:
            return rmsd, rmsd, n_atoms
        _, _, _, soft_clash, soft_anchor = self._clash_plan.evaluate()
        # clash term scaled by clash_weight; anchor term already weighted internally
        return rmsd + self.clash_weight * soft_clash + soft_anchor, rmsd, n_atoms

    def global_search_single_chi(
        self,
        ref_res: Residue,
        mob_res: Residue,
        chi_idx: int,
        step_degrees: float = 30.0
    ) -> Tuple[float, float, float, int]:
        """
        Global search for a single chi angle over full 360°.
        Returns (best_angle, best_cost, best_rmsd, n_evaluations).
        Selection is on COST (rmsd + clash penalty when active); rmsd kept for diagnostics.
        """
        best_cost = np.inf
        best_rmsd = np.inf
        best_angle = 0.0
        n_evals = 0

        angles = np.arange(-180, 180, step_degrees)
        for angle_deg in angles:
            angle_rad = np.radians(angle_deg)
            self.set_chi_angle(mob_res, chi_idx, angle_rad)
            cost, rmsd, _ = self.residue_cost(ref_res, mob_res)
            n_evals += 1

            if cost < best_cost:
                best_cost = cost
                best_rmsd = rmsd
                best_angle = angle_rad

        return best_angle, best_cost, best_rmsd, n_evals

    def local_refine_single_chi(
        self,
        ref_res: Residue,
        mob_res: Residue,
        chi_idx: int,
        center_angle: float,
        half_range: float = 20.0,
        step_degrees: float = 2.0
    ) -> Tuple[float, float, float, int]:
        """
        Local refinement around a center angle.
        Returns (best_angle, best_cost, best_rmsd, n_evaluations).
        """
        best_cost = np.inf
        best_rmsd = np.inf
        best_angle = center_angle
        n_evals = 0

        half_range_rad = np.radians(half_range)
        step_rad = np.radians(step_degrees)
        offsets = np.arange(-half_range_rad, half_range_rad + step_rad, step_rad)

        for offset in offsets:
            test_angle = center_angle + offset
            self.set_chi_angle(mob_res, chi_idx, test_angle)
            cost, rmsd, _ = self.residue_cost(ref_res, mob_res)
            n_evals += 1

            if cost < best_cost:
                best_cost = cost
                best_rmsd = rmsd
                best_angle = test_angle

        return best_angle, best_cost, best_rmsd, n_evals

    def optimize_single_residue(
        self,
        ref_res: Residue,
        mob_res: Residue,
        convergence_threshold: float = 0.001,
        max_passes: int = 10,
        coarse_grid_degrees: float = 30.0,
        fine_grid_degrees: float = 1.0,
    ) -> Tuple[float, float, int, List[float]]:
        """
        Thorough optimization of a single residue using multi-phase approach.

        Phase 1: Try reference chi angles directly
        Phase 2: Try all rotamers from library
        Phase 3: Full 360° global search on each chi (step = coarse_grid_degrees)
        Phase 4: Iterative local refinement with progressively finer grids
                 (sequence = [fine_grid_degrees*10, *5, *2, *1])

        Selection is on COST (rmsd + clash_weight * soft_clash_penalty when
        --apply_clash_penalty is on; otherwise cost == rmsd). The reported
        initial_rmsd/final_rmsd are still RMSD (the diagnostic), but the
        optimizer chose its trajectory by cost.

        Returns (initial_rmsd, final_rmsd, n_evaluations, final_chi_angles)
        """
        initial_cost, initial_rmsd, n_atoms = self.residue_cost(ref_res, mob_res)
        resname = mob_res.resname

        if resname not in CHI_DEFINITIONS:
            return initial_rmsd, initial_rmsd, 0, []

        n_chi = len(CHI_DEFINITIONS[resname]['chi_atoms'])
        if n_chi == 0:
            return initial_rmsd, initial_rmsd, 0, []

        ref_chi = self.get_all_chi_angles(ref_res)
        mob_chi = self.get_all_chi_angles(mob_res)
        total_evals = 0

        if self.verbose:
            print(f"      {resname} {mob_res.id[1]}: {n_chi} chi, {n_atoms} atoms")
            print(f"        Ref chi:  {[f'{np.degrees(a):.1f}°' if a else 'N/A' for a in ref_chi]}")
            print(f"        Mob chi:  {[f'{np.degrees(a):.1f}°' if a else 'N/A' for a in mob_chi]}")
            print(f"        Initial RMSD: {initial_rmsd:.4f} Å, cost: {initial_cost:.4f}")

        best_cost = initial_cost
        best_rmsd = initial_rmsd
        best_chi = mob_chi.copy()

        # PHASE 1: Try reference chi angles directly
        if all(a is not None for a in ref_chi[:n_chi]):
            self.set_all_chi_angles(mob_res, ref_chi)
            cost, rmsd, _ = self.residue_cost(ref_res, mob_res)
            total_evals += 1
            if cost < best_cost:
                best_cost = cost
                best_rmsd = rmsd
                best_chi = ref_chi.copy()
                if self.verbose:
                    print(f"        Phase 1 (ref angles): RMSD {rmsd:.4f} Å, cost {cost:.4f}")

        # PHASE 2: Try rotamer library
        if resname in ROTAMER_LIBRARY:
            for rotamer in ROTAMER_LIBRARY[resname]:
                if len(rotamer) != n_chi:
                    continue
                test_chi = [np.radians(a) for a in rotamer]
                self.set_all_chi_angles(mob_res, test_chi)
                cost, rmsd, _ = self.residue_cost(ref_res, mob_res)
                total_evals += 1
                if cost < best_cost:
                    best_cost = cost
                    best_rmsd = rmsd
                    best_chi = test_chi.copy()

            if self.verbose:
                print(f"        Phase 2 (rotamers): best RMSD {best_rmsd:.4f} Å, cost {best_cost:.4f}")

        # Apply current best before global search
        self.set_all_chi_angles(mob_res, best_chi)

        # PHASE 3: Full 360° global search on each chi (coarse)
        for chi_idx in range(n_chi):
            angle, cost, rmsd, n_evals = self.global_search_single_chi(
                ref_res, mob_res, chi_idx, step_degrees=coarse_grid_degrees
            )
            total_evals += n_evals

            if cost < best_cost:
                best_cost = cost
                best_rmsd = rmsd
                best_chi[chi_idx] = angle
                self.set_chi_angle(mob_res, chi_idx, angle)
            else:
                self.set_chi_angle(mob_res, chi_idx, best_chi[chi_idx])

        if self.verbose:
            print(f"        Phase 3 (global {coarse_grid_degrees:g}°): "
                  f"best RMSD {best_rmsd:.4f} Å, cost {best_cost:.4f}")

        # PHASE 4: Iterative local refinement with convergence
        # Grid sequence scales with fine_grid_degrees: [×10, ×5, ×2, ×1]
        fine_grid_sequence = [fine_grid_degrees * 10.0,
                              fine_grid_degrees * 5.0,
                              fine_grid_degrees * 2.0,
                              fine_grid_degrees * 1.0]
        prev_cost = best_cost
        for pass_num in range(max_passes):
            improved_this_pass = False

            for grid_size in fine_grid_sequence:
                for chi_idx in range(n_chi):
                    center = best_chi[chi_idx]
                    if center is None:
                        continue

                    angle, cost, rmsd, n_evals = self.local_refine_single_chi(
                        ref_res, mob_res, chi_idx,
                        center_angle=center,
                        half_range=grid_size * 2,
                        step_degrees=grid_size
                    )
                    total_evals += n_evals

                    if cost < best_cost - convergence_threshold:
                        best_cost = cost
                        best_rmsd = rmsd
                        best_chi[chi_idx] = angle
                        self.set_chi_angle(mob_res, chi_idx, angle)
                        improved_this_pass = True
                    else:
                        self.set_chi_angle(mob_res, chi_idx, best_chi[chi_idx])

            improvement = prev_cost - best_cost
            if self.verbose and improved_this_pass:
                print(f"        Pass {pass_num+1}: RMSD {best_rmsd:.4f} Å, "
                      f"cost {best_cost:.4f} (Δcost={improvement:.4f})")

            if improvement < convergence_threshold:
                if self.verbose:
                    print(f"        Converged after {pass_num+1} passes")
                break

            prev_cost = best_cost

        # Final application of chi angles
        self.set_all_chi_angles(mob_res, best_chi)

        # PHASE 5: SP3 bond angle optimization (if enabled)
        if self.sp3_angle_flexibility:
            pre_sp3_rmsd = best_rmsd
            best_rmsd, sp3_evals = self.optimize_sp3_angles(ref_res, mob_res)
            total_evals += sp3_evals
            if self.verbose:
                sp3_improvement = pre_sp3_rmsd - best_rmsd
                print(f"        Phase 5 (SP3 angles ±{self.sp3_angle_tolerance}°): "
                      f"RMSD {best_rmsd:.4f} Å (Δ={sp3_improvement:.4f})")

        final_rmsd, _ = self.calculate_residue_rmsd(ref_res, mob_res)

        if self.verbose:
            print(f"        Final RMSD: {final_rmsd:.4f} Å (improvement: {initial_rmsd - final_rmsd:.4f} Å)")
            print(f"        Final chi: {[f'{np.degrees(a):.1f}°' if a else 'N/A' for a in best_chi]}")
            print(f"        Total evaluations: {total_evals}")

        return initial_rmsd, final_rmsd, total_evals, best_chi

    def optimize_all_residues(
        self,
        n_cycles: int = 3,
        fine_grid_degrees: float = 1.0,
        coarse_grid_degrees: float = 30.0,
    ) -> Tuple[float, float, int]:
        """
        Optimize all catalytic residues with convergence-based approach.

        The optimization runs multiple global cycles over all residues,
        as optimizing one residue may affect the optimal configuration of others
        (through steric interactions or cumulative RMSD effects).

        Returns (initial_total_rmsd, final_total_rmsd, total_evaluations)
        """
        if len(self.ref_catres) != len(self.mob_catres):
            print(f"ERROR: Residue count mismatch: ref={len(self.ref_catres)}, mob={len(self.mob_catres)}")
            return np.inf, np.inf, 0

        if len(self.ref_catres) == 0:
            print("ERROR: No catalytic residues found")
            return np.inf, np.inf, 0

        # Calculate initial total RMSD
        initial_rmsd, _, _ = calculate_rmsd_with_residues(
            self.ref_catres, self.mob_catres, 'all_heavy'
        )

        if self.verbose:
            print(f"\n  === SIDECHAIN OPTIMIZATION (Convergence-Based) ===")
            print(f"  Initial catres all-atom RMSD: {initial_rmsd:.4f} Å")
            print(f"  Optimizing {len(self.ref_catres)} residues...")
            print(f"  Using: rotamer library + global search + iterative refinement")

        total_evals = 0
        per_residue_improvements = []
        per_residue_details = []

        # Multiple global cycles for inter-residue effects
        for global_cycle in range(n_cycles):
            if self.verbose:
                print(f"\n  --- Global Cycle {global_cycle + 1}/{n_cycles} ---")

            cycle_start_rmsd, _, _ = calculate_rmsd_with_residues(
                self.ref_catres, self.mob_catres, 'all_heavy'
            )

            for res_idx, (ref_res, mob_res) in enumerate(zip(self.ref_catres, self.mob_catres)):
                if self.verbose:
                    print(f"\n    Residue {res_idx + 1}/{len(self.ref_catres)}:")

                init_rmsd, final_rmsd, n_evals, final_chi = self.optimize_single_residue(
                    ref_res, mob_res,
                    convergence_threshold=0.001,
                    max_passes=10,
                    coarse_grid_degrees=coarse_grid_degrees,
                    fine_grid_degrees=fine_grid_degrees,
                )
                total_evals += n_evals

                if global_cycle == 0:
                    per_residue_improvements.append(init_rmsd - final_rmsd)
                    per_residue_details.append({
                        'resname': mob_res.resname,
                        'resnum': mob_res.id[1],
                        'init_rmsd': init_rmsd,
                        'final_rmsd': final_rmsd,
                        'improvement': init_rmsd - final_rmsd,
                        'final_chi': [np.degrees(a) if a else None for a in final_chi]
                    })

            cycle_end_rmsd, _, _ = calculate_rmsd_with_residues(
                self.ref_catres, self.mob_catres, 'all_heavy'
            )

            cycle_improvement = cycle_start_rmsd - cycle_end_rmsd
            if self.verbose:
                print(f"\n  Cycle {global_cycle + 1} improvement: {cycle_improvement:.4f} Å")
                print(f"  Current total RMSD: {cycle_end_rmsd:.4f} Å")

            # Early stopping if no significant improvement
            if global_cycle > 0 and cycle_improvement < 0.01:
                if self.verbose:
                    print(f"  Stopping early - minimal improvement")
                break

        # Calculate final total RMSD
        final_rmsd, _, _ = calculate_rmsd_with_residues(
            self.ref_catres, self.mob_catres, 'all_heavy'
        )

        if self.verbose:
            print(f"\n  === OPTIMIZATION SUMMARY ===")
            print(f"  Initial catres all-atom RMSD: {initial_rmsd:.4f} Å")
            print(f"  Final catres all-atom RMSD:   {final_rmsd:.4f} Å")
            print(f"  Total improvement: {initial_rmsd - final_rmsd:.4f} Å")
            print(f"  Total evaluations: {total_evals}")
            print(f"\n  Per-residue details:")
            for detail in per_residue_details:
                print(f"    {detail['resname']} {detail['resnum']}: "
                      f"{detail['init_rmsd']:.3f} -> {detail['final_rmsd']:.3f} Å "
                      f"(Δ={detail['improvement']:.3f})")
                if detail['final_chi']:
                    chi_str = ', '.join([f"χ{i+1}={a:.1f}°" if a else "N/A"
                                        for i, a in enumerate(detail['final_chi'])])
                    print(f"      Final: {chi_str}")

        return initial_rmsd, final_rmsd, total_evals


# ============================================================================
# WINNER SELECTION LOGIC
# ============================================================================

def select_winner(
    df: pd.DataFrame,
    tiebreaker_threshold: float = 0.1,
    verbose: bool = True
) -> str:
    """
    Select the winning alignment strategy using hierarchical criteria.

    Priority order:
    1. catres_subset_all_atom_rmsd_after_opt (primary)
    2. catres_subset_ca_rmsd (tiebreaker 1)
    3. all_backbone_rmsd (tiebreaker 2)

    Two values are considered "too close to call" if they differ by less than
    tiebreaker_threshold (default 0.1 Å).

    Returns the strategy name of the winner.
    """
    if df.empty:
        return ""

    # Sort by primary metric
    sorted_df = df.sort_values('catres_subset_all_atom_rmsd_after_opt').reset_index(drop=True)

    if verbose:
        print("\n  Winner selection:")
        print(f"    Tiebreaker threshold: {tiebreaker_threshold} Å")

    # Check if there's a clear winner on primary metric
    if len(sorted_df) == 1:
        winner = sorted_df.loc[0, 'strategy']
        if verbose:
            print(f"    Single strategy, winner: {winner}")
        return winner

    best_primary = sorted_df.loc[0, 'catres_subset_all_atom_rmsd_after_opt']

    # Find all strategies within threshold of the best
    candidates_mask = (sorted_df['catres_subset_all_atom_rmsd_after_opt'] - best_primary) <= tiebreaker_threshold
    candidates = sorted_df[candidates_mask]

    if verbose:
        print(f"    Primary metric (catres_all_atom): best={best_primary:.4f} Å")
        print(f"    Candidates within threshold: {list(candidates['strategy'])}")

    if len(candidates) == 1:
        winner = candidates.iloc[0]['strategy']
        if verbose:
            print(f"    Clear winner on primary metric: {winner}")
        return winner

    # Tiebreaker 1: catres_subset_ca_rmsd
    candidates = candidates.sort_values('catres_subset_ca_rmsd').reset_index(drop=True)
    best_secondary = candidates.loc[0, 'catres_subset_ca_rmsd']
    secondary_mask = (candidates['catres_subset_ca_rmsd'] - best_secondary) <= tiebreaker_threshold
    candidates = candidates[secondary_mask]

    if verbose:
        print(f"    Secondary metric (catres_ca): best={best_secondary:.4f} Å")
        print(f"    Candidates within threshold: {list(candidates['strategy'])}")

    if len(candidates) == 1:
        winner = candidates.iloc[0]['strategy']
        if verbose:
            print(f"    Winner after tiebreaker 1: {winner}")
        return winner

    # Tiebreaker 2: all_backbone_rmsd
    candidates = candidates.sort_values('all_backbone_rmsd').reset_index(drop=True)
    winner = candidates.iloc[0]['strategy']

    if verbose:
        best_tertiary = candidates.loc[0, 'all_backbone_rmsd']
        print(f"    Tertiary metric (backbone): best={best_tertiary:.4f} Å")
        print(f"    Final winner: {winner}")

    return winner


# ============================================================================
# MAIN ALIGNMENT WORKFLOW
# ============================================================================

# Set on the parent process just before a fork Pool is created; inherited by
# forked workers (never pickled). Lets workers reach the aligner without
# serializing it. Only the small (strategy_name, atom_selector, residue_set)
# spec is sent to workers, and only (metrics, aligned_structure) comes back.
_PARALLEL_ALIGNER = None


def _run_strategy_worker(spec):
    """Worker entry point for parallel strategy execution (fork model).

    Runs ONE strategy's alignment + sidechain optimization on the aligner's
    own deepcopy and returns (metrics, aligned_structure). Each strategy is
    fully independent and deterministic, so results are identical to the serial
    path regardless of worker count. The (PyRosetta) hydrogen-addition save is
    deliberately NOT done here — it happens serially in the parent so PyRosetta
    is only ever used in the process where it was initialized.
    """
    strategy_name, atom_selector, residue_set = spec
    aligner = _PARALLEL_ALIGNER
    try:
        metrics = aligner.run_single_strategy(strategy_name, atom_selector, residue_set)
        structure = getattr(aligner, 'mobile_structure_aligned', None)
        return (metrics, structure)
    except Exception as e:
        print(f"ERROR in {strategy_name}: {e}")
        import traceback
        traceback.print_exc()
        return (None, None)


class MultiStrategyAligner:
    """Main class that orchestrates multiple alignment strategies."""

    def __init__(
        self,
        ref_pdb_path: str,
        mobile_pdb_path: str,
        output_dir: str,
        catres_subset: Optional[List[int]] = None,
        ptm_specs: Optional[List[PTMSpec]] = None,
        outlier_rmsd_threshold: float = 0.5,
        convergence_threshold: float = 0.001,
        max_iterations: int = 50,
        min_residues: int = 10,
        sidechain_cycles: int = 3,
        sidechain_fine_grid: float = 1.0,
        sidechain_coarse_grid: float = 30.0,
        enable_sidechain_opt: bool = True,
        sp3_angle_flexibility: bool = False,
        sp3_angle_tolerance: float = 3.0,
        keep_all_outputs: bool = False,
        save_csv: bool = False,
        winner_threshold: float = 0.1,
        apply_clash_penalty: bool = False,
        clash_cutoff: float = 2.0,
        clash_cutoff_ref: float = 3.5,
        clash_weight: float = 1.0,
        anchor_tol_close: float = 0.05,
        anchor_tol_loose: float = 0.05,
        anchor_w_close: float = 15.0,
        anchor_w_loose: float = 3.0,
        rigid_dock_ligand_complex: bool = False,
        rigid_dock_max_translation: float = 0.5,
        rigid_dock_max_rotation: float = 5.0,
        rigid_dock_iterations: int = 1,
        n_workers: int = 1,
        verbose: bool = True
    ):
        self.n_workers = n_workers
        # Process that owns the temp file; forked parallel workers must NOT
        # unlink it on exit (the parent still owns its lifecycle).
        self._owner_pid = os.getpid()
        self.ref_pdb_path = ref_pdb_path
        self.mobile_pdb_path = mobile_pdb_path
        self.output_dir = output_dir
        self.catres_subset_indices = catres_subset
        self.ptm_specs = ptm_specs or []
        self.outlier_rmsd_threshold = outlier_rmsd_threshold
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.min_residues = min_residues
        self.sidechain_cycles = sidechain_cycles
        self.sidechain_fine_grid = sidechain_fine_grid
        self.sidechain_coarse_grid = sidechain_coarse_grid
        self.enable_sidechain_opt = enable_sidechain_opt
        self.apply_clash_penalty = apply_clash_penalty
        self.clash_cutoff = clash_cutoff
        self.clash_cutoff_ref = clash_cutoff_ref
        self.clash_weight = clash_weight
        self.anchor_tol_close = anchor_tol_close
        self.anchor_tol_loose = anchor_tol_loose
        self.anchor_w_close = anchor_w_close
        self.anchor_w_loose = anchor_w_loose
        self.rigid_dock_ligand_complex = rigid_dock_ligand_complex
        self.rigid_dock_max_translation = rigid_dock_max_translation
        self.rigid_dock_max_rotation = rigid_dock_max_rotation
        self.rigid_dock_iterations = rigid_dock_iterations
        self.ref_anchors: Dict[Tuple, List[float]] = {}
        self.sp3_angle_flexibility = sp3_angle_flexibility
        self.sp3_angle_tolerance = sp3_angle_tolerance
        self.keep_all_outputs = keep_all_outputs
        self.save_csv = save_csv
        self.winner_threshold = winner_threshold
        self.verbose = verbose

        # Store aligned structures and their paths
        self.aligned_structures: Dict[str, Structure.Structure] = {}
        self.output_paths: Dict[str, str] = {}

        self.mobile_basename = Path(mobile_pdb_path).stem
        os.makedirs(output_dir, exist_ok=True)

        # Parse REMARK 666 from reference
        self.all_remark_lines = extract_all_remark_lines(ref_pdb_path)
        _, self.all_catalytic_residues = parse_remark666_lines(ref_pdb_path)

        if not self.all_catalytic_residues:
            print("WARNING: No catalytic residues found in REMARK 666!")
        else:
            print(f"\nFound {len(self.all_catalytic_residues)} catalytic residues:")
            for cr in self.all_catalytic_residues:
                print(f"  #{cr.catres_index}: {cr.chain} {cr.resname} {cr.resnum}")

        # Get subset
        self.catres_subset = filter_catres_by_subset(self.all_catalytic_residues, catres_subset)

        if catres_subset:
            print(f"\nUsing catres_subset with {len(self.catres_subset)} residues")

        # Process prediction PDB - NCAA handling + strip ALL HETATM
        print(f"\nProcessing prediction PDB (NCAA + HETATM removal)...")
        self.mobile_processed = tempfile.NamedTemporaryFile(
            mode='w', suffix='.pdb', delete=False
        )
        self.mobile_processed.close()

        process_prediction_pdb(
            mobile_pdb_path,
            self.mobile_processed.name,
            self.ptm_specs,
            self.all_catalytic_residues,
            verbose=verbose
        )

        # Parse structures
        parser = PDBParser(QUIET=True)
        self.ref_structure = parser.get_structure('ref', ref_pdb_path)
        self.mobile_structure_original = parser.get_structure('mobile', self.mobile_processed.name)

        # HETATM from reference
        self.ref_hetatm_lines = extract_hetatm_lines(ref_pdb_path)
        print(f"\nFound {len(self.ref_hetatm_lines)} HETATM lines in reference")

        # Build ref-anchor dict (if --apply_clash_penalty)
        if self.apply_clash_penalty:
            self.ref_anchors = build_ref_anchor_dict(
                self.ref_structure, self.all_catalytic_residues,
                self.clash_cutoff_ref, verbose=True,
            )

        # HIS tautomer map
        self.his_tautomer_map = build_his_tautomer_map_from_raw_pdb(ref_pdb_path)

        self.metrics_list: List[AlignmentMetrics] = []

    def __del__(self):
        """Cleanup temp files (only in the process that created them)."""
        if getattr(self, '_owner_pid', None) != os.getpid():
            return
        if hasattr(self, 'mobile_processed') and os.path.exists(self.mobile_processed.name):
            os.unlink(self.mobile_processed.name)

    def run_all_strategies(self) -> pd.DataFrame:
        """Run all alignment strategies."""
        # RMSD-based strategies (participate in winner selection)
        strategies = [
            ('all_backbone_rmsd', 'backbone', 'all'),
            ('ca_rmsd', 'ca', 'all'),
            ('catres_backbone_rmsd', 'backbone', 'catres'),
            ('catres_ca_rmsd', 'ca', 'catres'),
        ]

        # FIX (strategy redundancy): only run catres_subset_* if user supplied
        # an actual subset via --catres_subset. Otherwise self.catres_subset ==
        # self.all_catalytic_residues, making catres_subset_* rows byte-identical
        # to catres_* rows — pure wasted CPU.
        if self.catres_subset_indices is not None:
            strategies.append(('catres_subset_backbone_rmsd', 'backbone', 'catres_subset'))
            strategies.append(('catres_subset_ca_rmsd', 'ca', 'catres_subset'))
        else:
            print("\nINFO: --catres_subset not supplied; skipping catres_subset_* strategies "
                  "(would be duplicates of catres_* strategies)")

        # TM-align strategy (global only - catres-specific TM-align doesn't work well)
        if HAS_BIOTITE:
            strategies.append(('global_TMalign', 'tmalign', 'all'))
        else:
            print("\nWARNING: Biotite not available, skipping TM-align strategies")

        # Resolve worker count. n_workers == 1 -> serial path (byte-identical to
        # the original). n_workers == 0 -> auto-detect available CPUs (honors
        # SLURM allocation: 1 CPU on SLURM -> serial). Never exceed #strategies.
        requested = self.n_workers if (self.n_workers and self.n_workers > 0) else detect_cpu_count()
        n_workers = max(1, min(requested, len(strategies)))

        if n_workers > 1:
            print(f"\nDetected {detect_cpu_count()} CPU(s); running {len(strategies)} independent "
                  f"strategies across {n_workers} parallel workers "
                  f"(hydrogen-add saves run serially in the parent).")
            self._run_strategies_parallel(strategies, n_workers)
        else:
            self._run_strategies_serial(strategies)

        if not self.metrics_list:
            # No successful strategies - still clean up any partial outputs
            if not self.keep_all_outputs and self.output_paths:
                print("\nNo successful strategies. Cleaning up partial outputs...")
                self._cleanup_strategy_files(exclude_path=None)
            return pd.DataFrame()

        df = self.create_metrics_dataframe()
        winner_strategy = ""

        try:
            # Select winner
            winner_strategy = select_winner(df, self.winner_threshold, self.verbose)

            if winner_strategy:
                self.finalize_outputs(winner_strategy)
            else:
                # No winner selected - clean up all files unless keep_all
                print("\nWARNING: No winner could be selected")
                if not self.keep_all_outputs:
                    self._cleanup_strategy_files(exclude_path=None)

        except Exception as e:
            print(f"\nERROR during output finalization: {e}")
            import traceback
            traceback.print_exc()
            # Attempt cleanup even on error
            if not self.keep_all_outputs:
                print("Attempting cleanup after error...")
                try:
                    self._cleanup_strategy_files(exclude_path=None)
                except Exception as cleanup_error:
                    print(f"Cleanup also failed: {cleanup_error}")

        # Save CSV only if requested
        if self.save_csv:
            csv_path = os.path.join(self.output_dir, f"{self.mobile_basename}_alignment_metrics.csv")
            df.to_csv(csv_path, index=False)
            print(f"\nMetrics saved to: {csv_path}")

        self.print_summary(df, winner_strategy)
        return df

    def _run_strategies_serial(self, strategies) -> None:
        """Original serial path: run + save each strategy inline, in order.
        Behavior is byte-identical to pre-parallelization code."""
        for strategy_name, atom_selector, residue_set in strategies:
            print(f"\n{'='*80}")
            print(f"STRATEGY: {strategy_name}")
            print(f"{'='*80}")

            try:
                metrics = self.run_single_strategy(strategy_name, atom_selector, residue_set)
                if metrics:
                    self.metrics_list.append(metrics)

                    # Store the aligned structure
                    self.aligned_structures[strategy_name] = copy.deepcopy(self.mobile_structure_aligned)

                    # Save to temporary location with strategy-specific name
                    output_pdb = os.path.join(
                        self.output_dir,
                        f"{self.mobile_basename}_aligned_{strategy_name}.pdb"
                    )
                    self.output_paths[strategy_name] = output_pdb
                    self.save_aligned_structure(
                        self.mobile_structure_aligned,
                        output_pdb,
                        strategy_name
                    )
            except Exception as e:
                print(f"ERROR in {strategy_name}: {e}")
                import traceback
                traceback.print_exc()

    def _run_strategies_parallel(self, strategies, n_workers) -> None:
        """Parallel path: run the independent strategies in forked workers, then
        save serially in the parent (in strategy order, so metrics_list ordering
        and all outputs match the serial path exactly). PyRosetta is only ever
        touched in the parent."""
        import multiprocessing as mp

        global _PARALLEL_ALIGNER
        _PARALLEL_ALIGNER = self
        try:
            ctx = mp.get_context('fork')
            with ctx.Pool(processes=n_workers) as pool:
                results = pool.map(_run_strategy_worker, strategies)
        finally:
            _PARALLEL_ALIGNER = None

        for (strategy_name, _atom_selector, _residue_set), (metrics, structure) in zip(strategies, results):
            if metrics and structure is not None:
                print(f"\n{'='*80}")
                print(f"SAVING STRATEGY: {strategy_name}")
                print(f"{'='*80}")
                self.metrics_list.append(metrics)
                self.aligned_structures[strategy_name] = structure
                output_pdb = os.path.join(
                    self.output_dir,
                    f"{self.mobile_basename}_aligned_{strategy_name}.pdb"
                )
                self.output_paths[strategy_name] = output_pdb
                self.save_aligned_structure(structure, output_pdb, strategy_name)

    def finalize_outputs(self, winner_strategy: str) -> None:
        """
        Finalize outputs: rename winner to _aligned.pdb, delete others unless keep_all.
        """
        winner_path = self.output_paths.get(winner_strategy)
        if not winner_path or not os.path.exists(winner_path):
            print(f"WARNING: Winner path not found: {winner_path}")
            # Still try to clean up other files
            if not self.keep_all_outputs:
                self._cleanup_strategy_files(exclude_path=None)
            return

        # Final path for winner
        final_winner_path = os.path.join(
            self.output_dir,
            f"{self.mobile_basename}_aligned.pdb"
        )

        try:
            # If final path already exists, remove it
            if os.path.exists(final_winner_path) and final_winner_path != winner_path:
                os.remove(final_winner_path)

            # Move winner to final location (more atomic than copy+delete)
            shutil.move(winner_path, final_winner_path)
            print(f"\nWinner ({winner_strategy}) saved as: {final_winner_path}")

            # Update output_paths so cleanup doesn't try to delete the moved file
            self.output_paths[winner_strategy] = final_winner_path

        except Exception as e:
            print(f"ERROR: Failed to move winner file: {e}")
            # Try copy as fallback
            try:
                shutil.copy2(winner_path, final_winner_path)
                print(f"\nWinner ({winner_strategy}) copied to: {final_winner_path}")
            except Exception as e2:
                print(f"ERROR: Failed to copy winner file: {e2}")
                return

        # Clean up strategy-specific files (unless keep_all)
        if not self.keep_all_outputs:
            self._cleanup_strategy_files(exclude_path=final_winner_path)

    def _cleanup_strategy_files(self, exclude_path: Optional[str] = None) -> None:
        """Remove all strategy-specific output files except the excluded path."""
        removed_count = 0
        failed_count = 0

        for strategy_name, output_path in self.output_paths.items():
            # Skip if it's the excluded path (winner)
            if exclude_path and os.path.abspath(output_path) == os.path.abspath(exclude_path):
                continue

            if not os.path.exists(output_path):
                continue

            try:
                os.remove(output_path)
                removed_count += 1
                if self.verbose:
                    print(f"  Removed: {output_path}")
            except Exception as e:
                failed_count += 1
                print(f"  WARNING: Could not remove {output_path}: {e}")

        if failed_count > 0:
            print(f"  WARNING: Failed to remove {failed_count} files")
        elif removed_count > 0 and self.verbose:
            print(f"  Cleaned up {removed_count} strategy files")

    def run_single_strategy(
        self,
        strategy_name: str,
        atom_selector: str,
        residue_set: str
    ) -> Optional[AlignmentMetrics]:
        """Run a single alignment strategy."""
        mobile_structure = copy.deepcopy(self.mobile_structure_original)

        # Determine residues for alignment
        if residue_set == 'catres_subset':
            ref_residues = get_catalytic_residues_from_structure(self.ref_structure, self.catres_subset)
            mobile_residues = get_catalytic_residues_from_structure(mobile_structure, self.catres_subset)
            catres_for_opt = self.catres_subset
            tmalign_residue_subset = self.catres_subset
        elif residue_set == 'catres':
            ref_residues = get_catalytic_residues_from_structure(self.ref_structure, self.all_catalytic_residues)
            mobile_residues = get_catalytic_residues_from_structure(mobile_structure, self.all_catalytic_residues)
            catres_for_opt = self.all_catalytic_residues
            tmalign_residue_subset = self.all_catalytic_residues
        else:
            matched = match_residues_by_chain_and_number(self.ref_structure, mobile_structure)
            ref_residues = [r1 for r1, r2 in matched]
            mobile_residues = [r2 for r1, r2 in matched]
            catres_for_opt = self.catres_subset if self.catres_subset else self.all_catalytic_residues
            tmalign_residue_subset = None  # Use all residues for global TM-align

        if not ref_residues or not mobile_residues or len(ref_residues) != len(mobile_residues):
            print(f"ERROR: Invalid residues for {strategy_name}")
            return None

        print(f"\nAligning {len(ref_residues)} residues with {atom_selector} method")

        # Variables for alignment results
        converged_rmsd = 0.0
        n_iterations = 0
        kept_indices = []
        tm_score = float('nan')  # FIX (Bug C): NaN default for non-TM-align strategies; overwritten by global_TMalign branch below if applicable

        # Alignment - either Kabsch (iterative) or TM-align
        if atom_selector == 'tmalign':
            # TM-align superposition
            tm_score, n_aligned = tmalign_superimpose(
                self.ref_structure,
                mobile_structure,
                residue_subset=tmalign_residue_subset,
                verbose=self.verbose
            )
            if n_aligned == 0:
                print(f"ERROR: TM-align failed")
                return None

            converged_rmsd = 0.0  # TM-align doesn't use RMSD convergence
            n_iterations = 1
            kept_indices = list(range(len(ref_residues)))
            print(f"\nTM-align: TM-score={tm_score:.4f}, aligned {n_aligned} CA atoms")
        else:
            # Standard Kabsch alignment with iterative outlier removal
            aligner = IterativeAligner(
                self.ref_structure,
                mobile_structure,
                atom_selector=atom_selector,
                rmsd_threshold=self.outlier_rmsd_threshold,
                max_iterations=self.max_iterations,
                convergence_threshold=self.convergence_threshold,
                min_atoms=self.min_residues,
                verbose=self.verbose
            )

            converged_rmsd, n_iterations, kept_indices = aligner.align_with_outlier_removal(
                ref_residues, mobile_residues
            )

            if np.isinf(converged_rmsd):
                print(f"ERROR: Alignment failed")
                return None

            print(f"\nConverged: {converged_rmsd:.4f}Å after {n_iterations} iters ({len(kept_indices)}/{len(ref_residues)} res)")

        self.mobile_structure_aligned = mobile_structure

        # Calculate standard metrics (without symmetry handling yet)
        all_matched = match_residues_by_chain_and_number(self.ref_structure, mobile_structure)
        ref_all = [r1 for r1, r2 in all_matched]
        mob_all = [r2 for r1, r2 in all_matched]

        all_backbone_rmsd, _, _ = calculate_rmsd_with_residues(ref_all, mob_all, 'backbone')
        ca_rmsd, _, _ = calculate_rmsd_with_residues(ref_all, mob_all, 'ca')

        ref_catres = get_catalytic_residues_from_structure(self.ref_structure, self.all_catalytic_residues)
        mob_catres = get_catalytic_residues_from_structure(mobile_structure, self.all_catalytic_residues)

        catres_backbone_rmsd, _, _ = calculate_rmsd_with_residues(ref_catres, mob_catres, 'backbone')
        catres_ca_rmsd, _, _ = calculate_rmsd_with_residues(ref_catres, mob_catres, 'ca')

        opt_catres = self.catres_subset if self.catres_subset else self.all_catalytic_residues
        ref_catres_subset = get_catalytic_residues_from_structure(self.ref_structure, opt_catres)
        mob_catres_subset = get_catalytic_residues_from_structure(mobile_structure, opt_catres)

        catres_subset_backbone_rmsd, _, _ = calculate_rmsd_with_residues(ref_catres_subset, mob_catres_subset, 'backbone')
        catres_subset_ca_rmsd, _, _ = calculate_rmsd_with_residues(ref_catres_subset, mob_catres_subset, 'ca')

        print(f"\nMetrics after alignment:")
        print(f"  All backbone RMSD: {all_backbone_rmsd:.4f} Å")
        print(f"  CA RMSD: {ca_rmsd:.4f} Å")
        print(f"  Catres backbone RMSD: {catres_backbone_rmsd:.4f} Å")
        print(f"  Catres CA RMSD: {catres_ca_rmsd:.4f} Å")
        if atom_selector == 'tmalign':
            print(f"  TM-score: {tm_score:.4f}")

        # Inject REF HETATM into mobile so rigid dock + anchor eval can see/move them.
        # This is post-alignment so mobile + ligand are in the same frame.
        if self.apply_clash_penalty or self.rigid_dock_ligand_complex:
            inject_ref_ligand_into_mobile(
                mobile_structure, self.ref_structure,
                self.all_catalytic_residues, verbose=self.verbose,
            )

        # Sidechain optimization (optional)
        if self.enable_sidechain_opt:
            optimizer = SidechainDihedralOptimizer(
                self.ref_structure,
                mobile_structure,
                catres_for_opt,
                verbose=self.verbose,
                sp3_angle_flexibility=self.sp3_angle_flexibility,
                sp3_angle_tolerance=self.sp3_angle_tolerance,
                apply_clash_penalty=self.apply_clash_penalty,
                clash_cutoff_mob=self.clash_cutoff,
                clash_weight=self.clash_weight,
                ref_anchors=self.ref_anchors,
                anchor_tol_close=self.anchor_tol_close,
                anchor_tol_loose=self.anchor_tol_loose,
                anchor_w_close=self.anchor_w_close,
                anchor_w_loose=self.anchor_w_loose,
            )

            rmsd_before_opt, rmsd_after_opt, n_opt_iter = optimizer.optimize_all_residues(
                n_cycles=self.sidechain_cycles,
                fine_grid_degrees=self.sidechain_fine_grid,
                coarse_grid_degrees=self.sidechain_coarse_grid,
            )
            improvement = rmsd_before_opt - rmsd_after_opt
        else:
            # Just calculate RMSD without optimization
            rmsd_before_opt, _, _ = calculate_rmsd_with_residues(
                ref_catres_subset, mob_catres_subset, 'all_heavy'
            )
            rmsd_after_opt = rmsd_before_opt
            n_opt_iter = 0
            improvement = 0.0
            optimizer = None  # may be needed later for dock+sidechain alternation
            print(f"\n  Sidechain optimization DISABLED")
            print(f"  Catres all-atom RMSD: {rmsd_before_opt:.4f} Å")

        # Optional: rigid-body dock the ligand complex to satisfy ref anchors,
        # then (if sidechain opt is enabled) re-optimize sidechains. Iterate.
        # Runs INDEPENDENTLY of sidechain opt — supports "dock-only" mode.
        if self.rigid_dock_ligand_complex and self.apply_clash_penalty:
            for dock_iter in range(self.rigid_dock_iterations):
                if self.verbose:
                    print(f"\n  --- Rigid ligand dock pass {dock_iter+1}/{self.rigid_dock_iterations} ---")
                rigid_dock_ligand_complex(
                    mobile_structure, self.all_catalytic_residues, self.ref_anchors,
                    clash_cutoff_mob=self.clash_cutoff,
                    anchor_tol_close=self.anchor_tol_close,
                    anchor_tol_loose=self.anchor_tol_loose,
                    anchor_w_close=self.anchor_w_close,
                    anchor_w_loose=self.anchor_w_loose,
                    clash_weight=self.clash_weight,
                    max_trans=self.rigid_dock_max_translation,
                    max_rot_deg=self.rigid_dock_max_rotation,
                    verbose=self.verbose,
                )
                # Re-optimize sidechains for ligand's new pose (skip on last iter,
                # and skip entirely if sidechain opt is disabled)
                if optimizer is not None and dock_iter < self.rigid_dock_iterations - 1:
                    _, rmsd_after_opt, n_opt2 = optimizer.optimize_all_residues(
                        n_cycles=1,
                        fine_grid_degrees=self.sidechain_fine_grid,
                        coarse_grid_degrees=self.sidechain_coarse_grid,
                    )
                    n_opt_iter += n_opt2
            # ALSO run a final sidechain pass after the LAST dock (codex point #8)
            if optimizer is not None:
                _, rmsd_after_opt, n_opt2 = optimizer.optimize_all_residues(
                    n_cycles=1,
                    fine_grid_degrees=self.sidechain_fine_grid,
                    coarse_grid_degrees=self.sidechain_coarse_grid,
                )
                n_opt_iter += n_opt2
                improvement = rmsd_before_opt - rmsd_after_opt

        # ============================================================
        # SYMMETRIC ATOM RESOLUTION (happens ONCE, after optimization)
        # ============================================================
        if self.verbose:
            print(f"\n  Resolving symmetric atoms...")

        swap_map = resolve_symmetric_atoms(
            self.ref_structure,
            mobile_structure,
            opt_catres,
            verbose=self.verbose
        )

        # Recalculate catres_subset all-atom RMSD with symmetry handling
        rmsd_after_opt_symmetric, _, _ = calculate_rmsd_with_symmetry(
            ref_catres_subset,
            mob_catres_subset,
            opt_catres,
            swap_map,
            atom_selector='all_heavy'
        )

        if self.verbose:
            print(f"  Catres all-atom RMSD (with symmetry): {rmsd_after_opt_symmetric:.4f} Å")
            if abs(rmsd_after_opt_symmetric - rmsd_after_opt) > 0.001:
                print(f"    (was {rmsd_after_opt:.4f} Å without symmetry handling)")

        # Use symmetric RMSD as the final metric
        rmsd_after_opt = rmsd_after_opt_symmetric

        # ============================================================
        # lDDT CALCULATION (uses resolved symmetric atom mapping)
        # ============================================================
        if self.verbose:
            print(f"\n  Calculating lDDT for catres_subset...")

        catres_subset_lddt = calculate_catres_subset_lddt(
            self.ref_structure,
            mobile_structure,
            opt_catres,
            swap_map,
            verbose=self.verbose
        )

        if self.verbose:
            print(f"  Catres subset internal lDDT: {catres_subset_lddt:.4f}")

        # Clash-penalty diagnostics (only when --apply_clash_penalty)
        n_clash_init = -1
        n_clash_final = -1
        n_overshot_final = -1
        n_loosened_final = -1
        soft_clash_final = float('nan')
        soft_anchor_final = float('nan')
        if self.apply_clash_penalty:
            (n_clash_final, n_overshot_final, n_loosened_final,
             soft_clash_final, soft_anchor_final) = evaluate_clash_penalty(
                mobile_structure, self.all_catalytic_residues,
                self.ref_anchors, self.clash_cutoff,
                anchor_tol_close=self.anchor_tol_close,
                anchor_tol_loose=self.anchor_tol_loose,
                anchor_w_close=self.anchor_w_close,
                anchor_w_loose=self.anchor_w_loose,
            )
            if self.verbose:
                print(f"  Clashes (after opt): {n_clash_final} unexpected (penalty {soft_clash_final:.4f} Å²); "
                      f"anchors: {n_overshot_final} overshot + {n_loosened_final} loosened "
                      f"(penalty {soft_anchor_final:.4f} Å²-cost)")

        return AlignmentMetrics(
            strategy_name=strategy_name,
            converged_rmsd=converged_rmsd,
            n_iterations=n_iterations,
            n_atoms_final=len(kept_indices),
            all_backbone_rmsd=all_backbone_rmsd,
            ca_rmsd=ca_rmsd,
            catres_backbone_rmsd=catres_backbone_rmsd,
            catres_ca_rmsd=catres_ca_rmsd,
            catres_subset_backbone_rmsd=catres_subset_backbone_rmsd,
            catres_subset_ca_rmsd=catres_subset_ca_rmsd,
            catres_subset_all_atom_rmsd_before_opt=rmsd_before_opt,
            catres_subset_all_atom_rmsd_after_opt=rmsd_after_opt,
            n_sidechain_opt_iterations=n_opt_iter,
            sidechain_opt_improvement=improvement,
            catres_subset_internal_lddt=catres_subset_lddt,
            tm_score=tm_score,
            n_unexpected_clashes_initial=n_clash_init,
            n_unexpected_clashes_final=n_clash_final,
            soft_clash_penalty_final=soft_clash_final,
            n_overshot_anchors_final=n_overshot_final,
            n_loosened_anchors_final=n_loosened_final,
            soft_anchor_penalty_final=soft_anchor_final,
        )

    def save_aligned_structure(
        self,
        structure: Structure.Structure,
        output_path: str,
        strategy_name: str
    ) -> None:
        """Save aligned structure with HETATM from reference."""
        io = PDBIO()
        io.set_structure(structure)

        temp_pdb = tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False)
        io.save(temp_pdb.name)
        temp_pdb.close()

        with open(temp_pdb.name, 'r') as f:
            content = f.readlines()

        # When ligand was injected into mobile (clash-penalty / rigid-dock pipelines),
        # the structure's HETATM lines reflect any rigid-dock movement and should be
        # used directly. Otherwise (legacy path) fall back to static ref_hetatm_lines.
        ligand_in_structure = self.apply_clash_penalty or self.rigid_dock_ligand_complex
        if ligand_in_structure:
            protein_lines, hetatm_lines_from_struct = separate_protein_and_hetatm(content)
        else:
            protein_lines, _ = separate_protein_and_hetatm(content)
            hetatm_lines_from_struct = None

        # Add hydrogens with PyRosetta
        if HAS_PYROSETTA:
            try:
                if self.verbose:
                    print(f"  Adding hydrogens with PyRosetta...")

                pose = pyrosetta.pose_from_file(temp_pdb.name)

                # Fix HIS tautomers
                for model in structure:
                    for chain in model:
                        for residue in chain:
                            if residue.resname == 'HIS':
                                key = (chain.id, residue.id[1])
                                if key in self.his_tautomer_map:
                                    his_type = self.his_tautomer_map[key]
                                    seqpos = pose.pdb_info().pdb2pose(chain.id, residue.id[1])
                                    if seqpos > 0:
                                        mutres = pyrosetta.rosetta.protocols.simple_moves.MutateResidue()
                                        mutres.set_res_name(his_type)
                                        mutres.set_target(seqpos)
                                        mutres.set_preserve_atom_coords(True)
                                        mutres.apply(pose)

                pdb_string = pyrosetta.distributed.io.to_pdbstring(pose)
                protein_with_h_lines = [line + '\n' for line in pdb_string.split('\n')
                                       if line.startswith(('ATOM', 'TER'))]
            except Exception as e:
                print(f"  WARNING: Could not add hydrogens: {e}")
                protein_with_h_lines = protein_lines
        else:
            protein_with_h_lines = protein_lines

        # Build final PDB
        final_lines = []
        final_lines.extend(self.all_remark_lines)
        final_lines.append(f"REMARK   Aligned using strategy: {strategy_name}\n")
        final_lines.extend(protein_with_h_lines)
        # Append ligand HETATM. Prefer the (possibly docked) coords from structure,
        # fall back to static REF lines if clash/dock pipelines weren't run.
        if hetatm_lines_from_struct is not None:
            final_lines.extend(hetatm_lines_from_struct)
        else:
            final_lines.extend(self.ref_hetatm_lines)
        # Strip any CONECT records from the final output PDB
        final_lines = [ln for ln in final_lines if not ln.startswith("CONECT")]
        final_lines = renumber_pdb_atoms(final_lines, start_number=1)
        if not any(line.startswith('END') for line in final_lines):
            final_lines.append('END\n')

        with open(output_path, 'w') as f:
            f.writelines(final_lines)

        os.unlink(temp_pdb.name)
        print(f"\nSaved: {output_path}")

    def create_metrics_dataframe(self) -> pd.DataFrame:
        """Convert metrics to DataFrame."""
        data = []
        for m in self.metrics_list:
            data.append({
                'strategy': m.strategy_name,
                'converged_rmsd': m.converged_rmsd,
                'n_iterations': m.n_iterations,
                'n_atoms_final': m.n_atoms_final,
                'all_backbone_rmsd': m.all_backbone_rmsd,
                'ca_rmsd': m.ca_rmsd,
                'catres_backbone_rmsd': m.catres_backbone_rmsd,
                'catres_ca_rmsd': m.catres_ca_rmsd,
                'catres_subset_backbone_rmsd': m.catres_subset_backbone_rmsd,
                'catres_subset_ca_rmsd': m.catres_subset_ca_rmsd,
                'catres_subset_all_atom_rmsd_before_opt': m.catres_subset_all_atom_rmsd_before_opt,
                'catres_subset_all_atom_rmsd_after_opt': m.catres_subset_all_atom_rmsd_after_opt,
                'n_sidechain_opt_iterations': m.n_sidechain_opt_iterations,
                'sidechain_opt_improvement': m.sidechain_opt_improvement,
                'catres_subset_internal_lddt': m.catres_subset_internal_lddt,
                'tm_score': m.tm_score,
                'n_unexpected_clashes_final': m.n_unexpected_clashes_final,
                'soft_clash_penalty_final': m.soft_clash_penalty_final,
                'n_overshot_anchors_final': m.n_overshot_anchors_final,
                'n_loosened_anchors_final': m.n_loosened_anchors_final,
                'soft_anchor_penalty_final': m.soft_anchor_penalty_final,
            })
        return pd.DataFrame(data)

    def print_summary(self, df: pd.DataFrame, winner_strategy: str = "") -> None:
        """Print summary with winner information."""
        print(f"\n{'='*80}")
        print("SUMMARY OF ALL STRATEGIES")
        print(f"{'='*80}\n")

        best_catres_idx = df['catres_subset_all_atom_rmsd_after_opt'].idxmin()
        best_catres_strategy = df.loc[best_catres_idx, 'strategy']
        best_catres_rmsd = df.loc[best_catres_idx, 'catres_subset_all_atom_rmsd_after_opt']

        best_ca_idx = df['ca_rmsd'].idxmin()
        best_ca_strategy = df.loc[best_ca_idx, 'strategy']
        best_ca_rmsd = df.loc[best_ca_idx, 'ca_rmsd']

        # Find best lDDT
        best_lddt_idx = df['catres_subset_internal_lddt'].idxmax()
        best_lddt_strategy = df.loc[best_lddt_idx, 'strategy']
        best_lddt = df.loc[best_lddt_idx, 'catres_subset_internal_lddt']

        print("BEST STRATEGIES BY METRIC:")
        print(f"  Best catres all-atom RMSD: {best_catres_strategy} ({best_catres_rmsd:.4f} Å)")
        print(f"  Best CA RMSD: {best_ca_strategy} ({best_ca_rmsd:.4f} Å)")
        print(f"  Best catres lDDT: {best_lddt_strategy} ({best_lddt:.4f})")

        # Show TM-align strategy details if present
        tmalign_rows = df[df['strategy'].str.contains('TMalign', case=False)]
        if not tmalign_rows.empty:
            print("\nTM-ALIGN STRATEGY:")
            for _, row in tmalign_rows.iterrows():
                print(f"  {row['strategy']}: TM-score={row['tm_score']:.4f}, "
                      f"catres_RMSD={row['catres_subset_all_atom_rmsd_after_opt']:.4f} Å, "
                      f"lDDT={row['catres_subset_internal_lddt']:.4f}")

        print("\nFULL METRICS TABLE:")
        print(df.to_string(index=False))

        if winner_strategy:
            winner_row = df[df['strategy'] == winner_strategy].iloc[0]
            print(f"\n{'='*80}")
            print(f"SELECTED WINNER: {winner_strategy}")
            print(f"{'='*80}")
            print(f"  Catres all-atom RMSD (optimized): {winner_row['catres_subset_all_atom_rmsd_after_opt']:.4f} Å")
            print(f"  Catres CA RMSD: {winner_row['catres_subset_ca_rmsd']:.4f} Å")
            print(f"  All backbone RMSD: {winner_row['all_backbone_rmsd']:.4f} Å")
            print(f"  CA RMSD: {winner_row['ca_rmsd']:.4f} Å")
            print(f"  Catres subset lDDT: {winner_row['catres_subset_internal_lddt']:.4f}")
            print(f"  Sidechain improvement: {winner_row['sidechain_opt_improvement']:.4f} Å")
            print(f"{'='*80}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Advanced PDB alignment with NCAA handling and sidechain optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic usage:
    python %(prog)s --ref_pdb ref.pdb --pdb_for_alignment pred.pdb --output_dir ./out

  With NCAA handling:
    python %(prog)s --ref_pdb ref.pdb --pdb_for_alignment pred.pdb --output_dir ./out \\
        --ptm_from_remark666 "A/LYS/6:KCX"

  Full options:
    python %(prog)s --ref_pdb ref.pdb --pdb_for_alignment pred.pdb --output_dir ./out \\
        --catres_subset "1,5,6,11" --ptm_from_remark666 "A/LYS/6:KCX" \\
        --sp3_angle_flexibility --keep_all --save_csv --verbose
"""
    )

    # Required arguments
    parser.add_argument('--ref_pdb', required=True, help='Reference PDB file path')
    parser.add_argument('--pdb_for_alignment', required=True, help='PDB file to align (e.g., AF3 prediction)')
    parser.add_argument('--output_dir', required=True, help='Output directory for aligned structures')

    # Catalytic residue arguments
    parser.add_argument('--catres_subset', type=str, default=None,
                        help='Comma-separated catres indices to use for optimization (e.g., "1,5,11")')
    parser.add_argument('--ptm_from_remark666', nargs='+', default=None,
                        help='PTM specifications: "CHAIN/RESNAME/CATRES_IDX:NCAA[-ATOMS]" (e.g., "A/LYS/6:KCX")')

    # Ligand params (Rosetta residue types for non-standard HETATM like the ligand)
    parser.add_argument('--ligand_params', nargs='+', default=None,
                        help='One or more Rosetta .params files for non-standard residues present '
                             'in the structure (e.g. the ligand PBJ.params). Passed to '
                             'pyrosetta.init via -extra_res_fa so the PyRosetta hydrogen-addition / '
                             'HIS-tautomer step can build the pose. WITHOUT this, Rosetta cannot '
                             'recognize the ligand, the protonation step is silently skipped, and '
                             'the output protein has NO hydrogens. Default: none (legacy behavior).')

    # Alignment thresholds
    parser.add_argument('--outlier_threshold', type=float, default=0.5,
                        help='RMSD threshold for outlier removal (default: 0.5 Å)')
    parser.add_argument('--convergence_threshold', type=float, default=0.001,
                        help='Convergence threshold for iterative alignment (default: 0.001 Å)')
    parser.add_argument('--max_iterations', type=int, default=50,
                        help='Maximum alignment iterations (default: 50)')
    parser.add_argument('--min_residues', type=int, default=10,
                        help='Minimum residues to keep during outlier removal (default: 10)')

    # Sidechain optimization
    parser.add_argument('--no_sidechain_opt', action='store_true',
                        help='Disable sidechain optimization entirely')
    parser.add_argument('--sidechain_cycles', type=int, default=3,
                        help='Number of global sidechain optimization cycles (default: 3)')
    parser.add_argument('--sidechain_fine_grid', type=float, default=1.0,
                        help='Fine grid base step in degrees (default: 1.0). Phase 4 refinement uses '
                             '[×10, ×5, ×2, ×1] of this value.')
    parser.add_argument('--sidechain_coarse_grid', type=float, default=30.0,
                        help='Coarse global-search step in degrees for Phase 3 (default: 30.0).')
    parser.add_argument('--sp3_angle_flexibility', action='store_true',
                        help='Enable SP3 C-C bond angle flexibility (±3° from 109.5°)')
    parser.add_argument('--sp3_angle_tolerance', type=float, default=3.0,
                        help='SP3 bond angle tolerance in degrees (default: 3.0)')

    # Clash-aware sidechain optimization
    parser.add_argument('--apply_clash_penalty', action='store_true',
                        help='Enable clash-aware sidechain optimization. The optimizer '
                             'minimizes (rmsd + clash_weight * soft_clash) where soft_clash = '
                             'Σ max(0, clash_cutoff - d)² over unexempt cross-residue heavy-atom '
                             'pairs. REF-confirmed close contacts (catres↔ligand and catres↔catres '
                             'within clash_cutoff_ref Å) are EXEMPTED — so KCX-NZ↔CX, Zn-coord HIS, '
                             'catalytic H-bonds are not penalised.')
    parser.add_argument('--clash_cutoff', type=float, default=2.0,
                        help='Hard-clash distance (Å) below which a mobile pair contributes to '
                             'soft_clash unless exempted by REF (default: 2.0).')
    parser.add_argument('--clash_cutoff_ref', type=float, default=3.5,
                        help='Distance (Å) below which a REF contact is treated as intentional '
                             'and added to the exempt set (default: 3.5; covers H-bonds, metal '
                             'coordination, PTM covalent bonds).')
    parser.add_argument('--clash_weight', type=float, default=1.0,
                        help='Weight on soft_clash term (non-exempt clashes) in cost (default: 1.0).')
    parser.add_argument('--ref_anchor_tolerance_close', type=float, default=0.05,
                        help='Allowed reduction (Å) of an exempt pair distance from REF before '
                             'penalty kicks in (default: 0.05; sweep-optimized 2026-05-13). '
                             'Catches "got too close to ligand".')
    parser.add_argument('--ref_anchor_tolerance_loose', type=float, default=0.05,
                        help='Allowed widening (Å) of an exempt pair distance from REF before '
                             'penalty kicks in (default: 0.05; sweep-optimized 2026-05-13). '
                             'Catches "lost the catalytic contact".')
    parser.add_argument('--ref_anchor_weight_close', type=float, default=15.0,
                        help='Weight on quadratic penalty for exempt-pair compression beyond '
                             'tolerance_close (default: 15; sweep-optimized 2026-05-13).')
    parser.add_argument('--ref_anchor_weight_loose', type=float, default=3.0,
                        help='Weight on quadratic penalty for exempt-pair widening beyond '
                             'tolerance_loose (default: 3; sweep-optimized 2026-05-13 — 5:1 close:loose '
                             'asymmetry preserves anchors in both directions while still penalising '
                             'close-overshoots more strongly).')

    # Rigid-body ligand-complex docking (subtle pose nudge)
    parser.add_argument('--rigid_dock_ligand_complex', action='store_true',
                        help='After sidechain opt, do a 6-DOF rigid-body optimization of the entire '
                             'ligand complex (substrate + metals + bound waters) to minimize the '
                             'anchor+clash cost. Iterates with sidechain opt. Requires '
                             '--apply_clash_penalty.')
    parser.add_argument('--rigid_dock_max_translation', type=float, default=0.5,
                        help='Max ligand-COM translation per axis in Å (default: 0.5).')
    parser.add_argument('--rigid_dock_max_rotation', type=float, default=5.0,
                        help='Max ligand rotation per axis in degrees (default: 5.0).')
    parser.add_argument('--rigid_dock_iterations', type=int, default=1,
                        help='Number of alternating dock + sidechain-opt cycles (default: 1; '
                             'sweep showed single pass converges, more iterations don\'t help).')

    # Reproducibility
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed for random + numpy.random (default: non-deterministic).')

    # Parallelism / resources
    parser.add_argument('--n_workers', type=int, default=1,
                        help='Number of parallel worker processes for the independent alignment '
                             'strategies (default: 1 = serial, identical to legacy behavior). '
                             'Use 0 to auto-detect available CPUs (honors SLURM_CPUS_PER_TASK, so '
                             'a 1-CPU SLURM task stays serial). Capped at the number of strategies. '
                             'Results are identical regardless of worker count; this only helps '
                             'wall-clock on multi-CPU nodes. Note: if you already batch many PDBs '
                             'in parallel (e.g. one core per design), keep this at 1 to avoid '
                             'oversubscription.')

    # Winner selection
    parser.add_argument('--winner_threshold', type=float, default=0.1,
                        help='Threshold for tiebreaker in winner selection (default: 0.1 Å)')

    # Output options
    parser.add_argument('--keep_all', action='store_true',
                        help='Keep all strategy output files (default: only keep winner)')
    parser.add_argument('--save_csv', action='store_true',
                        help='Save metrics to CSV file (default: no CSV output)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output with per-residue optimization details')

    args = parser.parse_args()

    # Apply --seed before any randomness
    if args.seed is not None:
        import random as _py_random
        _py_random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"Random seed set to {args.seed}")

    # Parse catres_subset
    catres_subset = None
    if args.catres_subset:
        catres_subset = [int(x.strip()) for x in args.catres_subset.split(',')]

    # Parse PTM specs
    ptm_specs = None
    if args.ptm_from_remark666:
        ptm_specs = parse_ptm_specs(args.ptm_from_remark666)
        print(f"\nParsed {len(ptm_specs)} PTM specifications:")
        for spec in ptm_specs:
            print(f"  #{spec.catres_index}: {spec.chain}/{spec.canonical_resname} -> {spec.ncaa_resname}")
            print(f"    Atoms to cut: {spec.atoms_to_cut}")

    # Initialize PyRosetta
    if HAS_PYROSETTA:
        print("Initializing PyRosetta...")
        init_flags = "-mute all -run:preserve_header"
        if args.ligand_params:
            missing = [p for p in args.ligand_params if not os.path.isfile(p)]
            if missing:
                print(f"ERROR: --ligand_params file(s) not found: {missing}")
                sys.exit(1)
            # Register custom residue types (e.g. the PBJ ligand) so pose_from_file
            # can build them and hydrogens / HIS tautomers get added. Without this,
            # the protonation step throws "Unrecognized residue" and is silently
            # skipped, producing un-protonated output.
            init_flags += " -extra_res_fa " + " ".join(os.path.abspath(p) for p in args.ligand_params)
            print(f"  Registering {len(args.ligand_params)} ligand params with Rosetta:")
            for p in args.ligand_params:
                print(f"    {p}")
        pyrosetta.init(init_flags)

    # Print configuration
    print(f"\n{'='*80}")
    print("MULTI-STRATEGY ALIGNMENT AND OPTIMIZATION")
    print(f"{'='*80}")
    print(f"\nReference: {args.ref_pdb}")
    print(f"Mobile: {args.pdb_for_alignment}")
    print(f"Output: {args.output_dir}")
    print(f"\nAlignment settings:")
    print(f"  Outlier threshold: {args.outlier_threshold} Å")
    print(f"  Convergence threshold: {args.convergence_threshold} Å")
    print(f"  Max iterations: {args.max_iterations}")
    print(f"  Min residues: {args.min_residues}")
    print(f"\nSidechain optimization:")
    if args.no_sidechain_opt:
        print(f"  DISABLED")
    else:
        print(f"  Global cycles: {args.sidechain_cycles}")
        print(f"  SP3 angle flexibility: {args.sp3_angle_flexibility}")
        if args.sp3_angle_flexibility:
            print(f"  SP3 tolerance: ±{args.sp3_angle_tolerance}°")
    print(f"\nOutput options:")
    print(f"  Keep all outputs: {args.keep_all}")
    print(f"  Save CSV: {args.save_csv}")
    print(f"  Winner threshold: {args.winner_threshold} Å")

    start_time = time.time()

    aligner = MultiStrategyAligner(
        ref_pdb_path=args.ref_pdb,
        mobile_pdb_path=args.pdb_for_alignment,
        output_dir=args.output_dir,
        catres_subset=catres_subset,
        ptm_specs=ptm_specs,
        outlier_rmsd_threshold=args.outlier_threshold,
        convergence_threshold=args.convergence_threshold,
        max_iterations=args.max_iterations,
        min_residues=args.min_residues,
        sidechain_cycles=args.sidechain_cycles,
        sidechain_fine_grid=args.sidechain_fine_grid,
        sidechain_coarse_grid=args.sidechain_coarse_grid,
        apply_clash_penalty=args.apply_clash_penalty,
        clash_cutoff=args.clash_cutoff,
        clash_cutoff_ref=args.clash_cutoff_ref,
        clash_weight=args.clash_weight,
        anchor_tol_close=args.ref_anchor_tolerance_close,
        anchor_tol_loose=args.ref_anchor_tolerance_loose,
        anchor_w_close=args.ref_anchor_weight_close,
        anchor_w_loose=args.ref_anchor_weight_loose,
        rigid_dock_ligand_complex=args.rigid_dock_ligand_complex,
        rigid_dock_max_translation=args.rigid_dock_max_translation,
        rigid_dock_max_rotation=args.rigid_dock_max_rotation,
        rigid_dock_iterations=args.rigid_dock_iterations,
        enable_sidechain_opt=not args.no_sidechain_opt,
        sp3_angle_flexibility=args.sp3_angle_flexibility,
        sp3_angle_tolerance=args.sp3_angle_tolerance,
        keep_all_outputs=args.keep_all,
        save_csv=args.save_csv,
        winner_threshold=args.winner_threshold,
        n_workers=args.n_workers,
        verbose=args.verbose
    )

    df = aligner.run_all_strategies()

    print(f"\n\nTotal time: {time.time() - start_time:.2f}s")
    print(f"Output: {args.output_dir}")


if __name__ == '__main__':
    main()
