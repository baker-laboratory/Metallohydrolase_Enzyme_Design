#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RFDiffusion3 Geometry Idealizer - Stage 1 of Predesign Pipeline

Takes RFDiffusion3 outputs (PDB + JSON with select_fixed_atoms) and performs
cartesian relaxation to idealize geometry while preserving fixed atoms in absolute space.

Key Features:
- Uses coordinate constraints to fix atoms in absolute space (from JSON)
- No CST file needed - minimal user input
- Fixes chain breaks, clashes, and poor geometry
- Preserves ligand and catalytic residue positions
- Uses modern beta_jan25 scorefunction

Workflow:
1. Parse JSON select_fixed_atoms
2. Apply coordinate constraints to fixed atoms + ligand
3. Define mobile region around ligand/catalytic residues
4. Cartesian idealization (optional: idealize SS → FastRelax → minimize)
5. Output cleaned PDB

Author: Created for woodbuse
Date: 2026-01-07
"""

import pyrosetta as pyr
import pyrosetta.rosetta
from pyrosetta.rosetta.core.select.residue_selector import (
    NeighborhoodResidueSelector,
    ResidueIndexSelector,
    OrResidueSelector,
    NotResidueSelector
)
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.protocols.minimization_packing import MinMover
from pyrosetta.rosetta.core.scoring import ScoreType
from pyrosetta.rosetta.core.scoring.constraints import CoordinateConstraint, AtomPairConstraint
from pyrosetta.rosetta.core.scoring.func import HarmonicFunc
from pyrosetta.rosetta.core.id import AtomID
from pyrosetta.rosetta.numeric import xyzVector_double_t
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
import time
from datetime import datetime


class RFDiffusion3GeometryIdealizer:
    """
    Geometry idealizer for RFDiffusion3 scaffolds using coordinate constraints.
    """

    def __init__(self, pdb_path, json_path=None, json_dir=None, params=None,
                 mobile_radius=10.0, cart_bonded_weight=0.5,
                 coord_cst_weight=100.0, coord_cst_stdev=0.01,
                 debug=False, no_metric_json_output=False):
        """
        Initialize the geometry idealizer.

        Args:
            pdb_path: Path to input PDB file
            json_path: Path to RFDiffusion3 JSON (if None, auto-detect)
            json_dir: Directory to look for JSON if not same as PDB
            params: List of ligand parameter files
            mobile_radius: Radius (Å) for mobile region around ligand/fixed residues
            cart_bonded_weight: Weight for cart_bonded score term (0.4-0.7)
            coord_cst_weight: Weight for coordinate constraints (100-1000 for near-zero movement)
            coord_cst_stdev: Standard deviation for coordinate constraint (0.01-0.1Å, lower=tighter)
            debug: If True, include full verbose metrics in JSON output
            no_metric_json_output: If True, skip JSON metrics file creation entirely
        """
        self.pdb_path = pdb_path
        self.pdbname = os.path.basename(pdb_path).replace(".pdb", "")
        self.pdb_dir = os.path.dirname(os.path.abspath(pdb_path))

        # Determine JSON path
        self.json_path = self._determine_json_path(json_path, json_dir)

        self.params = params or []
        self.mobile_radius = mobile_radius
        self.cart_bonded_weight = cart_bonded_weight
        self.coord_cst_weight = coord_cst_weight
        self.coord_cst_stdev = coord_cst_stdev
        self.debug = debug
        self.no_metric_json_output = no_metric_json_output

        # Will be populated
        self.pose = None
        self.fixed_atoms_data = None
        self.diffused_index_map = None  # INPUT → OUTPUT residue mapping from RFDiffusion3
        self.fixed_residues = {}  # {pose_idx: ['ALL'] or ['CB', 'CG', ...]}
        self.ligand_residues = []
        self.mobile_residues = []
        self.sfxn = None

        # Timing tracking
        self.timings = {}
        self.start_time = None

    def _determine_json_path(self, json_path, json_dir):
        """
        Determine JSON file path based on inputs.

        Priority:
        1. Explicit --json path if provided
        2. Same directory as PDB with same basename
        3. --corresponding_json_dir with same basename
        """
        if json_path is not None:
            return json_path

        # Auto-detect: same basename as PDB
        json_basename = self.pdbname + ".json"

        # Try same directory as PDB first
        candidate = os.path.join(self.pdb_dir, json_basename)
        if os.path.exists(candidate):
            return candidate

        # Try json_dir if specified
        if json_dir is not None:
            candidate = os.path.join(json_dir, json_basename)
            if os.path.exists(candidate):
                return candidate

        # Default: assume same dir as PDB (may not exist)
        return os.path.join(self.pdb_dir, json_basename)

    def load_json(self):
        """
        Load RFDiffusion3 JSON and extract select_fixed_atoms + diffused_index_map.

        The JSON structure can vary:
        - diffused_index_map: At top level (INPUT → OUTPUT residue mapping)
        - select_fixed_atoms: Can be at top level OR nested in specification{}
        """
        if not os.path.exists(self.json_path):
            print(f"WARNING: JSON file not found at {self.json_path}")
            print("Proceeding without fixed atom constraints from RFDiffusion3")
            self.fixed_atoms_data = {}
            self.diffused_index_map = {}
            return

        with open(self.json_path, 'r') as f:
            data = json.load(f)

        # Load diffused_index_map (INPUT → OUTPUT residue mapping)
        self.diffused_index_map = self._extract_field(
            data,
            ["diffused_index_map"],
            "diffused_index_map"
        )

        if self.diffused_index_map:
            print(f"✓ Loaded diffused_index_map from {self.json_path}")
            print(f"  Found {len(self.diffused_index_map)} residue mappings (input → output)")
        else:
            print(f"WARNING: 'diffused_index_map' not found in JSON")
            print(f"  Cannot map input residues to output residues")

        # Load select_fixed_atoms (try multiple locations)
        self.fixed_atoms_data = self._extract_field(
            data,
            ["select_fixed_atoms", "specification.select_fixed_atoms"],
            "select_fixed_atoms"
        )

        if self.fixed_atoms_data:
            print(f"✓ Loaded select_fixed_atoms from {self.json_path}")
            print(f"  Found {len(self.fixed_atoms_data)} fixed residue entries (input numbering)")
        else:
            print(f"WARNING: 'select_fixed_atoms' not found in JSON")
            print(f"  Top-level keys: {list(data.keys())}")
            if "specification" in data:
                print(f"  specification keys: {list(data['specification'].keys())[:10]}")
            print(f"  Proceeding with ligand-only constraints")

    def _extract_field(self, data, field_paths, field_name):
        """
        Extract a field from JSON, trying multiple possible locations.

        Args:
            data: JSON data dict
            field_paths: List of paths to try (e.g., ["field", "nested.field"])
            field_name: Field name for error messages

        Returns:
            Field value or empty dict if not found
        """
        for path in field_paths:
            try:
                # Navigate nested path (e.g., "specification.select_fixed_atoms")
                value = data
                for key in path.split('.'):
                    value = value[key]

                if value:
                    print(f"  Found {field_name} at: {path}")
                    return value
            except (KeyError, TypeError):
                continue

        return {}

    def map_json_residues_to_pose(self, pose):
        """
        Map JSON residue IDs to Rosetta pose indices using two-step process:
        1. INPUT residue (from select_fixed_atoms) → OUTPUT residue (via diffused_index_map)
        2. OUTPUT residue → Pose index (via PDBInfo)

        Args:
            pose: PyRosetta pose object

        Returns:
            dict: {pose_idx: atom_list}
        """
        fixed_residues = {}

        if not self.diffused_index_map:
            print("\nWARNING: No diffused_index_map available!")
            print("  Will attempt to use select_fixed_atoms residue IDs directly")
            print("  This may fail if RFDiffusion3 renumbered residues\n")

        print("Mapping fixed residues (INPUT → OUTPUT → Pose index):")

        for input_res_id, atom_list in self.fixed_atoms_data.items():
            # Step 1: Map INPUT residue → OUTPUT residue using diffused_index_map
            if self.diffused_index_map and input_res_id in self.diffused_index_map:
                output_res_id = self.diffused_index_map[input_res_id]
            else:
                if self.diffused_index_map:  # Only warn if map exists but key is missing
                    print(f"  WARNING: {input_res_id} not found in diffused_index_map, assuming no renumbering")
                output_res_id = input_res_id  # Fallback: use input residue ID directly

            # Step 2: Parse OUTPUT residue (e.g., "A251" → chain='A', resnum=251)
            chain = output_res_id[0]
            pdb_resnum = int(output_res_id[1:])

            # Step 3: Map OUTPUT residue → Pose index using PDBInfo
            pose_idx = pose.pdb_info().pdb2pose(chain, pdb_resnum)

            if pose_idx == 0:
                print(f"  WARNING: Could not map {output_res_id} to pose index")
                continue

            fixed_residues[pose_idx] = atom_list
            print(f"  {input_res_id} → {output_res_id} → pose {pose_idx}: {atom_list}")

        print(f"✓ Mapped {len(fixed_residues)} residues from JSON to pose")
        return fixed_residues

    def identify_ligands(self, pose):
        """
        Identify ligand residues (HETATM) in the pose.

        Args:
            pose: PyRosetta pose object

        Returns:
            list: Pose indices of ligand residues
        """
        ligand_res = []

        for i in range(1, pose.size() + 1):
            if pose.residue(i).is_ligand():
                ligand_res.append(i)
                resname = pose.residue(i).name()
                print(f"  Ligand at pose {i}: {resname}")

        print(f"✓ Found {len(ligand_res)} ligand residue(s)")
        return ligand_res

    def add_coordinate_constraints(self, pose, fixed_residues, ligand_residues):
        """
        Add coordinate constraints to fix atoms in absolute space.

        This is the KEY function - fixes atoms from JSON + ligand in place.

        Args:
            pose: PyRosetta pose (modified in-place)
            fixed_residues: dict {pose_idx: atom_spec}
                atom_spec is either "ALL" or comma-separated atom names "CB,CG,ND1,..."
            ligand_residues: list of ligand indices
        """
        print("\nAdding coordinate constraints to fix atoms in absolute space...")

        constraint_count = 0
        residue_details = []

        # 1. Add constraints for fixed residues from JSON
        for res_idx, atom_spec in fixed_residues.items():
            residue = pose.residue(res_idx)
            resname = residue.name3()
            pdb_info = f"{pose.pdb_info().chain(res_idx)}{pose.pdb_info().number(res_idx)}"

            # Parse atom specification
            if atom_spec == "ALL":
                # Constrain all heavy atoms
                atom_names = [residue.atom_name(i).strip()
                             for i in range(1, residue.natoms() + 1)
                             if not residue.atom_is_hydrogen(i)]
                print(f"  {pdb_info} ({resname}): Fixing ALL heavy atoms ({len(atom_names)} atoms)")
            else:
                # Split comma-separated atom names
                atom_names = [name.strip() for name in atom_spec.split(',')]
                print(f"  {pdb_info} ({resname}): Fixing {len(atom_names)} atoms: {', '.join(atom_names)}")

            # Add coordinate constraint for each atom
            added_for_this_res = 0
            missing_atoms = []

            for atom_name in atom_names:
                if not residue.has(atom_name):
                    missing_atoms.append(atom_name)
                    continue

                atom_idx = residue.atom_index(atom_name)
                xyz = residue.xyz(atom_name)

                # Create harmonic constraint function
                func = HarmonicFunc(0.0, self.coord_cst_stdev)

                # Create coordinate constraint
                cst = CoordinateConstraint(
                    AtomID(atom_idx, res_idx),
                    AtomID(1, 1),  # Dummy anchor (uses absolute coords)
                    xyz,
                    func
                )

                pose.add_constraint(cst)
                constraint_count += 1
                added_for_this_res += 1

            if missing_atoms:
                print(f"    WARNING: Missing {len(missing_atoms)} atoms: {', '.join(missing_atoms)}")

            residue_details.append(f"{pdb_info}:{added_for_this_res}")

        print(f"  ✓ Added {constraint_count} constraints for {len(fixed_residues)} fixed residues")

        # 2. Add constraints for ALL ligand atoms
        print(f"\n  Constraining ligand(s):")
        ligand_constraint_count = 0
        for lig_idx in ligand_residues:
            residue = pose.residue(lig_idx)
            resname = residue.name3()
            pdb_info = f"{pose.pdb_info().chain(lig_idx)}{pose.pdb_info().number(lig_idx)}"

            # Count heavy atoms
            heavy_atoms = [residue.atom_name(i).strip()
                          for i in range(1, residue.natoms() + 1)
                          if not residue.atom_is_hydrogen(i)]

            print(f"  {pdb_info} ({resname}): Fixing ALL heavy atoms ({len(heavy_atoms)} atoms)")

            # Constrain all heavy atoms in ligand
            for i in range(1, residue.natoms() + 1):
                if residue.atom_is_hydrogen(i):
                    continue

                atom_name = residue.atom_name(i).strip()
                xyz = residue.xyz(atom_name)

                func = HarmonicFunc(0.0, self.coord_cst_stdev)
                cst = CoordinateConstraint(
                    AtomID(i, lig_idx),
                    AtomID(1, 1),
                    xyz,
                    func
                )

                pose.add_constraint(cst)
                ligand_constraint_count += 1

        print(f"\n  ✓ Summary:")
        print(f"    - Fixed residues: {len(fixed_residues)} residues, {constraint_count} atom constraints")
        print(f"    - Ligands: {len(ligand_residues)} ligands, {ligand_constraint_count} atom constraints")
        print(f"    - TOTAL: {constraint_count + ligand_constraint_count} coordinate constraints")

    def define_mobile_region(self, pose, fixed_residues, ligand_residues, radius=10.0):
        """
        Define mobile region as neighborhood around ligands and fixed residues.

        Args:
            pose: PyRosetta pose object
            fixed_residues: dict of fixed residue indices
            ligand_residues: list of ligand residue indices
            radius: neighborhood radius in Angstroms

        Returns:
            list: Pose indices of mobile residues
        """
        print(f"\nDefining mobile region ({radius}Å around ligand/fixed residues)...")

        # Combine ligands and fixed residues as focus
        focus_residues = list(set(list(fixed_residues.keys()) + ligand_residues))

        if not focus_residues:
            print("  WARNING: No focus residues! Using all residues as mobile")
            return list(range(1, pose.size() + 1))

        # Create selector for focus residues
        focus_selector = ResidueIndexSelector(','.join(map(str, focus_residues)))

        # Create neighborhood selector
        neighborhood_selector = NeighborhoodResidueSelector(
            focus_selector,
            distance=radius,
            include_focus_in_subset=True  # Include focus for completeness
        )

        # Apply selector to get mobile+focus residues
        selected = neighborhood_selector.apply(pose)
        mobile_res = [i for i in range(1, pose.size() + 1) if selected[i]]

        # Don't explicitly remove fixed/ligand since coord constraints handle it
        # This allows mobile region to include them for packing, but constraints keep them in place

        print(f"✓ Mobile region: {len(mobile_res)} residues")
        return mobile_res

    def build_movemap(self, pose, mobile_residues, ligand_residues):
        """
        Build MoveMap allowing movement in mobile region.

        Note: Coordinate constraints handle fixing atoms, MoveMap just defines
        what DOFs are allowed to move.

        Args:
            pose: PyRosetta pose object
            mobile_residues: list of mobile residue indices
            ligand_residues: list of ligand indices

        Returns:
            MoveMap object
        """
        print("\nConfiguring MoveMap...")

        mm = pyrosetta.rosetta.core.kinematics.MoveMap()

        # Default: freeze everything
        mm.set_bb(False)
        mm.set_chi(False)
        mm.set_jump(False)

        # Enable movement for mobile residues
        for res_idx in mobile_residues:
            if res_idx not in ligand_residues:  # Don't enable ligand movement
                mm.set_bb(res_idx, True)
                mm.set_chi(res_idx, True)

        # Explicitly freeze ligands (belt + suspenders with coord constraints)
        for lig_idx in ligand_residues:
            mm.set_bb(lig_idx, False)
            mm.set_chi(lig_idx, False)

        mobile_non_lig = [r for r in mobile_residues if r not in ligand_residues]
        print(f"✓ MoveMap configured: {len(mobile_non_lig)} mobile residues (excluding ligands)")

        return mm

    def setup_scorefunction(self):
        """
        Setup cartesian scorefunction with proper weights.

        Returns:
            ScoreFunction object
        """
        print("\nSetting up cartesian scorefunction...")

        # Try to load a true cartesian scorefunction from weights file
        from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory

        sfxn = None
        cartesian_weights = ["ref2015_cart", "ref2015", "score12"]

        for weights_name in cartesian_weights:
            try:
                sfxn = ScoreFunctionFactory.create_score_function(weights_name)
                print(f"  Loaded scorefunction: {weights_name}")
                break
            except:
                pass

        if sfxn is None:
            # Last resort: use default
            print("  Using default scorefunction")
            sfxn = pyr.get_fa_scorefxn()

        # Explicitly set up for cartesian minimization
        # These weights are REQUIRED for cartesian scoring to work properly

        # 1. Cartesian bonded terms (CRITICAL for cartesian)
        sfxn.set_weight(ScoreType.cart_bonded, self.cart_bonded_weight)
        print(f"  Set cart_bonded weight: {self.cart_bonded_weight}")

        # 2. Turn OFF pro_close (incompatible with cart_bonded in some Rosetta versions)
        sfxn.set_weight(ScoreType.pro_close, 0.0)

        # 3. Add coordinate constraint weight (KEY for fixing atoms)
        sfxn.set_weight(ScoreType.coordinate_constraint, self.coord_cst_weight)
        print(f"  Set coordinate_constraint weight: {self.coord_cst_weight}")

        # 4. Verify scorefunction has cart_bonded
        if not sfxn.has_nonzero_weight(ScoreType.cart_bonded):
            print("  ERROR: Scorefunction does not have cart_bonded weight!")
            print("  Cartesian minimization WILL fail")
        else:
            print(f"  ✓ Cartesian scoring enabled (cart_bonded = {sfxn.get_weight(ScoreType.cart_bonded)})")

        return sfxn

    def detect_chain_breaks(self, pose):
        """
        Detect chain breaks in the pose.

        Returns:
            list: Residue indices where chain breaks occur
        """
        breaks = []

        for i in range(1, pose.size()):
            # Check if peptide bond is broken (distance > 2.0 Å)
            try:
                if not pose.residue(i).is_protein() or not pose.residue(i+1).is_protein():
                    continue

                c_xyz = pose.residue(i).xyz("C")
                n_xyz = pose.residue(i+1).xyz("N")
                dist = c_xyz.distance(n_xyz)

                if dist > 2.0:  # Typical peptide bond ~ 1.33 Å
                    breaks.append(i)
                    print(f"  Chain break detected: {i}-{i+1} (distance: {dist:.2f} Å)")
            except:
                continue

        return breaks

    def idealize_secondary_structure(self, pose):
        """
        Idealize secondary structure elements (optional preprocessing).

        Args:
            pose: PyRosetta pose (modified in-place)
        """
        print("\nIdealizing secondary structure...")
        try:
            from pyrosetta.rosetta.protocols.idealize import IdealizeMover
            idealize = IdealizeMover()
            idealize.apply(pose)
            print("✓ Secondary structure idealized")
        except Exception as e:
            print(f"  WARNING: Could not idealize: {e}")

    def run_cartesian_fastrelax(self, pose, sfxn, movemap, n_repeats=2):
        """
        Run cartesian FastRelax for refinement.

        Args:
            pose: PyRosetta pose (modified in-place)
            sfxn: ScoreFunction
            movemap: MoveMap
            n_repeats: Number of FastRelax cycles
        """
        print(f"\nRunning cartesian FastRelax ({n_repeats} repeats)...")

        relax = FastRelax(sfxn, n_repeats)
        relax.set_movemap(movemap)
        relax.cartesian(True)
        relax.min_type("lbfgs_armijo_nonmonotone")

        score_before = sfxn(pose)
        relax.apply(pose)
        score_after = sfxn(pose)

        print(f"✓ FastRelax complete")
        print(f"  Score: {score_before:.2f} → {score_after:.2f} (Δ {score_after - score_before:+.2f})")

    def run_cartesian_minimize(self, pose, sfxn, movemap, tolerance=0.0001):
        """
        Run cartesian minimization for final geometry cleanup.

        Args:
            pose: PyRosetta pose (modified in-place)
            sfxn: ScoreFunction
            movemap: MoveMap
            tolerance: Minimization tolerance
        """
        print(f"\nRunning cartesian minimization (tolerance={tolerance})...")

        min_mover = MinMover(movemap, sfxn, "lbfgs_armijo_nonmonotone", tolerance, True)
        min_mover.cartesian(True)

        score_before = sfxn(pose)
        min_mover.apply(pose)
        score_after = sfxn(pose)

        print(f"✓ Minimization complete")
        print(f"  Score: {score_before:.2f} → {score_after:.2f} (Δ {score_after - score_before:+.2f})")

    def calculate_validation_metrics(self, pose, sfxn, fixed_residues, ligand_residues, output_pdb_path=None):
        """
        Calculate comprehensive validation metrics for geometry quality.

        Args:
            pose: PyRosetta pose
            sfxn: ScoreFunction
            fixed_residues: dict of fixed residue indices
            ligand_residues: list of ligand indices
            output_pdb_path: Path to output PDB file (for metadata)

        Returns:
            dict: Nested dictionary of metrics (JSON-serializable)
        """
        print("\n  Calculating metrics...")
        metrics_start = time.time()

        metrics = {
            'metadata': {},
            'global_metrics': {},  # Always include - key summary stats
            'scores': {},
            'catalytic_residues': {},
            'quality_flags': {}
        }

        # Add debug-only sections if requested
        if self.debug:
            metrics['geometry'] = {}
            metrics['constraints'] = {}
            metrics['timing'] = self.timings.copy() if hasattr(self, 'timings') else {}

        # === METADATA ===
        metrics['metadata']['structure_name'] = self.pdbname
        if output_pdb_path:
            metrics['metadata']['pdb_path'] = output_pdb_path
        if self.debug:
            metrics['metadata']['timestamp'] = datetime.now().isoformat()
        metrics['metadata']['num_residues'] = int(pose.size())
        metrics['metadata']['num_fixed_residues'] = len(fixed_residues)
        metrics['metadata']['num_ligands'] = len(ligand_residues)
        if self.debug:
            metrics['metadata']['mobile_region_size'] = len(self.mobile_residues) if hasattr(self, 'mobile_residues') else 0

        # === ENERGY SCORES ===
        metrics['scores']['total_score'] = float(sfxn(pose))
        metrics['scores']['cart_bonded'] = float(pose.energies().total_energies()[ScoreType.cart_bonded])
        metrics['scores']['coordinate_constraint'] = float(pose.energies().total_energies()[ScoreType.coordinate_constraint])

        try:
            metrics['scores']['fa_rep'] = float(pose.energies().total_energies()[ScoreType.fa_rep])
            metrics['scores']['fa_atr'] = float(pose.energies().total_energies()[ScoreType.fa_atr])
            metrics['scores']['fa_sol'] = float(pose.energies().total_energies()[ScoreType.fa_sol])
            metrics['scores']['fa_elec'] = float(pose.energies().total_energies()[ScoreType.fa_elec])
            metrics['scores']['hbond_sc'] = float(pose.energies().total_energies()[ScoreType.hbond_sc])
            metrics['scores']['hbond_bb_sc'] = float(pose.energies().total_energies()[ScoreType.hbond_bb_sc])
        except:
            pass

        # === GEOMETRY QUALITY ===
        # Always calculate for quality flags, but only store in metrics if debug mode
        clash_threshold = 10.0  # fa_rep per residue

        # Chain breaks (always calculate for quality check)
        chain_breaks = self.detect_chain_breaks(pose)
        num_chain_breaks = len(chain_breaks)

        # Clashing residues (always calculate count for quality check)
        num_clashing_residues = 0
        clashing_residues_list = []
        for i in range(1, pose.size() + 1):
            try:
                res_energy = pose.energies().residue_total_energies(i)[ScoreType.fa_rep]
                if res_energy > clash_threshold:
                    num_clashing_residues += 1
                    if self.debug:  # Only build detailed list in debug mode
                        pdb_id = f"{pose.pdb_info().chain(i)}{pose.pdb_info().number(i)}"
                        clashing_residues_list.append({
                            'residue': int(i),
                            'pdb_id': pdb_id,
                            'resname': pose.residue(i).name3(),
                            'fa_rep': float(res_energy)
                        })
            except:
                continue

        # Store in metrics only if debug mode
        if self.debug:
            metrics['geometry']['num_chain_breaks'] = num_chain_breaks
            metrics['geometry']['chain_break_residues'] = [int(x) for x in chain_breaks]
            metrics['geometry']['num_clashing_residues'] = num_clashing_residues
            # Sort by fa_rep (worst first) and store top 20
            clashing_residues_list.sort(key=lambda x: x['fa_rep'], reverse=True)
            metrics['geometry']['clashing_residues'] = clashing_residues_list[:20]

        # === CONSTRAINT SATISFACTION ===
        # Calculate CA-RMSD and displacement stats for quality checks
        mean_fixed_rmsd = 0.0
        max_fixed_rmsd = 0.0

        if hasattr(self, 'original_pose'):
            # Overall CA-RMSD (shows global protein movement/alignment quality)
            from pyrosetta.rosetta.core.scoring import CA_rmsd
            ca_rmsd_val = CA_rmsd(self.original_pose, pose)

            if self.debug:
                metrics['constraints']['ca_rmsd_overall'] = float(ca_rmsd_val)

            # Fixed atom displacement in ABSOLUTE space (NO superposition!)
            # This is correct because coordinate constraints fix atoms in absolute coords
            fixed_res_displacements = []

            for res_idx in list(fixed_residues.keys()) + ligand_residues:
                if res_idx > pose.size():
                    continue

                try:
                    res_orig = self.original_pose.residue(res_idx)
                    res_final = pose.residue(res_idx)

                    pdb_id = f"{pose.pdb_info().chain(res_idx)}{pose.pdb_info().number(res_idx)}"
                    resname = res_final.name3()

                    # Determine which atoms to check
                    atom_spec = fixed_residues.get(res_idx, "ALL")
                    if res_idx in ligand_residues:
                        atom_spec = "ALL"  # Ligands are fully constrained

                    if atom_spec == "ALL":
                        atom_names = [res_final.atom_name(i).strip()
                                     for i in range(1, res_final.natoms() + 1)
                                     if not res_final.atom_is_hydrogen(i)]
                    else:
                        atom_names = [name.strip() for name in atom_spec.split(',')]

                    # Calculate displacement in ABSOLUTE coordinates
                    sq_dev = []
                    max_dev_atom = None
                    max_dev_val = 0.0
                    
                    for atom_name in atom_names:
                        if not res_orig.has(atom_name) or not res_final.has(atom_name):
                            continue
                        dist = res_orig.xyz(atom_name).distance(res_final.xyz(atom_name))
                        sq_dev.append(dist**2)
                        if dist > max_dev_val:
                            max_dev_val = dist
                            max_dev_atom = atom_name

                    if sq_dev:
                        rmsd = float(np.sqrt(np.mean(sq_dev)))
                        max_dev = float(np.sqrt(max(sq_dev)))

                        fixed_res_displacements.append({
                            'residue': int(res_idx),
                            'pdb_id': pdb_id,
                            'resname': resname,
                            'atom_spec': atom_spec,
                            'num_atoms': len(sq_dev),
                            'rmsd': rmsd,
                            'max_displacement': max_dev,
                            'max_displacement_atom': max_dev_atom
                        })
                except Exception as e:
                    continue

            # Sort by max displacement (worst first)
            fixed_res_displacements.sort(key=lambda x: x['max_displacement'], reverse=True)

            # Summary statistics (always calculate for quality flags)
            if fixed_res_displacements:
                all_rmsds = [x['rmsd'] for x in fixed_res_displacements]
                all_max_disps = [x['max_displacement'] for x in fixed_res_displacements]
                mean_fixed_rmsd = float(np.mean(all_rmsds))
                max_fixed_rmsd = float(np.max(all_rmsds))

                if self.debug:
                    metrics['constraints']['fixed_residue_displacements'] = fixed_res_displacements
                    metrics['constraints']['mean_fixed_rmsd'] = mean_fixed_rmsd
                    metrics['constraints']['max_fixed_rmsd'] = max_fixed_rmsd
                    metrics['constraints']['mean_max_displacement'] = float(np.mean(all_max_disps))
                    metrics['constraints']['max_displacement_overall'] = float(np.max(all_max_disps))

        # === CATALYTIC RESIDUE DETAILS ===
        # Always include this section (it's the most important!)
        cat_res_details = []

        for res_idx, atom_spec in fixed_residues.items():
            if res_idx > pose.size():
                continue

            residue = pose.residue(res_idx)
            pdb_id = f"{pose.pdb_info().chain(res_idx)}{pose.pdb_info().number(res_idx)}"

            # Per-residue cart_bonded score (TOTAL for residue)
            try:
                res_cart_bonded_total = float(pose.energies().residue_total_energies(res_idx)[ScoreType.cart_bonded])
            except:
                res_cart_bonded_total = 0.0

            # Per-residue fa_rep (clashes)
            try:
                res_fa_rep = float(pose.energies().residue_total_energies(res_idx)[ScoreType.fa_rep])
            except:
                res_fa_rep = 0.0

            # Calculate bond length deviations
            bond_geom = self.calculate_bond_length_deviations(residue)

            # Get displacement info from calculated displacements (always available)
            displacement_info = None
            if hasattr(self, 'original_pose'):
                # Find this residue in the fixed_res_displacements we calculated above
                for disp in fixed_res_displacements:
                    if disp['residue'] == res_idx:
                        displacement_info = {
                            'rmsd': disp['rmsd'],
                            'max_displacement': disp['max_displacement'],
                            'max_displacement_atom': disp.get('max_displacement_atom')
                        }
                        break

            # Determine if fully or partially constrained
            is_fully_constrained = (atom_spec == "ALL")

            res_detail = {
                'residue': int(res_idx),
                'pdb_id': pdb_id,
                'resname': residue.name3(),
                'atom_spec': atom_spec,
                'fully_constrained': is_fully_constrained,
                'cart_bonded': res_cart_bonded_total,
                'fa_rep': res_fa_rep,
                'is_clashing': res_fa_rep > clash_threshold,
                'bond_geometry': bond_geom
            }

            # Add displacement info if available
            if displacement_info:
                res_detail['displacement'] = displacement_info

            cat_res_details.append(res_detail)

        # Sort by cart_bonded (worst first)
        cat_res_details.sort(key=lambda x: x['cart_bonded'], reverse=True)
        metrics['catalytic_residues']['details'] = cat_res_details

        # Summary
        if cat_res_details:
            metrics['catalytic_residues']['mean_cart_bonded'] = float(np.mean([x['cart_bonded'] for x in cat_res_details]))
            metrics['catalytic_residues']['max_cart_bonded'] = float(np.max([x['cart_bonded'] for x in cat_res_details]))
            metrics['catalytic_residues']['num_clashing'] = sum(1 for x in cat_res_details if x['is_clashing'])

        # === QUALITY FLAGS & EXPLANATIONS ===
        passed_checks = []
        failed_checks = []

        # Check chain breaks
        if num_chain_breaks == 0:
            passed_checks.append('no_chain_breaks')
        else:
            failed_checks.append(f"{num_chain_breaks} chain breaks detected")

        # Check clashes
        if num_clashing_residues < 5:
            passed_checks.append('low_clashes')
        else:
            failed_checks.append(f"{num_clashing_residues} clashing residues (threshold: <5)")

        # Check fixed atom displacement
        if max_fixed_rmsd > 0:
            if max_fixed_rmsd < 0.1:
                passed_checks.append('tight_constraints')
            else:
                failed_checks.append(f"Fixed atoms moved {max_fixed_rmsd:.3f}Å (threshold: <0.1Å)")

        # Check catalytic residue geometry
        mean_cat_cart_bonded = metrics['catalytic_residues'].get('mean_cart_bonded', 0.0)
        if cat_res_details and mean_cat_cart_bonded < 2.0:
            passed_checks.append('good_catalytic_geometry')
        elif cat_res_details and mean_cat_cart_bonded > 0:
            failed_checks.append(f"Catalytic residues cart_bonded {mean_cat_cart_bonded:.2f} (threshold: <2.0)")

        # Overall geometry acceptable
        geometry_acceptable = (
            num_chain_breaks == 0 and
            num_clashing_residues < 5 and
            max_fixed_rmsd < 0.1
        )

        metrics['quality_flags']['geometry_acceptable'] = geometry_acceptable
        metrics['quality_flags']['passed_checks'] = passed_checks
        metrics['quality_flags']['failed_checks'] = failed_checks
        metrics['quality_flags']['explanation'] = (
            "All checks passed" if geometry_acceptable
            else "Failed: " + "; ".join(failed_checks)
        )

        # === GLOBAL METRICS (always included - summary stats) ===
        metrics['global_metrics']['num_chain_breaks'] = num_chain_breaks
        metrics['global_metrics']['num_clashing_residues'] = num_clashing_residues
        metrics['global_metrics']['total_score'] = metrics['scores']['total_score']
        metrics['global_metrics']['cart_bonded'] = metrics['scores']['cart_bonded']

        # CA-RMSD if available
        if hasattr(self, 'original_pose'):
            from pyrosetta.rosetta.core.scoring import CA_rmsd
            ca_rmsd_val = CA_rmsd(self.original_pose, pose)
            metrics['global_metrics']['ca_rmsd_overall'] = float(ca_rmsd_val)
            metrics['global_metrics']['mean_fixed_atom_displacement'] = mean_fixed_rmsd
            metrics['global_metrics']['max_fixed_atom_displacement'] = max_fixed_rmsd

        # Catalytic residue summary
        if cat_res_details:
            metrics['global_metrics']['mean_catalytic_cart_bonded'] = metrics['catalytic_residues'].get('mean_cart_bonded', 0.0)
            metrics['global_metrics']['num_catalytic_clashing'] = metrics['catalytic_residues'].get('num_clashing', 0)

        # Add metrics calculation time
        if self.debug:
            metrics['timing']['metrics_calculation'] = time.time() - metrics_start

        return metrics

    def strip_energy_table_from_pdb(self, pdb_path):
        """
        Remove the per-residue energy table from the bottom of the PDB file to save disk space.

        Args:
            pdb_path: Path to PDB file
        """
        with open(pdb_path, 'r') as f:
            lines = f.readlines()

        # Find the start of the energy table
        new_lines = []
        in_energy_table = False

        for line in lines:
            if line.startswith('# All scores below are weighted scores'):
                in_energy_table = True
                continue
            elif line.startswith('#END_POSE_ENERGIES_TABLE'):
                in_energy_table = False
                continue

            if not in_energy_table:
                new_lines.append(line)

        # Write back without energy table
        with open(pdb_path, 'w') as f:
            f.writelines(new_lines)

    def calculate_bond_length_deviations(self, residue):
        """
        Calculate maximum bond length deviation from ideal for a residue.

        Creates a reference residue with ideal geometry and compares bond lengths.

        Args:
            residue: PyRosetta residue object

        Returns:
            dict: {'max_bond_deviation': float, 'mean_bond_deviation': float, 'num_bonds_checked': int}
        """
        try:
            # Create ideal reference: A-X-A pose where X is the residue type
            ref_pose = pyr.pose_from_sequence("A" + residue.name1() + "A")
            ref_res = ref_pose.residue(2)  # Middle residue is our target type

            bond_deviations = []

            # Check all bonds (skip hydrogens)
            for atom_idx in range(1, residue.natoms() + 1):
                if residue.atom_type(atom_idx).element() == "H":
                    continue

                atom_name = residue.atom_name(atom_idx).strip()

                # Skip if reference doesn't have this atom (shouldn't happen for standard residues)
                if not ref_res.has(atom_name):
                    continue

                # Get bonded neighbors
                for bonded_idx in residue.bonded_neighbor(atom_idx):
                    if residue.atom_type(bonded_idx).element() == "H":
                        continue

                    bonded_name = residue.atom_name(bonded_idx).strip()

                    # Skip if reference doesn't have this atom
                    if not ref_res.has(bonded_name):
                        continue

                    # Calculate actual bond length
                    actual_dist = (residue.xyz(atom_name) - residue.xyz(bonded_name)).norm()

                    # Calculate ideal bond length from reference
                    ideal_dist = (ref_res.xyz(atom_name) - ref_res.xyz(bonded_name)).norm()

                    # Deviation
                    deviation = abs(actual_dist - ideal_dist)
                    bond_deviations.append(deviation)

            return {
                'max_bond_deviation': round(max(bond_deviations), 4) if bond_deviations else 0.0,
                'mean_bond_deviation': round(float(np.mean(bond_deviations)), 4) if bond_deviations else 0.0,
                'num_bonds_checked': len(bond_deviations)
            }

        except Exception as e:
            # If anything fails, return zeros (e.g., for non-standard residues like ligands)
            return {
                'max_bond_deviation': 0.0,
                'mean_bond_deviation': 0.0,
                'num_bonds_checked': 0
            }

    def round_metrics(self, obj, decimals=4):
        """
        Recursively round all float values in a nested dict/list structure.

        Args:
            obj: Dictionary, list, or primitive value
            decimals: Number of decimal places (default: 4)

        Returns:
            Rounded version of obj
        """
        if isinstance(obj, dict):
            return {k: self.round_metrics(v, decimals) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.round_metrics(item, decimals) for item in obj]
        elif isinstance(obj, float):
            # Round to specified decimals
            return round(obj, decimals)
        else:
            # Return as-is (int, str, bool, None)
            return obj

    def write_metrics_json(self, metrics, output_path):
        """
        Write validation metrics to JSON file (human-readable, nested structure).

        Args:
            metrics: Dictionary of metrics
            output_path: Path for output JSON (will use PDB basename)
        """
        # Round all floats to save memory and improve readability
        metrics = self.round_metrics(metrics, decimals=4)

        # Determine JSON path (same basename as PDB)
        json_path = output_path.replace('.pdb', '_metrics.json')

        # Write to JSON with nice formatting
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2, sort_keys=False)

        print(f"✓ Metrics written to: {json_path}")

        return json_path

    def run(self, output_path=None, do_idealize_ss=False,
            do_fastrelax=True, do_minimize=True, fastrelax_cycles=2,
            min_tolerance=0.0001, strip_pdb_energies=True):
        """
        Main execution function.

        Args:
            output_path: Path for output PDB (if None, auto-generated)
            do_idealize_ss: Whether to idealize secondary structure first
            do_fastrelax: Whether to run FastRelax
            do_minimize: Whether to run final minimization
            fastrelax_cycles: Number of FastRelax cycles (default: 2, more = better clash resolution)
            min_tolerance: Minimization tolerance (default: 0.0001, lower = more refined geometry)
            strip_pdb_energies: Remove per-residue energy table from PDB to save disk space

        Returns:
            PyRosetta pose object
        """
        print("\n" + "="*70)
        print("RFDiffusion3 Geometry Idealizer")
        print("="*70)

        # Start timing
        self.start_time = time.time()
        self.timings = {}



        # Load JSON
        json_start = time.time()
        self.load_json()
        self.timings["json_loading"] = time.time() - json_start

        # Load pose
        pose_start = time.time()
        print(f"\nLoading PDB: {self.pdb_path}")
        self.pose = pyr.pose_from_file(self.pdb_path)
        self.timings["pose_loading"] = time.time() - pose_start
        print(f"✓ Loaded pose with {self.pose.size()} residues")

        # Store original pose for RMSD calculation later
        self.original_pose = self.pose.clone()

        # Map JSON residues to pose indices
        if self.fixed_atoms_data:
            self.fixed_residues = self.map_json_residues_to_pose(self.pose)
        else:
            print("\nNo fixed atoms data - will only fix ligand")
            self.fixed_residues = {}

        # Identify ligands
        self.ligand_residues = self.identify_ligands(self.pose)

        if not self.ligand_residues and not self.fixed_residues:
            print("\n ERROR: No ligands or fixed residues found!")
            print("Cannot define constraints. Exiting.")
            sys.exit(1)

        # Setup scorefunction
        self.sfxn = self.setup_scorefunction()

        # Add coordinate constraints (KEY STEP!)
        cst_start = time.time()
        self.add_coordinate_constraints(
            self.pose,
            self.fixed_residues,
            self.ligand_residues
        )
        self.timings["constraint_setup"] = time.time() - cst_start

        # Define mobile region
        self.mobile_residues = self.define_mobile_region(
            self.pose,
            self.fixed_residues,
            self.ligand_residues,
            radius=self.mobile_radius
        )

        # Build MoveMap
        movemap = self.build_movemap(
            self.pose,
            self.mobile_residues,
            self.ligand_residues
        )

        # Check for chain breaks
        print("\nChecking for chain breaks...")
        breaks = self.detect_chain_breaks(self.pose)
        if breaks:
            print(f"  WARNING: Found {len(breaks)} chain breaks")
            print(f"  Minimization may help resolve small breaks")
        else:
            print("✓ No chain breaks detected")

        # Optional: Idealize secondary structure
        if do_idealize_ss:
            self.idealize_secondary_structure(self.pose)

        # Run geometry idealization
        print("\n" + "="*70)
        print("GEOMETRY IDEALIZATION PROTOCOL")
        print("="*70)

        initial_score = self.sfxn(self.pose)
        print(f"\nInitial score: {initial_score:.2f}")

        if do_fastrelax:
            fr_start = time.time()
            self.run_cartesian_fastrelax(self.pose, self.sfxn, movemap, n_repeats=fastrelax_cycles)
            self.timings["fastrelax"] = time.time() - fr_start

        if do_minimize:
            min_start = time.time()
            self.run_cartesian_minimize(self.pose, self.sfxn, movemap, tolerance=min_tolerance)
            self.timings["minimization"] = time.time() - min_start

        final_score = self.sfxn(self.pose)
        print(f"\n{'='*70}")
        print(f"Final score: {final_score:.2f} (Δ {final_score - initial_score:+.2f})")
        print(f"{'='*70}\n")

        # Check chain breaks again
        breaks_after = self.detect_chain_breaks(self.pose)
        if breaks_after:
            print(f"⚠  Still have {len(breaks_after)} chain breaks after idealization")
        else:
            print("✓ No chain breaks after idealization")

        # Output
        if output_path is None:
            output_path = f"{self.pdbname}_idealized.pdb"

        self.pose.dump_pdb(output_path)

        # Strip energy table to save disk space
        if strip_pdb_energies:
            self.strip_energy_table_from_pdb(output_path)

        print(f"\n✓ Output written to: {output_path}")

        # Calculate validation metrics (unless disabled)
        if not self.no_metric_json_output:
            print("\n" + "="*70)
            print("CALCULATING VALIDATION METRICS")
            print("="*70)
            metrics = self.calculate_validation_metrics(
                self.pose,
                self.sfxn,
                self.fixed_residues,
                self.ligand_residues,
                output_pdb_path=output_path
            )

            print(f"\n✓ Validation Summary:")

            # Extract actual numbers from failed checks
            num_chain_breaks = 0
            num_clashing = 0
            for check in metrics['quality_flags']['failed_checks']:
                if 'chain breaks' in check:
                    # Extract number from string like "2 chain breaks detected"
                    num_chain_breaks = int(check.split()[0])
                elif 'clashing residues' in check:
                    # Extract number from string like "17 clashing residues (threshold: <5)"
                    num_clashing = int(check.split()[0])

            print(f"  - Chain breaks: {num_chain_breaks}")
            print(f"  - Clashing residues: {num_clashing}")
            print(f"  - Catalytic residues cart_bonded (mean): {metrics['catalytic_residues'].get('mean_cart_bonded', 0):.2f}")
            print(f"  - Geometry acceptable: {metrics['quality_flags']['geometry_acceptable']}")
            if not metrics['quality_flags']['geometry_acceptable']:
                print(f"  - Reason: {metrics['quality_flags']['explanation']}")

            # Write metrics JSON
            self.timings["total_runtime"] = time.time() - self.start_time
            if self.debug:
                metrics["timing"] = self.timings  # Only include timing in debug mode
            json_path = self.write_metrics_json(metrics, output_path)

        # Print timing summary
        self.timings["total_runtime"] = time.time() - self.start_time
        print("\n" + "="*70)
        print("TIMING SUMMARY")
        print("="*70)
        print(f"  JSON loading:         {self.timings.get('json_loading', 0):.2f}s")
        print(f"  Pose loading:         {self.timings.get('pose_loading', 0):.2f}s")
        print(f"  Constraint setup:     {self.timings.get('constraint_setup', 0):.2f}s")
        if 'fastrelax' in self.timings:
            print(f"  FastRelax:            {self.timings.get('fastrelax', 0):.2f}s")
        if 'minimization' in self.timings:
            print(f"  Minimization:         {self.timings.get('minimization', 0):.2f}s")
        if 'metrics_calculation' in self.timings:
            print(f"  Metrics calculation:  {self.timings.get('metrics_calculation', 0):.2f}s")
        print(f"  {'─'*40}")
        print(f"  TOTAL RUNTIME:        {self.timings['total_runtime']:.2f}s")
        print("="*70)

        print("")
        return self.pose


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Idealize RFDiffusion3 scaffolds with cartesian relaxation (Stage 1 of predesign)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (auto-finds JSON in same directory)
  %(prog)s --pdb structure.pdb --params ligand.params

  # JSON in different directory
  %(prog)s --pdb /path/to/structure.pdb --corresponding_json_dir /path/to/jsons/ --params ligand.params

  # Explicit JSON path
  %(prog)s --pdb structure.pdb --json structure.json --params ligand.params

  # Custom mobile radius and constraint parameters
  %(prog)s --pdb structure.pdb --params ligand.params --mobile_radius 12.0 --coord_cst_weight 15.0

  # Fast mode (minimize only, no FastRelax)
  %(prog)s --pdb structure.pdb --params ligand.params --skip_fastrelax

  # With secondary structure idealization
  %(prog)s --pdb structure.pdb --params ligand.params --idealize_ss
        """
    )

    parser.add_argument("--pdb", type=str, required=True,
                        help="Input PDB file from RFDiffusion3")
    parser.add_argument("--json", type=str, default=None,
                        help="RFDiffusion3 JSON file (default: auto-detect from PDB basename)")
    parser.add_argument("--corresponding_json_dir", type=str, default=None,
                        help="Directory to search for JSON if not in same dir as PDB")
    parser.add_argument("--params", type=str, nargs="+",
                        help="Ligand parameter file(s)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PDB path (default: <input>_idealized.pdb)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory")

    # Geometry parameters
    parser.add_argument("--mobile_radius", type=float, default=10.0,
                        help="Radius (Å) for mobile region around ligand/fixed residues (default: 10.0)")
    parser.add_argument("--cart_bonded_weight", type=float, default=0.5,
                        help="Weight for cart_bonded score term (default: 0.5, range: 0.4-0.7)")
    parser.add_argument("--coord_cst_weight", type=float, default=100.0,
                        help="Weight for coordinate constraints (default: 100.0, range: 100-1000 for near-zero movement)")
    parser.add_argument("--coord_cst_stdev", type=float, default=0.01,
                        help="Standard deviation for coordinate constraints in Å (default: 0.01, lower=tighter)")

    # Protocol options
    parser.add_argument("--idealize_ss", action="store_true",
                        help="Idealize secondary structure before relaxation")
    parser.add_argument("--skip_fastrelax", action="store_true",
                        help="Skip FastRelax, only do minimization (faster)")
    parser.add_argument("--skip_minimize", action="store_true",
                        help="Skip final minimization (not recommended)")
    parser.add_argument("--fastrelax_cycles", type=int, default=2,
                        help="Number of FastRelax cycles (default: 2, more = better clash resolution, slower)")
    parser.add_argument("--min_tolerance", type=float, default=0.0001,
                        help="Minimization tolerance (default: 0.0001, lower = more refined but slower)")

    # Output options
    parser.add_argument("--debug", action="store_true",
                        help="Include full verbose metrics in JSON (timing, geometry, all displacements)")
    parser.add_argument("--no_metric_json_output", action="store_true",
                        help="Disable JSON metrics file creation entirely")
    parser.add_argument("--keep_pdb_energies", action="store_true",
                        help="Keep per-residue energy table in PDB (takes more disk space)")

    args = parser.parse_args()

    # Initialize PyRosetta
    extra_res_fa = ""
    if args.params:
        extra_res_fa = "-extra_res_fa " + " ".join(args.params)

    NPROC = os.cpu_count()
    if "OMP_NUM_THREADS" in os.environ:
        NPROC = int(os.environ["OMP_NUM_THREADS"])
    if "SLURM_CPUS_ON_NODE" in os.environ:
        NPROC = int(os.environ["SLURM_CPUS_ON_NODE"])

    init_flags = [
        extra_res_fa,
        "-run:preserve_header",
        f"-multithreading:total_threads {NPROC}",
        f"-multithreading:interaction_graph_threads {NPROC}",
        "-mute all",
        "-unmute core.scoring.ScoreFunction",
        "-unmute core.optimization.CartesianMinimizer"  # Unmute to see cartesian errors
    ]

    init_cmd = " ".join([f for f in init_flags if f])

    print("Initializing PyRosetta...")
    print(f"  Using {NPROC} threads")
    pyr.init(init_cmd)
    print("✓ PyRosetta initialized\n")

    # Setup output path
    output_path = args.output
    if output_path is None:
        pdbname = os.path.basename(args.pdb).replace(".pdb", "")
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            output_path = f"{args.output_dir}/{pdbname}_idealized.pdb"
        else:
            output_path = f"{pdbname}_idealized.pdb"

    # Run idealization
    idealizer = RFDiffusion3GeometryIdealizer(
        pdb_path=args.pdb,
        json_path=args.json,
        json_dir=args.corresponding_json_dir,
        params=args.params,
        mobile_radius=args.mobile_radius,
        cart_bonded_weight=args.cart_bonded_weight,
        coord_cst_weight=args.coord_cst_weight,
        coord_cst_stdev=args.coord_cst_stdev,
        debug=args.debug,
        no_metric_json_output=args.no_metric_json_output
    )

    idealizer.run(
        output_path=output_path,
        do_idealize_ss=args.idealize_ss,
        do_fastrelax=not args.skip_fastrelax,
        do_minimize=not args.skip_minimize,
        fastrelax_cycles=args.fastrelax_cycles,
        min_tolerance=args.min_tolerance,
        strip_pdb_energies=not args.keep_pdb_energies
    )

    print("="*70)
    print("✓ GEOMETRY IDEALIZATION COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
