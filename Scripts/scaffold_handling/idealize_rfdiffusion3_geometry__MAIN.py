#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RFDiffusion3 Geometry Idealizer - Stage 1 of Predesign Pipeline

Pipeline (single CPU-core friendly; scales to more threads when available):
1. Load RFDiffusion3 outputs (PDB + JSON select_fixed_atoms + diffused_index_map).
2. Normalize JSON atom-specs into a canonical {pose_idx: frozenset(atom_names)}.
3. Detect declared covalent contacts: fixed-fixed atom pairs within a distance
   threshold. These are "supposed to be close" (e.g. a catalytic LYS NZ bonded
   to a ligand carbon in a hacky PTM). We attempt declare_chemical_bond on
   each pair (best-effort, may exclude them from the fa_rep CountPair) AND
   subtract their fa_rep contribution from reported clash metrics so the
   output report isn't dominated by intentional close contacts.
4. (Optional, on by default) Run ligandMPNN pre-design at T=0.1 generating
   10 sequences with all 20 amino acids allowed. Catalytic residues and the
   ligand are frozen (identity AND rotamer). Select best candidate by:
       lowest (fa_rep_catalytic + fa_rep_ligand_interface + cart_bonded_catalytic
               - hbond_bonus)
   where hbond_bonus counts sidechain hbonds from non-catalytic residues to
   the ligand and/or catalytic residues. Restore HIS tautomers post-MPNN so
   Rosetta's default does not overwrite the user-specified tautomer.
5. Apply coordinate constraints to all fixed atoms (from JSON) + all ligand
   atoms (absolute-space pin).
6. Build MoveMap: fully-fixed residues are frozen at the residue level to
   reduce DOFs that fight stiff coord csts; ligand is always frozen.
7. Cartesian FastRelax + cartesian minimize.
8. Compute validation metrics, including a new per-catalytic-residue
   bond/angle deviation report classified as:
       all_fixed    — both endpoints in the fixed set (noted, never flagged)
       fixed_mobile — one fixed, one mobile (the interesting case)
       all_mobile   — both mobile (should be canonical post-idealization)
   Angles are auto-enumerated from the residue bond graph, and cross-residue
   peptide-bond neighbors are included (N[i+1]-C[i]-O[i], etc.) so the
   boundary between a fixed residue and its mobile neighbors is visible.

Defaults:
- Mobile region = whole protein (pass --mobile_radius to restrict).
- MPNN on (pass --skip_mpnn to disable).
- Single-thread-friendly but respects SLURM_CPUS_ON_NODE / OMP_NUM_THREADS.

Author: woodbuse (updated 2026-04-20)
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
import random
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
import time
from datetime import datetime


# ---------------------------------------------------------------------------
# Module-level caches
# ---------------------------------------------------------------------------

# A-X-A reference pose, keyed by one-letter code. Built once per type.
_IDEAL_REF_POSE_CACHE = {}


def _get_ideal_reference_residue(one_letter):
    """Return the middle residue of an A-X-A pose for given one-letter code.

    Cached per one-letter code to avoid rebuilding ~1400 poses for a full metric
    pass over a catalytic cluster.
    """
    if one_letter in _IDEAL_REF_POSE_CACHE:
        return _IDEAL_REF_POSE_CACHE[one_letter]
    try:
        ref_pose = pyr.pose_from_sequence("A" + one_letter + "A")
        ref_res = ref_pose.residue(2)
        _IDEAL_REF_POSE_CACHE[one_letter] = ref_res
        return ref_res
    except Exception:
        _IDEAL_REF_POSE_CACHE[one_letter] = None
        return None


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RFDiffusion3GeometryIdealizer:
    """Geometry idealizer for RFDiffusion3 scaffolds.

    Responsibilities are split into orthogonal steps; see run() for sequence.
    Internal fixed-residue representation is always
        self.fixed_residues: {pose_idx: frozenset(atom_names)}  # heavy atoms only
        self.fully_fixed_residues: set of pose_idx where every heavy atom is fixed
    "ALL" from JSON is expanded at map time against the residue type.
    """

    def __init__(self, pdb_path, json_path=None, json_dir=None, params=None,
                 mobile_radius=None,
                 cart_bonded_weight=4.0,
                 coord_cst_weight=100.0, coord_cst_stdev=0.01,
                 covalent_contact_threshold=2.5,
                 run_mpnn=True, mpnn_num_designs=10, mpnn_temperature=0.1,
                 mpnn_omit_aa="",
                 compute_ca_rmsd=True,
                 debug=False, no_metric_json_output=False,
                 compute_input_baseline_metrics=True,
                 include_input_in_mpnn_selection=True,
                 hbond_conserve_prob=0.0):
        self.pdb_path = pdb_path
        self.pdbname = os.path.basename(pdb_path).replace(".pdb", "")
        self.pdb_dir = os.path.dirname(os.path.abspath(pdb_path))

        self.json_path = self._determine_json_path(json_path, json_dir)

        self.params = params or []
        # None = whole protein mobile; a float = neighborhood radius in Å.
        self.mobile_radius = mobile_radius
        self.cart_bonded_weight = cart_bonded_weight
        self.coord_cst_weight = coord_cst_weight
        self.coord_cst_stdev = coord_cst_stdev
        self.covalent_contact_threshold = covalent_contact_threshold

        self.run_mpnn_flag = run_mpnn
        self.mpnn_num_designs = mpnn_num_designs
        self.mpnn_temperature = mpnn_temperature
        self.mpnn_omit_aa = mpnn_omit_aa or ""

        self.compute_ca_rmsd = compute_ca_rmsd
        self.debug = debug
        self.no_metric_json_output = no_metric_json_output
        self.compute_input_baseline_metrics = compute_input_baseline_metrics
        self._input_metrics_baseline = None  # populated once in run() before MPNN/relax
        self.include_input_in_mpnn_selection = include_input_in_mpnn_selection
        self.hbond_conserve_prob = float(hbond_conserve_prob or 0.0)

        # Populated during run()
        self.pose = None
        self.fixed_atoms_data_raw = None   # raw JSON dict (input-numbering)
        self.diffused_index_map = None
        self.fixed_residues = {}            # {pose_idx: frozenset(atom_names)}
        self.fully_fixed_residues = set()   # pose indices where atom_spec == "ALL"
        self.ligand_residues = []
        self.mobile_residues = []
        self.sfxn = None

        # Starting coords of fixed/ligand atoms — used for displacement metrics.
        # Replaces a full pose.clone() (saves ~2x pose memory).
        self.fixed_atom_initial_coords = {}  # {(pose_idx, atom_name): (x, y, z)}

        # Declared "covalent contacts" — fixed-fixed atom pairs < threshold.
        # Each entry: dict(res1, atom1, res2, atom2, dist, declared).
        self.declared_covalent_contacts = []

        self.original_pose = None  # only used if compute_ca_rmsd=True

        self.timings = {}
        self.start_time = None

        # MPNN integration details surfaced for metrics:
        self.mpnn_info = None

    # ------------------------------------------------------------------
    # JSON loading and normalization
    # ------------------------------------------------------------------

    def _determine_json_path(self, json_path, json_dir):
        """Priority: explicit --json > same dir as PDB > --corresponding_json_dir."""
        if json_path is not None:
            return json_path
        json_basename = self.pdbname + ".json"
        candidate = os.path.join(self.pdb_dir, json_basename)
        if os.path.exists(candidate):
            return candidate
        if json_dir is not None:
            candidate = os.path.join(json_dir, json_basename)
            if os.path.exists(candidate):
                return candidate
        return os.path.join(self.pdb_dir, json_basename)

    def load_json(self):
        """Load diffused_index_map + select_fixed_atoms from the RFdiffusion3 JSON.

        Stores the raw (input-numbering) dict in self.fixed_atoms_data_raw; the
        canonical form is only built after the pose is available (see
        map_json_residues_to_pose).
        """
        if not os.path.exists(self.json_path):
            print(f"WARNING: JSON file not found at {self.json_path}")
            print("Proceeding without fixed atom constraints from RFDiffusion3")
            self.fixed_atoms_data_raw = {}
            self.diffused_index_map = {}
            return

        with open(self.json_path, 'r') as f:
            data = json.load(f)

        self.diffused_index_map = self._extract_field(
            data, ["diffused_index_map"], "diffused_index_map")
        if self.diffused_index_map:
            print(f"✓ Loaded diffused_index_map from {self.json_path}")
            print(f"  Found {len(self.diffused_index_map)} residue mappings (input → output)")
        else:
            print(f"WARNING: 'diffused_index_map' not found in JSON")

        self.fixed_atoms_data_raw = self._extract_field(
            data,
            ["select_fixed_atoms", "specification.select_fixed_atoms"],
            "select_fixed_atoms")
        if self.fixed_atoms_data_raw:
            print(f"✓ Loaded select_fixed_atoms from {self.json_path}")
            print(f"  Found {len(self.fixed_atoms_data_raw)} fixed residue entries (input numbering)")
        else:
            print(f"WARNING: 'select_fixed_atoms' not found in JSON")
            print(f"  Top-level keys: {list(data.keys())}")

    def _extract_field(self, data, field_paths, field_name):
        """Walk candidate paths; return the first non-falsy nested value found."""
        for path in field_paths:
            try:
                value = data
                for key in path.split('.'):
                    value = value[key]
                if value:
                    print(f"  Found {field_name} at: {path}")
                    return value
            except (KeyError, TypeError):
                continue
        return {}

    @staticmethod
    def _normalize_atom_spec(raw, heavy_atom_names):
        """Canonicalize one JSON atom-spec value into a set of atom names.

        Accepts:
            - "ALL" / "all" / "All"          → every heavy atom
            - "NZ,CE" / "NZ, CE,"            → {"NZ", "CE"} (whitespace/empties trimmed)
            - ["NZ", "CE"]                   → {"NZ", "CE"}
            - "NZ CE" (space-separated)      → {"NZ", "CE"} (fallback)
        Returns a frozenset of atom name strings.
        """
        if isinstance(raw, (list, tuple, set, frozenset)):
            tokens = [str(x).strip() for x in raw]
        elif isinstance(raw, str):
            if raw.strip().upper() == "ALL":
                return frozenset(heavy_atom_names)
            sep = "," if "," in raw else None
            tokens = [t.strip() for t in raw.split(sep)]
        else:
            return frozenset()
        return frozenset(t for t in tokens if t)

    def map_json_residues_to_pose(self, pose):
        """Map JSON residue IDs to pose indices and expand atom specs.

        Builds:
            self.fixed_residues[pose_idx] = frozenset(atom_names)
            self.fully_fixed_residues = {pose_idx where spec == ALL}
        """
        if not self.diffused_index_map:
            print("\nWARNING: No diffused_index_map — assuming no renumbering")

        fixed_residues = {}
        fully_fixed = set()

        print("Mapping fixed residues (INPUT → OUTPUT → Pose index):")
        for input_res_id, atom_spec_raw in self.fixed_atoms_data_raw.items():
            # Step 1: input → output via diffused_index_map (fallback: identity).
            if self.diffused_index_map and input_res_id in self.diffused_index_map:
                output_res_id = self.diffused_index_map[input_res_id]
            else:
                if self.diffused_index_map:
                    print(f"  WARNING: {input_res_id} not in diffused_index_map, assuming identity")
                output_res_id = input_res_id

            # Step 2: parse chain + resnum.
            try:
                chain = output_res_id[0]
                pdb_resnum = int(output_res_id[1:])
            except (ValueError, IndexError):
                print(f"  WARNING: Unparseable residue id '{output_res_id}', skipping")
                continue

            # Step 3: PDB → pose index.
            pose_idx = pose.pdb_info().pdb2pose(chain, pdb_resnum)
            if pose_idx == 0:
                print(f"  WARNING: Could not map {output_res_id} to pose index")
                continue

            residue = pose.residue(pose_idx)
            heavy_atom_names = [residue.atom_name(i).strip()
                                for i in range(1, residue.natoms() + 1)
                                if not residue.atom_is_hydrogen(i)]
            atom_set = self._normalize_atom_spec(atom_spec_raw, heavy_atom_names)

            # Warn about any requested atoms the residue doesn't have.
            missing = atom_set - set(heavy_atom_names)
            if missing:
                print(f"  WARNING: {output_res_id} missing atoms {sorted(missing)}")
                atom_set = atom_set & set(heavy_atom_names)

            fixed_residues[pose_idx] = atom_set
            # "ALL" becomes fully-fixed iff it matches every heavy atom.
            if len(atom_set) == len(heavy_atom_names):
                fully_fixed.add(pose_idx)

            fully_tag = " [FULL]" if pose_idx in fully_fixed else ""
            print(f"  {input_res_id} → {output_res_id} → pose {pose_idx}{fully_tag}: "
                  f"{len(atom_set)} atoms ({sorted(atom_set)[:6]}{'...' if len(atom_set) > 6 else ''})")

        print(f"✓ Mapped {len(fixed_residues)} residues from JSON to pose "
              f"({len(fully_fixed)} fully fixed)")
        return fixed_residues, fully_fixed

    def identify_ligands(self, pose):
        """All residues flagged is_ligand() (typically HETATM, loaded via params)."""
        ligand_res = []
        for i in range(1, pose.size() + 1):
            if pose.residue(i).is_ligand():
                ligand_res.append(i)
                print(f"  Ligand at pose {i}: {pose.residue(i).name()}")
        print(f"✓ Found {len(ligand_res)} ligand residue(s)")
        return ligand_res

    # ------------------------------------------------------------------
    # Declared covalent contacts
    # ------------------------------------------------------------------

    def _collect_fixed_atoms_with_coords(self, pose, ligand_residues):
        """Yield (pose_idx, atom_name, xyz) for every fixed/ligand heavy atom."""
        out = []
        # Protein fixed atoms from JSON.
        for res_idx, atoms in self.fixed_residues.items():
            residue = pose.residue(res_idx)
            for atom_name in atoms:
                if not residue.has(atom_name):
                    continue
                xyz = residue.xyz(atom_name)
                out.append((res_idx, atom_name, np.array([xyz.x, xyz.y, xyz.z])))
        # Ligand heavy atoms (all ligand atoms are effectively fixed).
        for lig_idx in ligand_residues:
            residue = pose.residue(lig_idx)
            for i in range(1, residue.natoms() + 1):
                if residue.atom_is_hydrogen(i):
                    continue
                atom_name = residue.atom_name(i).strip()
                xyz = residue.xyz(atom_name)
                out.append((lig_idx, atom_name, np.array([xyz.x, xyz.y, xyz.z])))
        return out

    def _is_polymer_backbone_pair(self, pose, r1, a1, r2, a2):
        """True when (r1,a1)-(r2,a2) is a backbone-backbone pair between adjacent
        same-chain polymer-bonded protein residues.

        These pairs (C[i]-N[i+1], O[i]-N[i+1], CA[i]-N[i+1], C[i]-CA[i+1], ...)
        are geometrically close because of the peptide bond itself, not because
        of any unusual chemistry. We include them in the declared-contacts
        report for transparency, but we do NOT subtract their fa_rep
        contribution — any distortion of the polymer backbone (too-short or
        otherwise crushed peptide geometry between two fully-fixed residues)
        should still surface as a clash.
        """
        if abs(r1 - r2) != 1:
            return False
        lo, hi = (r1, r2) if r1 < r2 else (r2, r1)
        if not (pose.residue(lo).is_protein() and pose.residue(hi).is_protein()):
            return False
        try:
            if pose.pdb_info().chain(lo) != pose.pdb_info().chain(hi):
                return False
        except Exception:
            pass
        try:
            if not pose.conformation().residues_are_bonded(lo, hi):
                return False
        except Exception:
            pass
        backbone = {"N", "CA", "C", "O"}
        return a1 in backbone and a2 in backbone

    def detect_covalent_contacts(self, pose, ligand_residues):
        """Find fixed-fixed atom pairs closer than the threshold.

        Every close fixed-fixed pair is reported for transparency. Each is
        tagged `polymer_backbone_pair=True` if it is a backbone-backbone pair
        between adjacent same-chain polymer-bonded residues; those tagged
        pairs are excluded from fa_rep subtraction downstream so polymer
        distortion remains visible as a clash. Non-polymer pairs (sidechain
        PTMs, metal-ligand bonds, tight salt bridges) have their pair fa_rep
        subtracted from clash counts downstream.
        """
        threshold = self.covalent_contact_threshold
        print(f"\nScanning for declared covalent contacts (< {threshold:.2f} Å between fixed atoms)...")

        all_fixed = self._collect_fixed_atoms_with_coords(pose, ligand_residues)
        contacts = []
        n = len(all_fixed)
        for i in range(n):
            ri, ai, xi = all_fixed[i]
            for j in range(i + 1, n):
                rj, aj, xj = all_fixed[j]
                if ri == rj:
                    continue
                d = float(np.linalg.norm(xi - xj))
                if d >= threshold:
                    continue
                polymer_bb = self._is_polymer_backbone_pair(pose, ri, ai, rj, aj)
                contacts.append({
                    'res1': int(ri), 'atom1': ai,
                    'res2': int(rj), 'atom2': aj,
                    'distance': d, 'declared': False,
                    'polymer_backbone_pair': bool(polymer_bb),
                })

        if contacts:
            n_poly = sum(1 for c in contacts if c['polymer_backbone_pair'])
            n_chem = len(contacts) - n_poly
            print(f"  Found {len(contacts)} close fixed-fixed contact(s): "
                  f"{n_chem} chemistry-relevant (subtracted from clashes), "
                  f"{n_poly} polymer-backbone (listed only, NOT subtracted).")
            for c in contacts:
                r1 = pose.pdb_info().chain(c['res1']) + str(pose.pdb_info().number(c['res1']))
                r2 = pose.pdb_info().chain(c['res2']) + str(pose.pdb_info().number(c['res2']))
                tag = " [polymer-backbone]" if c['polymer_backbone_pair'] else ""
                print(f"    {r1}.{c['atom1']}  <-->  {r2}.{c['atom2']}   "
                      f"({c['distance']:.3f} Å)  "
                      f"[{pose.residue(c['res1']).name3()}/{pose.residue(c['res2']).name3()}]{tag}")
        else:
            print("  None found.")
        return contacts

    def declare_covalent_bonds(self, pose, contacts):
        """Best-effort declare_chemical_bond on each contact.

        Most fixed atoms on standard protein/ligand residues have no CONNECT
        records in their ResidueType, so the call raises. We silently record
        the failure — the metric-level exclusion (in calculate_validation_metrics)
        is what actually keeps these pairs out of clash counts. Rosetta's C++
        error printouts are redirected to a tempfile so they don't spam stderr.
        """
        if not contacts:
            return
        print("\nAttempting declare_chemical_bond for each contact...")
        n_success = 0
        import io as _io
        # Redirect stderr of C++ calls by dup'ing fd 2 to devnull.
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        saved_fd = os.dup(2)
        try:
            os.dup2(devnull_fd, 2)
            for c in contacts:
                try:
                    pose.conformation().declare_chemical_bond(
                        c['res1'], c['atom1'], c['res2'], c['atom2'])
                    c['declared'] = True
                    n_success += 1
                except Exception as e:
                    c['declared'] = False
                    c['declare_error'] = str(e).splitlines()[-1][:120] if str(e) else 'UtilityExit (no CONNECT record)'
        finally:
            os.dup2(saved_fd, 2)
            os.close(saved_fd)
            os.close(devnull_fd)
        print(f"  ✓ Declared {n_success}/{len(contacts)} bonds "
              f"(failures recorded; metric-level exclusion still applies)")

    # ------------------------------------------------------------------
    # Coordinate constraints
    # ------------------------------------------------------------------

    def add_coordinate_constraints(self, pose, fixed_residues, ligand_residues):
        """Pin fixed atoms (and all ligand heavy atoms) in absolute space.

        Populates self.fixed_atom_initial_coords for later displacement metrics,
        so we don't need to clone the whole pose.
        """
        print("\nAdding coordinate constraints to fix atoms in absolute space...")

        constraint_count = 0
        self.fixed_atom_initial_coords = {}

        # Protein fixed atoms from JSON.
        for res_idx, atom_set in fixed_residues.items():
            residue = pose.residue(res_idx)
            pdb_info = f"{pose.pdb_info().chain(res_idx)}{pose.pdb_info().number(res_idx)}"
            fully = res_idx in self.fully_fixed_residues
            tag = "[ALL]" if fully else f"[{len(atom_set)} atoms]"
            print(f"  {pdb_info} ({residue.name3()}) {tag}")

            for atom_name in atom_set:
                if not residue.has(atom_name):
                    continue
                atom_idx = residue.atom_index(atom_name)
                xyz = residue.xyz(atom_name)
                self.fixed_atom_initial_coords[(res_idx, atom_name)] = (xyz.x, xyz.y, xyz.z)

                func = HarmonicFunc(0.0, self.coord_cst_stdev)
                cst = CoordinateConstraint(
                    AtomID(atom_idx, res_idx),
                    AtomID(1, 1),
                    xyz,
                    func,
                )
                pose.add_constraint(cst)
                constraint_count += 1

        # All ligand heavy atoms.
        ligand_constraint_count = 0
        print(f"\n  Constraining ligand(s):")
        for lig_idx in ligand_residues:
            residue = pose.residue(lig_idx)
            pdb_info = f"{pose.pdb_info().chain(lig_idx)}{pose.pdb_info().number(lig_idx)}"
            n_heavy = sum(1 for i in range(1, residue.natoms() + 1)
                          if not residue.atom_is_hydrogen(i))
            print(f"  {pdb_info} ({residue.name3()}): {n_heavy} heavy atoms")

            for i in range(1, residue.natoms() + 1):
                if residue.atom_is_hydrogen(i):
                    continue
                atom_name = residue.atom_name(i).strip()
                xyz = residue.xyz(atom_name)
                self.fixed_atom_initial_coords[(lig_idx, atom_name)] = (xyz.x, xyz.y, xyz.z)

                func = HarmonicFunc(0.0, self.coord_cst_stdev)
                cst = CoordinateConstraint(
                    AtomID(i, lig_idx), AtomID(1, 1), xyz, func)
                pose.add_constraint(cst)
                ligand_constraint_count += 1

        print(f"\n  ✓ Summary:")
        print(f"    - Fixed residues: {len(fixed_residues)} residues, {constraint_count} atom constraints")
        print(f"    - Ligands: {len(ligand_residues)} ligands, {ligand_constraint_count} atom constraints")
        print(f"    - TOTAL: {constraint_count + ligand_constraint_count} coordinate constraints")

    # ------------------------------------------------------------------
    # Mobile region & MoveMap
    # ------------------------------------------------------------------

    def define_mobile_region(self, pose, fixed_residues, ligand_residues, radius=None):
        """If radius is None, return all residues (whole protein mobile).

        Otherwise use NeighborhoodResidueSelector around the union of fixed
        residues + ligand residues.
        """
        if radius is None:
            mobile_res = list(range(1, pose.size() + 1))
            print(f"\nMobile region: whole protein ({len(mobile_res)} residues)")
            return mobile_res

        print(f"\nDefining mobile region ({radius}Å around fixed/ligand residues)...")
        focus_residues = list(set(list(fixed_residues.keys()) + ligand_residues))
        if not focus_residues:
            print("  WARNING: No focus residues — using all residues as mobile")
            return list(range(1, pose.size() + 1))

        focus_selector = ResidueIndexSelector(','.join(map(str, focus_residues)))
        neighborhood_selector = NeighborhoodResidueSelector(
            focus_selector, distance=radius, include_focus_in_subset=True)
        selected = neighborhood_selector.apply(pose)
        mobile_res = [i for i in range(1, pose.size() + 1) if selected[i]]
        print(f"✓ Mobile region: {len(mobile_res)} residues")
        return mobile_res

    def build_movemap(self, pose, mobile_residues, ligand_residues, fully_fixed_residues):
        """Enable bb/chi only for residues that should move.

        A residue is enabled if it is in the mobile set AND it is not a ligand
        AND it is not fully fixed. Partially-fixed residues remain enabled
        because their mobile atoms still need DOFs to idealize — coord csts
        handle the fixed-atom subset.
        """
        print("\nConfiguring MoveMap...")
        mm = pyrosetta.rosetta.core.kinematics.MoveMap()
        mm.set_bb(False)
        mm.set_chi(False)
        mm.set_jump(False)

        mobile_set = set(mobile_residues)
        lig_set = set(ligand_residues)
        ff_set = set(fully_fixed_residues)

        n_enabled = 0
        for res_idx in range(1, pose.size() + 1):
            if res_idx in lig_set or res_idx in ff_set:
                continue
            if res_idx in mobile_set:
                mm.set_bb(res_idx, True)
                mm.set_chi(res_idx, True)
                n_enabled += 1

        # Belt-and-suspenders for ligands (coord csts also pin them).
        for lig_idx in lig_set:
            mm.set_bb(lig_idx, False)
            mm.set_chi(lig_idx, False)

        print(f"✓ MoveMap: {n_enabled} residues with bb+chi enabled; "
              f"{len(ff_set)} fully-fixed frozen; {len(lig_set)} ligand(s) frozen")
        return mm

    # ------------------------------------------------------------------
    # Scorefunction
    # ------------------------------------------------------------------

    def setup_scorefunction(self):
        from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory
        print("\nSetting up cartesian scorefunction...")
        sfxn = None
        for weights_name in ["ref2015_cart", "ref2015", "score12"]:
            try:
                sfxn = ScoreFunctionFactory.create_score_function(weights_name)
                print(f"  Loaded scorefunction: {weights_name}")
                break
            except Exception:
                pass
        if sfxn is None:
            print("  Using default scorefunction")
            sfxn = pyr.get_fa_scorefxn()

        sfxn.set_weight(ScoreType.cart_bonded, self.cart_bonded_weight)
        print(f"  Set cart_bonded weight: {self.cart_bonded_weight}")
        # pro_close and cart_bonded can conflict in some Rosetta builds.
        sfxn.set_weight(ScoreType.pro_close, 0.0)
        sfxn.set_weight(ScoreType.coordinate_constraint, self.coord_cst_weight)
        print(f"  Set coordinate_constraint weight: {self.coord_cst_weight}")

        if not sfxn.has_nonzero_weight(ScoreType.cart_bonded):
            print("  ERROR: Scorefunction does not have cart_bonded weight!")
        else:
            print(f"  ✓ Cartesian scoring enabled (cart_bonded = {sfxn.get_weight(ScoreType.cart_bonded)})")
        return sfxn

    # ------------------------------------------------------------------
    # Chain break detection (chain-aware + polymer-aware)
    # ------------------------------------------------------------------

    def detect_chain_breaks(self, pose):
        """Flag i/i+1 pairs that are same-chain polymer-bonded but distant.

        The original 2.0 Å distance cutoff alone mislabels chain boundaries as
        breaks. We now require same chain ID AND is_polymer_bonded(i, i+1)
        before treating a large C-N distance as a break.
        """
        breaks = []
        pdb_info = pose.pdb_info()
        for i in range(1, pose.size()):
            res_i = pose.residue(i)
            res_j = pose.residue(i + 1)
            if not res_i.is_protein() or not res_j.is_protein():
                continue
            try:
                if pdb_info and pdb_info.chain(i) != pdb_info.chain(i + 1):
                    continue
            except Exception:
                pass
            try:
                polymer_bonded = pose.conformation().residues_are_bonded(i, i + 1)
            except Exception:
                try:
                    polymer_bonded = pose.conformation().is_polymer_bonded(i, i + 1)
                except Exception:
                    polymer_bonded = True  # optimistic fallback
            if not polymer_bonded:
                continue
            try:
                c_xyz = res_i.xyz("C")
                n_xyz = res_j.xyz("N")
                dist = c_xyz.distance(n_xyz)
                if dist > 2.0:
                    breaks.append(i)
                    print(f"  Chain break detected: {i}-{i+1} (C-N distance: {dist:.2f} Å)")
            except Exception:
                continue
        return breaks

    # ------------------------------------------------------------------
    # HIS tautomer handling
    # ------------------------------------------------------------------

    @staticmethod
    def _his_tautomer(residue):
        """Return 'HIS' (HE2) or 'HIS_D' (HD1) based on which N has a proton.

        Falls back to residue.name3() if no H atoms present.
        """
        has_hd1 = residue.has("HD1")
        has_he2 = residue.has("HE2")
        if has_hd1 and not has_he2:
            return "HIS_D"
        if has_he2 and not has_hd1:
            return "HIS"
        # Both or neither — trust the current residue name.
        return residue.name3()

    def record_his_tautomers(self, pose, residues_to_record):
        """Capture tautomer state for each HIS-like residue in the set."""
        tautomers = {}
        for res_idx in residues_to_record:
            if res_idx > pose.size():
                continue
            residue = pose.residue(res_idx)
            if residue.name3() not in ("HIS", "HIS_D"):
                continue
            tautomers[res_idx] = self._his_tautomer(residue)
        return tautomers

    def restore_his_tautomers(self, pose, tautomer_map):
        """Apply MutateResidue to restore tautomers (preserves atom coords)."""
        from pyrosetta.rosetta.protocols.simple_moves import MutateResidue
        n_fixed = 0
        for res_idx, target_name in tautomer_map.items():
            if res_idx > pose.size():
                continue
            residue = pose.residue(res_idx)
            if residue.name3() == target_name:
                continue
            try:
                mut = MutateResidue()
                mut.set_target(res_idx)
                mut.set_res_name(target_name)
                mut.set_preserve_atom_coords(True)
                mut.apply(pose)
                n_fixed += 1
            except Exception as e:
                print(f"  WARNING: failed to restore HIS tautomer at {res_idx}: {e}")
        if n_fixed:
            print(f"  Restored {n_fixed} HIS tautomer(s) post-MPNN")

    # ------------------------------------------------------------------
    # MPNN pre-design
    # ------------------------------------------------------------------

    def _snapshot_protected_coords(self, pose, extra_residues=None):
        """Capture heavy-atom xyz for all fixed atoms + all ligand atoms.

        Returned dict maps {(pose_idx, atom_name): (x, y, z)}. Use with
        _restore_protected_coords to undo xyz drift that MPNN's sidechain
        packer introduces on 'do_not_repack' residues.
        """
        snap = {}
        for res_idx, atom_set in self.fixed_residues.items():
            if res_idx > pose.size():
                continue
            residue = pose.residue(res_idx)
            for atom_name in atom_set:
                if not residue.has(atom_name):
                    continue
                xyz = residue.xyz(atom_name)
                snap[(res_idx, atom_name)] = (xyz.x, xyz.y, xyz.z)
        for lig_idx in self.ligand_residues:
            if lig_idx > pose.size():
                continue
            residue = pose.residue(lig_idx)
            for i in range(1, residue.natoms() + 1):
                if residue.atom_is_hydrogen(i):
                    continue
                atom_name = residue.atom_name(i).strip()
                xyz = residue.xyz(atom_name)
                snap[(lig_idx, atom_name)] = (xyz.x, xyz.y, xyz.z)
        if extra_residues:
            for res_idx in extra_residues:
                if res_idx > pose.size():
                    continue
                residue = pose.residue(res_idx)
                for i in range(1, residue.natoms() + 1):
                    if residue.atom_is_hydrogen(i):
                        continue
                    atom_name = residue.atom_name(i).strip()
                    key = (res_idx, atom_name)
                    if key not in snap:
                        snap[key] = tuple(residue.xyz(atom_name))[0:3] if False else (
                            residue.xyz(atom_name).x,
                            residue.xyz(atom_name).y,
                            residue.xyz(atom_name).z,
                        )
        return snap

    def _restore_protected_coords(self, pose, snapshot):
        """Set xyz back to the snapshotted values. Returns max drift seen."""
        max_drift = 0.0
        for (res_idx, atom_name), (x, y, z) in snapshot.items():
            if res_idx > pose.size():
                continue
            residue = pose.residue(res_idx)
            if not residue.has(atom_name):
                continue
            current = residue.xyz(atom_name)
            d = ((current.x - x) ** 2 + (current.y - y) ** 2 + (current.z - z) ** 2) ** 0.5
            if d > max_drift:
                max_drift = d
            atom_idx = residue.atom_index(atom_name)
            target = xyzVector_double_t(x, y, z)
            pose.set_xyz(AtomID(atom_idx, res_idx), target)
        return max_drift

    def run_mpnn_predesign(self, pose):
        """Run ligandMPNN on the input pose; return best candidate pose + info.

        Fixed residues (do_not_repack_positions): catalytic fixed-atom residues
        from JSON + all REMARK 666 catres + ligand(s). Design residues:
        everything else that is protein. Returns (best_pose, info_dict).
        If MPNN is unavailable, returns (pose, {'skipped': <reason>}).

        NOTE: MPNN's internal sidechain packer (pack_sc=True) rebuilds xyz from
        backbone+chi tensors, which drifts fixed-atom coordinates even when
        the residue is in do_not_repack_positions. We snapshot the fixed/ligand
        atom positions BEFORE MPNN and restore them on each returned candidate
        so that the subsequent coord constraints pin the RFdiffusion3-original
        positions, not the MPNN-rebuilt ones.
        """
        info = {'requested_designs': self.mpnn_num_designs,
                'temperature': self.mpnn_temperature,
                'omit_aa': self.mpnn_omit_aa}
        # FastMPNNdesign is bundled as a submodule; $FASTMPNNDESIGN_DIR overrides.
        # Initialize it with:
        #     git submodule update --init --recursive Software/fastmpnndesign
        _repo_root = Path(__file__).resolve().parents[2]
        _candidate_roots = [r for r in (
            os.environ.get("FASTMPNNDESIGN_DIR"),
            str(_repo_root / "Software"),
        ) if r]
        _FMD = None
        _design_utils = None
        for root in _candidate_roots:
            fmd_dir = f"{root}/fastmpnndesign"
            utils_dir = f"{root}/fastmpnndesign/utils"
            if not os.path.isdir(fmd_dir):
                continue
            if fmd_dir not in sys.path:
                sys.path.insert(0, fmd_dir)
            if os.path.isdir(utils_dir) and utils_dir not in sys.path:
                sys.path.insert(0, utils_dir)
            try:
                import FastMPNNdesign as _FMD  # noqa: F401
            except Exception as e:
                info['skipped'] = f"MPNN library found at {fmd_dir} but import failed: {e}"
                continue
            try:
                import design_utils as _design_utils  # noqa: F401
            except Exception:
                _design_utils = None
            info['mpnn_library_path'] = _FMD.__file__
            break
        if _FMD is None:
            if 'skipped' not in info:
                info['skipped'] = ("MPNN library unavailable: FastMPNNdesign not found at "
                                   + " or ".join(_candidate_roots))
            print(f"\n⚠ MPNN skipped: {info['skipped']}")
            return pose, info

        # Union: JSON fixed residues + REMARK 666 catres (if parseable).
        fixed_set = set(self.fixed_residues.keys())
        catres_from_remark = {}
        if _design_utils is not None:
            try:
                catres_from_remark = _design_utils.get_matcher_residues(pose) or {}
                for k in catres_from_remark.keys():
                    fixed_set.add(int(k))
            except Exception as e:
                print(f"  (REMARK 666 parse failed: {e}; proceeding with JSON-only fixed set)")

        # Optionally expand the fixed set to preserve designable residues whose
        # SIDECHAIN hbonds to the ligand or to a catalytic residue. Each
        # candidate is promoted with probability `self.hbond_conserve_prob`,
        # giving MPNN diversity across multiple runs (different RNG draws per
        # subprocess) while biasing toward conserving observed active-site
        # sidechain interactions. Disabled by default (p=0.0).
        if self.hbond_conserve_prob > 0.0:
            pose.update_residue_neighbors()
            hbond_set = pyr.rosetta.core.scoring.hbonds.HBondSet()
            pyr.rosetta.core.scoring.hbonds.fill_hbond_set(pose, False, hbond_set)
            protected_set = fixed_set | set(self.ligand_residues)

            candidates = set()
            for h in range(1, hbond_set.nhbonds() + 1):
                hb = hbond_set.hbond(h)
                don_res, acc_res = hb.don_res(), hb.acc_res()
                don_is_prot, acc_is_prot = (don_res in protected_set), (acc_res in protected_set)
                # We want exactly one side protected (the catres/ligand side)
                # and the other side designable. Skip both-protected (intra-motif
                # hbond) and neither-protected (irrelevant).
                if don_is_prot == acc_is_prot:
                    continue
                if don_is_prot:
                    des_res, des_atm = acc_res, hb.acc_atm()
                else:
                    des_res = don_res
                    # Donor hbond atom is the H; we care about the heavy parent.
                    des_atm = pose.residue(don_res).atom_base(hb.don_hatm())
                res = pose.residue(des_res)
                if not res.is_protein():
                    continue
                if res.atom_is_backbone(des_atm):
                    continue   # backbone-only contribution — ignore per spec
                candidates.add(int(des_res))

            promoted = {r for r in candidates if random.random() < self.hbond_conserve_prob}
            fixed_set |= promoted
            info['hbond_conserve_prob'] = self.hbond_conserve_prob
            info['hbond_conserve_candidates'] = sorted(candidates)
            info['hbond_conserve_promoted'] = sorted(promoted)
            print(f"  hbond conservation (p={self.hbond_conserve_prob}): "
                  f"{len(promoted)}/{len(candidates)} designable-sidechain hbond residues "
                  f"to catres/ligand promoted into the MPNN fixed set "
                  f"(promoted={sorted(promoted)}).")

        do_not_repack = sorted(fixed_set | set(self.ligand_residues))
        design_positions = [
            i for i in range(1, pose.size() + 1)
            if i not in fixed_set and i not in self.ligand_residues
            and pose.residue(i).is_protein()
        ]
        info['num_fixed'] = len(do_not_repack)
        info['num_design'] = len(design_positions)
        info['catres_from_remark'] = sorted(int(k) for k in catres_from_remark.keys())

        if not design_positions:
            info['skipped'] = "No design positions (nothing to redesign)"
            print(f"\n⚠ MPNN skipped: {info['skipped']}")
            return pose, info

        # Capture HIS tautomers for all fixed residues before MPNN may disturb them.
        tautomer_map = self.record_his_tautomers(pose, fixed_set)
        info['his_tautomers_preserved'] = {int(k): v for k, v in tautomer_map.items()}

        # Snapshot fixed/ligand atom xyz so we can undo MPNN's sidechain-packer drift.
        protected_coords_snapshot = self._snapshot_protected_coords(pose)
        info['num_protected_atoms'] = len(protected_coords_snapshot)

        # Build MPNN scorefunction (cartesian-aware clone of our main sfxn).
        sfx_mpnn = self.sfxn.clone() if self.sfxn is not None else pyr.get_fa_scorefxn()
        sfx_mpnn.set_weight(ScoreType.cart_bonded, self.cart_bonded_weight)
        sfx_mpnn.set_weight(ScoreType.pro_close, 0.0)

        # Protocol: generate N sequences at T, then a single repack pass so
        # MPNN's output rotamers settle enough for cheap scoring.
        protocol = (
            f"scale:coordinate_constraint 0.0\n"
            f"scale:fa_rep 0.5\n"
            f"mpnn {self.mpnn_temperature} {self.mpnn_num_designs}\n"
            f"repack\n"
        )
        print(f"\nRunning ligandMPNN: T={self.mpnn_temperature}, N={self.mpnn_num_designs}, "
              f"omit_AA='{self.mpnn_omit_aa}', fixed={len(do_not_repack)}, "
              f"design={len(design_positions)}")

        try:
            fmd = _FMD.FastMPNNdesign(
                model_type="ligand_mpnn",
                params=self.params,
                scorefxn=sfx_mpnn,
                script_file=protocol,
                cartesian=True,
                design_positions=design_positions,
                do_not_repack_positions=do_not_repack,
                cst_io=None,
                omit_AA=self.mpnn_omit_aa,  # empty string = allow all 20
                mpnn_pack_sc=True,
                ligand_mpnn_use_side_chain_context=True,
            )
            poses = fmd.apply(pose.clone())
        except Exception as e:
            info['skipped'] = f"MPNN runtime error: {type(e).__name__}: {e}"
            print(f"\n⚠ MPNN failed at runtime: {info['skipped']}")
            print(f"  Falling back to input sequence.")
            return pose, info

        if not poses:
            info['skipped'] = "MPNN returned 0 poses"
            print(f"\n⚠ {info['skipped']}")
            return pose, info

        print(f"  Got {len(poses)} candidate poses; restoring protected xyz and scoring...")

        # Score each candidate and pick best. Restore fixed/ligand xyz (MPNN's
        # pack_sc=True rebuilds them from backbone+chi tensors and introduces
        # drift even for do_not_repack residues) and restore HIS tautomers.
        candidate_scores = []
        max_drift_seen = 0.0
        for idx, cand in enumerate(poses):
            drift = self._restore_protected_coords(cand, protected_coords_snapshot)
            if drift > max_drift_seen:
                max_drift_seen = drift
            self.restore_his_tautomers(cand, tautomer_map)
            score_dict = self._score_mpnn_candidate(
                cand, fixed_set, self.ligand_residues, catres_from_remark.keys())
            score_dict['index'] = idx
            score_dict['sequence'] = cand.sequence()
            candidate_scores.append(score_dict)
            print(f"    cand {idx:2d}: composite={score_dict['composite']:.2f}   "
                  f"(catres_fa_rep={score_dict['fa_rep_catalytic']:.2f}, "
                  f"lig_if_fa_rep={score_dict['fa_rep_ligand_interface']:.2f}, "
                  f"catres_cart={score_dict['cart_bonded_catalytic']:.2f}, "
                  f"sc_hbonds_to_active_site={score_dict['hbond_bonus_count']})")

        # Also score the original input pose with the same composite so we
        # can reject MPNN entirely when none of its N candidates improve on
        # what we started with. Scored as-is (no extra repack pass) — this
        # mildly favors the MPNN candidates (which did get a repack pass in
        # the protocol), so a win by the input is a strong signal that the
        # original sequence/rotamers were already in a good arrangement.
        input_won = False
        if self.include_input_in_mpnn_selection:
            input_score = self._score_mpnn_candidate(
                pose, fixed_set, self.ligand_residues, catres_from_remark.keys())
            input_score['index'] = 'input'
            input_score['sequence'] = pose.sequence()
            candidate_scores.append(input_score)
            info['input_pose_composite'] = input_score['composite']
            print(f"    input    : composite={input_score['composite']:.2f}   "
                  f"(catres_fa_rep={input_score['fa_rep_catalytic']:.2f}, "
                  f"lig_if_fa_rep={input_score['fa_rep_ligand_interface']:.2f}, "
                  f"catres_cart={input_score['cart_bonded_catalytic']:.2f}, "
                  f"sc_hbonds_to_active_site={input_score['hbond_bonus_count']})")

        best = min(candidate_scores, key=lambda d: d['composite'])
        info['candidate_scores'] = candidate_scores
        info['selected_index'] = best['index']
        info['selected_composite'] = best['composite']
        info['input_sequence_selected'] = (best['index'] == 'input')
        info['max_mpnn_protected_drift_A'] = float(max_drift_seen)

        if info['input_sequence_selected']:
            input_won = True
            n_mpnn = len(candidate_scores) - 1
            best_mpnn = min((c for c in candidate_scores if c['index'] != 'input'),
                            key=lambda d: d['composite'], default=None)
            mpnn_best_str = (f"; best MPNN was composite={best_mpnn['composite']:.2f}"
                             if best_mpnn is not None else "")
            print(f"\n  ✓ Input sequence retained "
                  f"(composite={best['composite']:.2f}{mpnn_best_str}) — "
                  f"none of {n_mpnn} MPNN candidates beat the original.")
            return pose.clone(), info

        print(f"\n  ✓ Selected candidate {best['index']} (composite={best['composite']:.2f})")
        print(f"  Max MPNN protected-atom drift restored: {max_drift_seen:.4f} Å")

        return poses[best['index']], info

    def _score_mpnn_candidate(self, pose, fixed_res_set, ligand_residues, catres_remark_keys):
        """Cheap per-candidate score for selection.

        Composite = fa_rep_catalytic + fa_rep_ligand_interface + cart_bonded_catalytic
                    - 1.0 * hbond_bonus_count
        Lower = better.
        """
        sfx = self.sfxn.clone() if self.sfxn is not None else pyr.get_fa_scorefxn()
        sfx(pose)  # fills energies

        cat_set = set(fixed_res_set) | set(int(k) for k in catres_remark_keys)
        lig_set = set(ligand_residues)

        fa_rep_cat = 0.0
        cart_bonded_cat = 0.0
        for r in cat_set:
            if r > pose.size():
                continue
            try:
                fa_rep_cat += float(pose.energies().residue_total_energies(r)[ScoreType.fa_rep])
                cart_bonded_cat += float(pose.energies().residue_total_energies(r)[ScoreType.cart_bonded])
            except Exception:
                pass

        # Ligand-interface fa_rep: sum fa_rep of residues within 5 Å of any ligand atom.
        interface_residues = set()
        for lig_idx in lig_set:
            lig_res = pose.residue(lig_idx)
            lig_coords = [lig_res.xyz(i) for i in range(1, lig_res.natoms() + 1)
                          if not lig_res.atom_is_hydrogen(i)]
            for r in range(1, pose.size() + 1):
                if r in lig_set:
                    continue
                res_r = pose.residue(r)
                if not res_r.is_protein():
                    continue
                # use nbr_atom for quick screen
                try:
                    nbr_xyz = res_r.nbr_atom_xyz()
                    if any(nbr_xyz.distance(lc) < 8.0 for lc in lig_coords):
                        # detailed distance check
                        for i in range(1, res_r.natoms() + 1):
                            if res_r.atom_is_hydrogen(i):
                                continue
                            axyz = res_r.xyz(i)
                            if any(axyz.distance(lc) < 5.0 for lc in lig_coords):
                                interface_residues.add(r)
                                break
                except Exception:
                    continue

        fa_rep_if = 0.0
        for r in interface_residues:
            try:
                fa_rep_if += float(pose.energies().residue_total_energies(r)[ScoreType.fa_rep])
            except Exception:
                pass

        hbond_bonus = self._count_hbonds_to_active_site(pose, cat_set, lig_set)

        composite = fa_rep_cat + fa_rep_if + cart_bonded_cat - 1.0 * hbond_bonus

        return {
            'composite': float(composite),
            'fa_rep_catalytic': float(fa_rep_cat),
            'fa_rep_ligand_interface': float(fa_rep_if),
            'cart_bonded_catalytic': float(cart_bonded_cat),
            'hbond_bonus_count': int(hbond_bonus),
            'num_interface_residues': len(interface_residues),
        }

    def _count_hbonds_to_active_site(self, pose, cat_set, lig_set):
        """Count sidechain hbonds where the donor is NOT a catalytic residue and
        the acceptor IS in cat_set or lig_set.

        Covers: (design_residue sidechain) → (catalytic or ligand), and
                (ligand) → (catalytic). Excludes catalytic→anything (we don't
        reward the catalytic residues themselves — they were frozen).
        """
        from pyrosetta.rosetta.core.scoring.hbonds import HBondSet, fill_hbond_set
        hb_set = HBondSet()
        try:
            fill_hbond_set(pose, False, hb_set)
        except Exception:
            return 0
        count = 0
        for i in range(1, hb_set.nhbonds() + 1):
            hb = hb_set.hbond(i)
            don_res = hb.don_res()
            acc_res = hb.acc_res()
            # Donor must not be a catalytic residue ("not from catalytic").
            if don_res in cat_set:
                continue
            # Target side (acceptor) must be in cat or lig.
            if acc_res not in cat_set and acc_res not in lig_set:
                continue
            # Sidechain-origin hbond only.
            try:
                if hb.don_hatm_is_backbone():
                    continue
            except Exception:
                pass
            count += 1
        return count

    # ------------------------------------------------------------------
    # Geometry deviations (new metric)
    # ------------------------------------------------------------------

    def _classify(self, res_idx, atom_name):
        """'fixed' if this atom is in the residue's fixed set, else 'mobile'.

        Ligand heavy atoms are always 'fixed' (all ligand atoms are constrained).
        """
        if res_idx in self.ligand_residues:
            return 'fixed'
        atoms = self.fixed_residues.get(res_idx, frozenset())
        return 'fixed' if atom_name in atoms else 'mobile'

    def _edge_classification(self, r1, a1, r2, a2):
        c1 = self._classify(r1, a1)
        c2 = self._classify(r2, a2)
        if c1 == c2 == 'fixed':
            return 'all_fixed'
        if c1 == c2 == 'mobile':
            return 'all_mobile'
        return 'fixed_mobile'

    def _enumerate_non_h_bonds(self, residue, res_idx):
        """Yield (res_idx, atom_name_a, res_idx, atom_name_b) for intra-residue non-H bonds.

        Each unordered pair yielded once (fixes the 2x counting bug from before).
        """
        seen = set()
        for atom_idx in range(1, residue.natoms() + 1):
            if residue.atom_type(atom_idx).element() == "H":
                continue
            name_a = residue.atom_name(atom_idx).strip()
            for bonded_idx in residue.bonded_neighbor(atom_idx):
                if residue.atom_type(bonded_idx).element() == "H":
                    continue
                name_b = residue.atom_name(bonded_idx).strip()
                key = tuple(sorted([name_a, name_b]))
                if key in seen:
                    continue
                seen.add(key)
                yield (res_idx, key[0], res_idx, key[1])

    def _enumerate_non_h_angles(self, residue, res_idx):
        """Yield 3-atom non-H angle tuples (res,a), (res,b), (res,c) where b is center.

        Only angles fully within this residue. Cross-residue angles (for peptide
        bonds) are added separately by _cross_residue_peptide_entries.
        """
        # Build adjacency: atom_name -> list of bonded heavy neighbor names
        name_of = {}
        adj = defaultdict(list)
        heavy_indices = [i for i in range(1, residue.natoms() + 1)
                         if residue.atom_type(i).element() != "H"]
        for i in heavy_indices:
            name_of[i] = residue.atom_name(i).strip()
        for i in heavy_indices:
            a = name_of[i]
            for j in residue.bonded_neighbor(i):
                if residue.atom_type(j).element() == "H":
                    continue
                adj[a].append(name_of[j])

        seen = set()
        for center, neighbors in adj.items():
            if len(neighbors) < 2:
                continue
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    a = neighbors[i]
                    c = neighbors[j]
                    key = (tuple(sorted([a, c])), center)
                    if key in seen:
                        continue
                    seen.add(key)
                    yield ((res_idx, a), (res_idx, center), (res_idx, c))

    def _cross_residue_peptide_entries(self, pose, res_idx):
        """Generate cross-residue bond/angle entries around peptide connectivity.

        For catalytic residue i (where relevant), yield:
          - bond: C[i] — N[i+1]
          - bond: C[i-1] — N[i]
          - angle: CA[i] — C[i] — N[i+1]
          - angle:  O[i] — C[i] — N[i+1]
          - angle: C[i-1] — N[i] — CA[i]
        Each entry returns a tuple (kind, tuple_of_(res,atom)) so callers can
        evaluate actual vs ideal.
        """
        out = []
        if res_idx <= pose.size() and pose.residue(res_idx).is_protein():
            # i → i+1 bond/angles
            if res_idx + 1 <= pose.size():
                try:
                    bonded = pose.conformation().residues_are_bonded(res_idx, res_idx + 1)
                except Exception:
                    bonded = True
                if bonded and pose.residue(res_idx + 1).is_protein():
                    same_chain = True
                    try:
                        same_chain = pose.pdb_info().chain(res_idx) == pose.pdb_info().chain(res_idx + 1)
                    except Exception:
                        pass
                    if same_chain:
                        if pose.residue(res_idx).has("C") and pose.residue(res_idx + 1).has("N"):
                            out.append(('bond', ((res_idx, "C"), (res_idx + 1, "N"))))
                        if pose.residue(res_idx).has("CA") and pose.residue(res_idx).has("C") and pose.residue(res_idx + 1).has("N"):
                            out.append(('angle', ((res_idx, "CA"), (res_idx, "C"), (res_idx + 1, "N"))))
                        if pose.residue(res_idx).has("O") and pose.residue(res_idx).has("C") and pose.residue(res_idx + 1).has("N"):
                            out.append(('angle', ((res_idx, "O"), (res_idx, "C"), (res_idx + 1, "N"))))
            # i-1 → i bond/angles
            if res_idx - 1 >= 1:
                try:
                    bonded = pose.conformation().residues_are_bonded(res_idx - 1, res_idx)
                except Exception:
                    bonded = True
                if bonded and pose.residue(res_idx - 1).is_protein():
                    same_chain = True
                    try:
                        same_chain = pose.pdb_info().chain(res_idx) == pose.pdb_info().chain(res_idx - 1)
                    except Exception:
                        pass
                    if same_chain:
                        if pose.residue(res_idx - 1).has("C") and pose.residue(res_idx).has("N"):
                            # Already covered by the (i-1, i) pair when res_idx-1 is catalytic;
                            # include here so lone catalytic residues see their N-terminal side.
                            out.append(('bond', ((res_idx - 1, "C"), (res_idx, "N"))))
                        if (pose.residue(res_idx - 1).has("C") and pose.residue(res_idx).has("N")
                                and pose.residue(res_idx).has("CA")):
                            out.append(('angle', ((res_idx - 1, "C"), (res_idx, "N"), (res_idx, "CA"))))
        return out

    @staticmethod
    def _measure_bond(pose, r1, a1, r2, a2):
        p1 = pose.residue(r1).xyz(a1)
        p2 = pose.residue(r2).xyz(a2)
        return float(p1.distance(p2))

    @staticmethod
    def _measure_angle(pose, r1, a1, r2, a2, r3, a3):
        p1 = pose.residue(r1).xyz(a1)
        p2 = pose.residue(r2).xyz(a2)
        p3 = pose.residue(r3).xyz(a3)
        v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return float('nan')
        cos = float(np.dot(v1, v2) / (n1 * n2))
        cos = max(-1.0, min(1.0, cos))
        return float(np.degrees(np.arccos(cos)))

    def _ideal_bond(self, residue, a1, a2):
        """Look up ideal bond length from the cached A-X-A reference residue.

        Returns None if unsupported.
        """
        ref = _get_ideal_reference_residue(residue.name1())
        if ref is None:
            return None
        if not (ref.has(a1) and ref.has(a2)):
            return None
        p1 = ref.xyz(a1)
        p2 = ref.xyz(a2)
        return float(p1.distance(p2))

    def _ideal_angle(self, residue, a1, a2, a3):
        ref = _get_ideal_reference_residue(residue.name1())
        if ref is None:
            return None
        if not (ref.has(a1) and ref.has(a2) and ref.has(a3)):
            return None
        p1 = ref.xyz(a1)
        p2 = ref.xyz(a2)
        p3 = ref.xyz(a3)
        v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return None
        cos = float(np.dot(v1, v2) / (n1 * n2))
        cos = max(-1.0, min(1.0, cos))
        return float(np.degrees(np.arccos(cos)))

    # Harmonic spring constants used for the approximate per-entry
    # cart_bonded contribution. These are Rosetta-style typical values for
    # protein bonds/angles; they're approximations, sufficient for producing
    # a comparable-scale strain signal that can be categorized by
    # fixed/mobile classification.
    _PSEUDO_K_BOND = 150.0       # 0.5 * ~300 kcal/mol/Å² (bond spring)
    _PSEUDO_K_ANGLE_DEG = 0.012  # 0.5 * 80/3282 kcal/mol/deg² (angle spring)

    @classmethod
    def _pseudo_cart_bonded_from_entries(cls, bond_entries, angle_entries, classes=None):
        """Estimate a cart_bonded-like strain using harmonic springs.

        Filters by classification if provided (list of strings). Missing
        ideal (delta is None) entries contribute 0.
        """
        total = 0.0
        for e in bond_entries:
            if e.get('delta') is None:
                continue
            if classes is not None and e['classification'] not in classes:
                continue
            total += cls._PSEUDO_K_BOND * (e['delta'] ** 2)
        for e in angle_entries:
            if e.get('delta') is None:
                continue
            if classes is not None and e['classification'] not in classes:
                continue
            total += cls._PSEUDO_K_ANGLE_DEG * (e['delta'] ** 2)
        return float(total)

    def calculate_catalytic_geometry_deviations(self, pose, res_idx):
        """New per-residue bond + angle deviation report.

        Returns a dict:
            {
              'supported': bool,
              'resname': str,
              'bonds': [ {'atoms':(..,..), 'actual':..., 'ideal':..., 'delta':...,
                          'classification': 'all_fixed'|'fixed_mobile'|'all_mobile'},
                         ... ],
              'angles': [ {'atoms':(..,..,..), 'actual':..., 'ideal':..., 'delta':..., 'classification': ...}, ... ],
              'summary': {by_class: {'max_bond_dev', 'mean_bond_dev', 'max_angle_dev', 'mean_angle_dev', 'n_bonds', 'n_angles'}},
            }
        Non-standard residues report supported=False but still include per-bond/angle entries (no 'ideal'/'delta') for reference.
        """
        residue = pose.residue(res_idx)
        ref_res = _get_ideal_reference_residue(residue.name1())
        supported = ref_res is not None

        bonds_out = []
        angles_out = []

        # Intra-residue bonds
        for (r1, a1, r2, a2) in self._enumerate_non_h_bonds(residue, res_idx):
            actual = self._measure_bond(pose, r1, a1, r2, a2)
            ideal = self._ideal_bond(residue, a1, a2) if supported else None
            classification = self._edge_classification(r1, a1, r2, a2)
            entry = {
                'atoms': (a1, a2),
                'actual': actual,
                'ideal': ideal,
                'delta': (actual - ideal) if ideal is not None else None,
                'classification': classification,
                'cross_residue': False,
            }
            bonds_out.append(entry)

        # Intra-residue angles
        for (n1, n2, n3) in self._enumerate_non_h_angles(residue, res_idx):
            r1, a1 = n1
            r2, a2 = n2
            r3, a3 = n3
            actual = self._measure_angle(pose, r1, a1, r2, a2, r3, a3)
            ideal = self._ideal_angle(residue, a1, a2, a3) if supported else None
            classification = self._edge_classification(r1, a1, r3, a3)  # endpoints determine it
            # If center is mobile, pull whole triangle toward fixed_mobile;
            # refine: use worst classification across three atoms.
            classes = [self._classify(r, a) for r, a in [(r1, a1), (r2, a2), (r3, a3)]]
            if all(c == 'fixed' for c in classes):
                classification = 'all_fixed'
            elif all(c == 'mobile' for c in classes):
                classification = 'all_mobile'
            else:
                classification = 'fixed_mobile'
            entry = {
                'atoms': (a1, a2, a3),
                'actual': actual,
                'ideal': ideal,
                'delta': (actual - ideal) if ideal is not None else None,
                'classification': classification,
                'cross_residue': False,
            }
            angles_out.append(entry)

        # Cross-residue peptide-bond entries (ideal values reserved — these
        # involve two residue types; a rigorous ideal would need a three-residue
        # reference. For now we report actual values and let users compare to
        # literature values 1.329 Å and 116/122°).
        for kind, tup in self._cross_residue_peptide_entries(pose, res_idx):
            if kind == 'bond':
                (r1, a1), (r2, a2) = tup
                actual = self._measure_bond(pose, r1, a1, r2, a2)
                # Literature peptide bond C-N ≈ 1.329 Å
                ideal = 1.329
                entry = {
                    'atoms': (f"{a1}[{r1}]", f"{a2}[{r2}]"),
                    'actual': actual,
                    'ideal': ideal,
                    'delta': actual - ideal,
                    'classification': self._edge_classification(r1, a1, r2, a2),
                    'cross_residue': True,
                }
                bonds_out.append(entry)
            else:
                (r1, a1), (r2, a2), (r3, a3) = tup
                actual = self._measure_angle(pose, r1, a1, r2, a2, r3, a3)
                # Literature ideals: CA-C-N ≈ 116.2°, O-C-N ≈ 123.0°, C-N-CA ≈ 121.7°
                if (a1, a2, a3) == ("CA", "C", "N"):
                    ideal = 116.2
                elif (a1, a2, a3) == ("O", "C", "N"):
                    ideal = 123.0
                elif (a1, a2, a3) == ("C", "N", "CA"):
                    ideal = 121.7
                else:
                    ideal = None
                classes = [self._classify(r, a) for r, a in [(r1, a1), (r2, a2), (r3, a3)]]
                if all(c == 'fixed' for c in classes):
                    classification = 'all_fixed'
                elif all(c == 'mobile' for c in classes):
                    classification = 'all_mobile'
                else:
                    classification = 'fixed_mobile'
                entry = {
                    'atoms': (f"{a1}[{r1}]", f"{a2}[{r2}]", f"{a3}[{r3}]"),
                    'actual': actual,
                    'ideal': ideal,
                    'delta': (actual - ideal) if ideal is not None else None,
                    'classification': classification,
                    'cross_residue': True,
                }
                angles_out.append(entry)

        # Summary by classification
        def _sumstats(entries, key):
            vals = [abs(e['delta']) for e in entries
                    if e.get('delta') is not None]
            if not vals:
                return {'n': len(entries), 'max': None, 'mean': None}
            return {'n': len(entries), 'max': float(max(vals)),
                    'mean': float(sum(vals) / len(vals))}

        summary = {}
        for cls in ('all_fixed', 'fixed_mobile', 'all_mobile'):
            b_sub = [e for e in bonds_out if e['classification'] == cls]
            a_sub = [e for e in angles_out if e['classification'] == cls]
            summary[cls] = {
                'bonds': _sumstats(b_sub, 'delta'),
                'angles': _sumstats(a_sub, 'delta'),
                'pseudo_cart_bonded': self._pseudo_cart_bonded_from_entries(b_sub, a_sub),
            }

        # The main strain signal the user cares about: everything the relax
        # could plausibly fix (fixed_mobile + all_mobile), excluding
        # intra-fixed QM-intentional geometry.
        pseudo_cb_actionable = self._pseudo_cart_bonded_from_entries(
            bonds_out, angles_out, classes=('fixed_mobile', 'all_mobile'))
        pseudo_cb_all_fixed_noted = self._pseudo_cart_bonded_from_entries(
            bonds_out, angles_out, classes=('all_fixed',))
        pseudo_cb_total = self._pseudo_cart_bonded_from_entries(bonds_out, angles_out)

        # Per-residue max/mean deviations restricted to actionable entries
        # (fixed_mobile + all_mobile only — never all_fixed).
        actionable_classes = ('fixed_mobile', 'all_mobile')
        actionable_bond_deltas = [abs(e['delta']) for e in bonds_out
                                  if e.get('delta') is not None
                                  and e['classification'] in actionable_classes]
        actionable_angle_deltas = [abs(e['delta']) for e in angles_out
                                   if e.get('delta') is not None
                                   and e['classification'] in actionable_classes]

        return {
            'supported': supported,
            'resname': residue.name3(),
            'bonds': bonds_out,
            'angles': angles_out,
            'summary': summary,
            'pseudo_cart_bonded_excluding_fixed_only': float(pseudo_cb_actionable),
            'pseudo_cart_bonded_all_fixed_noted': float(pseudo_cb_all_fixed_noted),
            'pseudo_cart_bonded_total': float(pseudo_cb_total),
            'actionable_bond_max_deviation': float(max(actionable_bond_deltas)) if actionable_bond_deltas else 0.0,
            'actionable_bond_mean_deviation': float(np.mean(actionable_bond_deltas)) if actionable_bond_deltas else 0.0,
            'actionable_angle_max_deviation': float(max(actionable_angle_deltas)) if actionable_angle_deltas else 0.0,
            'actionable_angle_mean_deviation': float(np.mean(actionable_angle_deltas)) if actionable_angle_deltas else 0.0,
            'actionable_num_bonds': int(len(actionable_bond_deltas)),
            'actionable_num_angles': int(len(actionable_angle_deltas)),
        }

    # ------------------------------------------------------------------
    # Relax / minimize
    # ------------------------------------------------------------------

    def idealize_secondary_structure(self, pose):
        print("\nIdealizing secondary structure...")
        try:
            from pyrosetta.rosetta.protocols.idealize import IdealizeMover
            IdealizeMover().apply(pose)
            print("✓ Secondary structure idealized")
        except Exception as e:
            print(f"  WARNING: Could not idealize: {e}")

    def run_cartesian_fastrelax(self, pose, sfxn, movemap, n_repeats=2):
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
        print(f"\nRunning cartesian minimization (tolerance={tolerance})...")
        min_mover = MinMover(movemap, sfxn, "lbfgs_armijo_nonmonotone", tolerance, True)
        min_mover.cartesian(True)
        score_before = sfxn(pose)
        min_mover.apply(pose)
        score_after = sfxn(pose)
        print(f"✓ Minimization complete")
        print(f"  Score: {score_before:.2f} → {score_after:.2f} (Δ {score_after - score_before:+.2f})")

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _residue_fa_rep_with_exclusions(self, pose, res_idx, excluded_pairs):
        """Return this residue's fa_rep minus its share of the residue-pair
        fa_rep for each chemistry-relevant declared contact it participates in.

        Polymer backbone pairs (tagged polymer_backbone_pair) are NOT subtracted
        — distortions of a peptide backbone between two fully-fixed residues
        must remain visible as a clash.

        Subtraction uses Rosetta's residue-pair fa_rep edge value × 0.5
        (Rosetta's residue_total_energies splits pair energies 50/50). This is
        an approximation: it also subtracts small contributions from ALL other
        atom-pair interactions between the two residues — including any
        non-fixed atoms of this residue that happen to clash with the partner.
        In practice, for a catalytic residue with only sidechain-tip atoms
        fixed (e.g. LYS87 fixed={NZ,CE}), the non-fixed atoms (backbone, CG,
        CD, CB) are ~3+ Å from the ligand and contribute < 1-2 Rosetta units
        to the pair fa_rep, dwarfed by the >100 units from the declared tight
        contact. For unusual cases where a non-fixed atom of this residue
        clashes closely with the partner, consult fa_rep_raw (always reported)
        for the uncorrected value.

        Computing an atom-pair-exact subtraction would require replicating
        Rosetta's piecewise spline fa_rep with linearized short-distance tail,
        which isn't reliably reproducible outside Rosetta itself.
        """
        try:
            raw = float(pose.energies().residue_total_energies(res_idx)[ScoreType.fa_rep])
        except Exception:
            return 0.0
        if not excluded_pairs:
            return raw

        try:
            egraph = pose.energies().energy_graph()
        except Exception:
            return raw

        subtract = 0.0
        subtracted_partners = set()
        for p in excluded_pairs:
            if p.get('polymer_backbone_pair'):
                continue
            if p['res1'] == res_idx:
                other = p['res2']
            elif p['res2'] == res_idx:
                other = p['res1']
            else:
                continue
            if other in subtracted_partners:
                continue  # only count each partner once (multiple atom pairs per partner collapse to one edge)
            subtracted_partners.add(other)
            try:
                edge = egraph.find_energy_edge(res_idx, other)
                if edge is None:
                    continue
                pair_fa_rep = float(edge.fill_energy_map()[ScoreType.fa_rep])
                subtract += 0.5 * pair_fa_rep
            except Exception:
                continue
        return max(0.0, raw - subtract)

    def calculate_validation_metrics(self, pose, sfxn, output_pdb_path=None):
        """Comprehensive validation metrics.

        Fixed-only violations (e.g. all_fixed bond/angle deltas) are *noted* but
        never pushed into the failed-checks list — the user pre-committed those
        geometries by marking them fixed.
        """
        print("\n  Calculating metrics...")
        metrics_start = time.time()

        metrics = {
            'metadata': {},
            'global_metrics': {},
            'scores': {},
            'declared_covalent_contacts': [],
            'catalytic_residues': {},
            'mpnn': self.mpnn_info or {},
            'quality_flags': {},
        }

        if self.debug:
            metrics['geometry'] = {}
            metrics['constraints'] = {}
            metrics['timing'] = self.timings.copy() if hasattr(self, 'timings') else {}

        # Metadata. structure_name mirrors the OUTPUT PDB's basename (without
        # the .pdb suffix) when output_pdb_path is provided, so multi-stage
        # runs don't end up with structure_name still pointing at the stage's
        # input PDB. Falls back to self.pdbname (the input basename) only when
        # there is no output path — e.g., the input-baseline-metrics pass at
        # the start of run().
        if output_pdb_path:
            metrics['metadata']['structure_name'] = (
                os.path.basename(output_pdb_path).replace(".pdb", "")
            )
            metrics['metadata']['pdb_path'] = output_pdb_path
        else:
            metrics['metadata']['structure_name'] = self.pdbname
        if self.debug:
            metrics['metadata']['timestamp'] = datetime.now().isoformat()
        metrics['metadata']['num_residues'] = int(pose.size())
        metrics['metadata']['num_fixed_residues'] = len(self.fixed_residues)
        metrics['metadata']['num_fully_fixed_residues'] = len(self.fully_fixed_residues)
        metrics['metadata']['num_ligands'] = len(self.ligand_residues)
        if self.debug:
            metrics['metadata']['mobile_region_size'] = len(self.mobile_residues)

        # Scores
        metrics['scores']['total_score'] = float(sfxn(pose))
        metrics['scores']['cart_bonded'] = float(pose.energies().total_energies()[ScoreType.cart_bonded])
        metrics['scores']['coordinate_constraint'] = float(pose.energies().total_energies()[ScoreType.coordinate_constraint])
        for st_name, st in (('fa_rep', ScoreType.fa_rep), ('fa_atr', ScoreType.fa_atr),
                            ('fa_sol', ScoreType.fa_sol), ('fa_elec', ScoreType.fa_elec),
                            ('hbond_sc', ScoreType.hbond_sc), ('hbond_bb_sc', ScoreType.hbond_bb_sc)):
            try:
                metrics['scores'][st_name] = float(pose.energies().total_energies()[st])
            except Exception:
                pass

        # Declared covalent contacts
        contacts_report = []
        for c in self.declared_covalent_contacts:
            c_report = dict(c)
            c_report['pdb_id1'] = f"{pose.pdb_info().chain(c['res1'])}{pose.pdb_info().number(c['res1'])}"
            c_report['pdb_id2'] = f"{pose.pdb_info().chain(c['res2'])}{pose.pdb_info().number(c['res2'])}"
            contacts_report.append(c_report)
        metrics['declared_covalent_contacts'] = contacts_report

        # Chain breaks (always)
        chain_breaks = self.detect_chain_breaks(pose)
        num_chain_breaks = len(chain_breaks)

        # Clashing residues (with chemistry-relevant declared-contact subtraction).
        # Polymer backbone pairs are listed but NOT subtracted, so any peptide
        # distortion between two fully-fixed residues still registers as a clash.
        clash_threshold = 10.0
        num_clashing_residues = 0
        clashing_residues_list = []
        chem_contacts_resset = set()
        polymer_contacts_resset = set()
        for c in self.declared_covalent_contacts:
            if c.get('polymer_backbone_pair'):
                polymer_contacts_resset.add(c['res1'])
                polymer_contacts_resset.add(c['res2'])
            else:
                chem_contacts_resset.add(c['res1'])
                chem_contacts_resset.add(c['res2'])
        for i in range(1, pose.size() + 1):
            try:
                raw = float(pose.energies().residue_total_energies(i)[ScoreType.fa_rep])
                effective = self._residue_fa_rep_with_exclusions(
                    pose, i, self.declared_covalent_contacts)
                if effective > clash_threshold:
                    num_clashing_residues += 1
                    if self.debug:
                        pdb_id = f"{pose.pdb_info().chain(i)}{pose.pdb_info().number(i)}"
                        clashing_residues_list.append({
                            'residue': int(i), 'pdb_id': pdb_id,
                            'resname': pose.residue(i).name3(),
                            'fa_rep_raw': raw, 'fa_rep_effective': effective,
                            'has_chemistry_declared_contact': i in chem_contacts_resset,
                            'has_polymer_backbone_declared_contact': i in polymer_contacts_resset,
                        })
            except Exception:
                continue

        if self.debug:
            metrics['geometry']['num_chain_breaks'] = num_chain_breaks
            metrics['geometry']['chain_break_residues'] = [int(x) for x in chain_breaks]
            metrics['geometry']['num_clashing_residues'] = num_clashing_residues
            clashing_residues_list.sort(key=lambda x: x['fa_rep_effective'], reverse=True)
            metrics['geometry']['clashing_residues'] = clashing_residues_list[:20]

        # Fixed atom displacement (using stored initial coords)
        fixed_res_displacements = []
        all_residues_with_fixed = list(self.fixed_residues.keys()) + list(self.ligand_residues)
        for res_idx in all_residues_with_fixed:
            if res_idx > pose.size():
                continue
            residue = pose.residue(res_idx)
            # Determine the relevant atoms.
            if res_idx in self.ligand_residues:
                atom_names = [residue.atom_name(i).strip()
                              for i in range(1, residue.natoms() + 1)
                              if not residue.atom_is_hydrogen(i)]
            else:
                atom_names = list(self.fixed_residues.get(res_idx, frozenset()))
            sq_dev = []
            max_dev_atom = None
            max_dev_val = 0.0
            for atom_name in atom_names:
                key = (res_idx, atom_name)
                if key not in self.fixed_atom_initial_coords or not residue.has(atom_name):
                    continue
                init = self.fixed_atom_initial_coords[key]
                now = residue.xyz(atom_name)
                d = ((now.x - init[0])**2 + (now.y - init[1])**2 + (now.z - init[2])**2) ** 0.5
                sq_dev.append(d * d)
                if d > max_dev_val:
                    max_dev_val = d
                    max_dev_atom = atom_name
            if sq_dev:
                pdb_id = f"{pose.pdb_info().chain(res_idx)}{pose.pdb_info().number(res_idx)}"
                fixed_res_displacements.append({
                    'residue': int(res_idx), 'pdb_id': pdb_id,
                    'resname': residue.name3(),
                    'num_atoms': len(sq_dev),
                    'rmsd': float(np.sqrt(np.mean(sq_dev))),
                    'max_displacement': float(np.sqrt(max(sq_dev))),
                    'max_displacement_atom': max_dev_atom,
                })

        fixed_res_displacements.sort(key=lambda x: x['max_displacement'], reverse=True)
        mean_fixed_rmsd = float(np.mean([x['rmsd'] for x in fixed_res_displacements])) if fixed_res_displacements else 0.0
        max_fixed_rmsd = float(np.max([x['rmsd'] for x in fixed_res_displacements])) if fixed_res_displacements else 0.0

        if self.debug:
            metrics['constraints']['fixed_residue_displacements'] = fixed_res_displacements
            metrics['constraints']['mean_fixed_rmsd'] = mean_fixed_rmsd
            metrics['constraints']['max_fixed_rmsd'] = max_fixed_rmsd
            if self.compute_ca_rmsd and self.original_pose is not None:
                from pyrosetta.rosetta.core.scoring import CA_rmsd
                metrics['constraints']['ca_rmsd_overall'] = float(CA_rmsd(self.original_pose, pose))

        # Catalytic residue details (new: full bond+angle deviation with classification)
        cat_res_details = []
        for res_idx in self.fixed_residues.keys():
            if res_idx > pose.size():
                continue
            residue = pose.residue(res_idx)
            pdb_id = f"{pose.pdb_info().chain(res_idx)}{pose.pdb_info().number(res_idx)}"
            try:
                raw_fa_rep = float(pose.energies().residue_total_energies(res_idx)[ScoreType.fa_rep])
            except Exception:
                raw_fa_rep = 0.0
            effective_fa_rep = self._residue_fa_rep_with_exclusions(
                pose, res_idx, self.declared_covalent_contacts)
            try:
                cart_bonded = float(pose.energies().residue_total_energies(res_idx)[ScoreType.cart_bonded])
            except Exception:
                cart_bonded = 0.0

            geom = self.calculate_catalytic_geometry_deviations(pose, res_idx)

            # Displacement from stored initial coords (already computed above)
            displacement = None
            for d in fixed_res_displacements:
                if d['residue'] == res_idx:
                    displacement = {k: d[k] for k in ('rmsd', 'max_displacement', 'max_displacement_atom')}
                    break

            cat_res_details.append({
                'residue': int(res_idx),
                'pdb_id': pdb_id,
                'resname': residue.name3(),
                'fixed_atoms': sorted(self.fixed_residues[res_idx]),
                'fully_constrained': res_idx in self.fully_fixed_residues,
                'cart_bonded': cart_bonded,
                'cart_bonded_excluding_fixed_only': geom.get('pseudo_cart_bonded_excluding_fixed_only', 0.0),
                'cart_bonded_all_fixed_noted': geom.get('pseudo_cart_bonded_all_fixed_noted', 0.0),
                # Actionable (fixed_mobile + all_mobile) bond/angle deviations
                # promoted to top-level for easy parsing without digging into geometry.
                'actionable_bond_max_deviation': geom.get('actionable_bond_max_deviation', 0.0),
                'actionable_bond_mean_deviation': geom.get('actionable_bond_mean_deviation', 0.0),
                'actionable_angle_max_deviation': geom.get('actionable_angle_max_deviation', 0.0),
                'actionable_angle_mean_deviation': geom.get('actionable_angle_mean_deviation', 0.0),
                'fa_rep_raw': raw_fa_rep,
                'fa_rep_effective': effective_fa_rep,
                'is_clashing_effective': effective_fa_rep > clash_threshold,
                'participates_in_chemistry_contact': res_idx in chem_contacts_resset,
                'participates_in_polymer_backbone_contact': res_idx in polymer_contacts_resset,
                'geometry': geom,
                'displacement': displacement,
            })

        cat_res_details.sort(key=lambda x: x['cart_bonded_excluding_fixed_only'], reverse=True)
        metrics['catalytic_residues']['details'] = cat_res_details
        if cat_res_details:
            metrics['catalytic_residues']['mean_cart_bonded'] = float(np.mean([x['cart_bonded'] for x in cat_res_details]))
            metrics['catalytic_residues']['max_cart_bonded'] = float(np.max([x['cart_bonded'] for x in cat_res_details]))
            metrics['catalytic_residues']['mean_cart_bonded_excluding_fixed_only'] = float(
                np.mean([x['cart_bonded_excluding_fixed_only'] for x in cat_res_details]))
            metrics['catalytic_residues']['max_cart_bonded_excluding_fixed_only'] = float(
                np.max([x['cart_bonded_excluding_fixed_only'] for x in cat_res_details]))
            metrics['catalytic_residues']['mean_cart_bonded_all_fixed_noted'] = float(
                np.mean([x['cart_bonded_all_fixed_noted'] for x in cat_res_details]))
            metrics['catalytic_residues']['num_clashing_effective'] = sum(1 for x in cat_res_details if x['is_clashing_effective'])

            # Aggregated actionable bond/angle deviations (fixed_mobile + all_mobile only).
            bond_max_per_res = [x['actionable_bond_max_deviation'] for x in cat_res_details]
            bond_mean_per_res = [x['actionable_bond_mean_deviation'] for x in cat_res_details]
            angle_max_per_res = [x['actionable_angle_max_deviation'] for x in cat_res_details]
            angle_mean_per_res = [x['actionable_angle_mean_deviation'] for x in cat_res_details]
            metrics['catalytic_residues']['max_actionable_bond_deviation'] = float(np.max(bond_max_per_res))
            metrics['catalytic_residues']['mean_actionable_bond_deviation'] = float(np.mean(bond_mean_per_res))
            metrics['catalytic_residues']['max_actionable_angle_deviation'] = float(np.max(angle_max_per_res))
            metrics['catalytic_residues']['mean_actionable_angle_deviation'] = float(np.mean(angle_mean_per_res))

            # Thresholds below also drive the violation-count quality flag.
            bond_dev_threshold = 0.03    # Å
            angle_dev_threshold = 3.0    # degrees

            # Top-5 worst actionable entries across ALL catalytic residues (for
            # eyeballing what the protein couldn't idealize).
            all_actionable = []
            for r in cat_res_details:
                for e in r['geometry']['bonds']:
                    if e.get('delta') is None or e['classification'] == 'all_fixed':
                        continue
                    all_actionable.append({
                        'residue': r['pdb_id'], 'resname': r['resname'],
                        'kind': 'bond', 'atoms': list(e['atoms']),
                        'classification': e['classification'],
                        'actual': e['actual'], 'ideal': e['ideal'],
                        'delta': e['delta'], 'abs_delta': abs(e['delta']),
                    })
                for e in r['geometry']['angles']:
                    if e.get('delta') is None or e['classification'] == 'all_fixed':
                        continue
                    all_actionable.append({
                        'residue': r['pdb_id'], 'resname': r['resname'],
                        'kind': 'angle', 'atoms': list(e['atoms']),
                        'classification': e['classification'],
                        'actual': e['actual'], 'ideal': e['ideal'],
                        'delta': e['delta'], 'abs_delta': abs(e['delta']),
                    })
            # Sort by |delta| normalized by threshold so bonds and angles mix fairly.
            bond_thr_norm = bond_dev_threshold
            angle_thr_norm = angle_dev_threshold
            all_actionable.sort(
                key=lambda x: x['abs_delta'] / (bond_thr_norm if x['kind']=='bond' else angle_thr_norm),
                reverse=True)
            metrics['catalytic_residues']['worst_actionable_entries'] = all_actionable[:5]

        # Quality flags — fixed-only violations DO NOT contribute to failed_checks.
        passed, failed = [], []
        if num_chain_breaks == 0:
            passed.append('no_chain_breaks')
        else:
            failed.append(f"{num_chain_breaks} chain breaks detected")
        if num_clashing_residues < 5:
            passed.append('low_clashes')
        else:
            failed.append(f"{num_clashing_residues} clashing residues (effective, post-exclusion) (threshold: <5)")
        if max_fixed_rmsd > 0:
            if max_fixed_rmsd < 0.1:
                passed.append('tight_constraints')
            else:
                failed.append(f"Fixed atoms moved {max_fixed_rmsd:.3f}Å (threshold: <0.1Å)")
        # Quality check uses the actionable cart_bonded (fixed_mobile + all_mobile
        # only). The all_fixed portion is QM-intentional and not a failure.
        mean_cb_actionable = metrics['catalytic_residues'].get('mean_cart_bonded_excluding_fixed_only', 0.0)
        if cat_res_details and mean_cb_actionable < 1.0:
            passed.append('good_catalytic_geometry_excluding_fixed_only')
        elif cat_res_details and mean_cb_actionable > 0:
            failed.append(f"Catalytic residues actionable cart_bonded {mean_cb_actionable:.2f} "
                          f"(fixed_mobile+all_mobile only, threshold: <1.0)")

        # Scan catalytic geometry deviations (only fixed_mobile and all_mobile
        # contribute to failed checks). Thresholds defined above.
        geom_violations_fm = 0
        geom_violations_am = 0
        geom_violations_ff_noted = 0
        for r in cat_res_details:
            for entry in r['geometry']['bonds']:
                if entry['delta'] is None:
                    continue
                if abs(entry['delta']) > bond_dev_threshold:
                    if entry['classification'] == 'all_fixed':
                        geom_violations_ff_noted += 1
                    elif entry['classification'] == 'fixed_mobile':
                        geom_violations_fm += 1
                    else:
                        geom_violations_am += 1
            for entry in r['geometry']['angles']:
                if entry['delta'] is None:
                    continue
                if abs(entry['delta']) > angle_dev_threshold:
                    if entry['classification'] == 'all_fixed':
                        geom_violations_ff_noted += 1
                    elif entry['classification'] == 'fixed_mobile':
                        geom_violations_fm += 1
                    else:
                        geom_violations_am += 1
        if geom_violations_fm + geom_violations_am == 0:
            passed.append('clean_catalytic_bond_angle_geometry')
        else:
            failed.append(f"{geom_violations_fm + geom_violations_am} catalytic bond/angle violations "
                          f"(fixed-mobile: {geom_violations_fm}; all-mobile: {geom_violations_am}; "
                          f"thresholds: {bond_dev_threshold}Å / {angle_dev_threshold}°)")
        if geom_violations_ff_noted:
            passed.append(f"noted_{geom_violations_ff_noted}_fixed_only_deviations_not_failing")

        geometry_acceptable = (num_chain_breaks == 0
                               and num_clashing_residues < 5
                               and max_fixed_rmsd < 0.1
                               and (geom_violations_fm + geom_violations_am) == 0)
        metrics['quality_flags']['geometry_acceptable'] = geometry_acceptable
        metrics['quality_flags']['passed_checks'] = passed
        metrics['quality_flags']['failed_checks'] = failed
        metrics['quality_flags']['explanation'] = ("All checks passed" if geometry_acceptable
                                                   else "Failed: " + "; ".join(failed))

        # Global metrics
        metrics['global_metrics']['num_chain_breaks'] = num_chain_breaks
        metrics['global_metrics']['num_clashing_residues_effective'] = num_clashing_residues
        metrics['global_metrics']['total_score'] = metrics['scores']['total_score']
        metrics['global_metrics']['cart_bonded'] = metrics['scores']['cart_bonded']
        metrics['global_metrics']['mean_fixed_atom_displacement'] = mean_fixed_rmsd
        metrics['global_metrics']['max_fixed_atom_displacement'] = max_fixed_rmsd
        if cat_res_details:
            metrics['global_metrics']['mean_catalytic_cart_bonded'] = metrics['catalytic_residues'].get('mean_cart_bonded', 0.0)
            metrics['global_metrics']['mean_catalytic_cart_bonded_excluding_fixed_only'] = \
                metrics['catalytic_residues'].get('mean_cart_bonded_excluding_fixed_only', 0.0)
            metrics['global_metrics']['max_catalytic_cart_bonded_excluding_fixed_only'] = \
                metrics['catalytic_residues'].get('max_cart_bonded_excluding_fixed_only', 0.0)
            metrics['global_metrics']['mean_catalytic_cart_bonded_all_fixed_noted'] = \
                metrics['catalytic_residues'].get('mean_cart_bonded_all_fixed_noted', 0.0)
            metrics['global_metrics']['num_catalytic_clashing_effective'] = metrics['catalytic_residues'].get('num_clashing_effective', 0)
            metrics['global_metrics']['catalytic_geom_violations_fixed_mobile'] = geom_violations_fm
            metrics['global_metrics']['catalytic_geom_violations_all_mobile'] = geom_violations_am
            metrics['global_metrics']['catalytic_geom_violations_all_fixed_noted'] = geom_violations_ff_noted
            # Precomputed actionable bond/angle deviations (fixed_mobile + all_mobile only).
            # These are the main quality signals — they're what relax could plausibly fix.
            metrics['global_metrics']['catalytic_max_actionable_bond_deviation'] = \
                metrics['catalytic_residues'].get('max_actionable_bond_deviation', 0.0)
            metrics['global_metrics']['catalytic_mean_actionable_bond_deviation'] = \
                metrics['catalytic_residues'].get('mean_actionable_bond_deviation', 0.0)
            metrics['global_metrics']['catalytic_max_actionable_angle_deviation'] = \
                metrics['catalytic_residues'].get('max_actionable_angle_deviation', 0.0)
            metrics['global_metrics']['catalytic_mean_actionable_angle_deviation'] = \
                metrics['catalytic_residues'].get('mean_actionable_angle_deviation', 0.0)
        if self.compute_ca_rmsd and self.original_pose is not None:
            from pyrosetta.rosetta.core.scoring import CA_rmsd
            metrics['global_metrics']['ca_rmsd_overall'] = float(CA_rmsd(self.original_pose, pose))

        if self.debug:
            metrics['timing']['metrics_calculation'] = time.time() - metrics_start

        return metrics

    # ------------------------------------------------------------------
    # PDB post-processing
    # ------------------------------------------------------------------

    def strip_energy_table_from_pdb(self, pdb_path):
        """Drop the per-residue energy table from a PDB file to save disk."""
        with open(pdb_path, 'r') as f:
            lines = f.readlines()
        new_lines, in_table = [], False
        for line in lines:
            if line.startswith('# All scores below are weighted scores'):
                in_table = True
                continue
            if line.startswith('#END_POSE_ENERGIES_TABLE'):
                in_table = False
                continue
            if not in_table:
                new_lines.append(line)
        with open(pdb_path, 'w') as f:
            f.writelines(new_lines)

    # ------------------------------------------------------------------
    # JSON write helper
    # ------------------------------------------------------------------

    def round_metrics(self, obj, decimals=4):
        if isinstance(obj, dict):
            return {k: self.round_metrics(v, decimals) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.round_metrics(v, decimals) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self.round_metrics(v, decimals) for v in obj)
        if isinstance(obj, float):
            return round(obj, decimals)
        if isinstance(obj, (set, frozenset)):
            return sorted(obj)
        return obj

    @staticmethod
    def _extract_numeric_skeleton(d):
        """
        Recursively prune `d`, keeping only finite numeric leaves at their
        original nested paths. Strings, lists, bools, NaN/inf, and now-empty
        sub-dicts are dropped.

        Used to build `input_metrics` as a sparse, dedup'd view of just the
        baseline values that could meaningfully change — non-numeric metadata
        (residue names, fixed-atom lists, quality_flags strings, etc.) lives
        only in the top-level metrics block written for the final pose.
        """
        import math

        def _walk(x):
            if isinstance(x, bool):
                return None
            if isinstance(x, (int, float)):
                return float(x) if math.isfinite(x) else None
            if isinstance(x, dict):
                out = {}
                for k, v in x.items():
                    sub = _walk(v)
                    if sub is not None:
                        out[k] = sub
                return out if out else None
            return None  # strings, lists, tuples, sets — all dropped

        result = _walk(d)
        return result if result is not None else {}

    @staticmethod
    def _assert_catalytic_residues_identity_unchanged(final_metrics: dict,
                                                       input_metrics: dict) -> None:
        """
        Sanity check: the set of catalytic residue identifiers must be
        identical between the input pose and the final (MPNN+relax) pose.

        Catalytic residues are FIXED by the protocol — if their keys diverge,
        something silently changed the catalytic set (accidental mutation,
        residue renumbering, an MPNN bug, etc.) and the input/final metrics
        are no longer directly comparable.
        """
        final_cat = (final_metrics.get('catalytic_residues') or {})
        input_cat = (input_metrics.get('catalytic_residues') or {})
        # The block can hold both per-residue sub-dicts (keyed by residue id)
        # and aggregate scalars (e.g. `mean_cart_bonded`). Only compare the
        # per-residue dict keys for the identity check; aggregates can differ.
        final_keys = {k for k, v in final_cat.items() if isinstance(v, dict)}
        input_keys = {k for k, v in input_cat.items() if isinstance(v, dict)}
        if final_keys != input_keys:
            only_final = sorted(final_keys - input_keys)
            only_input = sorted(input_keys - final_keys)
            raise AssertionError(
                "catalytic_residues identity diverged between input and final poses — "
                "this should not happen, catalytic residues are fixed.\n"
                f"  only in final: {only_final}\n"
                f"  only in input: {only_input}\n"
                f"  shared       : {sorted(final_keys & input_keys)}"
            )

    @staticmethod
    def _compute_metric_deltas(final_metrics: dict, input_metrics: dict) -> dict:
        """
        Recursively walk two metric dicts and return a nested dict of numeric
        leaf deltas (final - input).

        Rules:
          * Only numeric leaves (int / float, excluding bool) present in BOTH
            sides produce a delta.
          * NaN / inf on either side -> skip (no delta written for that leaf).
          * Strings, lists, sets, tuples, dicts-of-non-numeric, and leaves
            missing from one side are skipped silently.
          * Result preserves the nested shape, so e.g.
              change_from_input["global_metrics"]["catalytic_max_actionable_angle_deviation"]
            sits where the user expects it relative to the original schema.
        """
        import math

        def _is_num(v):
            # bool is a subclass of int; we don't want to treat True/False as numeric here.
            return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)

        def _walk(f, i):
            if isinstance(f, dict) and isinstance(i, dict):
                out = {}
                for k in f.keys() & i.keys():
                    sub = _walk(f[k], i[k])
                    if sub is not None:
                        out[k] = sub
                return out if out else None
            if _is_num(f) and _is_num(i):
                return float(f) - float(i)
            return None

        result = _walk(final_metrics, input_metrics)
        return result if result is not None else {}

    def write_metrics_json(self, metrics, output_path):
        metrics = self.round_metrics(metrics, decimals=4)
        # Convert any tuple atom-lists to lists for JSON serialization.
        def _json_safe(x):
            if isinstance(x, dict):
                return {k: _json_safe(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [_json_safe(v) for v in x]
            if isinstance(x, (set, frozenset)):
                return sorted(x)
            return x
        metrics = _json_safe(metrics)
        json_path = output_path.replace('.pdb', '_metrics.json')
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2, sort_keys=False)
        print(f"✓ Metrics written to: {json_path}")
        return json_path

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def run(self, output_path=None, do_idealize_ss=False,
            do_fastrelax=True, do_minimize=True, fastrelax_cycles=2,
            min_tolerance=0.0001, strip_pdb_energies=True):
        print("\n" + "=" * 70)
        print("RFDiffusion3 Geometry Idealizer")
        print("=" * 70)

        self.start_time = time.time()
        self.timings = {}

        # Load JSON first (doesn't need pose).
        json_start = time.time()
        self.load_json()
        self.timings["json_loading"] = time.time() - json_start

        # Load pose.
        pose_start = time.time()
        print(f"\nLoading PDB: {self.pdb_path}")
        self.pose = pyr.pose_from_file(self.pdb_path)
        self.timings["pose_loading"] = time.time() - pose_start
        print(f"✓ Loaded pose with {self.pose.size()} residues")

        # Optional — only kept when --compute_ca_rmsd is on, because full clones
        # are memory-expensive for large protein-ligand complexes.
        if self.compute_ca_rmsd:
            self.original_pose = self.pose.clone()

        # Normalize JSON atom-specs against the actual pose.
        if self.fixed_atoms_data_raw:
            self.fixed_residues, self.fully_fixed_residues = \
                self.map_json_residues_to_pose(self.pose)
        else:
            print("\nNo fixed atoms data — will only fix ligand")
            self.fixed_residues = {}
            self.fully_fixed_residues = set()

        self.ligand_residues = self.identify_ligands(self.pose)
        if not self.ligand_residues and not self.fixed_residues:
            print("\n ERROR: No ligands or fixed residues found — nothing to constrain.")
            sys.exit(1)

        # Scorefunction setup happens BEFORE MPNN so _score_mpnn_candidate can
        # use a cartesian-aware sfxn (otherwise cart_bonded_catalytic is 0).
        self.sfxn = self.setup_scorefunction()

        # Compute INPUT-pose validation metrics ONCE here, before MPNN/relax
        # touches anything. Cheap relative to the rosetta steps; gives us a
        # frozen "what the filter step handed us" baseline for change_from_input
        # deltas on the final output. Reuses self.pose at this point (it is
        # still the freshly-loaded input pose).
        if self.compute_input_baseline_metrics:
            print("\n" + "=" * 70)
            print("INPUT BASELINE METRICS  (computed once, pre-MPNN/relax)")
            print("=" * 70)
            self._input_metrics_baseline = self.calculate_validation_metrics(
                self.pose, self.sfxn, output_pdb_path=None,
            )

        # MPNN pre-design (on by default).
        if self.run_mpnn_flag:
            mpnn_start = time.time()
            # Snapshot protected coords against the ORIGINAL (pre-MPNN) pose so
            # the post-reload restore pins RFdiffusion3 xyz, not any drifted xyz.
            pre_mpnn_snapshot = self._snapshot_protected_coords(self.pose)
            self.pose, self.mpnn_info = self.run_mpnn_predesign(self.pose)
            # Dump+reload to ensure clean hydrogen placement on MPNN output.
            # Skipped when the input sequence won the composite comparison —
            # in that case self.pose IS the un-touched input and there is no
            # MPNN sidechain-packer artifact to clean up.
            if (self.mpnn_info and 'skipped' not in self.mpnn_info
                    and not self.mpnn_info.get('input_sequence_selected', False)):
                tmp_path = os.path.join(
                    os.path.dirname(output_path) if output_path else self.pdb_dir,
                    f"{self.pdbname}_postmpnn_tmp.pdb",
                )
                os.makedirs(os.path.dirname(tmp_path) or ".", exist_ok=True)
                pre_reload_tautomers = self.record_his_tautomers(
                    self.pose, self.fixed_residues.keys())
                self.pose.dump_pdb(tmp_path)
                self.pose = pyr.pose_from_file(tmp_path)
                self.restore_his_tautomers(self.pose, pre_reload_tautomers)
                # Re-apply original protected coords (belt+suspenders vs PDB
                # quantization and any pose_from_file idealization).
                post_reload_drift = self._restore_protected_coords(
                    self.pose, pre_mpnn_snapshot)
                self.mpnn_info['post_reload_drift_A'] = float(post_reload_drift)
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            self.timings["mpnn"] = time.time() - mpnn_start
        else:
            self.mpnn_info = {'skipped': 'disabled via --skip_mpnn'}

        # Declared covalent contacts (best-effort declare_chemical_bond + metric-level fix).
        self.declared_covalent_contacts = self.detect_covalent_contacts(
            self.pose, self.ligand_residues)
        self.declare_covalent_bonds(self.pose, self.declared_covalent_contacts)
        # Clear energies after conformation change so next scoring pass is clean.
        self.pose.energies().clear()

        # Constraints.
        cst_start = time.time()
        self.add_coordinate_constraints(
            self.pose, self.fixed_residues, self.ligand_residues)
        self.timings["constraint_setup"] = time.time() - cst_start

        # Mobile region (default = whole protein).
        self.mobile_residues = self.define_mobile_region(
            self.pose, self.fixed_residues, self.ligand_residues,
            radius=self.mobile_radius)

        # MoveMap (fully-fixed residues are frozen here).
        movemap = self.build_movemap(
            self.pose, self.mobile_residues, self.ligand_residues,
            self.fully_fixed_residues)

        # Chain break check.
        print("\nChecking for chain breaks...")
        breaks = self.detect_chain_breaks(self.pose)
        if breaks:
            print(f"  WARNING: Found {len(breaks)} chain breaks")
        else:
            print("✓ No chain breaks detected")

        if do_idealize_ss:
            self.idealize_secondary_structure(self.pose)

        print("\n" + "=" * 70)
        print("GEOMETRY IDEALIZATION PROTOCOL")
        print("=" * 70)
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
        print(f"\n{'=' * 70}")
        print(f"Final score: {final_score:.2f} (Δ {final_score - initial_score:+.2f})")
        print(f"{'=' * 70}\n")

        breaks_after = self.detect_chain_breaks(self.pose)
        if breaks_after:
            print(f"⚠  Still have {len(breaks_after)} chain breaks after idealization")
        else:
            print("✓ No chain breaks after idealization")

        if output_path is None:
            output_path = f"{self.pdbname}_idealized.pdb"
        self.pose.dump_pdb(output_path)
        if strip_pdb_energies:
            self.strip_energy_table_from_pdb(output_path)
        print(f"\n✓ Output written to: {output_path}")

        if not self.no_metric_json_output:
            print("\n" + "=" * 70)
            print("CALCULATING VALIDATION METRICS")
            print("=" * 70)
            metrics = self.calculate_validation_metrics(
                self.pose, self.sfxn, output_pdb_path=output_path)
            print(f"\n✓ Validation Summary:")
            print(f"  - Chain breaks: {metrics['global_metrics'].get('num_chain_breaks')}")
            print(f"  - Clashing residues (effective): {metrics['global_metrics'].get('num_clashing_residues_effective')}")
            print(f"  - Declared covalent contacts: {len(metrics['declared_covalent_contacts'])}")
            if 'ca_rmsd_overall' in metrics['global_metrics']:
                print(f"  - CA-RMSD (input → final): {metrics['global_metrics']['ca_rmsd_overall']:.3f} Å")
            print(f"  - Catalytic cart_bonded (Rosetta, mean): {metrics['catalytic_residues'].get('mean_cart_bonded', 0):.2f}")
            print(f"  - Catalytic cart_bonded ACTIONABLE (excl fixed-only, mean): "
                  f"{metrics['catalytic_residues'].get('mean_cart_bonded_excluding_fixed_only', 0):.2f}  "
                  f"[main strain signal — only fixed_mobile + all_mobile bonds/angles]")
            print(f"  - Catalytic cart_bonded FIXED-ONLY (QM-intentional, mean): "
                  f"{metrics['catalytic_residues'].get('mean_cart_bonded_all_fixed_noted', 0):.2f}  "
                  f"[noted; not counted toward quality]")
            print(f"  - Catalytic bond/angle violations (fixed_mobile / all_mobile / all_fixed_noted): "
                  f"{metrics['global_metrics'].get('catalytic_geom_violations_fixed_mobile', 0)} / "
                  f"{metrics['global_metrics'].get('catalytic_geom_violations_all_mobile', 0)} / "
                  f"{metrics['global_metrics'].get('catalytic_geom_violations_all_fixed_noted', 0)}")
            print(f"  - Geometry acceptable: {metrics['quality_flags']['geometry_acceptable']}")
            if not metrics['quality_flags']['geometry_acceptable']:
                print(f"  - Reason: {metrics['quality_flags']['explanation']}")
            # Top actionable bond/angle deviation signals (for quick scanning).
            print(f"  - Actionable catalytic deviations:  "
                  f"max_bond={metrics['global_metrics'].get('catalytic_max_actionable_bond_deviation', 0):.3f} Å, "
                  f"mean_bond={metrics['global_metrics'].get('catalytic_mean_actionable_bond_deviation', 0):.3f} Å, "
                  f"max_angle={metrics['global_metrics'].get('catalytic_max_actionable_angle_deviation', 0):.2f}°, "
                  f"mean_angle={metrics['global_metrics'].get('catalytic_mean_actionable_angle_deviation', 0):.2f}°")
            worst = metrics['catalytic_residues'].get('worst_actionable_entries', [])
            if worst:
                print(f"  - Top {len(worst)} worst actionable entries (fixed_mobile + all_mobile only):")
                for e in worst:
                    atoms = '-'.join(str(a) for a in e['atoms'])
                    unit = 'Å' if e['kind'] == 'bond' else '°'
                    fmt = '.3f' if e['kind'] == 'bond' else '.2f'
                    print(f"      {e['residue']} ({e['resname']}) [{e['classification']:<12}] "
                          f"{e['kind']:5} {atoms:<22} "
                          f"actual={e['actual']:{fmt}}{unit}  ideal={e['ideal']:{fmt}}{unit}  "
                          f"Δ={e['delta']:+{fmt}}{unit}")

            # Attach the input-pose baseline + per-leaf deltas to the metrics
            # dict before serializing. Baseline was computed once at the start
            # of run() (see "INPUT BASELINE METRICS" block); we just diff here.
            #
            # `input_metrics` is pruned to numeric leaves only — non-numeric
            # metadata (resnames, fixed-atom lists, quality_flags strings) is
            # identical to the corresponding entries already present in the
            # top-level metrics block written for the final pose, so we don't
            # duplicate it. Catalytic-residue identity is asserted to be
            # unchanged before we trust the deltas.
            if self._input_metrics_baseline is not None:
                self._assert_catalytic_residues_identity_unchanged(
                    metrics, self._input_metrics_baseline)
                metrics['input_metrics'] = self._extract_numeric_skeleton(
                    self._input_metrics_baseline)
                metrics['change_from_input'] = self._compute_metric_deltas(
                    metrics, self._input_metrics_baseline)

                # Headline delta — the catalytic max angle deviation is the
                # filter bottleneck, so surface its delta directly.
                gm_in = self._input_metrics_baseline.get('global_metrics', {}) or {}
                gm_fi = metrics.get('global_metrics', {}) or {}
                key = 'catalytic_max_actionable_angle_deviation'
                if key in gm_in and key in gm_fi:
                    v_in = gm_in[key]
                    v_fi = gm_fi[key]
                    if isinstance(v_in, (int, float)) and isinstance(v_fi, (int, float)):
                        print(f"  - Δ catalytic max angle dev: {v_fi - v_in:+.2f}° "
                              f"(input {v_in:.2f}° → final {v_fi:.2f}°)")

            self.timings["total_runtime"] = time.time() - self.start_time
            if self.debug:
                metrics["timing"] = self.timings
            self.write_metrics_json(metrics, output_path)

        self.timings["total_runtime"] = time.time() - self.start_time
        print("\n" + "=" * 70)
        print("TIMING SUMMARY")
        print("=" * 70)
        for k in ("json_loading", "pose_loading", "mpnn", "constraint_setup",
                  "fastrelax", "minimization", "metrics_calculation"):
            if k in self.timings:
                print(f"  {k:24s}  {self.timings[k]:.2f}s")
        print(f"  {'-' * 40}")
        print(f"  TOTAL RUNTIME         {self.timings['total_runtime']:.2f}s")
        print("=" * 70 + "\n")
        return self.pose


# ---------------------------------------------------------------------------
# OUTPUT PDB REMARK REORGANIZATION (post-processing)
# ---------------------------------------------------------------------------

def reorganize_output_pdb_remarks(
    output_pdb_path: str,
    input_pdb_path: str = None,
    include_all_header_lines: bool = False,
    add_predesign_design_path: bool = True,
    verbose: bool = True,
) -> None:
    """
    Rewrite the dumped output PDB's header so REMARK lines land in a canonical
    layout AND any custom REMARK types that PyRosetta silently drops on its
    pose_from_file -> dump_pdb round-trip (e.g. REMARK QCB, REMARK DESIGN_PATH,
    REMARK rfd3_property) are rescued from the original input PDB at
    `input_pdb_path` and re-injected.

    Layout above the first ATOM/HETATM record:
      1. Numbered `REMARK <int>` lines in ascending numeric order
         (REMARK 665 above REMARK 666, etc.). Within each numeric group,
         input-PDB lines come first, then any output-only lines that weren't
         already present. Dedup is by exact (rstripped) line content.
      2. `REMARK QCB ...`
      3. `REMARK rfd3_property ...`
      4. `REMARK DESIGN_PATH ...`. When `add_predesign_design_path` is True,
         a new
            `REMARK DESIGN_PATH predesign_cart_relax output <output_pdb_path>`
         line is appended at the *bottom* of this group (after any rescued
         upstream DESIGN_PATH lines like `rfd3 input` / `rfd3 output`).
      5. Any other REMARK types collected under "_misc".

    By default this drops `HEADER ... xx-MMM-xx`, `EXPDTA THEORETICAL MODEL`,
    and `REMARK 220 ...` from both sides. Pass `include_all_header_lines=True`
    to keep them.

    Runs at the very end of main(), AFTER all Rosetta IO is complete. The
    `input_pdb_path` is read off disk — the script does not rely on whatever
    PyRosetta happened to retain on its internal PDBInfo.
    """
    BODY_RECORDS = ("ATOM", "HETATM", "TER", "END", "ENDMDL", "MODEL", "CONECT", "MASTER")

    def _normalize_design_path_line(line: str) -> str:
        """Normalize the path token inside a 'REMARK DESIGN_PATH <stage> <kind> <path>'
        line — collapses '//' -> '/', strips redundant './' segments, strips PDB
        column-80 trailing whitespace from the path. Non-DESIGN_PATH lines and
        malformed DESIGN_PATH lines pass through unchanged.
        """
        if not line.startswith("REMARK DESIGN_PATH"):
            return line
        body = line.rstrip("\n")
        nl = line[len(body):]  # "" or "\n"
        tokens = body.split(None, 4)  # at most 5 fields
        if len(tokens) < 5:
            return line
        prefix = " ".join(tokens[:4])
        path_clean = tokens[4].rstrip()  # drop any column-80 padding spaces
        normalized = os.path.normpath(path_clean)
        return f"{prefix} {normalized}{nl}"

    def _parse(lines, capture_body):
        numbered: dict = {}
        grouped: dict = {"QCB": [], "rfd3_property": [], "DESIGN_PATH": [], "_misc": []}
        header_kept = []
        body_lines = []
        n_drop_header = 0
        n_drop_220 = 0
        n_drop_remark0 = 0
        in_body = False
        for line in lines:
            if not in_body and any(line.startswith(rec) for rec in BODY_RECORDS):
                in_body = True
            if in_body:
                if capture_body:
                    body_lines.append(line)
                continue

            if line.startswith("HEADER") or line.startswith("EXPDTA"):
                if include_all_header_lines:
                    header_kept.append(line)
                else:
                    n_drop_header += 1
                continue

            if line.startswith("REMARK"):
                tokens = line.split()
                if len(tokens) < 2:
                    continue
                tag = tokens[1]
                if tag.isdigit():
                    rem_num = int(tag)
                    # PyRosetta's dump_pdb serializes non-numeric REMARK tags
                    # (e.g. REMARK QCB, REMARK DESIGN_PATH, REMARK rfd3_property)
                    # as 'REMARK   0 ...' with the original tag stripped/garbled
                    # by fixed-width column padding (it pads '   0' into the
                    # 4-char REMARK-number slot and eats the original tag in
                    # place). The full versions are rescued by name from the
                    # input PDB, so we drop these PDB-spec-invalid REMARK 0
                    # artifacts unconditionally.
                    if rem_num == 0:
                        n_drop_remark0 += 1
                        continue
                    if rem_num == 220 and not include_all_header_lines:
                        n_drop_220 += 1
                        continue
                    numbered.setdefault(rem_num, []).append(line)
                elif tag == "QCB":
                    grouped["QCB"].append(line)
                elif tag == "rfd3_property":
                    grouped["rfd3_property"].append(line)
                elif tag == "DESIGN_PATH":
                    grouped["DESIGN_PATH"].append(line)
                else:
                    grouped["_misc"].append(line)
                continue

            # CRYST1 / SCALE / ORIGX / etc. — rare in Rosetta dumps but kept
            # in the pre-body slot when the user asked for all header lines.
            if include_all_header_lines:
                header_kept.append(line)
        return (numbered, grouped, header_kept, body_lines,
                n_drop_header, n_drop_220, n_drop_remark0)

    # Parse the dumped output PDB (this is the file we're going to rewrite).
    with open(output_pdb_path, "r") as fh:
        out_lines = fh.readlines()
    (numbered, grouped, header_kept, body_lines,
     n_dropped_header, n_dropped_remark220, n_dropped_remark0) = \
        _parse(out_lines, capture_body=True)

    # Track how many REMARKs we rescue from the input PDB for the summary.
    rescued_counts = {
        "numbered_by_num": {},   # num -> int
        "QCB": 0,
        "rfd3_property": 0,
        "DESIGN_PATH": 0,
        "_misc": 0,
    }

    if input_pdb_path:
        try:
            with open(input_pdb_path, "r") as fh:
                in_lines = fh.readlines()
        except OSError as e:
            if verbose:
                print(f"[WARN] could not read input PDB for REMARK preservation "
                      f"({input_pdb_path}): {e}")
            in_lines = None

        if in_lines is not None:
            in_numbered, in_grouped, _ih, _ib, _idh, _id220, _idr0 = \
                _parse(in_lines, capture_body=False)

            def _merge(input_list, output_list):
                """Input lines come first (authoritative). Output lines are
                appended only if they are not byte-identical (rstripped) to
                an input line AND not a Rosetta column-80 truncation prefix
                of an input line — PyRosetta's dump_pdb cuts long REMARKs at
                col 80, so e.g. an input 'REMARK 665 fmt: ... MATCH MOTIF ...'
                shows up in the dumped output as 'REMARK 665 fmt: ... MATCH MOTI'
                (truncated mid-word). We treat those as the same line."""
                in_stripped = [l.rstrip() for l in input_list]
                in_set = set(in_stripped)

                def _is_truncation_of_input(s):
                    return bool(s) and any(
                        i.startswith(s) and len(i) > len(s) for i in in_stripped)

                surviving_output = []
                for o in output_list:
                    os_ = o.rstrip()
                    if os_ in in_set:
                        continue
                    if _is_truncation_of_input(os_):
                        continue
                    surviving_output.append(o)
                return list(input_list) + surviving_output, len([
                    l for l in input_list if l.rstrip() not in
                    {x.rstrip() for x in output_list}
                ])

            for num, lst in in_numbered.items():
                merged, rescued = _merge(lst, numbered.get(num, []))
                numbered[num] = merged
                if rescued:
                    rescued_counts["numbered_by_num"][num] = rescued

            for grp_name, lst in in_grouped.items():
                merged, rescued = _merge(lst, grouped[grp_name])
                grouped[grp_name] = merged
                if rescued:
                    rescued_counts[grp_name] = rescued

    # Append the new predesign_cart_relax DESIGN_PATH line at the bottom of
    # the DESIGN_PATH group (after any rescued/preserved DESIGN_PATH entries).
    # The path is normalized so things like '<dir>//<file>.pdb' (from a
    # trailing-slashed --output_dir) collapse to '<dir>/<file>.pdb'.
    added_predesign_line = False
    if add_predesign_design_path:
        grouped["DESIGN_PATH"].append(
            f"REMARK DESIGN_PATH predesign_cart_relax output "
            f"{os.path.normpath(output_pdb_path)}\n"
        )
        added_predesign_line = True

    # Normalize every DESIGN_PATH line in the group (rescued + preserved + new)
    # so any upstream double-slash artifacts are cleaned up consistently.
    grouped["DESIGN_PATH"] = [_normalize_design_path_line(l) for l in grouped["DESIGN_PATH"]]

    new_lines = []
    new_lines.extend(header_kept)
    for n in sorted(numbered.keys()):
        new_lines.extend(numbered[n])
    new_lines.extend(grouped["QCB"])
    new_lines.extend(grouped["rfd3_property"])
    new_lines.extend(grouped["DESIGN_PATH"])
    new_lines.extend(grouped["_misc"])
    new_lines.extend(body_lines)

    with open(output_pdb_path, "w") as fh:
        fh.writelines(new_lines)

    if verbose:
        def _rescued_suffix(n):
            return f"  [rescued from input: {n}]" if n else ""

        print("\n" + "=" * 70)
        print("POST-PROCESS: REMARK REORGANIZATION")
        print("=" * 70)
        print(f"  output PDB                : {os.path.normpath(output_pdb_path)}")
        if input_pdb_path:
            print(f"  input  PDB (for rescue)   : {os.path.normpath(input_pdb_path)}")
        if numbered:
            numbered_str = ", ".join(
                f"REMARK {n} (n={len(numbered[n])}"
                f"{', rescued ' + str(rescued_counts['numbered_by_num'][n]) if rescued_counts['numbered_by_num'].get(n) else ''})"
                for n in sorted(numbered.keys())
            )
        else:
            numbered_str = "(none)"
        print(f"  numbered REMARK groups    : {numbered_str}")
        print(f"  REMARK QCB                : {len(grouped['QCB'])}{_rescued_suffix(rescued_counts['QCB'])}")
        print(f"  REMARK rfd3_property      : {len(grouped['rfd3_property'])}{_rescued_suffix(rescued_counts['rfd3_property'])}")
        dp_total = len(grouped["DESIGN_PATH"])
        dp_rescued = rescued_counts["DESIGN_PATH"]
        dp_extras = []
        if dp_rescued:
            dp_extras.append(f"rescued {dp_rescued} from input")
        if added_predesign_line:
            dp_extras.append("+1 new predesign_cart_relax output line")
        print(f"  REMARK DESIGN_PATH        : {dp_total}"
              + (f"  ({'; '.join(dp_extras)})" if dp_extras else ""))
        if grouped["_misc"]:
            print(f"  REMARK other              : {len(grouped['_misc'])}{_rescued_suffix(rescued_counts['_misc'])}")
        if include_all_header_lines:
            print(f"  HEADER/EXPDTA kept        : {len(header_kept)}")
        else:
            if n_dropped_header:
                print(f"  HEADER/EXPDTA dropped     : {n_dropped_header}  (pass --include_all_header_lines to keep)")
            if n_dropped_remark220:
                print(f"  REMARK 220 dropped        : {n_dropped_remark220}  (pass --include_all_header_lines to keep)")
        if n_dropped_remark0:
            print(f"  REMARK 0 corrupt-tag      : {n_dropped_remark0} dropped  "
                  "(PyRosetta serialization of non-numeric tags; full versions rescued from input)")
        print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Idealize RFDiffusion3 scaffolds (MPNN pre-design + cartesian relax)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --pdb structure.pdb --params ligand.params
  %(prog)s --pdb s.pdb --params L.params --skip_mpnn
  %(prog)s --pdb s.pdb --params L.params --mobile_radius 12.5   # restrict mobile region
""",
    )
    parser.add_argument("--pdb", type=str, required=True, help="Input PDB from RFDiffusion3")
    parser.add_argument("--json", type=str, default=None, help="RFDiffusion3 JSON (auto-detect if omitted)")
    parser.add_argument("--corresponding_json_dir", type=str, default=None,
                        help="Directory to search for JSON if not next to PDB")
    parser.add_argument("--params", type=str, nargs="+", help="Ligand parameter file(s)")
    parser.add_argument("--output", type=str, default=None, help="Output PDB path")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")

    # Geometry / constraint parameters
    parser.add_argument("--mobile_radius", type=float, default=None,
                        help="Å radius for mobile region around fixed/ligand residues. "
                             "Omit this flag (default) to make the whole protein mobile.")
    parser.add_argument("--cart_bonded_weight", type=float, default=4.0,
                        help="Weight for the cart_bonded score term. Default 4.0 "
                             "(sweet spot from weight sweep — drives actionable bond/angle "
                             "deviations ~60%% lower than cb=1 while keeping a negative total "
                             "score and no induced clashes). Going to 10 wrings out another "
                             "~30%% strain but makes total_score positive and can introduce "
                             "clashes. Drop to 0.5-1.0 for a more balanced scorefunction.")
    parser.add_argument("--coord_cst_weight", type=float, default=100.0)
    parser.add_argument("--coord_cst_stdev", type=float, default=0.01)
    parser.add_argument("--covalent_contact_threshold", type=float, default=2.5,
                        help="Å heavy-atom distance cutoff below which fixed-fixed atom "
                             "pairs are treated as declared covalent contacts (auto-detected, "
                             "excused from clash metrics). Reference ranges: covalent bonds "
                             "< 1.9, Zn/Fe-N/O metal-ligand ~2.0-2.2, Zn-S ~2.4, H-bonds "
                             "(heavy-atom) > 2.7. Default 2.5 covers bonds + metal coordination "
                             "+ close ionic contacts without catching H-bonds.")

    # MPNN options
    parser.add_argument("--skip_mpnn", action="store_true",
                        help="Skip the ligandMPNN pre-design step (MPNN is on by default).")
    parser.add_argument("--mpnn_num_designs", type=int, default=10)
    parser.add_argument("--mpnn_temperature", type=float, default=0.1)
    parser.add_argument("--mpnn_omit_aa", type=str, default="",
                        help="Amino acids to omit from MPNN sampling. '' (default) = all 20 allowed.")

    # Protocol options
    parser.add_argument("--idealize_ss", action="store_true")
    parser.add_argument("--skip_fastrelax", action="store_true")
    parser.add_argument("--skip_minimize", action="store_true")
    parser.add_argument("--fastrelax_cycles", type=int, default=3,
                        help="Number of FastRelax cycles. Default 3 — more cycles = "
                             "better idealization at higher cost. 2 is faster; 5+ is overkill.")
    parser.add_argument("--min_tolerance", type=float, default=5e-5,
                        help="Final minimization tolerance. Default 5e-5 (tight). "
                             "Use 1e-4 for faster but less refined geometry.")

    # Output options
    parser.add_argument("--debug", action="store_true",
                        help="Include verbose metrics (per-atom displacements, timing, clashing list).")
    parser.add_argument("--no_ca_rmsd", action="store_true",
                        help="Skip CA-RMSD computation (saves one pose clone, ~10-50 MB). "
                             "CA-RMSD is computed by default and compares the input PDB backbone "
                             "to the final idealized+MPNN'd backbone (input pose captured BEFORE MPNN).")
    parser.add_argument("--no_metric_json_output", action="store_true")
    parser.add_argument("--keep_pdb_energies", action="store_true")
    parser.add_argument("--no_input_baseline_metrics", action="store_true",
                        help="Skip computing validation metrics on the input pose and skip the "
                             "'input_metrics' / 'change_from_input' blocks in the output JSON. "
                             "Default (off) computes the input baseline ONCE at the start of run() "
                             "and writes per-metric deltas (final - input).")
    parser.add_argument("--no_input_seq_in_mpnn_selection", action="store_true",
                        help="Disable the default behavior of scoring the input pose's sequence "
                             "as an extra candidate alongside MPNN's N designs. When this flag is "
                             "set, the script picks strictly the best-of-N MPNN candidate even if "
                             "all of them score worse on the composite than the input. Default "
                             "(off) means: if MPNN doesn't improve on the input, the original "
                             "sequence is retained and the dump+reload H-fix is skipped.")
    parser.add_argument("--hbond_conserve_prob", type=float, default=0.0,
                        help="Probability (0.0-1.0) of fixing each non-catalytic, non-ligand "
                             "residue whose SIDECHAIN makes an hbond to the ligand or to a "
                             "catalytic residue (any heavy atom on the catres/ligand side). "
                             "Promoted residues are added to MPNN's do_not_repack_positions set "
                             "so their identity and rotamer are preserved. Backbone hbonds are "
                             "ignored. Default 0.0 = disabled (legacy behavior). "
                             "Typical production: 0.8.")

    # Post-processing of REMARK lines in the output PDB. Reorder is always on
    # at the end of the run; these flags only tune what gets kept and whether
    # the new predesign DESIGN_PATH line is appended.
    parser.add_argument("--include_all_header_lines", action="store_true",
                        help="Keep HEADER, EXPDTA THEORETICAL MODEL, and REMARK 220 lines in the output PDB. "
                             "By default these are stripped during REMARK reorganization.")
    parser.add_argument("--no_predesign_design_path_remark", action="store_true",
                        help="Disable the default behavior of appending "
                             "'REMARK DESIGN_PATH predesign_cart_relax output <output_path>' "
                             "below the existing DESIGN_PATH lines in the output PDB.")

    args = parser.parse_args()

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
        "-unmute core.optimization.CartesianMinimizer",
    ]
    init_cmd = " ".join([f for f in init_flags if f])

    print("Initializing PyRosetta...")
    print(f"  Using {NPROC} thread(s)")
    pyr.init(init_cmd)
    print("✓ PyRosetta initialized\n")

    output_path = args.output
    if output_path is None:
        pdbname = os.path.basename(args.pdb).replace(".pdb", "")
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            output_path = f"{args.output_dir}/{pdbname}_idealized.pdb"
        else:
            output_path = f"{pdbname}_idealized.pdb"

    idealizer = RFDiffusion3GeometryIdealizer(
        pdb_path=args.pdb,
        json_path=args.json,
        json_dir=args.corresponding_json_dir,
        params=args.params,
        mobile_radius=args.mobile_radius,
        cart_bonded_weight=args.cart_bonded_weight,
        coord_cst_weight=args.coord_cst_weight,
        coord_cst_stdev=args.coord_cst_stdev,
        covalent_contact_threshold=args.covalent_contact_threshold,
        run_mpnn=not args.skip_mpnn,
        mpnn_num_designs=args.mpnn_num_designs,
        mpnn_temperature=args.mpnn_temperature,
        mpnn_omit_aa=args.mpnn_omit_aa,
        compute_ca_rmsd=not args.no_ca_rmsd,
        debug=args.debug,
        no_metric_json_output=args.no_metric_json_output,
        compute_input_baseline_metrics=not args.no_input_baseline_metrics,
        include_input_in_mpnn_selection=not args.no_input_seq_in_mpnn_selection,
        hbond_conserve_prob=args.hbond_conserve_prob,
    )
    idealizer.run(
        output_path=output_path,
        do_idealize_ss=args.idealize_ss,
        do_fastrelax=not args.skip_fastrelax,
        do_minimize=not args.skip_minimize,
        fastrelax_cycles=args.fastrelax_cycles,
        min_tolerance=args.min_tolerance,
        strip_pdb_energies=not args.keep_pdb_energies,
    )

    # Post-processing: reorder REMARK lines into a canonical layout, RESCUE
    # any REMARK types PyRosetta silently drops on its load/dump round-trip
    # (REMARK QCB, REMARK DESIGN_PATH, REMARK rfd3_property) by reading them
    # off disk from the original input PDB, and append the predesign_cart_relax
    # DESIGN_PATH line. Runs at the very end of main() so all Rosetta IO is
    # already complete — this only ever touches the dumped output PDB on disk.
    reorganize_output_pdb_remarks(
        output_pdb_path=output_path,
        input_pdb_path=args.pdb,
        include_all_header_lines=args.include_all_header_lines,
        add_predesign_design_path=not args.no_predesign_design_path_remark,
        verbose=True,
    )

    print("=" * 70)
    print("✓ GEOMETRY IDEALIZATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
