#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 19:23:16 2023

@author: ikalvet
"""
import pyrosetta as pyr
import pyrosetta.rosetta
import os, sys
import pandas as pd

sys.path.append("./scripts/enzyme_design")
import no_ligand_repack
import scoring_utils



comparisons = {'<=': '__le__',
               '<': '__lt__',
               '>': '__gt__',
               '>=': '__ge__',
               '=': '__eq__'}


def fix_scorefxn(sfxn, allow_double_bb=False):
    opts = sfxn.energy_method_options()
    opts.hbond_options().decompose_bb_hb_into_pair_energies(True)
    opts.hbond_options().bb_donor_acceptor_check(not allow_double_bb)
    sfxn.set_energy_method_options(opts)


def score_design(pose, sfx, catres):
    df_scores = pd.DataFrame()
    from scoring_utils import calculate_ddg

    sfx(pose)
    for k in pose.scores:
        df_scores.at[0, k] = pose.scores[k]

    fix_scorefxn(sfx)
    
    if pose.constraint_set().has_constraints():
        sfx.set_weight(pyrosetta.rosetta.core.scoring.score_type_from_name("atom_pair_constraint"), 1.0)
        sfx.set_weight(pyrosetta.rosetta.core.scoring.score_type_from_name("angle_constraint"), 1.0)
        sfx.set_weight(pyrosetta.rosetta.core.scoring.score_type_from_name("dihedral_constraint"), 1.0)
        sfx(pose)
        df_scores.at[0, 'all_cst'] = sum([pose.scores[s] for s in pose.scores if "constraint" in s])

    df_scores.at[0, 'corrected_ddg'] = calculate_ddg(pose, sfx, catres[0])

    # Calculating relative ligand SASA
    # First figuring out what is the path to the ligand PDB file
    ligands = [res for res in pose.residues if res.is_ligand() and not res.is_virtual_residue()]
    ligand = ligands[0]

    ligand_seqpos = ligand.seqpos()
    ligand_name = ligand.name3()

    ligand_pose = pyrosetta.rosetta.core.pose.Pose()
    pyrosetta.rosetta.core.pose.append_subpose_to_pose(ligand_pose, pose, pose.size(), pose.size(), 1)

    free_ligand_sasa = scoring_utils.getSASA(ligand_pose, resno=1)
    ligand_sasa = scoring_utils.getSASA(pose, resno=ligand_seqpos)
    df_scores.at[0, 'L_SASA'] = ligand_sasa / free_ligand_sasa

    ## Calculating SASA of substrate atoms
    ligand = pose.residue(ligand_seqpos)
    substrate_atomnos = [n for n in range(1, ligand.natoms()+1) if ligand.atom_name(n).strip() not in ["ZN1", "O1"] and ligand.atom_type(n).element() != "H"]
    df_scores.at[0, 'substrate_SASA'] = scoring_utils.getSASA(pose, ligand_seqpos, substrate_atomnos)


    # Finding H-bond partners to the activated water
    target_atoms = ["H1"]
    HBond_res = 0
    for target_atom in target_atoms:
        for res in pose.residues:
            if not res.is_protein():
                continue
            if res.name3() not in ["ASP", "GLU"]:
                continue
            if (pose.residue(ligand_seqpos).xyz(target_atom) - res.xyz('CA')).norm() < 10.0:
                for atomno in range(1, res.natoms()+1):
                    if not res.heavyatom_is_an_acceptor(atomno):
                        continue
                    if (pose.residue(ligand_seqpos).xyz(target_atom) - res.xyz(atomno)).norm() < 2.0:
                        HBond_res += 1
                        break

    df_scores.at[0, "H2O_hbond"] = 0.0
    if HBond_res > 0:
        df_scores.at[0, "H2O_hbond"] = 1.0

    ## Finding Hbonds to oxyanion
    # Assuming it's always called O2
    df_scores.at[0, "O2_hbond"] = scoring_utils.find_hbonds_to_residue_atom(pose, ligand_seqpos, "O2")
    df_scores.at[0, 'oxy_hbond'] = float(df_scores.at[0, "O2_hbond"] > 1)


    # Calculating ContactMolecularSurface
    cms = pyrosetta.rosetta.protocols.simple_filters.ContactMolecularSurfaceFilter()
    lig_sel = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(ligand_seqpos)
    protein_sel = pyrosetta.rosetta.core.select.residue_selector.ChainSelector("A")
    cms.use_rosetta_radii(True)
    cms.distance_weight(0.5)
    cms.selector1(protein_sel)
    cms.selector2(lig_sel)
    df_scores.at[0, "cms"] =  cms.compute(pose)  # ContactMolecularSurface
    df_scores.at[0, "cms_per_atom"] =  df_scores.at[0, "cms"] / pose.residue(ligand_seqpos).natoms()  # ContactMolecularSurface per atom

    ## Calculating shape complementarity
    sc = pyrosetta.rosetta.protocols.simple_filters.ShapeComplementarityFilter()
    sc.use_rosetta_radii(True)
    sc.selector1(protein_sel)
    sc.selector2(lig_sel)
    df_scores.at[0, "sc"] = sc.score(pose)


    # Running no-ligand-repack
    nlr_scores = no_ligand_repack.no_ligand_repack(pose, pyr.get_fa_scorefxn(), ligand_resno=ligand_seqpos)
    for k in nlr_scores.keys():
        df_scores.at[0, k] = nlr_scores.iloc[0][k]

    df_scores.at[0, "score_per_res"] = df_scores.at[0, "total_score"]/pose.size()

    return df_scores


def filter_scores(scores, filter_dict=None):
    """
    Filters are defined in this importable module
    """
    filtered_scores = scores.copy()

    if filter_dict is None:
        filter_dict = filters

    for s in filter_dict.keys():
        if filter_dict[s] is not None and s in scores.keys():
            val = filter_dict[s][0]
            sign = comparisons[filter_dict[s][1]]
            filtered_scores =\
              filtered_scores.loc[(filtered_scores[s].__getattribute__(sign)(val))]
            n_passed = len(scores.loc[(scores[s].__getattribute__(sign)(val))])
            print(f"{s:<24} {filter_dict[s][1]:<2} {val:>7.3f}: {len(filtered_scores)} "
                  f"designs left. {n_passed} pass ({(n_passed/len(scores))*100:.0f}%).")
    return filtered_scores


filters = {"L_SASA": [0.20, "<="],
           "substrate_SASA": [2.0, ">="],
           "H2O_hbond": [1.0, "="],
           "oxy_hbond": [1.0, "="],
           "cms_per_atom": [5.0, ">="],
           "corrected_ddg": [-30.0, "<="],
           "nlr_totrms": [0.8, "<="]}
           # "nlr_SR1_rms": [0.6, "<="],
           # "nlr_SR2_rms": [0.6, "<="],
           # "nlr_SR3_rms": [0.6, "<="]}

align_atoms = ["ZN1", "O1", "C1"]

