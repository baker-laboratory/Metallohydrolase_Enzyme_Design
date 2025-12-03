#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 13:11:08 2023

@author: ikalvet, donghyo
"""

import pandas as pd
import matplotlib.pyplot as plt
import pyrosetta as pyr
import pyrosetta.rosetta
import pyrosetta.distributed.io
import numpy as np
import glob, os, sys
from shutil import copy2
import multiprocessing
import argparse
import math
from textwrap import wrap

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "enzyme_design", "utils"))
import design_utils
import scoring_utils
sys.path.append(os.path.join(script_dir, "enzyme_design", "invrotzyme", "utils"))
import align_pdbs

import warnings
warnings.filterwarnings('ignore')

comparisons = {'<=': '__le__',
               '<': '__lt__',
               '>': '__gt__',
               '>=': '__ge__',
               '=': '__eq__'}


def get_pocket_residues(pose):
    ligands = [res for res in pose.residues if res.is_ligand()]
    # heavyatoms = [ligand.atom_name(n+1).strip() for n in range(ligand.natoms()) if ligand.atom_type(n+1).element() != "H"]
    heavyatoms = {ligand.seqpos(): [ligand.atom_name(n+1).strip() for n in range(ligand.natoms()) if ligand.atom_type(n+1).element() != "H"] for ligand in ligands}

    pocket_residues = []
    for res in pose.residues:
        if res.is_ligand():
            continue
        if min([(res.nbr_atom_xyz() - lig.nbr_atom_xyz()).norm() for lig in ligands]) > 15.0:
            continue
        for ligand in ligands:
            if res.seqpos() in pocket_residues:
                continue
            for ha in heavyatoms[ligand.seqpos()]:
                if res.seqpos() in pocket_residues:
                    break
                if (res.xyz("CA") - ligand.xyz(ha)).norm() < 6.0:
                    pocket_residues.append(res.seqpos())
                    break
                for an in range(res.natoms()):
                    if (res.xyz(an) - ligand.xyz(ha)).norm() < 4.0:
                        pocket_residues.append(res.seqpos())
                        break
    pocket_residues_pdb = [pose.pdb_info().number(rn) for rn in pocket_residues]
    return pocket_residues, pocket_residues_pdb

def get_residue_rmsd(residue1, residue2, specified_atoms=None):
    if residue1.name3() != residue2.name3():
        return None
    else:
        atoms = [residue1.atom_name(n).strip() for n in range(1, residue1.natoms()+1) if not residue1.atom_is_hydrogen(n)]        
        if specified_atoms:
            atoms = specified_atoms
        else:
            atoms = [residue1.atom_name(n).strip() for n in range(1, residue1.natoms()+1) if not residue1.atom_is_hydrogen(n)]
        ref_coords = [residue2.xyz(a) for a in atoms]
        mdl_coords = [residue1.xyz(a) for a in atoms]
        rmsd = np.sqrt(sum([(np.linalg.norm(c1-c2))**2 for c1, c2 in zip(ref_coords, mdl_coords)])/len(atoms))
        """
        if specified_atoms:
            print (atoms)
            print ([list(x) for x in ref_coords])
            print ([list(x) for x in mdl_coords])
            print ([np.linalg.norm(c1-c2) for c1, c2 in zip(ref_coords, mdl_coords)])
        """
        return rmsd


#Pnear calculation function by Vikram from tools/analysis/compute_pnear.py
# Given a vector of scores, a matching vector of rmsds, and values for lambda and kbt,
# compute the PNear value.
def calculate_pnear( scores, rmsds, lambda_val=1.5, kbt=0.62 ):
    nscores = len(scores)
    assert nscores == len(rmsds), "Error in calculate_pnear(): The scores and rmsds lists must be of the same length."
    assert nscores > 0, "Error in calculate_pnear(): At least one score/rmsd pair must be provided."
    assert kbt > 1e-15, "Error in calculate_pnear(): kbt must be greater than zero!"
    assert lambda_val > 1e-15, "Error in calculate_pnear(): lambda must be greater than zero!"
    minscore = min( scores )
    weighted_sum = 0.0
    Z = 0.0
    lambdasq = lambda_val * lambda_val
    for i in range( nscores ) :
        val1 = math.exp( -( rmsds[i] * rmsds[i] ) / lambdasq )
        val2 = math.exp( -( scores[i] - minscore ) / kbt )
        weighted_sum += val1*val2
        Z += val2
    assert Z > 1e-15, "Math error in calculate_pnear()!  This shouldn't happen."
    return weighted_sum/Z


def load_poses(models):
    poses = {}
    for i, mdl in enumerate(models):
        poses[i] = pyrosetta.distributed.io.pose_from_pdbstring(mdl).pose.clone()
    return [poses[n].clone() for n in range(len(poses))]


def filter_scores(scores, filters):
    """
    Filters are defined in this importable module
    """
    filtered_scores = scores.copy()

    for s in filters.keys():
        if filters[s] is not None and s in scores.keys():
            val = filters[s][0]
            sign = comparisons[filters[s][1]]
            filtered_scores =\
              filtered_scores.loc[(filtered_scores[s].__getattribute__(sign)(val))]
            n_passed = len(scores.loc[(scores[s].__getattribute__(sign)(val))])
            print(f"{s:<24} {filters[s][1]:<2} {val:>7.3f}: {len(filtered_scores)} "
                  f"designs left. {n_passed} pass ({(n_passed/len(scores))*100:.0f}%).")
    return filtered_scores

arguments = sys.argv.copy()


parser = argparse.ArgumentParser()
parser.add_argument("--pdb", type=str, nargs="+", help="PDB files used as input(s) for PLACER")
parser.add_argument("--pdb_path", type=str, help="Path to PDB files used as input(s) for PLACER")

parser.add_argument("--placer_pdb_path", type=str, help="Path to PDB files used as output(s) for PLACER")

parser.add_argument("--lig_name", type=str, help="Name of target ligand")
parser.add_argument("--lig_atom", type=str, nargs="+", help="Name of ions of target ligand")

parser.add_argument("--top", default=5, required=False, help="Number of top ranked structures to investigate")

parser.add_argument("--scorefile", type=str, nargs="+", help="PLACER output scorefiles")
parser.add_argument("--scorefile_list", type=str, help="File with a list of PLACER output scorefiles")
parser.add_argument("--params", nargs="+", required=False, help="params files")
parser.add_argument("--nproc", type=int, default=os.cpu_count(), help="How many CPU cores")
parser.add_argument("--outdir", type=str, default="./", help="Output directory")
parser.add_argument("--scorefile_out", type=str, default="scorefile.txt", help="Output scorefile name")
parser.add_argument("--dump", action="store_true", default=False, help="Dump top 5 models as full protein PDB files")

args = parser.parse_args()


extra_res_fa = ""
if args.params is not None:
    extra_res_fa = "-extra_res_fa "
    for p in args.params:
        extra_res_fa += f"{p} "
else:
    extra_res_fa = ""

pyr.init(f"{extra_res_fa} -mute all -beta_nov16 -run:preserve_header")
sfx = pyr.get_fa_scorefxn()

#os.makedirs("top5_models", exist_ok=True)

if args.scorefile_list is not None and args.scorefile is None:
    args.scorefile = open(args.scorefile_list, "r").readlines()
    args.scorefile = [x.strip()+".csv" if ".csv" not in x else x.strip() for x in args.scorefile]

if args.pdb_path is not None and args.pdb is None:
    args.pdb = glob.glob(args.pdb_path + "/*.pdb")

## Reading the scorefiles
scores = pd.DataFrame()
for scf in args.scorefile:
    scores = pd.concat([scores, pd.read_csv(scf)], ignore_index=True)

designs = sorted(list(set(scores.label.values)))

# Assigning input PDB's to design names
ref_pdbs = {}
for d in designs:
    for p in args.pdb:
        if os.path.basename(p).replace(".pdb", "") == d:
            ref_pdbs[d] = p
    if d not in ref_pdbs.keys():
        print(f"Can't find reference PDB for design: {d}")

chi_pairs = [(1, 4), (2, 5), (3, 6), (4, 1), (5, 2), (6, 3)]

############################
### MAIN FILTERING BLOCK ###
############################

the_queue = multiprocessing.Queue()
manager = multiprocessing.Manager()

results = manager.dict()

for i, d in enumerate(designs):
    the_queue.put((i, d))

def process(q):
    while True:
        asd = q.get(block=True)
        if asd is None:
            return
        i = asd[0]
        d = asd[1]

        pocket_residues = {}
        pocket_residues_pdb = {}
        pocket_residue_rmsds = {}
        catalytic_residues = {}
        catalytic_residue_rmsds = {}
        model_res_scores = {}
    
        DF = pd.DataFrame()
        _df = scores.loc[scores.label.str.contains(d)]
    
        DF.at[i, "description"] = d
        DF.at[i, "kabsch"] = _df.kabsch.mean()
    
        ## Loading models
        pdbstr = open(os.path.join(args.placer_pdb_path, d + "_model.pdb"), "r").read()
        models = []
        for mdl in pdbstr.split("ENDMDL"):
            if len(mdl) < 10:
                continue
            models.append(mdl)
        model_poses = load_poses(models)  # not multiprocessed
        
        ref_pose = pyr.pose_from_file(ref_pdbs[d])
        
        ## Extracting pocket residue numbers
        best_plddt_idx = _df.sort_values("plddt", ascending=False).iloc[0]['model_idx']
        p = model_poses[best_plddt_idx-1].clone()

        pr, prpdb = get_pocket_residues(p)
        
        pocket_residues[d] = []
        pocket_residues_pdb[d] = []
        
        for r in prpdb:
            if p.pdb_rsd((ref_pose.pdb_info().chain(r), r)) is None:
                continue
            pocket_residues_pdb[d].append(r)
            pocket_residues[d].append(p.pdb_rsd((ref_pose.pdb_info().chain(r), r)).seqpos())
            assert p.residue(pocket_residues[d][-1]).name3() == ref_pose.residue(r).name3()

        catalytic_residues[d] = design_utils.get_matcher_residues(ref_pdbs[d])
        print (catalytic_residues)
        
        ## Extracting pocket residue scores
        model_res_scores[d] = pd.DataFrame()        
        for j, mdl in enumerate(models):
            _mdl = mdl.split("\n")
            model_res_scores[d].at[len(model_res_scores[d])+1, "iter"] = i
            for resno in pocket_residues_pdb[d]:
                res_lines = [l for l in _mdl if f"A{resno:>4}" in l]
                res_scores = [float(l[61:67]) for l in res_lines]
                model_res_scores[d].at[len(model_res_scores[d]), resno] = np.average(res_scores)
                
        ## Calculating catalytic residue info
        catalytic_residue_rmsds[d] = pd.DataFrame()
        
        ## Extracting ligand info
        ref_lig_seqpos = None
        ref_ligands = [res for res in ref_pose.residues if res.is_ligand()]
        if len(ref_ligands) > 1:
            for res in ref_ligands:
                if res.name3() == args.lig_name:
                    ref_lig_seqpos = res.seqpos()
        else:
            assert ref_ligands[0].name3() == args.lig_name
            ref_lig_seqpos = ref_ligands[0].seqpos()

        ## Calculating ligand metrics
        lig_rmsds = []
        
        mdl_ligands = [res for res in model_poses[0].residues if res.is_ligand()]
        
        for j, p in enumerate(model_poses):
            lig_mdl_pos = p.size()
            for mdl_ligand in mdl_ligands:
                if mdl_ligand.name3() == args.lig_name:
                    mdl_lig_seqpos = mdl_ligand.seqpos()
                    break
            lig_rmsds.append(get_residue_rmsd(ref_pose.residue(ref_lig_seqpos), p.residue(mdl_lig_seqpos), args.lig_atom))
            _df.at[j, "rmsd_ligand"] = lig_rmsds[-1]
        DF.at[i, "rmsd_ligand"] = np.average(lig_rmsds)
        #DF.at[i, "rmsd_ligand"] = np.median(lig_rmsds)
        DF.at[i, "rmsd_std_ligand"] = np.std(lig_rmsds)
        
        ## prmsd average / min
        DF.at[i, f"prmsd_avr"] = _df.plddt.mean()
        DF.at[i, f"prmsd_min"] = _df.plddt.min()
                
        ## plddt and rmsd based on sorting by plddt (1D track information)
        DF.at[i, f"plddt_top{args.top}"] = _df.sort_values("plddt", ascending=False).plddt.iloc[:int(args.top)].mean()

        ## plddt and rmsd based on sorting by plddt-pde (2D track information)
        DF.at[i, f"rmsd_pde_top{args.top}"] = _df.sort_values("plddt_pde", ascending=False).rmsd_ligand.iloc[:int(args.top)].mean()
        DF.at[i, f"plddt_pde_top{args.top}"] = _df.sort_values("plddt_pde", ascending=False).plddt_pde.iloc[:int(args.top)].mean()

        ## prmsd and rmsd based on sorting by prmsd (ligand prediction confidence information)
        DF.at[i, f"prmsd_top{args.top}"] = _df.sort_values("prmsd", ascending=True).prmsd.iloc[:5].mean()
        DF.at[i, f"rmsd_prmsd_top{args.top}"] = _df.sort_values("prmsd", ascending=True).rmsd_ligand.iloc[:5].mean()

        ## pnear values for rmsd of ligand0 vs lddt / plddt / plddt_pde
        DF.at[i, "lddt_pnear"] = calculate_pnear(np.array(-1*_df["lddt"]), np.array(_df["rmsd_ligand"]), lambda_val=1.5, kbt=0.62 )
        DF.at[i, "plddt_pnear"] = calculate_pnear(np.array(-1*_df["plddt"]), np.array(_df["rmsd_ligand"]), lambda_val=1.5, kbt=0.62 )
        DF.at[i, "plddt_pde_pnear"] = calculate_pnear(np.array(-1*_df["plddt_pde"]), np.array(_df["rmsd_ligand"]), lambda_val=1.5, kbt=0.62 )
        DF.at[i, "prmsd_pnear"] = calculate_pnear(np.array(_df["prmsd"]), np.array(_df["rmsd_ligand"]), lambda_val=1.5, kbt=0.62 )

        ## pnear values for rmsd of ligand0 vs lddt / plddt / plddt_pde
        for jj, cr in enumerate(catalytic_residues[d]):
            for j, p in enumerate(model_poses):
                try:
                    _cr_in_crop = pocket_residues[d][pocket_residues_pdb[d].index(cr)]
                except ValueError:
                    print(d)
                    print(best_plddt_idx)
                    print(cr)
                    print(pocket_residues[d])
                    print(pocket_residues_pdb[d])
                    sys.exit(1)
                catalytic_residue_rmsds[d].at[j, cr] = get_residue_rmsd(ref_pose.residue(cr), p.residue(_cr_in_crop))
            
            # If catalytic residues are not in the pocket.
            if not cr in catalytic_residue_rmsds[d]:
                DF.at[i, f"rmsd_catres{jj}"] = np.nan
                DF.at[i, f"rmsd_std_catres{jj}"] = np.nan
            else:
                DF.at[i, f"rmsd_catres{jj}"] = np.median(list(filter(lambda x: ~np.isnan(x), catalytic_residue_rmsds[d][cr])))
                DF.at[i, f"rmsd_std_catres{jj}"] = np.std(list(filter(lambda x: ~np.isnan(x), catalytic_residue_rmsds[d][cr])))
            if not cr in model_res_scores[d]:    
                DF.at[i, f"u_catres{jj}"] = np.nan
                DF.at[i, f"u_std_catres{jj}"] = np.nan
            else:
                DF.at[i, f"u_catres{jj}"] = np.average(list(filter(lambda x: ~np.isnan(x), model_res_scores[d][cr])))
                DF.at[i, f"u_std_catres{jj}"] = np.std(list(filter(lambda x: ~np.isnan(x), model_res_scores[d][cr])))
                
        ## Calculating pocket residue metrics
        pocket_residue_rmsds[d] = pd.DataFrame()
        for pr_no, prpdb_no in zip(pocket_residues[d], pocket_residues_pdb[d]):
            for j, p in enumerate(model_poses):
                pocket_residue_rmsds[d].at[j, prpdb_no] = get_residue_rmsd(ref_pose.residue(prpdb_no), p.residue(pr_no))
    
        DF.at[i, "rmsd_pocket"] = pocket_residue_rmsds[d][pocket_residues_pdb[d]].median().median()
        DF.at[i, "rmsd_pocket_worst"] = pocket_residue_rmsds[d][pocket_residues_pdb[d]].median().max()
    
        DF.at[i, "u_pocket"] = model_res_scores[d][pocket_residues_pdb[d]].median().median()
        DF.at[i, "u_pocket_std"] = model_res_scores[d][pocket_residues_pdb[d]].std().median()
        DF.at[i, "u_pocket_worst"] = model_res_scores[d][pocket_residues_pdb[d]].median().max()
        

        # Dumping the CSV file again because of an added rmsd column
        #_df.to_csv(args.scorefile, index=False)
        results[i] = DF.copy()

pool = multiprocessing.Pool(processes=args.nproc,
                        initializer=process,
                        initargs=(the_queue, ))
# None to end each process
for _i in range(args.nproc):
    the_queue.put(None)

# Closing the queue and the pool
the_queue.close()
the_queue.join_thread()
pool.close()
pool.join()

DF = pd.DataFrame()
for i in results.keys():
    for k in results[i].keys():
        DF.at[i, k] = results[i].at[i, k]

DF.to_csv(args.scorefile_out)