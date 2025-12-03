#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Indrek Kalvet
ikalvet@uw.edu
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import time
import queue
import threading
import multiprocessing
import argparse
from shutil import copy2
import pyrosetta as pyr
import pyrosetta.rosetta
import pyrosetta.distributed.io
#sys.path.append("./../git/rf_diffusion_repo")
import itertools
import json
import copy

script_dir = os.path.dirname(os.path.abspath(__file__))

DAB = os.path.join(script_dir, "enzyme_design", "DAlphaBall.gcc")

aa3to1 = {
    "ALA":'A', "ARG":'R', "ASN":'N', "ASP":'D', "CYS":'C',
    "GLN":'Q', "GLU":'E', "GLY":'G', "HIS":'H', "ILE":'I',
    "LEU":'L', "LYS":'K', "MET":'M', "PHE":'F', "PRO":'P',
    "SER":'S', "THR":'T', "TRP":'W', "TYR":'Y', "VAL":'V' }

aa1to3 = {val: k for k, val in aa3to1.items()}


def align_coords(ref_coords, mobile_coords, align_atoms):
    
    def centroid(X):
        # return the mean X,Y,Z down the atoms
        return np.mean(X, axis=0, keepdims=True)
    # Aligning xyz2 to xyz1
    xyz1 = np.array([ref_coords[x] for x in align_atoms["ref"]])
    xyz2 = np.array([mobile_coords[x] for x in align_atoms["mobile"]])
    
    rmsd, _, R = kinematics.np_kabsch(xyz1, xyz2)
    centroid1 = centroid(xyz1)
    centroid2 = centroid(xyz2)
    mobile_coords2 = np.copy(mobile_coords) - centroid2
    mobile_coords2 = mobile_coords2 @ R
    mobile_coords2 += centroid1

    return mobile_coords2


def read_pose_from_str_and_fix_ligand(pdbfile, ref_pdb, align_atoms):
    """
    align_atoms = [n1, n2, n3]
    """
    # _str = pyrosetta.distributed.io.to_pdbstring(pose2)
    pdbff = open(pdbfile, "r").readlines()

    new_pdb = []
    for i, l in enumerate(pdbff):
        if "HETATM" in l:
            break
        new_pdb.append(l)

    ligand_start_line = i
    target_ligand_txt = [l for l in pdbff[ligand_start_line:] if "HETATM" in l]
    ligand_resno = target_ligand_txt[0].split()[5]
    ligand_start_atomno = int(target_ligand_txt[0].split()[1])

    target_coords = np.array([[float(x) for x in l.split()[6:9]] for l in target_ligand_txt])

    ref_pdbf = open(ref_pdb, "r").readlines()
    ref_ligand_txt = [l for l in ref_pdbf if "HETATM" in l]
    ref_coords = np.array([[float(x) for x in l.split()[6:9]] for l in ref_ligand_txt])

    if align_atoms[0][0].isnumeric():
        align_atoms = {"ref": [int(x) for x in align_atoms],
                       "mobile": [int(x) for x in align_atoms]}
    else:
        
        align_atoms = {"ref": [i for i, l in enumerate(target_ligand_txt) if l.split()[2] in align_atoms],
                       "mobile": [i for i, l in enumerate(ref_ligand_txt) if l.split()[2] in align_atoms]}

    aligned_ligand_coords = align_coords(target_coords, ref_coords, align_atoms)

    new_pdb.append(f"{('TER'):<80}\n")
    for i, l in enumerate(ref_ligand_txt):
        line = ""
        line += "HETATM"
        line += f"{(ligand_start_atomno+i):>5}"
        line += l[11:22]  # atomname, resname, chain
        line += f"{ligand_resno:>4}    "
        for n in range(3):
            line += f"{aligned_ligand_coords[i][n]:>8.3f}"
        line += l[54:]
        new_pdb.append(line)

    pose3 = pyrosetta.Pose()
    pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(pose3, "\n".join(new_pdb))
    return pose3


def read_pose_from_str_and_fix_issues(pdbfile, trb):
    """
    align_atoms = [n1, n2, n3]
    """
    # _str = pyrosetta.distributed.io.to_pdbstring(pose2)
    if os.path.exists(pdbfile):
        pdbff = open(pdbfile, "r").readlines()
    else:
        pdbff = pdbfile.split("\n")  # In case contents of a PDB file are provided

    # Replacing all hallucinated longer residues with GLY
    new_pdb = []
    for i, l in enumerate(pdbff):
        if "ATOM" in l:
            if int(l.split()[5])-1 not in trb["con_hal_idx0"]:
                if l.split()[3] in ["VAL", "GLN", "ARG", "LYS", "GLU", "ASN", "ASP", "MET", "PRO"]:
                    new_pdb.append(l[:17]+"GLY"+l[20:])
                else:
                    new_pdb.append(l)
            else:
                new_pdb.append(l)
        elif "HETATM" in l:
            # Fixing the column formatting of ligand atom names
            atom_name = l[11:16].strip()
            ltrs = "".join([x for x in atom_name if not x.isnumeric()])
            nmbrs = "".join([x for x in atom_name if x.isnumeric()])
            atom_name_fix = f"{ltrs:>3}{nmbrs:<2}"
            new_pdb.append(l[:11]+atom_name_fix+l[16:])
        else:
            new_pdb.append(l)

    pose3 = pyrosetta.Pose()
    pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(pose3, "\n".join(new_pdb))
    return pose3


def add_matcher_line_to_pose(pose, resno, ligand_name):
    _str = pyrosetta.distributed.io.to_pdbstring(pose)
    pdbff = _str.split("\n")

    new_pdb = []
    if "ATOM" in pdbff[0]:
        new_pdb.append(f"REMARK 666 MATCH TEMPLATE X {ligand_name}    0 MATCH MOTIF A {pose.residue(resno).name3()}  {resno}  1  1               \n")
        for l in pdbff:
            new_pdb.append(l)
    else:
        for l in pdbff:
            new_pdb.append(l)
            if "HEADER" in l:
                new_pdb.append(f"REMARK 666 MATCH TEMPLATE X {ligand_name}    0 MATCH MOTIF A {pose.residue(resno).name3()}  {resno}  1  1               \n")

    pose3 = pyrosetta.Pose()
    pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(pose3, "\n".join(new_pdb))
    return pose3


def add_matcher_line_to_pose(pose, ref_pose, tgt_residues, ref_residues):
    """
    Takes REMARK 666 lines from ref pose and adjusts them based on the new positions in tgt_residues
    """
    if len(tgt_residues) == 0:
        return pose

    ligand_name = pose.residue(pose.size()).name3()
    # _str_ref = pyrosetta.distributed.io.to_pdbstring(ref_pose).split("\n")
    # _ref_remarks = [l for l in _str_ref if "REMARK 666" in l]

    _new_remarks = []

    for i, r in enumerate(tgt_residues):
        _new_remarks.append(f"REMARK 666 MATCH TEMPLATE {tgt_residues[r]['target_chain']} {tgt_residues[r]['target_name']}"
                            f"  {tgt_residues[r]['target_resno']:>3} MATCH MOTIF {tgt_residues[r]['chain']} "
                            f"{tgt_residues[r]['name3']}  {r:>3}  {tgt_residues[r]['cst_no']}  "
                            f"{tgt_residues[r]['cst_no_var']}               \n")

    _str = pyrosetta.distributed.io.to_pdbstring(pose)
    pdbff = _str.split("\n")

    new_pdb = []
    if "ATOM" in pdbff[0]:
        for lr in _new_remarks:
            new_pdb.append(lr)
        for l in pdbff:
            new_pdb.append(l)
    else:
        for l in pdbff:
            if "HEADER" in l:
                new_pdb.append(l)
                for lr in _new_remarks:
                    new_pdb.append(lr)
            elif "REMARK 666" in l:  # Skipping existing REMARK 666 lines
                continue
            else:
                new_pdb.append(l)
    pose2 = pyrosetta.Pose()
    pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(pose2, "\n".join(new_pdb))
    return pose2


def get_matcher_residues(filename):
    pdbfile = open(filename, 'r').readlines()

    matches = {}
    for line in pdbfile:
        if "ATOM" in line:
            break
        if "REMARK 666" in line:
            lspl = line.split()
            resno = int(lspl[11])

            matches[resno] = {'target_name': lspl[5],
                              'target_chain': lspl[4],
                              'target_resno': int(lspl[6]),
                              'chain': lspl[9],
                              'name3': lspl[10],
                              'cst_no': int(lspl[12]),
                              'cst_no_var': int(lspl[13])}
    return matches


def getSASA(pose, resno=None, SASA_atoms=None, ignore_sc=False):
    """
    Takes in a pose and calculates its SASA.
    Or calculates SASA of a given residue.
    Or calculates SASA of specified atoms in a given residue.

    Procedure by Brian Coventry
    """

    atoms = pyr.rosetta.core.id.AtomID_Map_bool_t()
    atoms.resize(pose.size())

    n_ligands = 0
    for res in pose.residues:
        if res.is_ligand():
            n_ligands += 1

    for i, res in enumerate(pose.residues):
        if res.is_ligand():
            atoms.resize(i+1, res.natoms(), True)
        else:
            atoms.resize(i+1, res.natoms(), not(ignore_sc))
            if ignore_sc is True:
                for n in range(1, res.natoms()+1):
                    if res.atom_is_backbone(n) and not res.atom_is_hydrogen(n):
                        atoms[i+1][n] = True

    surf_vol = pyr.rosetta.core.scoring.packing.get_surf_vol(pose, atoms, 1.4)

    if resno is not None:
        res_surf = 0.0
        for i in range(1, pose.residue(resno).natoms()+1):
            if SASA_atoms is not None and i not in SASA_atoms:
                continue
            res_surf += surf_vol.surf(resno, i)
        return res_surf
    else:
        return surf_vol


def get_ROG(pose):
    centroid = np.array([np.average([res.xyz("CA").__getattribute__(c) for res in pose.residues if res.is_protein()]) for c in "xyz"])
    ROG = max([np.linalg.norm(centroid - res.xyz("CA")) for res in pose.residues if res.is_protein()])
    return ROG


def sidechain_connectivity(res):
    """
    Evaluates the physical correctness of the sidechain of a residue
    """
    ref_res_pose = pyr.pose_from_sequence("A"+res.name1()+"A")
    ref_res = ref_res_pose.residue(2)
    bondlen_deviations = []
    for an in range(1, res.natoms()+1):
        if res.atom_type(an).element() == "H":
            continue
        for nn in res.bonded_neighbor(an):
            if res.atom_type(nn).element() == "H":
                continue
            bondlen_deviations.append(abs((res.xyz(an)-res.xyz(nn)).norm() - (ref_res.xyz(an)-ref_res.xyz(nn)).norm()))
    return max(bondlen_deviations)


def thread_seq_to_pose(pose, sequence, skip_resnos=None):
    if skip_resnos is None:
        skip_resnos = []
    pose2 = pose.clone()
    for i, r in enumerate(sequence):
        if i+1 in skip_resnos:
            continue
        if pose.residue(i+1).is_ligand():
            continue
        mutres = pyrosetta.rosetta.protocols.simple_moves.MutateResidue()
        mutres.set_target(i+1)
        mutres.set_res_name(aa1to3[r])
        mutres.apply(pose2)
    return pose2


def load_multimodel_PDB_to_poses(pdbfile, trb, count=None, random=None):
    pdbstr = open(pdbfile, "r").read()
    models = pdbstr.split("ENDMDL")
    poses = []

    if count == random:
        traj_ids_to_save = list(range(0, traj_N_save+1))
    else:
        traj_ids_to_save = []
        # Figuring out <traj_N_save> random unique step id's
        while len(traj_ids_to_save) < random:
            random_id = np.random.randint(1, count)
            if random_id not in traj_ids_to_save:
                traj_ids_to_save.append(random_id)

    for i, mdl in enumerate(models):
        if len(mdl) < 10:
            continue
        if i > count:
            break
        if i not in traj_ids_to_save:
            continue
        poses.append(read_pose_from_str_and_fix_issues(mdl, trb))
    return poses


def dump_scorefile(df, filename):
    widths = {}
    for k in df.keys():
        if k in ["SCORE:", "description", "name"]:
            widths[k] = 0
        if len(k) >= 12:
            widths[k] = len(k) + 1
        else:
            widths[k] = 12

    with open(filename, "w") as file:
        title = ""
        for k in df.keys():
            if k == "SCORE:":
                title += k
            elif k in ["description", "name"]:
                title += f" {k}"
            else:
                title += f"{k:>{widths[k]}}"
        if all([t not in df.keys() for t in ["description", "name"]]):
            title += f" {'description'}"
        file.write(title + "\n")
        
        for index, row in df.iterrows():
            line = ""
            for k in df.keys():
                if isinstance(row[k], (float, np.float16)):
                    val = f"{row[k]:.3f}"
                else:
                    val = row[k]
                if k == "SCORE:":
                    line += val
                elif k in ["description", "name"]:
                    line += f" {val}"
                else:
                    line += f"{val:>{widths[k]}}"
            if all([t not in df.keys() for t in ["description", "name"]]):
                line += f" {index}"
            file.write(line + "\n")

if __name__ == "__main__":
# def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--pdb", nargs="+", type=str, help="Input PDBs from RFD2")
    parser.add_argument("--pdbpath", nargs="+", type=str, help="Directories where input PDBs can be found")
    parser.add_argument("--trb", nargs="+", type=str, help="(optional) TRB files from RFD2")
    parser.add_argument("--ref", nargs="+", type=str, help="Reference PDBs used as input for RFD2. If not provided, then the reference structure will be taken from TRB files.")
    parser.add_argument("--analyze", action="store_true", default=False, help="Analyze only. Will not move any files. Will calculate all metrics and take a bit longer.")

    parser.add_argument("--fix", action="store_true", default=False, help="Fixes issues with broken residue sidechains. Need to use it if the script fails because RFD2 has produced broken sidechains.")
    parser.add_argument("--traj", help="How many steps of the trajectory of a design should be parsed.\nUse <N> to pick last N structures from the trajcetory. Use <N>/<M> to pick random <N> structures from the last <M> steps."
                        "Disabled by default.\nAssumes that for a design PDB 'path/pdbfile.pdb' a trajectory file can be found at '/path/traj/pdbfile_pX0_traj.pdb'")
    parser.add_argument("--rethread", action="store_true", default=False, help="Rethread the existing sequence to the backbone. This will fix sidechain weirdnesses coming from RFD2.")
    parser.add_argument("--params", nargs="+", type=str, help="Params files of ligands and noncanonicals")

    parser.add_argument("--lig_dist", default=2.5, type=float, help="(default 2.5) Cutoff for smallest allowed backbone to ligand heavyatom distance.")
    parser.add_argument("--SASA_upper_limit", default=1.00, type=float, help="(default 1.0) Cutoff for ligand relative SASA")
    parser.add_argument("--SASA_lower_limit", default=0.20, type=float, help="(default 0.2) Cutoff for ligand relative SASA")
    parser.add_argument("--loop_limit", default=0.30, type=float, help="(default 0.3) Cutoff for maximum allowed loop content")
    parser.add_argument("--longest_helix", default=30, type=int, help="(default 30) Longest allowed heix length")
    parser.add_argument("--rog", default=30.0, type=float, help="(default 30.0) Largest allowed radius of gyration")
    parser.add_argument("--term_limit", default=15.0, type=float, help="(default 15.0) Cutoff for relative SASA filtering")
    parser.add_argument("--bondlen_dev", default=0.1, type=float, help="(default 0.1) Maximum allowed sidechain bondlength deviation from normal in case of RFD2.")
    parser.add_argument("--exclude_clash_atoms", type=str, nargs="+", help="Ligand atom names that will be excluded from ligand clashchecking")
    parser.add_argument("--ligand_exposed_atoms", type=str, nargs="+", help="Ligand atoms with --ligand_exposed_atoms should have SASA above this cutoff")
    parser.add_argument("--exposed_atom_SASA", type=float, help="Relative SASA cutoff for ligand atoms defined with --exposed_atom_SASA")
    parser.add_argument("--ref_catres", type=str, nargs="+", help="(optional) Catalytic residue positions in reference structure. Ranges can be represented with a dash.")
    parser.add_argument("--loop_catres", action="store_false", default=True, help="(default = True) If enabled, structures where any catalytic residue has 2 loopy residues on both side will be filtered out.")

    parser.add_argument("--outdir", type=str, default="./", help="Where are outputs moved?")
    parser.add_argument("--partial", action="store_true", default=False, help="Are you running this on partial RFD2 output?")
    parser.add_argument("--nproc", type=int, help="# of CPU cores used")

    args = parser.parse_args()

    assert any([x is not None for x in [args.pdb, args.pdbpath]]), "Need to provide either --pdb or --pdbpath"

    if args.pdb is not None:
        pdbfiles = args.pdb
    elif args.pdbpath is not None:
        pdbfiles = []
        for pth in args.pdbpath:
            pdbfiles += glob.glob(pth+"/*.pdb")
    params = args.params
    NPROC = args.nproc
    # SASA_limit = args.limit

    if args.traj is not None:
        if "/" in args.traj:
            traj_N_save = int(args.traj.split("/")[0])
            traj_N_steps = int(args.traj.split("/")[1])
            assert traj_N_save <= traj_N_steps
        else:
            traj_N_save = int(args.traj)
            traj_N_steps = traj_N_save


    ref_catres = []
    if args.ref_catres is not None:
        for r in args.ref_catres:
            if "-" in r:
                _start_pos, _end_pos = r.split("-")[0], r.split("-")[1]
                _ch = r[0]
                for n in range(int(_start_pos[1:]), int(_end_pos)+1):
                    ref_catres.append(f"{_ch}{n}")
            else:
                ref_catres.append(r)

    if args.ligand_exposed_atoms is not None and args.exposed_atom_SASA is None:
        sys.exit("Defined --ligand_exposed_atoms but not --exposed_atom_SASA")
    if args.exposed_atom_SASA is not None and args.ligand_exposed_atoms is None:
        sys.exit("Defined --exposed_atom_SASA but not --ligand_exposed_atoms")

    # if args.fix is True:
    #     assert args.align_atoms is not None, "Need to provide align atom numbers if you want to fix ligand"

    filtered_dir = os.path.join(args.outdir, "filtered_structures")
    try:
        if not os.path.exists(filtered_dir):
            os.mkdir(filtered_dir)
    except PermissionError:
        pass


    ### Getting PyRosetta started
    extra_res_fa = ""
    if len(params) > 0:
        extra_res_fa = "-extra_res_fa"
        for p in params:
            extra_res_fa += " " + p


    print(extra_res_fa)


    assert DAB is not None, "Please compile DAlphaBall.gcc and manually provide a path to it in this script under the variable `DAB`\n"\
                            "For more info on DAlphaBall, visit: https://www.rosettacommons.org/docs/latest/scripting_documentation/RosettaScripts/Filters/HolesFilter"

    pyr.init(f"{extra_res_fa} -mute all -dalphaball {os.path.abspath(os.path.join('./', DAB))} -run:preserve_header")

    ### Starting processing
    start = time.time()

    ligand_found = False
    for p in params:
        par = open(p, 'r').readlines()
        for l in par:
            if "TYPE LIGAND" in l:
                ligand_pdb = p.replace(".params", ".pdb")
                ligand = pyr.pose_from_file(ligand_pdb)
                ligand_SASA = getSASA(ligand).tot_surf
                ligand_found = True
                break
        if ligand_found:
            break

    if not ligand_found:
        sys.exit("Was not able to find a ligand from the params files specified.")

    q = queue.Queue()

    datacolumns = ["chainbreak", "rCA_nonadj", "lig_dist", "bondlen_dev", "loop_frac", "longest_helix", "rog", "loop_at_motif",
                   "term_mindist", "SASA", "SASA_rel"]
    if args.ligand_exposed_atoms is not None:
        datacolumns.append("SASA_exposed_atoms")

    datacolumns.append("description")
    df = pd.DataFrame(columns=datacolumns)


    the_queue = multiprocessing.Queue()  # Queue stores the iterables

    manager = multiprocessing.Manager() 
    ref_poses = manager.dict()  # Need a special dictionary to store outputs from multiple processes
    scores = manager.dict()
    dssps = manager.dict()

    CA_CA_dists = manager.dict()

    print(len(pdbfiles), "designs to analyze.")
    
    count = 0
    for i, pdbfile in enumerate(pdbfiles):
        scores[count] = manager.dict()
        the_queue.put((count, pdbfile))
        count += 1

    # reserving additional entries in the scores dictionary for the trajectory models
    if args.traj is not None:
        for i, pdbfile in enumerate(pdbfiles):
            traj_file = os.path.dirname(os.path.realpath(pdbfile)) + "/traj/" + os.path.basename(pdbfile).replace(".pdb", "_pX0_traj.pdb")
            if os.path.exists(traj_file):
                for n in range(1, traj_N_save):
                    scores[i+len(pdbfiles)*n] = manager.dict()
                    # scores[len(scores)+1] = manager.dict()
            else:
                print(f"No trajectory file found for {pdbfile}, {traj_file}")


    def process(q, ref_poses):
        while True:
            p = q.get(block=True)
            if p is None:
                return
            i = p[0]
            pdbfile = p[1]
            pdbfile_orig = pdbfile
            scores[i]["description"] = pdbfile

            if args.trb is None:
                trbfile = pdbfile.replace(".pdb", ".trb")
            else:
                __trbfs = [x for x in args.trb if os.path.basename(pdbfile).replace(".pdb", "") in x]
                assert len(__trbfs) == 1, f"Bad number of trbs for {pdbfile}: {__trbfs}"
                trbfile = __trbfs[0]

            ### Loading trb file and figuring out fixed positions between ref and hal
            try:
                trb = np.load(trbfile, allow_pickle=True)
            except FileNotFoundError:
                print(trbfile, "not found!!!!!!!!!!!")
                continue

            if args.ref is not None:
                __refs = [r for r in args.ref if os.path.basename(r).replace(".pdb", "_") in os.path.basename(pdbfile)]
                assert len(__refs) == 1, f"Bad number of reference PDBS found for {pdbfile}: {__refs}"
                ref_pdb = __refs[0]
            else:
                ref_pdb = trb["config"]["inference"]["input_pdb"]

            if ref_pdb not in ref_poses.keys():
                ref_poses[ref_pdb] = pyr.pose_from_file(ref_pdb)
            ref_pose = ref_poses[ref_pdb].clone()
            numbering_offset = ref_pose.pdb_info().number(1) -1

            if args.partial is True:
                fixed_pos_in_hal0 = trb["con_hal_idx0"]
                fixed_pos_in_hal = [x+1 for x in fixed_pos_in_hal0]
                fixed_pos_in_ref = trb["con_ref_pdb_idx"]
                _ref_catres = [f"{x[0]}{x[1]}" for x in fixed_pos_in_ref]
            else:
                fixed_pos_in_hal0 = trb["con_hal_idx0"]
                fixed_pos_in_hal = [x+1 for x in fixed_pos_in_hal0]
                fixed_pos_in_ref = trb["con_ref_pdb_idx"]
                _ref_catres = ref_catres

            if args.fix is True:
                # This fixes ligand atom name issues and stuff like that
                # Only for very old AA-diffusion outputs
                # pose = read_pose_from_str_and_fix_ligand(pdbfile, ref_pdb, args.align_atoms)
                pose = read_pose_from_str_and_fix_issues(pdbfile, trb)
            else:
                pose = pyr.pose_from_file(pdbfile)

            poses_to_parse = [pose]

            if args.traj is not None:
                if "atomize_indices2atomname" not in trb.keys() or len(trb["atomize_indices2atomname"]) == 0:
                    traj_file = os.path.dirname(os.path.realpath(pdbfile)) + "/traj/" + os.path.basename(pdbfile).replace(".pdb", "_pX0_traj.pdb")
                    if os.path.exists(traj_file):
                        poses_to_parse += load_multimodel_PDB_to_poses(traj_file, trb, count=traj_N_steps, random=traj_N_save)[1:traj_N_save+1]
                else:
                    print("Trajectory is useless in case of atomized motifs.")

            for _j, pose in enumerate(poses_to_parse):
                if _j != 0:
                    # Setting correct output PDB name and scorefile line for each trajectory pose
                    pdbfile = pdbfile_orig.replace(".pdb", f"_traj{_j}.pdb")
                    idx = p[0] + len(pdbfiles)*_j
                    if "description" in scores[idx].keys():
                        print(f"{pdbfile} {idx} = {i} {_j}: Can't figure out where to store scores !!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    i = idx
                    scores[i]["description"] = pdbfile

                if scores[i]["description"] != pdbfile:
                    print(pdbfile, "Filling out wrong line in scores !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    sys.exit(1)

                pose2 = pose.clone()

                ## A legacy function for cases when partial RFD2 generated backbone doesn't contain a ligand, but reference structure does
                ## This was from the era when allatom partial RFD2 did not exist
                if args.partial is True and any([res.is_ligand() for res in ref_pose.residues]) and not any([res.is_ligand() for res in pose2.residues]):
                    # Add ligand to pose if it exists in the reference structure
                    # But not in the diffused structure
                    align_map = pyrosetta.rosetta.std.map_core_id_AtomID_core_id_AtomID()
                    aln_atoms = ['N', 'CA', 'C', 'O']
                    for template_i, target_i in zip(trb["con_ref_idx0"], trb["con_hal_idx0"]):
                        res_template_i = ref_pose.residue(template_i+1)
                        res_target_i = pose2.residue(target_i+1)
                        for n in aln_atoms:
                            template_atom_idx = res_template_i.atom_index(n)
                            atom_id_template = pyrosetta.rosetta.core.id.AtomID(template_atom_idx, template_i+1)
                            target_atom_idx = res_target_i.atom_index(n)
                            atom_id_target = pyrosetta.rosetta.core.id.AtomID(target_atom_idx, target_i+1)
                            align_map[atom_id_target] = atom_id_template

                    rmsd = pyrosetta.rosetta.core.scoring.superimpose_pose(pose2, ref_pose, align_map)
                    print(f"{pdbfile}: alignment RMSD = {rmsd:.3f}")

                    ### Adding ligand to pose ###
                    ligands = [res for res in ref_pose.residues if res.is_ligand()]
                    for lig in ligands:
                        pyrosetta.rosetta.core.pose.append_subpose_to_pose(pose2, ref_pose, lig.seqpos(), lig.seqpos(), True)


                """
                First some scaffold quality analysis
                """
                ### Checking for chainbreaks
                dists = []
                for n in range(1, pose.size()):
                    if pose.residue(n).is_ligand():
                        continue
                    if pose.residue(n+1).is_ligand():
                        continue
                    if pose.chain(n) != pose.chain(n+1):
                        continue
                    dists.append((pose.residue(n).xyz("CA") - pose.residue(n+1).xyz("CA")).norm())
    
                scores[i]["chainbreak"] = max(dists)
                if args.analyze is False and max(dists) > 4.5:
                    print(f"{pdbfile}: chainbreak found! {max(dists):.2f}")
                    continue

                ### Checking if there are non-adjacent CA-CA contacts that are too short
                ### It seems there are veeeery few cases in nature where non-adj CA atoms are closer than 3.6A from each other
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
    
                scores[i]["rCA_nonadj"] = min(nonadjacentCAs)
                if args.analyze is False and min(nonadjacentCAs) < 3.0:
                    print(f"{pdbfile}: some residues are too close to each other: {min(nonadjacentCAs):.2f}")
                    continue


                ### Checking if there are clashes with any ligands
                ligands = []
                if any([res.is_ligand() for res in pose2.residues]):
                    ligands = [res for res in pose2.residues if res.is_ligand()]
                    lig_dists = []
                    for lig in ligands:
                        ligand_HAs = [n for n in range(1, lig.natoms()+1) if not lig.atom_is_hydrogen(n)]
                        for res in pose2.residues:
                            # excluding motif residues
                            if res.seqpos() in fixed_pos_in_hal:
                                continue
                            if (res.nbr_atom_xyz() - lig.nbr_atom_xyz()).norm() > 15.0:
                                continue
                            if res.is_ligand():
                                continue
                            for lha in ligand_HAs:
                                if args.exclude_clash_atoms is not None:
                                    if lig.atom_name(lha).strip() in args.exclude_clash_atoms:
                                        continue
                                for n in range(1, 5):
                                    lig_dists.append((res.xyz(n) - lig.xyz(lha)).norm())

                    scores[i]["lig_dist"] = min(lig_dists)
                    if args.analyze is False and min(lig_dists) < args.lig_dist:
                        print(f"{pdbfile}: ligand is too close to the backbone {min(lig_dists):.2f}")
                        continue

                ## If atomized residues were used (i.e. tip-atom RFD2), then check if the sidechain is physically real or has too long bonds
                ## Currently doing it for all atomized residues, and not just catalytic ones
                if "atomize_indices2atomname" in trb.keys() and len(trb["atomize_indices2atomname"]) > 0:
                    motif_res_bond_deviations = []
                    for resno in trb["atomize_indices2atomname"].keys():
                        motif_res_bond_deviations.append(sidechain_connectivity(pose2.residue(resno+1)))

                    scores[i]["bondlen_dev"] = max(motif_res_bond_deviations)
                    if args.analyze is False and scores[i]["bondlen_dev"] > args.bondlen_dev:
                        print(f"{pdbfile}: some motif residue geometry is too distorted: {scores[i]['bondlen_dev']:.2f}")
                        continue




                """
                Then let's do some more subjective scaffold quality analysis:
                1) loop fraction
                2) longest helix
                3) radius of gyration
                4) whether motif residues are on loops
                5) how far are the termini from the ligands
                """
                ### Finding how much loop content the structure has
                dssp = pyrosetta.rosetta.core.scoring.dssp.Dssp(pose2)
                secstruct = dssp.get_dssp_secstruct()
                loop_frac = secstruct.count("L") / pose2.size()
                scores[i]["loop_frac"] = loop_frac

                dssps[pdbfile] = secstruct

                if args.analyze is False and loop_frac > args.loop_limit:
                    if loop_frac > 0.9 and _j > 0:
                        print(f"{pdbfile}: something wrong with trajectory loopyness? loop_frac = {loop_frac:.3f}")
                    else:
                        print(f"{pdbfile}: protein too loopy: {loop_frac:.3f}")
                        continue


                ###  Analyzing how long is the longest helix
                if "H" in secstruct:
                    longest_helix = max([len(x.replace("E", "")) for x in secstruct.split("L") if "H" in x])
                else:
                    longest_helix = 0
                scores[i]["longest_helix"] = longest_helix

                if args.analyze is False and longest_helix > args.longest_helix:
                    print(f"{pdbfile}: longest helix too long: {longest_helix}")
                    continue


                ### Calculating radius of gyration
                scores[i]["rog"] = get_ROG(pose)
                if args.analyze is False and scores[i]["rog"] > args.rog:
                    print(f"{pdbfile}: radius of gyration too high: {scores[i]['rog']:.1f}")
                    continue


                ### Finding out if catalytic residues are between loops
                if args.loop_catres is True:
                    loops_next_to_catres = False
                    for r, _or in zip(fixed_pos_in_hal0, fixed_pos_in_ref):
                        # Only calculating it for true catalytic residues that the user provides
                        # Not calculated for partial RFD2 outputs
                        if f"{_or[0]}{_or[1]}" not in ref_catres:
                            continue
                        if secstruct[r-2:r] == "LL" and secstruct[r+1:r+3] == "LL":
                            loops_next_to_catres = True
                            break
                    scores[i]["loop_at_motif"] = int(loops_next_to_catres)
        
                    if args.analyze is False and loops_next_to_catres is True:
                        print(f"{pdbfile}: catalytic residue between loops")
                        continue


                ### Checking how far C and N termini are from the ligands
                if any([res.is_ligand() for res in pose2.residues]):

                    ligands = [res for res in pose2.residues if res.is_ligand()]
                    term_mindists = []
                    for lig in ligands:
                        d_Nt_lig = (pose2.residue(1).xyz("CA") - lig.nbr_atom_xyz()).norm()
                        d_Ct_lig = (pose2.residue(pose2.size()-len(ligands)).xyz("CA") - lig.nbr_atom_xyz()).norm()

                        term_mindist = min([d_Nt_lig, d_Ct_lig])
                        term_mindists.append(term_mindist)
                    scores[i]["term_mindist"] = min(term_mindists)
    
                    if args.analyze is False and scores[i]["term_mindist"] < args.term_limit:
                        print(f"{pdbfile}: terminus too close to ligand: {scores[i]['term_mindist']:.2f}")
                        continue


                #############################################################
                ### DOING ADJUSTMENTS ON STRUCTURES THAT PASS ALL FILTERS ###
                ############################################################# 

                ref_catres_nos = []
                hal_catres_nos = []
                for r in _ref_catres:
                    _ch = r[0]
                    _ref_resno = int(r[1:])
                    assert (_ch, _ref_resno) in fixed_pos_in_ref, f"Can't find residue {r} in trb con_ref_pdb_idx: {fixed_pos_in_ref}"
                    # re-calculating numbering offset because gaps in reference structure might throw it off otherwise
                    ref_resno_in_pose = None
                    for res in ref_pose.residues:
                        if ref_pose.pdb_info().chain(res.seqpos())+str(ref_pose.pdb_info().number(res.seqpos())) == r:
                            ref_resno_in_pose = res.seqpos()
                            break
                    if ref_resno_in_pose is None:
                        print(f"Could not find what is the ref_pose residue number of reference residue {r}")
                        sys.exit(1)
                    ref_catres_nos.append(ref_resno_in_pose)
                    hal_catres_nos.append(fixed_pos_in_hal[fixed_pos_in_ref.index((_ch, _ref_resno))])

                for j, ref_catres_no in enumerate(ref_catres_nos):
                    catres_seqpos = hal_catres_nos[j]
                    catres_AA = ref_pose.residue(ref_catres_no).name().split(":")[0]
                    catres_AA3 = ref_pose.residue(ref_catres_no).name3()

                    # Fixing catalytic residue identity to be the same as in the reference
                    print(f"{pdbfile}: fixing {pose2.residue(catres_seqpos).name()}-{catres_seqpos} with reference {catres_AA}-{ref_catres_no}")
                    mutres = pyrosetta.rosetta.protocols.simple_moves.MutateResidue()
                    mutres.set_res_name(catres_AA)  # fixes HIS_D as well
                    mutres.set_target(catres_seqpos)
                    mutres.apply(pose2)

                    # Fixing catalytic residue rotamer
                    # Skipping residues that were atomized (tip-atom RFD2)
                    if "atomize_indices2atomname" not in trb.keys() or ("atomize_indices2atomname" in trb.keys() and catres_seqpos-1 not in trb["atomize_indices2atomname"].keys()):
                        for n in range(ref_pose.residue(ref_catres_no).nchi()):
                            pose2.residue(catres_seqpos).set_chi(n+1, ref_pose.residue(ref_catres_no).chi(n+1))


                if args.rethread is True:
                    pose2 = thread_seq_to_pose(pose2, pose2.sequence(), skip_resnos=hal_catres_nos)


                if len(ligands) != 0:
                    ### Checking ligand SASA
                    target_ligand = None
                    for lig in pose2.residues:
                        if lig.is_protein():
                            continue
                        if lig.name3() != ligand.residue(1).name3():
                            continue
                        target_ligand = lig.seqpos()
                    if target_ligand is None:
                        print(f"{pdbfile} COULD NOT FIND MATCHING LIGAND IN PDB: {ligand.residue(1).name3()}")

                    scores[i]["SASA"] = getSASA(pose2, resno=target_ligand)
                    scores[i]["SASA_rel"] = scores[i]["SASA"] / ligand_SASA
    
                    if args.analyze is False and scores[i]["SASA_rel"] > args.SASA_upper_limit:
                        print(f"{pdbfile}: ligand too exposed, L_SASA = {scores[i]['SASA_rel']:.3f}")
                        continue
    
                    if args.analyze is False and scores[i]["SASA_rel"] < max(0.01, args.SASA_lower_limit):
                        print(f"{pdbfile}: ligand too buried, L_SASA = {scores[i]['SASA_rel']:.3f}")
                        continue
    
                    if args.ligand_exposed_atoms is not None and args.exposed_atom_SASA is not None:
                        indexes = [pose2.residue(target_ligand).atom_index(x) for x in args.ligand_exposed_atoms]
                        surf_vol_nosc = getSASA(pose2, ignore_sc=True)
                        scores[i]["SASA_exposed_atoms"] = sum([surf_vol_nosc.surf(pose2.size(), i) for i in indexes])
    
                        if args.analyze is False and scores[i]["SASA_exposed_atoms"] < args.exposed_atom_SASA:
                            print(f"{pdbfile}: ligand atoms {args.ligand_exposed_atoms} too buried: {scores[i]['SASA_exposed_atoms']:.3f}")
                            continue


                CA_CA_dists[pdbfile] = []
                for r1, r2 in itertools.combinations(pose2.residues, 2):
                    if r1.is_ligand() or r2.is_ligand():
                        continue
                    if abs(r1.seqpos() - r2.seqpos()) == 1:
                        continue
                    _d = (r1.xyz("CA") - r2.xyz("CA")).norm()
                    if _d < 3.9:
                        CA_CA_dists[pdbfile].append(_d)

                ### Trying to add matcher catalytic residue info to fixed PDB's, if available
                if args.analyze is False:
                    print(f"{pdbfile}: GOOD design")
                    matched_residues = get_matcher_residues(ref_pdb)
                    if len(matched_residues) != 0:
                        matched_residues_in_design = {}
                        for r in matched_residues:
                            for i, res in enumerate(trb["con_ref_pdb_idx"]):
                                if res == (matched_residues[r]["chain"], np.int64(r)):
                                    resno_in_design = trb["con_hal_pdb_idx"][i][1]
                                    matched_residues_in_design[resno_in_design] = copy.deepcopy(matched_residues[r])
                                    matched_residues_in_design[resno_in_design]["chain"] = trb["con_hal_pdb_idx"][i][0]
                                    # Adjusting target residue number if it's not ligand. In case of an upstream match
                                    tgt_resno_orig = matched_residues_in_design[resno_in_design]["target_resno"]
                                    if tgt_resno_orig != 0 and pose2.residue(tgt_resno_orig).is_protein()\
                                        and (matched_residues_in_design[resno_in_design]["target_chain"], np.int64(tgt_resno_orig)) not in trb["con_hal_pdb_idx"]:
                                            (_ch, _rn) = trb["con_hal_pdb_idx"][trb["con_ref_pdb_idx"].index((matched_residues[r]["target_chain"],
                                                                                                              np.int64(matched_residues[r]["target_resno"])))]
                                            matched_residues_in_design[resno_in_design]["target_chain"] = _ch
                                            matched_residues_in_design[resno_in_design]["target_resno"] = _rn
                                    break

                        # Adjusting matched residue info in case some of the reference REMARK 666 residues were not included in RFD2 contigs
                        missing_matched_resnos = []
                        for k in matched_residues:
                            if (matched_residues[k]["chain"], k) not in trb["con_ref_pdb_idx"]:
                                missing_matched_resnos.append(k)
                        for k in missing_matched_resnos:
                                matched_residues.__delitem__(k)

                        pose2 = add_matcher_line_to_pose(pose2, ref_pose, matched_residues_in_design, matched_residues)

                    # Adding motif residues also a REMARK PDBinfo-LABEL:
                    if any(["literal" in x for x in trb.keys()]):  # guideposted RFD2 has broken inpaint_seq TRB
                        motif_residues_fixed = [x for x in trb["con_hal_idx0"]]
                    else:
                        motif_residues_fixed = [x for x in trb["con_hal_idx0"] if trb["inpaint_seq"][x] == True]
                    for rn in motif_residues_fixed:
                        pose2.pdb_info().add_reslabel(rn+1, "motif")


                    pose2.dump_pdb(f"{filtered_dir}/{os.path.basename(pdbfile)}")
                    if _j == 0:
                        copy2(trbfile, f"{filtered_dir}/{os.path.basename(trbfile)}")
                    else:
                        copy2(trbfile, f"{filtered_dir}/{os.path.basename(pdbfile).replace('.pdb', '.trb')}")



    if args.nproc is not None:
        N_PROCESSES = args.nproc
    elif "SLURM_CPUS_ON_NODE" in os.environ:
        N_PROCESSES = int(os.environ["SLURM_CPUS_ON_NODE"])
    elif "OMP_NUM_THREADS" in os.environ:
        N_PROCESSES = int(os.environ["OMP_NUM_THREADS"])
    elif args.nproc is None:
        N_PROCESSES = os.cpu_count()

    print(f"Using {N_PROCESSES} processes")
    pool = multiprocessing.Pool(processes=N_PROCESSES,
                                initializer=process,
                                initargs=(the_queue, ref_poses, ))

    # None to end each process
    for _i in range(N_PROCESSES):
        the_queue.put(None)

    # Closing the queue and the pool
    the_queue.close()
    the_queue.join_thread()
    pool.close()
    pool.join()

    end = time.time()

    print("Sorting the matches took {:.3f} seconds.".format(end - start))

    for i in scores.keys():
        for k in datacolumns:
            if k not in scores[i].keys():
                df.at[i, k] = np.nan
            else:
                df.at[i, k] = scores[i][k]

    for k in datacolumns[:-1]:
        df = df.sort_values(k)

    print(df)
    dump_scorefile(df, os.path.join(args.outdir, "RFD2_analysis.sc"))


# if __name__ == "__main__":
#     main()
