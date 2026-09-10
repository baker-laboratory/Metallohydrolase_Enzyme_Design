#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Indrek Kalvet
ikalvet@uw.edu
"""

import os
import sys
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
import json


aa3to1 = {
    "ALA":'A', "ARG":'R', "ASN":'N', "ASP":'D', "CYS":'C',
    "GLN":'Q', "GLU":'E', "GLY":'G', "HIS":'H', "ILE":'I',
    "LEU":'L', "LYS":'K', "MET":'M', "PHE":'F', "PRO":'P',
    "SER":'S', "THR":'T', "TRP":'W', "TYR":'Y', "VAL":'V' }

aa1to3 = {val: k for k, val in aa3to1.items()}



def main(args):

    pdbfiles = args.pdb


    extra_res_fa = ""
    if args.params is not None:
        extra_res_fa = "-extra_res_fa"
        for p in args.params:
            extra_res_fa += " " + p

    pyr.init(f"{extra_res_fa} -mute all -run:preserve_header")


    inpaint_dict = {"pdb": None,
        "task": "hal",
        "dump_all": True,
        "inf_method": "multi_shot",
        "num_designs": args.nstruct,
        "tmpl_conf": "1.0",
        "exclude_aa": "C",
        "inpaint_seq": None,
        "contigs": None,
        "out": None}


    # parsing/expanding ref_catres, if provided
    if args.ref_catres is not None:
        ref_catres = []
        for r in args.ref_catres:
            if "-" in r:
                _start_pos, _end_pos = r.split("-")[0], r.split("-")[1]
                _ch = r[0]
                for n in range(int(_start_pos[1:]), int(_end_pos)+1):
                    ref_catres.append(f"{_ch}{n}")
            else:
                ref_catres.append(r)
        args.ref_catres = ref_catres


    # inpaint_dict_list = []

    start = time.time()

    q = queue.Queue()

    the_queue = multiprocessing.Queue()  # Queue stores the iterables
    
    manager = multiprocessing.Manager()
    inpaint_dict_list = manager.list()

    for i, pdbfile in enumerate(pdbfiles):
        the_queue.put((i, pdbfile))

    def process(q):
        while True:
            p = q.get(block=True)
            if p is None:
                return
            i = p[0]
            pdbfile = p[1]
            
            if args.trb is None:
                trbfile = pdbfile.replace(".pdb", ".trb")
                if not os.path.exists(trbfile) and "traj" in pdbfile.split("_")[-1]:
                    trbfile = "_".join(pdbfile.split("_")[:-1]) + ".trb"

            else:
                __trbfs = [x for x in args.trb if os.path.basename(pdbfile).replace(".pdb", "") in x]
                assert len(__trbfs) == 1, f"Bad number of trbs for {pdbfile}: {__trbfs}"
                trbfile = __trbfs[0]


            pose = pyr.pose_from_file(pdbfile)
    
            dssp = pyrosetta.rosetta.core.scoring.dssp.Dssp(pose)
            if pose.residue(pose.size()).is_ligand():
                secstruct = dssp.get_dssp_secstruct()[:-1]  # excluding the ligand
            else:
                secstruct = dssp.get_dssp_secstruct()

            N_ligands = len([r for r in pose.residues if r.is_ligand()])


            ### Loading trb file and figuring out fixed positions between ref and hal
            trb = np.load(trbfile, allow_pickle=True)
            fixed_pos_in_hal0 = trb["con_hal_idx0"]
            fixed_pos_in_hal = [x+1 for x in fixed_pos_in_hal0]
            fixed_pos_in_ref = trb["con_ref_pdb_idx"]

            if args.ref_designable is not None:
                fixed_pos_in_hal = []
                fixed_pos_in_ref = []
                for y,x in zip(trb["con_hal_idx0"], trb["con_ref_pdb_idx"]):
                    if f"{x[0]}{x[1]}" not in args.ref_designable:
                        fixed_pos_in_hal.append(y+1)
                        fixed_pos_in_ref.append(x)

            TIP_ATOM = False
            if "atomize_indices2atomname" in trb.keys() or any(["literal" in x for x in trb.keys()]):
                TIP_ATOM = True
                if len(trb["atomize_indices2atomname"]) == 0:
                    TIP_ATOM = False

            # RFD2 fixed positions will not be touched
            if TIP_ATOM:  # tip-atom RFD2 TRB has a bug
                forbidden_positions = [x for x in fixed_pos_in_hal]  # allowing design on motifs where sequence was designable. Is it safe?
            else:
                forbidden_positions = [x for x in fixed_pos_in_hal if trb["inpaint_seq"][x-1] == True]  # allowing design on motifs where sequence was designable. Is it safe?
            catalytic_positions = []
            if args.ref_catres is not None:
                for _nr, _or in zip(fixed_pos_in_hal, fixed_pos_in_ref):
                    if f"{_or[0]}{_or[1]}" in args.ref_catres:
                        catalytic_positions.append(_nr)

            if args.only_seq is False:
                # Positions 5 res upstream and downstream of fixed or catalytic positions will not be touched by backbone inpainting
                for x in fixed_pos_in_hal:
                    if x in catalytic_positions or args.ref_catres is None:
                        for n in range(x-4, x+4):
                            if n > 0 and n not in forbidden_positions and n < pose.size():
                                forbidden_positions.append(n)
                    else:
                        forbidden_positions.append(x)

                # Finding which positions are close to ligand and also not allowing remodeling of these
                if pose.residue(pose.size()).is_ligand():
                    ligand = pose.residue(pose.size())
                    lig_heavyatoms = [atomno for atomno in range(1, ligand.natoms()+1) if ligand.atom_type(atomno).element() != "H"]
                    for res in pose.residues:
                        if res.is_ligand() or res.is_virtual_residue():
                            continue
                        for lig_ha in lig_heavyatoms:
                            if (ligand.xyz(lig_ha) - res.xyz("CA")).norm() < 5.0:
                                forbidden_positions.append(res.seqpos())
                                break
        
                forbidden_positions = sorted(list(set(forbidden_positions)))
        
        
                #### BELOW IS A VERY PRIMITIVE WAY TO FIGURE OUT WHAT POSITIONS TO INPAINT ###
                # TODO: split it into functions when things work reliably
        
                # Finding loops in the structure
                loop_start = None
                loops = []
                for i, l in enumerate(secstruct):
                    if l == "L" and (i == 0 or secstruct[i-1] != "L"):
                        loop_start = i
                    if l == "L" and (i == len(secstruct)-1 or secstruct[i+1] != "L"):
                        loops.append((loop_start+1, i+1))
                        loop_start = None
        
                # Expand positions around loops to up to 3 residues for helix, 1 residue for strand
                expanded_loops = []
                for j, loop in enumerate(loops):
                    if loop[1] - loop[0] + 1 == 1:  # Dropping loops of length 1
                        continue
                    ns = [n for n in range(loop[0], loop[1]+1)]
                    if any([n in forbidden_positions for n in ns]):  # Dropping loops that contain any forbidden residues
                        continue
                    # Expanding the loops
                    expanded_loop = [n for n in range(loop[0], loop[1]+1)]
                    # C-term
                    num_to_expand = 3
                    if "E" in secstruct[loop[1]:loop[1]+3]:
                        num_to_expand = 1
                    for _ in range(num_to_expand):
                        if expanded_loop[-1] + 1 not in forbidden_positions and expanded_loop[-1] + 1 in range(1, pose.size()):
                            expanded_loop.append(expanded_loop[-1] + 1)
                        else:
                            break
                    # N-term
                    num_to_expand = 3
                    if "E" in secstruct[loop[0]-3:loop[0]]:
                        num_to_expand = 1
                    for _ in range(num_to_expand):
                        if expanded_loop[0] - 1 not in forbidden_positions and expanded_loop[0] - 1 in range(1, pose.size()):
                            if j > 0 and loops[j-1][-1] == expanded_loop[0] - 2:  # Not adding the last bit if it makes two regions run into eachother
                                break
                            expanded_loop = [expanded_loop[0] - 1] + expanded_loop
                        else:
                            break
                    expanded_loops.append(expanded_loop)

                if len(expanded_loops) == 0:
                    print(f"{pdbfile} No valid loops found.")
                    continue

                # Adding terminal chunks to inpaint regions
                if not any([x < 6 for x in forbidden_positions]):
                    if 1 < expanded_loops[0][0] <= 6:  # There's already a loop that's close to N-term
                        expanded_loops[0] = [x for x in range(1, expanded_loops[0][0])] + expanded_loops[0]
                    elif expanded_loops[0][0] > 6:  # No loop that includes any of the first 6 residues
                        expanded_loops = [[x for x in range(1, 7)]] + expanded_loops
                if not any([x > pose.size()-N_ligands-6 for x in forbidden_positions]):
                    if pose.size() > expanded_loops[-1][-1] >= pose.size()-N_ligands-6:  # There's already a loop that's close to C-term
                        expanded_loops[-1] = expanded_loops[-1] + [x for x in range(expanded_loops[-1][-1], pose.size())]
                    elif expanded_loops[-1][-1] < pose.size()-N_ligands-6:  # No loop that includes any of the last 6 residues
                        expanded_loops.append([x for x in range(pose.size()-N_ligands-6, pose.size())])

                # Checking if loop expansion caused some of them to overlap
                # Combining overlapping regions
                _tmp_exp_loops = []
                for j, loop in enumerate(expanded_loops):
                    if j > 0 and any([x in _tmp_exp_loops[-1] for x in loop]):
                        continue
                    _combined_loops = [x for x in loop]
                    jjj = j+1
                    jj = j
                    if jjj <= len(expanded_loops)-1:
                        while any([x in expanded_loops[jjj] for x in expanded_loops[jj]]) or expanded_loops[jj][-1]+1 == expanded_loops[jjj][0]:
                            _combined_loops += expanded_loops[jjj]
                            jjj += 1
                            jj += 1
                            if jjj > len(expanded_loops)-1:
                                break
                    _tmp_exp_loops.append(sorted(list(set(_combined_loops))))
                expanded_loops = [x for x in _tmp_exp_loops]

                n_des = args.nstruct
                contig = ""
                for j, loop in enumerate(expanded_loops):
                    loop_length = len(loop)
                    if args.var is True:
                        if 10 <= loop_length < 15:
                            loop_length = f"{loop_length}-{loop_length+1}"
                            n_des = args.nstruct + 1
                        elif 15 <= loop_length < 20:
                            loop_length = f"{loop_length-1}-{loop_length+2}"
                            n_des = args.nstruct + 2
                        elif 20 <= loop_length <= 25:
                            loop_length = f"{loop_length-2}-{loop_length+3}"
                            n_des = args.nstruct + 4
                        elif loop_length > 25:
                            loop_length = f"{loop_length-int(0.1*loop_length)}-{loop_length+int(0.1*loop_length)}"
                            n_des = args.nstruct + 4

                    if j == 0 and loop[0] == 1:
                        contig += f"{loop_length},A{loop[-1]+1}-"
                    elif j == 0:
                        contig += f"A1-{loop[0]-1},{loop_length},A{loop[-1]+1}-"
                    elif j == len(expanded_loops)-1 and loop[-1] == pose.size()-N_ligands:
                        contig += f"{loop[0]-1},{loop_length}"
                    elif j == len(expanded_loops)-1:
                        contig += f"{loop[0]-1},{loop_length},A{loop[-1]+1}-{pose.size()-N_ligands}"
                    else:
                        contig += f"{loop[0]-1},{loop_length},A{loop[-1]+1}-"


                keep_regions = [x for x in contig.split(",") if "A" in x]
                try:
                    keep_regions_expanded = [[n for n in range(int(rgn.split("-")[0][1:]), int(rgn.split("-")[1])+1)] for rgn in keep_regions]
                except ValueError:
                    print(pdbfile)
                    print(expanded_loops)
                    print(contig)
                    print(keep_regions)
                    sys.exit(1)

                # print(fixed_pos_in_hal)

                keep_regions_expanded_no_catres = []
                
                if args.design_full is True:
                    _positions_to_not_design = catalytic_positions
                else:
                    _positions_to_not_design = fixed_pos_in_hal

                ## Generating 'inpaint_seq' entry
                for rgn in keep_regions_expanded:
                    if any([n in _positions_to_not_design for n in rgn]):  # any fixed/motif residue in a contig region?
                        _tmp_rgn = []
                        for resno in rgn:
                            if resno not in _positions_to_not_design:
                                _tmp_rgn.append(resno)
                                if resno == rgn[-1]:
                                    keep_regions_expanded_no_catres.append(_tmp_rgn)
                            elif resno in _positions_to_not_design and trb["inpaint_seq"][resno-1] == False and not TIP_ATOM:  # was only backbone motif - sequence design allowed
                                # inpaint_seq positions are designable
                                # unless tip-atom RFD2 was used. Then there's a bug...
                                _tmp_rgn.append(resno)
                                if resno == rgn[-1]:
                                    keep_regions_expanded_no_catres.append(_tmp_rgn)
                                # print(resno)
                                # print(resno in fixed_pos_in_hal)
                                # print(trb["inpaint_seq"][resno-1])
                                # print(trb["inpaint_seq"])
                                # print(trb["inpaint_seq"][0])
                                # print([n for n,t in enumerate(trb["inpaint_seq"]) if t == True])
                            elif resno in _positions_to_not_design and TIP_ATOM is True and resno-1 not in trb["atomize_indices2atomname"].keys():
                                _tmp_rgn.append(resno)
                                if resno == rgn[-1]:
                                    keep_regions_expanded_no_catres.append(_tmp_rgn)
                            else:
                                # Storing the region up until the motif residue, and skipping over it
                                # inpaint_seq of contig A1-7 would become A1-3,A5-7 if A4 is motif
                                if len(_tmp_rgn) != 0:
                                    keep_regions_expanded_no_catres.append(_tmp_rgn)
                                    _tmp_rgn = []
                    else:
                        keep_regions_expanded_no_catres.append(rgn)
                    
            elif args.only_seq is True:
                keep_regions_expanded_no_catres = [[]]
                if args.design_full is False:
                    if not TIP_ATOM:  # tip-atom RFD2 TRB has a bug
                        forbidden_positions = [x for x in forbidden_positions if trb["inpaint_seq"][x-1] == True]
                else:
                    forbidden_positions = [x for x in catalytic_positions]
                for n in range(1, len(secstruct)+1):
                    if n not in forbidden_positions:
                        keep_regions_expanded_no_catres[-1].append(n)
                    else:
                        keep_regions_expanded_no_catres.append([])
                keep_regions_expanded_no_catres = [x for x in keep_regions_expanded_no_catres if len(x) != 0]
                contig = f"A1-{len(secstruct)}"
                n_des = 1

                
            inpaint_seq = ",".join([f"A{x[0]}-{x[-1]}" for x in keep_regions_expanded_no_catres])
            print(f"{pdbfile} {secstruct}\n"
                  f"{pdbfile}: contigs = {contig}\n"
                  f"{pdbfile}: inpaint_seq = {inpaint_seq}")

            ### Setting up the commands dictionary for inpainting
            _dict = {k: val for k, val in inpaint_dict.items()}
            _dict["pdb"] = pdbfile
            _dict["out"] = os.path.join(str(args.outdir), os.path.basename(pdbfile).replace(".pdb", "_inp"))
            _dict["contigs"] = [contig]
            _dict["inpaint_seq"] = [inpaint_seq]

            if "EEE" in secstruct and args.var is False and args.only_seq is False:
                _dict["num_designs"] = args.nstruct + 1
            elif args.var is True:
                _dict["num_designs"] = n_des
            else:
                _dict["num_designs"] = n_des


            inpaint_dict_list.append(_dict)


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

    end = time.time()
    if args.group is None:
        with open("cmds.json", "w") as file:
            json.dump(list(inpaint_dict_list), file, separators=(",\n", ":"))
    else:
        for j, i in enumerate(range(0, len(inpaint_dict_list), args.group)):
            _tmp = inpaint_dict_list[i:i+args.group]
            with open(f"cmds_{j}.json", "w") as file:
                json.dump(_tmp, file, separators=(",\n", ":"))

    print("Creating inpainting inputs took {:.3f} seconds.".format(end - start))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--pdb", nargs="+", type=str, required=True, help="Input PDBs from RFD2")
    parser.add_argument("--trb", nargs="+", type=str, help="TRB files from RFD2")
    parser.add_argument("--params", nargs="+", type=str, help="Params files of ligands and noncanonicals")
    parser.add_argument("--group", type=int, help="How many designs will be in each JSON file")
    parser.add_argument("--nstruct", type=int, default=1, help="At minimum, how many structures will be produced per input. Actual number depends on the lengths of loop regions that are going to be rebuilt.")
    parser.add_argument("--var", action="store_true", default=False, help="Inpaint with variable contig lengths?")
    parser.add_argument("--only_seq", action="store_true", default=False, help="Do not generate backbone, just design sequence")
    parser.add_argument("--design_full", action="store_true", default=False, help="Design all non-catalytic (defined with --ref_catres) residues.")
    parser.add_argument("--nproc", type=int, default=os.cpu_count(), help="How many CPU cores are used")
    # parser.add_argument("--pdif", action="store_true", default=False, help="Setting up for partial RFD2 instead of inpainting")
    parser.add_argument("--ref_catres", type=str, nargs="+", help="Format: 'A15 Bxx'. Residue that is defined as catalytic in the RFD2 input. This will be parsed from the trb con_ref_pdb_idx")
    parser.add_argument("--ref_designable", type=str, nargs="+", help="Format: 'A15 Bxx'. Residues that were motifs for RFD2, but can be redesigned during inpainting. This will be parsed from the trb con_ref_pdb_idx")
    parser.add_argument("--outdir", type=str, help="Output path")

    args = parser.parse_args()
    
    main(args)
