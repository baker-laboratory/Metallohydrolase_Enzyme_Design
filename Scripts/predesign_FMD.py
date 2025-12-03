# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 17:45:33 2023

@author: ikalvet
"""
import pyrosetta as pyr
import pyrosetta.rosetta
import pyrosetta.distributed.io
import os, sys
import glob
import random
import json
import argparse
import numpy as np


script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "enzyme_design", "FastMPNNDesign"))
import FastMPNNdesign
from Selectors import SelectHBondsToResidue
sys.path.append(os.path.join(script_dir, "enzyme_design"))
import design_utils
import scoring_utils

rfd2_path = os.path.normpath(os.path.join(script_dir, "..", "Software", "RFdiffusion2"))
sys.path.append(rfd2_path)
#sys.path.append("./software/rf_diffusion_repo")
import rf_diffusion


class CSTs():
    def __init__(self, cstfile, scorefxn):
        self.__scorefxn = scorefxn
        self.__addcst_mover = pyrosetta.rosetta.protocols.enzdes.AddOrRemoveMatchCsts()
        self.__chem_manager = pyrosetta.rosetta.core.chemical.ChemicalManager.get_instance()
        self.__residue_type_set = self.__chem_manager.residue_type_set("fa_standard")
        self.__cst_io = pyrosetta.rosetta.protocols.toolbox.match_enzdes_util.EnzConstraintIO(self.__residue_type_set)
        self.__cst_io.read_enzyme_cstfile(cstfile)
        pass
    
    def add_cst(self, pose):
        self.__cst_io.add_constraints_to_pose(pose, self.__scorefxn, True)
    
    def cst_io(self):
        return self.__cst_io

    def remove_cst(self, pose):
        self.__cst_io.remove_constraints_from_pose(pose, True, True)
        
    def cst_score(self, pose):
        """
        To be implemented
        """
        return None


parser = argparse.ArgumentParser()

parser.add_argument("--pdb", type=str, required=True, help="Input PDB file, containing a ligand and matcher CST lines in header.")
parser.add_argument("--suffix", type=str, help="Suffix to be added to the end the output filename")
parser.add_argument("--cstfile", type=str, help="Matcher/enzdes CSTfile")
parser.add_argument("--params", type=str, nargs="+", help="Matcher/enzdes CSTfile")
parser.add_argument("--output_dir", type=str, help="Output directory for structure")
parser.add_argument("--scoring", type=str, help="Scoring file path")

args = parser.parse_args()

## Loading the user-provided scoring module
sys.path.append(os.path.dirname(str(args.scoring)))
scoring = __import__(os.path.basename(str(args.scoring).replace(".py", "")))
assert "score_design" in scoring.__dir__()
assert "filter_scores" in scoring.__dir__()
assert "filters" in scoring.__dir__()

pdbname = os.path.basename(args.pdb).replace(".pdb", "")


suffix = ""
if args.suffix is not None:
    suffix = "_" + args.suffix

output_dir = ""
if args.output_dir is not None:
    output_dir = args.output_dir + "/"

## Saving scorefiles separately
os.makedirs(f"{output_dir}scores/", exist_ok=True)
scorefilename = f"{output_dir}scores/{pdbname}{suffix}.sc"


"""
Rosetta stuff
"""
extra_res_fa = "-extra_res_fa " + " ".join(args.params)

NPROC = os.cpu_count()
if "OMP_NUM_THREADS" in os.environ:
    NPROC = os.environ["OMP_NUM_THREADS"]
if "SLURM_CPUS_ON_NODE" in os.environ:
    NPROC = os.environ["SLURM_CPUS_ON_NODE"]


DAB = os.path.join(script_dir, "enzyme_design", "DAlphaBall.gcc")
pyr.init(f"{extra_res_fa} -dalphaball {DAB} -beta_nov16 -run:preserve_header "
         f"-multithreading true -multithreading:total_threads {NPROC} -multithreading:interaction_graph_threads {NPROC}")


sfx = pyr.get_fa_scorefxn()

if args.cstfile is not None:
    sfx.set_weight(pyrosetta.rosetta.core.scoring.score_type_from_name("atom_pair_constraint"), 1.0)
    sfx.set_weight(pyrosetta.rosetta.core.scoring.score_type_from_name("angle_constraint"), 1.0)
    sfx.set_weight(pyrosetta.rosetta.core.scoring.score_type_from_name("dihedral_constraint"), 1.0)     
    cst_mover = CSTs(args.cstfile, sfx)



pose = pyr.pose_from_file(args.pdb)

ligand_seqpos = pose.size()
assert pose.residue(ligand_seqpos).is_ligand()


"""
Setting up design/repack layers
"""
catres = design_utils.get_matcher_residues(args.pdb)

keep_pos = []
keep_pos += list(catres.keys())

trb = np.load(args.pdb.replace(".pdb", ".trb"), allow_pickle=True)


if any(["literal" in x for x in trb.keys()]):
    keep_pos += [int(x+1) for x in trb["con_hal_idx0"]]
else:
    keep_pos += [int(x+1) for x in trb["con_hal_idx0"] if trb["inpaint_seq"][x] == True]

## Identifying any motif residues based on pdbinfo reslabel
motif_label_sel = pyrosetta.rosetta.core.select.residue_selector.ResiduePDBInfoHasLabelSelector(label_str="motif")
keep_pos += list(pyrosetta.rosetta.core.select.get_residue_set_from_subset(motif_label_sel.apply(pose)))
keep_pos = list(set(keep_pos))

design_residues = [res.seqpos() for res in pose.residues if res.seqpos() not in keep_pos and not res.is_ligand()]

print("Design positions: ", "+".join([str(x) for x in design_residues]))

assert len(design_residues) > 0
assert len(keep_pos) > 0

sfx_cart = sfx.clone()
sfx_cart.set_weight(pyrosetta.rosetta.core.scoring.score_type_from_name("cart_bonded"), 0.5)
sfx_cart.set_weight(pyrosetta.rosetta.core.scoring.score_type_from_name("pro_close"), 0.0)

# _pose2.dump_pdb(f"{pdbname}_prerelax3_{suffix}.pdb")


## Defining the design protocol for FastMPNNDesign
## Putting 1 sequence onto the backbone with MPNN
## Repacking the sidechains and doing cartesian minimization twice

# 7
protocol = """
scale:coordinate_constraint 1.0
scale:fa_rep 0.5
mpnn 0.1 1
repack
min 0.01
"""
cartesian=True

## Excluding the ligand from minimization - it can mess it up.
mm = pyrosetta.rosetta.core.kinematics.MoveMap()
mm.set_chi(True)
mm.set_bb(True)
mm.set_jump(True)
for resno in [ligand_seqpos]:
    mm.set_chi(resno, False)
    mm.set_bb(resno, False)


fastrelax = design_utils.setup_fastrelax(sfx, crude=True)

if len(glob.glob(f"{output_dir}{pdbname}{suffix}_*.pdb")) > 0:
    print(f"{output_dir}{pdbname}{suffix}_* outputs already exist")
    sys.exit(0)

pose2 = pose.clone()

if args.cstfile is not None:
    print(pose2.constraint_set().has_constraints())
    # cst_mover.remove_cst(pose2)

    cst_mover.add_cst(pose2)

fmd = FastMPNNdesign.FastMPNNdesign(model_type="ligand_mpnn", params=args.params, scorefxn=sfx_cart, script_file=protocol, cartesian=cartesian,
                                    design_positions=design_residues,
                                    repack_positions=keep_pos,
                                    cst_io=cst_mover.cst_io(), omit_AA="CM")
fmd.set_mpnn_bias({"A": -0.5})  # small bias against ALA
fmd.set_minimizer_movemap(mm)
poses = fmd.apply(pose2)

for i, p in enumerate(poses):
    cst_mover.add_cst(p)
    fastrelax.apply(p)  # crude fastrelax to fix the constraints
    scores_df = scoring.score_design(p, pyr.get_fa_scorefxn(), list(catres.keys()))
    print(scores_df.iloc[0])
    scores_df.at[0, "description"] = f"{output_dir}{pdbname}{suffix}_{i}"
    scoring_utils.dump_scorefile(scores_df, scorefilename)
    p.dump_pdb(f"{output_dir}{pdbname}{suffix}_{i}.pdb")
    print("PDB Dumped Here:", f"{output_dir}{pdbname}{suffix}_{i}.pdb")

