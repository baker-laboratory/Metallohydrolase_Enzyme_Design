# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 17:45:33 2023

@author: ikalvet, Donghyo
"""
import pyrosetta as pyr
import pyrosetta.rosetta
import pyrosetta.distributed.io
import os, sys
import glob
import random
import json
import argparse

sys.path.append("../Software/fastmpnndesign/")
import FastMPNNdesign
from Selectors import SelectHBondsToResidue
import ScoringCalculators
sys.path.append("./scripts/enzyme_design/utils")
import design_utils
import scoring_utils


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


def perform_quick_prerelax_and_mutate_clashes_to_ALA(pose, design_positions, target_seqpos: list, cst_io=None, cartesian=False):
    print(f"Performing quick constrained cartesian={cartesian} FastRelax on the input structure to get the ligand placement correct")
    sfx_cart = sfx.clone()
    if cartesian:
        sfx_cart.set_weight(pyrosetta.rosetta.core.scoring.score_type_from_name("cart_bonded"), 0.5)
        sfx_cart.set_weight(pyrosetta.rosetta.core.scoring.score_type_from_name("pro_close"), 0.0)
    fastRelax = design_utils.setup_fastrelax(sfx_cart, crude=True, disable_min_resons=target_seqpos)
    fastRelax.cartesian(cartesian)
    fastRelax.constrain_coords(True)
    clashes = []
    for resno in target_seqpos:
        ligand = pose.residue(resno)
        clashes += design_utils.find_clashes_between_target_and_sidechains(pose, pose.size(),
                                                                          target_atoms=[n for n in range(1, ligand.natoms()+1) if not ligand.atom_is_hydrogen(n)],
                                                                          residues=design_positions)
    clashes = [x for x in clashes if pose.residue(x).name3() not in ["ALA", "GLY", "PRO"] and x not in keep_pos]
    _pose2 = design_utils.mutate_residues(pose, clashes, "ALA")
    if cst_io is not None:
        cst_io.add_constraints_to_pose(_pose2, sfx_cart, True)
    # _pose3 = _pose2.clone()
    fastRelax.apply(_pose2)
    sfx_cart(_pose2)
    if cst_io is not None:
        cst_score = sum([_pose2.scores[s] for s in _pose2.scores if "constraint" in s])
        print(f"CST score after ALA FastRelax: {cst_score:.2f}")
    return _pose2


def get_2nd_layer_fixed_pos(pose, target_resnos, heavyatoms, keep_pos):
    dist_bb = 6.0
    dist_sc = 5.0
    motif_label_sel = pyrosetta.rosetta.core.select.residue_selector.ResiduePDBInfoHasLabelSelector(label_str="keep_hbonds_to_ligand_and_catres")
    pocket_positions = keep_pos+list(pyrosetta.rosetta.core.select.get_residue_set_from_subset(motif_label_sel.apply(pose)))
    pocket_positions = list(set(pocket_positions))
    design_residues = []

    for resno in target_resnos:
        SEL_mutate_residues, SEL_repack_residues, SEL_do_not_repack, residues\
            = design_utils.get_layer_selections(pose, repack_only_pos=pocket_positions,
                                                design_pos=[], ref_resno=resno, heavyatoms=heavyatoms,
                                                cuts=[dist_bb, dist_bb+2.0, dist_bb+4.0, dist_bb+6.0], design_GP=True)

        # Need to somehow pick pocket residues that have SC atoms close to the ligand.
        # Exclude from design: residues[0] and those that have SC atoms very close.
        close_ones = design_utils.get_residues_with_close_sc(pose, heavyatoms, residues[1]+residues[2], exclude_residues=pocket_positions, cutoff=dist_sc, ref_seqpos=resno)
        pocket_positions += residues[0] + close_ones
        design_residues += [x for x in residues[0]+residues[1]+residues[2]+residues[3] if x not in pocket_positions]
    pocket_positions = list(set(pocket_positions))
    design_residues = list(set([x for x in design_residues if x not in pocket_positions]))

    # Also including all Alanines that are not in the pocket.
    # ala_positons = [res.seqpos() for res in pose.residues if res.seqpos() not in pocket_positions+design_residues and res.name3() == "ALA"]
    # print("ALA positions", '+'.join([str(x) for x in ala_positons]))
    # design_residues += ala_positons
    fixed_residues = [f"{pose.pdb_info().chain(r.seqpos())}{pose.pdb_info().number(r.seqpos())}" for r in pose.residues if r.seqpos() not in design_residues and r.is_protein()]
    return fixed_residues

## Defining the design protocol for FastMPNNDesign. It can be provided using --protocol
protocol = """
scale:coordinate_constraint 1.0
scale:fa_rep 0.150
mpnn 0.4 10
repack
scale:fa_rep 0.200
min 0.01
keep_best 5
task_operation keep_hbonds_to_ligand_and_catres
scale:coordinate_constraint 0.5
scale:fa_rep 0.365
mpnn 0.2 2
repack
keep_best 5
scale:fa_rep 0.480
min 0.01
task_operation keep_hbonds_to_ligand_and_catres
scale:coordinate_constraint 0.0
scale:fa_rep 0.659
mpnn 0.1 2
repack
keep_best 5
scale:fa_rep 0.750
min 0.01
task_operation keep_hbonds_to_ligand_and_catres
scale:coordinate_constraint 0.0
scale:fa_rep 1
mpnn 0.1 2
repack
min 0.00001
"""


parser = argparse.ArgumentParser()

parser.add_argument("--pdb", type=str, required=True, help="Input PDB file, containing a ligand and matcher CST lines in header.")
parser.add_argument("--nstruct", type=int, default=1, help="How many design iterations?")
parser.add_argument("--suffix", type=str, help="Suffix to be added to the end the output filename")
parser.add_argument("--outdir", type=str, default="./", help="Suffix to be added to the end the output filename")
parser.add_argument("--params", type=str, nargs="+", help="Params files of ligands for Rosetta")
parser.add_argument("--design_pos", type=int, nargs="+", help="Positions that will be redesigned.")
parser.add_argument("--keep_pos", type=int, nargs="+", help="Positions that will be kept fixed. Repack is allowed.")
parser.add_argument("--detect_pocket", action="store_true", default=False, help="Figure out designable positions around the ligand algorithmically.")
parser.add_argument("--redesign", action="store_true", default=False, help="Only focuses on redesigning positions above the porphyrin.")
parser.add_argument("--bias_atoms", nargs="+", type=str, help="Ligand atom names for which the surrounding residues will receive a bias towards/agains KREDYQWSTH aa's")
parser.add_argument("--bias_AAs", type=str, default="KREDYQWSTH", help="(default = KREDYQWSTH) AA1 letters of amino acids that should be biased near atoms defined with --bias_atoms with a bias defined with --position_bias")
parser.add_argument("--position_bias", type=float, default=-1.0, help="(default = -1.0) Bias that will be applied to polar AAs at positions selected with distance from --bias_atoms.")
parser.add_argument("--protocol", type=str, help="Text file defining the FastMPNNDesign protocol that will be applied.")
parser.add_argument("--scoring", type=str, help="Python script that implements scoring of designs.")
parser.add_argument("--cstfile", type=str, help="Matcher/enzdes CSTfile")
parser.add_argument("--filter", action="store_true", default=False, help="Only dump outputs that meet filtering criteria set in scoring script")
parser.add_argument("--mpnn", action="store_true", default=False, help="Performs additional 2nd layer MPNN on successful outputs")

parser.add_argument("--debug", action="store_true", default=False, help="Dumping stuff along the way")

args = parser.parse_args()

if args.scoring is None:
    args.scoring = os.path.dirname(__file__) + "/esterase_scoring.py"

## Loading the user-provided scoring module
sys.path.append(os.path.dirname(args.scoring))
scoring = __import__(os.path.basename(args.scoring.replace(".py", "")))
assert "score_design" in scoring.__dir__()
assert "filter_scores" in scoring.__dir__()
assert "filters" in scoring.__dir__()

pdbname = os.path.basename(args.pdb).replace(".pdb", "")


if args.mpnn is True:
    os.makedirs(f"{args.outdir}/seqs/", exist_ok=True)

suffix = ""
if args.suffix is not None:
    suffix = "_" + args.suffix

## Saving scorefiles separately
os.makedirs(f"{args.outdir}/scores/", exist_ok=True)
scorefilename = f"{args.outdir}/scores/{pdbname}{suffix}.sc"


"""
Rosetta stuff
"""

extra_res_fa = ""
if args.params is not None:
    extra_res_fa = "-extra_res_fa"
    for p in args.params:
        extra_res_fa += f" {p}"

NPROC = os.cpu_count()
if "OMP_NUM_THREADS" in os.environ:
    NPROC = os.environ["OMP_NUM_THREADS"]
if "SLURM_CPUS_ON_NODE" in os.environ:
    NPROC = os.environ["SLURM_CPUS_ON_NODE"]


DAB = "./scripts/enzyme_design/DAlphaBall.gcc"
pyr.init(f"{extra_res_fa} -dalphaball {DAB} -beta_nov16 -run:preserve_header "
         f"-multithreading true -multithreading:total_threads {NPROC} -multithreading:interaction_graph_threads {NPROC}")


sfx = pyr.get_fa_scorefxn()

if args.cstfile is not None:
    sfx.set_weight(pyrosetta.rosetta.core.scoring.score_type_from_name("atom_pair_constraint"), 1.0)
    sfx.set_weight(pyrosetta.rosetta.core.scoring.score_type_from_name("angle_constraint"), 1.0)
    sfx.set_weight(pyrosetta.rosetta.core.scoring.score_type_from_name("dihedral_constraint"), 1.0)     
    cst_mover = CSTs(args.cstfile, sfx)


if args.design_pos is None:
    design_pos = [] 
else:
    design_pos = args.design_pos

keep_pos = []
if args.keep_pos is not None:
    keep_pos = args.keep_pos


pose = pyr.pose_from_file(args.pdb)

ligand_seqpos = pose.size()
assert pose.residue(ligand_seqpos).is_ligand()
ligand = pose.residue(ligand_seqpos)


"""
Setting up design/repack layers
"""
catres = design_utils.get_matcher_residues(pose)

keep_pos += [x for x in catres.keys() if pose.residue(x).is_protein()]
keep_pos += [catres[r]["target_resno"] for r in catres if pose.pdb_rsd((catres[r]["target_chain"], catres[r]["target_resno"])) != 0]

## Identifying any motif residues based on pdbinfo reslabel
motif_label_sel = pyrosetta.rosetta.core.select.residue_selector.ResiduePDBInfoHasLabelSelector(label_str="motif")
keep_pos += list(pyrosetta.rosetta.core.select.get_residue_set_from_subset(motif_label_sel.apply(pose)))
keep_pos = list(set(keep_pos))
heavyatoms = design_utils.get_ligand_heavyatoms(pose)

# Finding out what residues belong to what layer, based on the CA distance
# from ligand heavyatoms.
SEL_mutate_residues, SEL_repack_residues, SEL_do_not_repack, residues\
    = design_utils.get_layer_selections(pose, keep_pos,
                                        design_pos, ligand_seqpos, heavyatoms)

CUTS = [9.0, 11.0, 13.0, 15.0]
if args.redesign is True:
    CUTS = [7.0, 9.0, 11.0, 13.0]  # smaller design range

if args.detect_pocket is False:
    if args.design_pos is not None:
        design_residues = design_pos
    else:
         ## Designing all residues that are not meant to stay fixed
        design_residues = [res.seqpos() for res in pose.residues if not res.is_ligand() and res.seqpos() not in keep_pos]
        repack_residues = list(set(keep_pos + [res.seqpos() for res in pose.residues if res.seqpos() not in design_residues]))
        do_not_touch_residues = []  # not really relevant with mpnn-packing. Was used formerly to not repack certain sidechains at all.
else:
    substrate_atoms_ref = [ligand.atom_name(n).strip() for n in range(1, ligand.natoms()+1) if ligand.atom_name(n).strip() not in ["ZN1", "O1"]]
    __a, __b, __c, residues_substrate\
        = design_utils.get_layer_selections(pose, keep_pos,
                                            design_pos, ligand_seqpos, substrate_atoms_ref, cuts=[7.0, 9.0, 11.0, 13.0])
    design_residues = [x for x in residues_substrate[0]+residues_substrate[1]]
    design_residues += design_utils.get_residues_with_close_sc(pose, substrate_atoms_ref, residues_substrate[2]+residues_substrate[3], keep_pos, 8.0)
    design_residues = list(set(design_residues))


repack_residues = residues[2] + residues[3] + residues[4]+ [ligand_seqpos]
do_not_touch_residues = []

for res in residues[0]+residues[1]:
    if res not in design_residues:
        repack_residues.append(res)
repack_residues = [x for x in repack_residues if x not in design_residues]

unclassified_residues = [res.seqpos() for res in pose.residues if res.seqpos() not in design_residues+repack_residues+do_not_touch_residues]
assert len(unclassified_residues) == 0, f"Some residues have not been layered: {unclassified_residues}"

print("Design positions: ", "+".join([str(x) for x in design_residues]))


## Defining the design protocol for FastMPNNDesign
if args.protocol is not None:
    protocol = args.protocol


aa_biases = {a: args.position_bias for a in "KREQDNHY"}
aa_biases["A"] = -0.5
aa_biases["R"] = args.position_bias-2.0
# aa_biases["Y"] = args.position_bias-0.5
if args.position_bias <= -1.0:
    aa_biases["W"] = args.position_bias+0.5  # not downweighting TRP as much as the rest

#### Setting up methods that integrate into FastMPNNdesign

## Defining a method that keeps H-bond contacts to ligand and motif fixed
ligand_and_catres_hbond_keeper = SelectHBondsToResidue(name="keep_hbonds_to_ligand_and_catres")
ligand_and_catres_hbond_keeper.target([ligand_seqpos]+list(catres))  # ligand and catalytic residues
ligand_and_catres_hbond_keeper.allow_updating(True)  # allowing updating the target set based on found H-bond contacts to these
ligand_and_catres_hbond_keeper.accept_probability(0.75)  # 75% chance of keeping an identified H-bond contact

## Defining scoring methods that are used to rank designs in the protocol
ddg_calculator = ScoringCalculators.ScoringCalculator(name="ddg")
ddg_calculator.descending(False)
ddg_calculator.set_calculator(ScoringCalculators.ddG)
ddg_calculator.scorefunction(sfx)

cms_calculator = ScoringCalculators.ScoringCalculator(name="cms")
cms_calculator.descending(True)
cms_calculator.set_calculator(ScoringCalculators.ContactMolecularSurface)


# Creating an instance of MPNN for 2nd layer mpnn redesign
if args.mpnn is True:
    mpnnrunner = FastMPNNdesign.mpnn_api.MPNNRunner("ligand_mpnn", verbose=True, pack_sc=False, ligand_mpnn_use_side_chain_context=True)

"""
DESIGN BLOCK
"""
for N_iter in range(args.nstruct):

    if len(glob.glob(f"{args.outdir}/{pdbname}{suffix}_{N_iter}_*.pdb")) > 0:
        print(f"{pdbname}{suffix}_{N_iter}_* outputs already exist")
        continue

    ## Performing quick constrained cartesian FastRelax on the input structure to get the ligand placement correct
    ## Mutating clashing non-motif residues to ALA
    ## Redoing this at each iteration because the outcome of this step can vary and affect downstream design
    _pose2 = perform_quick_prerelax_and_mutate_clashes_to_ALA(pose, design_residues, [ligand.seqpos()], cst_io=cst_mover.cst_io(), cartesian=False)
    if args.debug:
        _pose2.dump_pdb(f"{args.outdir}/{pdbname}_prerelax_{suffix}.pdb")

    ## Setting up apolar bias on positions close to user-defined atoms
    ## It really means applying a small bias against polar residues though...
    bias_positions_dict = {}
    if args.bias_atoms is not None:


        assert all([_pose2.residue(ligand_seqpos).has(a) for a in args.bias_atoms]), "Some --bias_atoms atom names are invalid."
        __a, __b, __c, residues_bias\
            = design_utils.get_layer_selections(_pose2, keep_pos, design_pos, ligand_seqpos, args.bias_atoms, cuts=[5.0, 7.0, 9.0, 11.0])
        bias_positions = [x for x in residues_bias[0]+residues_bias[1]]
        # bias_positions += design_utils.get_residues_with_close_sc(_pose2, args.bias_atoms, residues_bias[2]+residues_bias[3], keep_pos, 4.0)
        bias_positions = list(set(bias_positions))

        # Excluding positions that are closer to ligand H1 than to O1
        if "O1" not in args.bias_atoms and "H1" in args.bias_atoms:
            bias_positions2 = []
            for pos in bias_positions:
                _dist_nbr = (_pose2.residue(pos).xyz("CA") - _pose2.residue(ligand_seqpos).xyz("O1")).norm()
                _dists = [(_pose2.residue(pos).xyz("CA") - _pose2.residue(ligand_seqpos).xyz(a)).norm() for a in args.bias_atoms]
                if min(_dists) < _dist_nbr:
                    bias_positions2.append(pos)
            bias_positions = bias_positions2

        print(f"Bias positions for {args.bias_AAs}: ", "+".join([str(x) for x in bias_positions]))
        bias_positions_dict = {}
        for pos in bias_positions:
            bias_positions_dict[f"{_pose2.pdb_info().chain(pos)}{pos}"] = {a: args.position_bias for a in args.bias_AAs}

    else:
        aa_biases = {"A": -0.5, "R": -0.2, "E": -0.01, "W": -0.4, "Y": -0.4}


    ### Design part ###
    pose2 = _pose2.clone()

    # Adding constraints
    if args.cstfile is not None:
        cst_mover.remove_cst(pose2)
        if pose2.constraint_set().has_constraints():
            pose2.constraint_set().clear()
        cst_mover.add_cst(pose2)

    # Creating an instance of FastMPNNdesign
    fmd = FastMPNNdesign.FastMPNNdesign(model_type="ligand_mpnn", scorefxn=sfx, protocol=protocol,
                                        design_positions=design_residues,
                                        repack_positions=repack_residues, do_not_repack_positions=do_not_touch_residues,
                                        omit_AA="CM", mpnn_pack_sc=False)

    if args.bias_atoms is not None:
        if len(bias_positions_dict) != 0:
            fmd.mpnn_bias_per_residue(bias_positions_dict)
    else:
        fmd.mpnn_bias(aa_biases)

    fmd.add_mpnn_selector(ligand_and_catres_hbond_keeper)

    fmd.add_scoring_calculator(ddg_calculator)
    fmd.add_scoring_calculator(cms_calculator)

    # Running design
    poses = fmd.apply(pose2)

    # Analyzing design outputs
    for i, p in enumerate(poses):
        cst_mover.add_cst(p)
        scores_df = scoring.score_design(p, pyr.get_fa_scorefxn(), list(catres.keys()))
        if args.filter is True and len(scoring.filter_scores(scores_df)) == 0:
            print(f"BAD design: {pdbname}{suffix}_{N_iter}_{i}")
            continue
        scores_df.at[0, "description"] = f"{pdbname}{suffix}_{N_iter}_{i}"
        scoring_utils.dump_scorefile(scores_df, scorefilename)
        p.dump_pdb(f"{args.outdir}/{pdbname}{suffix}_{N_iter}_{i}.pdb")

        ## Performing 2nd layer MPNN on successful outputs
        if args.mpnn is True:
            fixed_residues = get_2nd_layer_fixed_pos(p, [ligand.seqpos()], heavyatoms, keep_pos)
            pdbstr = pyrosetta.distributed.io.to_pdbstring(p)
            for T in [0.1, 0.2]:
                mpnn_input = mpnnrunner.MPNN_Input()
                mpnn_input.fixed_residues = fixed_residues
                mpnn_input.omit_AA = ["C", "M"]
                mpnn_input.bias_AA = {"A": -0.5, "R": -1.0}
                mpnn_input.pdb = pdbstr
                mpnn_input.name = f"{pdbname}{suffix}_{N_iter}_{i}"
                mpnn_input.number_of_batches = 1
                mpnn_input.batch_size = 5
                mpnn_input.temperature = T
                mpnn_out = mpnnrunner.run(mpnn_input)

                with open(f"{args.outdir}/seqs/{pdbname}{suffix}_{N_iter}_{i}.fasta", "a") as file:
                    file.write(f">{pdbname}{suffix}_{N_iter}_{i}_native\n")
                    file.write(mpnn_out["native_sequence"]+"\n")
                    for j, s in enumerate(mpnn_out["generated_sequences"]):
                        file.write(f">{pdbname}{suffix}_{N_iter}_{i}_T{T}_s0_{j}\n")
                        file.write(s+"\n")
