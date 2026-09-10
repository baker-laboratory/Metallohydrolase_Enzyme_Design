"""
Extra modules for scoring protein structures
Authors: Chris Norn, Indrek Kalvet
"""
import os, sys
import pyrosetta
import pyrosetta.rosetta
import numpy as np
from pyrosetta.rosetta.core.scoring import fa_rep
import math
import pandas as pd
import itertools
import Bio.PDB
BIO_PDB_parser = Bio.PDB.PDBParser(QUIET=True)


comparisons = {'<=': '__le__',
               '<': '__lt__',
               '>': '__gt__',
               '>=': '__ge__',
               '=': '__eq__'}


def get_one_and_twobody_energies(p, scorefxn):
    nres = p.size()    
    res_energy_z = np.zeros(nres)
    res_pair_energy_z = np.zeros( (nres,nres) )
    res_energy_no_two_body_z = np.zeros ( (nres) )

    totE = scorefxn(p)
    energy_graph = p.energies().energy_graph()

    twobody_terms = p.energies().energy_graph().active_2b_score_types()
    onebody_weights = pyrosetta.rosetta.core.scoring.EMapVector()
    onebody_weights.assign(scorefxn.weights())

    for term in twobody_terms:
        if 'intra' not in pyrosetta.rosetta.core.scoring.name_from_score_type(term):
            onebody_weights.set(term, 0)

    for i in range(1,p.size()+1):
        res_energy_no_two_body_z[i-1] = p.energies().residue_total_energies(i).dot(onebody_weights)
        res_energy_z[i-1]= p.energies().residue_total_energy(i)
    
        for j in range(1,p.size()+1):
            if i == j: continue
            edge = energy_graph.find_edge(i,j)
            if edge is None:
                energy = 0.0
            else:
                res_pair_energy_z[i-1][j-1]= edge.fill_energy_map().dot(scorefxn.weights())

    one_body_tot = np.sum(res_energy_z)
    one_body_no_two_body_tot = np.sum(res_energy_no_two_body_z)
    two_body_tot = np.sum(res_pair_energy_z)

    onebody_energies = res_energy_no_two_body_z
    twobody_energies = res_pair_energy_z  # This matrix is symmetrical, 0 diagonal, and when summed only the half matrix sohuld be summed

    return onebody_energies, twobody_energies


def fix_scorefxn(sfxn, allow_double_bb=False):
    opts = sfxn.energy_method_options()
    opts.hbond_options().decompose_bb_hb_into_pair_energies(True)
    opts.hbond_options().bb_donor_acceptor_check(not allow_double_bb)
    sfxn.set_energy_method_options(opts)


def calculate_ddg(pose, scorefxn, ser_idx=None):
    """
    Calculates interaction ddG based on twobody energies.
    Corrected for repulsive terms at cst-bonded residuepairs.

    Parameters
    ----------
    pose : TYPE
        DESCRIPTION.
    scorefxn : TYPE
        scorefunction where 'decompose_bb_hb_into_pair_energies' is True.
    ser_idx : TYPE, optional
        Not used anymore.

    Returns
    -------
    ddg : float
        DESCRIPTION.

    """
    ligands = [r for r in pose.residues if r.is_ligand()]

    twobody_energies = get_one_and_twobody_energies(pose, scorefxn)[1]
    scorefxn.set_weight(pyrosetta.rosetta.core.scoring.fa_rep, 0.0)
    twobody_energies_no_fa_rep = get_one_and_twobody_energies(pose, scorefxn)[1]

    # Figuring out which residues have any covalent bonds defined through constraints
    cst_covalents = []
    for lig in ligands:
        for res2 in pose.residues:
            if lig.seqpos() == res2.seqpos():
                continue
            if lig.is_bonded(res2):
                cst_covalents.append((lig, res2))

    protein_pos = np.array([r.seqpos()-1 for r in pose.residues if r.is_protein()])

    ddg = 0.0
    for lig in ligands:
        ddg += np.sum(twobody_energies[lig.seqpos()-1][protein_pos])
    # adding repulsion-corrected twobody energies of cst-bonded residuepairs
    if len(cst_covalents) > 0:
        for covpair in cst_covalents:
            ddg += (- twobody_energies[covpair[0].seqpos()-1][covpair[1].seqpos()-1] \
                    + twobody_energies_no_fa_rep[covpair[0].seqpos()-1][covpair[1].seqpos()-1])
    return ddg


def calculate_ddg_PPI(pose, scorefxn, binder_chain):
    """
    Sums up pairwise energies between binder and target chains. Ignoring intra-chain pairwise interactions.
    """
    twobody_energies = get_one_and_twobody_energies(pose, scorefxn)[1]
    ddg = 0.0
    target_chains = [n+1 for n in range(pose.num_chains()) if n+1 != binder_chain]
    for tc in target_chains:
        #TODO: is there a bug in how the target chain residue numbers are handled? pose.chain_begin(tc)+1: seems to assume the target chain is after binder chain?
        ddg += sum([np.sum(twobody_energies[n][pose.chain_begin(tc)+1:]) for n in range(pose.chain_begin(binder_chain), pose.chain_end(binder_chain)+1) ])
    return ddg


def calculate_dG(pose, scorefxn, target_resnos=None):
    """
    Calculates dG of protein-ligand interaction using the GALigandDock method
    with entropy estimation.
    """
    entropy_method_ = "MCEntropy"
    ligids = pyrosetta.rosetta.utility.vector1_unsigned_long()
    if target_resnos is None:
        for res in pose.residues:
            if res.is_ligand():
                ligids.append(res.seqpos())
    else:
        for rn in target_resnos:
            ligids.append(rn)

    entropy_estimator = pyrosetta.rosetta.protocols.ligand_docking.ga_ligand_dock.EntropyEstimator(scorefxn, pose, ligids, entropy_method_)
    entropy_estimator.set_niter( 2000 )
    TdS = entropy_estimator.apply(pose)
    cplx_score = scorefxn(pose)
    pose_tmp = pose.clone()
    pose_tmp.energies().clear()
    pose_tmp.data().clear()

    movable_scs = pyrosetta.rosetta.protocols.ligand_docking.ga_ligand_dock.get_atomic_contacting_sidechains(pose_tmp, ligids, 4.5)
    ligscore = calculate_free_ligand_score(pose_tmp.clone(), ligids, scorefxn)
    recscore = calculate_free_receptor_score(pose_tmp.clone(), ligids, movable_scs, True, scorefxn)
    dH = cplx_score - ligscore - recscore
    dG = dH + TdS
    return dG


def calculate_free_ligand_score(pose_ref, lig_resnos, sfx):
    """
    Ported from C++
    rosetta/main/source/src/protocols/ligand_docking/GALigandDock/GALigandDock.cc
    GALigandDock::calculate_free_ligand_score
    """
	# make a ligand-only pose; root ligand on virtual if no residues to anchor jump
    # utility.vector1< core.Size > freeligresids
    pose = pyrosetta.rosetta.core.pose.Pose()
    pose.append_residue_by_jump( pose_ref.residue(lig_resnos[1]), 0 )

    for i in range(2, len(lig_resnos)+1):
        pose.append_residue_by_bond( pose_ref.residue(lig_resnos[i]) )


    pyrosetta.rosetta.core.pose.initialize_disulfide_bonds( pose )
    pyrosetta.rosetta.core.pose.addVirtualResAsRoot(pose)

    # optimize slightly...
    scfxn_ligmin = sfx.clone()
    scfxn_ligmin.set_weight( pyrosetta.rosetta.core.scoring.coordinate_constraint, 1.0 )

    anchorid = pyrosetta.rosetta.core.id.AtomID(1, pose.fold_tree().root())
    for ires in range(1, pose.total_residue()):
        for iatm in range(1, pose.residue(ires).natoms()):
            atomid = pyrosetta.rosetta.core.id.AtomID(iatm, ires)
            xyz = pose.xyz( atomid )
            fx = pyrosetta.rosetta.core.scoring.func.Func(pyrosetta.rosetta.core.scoring.func.HarmonicFunc(0.0, 1.0))
            pose.add_constraint(pyrosetta.rosetta.core.scoring.constraints.Constraint( pyrosetta.rosetta.core.scoring.constraints.CoordinateConstraint( atomid, anchorid, xyz, fx ) ))

    mm = pyrosetta.rosetta.core.kinematics.MoveMap()
    mm.set_bb( True )
    mm.set_chi( True )
    mm.set_jump( True )

    min_mover = pyrosetta.rosetta.protocols.minimization_packing.MinMover(pyrosetta.rosetta.protocols.minimization_packing.MinMover( mm, scfxn_ligmin, "linmin", 0.01, True ))
    min_mover.max_iter( 30 )
    min_mover.apply( pose )
    # turn off coordinate constraint in the actual scoring
    scfxn_ligmin.set_weight( pyrosetta.rosetta.core.scoring.coordinate_constraint, 0.0 )
    ligandscore = scfxn_ligmin.score( pose )

    return ligandscore


def calculate_free_receptor_score(pose, lig_resnos, moving_scs, simple, sfx):
    """
    Ported from C++
    rosetta/main/source/src/protocols/ligand_docking/GALigandDock/GALigandDock.cc
    GALigandDock.calculate_free_receptor_score
    """
    if pose.size() == 1:
        return 0.0  # ligand-only

    # necessary for memory issue?
    pose.energies().clear()
    pose.data().clear()

    # delete ligand
    # startid = ( lig_resnos[1] <= lig_resnos.back())? lig_resnos[1] : lig_resnos.back()
    # endid = ( lig_resnos[1] > lig_resnos.back())? lig_resnos[1] : lig_resnos.back()
    startid = lig_resnos[1] if lig_resnos[1] <= lig_resnos[-1] else lig_resnos[-1]
    endid = lig_resnos[1] if lig_resnos[1] > lig_resnos[-1] else lig_resnos[-1]
    pose.delete_residue_range_slow(startid, endid)

    if simple:
        score = sfx(pose)
        return score

    task = pyrosetta.rosetta.core.pack.task.TaskFactory.create_packer_task( pose )
    task.initialize_from_command_line()
    task.or_optimize_h_mode( False )  # to reduce noise
    task.or_flip_HNQ( False )  # to reduce noise

    moving_his_scs = []
    # utility.vector1< core.Size > moving_his_scs
    for ires in range(1, moving_scs.size()):
        if pose.residue(moving_scs[ires]).aa() == pyrosetta.rosetta.core.chemical.aa_his:
            moving_his_scs.push_back( moving_scs[ires] )

    task.or_fix_his_tautomer( moving_his_scs, True )  # to reduce noise
    # task.or_multi_cool_annealer( true )

    tf = pyrosetta.rosetta.core.pack.task.TaskFactory()
    tf.modify_task( pose, task )

    mm = pyrosetta.rosetta.core.kinematics.MoveMap()
    mm.set_bb( False )
    mm.set_chi( False )
    mm.set_jump( False )
    for resid in moving_scs:
        mm.set_chi( resid, True )


    # let's use hard-coded scopt schedule for now...
    relax = pyrosetta.rosetta.protocols.relax.FastRelax()
    relax.set_task_factory( tf )

    # sometimes just stuck here wasting huge memory... why is it?

    lines = pyrosetta.rosetta.std.vector_std_string()
    fast_relax_script_file_ = ""
    if fast_relax_script_file_ != "":
        # TR << "==== Use FastRelax script: " << fast_relax_script_file_ << std.endl
        relax = pyrosetta.rosetta.protocols.relax.FastRelax( sfx, fast_relax_script_file_ )
    else:
        # TR << "==== Use FastRelax hardcoded. "<< std.endl
        lines.append( "switch:torsion" )
        lines.append( "repeat 3" )
        lines.append( "ramp_repack_min 0.02 0.01 1.0 50" )
        lines.append( "ramp_repack_min 1.0  0.00001 0.0 50" )
        lines.append( "accept_to_best" )
        lines.append( "endrepeat" )
        relax.set_script_from_lines( lines )
        relax.set_scorefxn( sfx )

    relax.set_movemap( mm )
    relax.set_movemap_disables_packing_of_fixed_chi_positions( True )
    relax.apply( pose )

    score = sfx(pose)

    return score


def mutate_pdb(pdb, site, mutant_aa, output_file):
    pose = pyrosetta.pose_from_pdb(pdb)
    pyrosetta.toolbox.mutants.mutate_residue(pose, site, mutant_aa)
    pyrosetta.dump_pdb(pose, output_file)


def apply_score_from_filter(pose, filter_obj):
    pyrosetta.rosetta.core.pose.setPoseExtraScore(pose,
                                                  filter_obj.get_user_defined_name(),
                                                  filter_obj.score(pose))


def dump_scorefile(df, filename):
    widths = {}
    for k in df.keys():
        if k in ["SCORE:", "description", "name"]:
            widths[k] = 0
        elif isinstance(df.at[df.index.values[0], k], str):
            max_val_len = max([len(row[k]) for index, row in df.iterrows()])
            widths[k] = max(max_val_len, len(k)) + 1
        elif len(k) >= 12:
            widths[k] = len(k) + 1
        else:
            widths[k] = 12

    keys = df.keys()
    write_title = True
    if os.path.exists(filename):
        write_title = False
        keys = open(filename, 'r').readlines()[0].split()
        keys = [x.rstrip() for x in keys]
        if len(keys) != len(df.keys()):
            print(f"Number of columns in existing scorefile {filename} and "
                  f"scores dataframe does not match: {len(keys)} != {len(df.keys())}")

    with open(filename, "a") as file:
        title = ""
        if write_title is True:
            for k in df.keys():
                if k == "SCORE:":
                    title += k
                elif k in ["description", "name"]:
                    continue
                else:
                    title += f"{k:>{widths[k]}}"
            if 'description' in df.keys():
                title += " description"
            file.write(title + "\n")

        for index, row in df.iterrows():
            line = ""
            for k in keys:
                if k not in df.keys():
                    val = f"{np.nan}"
                    widths[k] = 11
                elif isinstance(row[k], (float, np.float16, np.float64, np.float32)):
                    val = f"{row[k]:.3f}"
                else:
                    val = row[k]
                if k == "SCORE:":
                    line += val
                elif k in ["description", "name"]:
                    continue
                else:
                    line += f"{val:>{widths[k]}}"
            # Always writing PDB name as the last item
            if 'description' in df.keys():
                line += f" {row['description']}"
            file.write(line + "\n")


def getSASA(pose, resno=None, SASA_atoms=None, ignore_sc=False):
    """
    Takes in a pose and calculates its SASA.
    Or calculates SASA of a given residue.
    Or calculates SASA of specified atoms in a given residue.

    Procedure by Brian Coventry
    """

    atoms = pyrosetta.rosetta.core.id.AtomID_Map_bool_t()
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

    surf_vol = pyrosetta.rosetta.core.scoring.packing.get_surf_vol(pose, atoms, 1.4)

    if resno is not None:
        res_surf = 0.0
        for i in range(1, pose.residue(resno).natoms()+1):
            if SASA_atoms is not None and i not in SASA_atoms:
                continue
            res_surf += surf_vol.surf(resno, i)
        return res_surf
    else:
        return surf_vol


def find_hbonds_to_residue_atom(pose, target_seqpos, target_atom):
    """
    Counts how many Hbond contacts input atom has with the protein.
    """
    HBond_res = 0

    target = pose.residue(target_seqpos)
    
    if isinstance(target_atom, int):
        target_atomno = target_atom
        target_atom = target.atom_name(target_atomno)
    else:
        target_atomno = target.atom_index(target_atom)

    for res in pose.residues:
        if res.seqpos() == target_seqpos or res.is_ligand():
            break
        if (target.xyz(target_atom) - res.xyz('CA')).norm() < 10.0:
            if target.atom_type(target_atomno).element() != "H":
                for polar_H in res.Hpos_polar():
                    if (target.xyz(target_atom) - res.xyz(polar_H)).norm() < 2.5:
                        # If the polar atom is from the backbone then check that the X-H...Y angle is close to linear.
                        # It is assumed that polar backbone H is only attached to backbone N
                        if res.atom_is_backbone(polar_H):
                            # print(res.seqpos(), target_atom, res.atom_name(polar_H), get_angle(res.xyz(1), res.xyz(polar_H), pose.residue(lig_seqpos).xyz(target_atom)))
                            if get_angle(res.xyz(1), res.xyz(polar_H), target.xyz(target_atom)) < 140.0:
                                continue
                        HBond_res += 1
                        break
            else:
                for acceptor in res.accpt_pos():
                    if (target.xyz(target_atom) - res.xyz(acceptor)).norm() < 2.5:
                        # check that the X-H...Y angle is close to linear.
                        if get_angle(res.xyz(acceptor), target.xyz(target_atom), target.xyz(target.get_adjacent_heavy_atoms(target_atomno)[1])) < 140.0:
                            continue
                        HBond_res += 1
                        break
    return HBond_res


def find_res_with_hbond_to_residue_atom(pose, lig_seqpos, target_atom):
    """
    Returns the residue numbers that form Hbond contacts with input atom
    """
    HBond_res = 0
    residues = []
    for res in pose.residues:
        if res.seqpos() == lig_seqpos or res.is_ligand():
            continue
        if (pose.residue(lig_seqpos).xyz(target_atom) - res.xyz('CA')).norm() < 10.0:
            for polar_H in res.Hpos_polar():
                if (pose.residue(lig_seqpos).xyz(target_atom) - res.xyz(polar_H)).norm() < 2.5:
                    # If the polar atom is from the backbone then check that the X-H...Y angle is close to linear.
                    # It is assumed that polar backbone H is only attached to backbone N
                    if res.atom_is_backbone(polar_H):
                        # print(get_angle(res.xyz(1), res.xyz(polar_H), pose.residue(lig_seqpos).xyz(target_atom)))
                        if get_angle(res.xyz(1), res.xyz(polar_H), pose.residue(lig_seqpos).xyz(target_atom)) < 140.0:
                            continue
                    residues.append(res.seqpos())
                    HBond_res += 1
                    break
    return residues


def get_hbonds_to_residue(pose, resnos):

    ## Calculating HBonds in the pose (Sam Pellock implementation)
    hbonds = pose.get_hbonds()

    # Create empty lists to store the data
    donor_residues = []
    acceptor_residues = []
    donor_atoms = []
    acceptor_atoms = []
    hbond_energies = []

    # Loop over each HBond in the HBondSet and extract the information
    for hbond in hbonds.hbonds():
        donor_residues.append(hbond.don_res())
        acceptor_residues.append(hbond.acc_res())
        donor_atoms.append(pose.residue(hbond.don_res()).atom_name(hbond.don_hatm()))
        acceptor_atoms.append(pose.residue(hbond.acc_res()).atom_name(hbond.acc_atm()))
        hbond_energies.append(hbond.energy())

    # Create the pandas DataFrame from the lists
    hbond_df = pd.DataFrame({
        'donor_residue': donor_residues,
        'acceptor_residue': acceptor_residues,
        'donor_atom': [x.strip() for x in donor_atoms],
        'acceptor_atom': [x.strip() for x in acceptor_atoms],
        'energy': hbond_energies
        })

    # Filter the DataFrame to only include hydrogen bonds involving the ligand
    mask = (hbond_df['donor_residue'].isin(resnos)) | (hbond_df['acceptor_residue'].isin(resnos))
    target_hbond_df = hbond_df[mask]
    return target_hbond_df


def get_angle(a1, a2, a3):
    a1 = np.array(a1)
    a2 = np.array(a2)
    a3 = np.array(a3)

    ba = a1 - a2
    bc = a3 - a2

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return round(np.degrees(angle), 1)


def get_dihedral(a1, a2, a3, a4):
    """
    a1, a2, a3, a4 (np.array)
    Each array has to contain 3 floats corresponding to X, Y and Z of an atom.
    Solution by 'Praxeolitic' from Stackoverflow:
    https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python#
    1 sqrt, 1 cross product
    Calculates the dihedral/torsion between atoms a1, a2, a3 and a4
    Output is in degrees
    """

    b0 = a1 - a2
    b1 = a3 - a2
    b2 = a4 - a3

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= np.linalg.norm(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.dot(b0, b1)*b1
    w = b2 - np.dot(b2, b1)*b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))


def calculate_CA_rmsd(input_pose, design_pose, residue_list=None):
    reslist = pyrosetta.rosetta.std.list_unsigned_long_t()
    if residue_list is None:
        for n in range(1, input_pose.size()+1):
            if input_pose.residue(n).is_ligand():
                continue
            if input_pose.residue(n).is_virtual_residue():
                continue
            reslist.append(n)
    else:
        for n in residue_list:
            reslist.append(n)

    rmsd_CA = pyrosetta.rosetta.core.scoring.CA_rmsd(input_pose, design_pose, residue_selection=reslist)
    return rmsd_CA


def get_per_res_rmsd(design: pyrosetta.rosetta.core.pose.Pose, prediction: pyrosetta.rosetta.core.pose.Pose) -> list:
    """
    calculate per residue rmsd (sc) of prediction to design.
    Using Rosetta automorphic rmsd calculator.
    Poses are NOT aligned by this function.
    """
    # per_res_rmsd = pyrosetta.rosetta.core.simple_metrics.per_residue_metrics.PerResidueRMSDMetric()
    
    result = {}
    for resno in range(1, prediction.size()+1):
        resp = prediction.residue(resno)
        resd = design.residue(resno)
        if resd.is_ligand():
            continue
        if resp.name3() != resd.name3():
            result[resno-1] = 0.0
        else:
            try:
                result[resno-1] = pyrosetta.rosetta.core.scoring.automorphic_rmsd(resp, resd, False)
            except RuntimeError:
                print(resp.name(), resd.name())
                sys.exit(0)
            # atoms = [resp.atom_name(n).strip() for n in range(1, resp.natoms()+1) if not resp.atom_is_hydrogen(n)]
            # ref_coords = [resd.xyz(a) for a in atoms]
            # mdl_coords = [resp.xyz(a) for a in atoms]
            # result[resno-1] = np.sqrt(sum([(np.linalg.norm(c1-c2))**2 for c1, c2 in zip(ref_coords, mdl_coords)])/len(atoms))
    return result


def get_rmsd(coords1, coords2):
    """
    Calculates the rmsd between two sets of coordinates as Numpy arrays
    """
    assert len(coords1) == len(coords2)
    return np.sqrt(sum([(np.linalg.norm(c1-c2))**2 for c1, c2 in zip(coords1, coords2)])/len(coords2))


def calculate_pnear( scores, rmsds, lambda_val=1.5, kbt=0.62 ):
    #Pnear calculation function by Vikram from tools/analysis/compute_pnear.py
    # Given a vector of scores, a matching vector of rmsds, and values for lambda and kbt,
    # compute the PNear value.
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



def find_mutations(parent_pose, designed_pose):
    mutations = {}
    for res_p, res_d in zip(parent_pose.residues, designed_pose.residues):
        if res_p.name3() != res_d.name3():
            mutations[res_p.seqpos()] = {'from': res_p.name3(),
                                         'to': res_d.name3()}
    return mutations


def get_residue_subset_lddt(lddt, residues):
    """
    Calculates the average lDDT of a subset of residues
    """
    lddt_desres = [lddt[x-1] for x in residues]
    return np.average(lddt_desres)


def _fix_CYX_pdbfile(pdbfile):
    pdbf = open(pdbfile, "r").readlines()
    
    new_pdbf = []
    for l in pdbf:
        if "CYX" in l:
            new_pdbf.append(l.replace("CYX", "CYS"))
        else:
            new_pdbf.append(l)
    temp_pdb = pdbfile.replace(".pdb", "_tmp.pdb")
    with open(temp_pdb, "w") as file:
        for l in new_pdbf:
            file.write(l)
    return temp_pdb


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

