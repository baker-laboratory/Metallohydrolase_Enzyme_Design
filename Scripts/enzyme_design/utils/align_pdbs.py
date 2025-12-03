#!/home/dimaio/.conda/envs/SE3nv/bin/python
# script for aligning two proteins on arbitrary sets of their (heavy) atoms with the Kabsch algorithm 
import os,sys,argparse 
import kinematics 
import util 
import parsers
# from icecream import ic 
# import torch 
import numpy as np
import pyrosetta as pyr
import pyrosetta.rosetta


def find_atom_idx(atom, mapping):
    for i,A in enumerate(mapping):
        try:
            if A.strip() == atom:
                return i
        except AttributeError:
            print('This is atom ',A)

    raise KeyError(f'Could not find atom {atom} in mapping {mapping}')

def get_xyz_stack(pdb, atoms_list):
    """
    Extracts the xyz crds corresponding to every atom in atoms_list 
    """

    parsed = parsers.parse_pdb(pdb)
    xyz_all = parsed['xyz']
    pdb_idx = parsed['pdb_idx']
    seq     = parsed['seq']
    xyz_out = []
    
    # for each atom, get residue index and atom index 
    # store crds 
    for (resn, atom) in atoms_list:
        
        print(resn,atom)
        # get index of residue and its Heavy atom mapping
        res_idx0 = pdb_idx.index((resn[0], int(resn[1:])))
        AA_int = seq[res_idx0]
        AA_long_map = util.aa2long[AA_int]
        
        # get index of atom in residue 
        atom_idx0 = find_atom_idx(atom, AA_long_map)

        # crds of this atom 
        xyz_atom = xyz_all[res_idx0, atom_idx0, :]

        xyz_out.append(xyz_atom)

    return np.array(xyz_out), parsed


def parse_pose_coords(pose):
    res = [r.seqpos() for r in pose.residues if not r.is_ligand() and not r.is_virtual_residue()]
    xyz = np.full((len(res), 26, 3), np.nan, dtype=np.float32)
    for r in pose.residues:
        if r.is_ligand() or r.is_virtual_residue():
            continue
        # rc = np.ndarray((res.natoms(), 3), dtype=np.float32)
        for n in range(r.natoms()):
            try:
                xyz[r.seqpos()-1][n] = r.xyz(n+1)
            except IndexError:
                print(r.name())
                print(r.seqpos())
                print(r.natoms())
                sys.exit(1)
    return xyz


def parse_residue_coords(residue):
    xyz = np.full((1, 26, 3), np.nan, dtype=np.float32)
    if residue.is_ligand() or residue.is_virtual_residue():
        return None
    # rc = np.ndarray((res.natoms(), 3), dtype=np.float32)
    for n in range(residue.natoms()):
        xyz[0][n] = residue.xyz(n+1)
    return xyz


def get_xyz_stack_pose(pose, atoms_list):
    """
    Extracts the xyz crds corresponding to every atom in atoms_list 
    atoms_list format: [(resno, atomname), (resno, atomname), ...]
    """

    xyz_all = parse_pose_coords(pose)
    seq = [util.alpha_1.index(r.name1()) for r in pose.residues if not r.is_ligand() and not r.is_virtual_residue()]
    xyz_out = []

    # for each atom, get residue index and atom index 
    # store crds 
    for (resn, atom) in atoms_list:
        # get index of residue and its Heavy atom mapping
        AA_int = seq[resn-1]
        if pose.residue(resn).is_lower_terminus():
            AA_long_map = util.aa2longH_Nterm[AA_int]
        elif pose.residue(resn).is_upper_terminus():
            AA_long_map = util.aa2longH_Cterm[AA_int]
        else:
            AA_long_map = util.aa2longH[AA_int]

        # get index of atom in residue 
        atom_idx0 = find_atom_idx(atom.strip(), AA_long_map)

        # crds of this atom 
        xyz_atom = xyz_all[resn-1, atom_idx0, :]

        xyz_out.append(xyz_atom)

    return np.array(xyz_out), xyz_all


def get_xyz_stack_residue(residue, atoms_list):
    """
    Extracts the xyz crds corresponding to every atom in atoms_list 
    atoms_list format: [(resno, atomname), (resno, atomname), ...]
    """
    if residue.is_ligand() or residue.is_virtual_residue():
        return None, None

    xyz_all = parse_residue_coords(residue)
    seq = [util.alpha_1.index(residue.name1())]
    xyz_out = []

    # for each atom, get residue index and atom index 
    # store crds 
    for atom in atoms_list:
        # get index of residue and its Heavy atom mapping
        AA_int = seq[0]

        if residue.is_lower_terminus():
            AA_long_map = util.aa2longH_Nterm[AA_int]
        elif residue.is_upper_terminus():
            AA_long_map = util.aa2longH_Cterm[AA_int]
        else:
            AA_long_map = util.aa2longH[AA_int]

        # get index of atom in residue 
        atom_idx0 = find_atom_idx(atom.strip(), AA_long_map)

        # crds of this atom 
        xyz_atom = xyz_all[0, atom_idx0, :]

        xyz_out.append(xyz_atom)

    return np.array(xyz_out), xyz_all


def align_poses(ref_pose, mobile_pose, ref_atoms):
    xyz1, parsed1 = get_xyz_stack_pose(ref_pose, ref_atoms["atoms1"])
    xyz2, parsed2 = get_xyz_stack_pose(mobile_pose, ref_atoms["atoms2"])

    # run Kabsch to get rotation matrix for atoms and rmsd
    # aligns xyz2 onto xyz1
    rmsd, _, R = kinematics.np_kabsch(xyz1, xyz2)
    print('RMSD between atoms: ',rmsd)

    # (1) now translate both proteins such that centroid(xyz1/xyz2) is at origin
    # (2) rorate xyz2 onto xyz1 with R
    # (3) write pdbs into outdir

    def centroid(X):
        # return the mean X,Y,Z down the atoms
        return np.mean(X, axis=0, keepdims=True)

    # centroid of just the points being aligned
    centroid1 = centroid(xyz1)
    centroid2 = centroid(xyz2)

    # (1)
    #xyz_protein1 = np.copy(parsed1['xyz']) - centroid1
    xyz_protein2 = np.copy(parsed2) - centroid2

    # (2)
    xyz_protein2 = xyz_protein2 @ R

    # Translate protein 2 to where it aligns with original protein 1
    xyz_protein2 += centroid1
    
    out_pose = mobile_pose.clone()
    for resno, res_coords in enumerate(xyz_protein2):
        for i, ac in enumerate(res_coords):
            if np.isnan(ac[0]):
                break
            out_pose.residue(resno+1).set_xyz(i+1, pyrosetta.rosetta.numeric.xyzVector_double_t(*ac))
            continue
    return out_pose


def align_pose_to_residue(ref_residue, mobile_pose, ref_atoms):
    xyz1, parsed1 = get_xyz_stack_residue(ref_residue, ref_atoms["atoms1"])
    xyz2, parsed2 = get_xyz_stack_pose(mobile_pose, ref_atoms["atoms2"])

    # run Kabsch to get rotation matrix for atoms and rmsd
    # aligns xyz2 onto xyz1
    rmsd, _, R = kinematics.np_kabsch(xyz1, xyz2)
    print('RMSD between atoms: ',rmsd)

    # (1) now translate both proteins such that centroid(xyz1/xyz2) is at origin
    # (2) rorate xyz2 onto xyz1 with R
    # (3) write pdbs into outdir

    def centroid(X):
        # return the mean X,Y,Z down the atoms
        return np.mean(X, axis=0, keepdims=True)

    # centroid of just the points being aligned
    centroid1 = centroid(xyz1)
    centroid2 = centroid(xyz2)

    # (1)
    #xyz_protein1 = np.copy(parsed1['xyz']) - centroid1
    xyz_protein2 = np.copy(parsed2) - centroid2

    # (2)
    xyz_protein2 = xyz_protein2 @ R

    # Translate protein 2 to where it aligns with original protein 1
    xyz_protein2 += centroid1
    
    out_pose = mobile_pose.clone()
    for resno, res_coords in enumerate(xyz_protein2):
        for i, ac in enumerate(res_coords):
            if np.isnan(ac[0]):
                break
            out_pose.residue(resno+1).set_xyz(i+1, pyrosetta.rosetta.numeric.xyzVector_double_t(*ac))
            continue
    return out_pose


def align_residue_to_residue(ref_residue, mobile_residue, ref_atoms):
    xyz1, parsed1 = get_xyz_stack_residue(ref_residue, ref_atoms["atoms1"])
    xyz2, parsed2 = get_xyz_stack_residue(mobile_residue, ref_atoms["atoms2"])

    # run Kabsch to get rotation matrix for atoms and rmsd
    # aligns xyz2 onto xyz1
    rmsd, _, R = kinematics.np_kabsch(xyz1, xyz2)
    if rmsd > 0.1:
        print('RMSD between atoms: ',rmsd)

    # (1) now translate both proteins such that centroid(xyz1/xyz2) is at origin
    # (2) rorate xyz2 onto xyz1 with R
    # (3) write pdbs into outdir

    def centroid(X):
        # return the mean X,Y,Z down the atoms
        return np.mean(X, axis=0, keepdims=True)

    # centroid of just the points being aligned
    centroid1 = centroid(xyz1)
    centroid2 = centroid(xyz2)

    # (1)
    #xyz_protein1 = np.copy(parsed1['xyz']) - centroid1
    xyz_protein2 = np.copy(parsed2) - centroid2

    # (2)
    xyz_protein2 = xyz_protein2 @ R

    # Translate protein 2 to where it aligns with original protein 1
    xyz_protein2 += centroid1
    
    out_residue = mobile_residue.clone()

    for i, ac in enumerate(xyz_protein2[0]):
        if np.isnan(ac[0]):
            break
        out_residue.set_xyz(i+1, pyrosetta.rosetta.numeric.xyzVector_double_t(*ac))
        continue
    return out_residue


def align_poses_atom_subset(ref_pose, mobile_pose, ref_atoms):
    xyz1, parsed1 = get_xyz_stack_pose(ref_pose, ref_atoms["atoms1"])
    xyz2, parsed2 = get_xyz_stack_pose(mobile_pose, ref_atoms["atoms2"])

    # run Kabsch to get rotation matrix for atoms and rmsd
    # aligns xyz2 onto xyz1
    rmsd, _, R = kinematics.np_kabsch(xyz1, xyz2)
    print('RMSD between atoms: ',rmsd)

    # (1) now translate both proteins such that centroid(xyz1/xyz2) is at origin
    # (2) rorate xyz2 onto xyz1 with R
    # (3) write pdbs into outdir

    def centroid(X):
        # return the mean X,Y,Z down the atoms
        return np.mean(X, axis=0, keepdims=True)

    # centroid of just the points being aligned
    centroid1 = centroid(xyz1)
    centroid2 = centroid(xyz2)

    # (1)
    #xyz_protein1 = np.copy(parsed1['xyz']) - centroid1
    xyz_protein2 = np.copy(parsed2) - centroid2

    # (2)
    xyz_protein2 = xyz_protein2 @ R

    # Translate protein 2 to where it aligns with original protein 1
    xyz_protein2 += centroid1
    
    out_pose = mobile_pose.clone()
    for resno, res_coords in enumerate(xyz_protein2):
        for i, ac in enumerate(res_coords):
            if np.isnan(ac[0]):
                break
            out_pose.residue(resno+1).set_xyz(i+1, pyrosetta.rosetta.numeric.xyzVector_double_t(*ac))
            continue
    out_pose.scores["rmsd"] = rmsd
    return out_pose


def get_pose_coords(pose, atoms_list=None):
    # xyz = []
    xyz_subset = []
    longest_residue = max([res.natoms() for res in pose.residues])
    # xyz = np.ndarray((pose.size(), longest_residue, 3))
    xyz = np.full((pose.size(), longest_residue, 3), np.nan, dtype=np.float32)
    for res in pose.residues:
        for n in range(res.natoms()):
            xyz[res.seqpos()-1][n] = np.array(res.xyz(n+1))
        
    if atoms_list is not None:
        for (resno, aname) in atoms_list:
            xyz_subset.append(np.array(pose.residue(resno).xyz(aname)))
    return np.array(xyz_subset), np.array(xyz)


def align_pose_to_coords(ref_coords, mobile_pose, align_atoms):
    """
    align atoms: [(resno1, atom1), (resno2, atom2)...]
    """
    # xyz1, parsed1 = get_xyz_stack_pose(ref_pose, ref_atoms["atoms1"])
    xyz1 = ref_coords
    xyz2, parsed2 = get_pose_coords(mobile_pose, align_atoms)

    # run Kabsch to get rotation matrix for atoms and rmsd
    # aligns xyz2 onto xyz1
    rmsd, _, R = kinematics.np_kabsch(xyz1, xyz2)
    # print('RMSD between atoms: ',rmsd)

    # (1) now translate both proteins such that centroid(xyz1/xyz2) is at origin
    # (2) rorate xyz2 onto xyz1 with R
    # (3) write pdbs into outdir

    def centroid(X):
        # return the mean X,Y,Z down the atoms
        return np.mean(X, axis=0, keepdims=True)

    # centroid of just the points being aligned
    centroid1 = centroid(xyz1)
    centroid2 = centroid(xyz2)

    # (1)
    #xyz_protein1 = np.copy(parsed1['xyz']) - centroid1
    xyz_protein2 = np.copy(parsed2) - centroid2

    # (2)
    xyz_protein2 = xyz_protein2 @ R

    # Translate protein 2 to where it aligns with original protein 1
    xyz_protein2 += centroid1
    
    out_pose = mobile_pose.clone()
    for resno, res_coords in enumerate(xyz_protein2):
        for i, ac in enumerate(res_coords):
            if np.isnan(ac[0]):
                break
            out_pose.residue(resno+1).set_xyz(i+1, pyrosetta.rosetta.numeric.xyzVector_double_t(*ac))
            continue
    out_pose.scores["rmsd"] = rmsd
    return out_pose


# def align_pose_coords(mobile_pose, ref_coords, ref_atoms):
#     # xyz1, parsed1 = get_xyz_stack_pose(ref_pose, ref_atoms["atoms1"])
#     assert len(ref_coords) == len(ref_atoms)
#     xyz2, parsed2 = get_xyz_stack_ligand_pose(mobile_pose, ref_atoms)

#     # run Kabsch to get rotation matrix for atoms and rmsd
#     # aligns xyz2 onto xyz1
#     rmsd, _, R = kinematics.np_kabsch(ref_coords, xyz2)
#     print('RMSD between atoms: ',rmsd)

#     # (1) now translate both proteins such that centroid(xyz1/xyz2) is at origin
#     # (2) rorate xyz2 onto xyz1 with R
#     # (3) write pdbs into outdir

#     def centroid(X):
#         # return the mean X,Y,Z down the atoms
#         return np.mean(X, axis=0, keepdims=True)

#     # centroid of just the points being aligned
#     centroid1 = centroid(ref_coords)
#     centroid2 = centroid(xyz2)

#     # (1)
#     #xyz_protein1 = np.copy(parsed1['xyz']) - centroid1
#     xyz_protein2 = np.copy(parsed2) - centroid2

#     # (2)
#     xyz_protein2 = xyz_protein2 @ R

#     # Translate protein 2 to where it aligns with original protein 1
#     xyz_protein2 += centroid1
    
#     out_pose = mobile_pose.clone()
#     for resno, res_coords in enumerate(xyz_protein2):
#         for i, ac in enumerate(res_coords):
#             if np.isnan(ac[0]):
#                 break
#             out_pose.residue(resno+1).set_xyz(i+1, pyrosetta.rosetta.numeric.xyzVector_double_t(*ac))
#             continue
#     return out_pose
