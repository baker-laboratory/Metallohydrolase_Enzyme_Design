from collections import namedtuple
import attr
import logging
import numpy as np
import os
import math
import re

from io import StringIO #to be able to process pdb as StringIO in read_in_stubs_StringIO(s: StringIO)

import sys
#sys.path.append("/home/kipnis/scripts") #to import SimpleXyzMath3
from SimpleXyzMath3 import *

###############################################################################################
##################################( Classes )##################################################
###############################################################################################
@attr.s
class AtomRecord(object):
    record_name = attr.ib()
    serial = attr.ib()
    name = attr.ib()
    altLoc = attr.ib()
    resName = attr.ib()
    chainID = attr.ib()
    resSeq = attr.ib()
    iCode = attr.ib()
    x = attr.ib()
    y = attr.ib()
    z = attr.ib()
    occupancy = attr.ib()
    tempFactor = attr.ib()
    element = attr.ib()
    charge = attr.ib()

    @classmethod #YK wrote this and it should not be a classmethod ???
    def clone(cls, atomrecord):
        record_name = atomrecord.record_name
        serial = atomrecord.serial
        name = atomrecord.name
        altLoc = atomrecord.altLoc
        resName = atomrecord.resName
        chainID = atomrecord.chainID
        resSeq = atomrecord.resSeq
        iCode = atomrecord.iCode
        x = atomrecord.x
        y = atomrecord.y
        z = atomrecord.z
        occupancy = atomrecord.occupancy
        tempFactor = atomrecord.tempFactor
        element = atomrecord.element
        charge = atomrecord.charge
        return cls(
            record_name,
            serial,
            name,
            altLoc,
            resName,
            chainID,
            resSeq,
            iCode,
            x,
            y,
            z,
            occupancy,
            tempFactor,
            element,
            charge,
        )
    
    @classmethod
    def from_str(cls, record):
        record_name = record[:6].strip()
        serial = int(record[6:11].strip())
        name = record[12:16].strip()
        altLoc = record[16].strip()
        resName = record[17:20].strip()
        chainID = record[21].strip()
        resSeq = int(record[22:26].strip())
        iCode = record[26].strip()
        x = float(record[30:38].strip())
        y = float(record[38:46].strip())
        z = float(record[46:54].strip())
        occupancy, tempFactor, element, charge = [None] * 4
        
        try:
            occupancy = float(record[54:60].strip())
            tempFactor = float(record[60:66].strip())
            element = record[76:78].strip()
            charge = record[78:80].strip()
        except (IndexError, ValueError):
            pass
        
        return cls(
            record_name,
            serial,
            name,
            altLoc,
            resName,
            chainID,
            resSeq,
            iCode,
            x,
            y,
            z,
            occupancy,
            tempFactor,
            element,
            charge,
        )
        
    def to_vec(self):
        return Vec(self.x,self.y,self.z)

    def __str__(self):
        return "{record_name:6}{serial:5d} {name:4}{altLoc:1}{resName:3} {chainID:1}{resSeq:>4}{iCode:1}   {x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{tempFactor:6.2f}           {element:2}{charge:2}".format(
            record_name=self.record_name,
            serial=self.serial,
            name=self.name,
            altLoc=self.altLoc,
            resName=self.resName,
            chainID=self.chainID,
            resSeq=self.resSeq,
            iCode=self.iCode,
            x=self.x,
            y=self.y,
            z=self.z,
            occupancy=self.occupancy if self.occupancy is not None else "",
            tempFactor=self.tempFactor if self.tempFactor is not None else "",
            element=self.element if self.element is not None else "",
            charge=self.charge if self.charge is not None else "",
        )


@attr.s
class Residue(object):
    atom_records = attr.ib()
    coords = attr.ib()

    @classmethod
    def from_records(cls, atom_records):
        coords = np.array([[ar.x, ar.y, ar.z] for ar in atom_records])
        return cls(atom_records, coords)

    @property
    def name(self):
        names = set(ar.resName for ar in self.atom_records)
        assert(len(names) == 1)
        return names.pop()
        
    def to_vec(self):
        return [ar.to_vec() for ar in self.atom_records]

    def __str__(self):
        return "\n".join([str(ar) for ar in self.atom_records])
    
############################( Classes to store Matcher header info #####################################
@attr.s
class R666Record(object):
    REMARK = attr.ib()
    REMARK_id = attr.ib()
    MATCH1 = attr.ib()
    TEMPLATE = attr.ib()
    u_ch = attr.ib()
    u_rn = attr.ib()
    u_ri = attr.ib()
    MATCH2 = attr.ib()
    MOTIF = attr.ib()
    d_ch = attr.ib()
    d_rn = attr.ib()
    d_ri = attr.ib()
    cst_block = attr.ib()
    cst_subblock = attr.ib()
    
    @classmethod
    def from_str(cls, record):
        REMARK = record.strip().split()[0]
        REMARK_id = record.strip().split()[1]
        MATCH1 = record.strip().split()[2]
        TEMPLATE = record.strip().split()[3]
        u_ch = record.strip().split()[4]
        u_rn = record.strip().split()[5]
        u_ri = record.strip().split()[6]
        MATCH2 = record.strip().split()[7]
        MOTIF = record.strip().split()[8]
        d_ch = record.strip().split()[9]
        d_rn = record.strip().split()[10]
        d_ri = record.strip().split()[11]
        cst_block = record.strip().split()[12]
        cst_subblock = record.strip().split()[13]

        return cls(
            REMARK,
            REMARK_id,
            MATCH1,
            TEMPLATE,
            u_ch,
            u_rn,
            u_ri,
            MATCH2,
            MOTIF,
            d_ch,
            d_rn,
            d_ri,
            cst_block,
            cst_subblock,
        )

    def __str__(self):
        return "{REMARK}{REMARK_id: >4} {MATCH1} {TEMPLATE} {u_ch} {u_rn} {u_ri: >5} {MATCH2} {MOTIF} {d_ch} {d_rn} {d_ri: >5} {cst_block: >3} {cst_subblock: >3}".format(
           REMARK = self.REMARK,
           REMARK_id = self.REMARK_id,
           MATCH1 = self.MATCH1,
           TEMPLATE = self.TEMPLATE,
           u_ch = self.u_ch,
           u_rn = self.u_rn,
           u_ri = self.u_ri,
           MATCH2 = self.MATCH2,
           MOTIF = self.MOTIF,
           d_ch = self.d_ch,
           d_rn = self.d_rn,
           d_ri = self.d_ri,
           cst_block = self.cst_block,
           cst_subblock = self.cst_subblock,
        )


@attr.s
class MatcherHeader(object):
    r666_records = attr.ib()
    theozyme = attr.ib()

    @classmethod
    def from_records(cls, r666_records):
        theozyme = [((r666.u_ch,r666.u_rn,r666.u_ri),(r666.d_ch,r666.d_rn,r666.d_ri)) for r666 in r666_records]
        return cls(r666_records, theozyme)

    @property
    def ress(self):
        all_ress = []
        for cst in self.theozyme:
            all_ress += cst
        return set(all_ress)

    def __str__(self):
        return "\n".join([str(record) for record in self.r666_records])

############################( Classes to store etable from pdb )#####################################
class POSE_ETABLE:
#     import pandas as pd
    import re
    
    def __init__(self,labels=[],pose=[],etable=[],filters={},metrics={}):
        self.labels = labels
        self.pose = pose
        self.etable = etable
        self.filters = filters
        self.metrics = metrics

    def read_pdb_file(self,pdb_path):
        etable_start = False
        etable_end = False
        with open(pdb_path, "r") as inp:
            labels = []
            weights = []
            pose = []
            etable = []
            filters = {}
            metrics = {}
            for line in inp:
                if line.startswith("#BEGIN_POSE_ENERGIES_TABLE"):
                    etable_start = True
                    continue
                elif line.startswith("#END_POSE_ENERGIES_TABLE") and etable_start and not(etable_end):
                    etable_end = True
                    continue
                elif line.startswith("label") and etable_start and not(etable_end):
                    labels = line.strip().split()
                    continue
                elif line.startswith("weights") and etable_start and not(etable_end):
                    weights = line.strip().split()
                    continue
                elif line.startswith("pose") and etable_start and not(etable_end):
                    pose = line.strip().split()
                    continue
                elif etable_start and not(etable_end):
                    etable.append(line.strip().split())
                elif line.startswith("filt_") or line.startswith("ffilt_"):#ffilt is temporary misspelled one filter in set of pdbs
                    try:
                        [filt,value] = line.strip().split()
                        filters[filt] = value
                    except:
                        pass
                    else:
                        pass
                elif line.startswith("metrics_"):
                    try:
                        [metric,value] = line.strip().split()
                        metrics[metric] = value
                    except:
                        pass
                    else:
                        pass
                else:
                    continue
    
        return POSE_ETABLE(labels,pose,etable,filters,metrics)
        
    def __str__(self):
        return f"{self.labels}\n{self.pose}\n{self.etable}\n{self.filters}\n{self.metrics}"  
        

###############################################################################################
##############################( Pdb parsing )##################################################
###############################################################################################
def read_in_stubs_file(fname: str):
    '''
    takes:
            path to pdb,reads in pdb file, extracts ATOM and HETATM records,
            combines individual AtomRecords into Residue objects;
            separates multimodel pdb into individual models.
    returns:
            list of models 
    '''
    models = []
    model = []
    detected_multimodel = 0 #YK: triggered by the presence of MODEL tag 
    curr_res = []
    resSeq = 0

    with open(fname, "r") as f:
        for l in f:
            if not l:
                continue

            if l.startswith("MODEL"):
                # clear previous model data
                model = []
                curr_res = []
                resSeq = 0
                detected_multimodel = 1
                continue

            if l.startswith("ENDMDL"):
                # finalize current model
                model.append(Residue.from_records(curr_res))
                models.append(model)
                continue
            
            if l.startswith("ATOM") or l.startswith("HETATM"): #YK: to skip all non-atom lines
                record = AtomRecord.from_str(l)
                if record.resSeq == resSeq:
                    curr_res.append(record)
                else:
                    if curr_res:
                        model.append(Residue.from_records(curr_res))
                    curr_res = [record]
                    resSeq = record.resSeq
                    
        if not detected_multimodel: #YK: write the only model found in the fname, and make sure last curr_res gets added to the model
            model.append(Residue.from_records(curr_res))
            models.append(model)
            
    return models

def read_in_stubs_StringIO(s: StringIO):
    '''
    takes:
            StringIO pdb file, extracts ATOM and HETATM records,
            combines individual AtomRecords into Residue objects;
            separates multimodel pdb into individual models.
    returns:
            list of models 
    '''
    models = []
    model = []
    detected_multimodel = 0 #YK: triggered by the presence of MODEL tag 
    curr_res = []
    resSeq = 0

    assert type(s) == StringIO, f"Wrong input data type, expecting StringIO, got {type(s)}"
    
    s.seek(0)
    line_list = s.read().strip().split("\n")

    for l in line_list:
        if not l:
            continue

        if l.startswith("MODEL"):
            # clear previous model data
            model = []
            curr_res = []
            resSeq = 0
            detected_multimodel = 1
            continue

        if l.startswith("ENDMDL"):
            # finalize current model
            model.append(Residue.from_records(curr_res))
            models.append(model)
            continue

        if l.startswith("ATOM") or l.startswith("HETATM"): #YK: to skip all non-atom lines
            record = AtomRecord.from_str(l)
            if record.resSeq == resSeq:
                curr_res.append(record)
            else:
                if curr_res:
                    model.append(Residue.from_records(curr_res))
                curr_res = [record]
                resSeq = record.resSeq

    if not detected_multimodel: #YK: write the only model found in the fname, and make sure last curr_res gets added to the model
        model.append(Residue.from_records(curr_res))
        models.append(model)

    return models

def get_r666(fname):
    "Creates MatcherHeader object from pdb. Returns None if no REMARK 666 lines found in pdb"
    
    header = None
    r666s = []
    
    with open(fname,"r") as inp:
        for line in inp:
            if line.startswith("REMARK 666 MATCH"):
                r666s.append(R666Record.from_str(line))
                
    if len(r666s) > 0:
        header = MatcherHeader.from_records(r666s)
    
    return header

def get_theozyme_resids(pdb_path):
    '''
    function to directly parse REMARK 666 lines to extract resi and 
    simple heuristic to guess ligand resi as resi after last protein resi
    Single chain pdbs only! Better to use "get_r666()" to process matcher headers
    May be problematic if there are modified residues using HETATM instead of ATOM
    record (like LYS_D)
    '''
    r666=[]
    last_protein_residue=0
    with open(pdb_path) as fp:
        for line in fp:
            if "REMARK 666" in line:
                r666.append(line.strip().split()[11])
            if line.startswith("ATOM"):
                last_protein_residue=int(line[22:27].strip())

    lig_resid=str(last_protein_residue+1)
    r666.append(lig_resid)
    
    return r666

###############################################################################################
#################################( Geometry )##################################################
###############################################################################################

#Kabsch algorithm implementation to aligh models from
#https://github.com/charnley/rmsd/blob/master/rmsd/calculate_rmsd.py
#There is no computation to return translation vectors, but this can be obtained as
# kabsch_t = centroid_static - centroid_movable.dot(kabsch_U)

def RMSD(V, W):
    """
    Calculate Root-mean-square deviation from two sets of vectors V and W.
    Parameters
    ----------
    V : array
        (N,D) matrix, where N is points and D is dimension.
    W : array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    rmsd : float
        Root-mean-square deviation between the two vectors
    """
    D = len(V[0])
    N = len(V)
    result = 0.0
    for v, w in zip(V, W):
        result += sum([(v[i] - w[i])**2.0 for i in range(D)])
    return np.sqrt(result/N)

def centroid(X):
    """
    Centroid is the mean position of all the points in all of the coordinate
    directions, from a vectorset X.
    https://en.wikipedia.org/wiki/Centroid
    C = sum(X)/len(X)
    Parameters
    ----------
    X : array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    C : float
        centroid
    """
    C = X.mean(axis=0)
    return C

def kabsch_rotate(P, Q):
    """
    Rotate matrix P unto matrix Q using Kabsch algorithm.
    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    P : array
        (N,D) matrix, where N is points and D is dimension,
        rotated
    """
    U = kabsch(P, Q)

    # Rotate P
    P = np.dot(P, U)
    return P


def kabsch(P, Q):
    """
    Using the Kabsch algorithm with two sets of paired point P and Q, centered
    around the centroid. Each vector set is represented as an NxD
    matrix, where D is the the dimension of the space.
    The algorithm works in three steps:
    - a centroid translation of P and Q (assumed done before this function
      call)
    - the computation of a covariance matrix C
    - computation of the optimal rotation matrix U
    For more info see http://en.wikipedia.org/wiki/Kabsch_algorithm
    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    U : matrix
        Rotation matrix (D,D)
    """

    # Computation of the covariance matrix
    C = np.dot(np.transpose(P), Q)

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    # Create Rotation matrix U
    U = np.dot(V, W)

    return U

def get_model_coords(model):
    '''
        Takes:
        model as list of Residue objects.
        
        Returns:
        numpy array of atomic coordinates useful to compute 
        Kabsch transformation matrix, centroid coords etc.
    '''
    all_coords = None
    for res in model:
        if all_coords is None:
            all_coords = res.coords
        else:
            all_coords = np.append(all_coords,res.coords, axis = 0)
    return all_coords

def TR_model(model, rotate_by = np.identity((3)), translate_by = np.zeros(3)):
    '''
        Takes a model as a list of Residue objects,
        applies translation as defined by translate_by vector and
        rotation as defined by rotate_by matrix to individual AtomRrecords.
        Returns copy of the input model after transformation
    '''
    transformed = []
    for res in model:
        cur_res = []
        for ar in res.atom_records:
            #possible to modify xyz coords in AtomRecord directly
            #but not sure if might need original models for something, so made clone() class method
            #potentially may move this function into AtomRecord class as AtomRecord.coord_update() method
            TR_ar = AtomRecord.clone(ar)
            xyz = np.array([ar.x,ar.y,ar.z]) 
            
            TR_ar.x,TR_ar.y,TR_ar.z = xyz.dot(rotate_by) + translate_by
            
            cur_res.append(TR_ar)
    
        transformed.append(Residue.from_records(cur_res))
    return transformed

def TR_coords(coords, rotate_by = np.identity((3)), translate_by = np.zeros(3)):
    '''
        Takes:
            a model as a numpy.array of coords,
            applies translation as defined by translate_by vector and
            rotation as defined by rotate_by matrix to coords.
        Returns:
            copy of the input model after transformation
    '''
    transformed = np.dot(coords,rotate_by) - translate_by
    return transformed

def normalized(vec): # takes numpy array/matrix
    vec_norm=np.linalg.norm(vec)
    return np.divide(vec,vec_norm)

def distance_np(vec1,vec2):
    return np.linalg.norm(vec1-vec2)
    
def angle_np(vec1,vec2,vec3=None): # takes numpy array/matrix
    if not(vec3 is None): #looks like input is coords of 3 points
        vec1 = normalized(vec1 - vec2)
        vec2 = normalized(vec2 - vec3)
    else:
        vec1 = normalized(vec1)
        vec2 = normalized(vec2)
    
    if type(vec2) is np.matrixlib.defmatrix.matrix:
        vec2=vec2.getT()
    dot=np.dot(vec1,vec2)
    
    return np.degrees(np.arccos(vec1.dot(vec2)))


def compute_normal(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    '''
        A helper function for computing the normal to a triangular facet
    '''
    nx = (y2-y1)*(z3-z2) - (z2-z1)*(y3-y2)
    ny = (z2-z1)*(x3-x2) - (x2-x1)*(z3-z2)
    nz = (x2-x1)*(y3-y2) - (y2-y1)*(x3-x2)
    return (nx,ny,nz)
 
def dihedral_np(a1_xyz,a2_xyz,a3_xyz,a4_xyz):
    '''
        code for dihedral from
        http://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
        function dihedral2() 
    '''
    p = np.array([a1_xyz,a2_xyz,a3_xyz,a4_xyz])
    b = p[:-1] - p[1:] #convert 4 points into 3 vectors
    b[0] *= -1
    v = np.array( [ v - (v.dot(b[1])/b[1].dot(b[1])) * b[1] for v in [b[0], b[2]] ] ) #vec - projection of vec
    # Normalize vectors
    v /= np.sqrt(np.einsum('...i,...i', v, v)).reshape(-1,1)
    b1 = b[1] / np.linalg.norm(b[1])
    x = np.dot(v[0], v[1])
    m = np.cross(v[0], b1)
    y = np.dot(m, v[1])
    return np.degrees(np.arctan2( y, x ))
    
def same_direction(u,v): #cosine (0,1]
    assert len(u.shape) == 1 and u.shape[0] == 3, f"Not implemented for arrays of shape {u.shape}"
    assert len(v.shape) == 1 and v.shape[0] == 3, f"Not implemented for arrays of shape {v.shape}"
    return u.dot(v) > 0
    
def projection_matrix(v):
    assert len(v.shape) == 1 and v.shape[0] == 3, f"Not implemented for arrays of shape {v.shape}"
    x, y, z = v
    m = np.array([[x*x, x*y, x*z], [y*x, y*y, y*z], [z*x, z*y, z*z]])
    return m/v.dot(v)

def proj(u, v): #projection of v onto u
    '''
    v=np.array([1,1,1])
    u=np.array([1,0,0])
    proj(u,v)
    Returns: array([1., 0., 0.])
    '''
    assert len(u.shape) == 1 and u.shape[0] == 3, f"Not implemented for arrays of shape {u.shape}"
    assert len(v.shape) == 1 and v.shape[0] == 3, f"Not implemented for arrays of shape {v.shape}"
    return projection_matrix(u).dot(v)

def proj_perp(u,v):
    '''
    v=np.array([1,1,1])
    u=np.array([1,0,0])
    proj_perp(u,v)
    Returns: array([0., 1., 1.])
    '''
    assert len(u.shape) == 1 and u.shape[0] == 3, f"Not implemented for arrays of shape {u.shape}"
    assert len(v.shape) == 1 and v.shape[0] == 3, f"Not implemented for arrays of shape {v.shape}"
    return v - proj(u,v)    


def embed_node_3d(avec, bvec, cvec, dst, ang, tor):
    '''
    For 4 connected atoms A->B->C->(D)
    given xyz coords for A,B and C (as np.arrays) and 
    CD distance, BCD angle (in degrees) and ABCD torsion (in degrees)
    compute cartesian embedding (xyz coords) for D.
    
    Using NeRF method from
    https://pubmed.ncbi.nlm.nih.gov/15898109/
    
    ========Embedding inputs=======
    [ 9.88  5.2  -0.31]
    [ 6.82  3.13 -1.23]
    [3.69 2.94 0.9 ]
    length:       3.80
    angle:       66.97
    dihedral:  -163.18
    ========Embedded dvec===========
    [ 1.46  0.   -0.  ]
    length:       3.80
    angle:       66.97
    dihedral:  -163.18
    ========Ground truth Dvec=======
    [1.46 0.   0.  ]
    length:       3.80
    angle:       66.97
    dihedral:  -163.18
    
    '''
    
    ang = ang * np.pi / 180.0
    tor = tor * np.pi / 180.0
    
    # Computes coordinates of D in local frame 
    # where C (0,0,0), B is along the axis, A is in CBA plane.
    # therefore d2 is simply polar to cartesian conversion
    d2 = np.array([ dst*np.cos(ang), dst*np.sin(ang)*np.cos(tor), dst*np.sin(ang)*np.sin(tor) ])

    #Compute M_hat which transformes local frame into global
    # "hat" == normalized
    bc_hat = normalized(cvec - bvec)
    ab_hat = normalized(bvec - avec)
    n_hat = normalized(np.cross(ab_hat, bc_hat)) #axis perpendicular to CBA plane

    M_hat = np.array([bc_hat, np.cross(n_hat, bc_hat), n_hat])

    dvec = d2.dot(M_hat) + cvec
    
    return dvec

def get_lig_orientation_in_pdb(pdb_path,
                               re_P1="ATOM .* CA .* 10 ",
                               re_P2="ATOM .* CA .* 19 ",
                               re_L1="HETATM .* O3 ",
                               re_L2="HETATM .* N1 "):
    
    '''takes path to pdb and 4 regex's to extract reference atoms from pdb
       re_P1, re_P2 are from protein
       re_L1, re_L2 are from ligand (dislike lowercase l, annoyingly similar to 1)
       _______________________________
       x_coord   = line[30:38].strip()
       y_coord   = line[38:46].strip()
       z_coord   = line[46:54].strip()
       _______________________________
       returns angle between vectors P1_P2 and L1_L2
    '''
    def normalized(vec): # takes numpy array/matrix
        vec_norm=np.linalg.norm(vec)
        return np.divide(vec,vec_norm)

    def angle(vec1,vec2): # takes numpy array
        dot=np.dot(vec1,vec2)
        return math.acos(dot)*180.0/3.1412927
    
    
    reP1=re.compile(re_P1)
    reP2=re.compile(re_P2)
    reL1=re.compile(re_L1)
    reL2=re.compile(re_L2)
    
    atm_coords = {"P1":[],"P2":[],"L1":[],"L2":[]}
    
    with open(pdb_path,"r") as pdb:
        for line in pdb:
            if reL1.search(line):
                atm_name = "L1"
                coords = (line[30:38].strip(),line[38:46].strip(),line[46:54].strip())
                atm_coords[atm_name] = np.array(coords, dtype=float)
            if reL2.search(line):
                atm_name="L2"
                coords = (line[30:38].strip(),line[38:46].strip(),line[46:54].strip())
                atm_coords[atm_name] = np.array(coords, dtype=float)
            if reP1.search(line):
                atm_name="P1"
                coords = (line[30:38].strip(),line[38:46].strip(),line[46:54].strip())
                atm_coords[atm_name] = np.array(coords, dtype=float)
            if reP2.search(line):
                atm_name="P2"
                coords = (line[30:38].strip(),line[38:46].strip(),line[46:54].strip())
                atm_coords[atm_name] = np.array(coords, dtype=float)

    L_vec=atm_coords["L2"] - atm_coords["L1"]
    P_vec=atm_coords["P2"] - atm_coords["P1"]

    L_vec_n=normalized(L_vec)
    P_vec_n=normalized(P_vec)

    return angle(L_vec_n,P_vec_n)


###############################################################################################
###############################( Dealing with fragments for alphabets )########################
###############################################################################################
        
def trim_model_by_resn(model, ress_keep=[], atms_keep=[]):
    '''
        Takes:
        model as a list of Residue objects,
        list of residue types [res_type1, .. , res_typeN] to keep 
        list of atom names to keep in each residue [at_list_1, .. at_list_N]
        
        If list of residues is empty [], then all residues will be returned.
        If list of atoms for a residue is empty [], then all atoms for a residue will be returned.
    '''
    assert len(ress_keep) == len(atms_keep), "Lenght of ress_keep and atms_keep needs to be equal"
    
    if len(ress_keep) == 0:
        keepers = [ res for res in model]
    else:
        keepers = []
        for res in model:

            if not(res.name in ress_keep):
                continue

            if len(atms_keep[ress_keep.index(res.name)]) == 0: #empty list of atoms to keep, keep all atoms, kind of silly
                cur_res = [ atom_record for atom_record in res.atom_records ]
            else:
                cur_res = []
                for atom_record in res.atom_records:
                    if atom_record.name in atms_keep[ress_keep.index(res.name)]:
                        cur_res.append(atom_record)
            
            if len(cur_res) > 0:    
                keepers.append(Residue.from_records(cur_res))
        
    return keepers

def trim_model_by_resi(model,ress_keep=[],atms_keep=[]):#0-based resi indexing
    '''
        Takes: 
        model as a list of Residue objects,
        list of residue indices [resi_1, resi_2, ...] to keep
        list of atom names to keep in each residue [[at1,...],[at1, ...], ...]
        
        Can extract: 
        1) specified atoms from specified residues
        2) specified atoms from each residue
        3) all atoms from specified residues
        4) all atoms from all residues
        Doesn't test if residue actually has the required atom or if residue index is smaller than model length
    '''
    #trim_model_by_resi(model,[],[]) extracts all residues and all atoms
    if len(ress_keep) == len(atms_keep) == 0:
        keepers = [ res for res in model]
    
    #trim_model_by_resi(model,[0,2,5],[]) extracts all atoms from residues 1,3,6
    elif (len(ress_keep) != 0 and len(atms_keep) == 0):
        keepers = [ model[i] for i,res in enumerate(model) if i in ress_keep]
    
    #trim_model_by_resi(model,[],["CA"]) extracts CA's from all residues in model
    elif (len(ress_keep) == 0 and len(atms_keep) != 0):
        keepers = []
        for res in model:
            cur_res = []
            for atom_record in res.atom_records:
                if atom_record.name in atms_keep:
                    cur_res.append(atom_record)
            if len(cur_res) > 0:    
                keepers.append(Residue.from_records(cur_res))
    
    ##trim_model_by_resi(model,[0,2,5],[["CA","CB"],["CA"],["CA"]]) extracts CA and CB from residue 1 and CA's from residues 3,6 and  in model
    elif (len(ress_keep) != 0 and len(atms_keep) != 0 and len(ress_keep) == len(atms_keep)):
        keepers = []
        for i,res in enumerate(model):
            if i in ress_keep:
                cur_res = []
                for atom_record in res.atom_records:
                    if atom_record.name in atms_keep[ress_keep.index(i)]:
                        cur_res.append(atom_record)
                if len(cur_res) > 0:
                    keepers.append(Residue.from_records(cur_res))
    else:
        print("Cannot figure out what residues from {ress_keep} and what atoms from {atms_keep} to keep. Stopping ...".format(ress_keep=ress_keep, atms_keep=atms_keep))
        sys.exit()
        
    return keepers

def frag_is_good(frag,dist_cutoff=5.0):
    '''
        Takes:
        fragment <list of Residue objects with single AtomRecord in each>
        dist_cutoff <acceptable length of pseudo-bond between consequtive atoms in fragment>
        
        Returns:
        True if all pseudo bonds in fragment are acceptable
    '''
    good = True
    
    #atms_xyz = np.array([[ar.x,ar.y,ar.z] for res in frag for ar in res.atom_records])
    atms_xyz = get_model_coords(frag)
    bonds =atms_xyz[:-1] - atms_xyz[1:]
    for bond in bonds:
        if np.linalg.norm(bond) >= dist_cutoff:
            good = False
            break
    
    return good

def get_frags_from_bbtrace(trace,frag_length=5,pseudo_bond_cutoff=5.0,overlapping=True):
    '''
        Takes:
        trace = list of Residue objects with each residue having one AtomRecord for CA
        fragment size
        generate overlapping or consecutive
        
        Returns:
        frags = list of lists of Residue objects all with frag_length size
    '''
    trace_length = len(trace)
    
    frags = []
    
    if overlapping:
        for i in range(0,trace_length - frag_length + 1,1):
            if frag_is_good(trace[i:i+frag_length],dist_cutoff=pseudo_bond_cutoff):
                frags.append(trace[i:i+frag_length])
    else:
        for i in range(0,trace_length - frag_length + 1,frag_length):
            if frag_is_good(trace[i:i+frag_length],dist_cutoff=pseudo_bond_cutoff):
                frags.append(trace[i:i+frag_length])
    
    return frags

def align_models(target, mobiles, target_ress_align=[], mobile_ress_align=[], target_atms_align=[], mobile_atms_align=[]):
    '''
        Takes:
        target model
        list of mobile models
        list of res ids to use for alignment 0-indexed
        list of atoms to use for alignment
        f.e. each model is 5 residues with "CA" and "CB" atoms
        calling align_models(models,target_ress_align=[0,1,2], mobile_ress_align=[0,1,2], atms_align = [["CA"],["CA"],["CA"]]) should 
        return models aligned on CA's of residues 1,2 and 3
        
        Returns:
        list of models each translated to COM at (0,0,0) and rotated by Kabsch matrix
        aligned to first model as a reference model
    '''
    aligned_models = []
    rmsds = []
    
    # set the reference model
    assert len(mobile_ress_align) == len(target_ress_align)
    target_trimmed = trim_model_by_resi(target, target_ress_align, target_atms_align)

    # get the reference centroid
    target_centroid = centroid(get_model_coords(target_trimmed))

    # translate the atoms by the centroid to origin
    target_align = TR_model(target_trimmed, translate_by=target_centroid*-1)
    target_coords = np.array([get_model_coords(trim_model_by_resi(target_align, atms_keep=atm))[0] for atm in target_atms_align[0]])
    
    # also translate the entire model by the centroid to the origin
    entire_target_align = TR_model(target, translate_by=target_centroid*-1)
    entire_target_align_coords = get_model_coords(trim_model_by_resi(entire_target_align, target_ress_align, target_atms_align))

    # add to the list
    aligned_models.append(entire_target_align)
    rmsds.append(0.0) # aligning model to itself
    
    for mobile in mobiles:
        mobile_trimmed = trim_model_by_resi(mobile, mobile_ress_align, mobile_atms_align)
        mobile_coords = np.array([get_model_coords(trim_model_by_resi(mobile_trimmed, atms_keep=atm))[0] for atm in mobile_atms_align[0]])
        mobile_centroid = centroid(mobile_coords)

        # translate the model by the centroid to origin 
        mobile_align = TR_model(mobile_trimmed, translate_by=mobile_centroid*-1)
        mobile_coords = np.array([get_model_coords(trim_model_by_resi(mobile_align, atms_keep=atm))[0] for atm in mobile_atms_align[0]])

        # get the rotation matrix between target and mobile atoms
        rotmat = kabsch(mobile_coords, target_coords)
        
        # apply rotation matrix and centroid translation to the whole model
        # Anna's note: by some math that I don't understand, this only works if you 
        # apply the translation and rotation in seperate steps...
        mobile_translate = TR_model(mobile, translate_by=mobile_centroid*-1) # translate
        aligned_model = TR_model(mobile_translate, rotate_by=rotmat) # rotate
        aligned_models.append(aligned_model)
        
        # rmsd calculation
        aligned_atoms = TR_model(mobile_align, rotate_by=rotmat)
        aligned_coords = np.array([get_model_coords(trim_model_by_resi(aligned_atoms, atms_keep=atm))[0] for atm in mobile_atms_align[0]])
        aligned_rmsd = round(RMSD(target_coords, aligned_coords),5)
        rmsds.append(aligned_rmsd)

    return aligned_models, rmsds

def align_coords(set1,set2,idxs1=[0,1,2],idxs2=[0,1,2]):
    '''
        Takes:
            two sets of atom coords as numpy.arrays with lists of equal length indicating which atoms to align
        Returns:
            set2 translated/rotated in a way that ovelays alignmnet group of set1 and set2
    '''

    coord1=set1[idxs1,:]
    coord2=set2[idxs2,:]
    
    centroid1 = centroid(coord1)
    centroid2 = centroid(coord2)
    
    overlap_vertices_1_COM = TR_coords(coord1,translate_by=centroid1)
    overlap_vertices_2_COM = TR_coords(coord2,translate_by=centroid2)
    
    rotmat = kabsch(overlap_vertices_2_COM,overlap_vertices_1_COM)
    
    set2 = TR_coords(set2,translate_by=centroid2)
    set2 = TR_coords(set2,rotate_by=rotmat)
    set2 = TR_coords(set2,translate_by=(centroid1 * (-1.0)))
    
    return set2

def write_models_to_file(models,path_fname):
    '''
        Takes:
        list of models as list of Residue objects
        path and file name (doesn't have any heuristic to generate this)
        
        Returns:
        True
    '''
    
    with open(path_fname,"w") as out:
        for model in models:
            out.write("MODEL\n")
            out.write("\n".join([str(res) for res in model]))
            out.write("\nENDMDL")  
    return True

def get_model_string(models):
    '''
        Takes:
        list of models as list of Residue objects
        
        Returns:
        StringIO object of pdb
    '''
    s = ''
    for model in models:
        s += "MODEL\n"
        s += "\n".join([str(res) for res in model])
        s += "\nENDMDL"  

    return s

###############################################################################################
###############################( Wrappers for fragment clustering )############################
###############################################################################################
def cluster_models(models,epsilon=0.25):
    '''
        Takes:
        list of models (each model is a list of Residue objects)
        To cluster by model shape, all models need to be aligned (COM at (0,0,0) Kabsch rotation applied)
        for clustering to be meaningful
        
        Returns:
        clusters object
    '''
    cluster_by_atoms = []
    for model in models:
        atms_coords = get_model_coords(model)
        cluster_by_atoms.append(atms_coords)

    cluster_by_atoms = np.array(cluster_by_atoms)
    cluster_by_atoms = cluster_by_atoms.reshape(
        cluster_by_atoms.shape[:-2] + (cluster_by_atoms.shape[-1] * cluster_by_atoms.shape[-2],)
    )

    from sklearn.cluster import DBSCAN
    clusters = DBSCAN(eps=epsilon, metric="euclidean", n_jobs=-1).fit(cluster_by_atoms)
    
    return clusters
    
def compute_cluster_centroid(frags):
    '''
        Takes:
        list of fragments, each fragment is a list of Residue objects with single AtomRecord
        
        Returns:
        single fragment as a list of Residues with atoms coordinates corresponding to
        averages of atom coordinates of input frags
    '''
    all_coords = np.array([get_model_coords(frag) for frag in frags])
    centroid_coords = all_coords.mean(axis=0)
    
    ###############( Making frag with centroid coords)###################
    cluster_centroid_frag = []
    for i,res in enumerate(frags[0]): #using first frag in cluster as guide for iteration, centroid will have resn, resi etc. from this fragment
        cur_res = []
        for j,ar in enumerate(res.atom_records):
            centroid_ar = AtomRecord.clone(ar)
            centroid_ar.x,centroid_ar.y,centroid_ar.z = centroid_coords[i]
            cur_res.append(centroid_ar)
    
        cluster_centroid_frag.append(Residue.from_records(cur_res))
    ######################################################################
    
    return cluster_centroid_frag
    
def cluster_by_angles(angles,epsilon=0.25):
    '''
        Takes:
        list of tuples, representing coordinates in phi1_phi2_theta space
        
        Returns:
        clusters object
    '''
    angles = np.array(angles)
    
    from sklearn.cluster import DBSCAN
    clusters = DBSCAN(eps=epsilon, metric="euclidean", n_jobs=-1).fit(angles)
    
    return clusters
    
def process_clusters(clustering,n_clusters=20,keep_singletons=True):
    '''
        Takes:
        cluster object
        number of most populous clusters to keep
        return or not cluster labeled "-1", representing objects not belonging to any other cluster
        
        Returns:
        list of IDs for models in top_n_clusters
    '''
    from itertools import groupby

    # identify top N clusters
    top_n_clusters = sorted(
        [(key, len(list(group))) for key, group in groupby(sorted(clustering.labels_))],
        key=lambda x: x[-1],
        reverse=True,
    )[:n_clusters]

    # "noisy" samples are given the label -1, so if -1 is included in the top N,
    # we should remove it

    if not keep_singletons:
        top_n_clusters = [cluster for cluster in top_n_clusters if cluster[0] >= 0]

    # collect groups of model IDs for the top N clusters
    # ensure they are consistent with the previous step
    models_IDs = []
    for clusID, n_occ in top_n_clusters:
        model_indices_for_cluster = [
            i for i, elem in enumerate(clustering.labels_) if elem == clusID
        ]
        assert len(model_indices_for_cluster) == n_occ
        models_IDs.append(model_indices_for_cluster)
        
    return models_IDs
    

###############################################################################################
####################( Get fragments aligned for backbone assembly )############################
###############################################################################################

def pairwise_align(let1,let2): # the only difference with compute_endchain_placement() is this uses all atoms in the trace instead of 3 last 3 first
    '''
    Takes:
        two fragments of equal lenghth, aligns them let2 will be moved, let1 remain stationary
    Returns:
        letter2 translated/rotated and RMSD
    '''
    overlap_vertices_1=trim_model_by_resi(let1,ress_keep=[],atms_keep=[]) #the only difference with compute_endchain_placement()
    overlap_vertices_2=trim_model_by_resi(let2,ress_keep=[],atms_keep=[]) #the only difference with compute_endchain_placement()
    
    coord1=get_model_coords(overlap_vertices_1)
    coord2=get_model_coords(overlap_vertices_2)
    
    centroid1 = centroid(coord1)
    centroid2 = centroid(coord2)
    
    overlap_vertices_1_COM = TR_model(overlap_vertices_1,translate_by=centroid1)
    overlap_vertices_2_COM = TR_model(overlap_vertices_2,translate_by=centroid2)
    
    overlap_vertices_1_COM_coords = get_model_coords(overlap_vertices_1_COM)
    overlap_vertices_2_COM_coords = get_model_coords(overlap_vertices_2_COM)
    
    rotmat = kabsch(overlap_vertices_2_COM_coords,overlap_vertices_1_COM_coords)
    
    #compute rmsd of the overlap
    overlap_vertices_2_COM_R = TR_model(overlap_vertices_2_COM,rotate_by=rotmat)
    overlap_vertices_2_COM_R_coords = get_model_coords(overlap_vertices_2_COM_R)
    
    r = rmsd(overlap_vertices_1_COM_coords,overlap_vertices_2_COM_R_coords)
    
    
    let2 = TR_model(let2,translate_by=centroid2)
    let2 = TR_model(let2,rotate_by=rotmat)
    let2 = TR_model(let2,translate_by=(centroid1 * (-1.0)))
    
    return (let2,r)

def compute_endchain_placement(let1,let2):
    '''
    Takes:
        two 4-res fragments, aligns on last three res of first and first three res of second
        let2 will be moved, let1 remain stationary
    Returns:
        letter2 translated/rotated in a way that ovelays alignmnet group of let1 and let2
    '''
    overlap_vertices_1=trim_model_by_resi(let1,ress_keep=[1,2,3],atms_keep=[])
    overlap_vertices_2=trim_model_by_resi(let2,ress_keep=[0,1,2],atms_keep=[])
    
    coord1=get_model_coords(overlap_vertices_1)
    coord2=get_model_coords(overlap_vertices_2)
    
    centroid1 = centroid(coord1)
    centroid2 = centroid(coord2)
    
    overlap_vertices_1_COM = TR_model(overlap_vertices_1,translate_by=centroid1)
    overlap_vertices_2_COM = TR_model(overlap_vertices_2,translate_by=centroid2)
    
    overlap_vertices_1_COM_coords = get_model_coords(overlap_vertices_1_COM)
    overlap_vertices_2_COM_coords = get_model_coords(overlap_vertices_2_COM)
    
    rotmat = kabsch(overlap_vertices_2_COM_coords,overlap_vertices_1_COM_coords)
    
    #compute rmsd of the overlap
    overlap_vertices_2_COM_R = TR_model(overlap_vertices_2_COM,rotate_by=rotmat)
    overlap_vertices_2_COM_R_coords = get_model_coords(overlap_vertices_2_COM_R)
    
    r = rmsd(overlap_vertices_1_COM_coords,overlap_vertices_2_COM_R_coords)
    
    
    let2 = TR_model(let2,translate_by=centroid2)
    let2 = TR_model(let2,rotate_by=rotmat)
    let2 = TR_model(let2,translate_by=(centroid1 * (-1.0)))
    
    return (let2,r)

def clash_count(theozyme,trace_structure,clash_cutoff = 2.2):
    '''
    Takes:
        coordinates of atoms forming trace as np.array() (need to keep in mind fuzziness of atoms positions)
    Returns:
        clash_score as total number of clashing atoms 
    '''
    all_atoms = np.append(theozyme,trace_structure,axis=0)
    
    clash_score = 0
    for i,atom1 in enumerate(all_atoms):
        for j,atom2 in enumerate(all_atoms):
            if (i < len(theozyme)) and (j < len(theozyme)):
                continue
            elif i < j:
                if np.linalg.norm(atom1 - atom2) <= clash_cutoff:
                    clash_score += 1
            else:
                continue
    return clash_score
    
    
###############################################################################################
############################( Computing CA grid properties )###################################
###############################################################################################    
     
def rg(coords,atoms=None,masses=None):
    '''
       adopted from:
       https://github.com/sarisabban/Rg/blob/master/Rg.py
       
       takes:
             coords of atoms as NumPy array, list of atom types, list of masses
       returns:
             radius of gyration for these atoms
       rg(atom_coords)
       rg(atom_coords,atoms="C")
       rg(atom_coords,atoms=["N","C","C","O"])
       rg(atom_coords,masses=[14,12,12,16])
    '''
    import math
    
    atm_types={
    'C':12.010,
    'O':15.999,
    'N':14.006,
    'S':32.065,
    'H':1.0,
    }
    
    if atoms is None and masses is None:
        masses = np.array([atm_types["C"],]*len(coords),dtype=float)
    elif type(atoms) is str and masses is None:
        try:
            masses = np.array([atm_types[atoms],]*len(coords),dtype=float)
        except:
            print("Atom type {} not supported".format(atoms))
    elif type(atoms) is list and masses is None:
        try:
            masses = np.array([atm_types[atom] for atom in atoms],dtype=float)
        except:
            print("Failed to obtain mass for some atoms in {}".print(atoms))
    elif atoms is None and type(masses) is list:
        masses = np.array(masses,dtype=float)
    else:
        print("Handling this mode of Rg computation is not yet implemented")
        sys.exit()
    
    assert len(coords) == len(masses), "Mismatch in number of atoms and their masses"
    
    xm = [(m*i, m*j, m*k) for (i, j, k), m in zip(coords, masses)]
    rr = sum(mi*i + mj*j + mk*k for (i, j, k), (mi, mj, mk) in zip(coords, xm))
    tmass = sum(masses)
    mm = sum((sum(i) / tmass)**2 for i in zip(*xm))
    rg = math.sqrt(rr / tmass-mm)
    
    return(round(rg, 3))

def find_clashes(mdl, clash=2.4, substrate_idx=[-1]):
    """
    Find residues that are within clash distance of the substrate.
    Substrate assumed to be the last residue in the sequence.
    
    Input:
    mdl - SimplePdbLib model
    clash - max distance considered a clash
    substrate_idx - negative list of substrate indices, default [-1]

    Returns:
    ca_clashes - the residues with backbone clashes to the substrate
    sc_clashes - the residues with sidechain clashes to the substrate

    *returns residues indices in pdb numbering!
    """

    # get substrate coordinates
    substrate = []
    for i in substrate_idx:
        substrate.append(mdl[i])
    substrate_coords = get_model_coords(substrate)

    # remove substrate from aligned design
    substrate_pos_idx = [len(mdl) + i for i in substrate_idx] # convert to positive numbers
    keep_resi = [i for i in range(len(mdl)) if i not in substrate_pos_idx]
    mdl = trim_model_by_resi(mdl, ress_keep=keep_resi)
    
    # calculate pairwise distances bw substrate & all other atoms
    pdb_coords = get_model_coords(trim_model_by_resi(mdl))
    pairwise_distances = np.array([np.linalg.norm(substrate_coords - p, axis=1) for p in pdb_coords])

    # for each af2 atom get distance to closest substrate atom
    min_distances = np.amin(pairwise_distances, axis=1)

    # get clashing atoms
    clash_atm = np.where(min_distances <= clash)[0]
    
    # split clashing atoms into lists of clashing residues by backbone and sidechain
    ca_clashes = []
    sc_clashes = []
    
    # throw all the atom records in one list
    atom_records = []
    for res in mdl:
        for atom_record in res.atom_records:
            atom_records.append(atom_record)
    
    for i, atom_record in enumerate(atom_records):
        if i in clash_atm:

            # ignore hydrogens?
            if 'H' in atom_record.name:
                continue

            # print('found clash', atom_record.resSeq, atom_record.resName)
            # print(atom_record.serial)

            # check if atom is a backbone atom
            if atom_record.name in ['C', 'N', 'CA', 'O']:
                ca_clashes.append(atom_record.resSeq)
            else:
                sc_clashes.append(atom_record.resSeq)

    ca_clashes = [*set(ca_clashes)]
    sc_clashes = [*set(sc_clashes)]

    return ca_clashes, sc_clashes
