#!/usr/bin/env python3
"""
protein_size_shape_metrics__STEP1_ligand_metrics.py

Compute ligand-related metrics for a given PDB and append them to the existing
protein shape/size metrics CSV (created by protein_size_shape_metrics__MAIN.py).

Logic:
  1. Identify all heteroatom ligands in the PDB (exclude waters: HOH, WAT).
  2. Sort ligand codes (residue names) alphanumerically.
     - If only one code: use suffix `_lig` and set `ligand_code` column.
     - If multiple: set `ligand_code_1`, `ligand_code_2`, ... and use suffix `_lig_<CODE>`.
  3. Compute per-ligand metrics:
     - `dist_COM`: distance between ligand center of mass and protein COM
     - `nearest_prot_dist`: minimum atom–atom distance ligand → protein
     - `ligand_Rg`: radius of gyration of the ligand atoms
     - `N_atoms`: total number of ligand atoms
     - `N_heavy_atoms`: count of non-H atoms in the ligand

Usage:
    python protein_size_shape_metrics__STEP1_ligand_metrics.py /path/to/structure.pdb

Dependencies:
    numpy, pandas, scipy
"""
import os
import argparse
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

WATER_RESNAMES = {'HOH', 'WAT'}


def parse_pdb_atoms(pdb_path):
    """
    Returns two lists:
      - prot_coords: list of 3-arrays for ATOM records
      - lig_atoms: list of tuples (resName, coord-array, element) for HETATM excluding water
    """
    prot = []
    lig = []
    with open(pdb_path) as fh:
        for line in fh:
            rec = line[0:6].strip()
            if rec == 'ATOM':
                x,y,z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                prot.append(np.array([x,y,z]))
            elif rec == 'HETATM':
                resName = line[17:20].strip()
                if resName in WATER_RESNAMES:
                    continue
                x,y,z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                elem = line[76:78].strip() or line[12:14].strip()[0]
                lig.append((resName, np.array([x,y,z]), elem.upper()))
    return prot, lig


def compute_ligand_metrics(prot_coords, ligand_atoms):
    """
    Given protein coords list and ligand atom coords list, compute:
      - ligand COM
      - protein COM
      - COM distance
      - nearest atom distance
      - ligand Rg
      - N atoms, N heavy atoms
    """
    prot_arr = np.vstack(prot_coords) if prot_coords else np.zeros((0,3))
    lig_coords = np.vstack([a[1] for a in ligand_atoms]) if ligand_atoms else np.zeros((0,3))
    elements = [a[2] for a in ligand_atoms]

    # Centers of mass
    prot_com = prot_arr.mean(axis=0) if len(prot_arr)>0 else np.zeros(3)
    lig_com  = lig_coords.mean(axis=0) if len(lig_coords)>0 else np.zeros(3)

    # COM distance
    dist_com = float(np.linalg.norm(lig_com - prot_com))

    # nearest atom-atom distance
    if len(prot_arr)>0 and len(lig_coords)>0:
        dmin = float(cdist(lig_coords, prot_arr).min())
    else:
        dmin = np.nan

    # ligand Rg
    disp = lig_coords - lig_com
    if len(lig_coords)>0:
        Rg = float(np.sqrt((disp**2).sum(axis=1).mean()))
    else:
        Rg = np.nan

    # counts
    N_atoms = len(lig_coords)
    N_heavy = sum(1 for e in elements if e!='H')

    return {
        'dist_COM': dist_com,
        'nearest_prot_dist': dmin,
        'ligand_Rg': Rg,
        'N_atoms': N_atoms,
        'N_heavy_atoms': N_heavy
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('pdb', help='Input PDB file.')
    args = parser.parse_args()

    pdb_path = args.pdb
    base = os.path.splitext(pdb_path)[0]
    metrics_csv = base + '_protein_shapeSIZE_metrics.csv'
    if not os.path.isfile(metrics_csv):
        print(f"[ERROR] Metrics CSV not found: {metrics_csv}")
        return

    # parse atoms
    prot_coords, lig_atoms = parse_pdb_atoms(pdb_path)
    # identify ligand codes
    codes = sorted({res for res,_,_ in lig_atoms})
    if not codes:
        print("[INFO] No ligands detected; skipping ligand metrics.")
        return

    # count them
    new_cols = {}
    new_cols['number_of_ligands'] = len(codes)

    df = pd.read_csv(metrics_csv)
    new_cols = {}

    # map ligand_code(s)
    if len(codes)==1:
        code = codes[0]
        new_cols['ligand_code'] = code
        groups = [(code, lig_atoms)]
        suffix = '_lig'
    else:
        groups = []
        for i,code in enumerate(codes, start=1):
            new_cols[f'ligand_code_{i}'] = code
            # select atoms of this code
            grp = [(r,c,e) for (r,c,e) in lig_atoms if r==code]
            groups.append((code, grp))
        suffix = '_lig_{}'  # will format with code

    # compute metrics per ligand group
    for code, atoms in groups:
        mets = compute_ligand_metrics(prot_coords, atoms)
        for k,v in mets.items():
            if len(codes)==1:
                col = f"{k}{suffix}"
            else:
                col = f"{k}{suffix.format(code)}"
            new_cols[col] = v

    # append and write
    out = pd.concat([df, pd.DataFrame([new_cols])], axis=1)
    out.to_csv(metrics_csv, index=False)
    print(f"[INFO] Appended ligand metrics to {metrics_csv}")

if __name__=='__main__':
    main()
