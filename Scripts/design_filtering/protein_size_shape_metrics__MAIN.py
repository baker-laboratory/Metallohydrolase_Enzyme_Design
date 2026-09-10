#!/usr/bin/env python3
"""
protein_size_shape_metrics__MAIN.py

Compute basic protein size and shape metrics for a given PDB file.

Metrics:
  - Amino acid length (number of standard residues, excluding HETATM)
  - Number of Cα atoms
  - Radius of gyration (R_g) computed on Cα coordinates
  - Normalized R_g = R_g / N^(1/3)
  - Inertia tensor eigenvalues (λ1 ≥ λ2 ≥ λ3) on Cα coords
  - Relative shape anisotropy κ² = 1 - 3*(λ1λ2 + λ2λ3 + λ1λ3)/(λ1 + λ2 + λ3)²
  - Maximum Cα–Cα distance (approx. diameter)

Usage:
    python protein_size_shape_metrics__MAIN.py /path/to/structure.pdb

Outputs:
    /path/to/structure_protein_shapeSIZE_metrics.csv
    containing a single row with all metrics.

Dependencies:
    numpy, pandas
"""
from pathlib import Path
import os
import argparse
import numpy as np
import pandas as pd
import subprocess
import sys
import shlex

# Sibling helper scripts resolve next to this file, so the whole set can be
# relocated together.
_SCRIPT_DIR = Path(__file__).resolve().parent

STEP1_LIGAND_SCRIPT = str(_SCRIPT_DIR / "protein_size_shape_metrics__STEP1_ligand_metrics.py")


def parse_pdb_ca(pdb_path):
    """
    Parse PDB and return list of (residue_id, coord) for Cα atoms.
    Excludes HETATM and non-ALA/... residues are included but only CA atom.
    """
    cas = []
    seen_res = set()
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith('ATOM'):
                continue
            atom_name = line[12:16].strip()
            if atom_name != 'CA':
                continue
            res_name = line[17:20].strip()
            chain_id = line[21].strip()
            res_seq = line[22:26].strip()
            res_id = (chain_id, res_seq, res_name)
            if res_id in seen_res:
                continue
            seen_res.add(res_id)
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            cas.append((res_id, np.array([x, y, z])))
    return cas


def compute_metrics(coords):
    """
    Given N×3 array of coordinates, compute:
      - Rg
      - normalized Rg (Rg / N^(1/3))
      - inertia eigenvalues and relative shape anisotropy
      - max pairwise distance
    """
    N = len(coords)
    if N == 0:
        return { 'Rg': np.nan, 'Rg_norm': np.nan,
                 'eig1': np.nan, 'eig2': np.nan, 'eig3': np.nan,
                 'shape_anisotropy': np.nan, 'max_dist': np.nan }
    # center coords
    center = coords.mean(axis=0)
    disp = coords - center
    # Radius of gyration
    Rg = np.sqrt((disp**2).sum(axis=1).mean())
    Rg_norm = Rg / (N**(1/3))
    # Inertia tensor (covariance-like)
    I = np.dot(disp.T, disp) / N  # 3x3
    evals = np.linalg.eigvalsh(I)
    # sort descending
    e1, e2, e3 = sorted(evals, reverse=True)
    # relative shape anisotropy κ^2
    denom = (e1+e2+e3)**2
    if denom > 0:
        shape_aniso = 1.0 - 3.0*(e1*e2 + e2*e3 + e1*e3)/denom
    else:
        shape_aniso = np.nan
    # max pairwise distance
    # efficient: compute condensed distances
    from scipy.spatial.distance import pdist
    max_dist = pdist(coords).max() if N > 1 else 0.0
    return { 'Rg': Rg, 'Rg_norm': Rg_norm,
             'eig1': e1, 'eig2': e2, 'eig3': e3,
             'shape_anisotropy': shape_aniso, 'max_dist': max_dist }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('pdb', help='Input PDB file.')
    args = parser.parse_args()

    pdb_path = args.pdb
    base = os.path.splitext(pdb_path)[0]
    out_csv = base + '_protein_shapeSIZE_metrics.csv'

    # parse Cα coordinates
    ca_list = parse_pdb_ca(pdb_path)
    length = len(ca_list)
    res_ids, coords = zip(*ca_list) if length>0 else ([], np.zeros((0,3)))
    coords = np.vstack(coords) if length>0 else np.zeros((0,3))

    # compute metrics
    met = compute_metrics(coords)

    # ——————————————————————————————
    # ideal‐sphere normalization of Rg
    # assume 110 Å³ per residue, convert to nm³ with packing correction
    v_res = 110.0                  # Å³ per amino acid (average)
    conv  = 1.212e-3               # Å³ → nm³ (incl. packing/hydration)
    V_nm3 = length * v_res * conv  # total volume in nm³
    # radius (nm) of a sphere with volume V_nm3: R = (3V/4π)^(1/3)
    R_sphere_nm = (3 * V_nm3 / (4 * np.pi)) ** (1/3)
    R_sphere_A  = R_sphere_nm * 10  # back to Å
    # dimensionless “sphericity” factor
    shape_factor = met['Rg'] / R_sphere_A

    # add to your metrics dict
    met['ideal_sphere_Rg'] = R_sphere_A
    met['Rg_norm_by_ideal_sphere'] = shape_factor
    # ——————————————————————————————

    # prepare output
    data = {
        'residue_length': length,
        'n_CA_atoms': length,
        **met
    }
    df = pd.DataFrame([data])
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Wrote metrics to {out_csv}")

    # ——————————————————————————————————————————————————————————————
    # If any non‐water HETATM (ligand) is present, run ligand‐metrics STEP1
    has_ligand = False
    with open(pdb_path) as f:
        for ln in f:
            if ln.startswith("HETATM") and ln[17:20].strip() not in ("HOH","WAT"):
                has_ligand = True
                break

    if has_ligand:
        lig_cmd = [
            sys.executable,
            STEP1_LIGAND_SCRIPT,
            pdb_path
        ]
        print(f"[INFO] Detected ligand(s), executing ligand‐metrics STEP1: {' '.join(shlex.quote(x) for x in lig_cmd)}")
        try:
            subprocess.run(lig_cmd, check=True)
            print("[INFO] Ligand metrics appended successfully.")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Ligand metrics STEP1 failed with exit code {e.returncode}")


if __name__ == '__main__':
    main()
