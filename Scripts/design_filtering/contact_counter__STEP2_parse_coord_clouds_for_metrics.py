#!/usr/bin/env python3
"""
Author: Seth M. Woodbury
contact_counter__STEP2_parse_coord_clouds_for_metrics.py

------------------------------
This script extends fpocket’s pocket-detection by analyzing the “coordinate clouds” generated around a ligand at multiple distance cutoffs. It reads in per‑cutoff PDB files of contacting protein atoms plus the unmodified ligand (created by STEP1), recenters them on the ligand centroid, and computes a suite of geometric and topological metrics to characterize binding-site shape, volume, and potential tunnel entrances.

**Logic & Workflow**
1. **Locate input files**: For each user‑specified PDB, find `<base>__contacts_summary.csv` and all `<base>__CONTACTcutoff_<d>A.pdb` files in the same directory.
2. **Parse coordinates**: Separate ATOM records (protein) and HETATM records (ligand). Compute ligand centroid `L` and recenter protein points at the origin.
3. **Convex hull**: Build a 3D convex hull around the protein contact cloud. Report its **volume** (Å³) and **surface area** (Å²), giving pocket size.
4. **PCA elongation**: Perform PCA on the centered protein points. Compute the ratio λ₁/(λ₂+λ₃) to quantify how rod‑like (high ratio) versus sphere‑like (ratio≈1) the cloud is.
5. **Spherical‐occupancy tunnel detection**:
   - Uniformly sample `--sphere-samples` directions on the unit sphere.
   - For each angle threshold θ in `--angle-thresholds`, mark directions blocked if any protein atom lies within ±θ of that ray.
   - Identify the remaining “empty” directions, cluster them with DBSCAN over parameters `eps_list` (angular spread) and `min_list` (minimum cluster size). Each cluster represents a putative tunnel mouth.
   - For each cluster, compute:
     - **Entrance axis**: mean direction vector of the cluster.
     - **Aperture half‐angle**: max angular deviation from the mean (degrees).
     - **Diameter**: 2×(cutoff)×sin(aperture half‐angle).
     - **Occupancy fraction**: fraction of sphere directions unblocked.
6. **Rim residues**: For each valid cluster, project protein atoms onto its axis. Atoms within 1Å of the furthest projection form the tunnel rim—list their residue IDs.
7. **Δ‑metrics**: Compute differences in convex‐hull volume and area between the smallest and largest cutoffs (e.g. 4Å→6Å) to characterize pocket growth.
8. **Batch insertion**: Collate all metrics into a single pandas DataFrame row and append to the existing summary file in one operation for performance.

**Usage**
```bash
python contact_counter__STEP2_parse_coord_clouds_for_metrics.py \
  structure.pdb \
  --sphere-samples 1000 \
  --angle-thresholds 10 15 20 25
```
Ensure you have already run STEP1 with `--return_HETATM_coordinates_from_cutoffs` so that the `__CONTACTcutoff_*A.pdb` clouds exist alongside the `__contacts_summary.csv`.

**Adjusting Metrics**
- **`--sphere-samples`**: More samples → higher angular resolution (slower). Default 1000.
- **`--angle-thresholds`**: Search for entrances under different angular tolerances. Smaller angles catch narrow tunnels; larger angles capture wider openings.
- **`eps_list`** / **`min_list`**: Controls clustering of empty directions:
  - `eps_list` values correspond roughly to angular neighbor tolerances (e.g. 0.1→~5.7°).
  - `min_list` sets the minimum number of sampled directions to call a cluster real. Lower → small leaks flagged; higher → require broad openings.

**Expected Results & Interpretation**
- **Convex hull volume/area** quantify site size. Large Δ‑values suggest deep cavities.
- **Elongation ratio** near 1 implies near‑spherical pockets; high values indicate elongated tunnels.
- **Entrance metrics** reveal directionality and size of any tunnels. Occupancy fraction gives the proportion of unblocked directions.
- **Rim residues** identify which amino acids line the mouth.

**Potential Bugs & Pitfalls**
- Missing or misnamed `__CONTACTcutoff_*A.pdb` or summary CSV → script skips that PDB.
- Extremely uniform or sparse clouds may yield no valid clusters (no entrance).
- Very large `sphere_samples` with exhaustive grids can be time‑consuming.
- DataFrame fragmentation avoided by batch insertion.

**Debugging & Precautions**
- To diagnose tunnel detection, print intermediate counts:
  ```python
  print(len(empty_dirs), "empty directions at angle", angle)
  ```
- If no entrances appear, try relaxing `min_list` or increasing `eps_list`.
- For reproducibility, set a fixed random seed (e.g. in direction sampling).
- Always inspect a few `__CONTACTcutoff_*A.pdb` clouds visually to confirm that STEP1 outputs are correct before running STEP2.

Usage:
    python contact_counter__STEP2_parse_coord_clouds_for_metrics.py file1.pdb [file2.pdb ...]

Dependencies:
    numpy, pandas, scipy, sklearn, sklearn.cluster
"""
import os
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import warnings
warnings.simplefilter("ignore", pd.errors.PerformanceWarning)


def parse_coord_pdb(pdb_path):
    prot = []
    lig = []
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("ATOM"):
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                prot.append((x, y, z, line))
            elif line.startswith("HETATM"):
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                lig.append((x, y, z, line))
    return prot, lig


def fibonacci_sphere(samples=10000):
    # generate approximately uniform directions on unit sphere
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y from 1 to -1
        radius = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append((x, y, z))
    return np.array(points)


def compute_spherical_entrance(prot_vecs, cutoff, n_dirs=10000, angle_thresh_deg=15, db_eps=0.1, db_min=5):
    norms = np.linalg.norm(prot_vecs, axis=1)
    valid = norms > 1e-6
    if valid.sum() == 0:
        return None, None, None, 0.0
    unit_prot = prot_vecs[valid] / norms[valid][:, None]
    samples = fibonacci_sphere(samples=n_dirs)
    cos_thresh = np.cos(np.deg2rad(angle_thresh_deg))
    blocked = np.any(unit_prot @ samples.T >= cos_thresh, axis=0)
    empty_dirs = samples[~blocked]
    occ_frac = len(empty_dirs) / samples.shape[0]
    if len(empty_dirs) < db_min:
        return None, None, None, occ_frac
    clustering = DBSCAN(eps=db_eps, min_samples=db_min).fit(empty_dirs)
    labels = clustering.labels_
    clusters = [empty_dirs[labels == lbl] for lbl in set(labels) if lbl != -1]
    if not clusters:
        return None, None, None, occ_frac
    biggest = max(clusters, key=lambda c: len(c))
    mean_dir = biggest.mean(axis=0)
    mean_dir /= np.linalg.norm(mean_dir)
    angles = np.arccos(np.clip(biggest @ mean_dir, -1, 1))
    aperture_angle = float(np.degrees(angles.max()))
    aperture_diameter = 2 * cutoff * np.sin(np.radians(aperture_angle))
    return mean_dir, aperture_angle, aperture_diameter, occ_frac


def parse_residue_id(line):
    resName = line[17:20].strip()
    chainID = line[21].strip()
    resSeq = int(line[22:26])
    return resName, chainID, resSeq


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("pdb", nargs='+', help="Input PDB(s) for STEP2 parsing.")
    parser.add_argument("--sphere-samples", type=int, default=10000,
                        help="# directions to sample on sphere.")
    parser.add_argument("--angle-thresholds", nargs='+', type=float,
                        default=[10.0, 15.0, 20.0, 25.0],
                        help="List of half-angle thresholds (deg) to test.")
    parser.add_argument("--dbscan-eps", type=float, default=0.1,
                        help="DBSCAN eps default for clustering empty directions.")
    parser.add_argument("--dbscan-min", type=int, default=5,
                        help="DBSCAN min_samples default.")
    args = parser.parse_args()

    # Parameter grid for DBSCAN sampling
    eps_list = [0.05, 0.1, 0.2]
    min_list = [3, 5, 10]

    for pdb_file in args.pdb:
        base = os.path.splitext(pdb_file)[0]
        summary_csv = base + "__contacts_summary.csv"
        print(f"[STEP2] Processing {pdb_file}")
        if not os.path.isfile(summary_csv):
            print(f"[WARN] Summary CSV not found: {summary_csv}, skipping STEP2")
            continue
        pat = base + "__CONTACTcutoff_*A.pdb"
        cutoff_files = sorted(glob.glob(pat))
        if not cutoff_files:
            print(f"[WARN] No cutoff PDBs found for pattern {pat}")
            continue
        print(f"[INFO] Found {len(cutoff_files)} cutoff PDB clouds:")
        for f in cutoff_files:
            print("   ", os.path.basename(f))

        df = pd.read_csv(summary_csv)
        new_metrics = {}
        volumes = {}
        areas = {}

        for file in cutoff_files:
            c_str = os.path.basename(file).split('_CONTACTcutoff_')[-1].rstrip('A.pdb')
            cutoff = float(c_str)
            print(f"\n[INFO] Computing metrics for cutoff = {cutoff} Å")
            suf = f"__CUTOFF_{int(cutoff)}A"

            prot, lig = parse_coord_pdb(file)
            prot_coords = np.array([(x, y, z) for x, y, z, _ in prot])
            lig_coords = np.array([(x, y, z) for x, y, z, _ in lig])
            if prot_coords.size == 0 or lig_coords.size == 0:
                print("[WARN] Missing ligand or protein coords, skipping this cutoff")
                continue
            L = lig_coords.mean(axis=0)
            prot_trans = prot_coords - L

            # Convex hull volume & area
            if len(prot_trans) >= 4:
                hull = ConvexHull(prot_trans)
                vol = float(hull.volume)
                area = float(hull.area)
            else:
                vol = np.nan
                area = np.nan
            volumes[cutoff] = vol
            areas[cutoff] = area
            print(f"  Convex hull volume = {vol:.2f}, surface area = {area:.2f}")

            # PCA elongation ratio
            try:
                pca = PCA(n_components=3).fit(prot_trans)
                lambdas = pca.explained_variance_
                elong = float(lambdas[0] / (lambdas[1] + lambdas[2]))
            except Exception as e:
                elong = np.nan
                print(f"[WARN] PCA failed: {e}")
            print(f"  Elongation ratio λ1/(λ2+λ3) = {elong:.3f}")

            # Tunnel entrance + occupancy for each angle-threshold, eps, min
            for angle in args.angle_thresholds:
                for eps in eps_list:
                    for m in min_list:
                        md, ap_ang, ap_diam, occ = compute_spherical_entrance(
                            prot_trans, cutoff,
                            n_dirs=args.sphere_samples,
                            angle_thresh_deg=angle,
                            db_eps=eps,
                            db_min=m
                        )
                        tag = f"{suf}__ANGLE_{int(angle)}__EPS_{eps}_MIN_{m}"
                        if md is None:
                            print(f"[WARN] No entrance at {angle}° eps={eps} min={m}")
                            md = (np.nan, np.nan, np.nan)
                            ap_ang = np.nan
                            ap_diam = np.nan
                        else:
                            print(f"  angle={angle}°, eps={eps}, min={m} → axis=[{md[0]:.3f},{md[1]:.3f},{md[2]:.3f}], aperture={ap_ang:.1f}°, diam≈{ap_diam:.2f} Å, free_frac={occ:.2f}")
                        new_metrics[f"entrance_x{tag}"] = md[0]
                        new_metrics[f"entrance_y{tag}"] = md[1]
                        new_metrics[f"entrance_z{tag}"] = md[2]
                        new_metrics[f"aperture_half_angle_deg{tag}"] = ap_ang
                        new_metrics[f"aperture_diameter{tag}"] = ap_diam
                        new_metrics[f"occupancy_fraction{tag}"] = occ

                        # Rim residues at mouth for this configuration
                        projs = prot_trans @ np.array(md)
                        thresh = projs.max() - 1.0
                        rim_ids = set()
                        for idx in np.where(projs >= thresh)[0]:
                            resName, chain, resSeq = parse_residue_id(prot[idx][3])
                            rim_ids.add(f"{resName}{chain}{resSeq}")
                        new_metrics[f"rim_residues{tag}"] = ",".join(sorted(rim_ids))
                        print(f"    rim residues: {', '.join(sorted(rim_ids))}")

            # store core metrics
            new_metrics[f"convex_hull_volume{suf}"] = vol
            new_metrics[f"convex_hull_area{suf}"] = area
            new_metrics[f"elongation_ratio{suf}"] = elong

        # multi-cutoff differences
        cutoffs = sorted(volumes.keys())
        if len(cutoffs) >= 2:
            small, large = cutoffs[0], cutoffs[-1]
            dv = volumes[large] - volumes[small]
            da = areas[large] - areas[small]
            print(f"\n[INFO] Δvolume ({large}A - {small}A) = {dv:.2f}")
            print(f"[INFO] Δarea   ({large}A - {small}A) = {da:.2f}")
            new_metrics[f"delta_volume_{int(large)}A_minus_{int(small)}A"] = dv
            new_metrics[f"delta_area_{int(large)}A_minus_{int(small)}A"] = da

        # append to summary
        print(f"[INFO] Updating summary CSV {summary_csv} with new metrics (batch insert)...")
        # turn your dict of metrics into a one‐row DataFrame
        extras = pd.DataFrame([new_metrics])
        # concatenate side by side
        df = pd.concat([df, extras], axis=1)
        # write out
        df.to_csv(summary_csv, index=False)

        print(f"[STEP2] Done with {pdb_file}\n")

if __name__ == "__main__":
    main()
