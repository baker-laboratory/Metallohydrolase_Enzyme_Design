#!/usr/bin/env python3
"""
postprocess_fpocket_info.py

This script parses a fpocket info text file and a fpocket pockets PDB,
reindexes pockets sequentially, and outputs raw and aggregated metrics into CSV(s).
By default the raw per-pocket CSV is deleted unless --do_not_delete_raw_csv is specified.

Inputs:
  --fpocket_txt_file             Path to the fpocket *_info.txt file.
  --fpocket_pdb_pockets_file     Path to the fpocket *_out.pdb file (for validation).
  --do_not_delete_raw_csv        Keep the raw per-pocket CSV after aggregation.

Usage:
  python postprocess_fpocket_info.py \
    --fpocket_txt_file /path/to/group1_info.txt \
    --fpocket_pdb_pockets_file /path/to/group1_out.pdb \
    [--do_not_delete_raw_csv]
"""
import argparse
import os
import re
import sys
import pandas as pd


def slugify(key: str) -> str:
    s = key.lower()
    s = re.sub(r"[^0-9a-z]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def parse_info_txt(txt_path):
    # Read all lines
    with open(txt_path) as f:
        raw_lines = [line.rstrip() for line in f.readlines()]

    pocket_dicts = []
    current = None
    for line in raw_lines:
        # Header for a new pocket
        header = re.match(r"^Pocket\s+(\d+)", line)
        if header:
            # save previous
            if current:
                pocket_dicts.append(current)
            pid = int(header.group(1))
            current = {'orig_pocket_id': pid}
            continue
        if current is None:
            continue  # skip lines before first Pocket
        # metric lines: look for 'key : value'
        if ':' in line:
            parts = line.split(':', 1)
            key = slugify(parts[0])
            val = parts[1].strip()
            # try numeric
            try:
                if re.match(r"^[0-9]*\.[0-9]+$", val) or 'e' in val.lower():
                    current[key] = float(val)
                else:
                    current[key] = int(val)
            except ValueError:
                current[key] = val
    # append last
    if current:
        pocket_dicts.append(current)
    return pocket_dicts


def main():
    parser = argparse.ArgumentParser(
        description="Convert and aggregate fpocket metrics"
    )
    parser.add_argument("--fpocket_txt_file", required=True)
    parser.add_argument("--fpocket_pdb_pockets_file", required=True)
    parser.add_argument(
        "--do_not_delete_raw_csv", action="store_true",
        help="Keep the raw per-pocket CSV after aggregation"
    )
    args = parser.parse_args()

    txt_file = os.path.abspath(args.fpocket_txt_file)
    if not os.path.isfile(txt_file):
        print(f"[ERROR] txt file not found: {txt_file}", file=sys.stderr)
        sys.exit(1)
    pdb_file = os.path.abspath(args.fpocket_pdb_pockets_file)
    if not os.path.isfile(pdb_file):
        print(f"[WARNING] pockets pdb file not found: {pdb_file}", file=sys.stderr)

    pocket_list = parse_info_txt(txt_file)
    if not pocket_list:
        print("[ERROR] No pocket blocks parsed.", file=sys.stderr)
        sys.exit(1)

    # Build per-pocket DataFrame
    record = {}
    for new_idx, pocket in enumerate(pocket_list, start=1):
        for k, v in pocket.items():
            if k == 'orig_pocket_id':
                continue
            col = f"pocket_{new_idx}__{k}"
            record[col] = v
    df = pd.DataFrame([record])

    # Save raw CSV
    raw_csv = os.path.splitext(txt_file)[0] + '.csv'
    df.to_csv(raw_csv, index=False)
    print(f"[INFO] Saved raw metrics CSV to {raw_csv}")

    # Determine pocket indices
    pocket_indices = sorted(
        int(re.match(r"pocket_(\d+)__.*", col).group(1))
        for col in df.columns if col.startswith('pocket_') and '__' in col
    )

    # Aggregated metrics
    agg = {}
    # Max scores
    score_cols = [f"pocket_{i}__score" for i in pocket_indices if f"pocket_{i}__score" in df]
    agg['max_raw_score'] = df[score_cols].max(axis=1).iloc[0] if score_cols else None
    drg_cols = [f"pocket_{i}__druggability_score" for i in pocket_indices if f"pocket_{i}__druggability_score" in df]
    agg['max_druggability_score'] = df[drg_cols].max(axis=1).iloc[0] if drg_cols else None

    # Cumulative sums
    cum_keys = ['total_sasa','polar_sasa','apolar_sasa','volume','number_of_alpha_spheres',
                'flexibility','cent_of_mass_alpha_sphere_max_dist','charge_score',
                'polarity_score','volume_score']
    for key in cum_keys:
        cols = [f"pocket_{i}__{key}" for i in pocket_indices if f"pocket_{i}__{key}" in df]
        agg[f'cumulative_{key}'] = df[cols].sum(axis=1).iloc[0] if cols else 0.0

    # Compute weights from any occupancy_of_this_pocket column
    weights = {}
    for i in pocket_indices:
        wcol = [c for c in df.columns if c.startswith(f"pocket_{i}__") and 'occupation_of_this_pocket' in c]
        weights[i] = df[wcol].iloc[0,0] if wcol else 0.0
        print(f"[VERBOSE] Pocket {i} weight: {weights[i]}")

    # Weighted metrics
    weight_keys = ['score','druggability_score','mean_local_hydrophobic_density',
                   'mean_alpha_sphere_radius','mean_alp_sph_solvent_access','flexibility',
                   'alpha_sphere_density','charge_score','polarity_score','volume_score',
                   'hydrophobicity_score','apolar_alpha_sphere_proportion']
    for key in weight_keys:
        total=0.0
        for i in pocket_indices:
            col = f"pocket_{i}__{key}"
            if col in df.columns:
                total += df.at[0,col]*weights.get(i,0.0)
        agg[f'weighted_{key}'] = total
        print(f"[VERBOSE] weighted_{key}: {total}")

    # Write aggregated CSV
    agg_df = pd.DataFrame([agg])
    agg_csv = os.path.splitext(txt_file)[0] + '_aggregated.csv'
    agg_df.to_csv(agg_csv, index=False)
    print(f"[INFO] Saved aggregated metrics CSV to {agg_csv}")

    # Remove raw CSV if not requested
    if not args.do_not_delete_raw_csv:
        os.remove(raw_csv)
        print(f"[INFO] Removed raw CSV {raw_csv}")

if __name__ == '__main__':
    main()
