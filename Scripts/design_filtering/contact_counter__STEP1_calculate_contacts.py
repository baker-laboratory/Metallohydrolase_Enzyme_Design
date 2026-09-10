#!/usr/bin/env python3
"""
contact_counter.py

Counts protein–ligand contacts at custom distance cutoffs around all TS atoms (grouped or split),
writes out a detailed CSV, then (by default) aggregates across ligands into one-row summary,
and can optionally dump per‑cutoff PDBs of ligand + contacting protein atoms.

Usage:
    python contact_counter.py input.pdb \
        [--ligands LIG1 LIG2 ...] \
        [--split-ligands] \
        [--include-hydrogens] \
        [--cutoffs 4 5 6 7.2] \
        [--return_HETATM_coordinates_from_cutoffs] \
        [--keep_original_csv_ligANDcutoff_separated_by_row]

Dependencies:
    numpy, pandas
"""

import argparse, os, math
import numpy as np
import pandas as pd

# definitions
BACKBONE_ATOMS     = {"N","CA","C","O"}
AROMATIC_RESIDUES  = {"PHE","TYR","TRP","HIS"}
WATER_RESNAMES     = {"HOH","WAT"}

def parse_args():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("pdb", help="Path to input PDB file")
    p.add_argument(
        "--ligands", nargs="+",
        help="Residue names of ligands to consider. "
             "If omitted, all HETATM (minus water) are grouped together."
    )
    p.add_argument(
        "--split-ligands", action="store_true",
        help="Treat each HETATM residue separately (labeled by resname+chain+resid)."
    )
    p.add_argument(
        "--include-hydrogens", action="store_true",
        help="Do not filter out H atoms (by default hydrogens are excluded)."
    )
    p.add_argument(
        "--cutoffs", nargs="+", type=float, default=[4.0, 5.0, 6.0],
        help="List of distance cutoffs (in Å). Default: 4.0 5.0 6.0"
    )
    p.add_argument(
        "--return_HETATM_coordinates_from_cutoffs",
        action="store_true",
        help="Dump per‑cutoff PDBs containing ligand HETATM + contacting protein ATOM lines."
    )
    p.add_argument(
        "--keep_original_csv_ligANDcutoff_separated_by_row",
        action="store_true",
        help="Skip the across‑ligand aggregation step (and keep detailed CSV)."
    )
    return p.parse_args()

def read_pdb_atoms(pdb_path):
    atoms = []
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            record   = line[:6].strip()
            atom_name= line[12:16].strip()
            resName  = line[17:20].strip()
            chainID  = line[21].strip()
            resSeq   = int(line[22:26])
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            element  = line[76:78].strip() or atom_name[0]
            atoms.append({
                "record": record,
                "atom_name": atom_name,
                "resName": resName,
                "chainID": chainID,
                "resSeq": resSeq,
                "coord": (x,y,z),
                "element": element.upper(),
                "line": line.rstrip("\n")
            })
    return atoms

def classify_protein_atoms(atoms, include_h):
    prot = [a for a in atoms if a["record"]=="ATOM"]
    if not include_h:
        prot = [a for a in prot if a["element"]!="H"]
    coords  = np.array([a["coord"] for a in prot])
    idx_map = {i:a for i,a in enumerate(prot)}
    bb_idx     = {i for i,a in idx_map.items() if a["atom_name"] in BACKBONE_ATOMS}
    sc_idx     = {i for i in idx_map if i not in bb_idx}
    polar_idx  = {i for i,a in idx_map.items() if a["element"] in {"N","O","S"}}
    nonpol_idx = {i for i,a in idx_map.items() if a["element"]=="C"}
    arom_idx   = {i for i,a in idx_map.items()
                  if a["resName"] in AROMATIC_RESIDUES and a["atom_name"] not in BACKBONE_ATOMS}
    return prot, coords, bb_idx, sc_idx, polar_idx, nonpol_idx, arom_idx

def build_ligand_groups(atoms, args):
    het = [a for a in atoms if a["record"]=="HETATM" and a["resName"] not in WATER_RESNAMES]
    if args.ligands:
        groups = { lig:[a for a in het if a["resName"]==lig] for lig in args.ligands }
    elif args.split_ligands:
        groups = {}
        for a in het:
            key = f"{a['resName']}{a['chainID']}{a['resSeq']}"
            groups.setdefault(key,[]).append(a)
    else:
        groups = {"all_HETATM": het}
    if not args.include_hydrogens:
        for k,v in groups.items():
            groups[k] = [a for a in v if a["element"]!="H"]
    return groups

def count_contacts(prot_coords, prot_idx_sets, lig_atoms, cutoffs):
    bb_idx, sc_idx, polar_idx, nonpol_idx, arom_idx = prot_idx_sets
    lig_coords = np.array([a["coord"] for a in lig_atoms])
    if lig_coords.size==0 or prot_coords.size==0:
        return { c:(0,0,0,0,0,0) for c in cutoffs }
    D = np.sqrt(((lig_coords[:,None,:] - prot_coords[None,:,:])**2).sum(axis=2))
    results = {}
    for c in cutoffs:
        cols = np.unique(np.where(D<=c)[1])
        contacting = set(cols.tolist())
        results[c] = (
            len(contacting),
            len(contacting & sc_idx),
            len(contacting & bb_idx),
            len(contacting & polar_idx),
            len(contacting & nonpol_idx),
            len(contacting & arom_idx)
        )
    return results

def main():
    args = parse_args()
    print(f"[INFO] Parsing PDB: {args.pdb}")
    atoms = read_pdb_atoms(args.pdb)

    print(f"[INFO] Using cutoffs: {args.cutoffs} Å")
    prot, prot_coords, bb_idx, sc_idx, polar_idx, nonpol_idx, arom_idx = \
        classify_protein_atoms(atoms, args.include_hydrogens)
    print(f"  total protein atoms: {len(prot)}")

    ligand_groups = build_ligand_groups(atoms, args)
    print(f"[INFO] Built {len(ligand_groups)} ligand groups: {list(ligand_groups.keys())}")

    records = []
    for name, lig in ligand_groups.items():
        print(f"[INFO] Ligand '{name}' with {len(lig)} atoms")
        res = count_contacts(
            prot_coords,
            (bb_idx, sc_idx, polar_idx, nonpol_idx, arom_idx),
            lig, args.cutoffs
        )
        for c in args.cutoffs:
            all_c, sc_c, bb_c, polar_c, nonpol_c, arom_c = res[c]
            print(f"  cutoff {c}Å → total={all_c}, sc={sc_c}, bb={bb_c}, pol={polar_c}, np={nonpol_c}, arom={arom_c}")
            records.append({
                "ligand": name,
                "cutoff": c,
                "all_hetatm_contacts":      all_c,
                "sidechain_contacts":       sc_c,
                "backbone_contacts":        bb_c,
                "protein_polar_contacts":   polar_c,
                "protein_nonpolar_contacts":nonpol_c,
                "aromatic_sc_contacts":     arom_c
            })

    # write detailed CSV
    df = pd.DataFrame(records)
    out_csv = os.path.splitext(args.pdb)[0] + "__contacts_by_cutoff.csv"
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Wrote detailed CSV → {out_csv}")

    # optionally dump PDBs per cutoff
    if args.return_HETATM_coordinates_from_cutoffs:
        print("[INFO] Writing per‑cutoff PDBs of ligand + contacting protein atoms…")
        # collect all ligand atoms
        lig_atoms_all = [a for group in ligand_groups.values() for a in group]
        lig_coords = np.array([a["coord"] for a in lig_atoms_all])
        lig_keys = {(a["chainID"], a["resSeq"], a["atom_name"], a["record"]) for a in lig_atoms_all}

        for c in args.cutoffs:
            prot_idxs = set()
            if lig_coords.size and prot_coords.size:
                D = np.sqrt(((prot_coords[:,None,:] - lig_coords[None,:,:])**2).sum(axis=2))
                prot_idxs = set(np.unique(np.where(D<=c)[0].tolist()))
            out_pdb = os.path.splitext(args.pdb)[0] + f"__CONTACTcutoff_{c}A.pdb"
            with open(out_pdb, "w") as fh:
                # write contacting protein ATOMs
                for i in sorted(prot_idxs):
                    fh.write(prot[i]["line"] + "\n")
                # write all ligand HETATMs
                for a in lig_atoms_all:
                    fh.write(a["line"] + "\n")
            print(f"[INFO] Wrote {len(prot_idxs)} protein + {len(lig_atoms_all)} ligand lines to {out_pdb}")

    if args.keep_original_csv_ligANDcutoff_separated_by_row:
        print("[INFO] Skipping aggregation, done.")
        return

    # aggregate by cutoff
    print("[INFO] Aggregating across ligands…")
    num_cols = [c for c in df.columns if c not in ("ligand","cutoff")]
    agg = df.drop(columns=["ligand"]).groupby("cutoff", as_index=False)[num_cols].sum()

    # pivot into one row
    summary = {}
    for _, row in agg.iterrows():
        co = int(row["cutoff"])
        for m in num_cols:
            summary[f"{m}__CUTOFF_{co}A"] = int(row[m])

    summary_df = pd.DataFrame([summary])
    out_sum = os.path.splitext(args.pdb)[0] + "__contacts_summary.csv"
    summary_df.to_csv(out_sum, index=False)
    print(f"[INFO] Wrote summary CSV → {out_sum}")

    # remove detailed CSV unless keep flag
    os.remove(out_csv)
    print(f"[INFO] Deleted detailed CSV → {out_csv}")

if __name__ == "__main__":
    main()
