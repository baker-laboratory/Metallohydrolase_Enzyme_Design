#!/usr/bin/env python3
"""Measure steric clashes between designed side chains and fixed side chains.

Used to check whether making fixed residues visible to the packer as context
actually reduces clashes. Run the same input twice -- once normally, once with
LIGANDMPNN_NO_FIXED_CONTEXT=1 -- then point this at the two output directories:

    python test_fixed_residue_clashes.py <with_dir> <without_dir> 1,2,3


For each packed PDB, finds heavy-atom pairs where one atom belongs to a fixed
residue's side chain and the other to a designed residue's side chain, and
reports how close they get. Residues adjacent in sequence are excluded, since
near contacts there are backbone geometry, not packing decisions.
"""
import sys, glob, os
from collections import defaultdict

# Rough heavy-atom van der Waals radii; a pair is "clashing" when its
# separation falls below the sum of radii minus a tolerance.
VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "ZN": 1.39, "P": 1.80}
TOLERANCE = 0.4          # A of allowed interpenetration before calling it a clash
BACKBONE = {"N", "CA", "C", "O", "OXT"}


def element_of(name, raw_elem):
    e = (raw_elem or "").strip().upper()
    if e in VDW:
        return e
    n = name.strip().upper()
    return "ZN" if n.startswith("ZN") else (n[0] if n[:1] in "CNOSP" else "C")


def read_sidechains(pdb, fixed):
    """Return (fixed_atoms, designed_atoms) as lists of (resnum, name, elem, xyz)."""
    fx, dz = [], []
    for l in open(pdb):
        if not l.startswith(("ATOM", "HETATM")):
            continue
        name = l[12:16].strip()
        if name.startswith("H") or name in BACKBONE:
            continue                                  # side-chain heavy atoms only
        try:
            rn = int(l[22:26])
        except ValueError:
            continue
        xyz = (float(l[30:38]), float(l[38:46]), float(l[46:54]))
        rec = (rn, name, element_of(name, l[76:78]), xyz)
        (fx if rn in fixed else dz).append(rec)
    return fx, dz


def clashes(pdb, fixed):
    fx, dz = read_sidechains(pdb, fixed)
    worst, n_clash, closest = 0.0, 0, 9e9
    for rn_f, nm_f, el_f, (xf, yf, zf) in fx:
        for rn_d, nm_d, el_d, (xd, yd, zd) in dz:
            if abs(rn_f - rn_d) <= 1:                 # sequence neighbors: skip
                continue
            d = ((xf - xd) ** 2 + (yf - yd) ** 2 + (zf - zd) ** 2) ** 0.5
            if d > 6.0:
                continue
            closest = min(closest, d)
            allowed = VDW.get(el_f, 1.7) + VDW.get(el_d, 1.7) - TOLERANCE
            if d < allowed:
                n_clash += 1
                worst = max(worst, allowed - d)
    return n_clash, worst, (closest if closest < 9e9 else float("nan"))


def summarize(label, outdir, fixed):
    files = sorted(glob.glob(os.path.join(outdir, "packed", "*.pdb")))
    tot_c, tot_w, mins = 0, 0.0, []
    print(f"\n### {label}  ({len(files)} packed structures)")
    for f in files:
        c, w, closest = clashes(f, fixed)
        tot_c += c
        tot_w = max(tot_w, w)
        mins.append(closest)
        print(f"    {os.path.basename(f)[-24:]:26s} clashes={c:3d}  worst_overlap={w:.3f} A  closest={closest:.3f} A")
    mean_min = sum(mins) / len(mins) if mins else float("nan")
    print(f"    ---- total clashes {tot_c}, worst overlap {tot_w:.3f} A, mean closest approach {mean_min:.3f} A")
    return tot_c, tot_w, mean_min


if __name__ == "__main__":
    fixed = {int(x) for x in sys.argv[3].split(",")}
    a = summarize("WITH fixed-side-chain context (patched)", sys.argv[1], fixed)
    b = summarize("WITHOUT that context (coords still preserved)", sys.argv[2], fixed)
    print("\n" + "=" * 74)
    print(f"  clashes          : {a[0]:4d}  vs {b[0]:4d}   (with vs without context)")
    print(f"  worst overlap    : {a[1]:.3f} A vs {b[1]:.3f} A")
    print(f"  mean closest     : {a[2]:.3f} A vs {b[2]:.3f} A")
    if a[0] < b[0]:
        print("  -> context REDUCES clashes against fixed residues")
    elif a[0] == b[0]:
        print("  -> no difference in clash count on this input")
    else:
        print("  -> context INCREASED clashes; investigate before relying on it")
