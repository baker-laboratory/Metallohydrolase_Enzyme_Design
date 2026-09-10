#!/usr/bin/env python3
"""Head-to-head test: does side-chain packing preserve fixed-residue coordinates?

Runs LigandMPNN's make_torsion_features with repack_everything=False on a real
structure, once with the stock sc_utils and once with the patched copy, and
reports the largest per-atom displacement of the residues marked fixed.

Expected: stock moves them (idealized rebuild); patched leaves them at 0.
"""
import sys, importlib, argparse
import numpy as np
import torch

def _repo_root():
    from pathlib import Path as _P
    for anc in _P(__file__).resolve().parents:
        if (anc / "Scripts").is_dir() and (anc / "Software").is_dir():
            return str(anc)
    raise RuntimeError("repository root not found")

REPO = _repo_root()
LIG = f"{REPO}/Software/fastmpnndesign/lib/LigandMPNN"
PATCHED = f"{REPO}/Scripts/mpnn_relevant_utils/ligandmpnn_patched"


def load_sc_utils(patched: bool):
    """Import sc_utils fresh, from the patched dir or the stock one."""
    for m in list(sys.modules):
        if m in ("sc_utils", "data_utils", "model_utils"):
            del sys.modules[m]
    sys.path[:] = [p for p in sys.path if p not in (LIG, PATCHED)]
    if patched:
        sys.path.insert(0, PATCHED)
    sys.path.insert(0 if not patched else 1, LIG)
    return importlib.import_module("sc_utils")


def build_features(pdb, fixed_resnums, device="cpu"):
    """Build the minimal feature_dict make_torsion_features needs, from a PDB."""
    sys.path.insert(0, LIG)
    import data_utils as du
    out, _, _, _, _ = du.parse_PDB(pdb, device=device, parse_all_atoms=True)
    L = out["R_idx"].shape[0]
    chain_mask = torch.ones([1, L], device=device)
    resnums = out["R_idx"].cpu().numpy().tolist()
    fixed_pos = [i for i, r in enumerate(resnums) if r in fixed_resnums]
    if not fixed_pos:                      # fall back to the first few residues
        fixed_pos = list(range(min(4, L)))
    for i in fixed_pos:
        chain_mask[0, i] = 0.0
    fd = {
        "X": out["X"][None].to(device),
        "S": out["S"][None].to(device).long(),
        "mask": out["mask"][None].to(device),
        "chain_mask": chain_mask,
        "xyz_37": out["xyz_37"][None].to(device),
    }
    return fd, fixed_pos, out


def max_fixed_displacement(sc_utils, fd, fixed_pos):
    """Largest per-atom shift of fixed residues between input and packed output."""
    td = sc_utils.make_torsion_features({k: v.clone() for k, v in fd.items()},
                                        repack_everything=False)
    xyz14 = td["xyz14_noised"]                      # [1, L, 14, 3]
    masks = sc_utils.make_atom14_masks({"aatype": td["S_af2"]})
    idx = masks["residx_atom14_to_atom37"][..., None].expand(-1, -1, -1, 3).long()
    truth = torch.gather(fd["xyz_37"], 2, idx) * masks["atom14_atom_exists"][..., None]
    exists = masks["atom14_atom_exists"][..., None]
    worst = 0.0
    for i in fixed_pos:
        d = ((xyz14[0, i] - truth[0, i]) * exists[0, i]).norm(dim=-1)
        worst = max(worst, float(d.max()))
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--fixed", type=int, nargs="*", default=[])
    a = ap.parse_args()

    print(f"structure : {a.pdb}")
    sc = load_sc_utils(patched=False)
    fd, fixed_pos, out = build_features(a.pdb, set(a.fixed))
    print(f"residues  : {fd['X'].shape[1]}   fixed positions: {fixed_pos}")
    print()

    stock = max_fixed_displacement(sc, fd, fixed_pos)
    print(f"  STOCK   LigandMPNN : max fixed-residue displacement = {stock:.4f} A")

    sc_p = load_sc_utils(patched=True)
    assert "ligandmpnn_patched" in sc_p.__file__, sc_p.__file__
    patched = max_fixed_displacement(sc_p, fd, fixed_pos)
    print(f"  PATCHED            : max fixed-residue displacement = {patched:.4f} A")
    print()

    ok = patched < 1e-4 and stock > patched
    print(f"  RESULT: {'PASS' if ok else 'FAIL'} "
          f"(patched preserves coordinates; stock moves them by {stock:.3f} A)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
