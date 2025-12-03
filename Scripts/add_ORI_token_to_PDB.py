#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
ORI Token Generator (Deterministic, Modular, Verbose)
===============================================================================
Purpose
-------
Create an ORI pseudoatom record (or a *set* of them arranged as a sphere)
and append it/them to a PDB — without ever exceeding or dipping below the
requested `--sampling_size`. Defaults to writing a single ORI token at a
specified (or discovered) coordinate. Sphere modes are opt-in.

Key Behaviors
-------------
• Default: write **one** ORI at the center (no sphere).
• If `--sphere` is set:
    - The **first** ORI is the sphere center (the input coordinate).
    - The remaining (N-1) ORIs are *deterministically* placed:
        - `--mode surface`: on the sphere surface via Fibonacci distribution.
        - `--mode volume`: in shells (r ∝ (i/(N-1))^(1/3)) with Fibonacci angles.
    - Exactly `--sampling_size` ORIs are written. No more, no less.
• Coordinate source priority:
    1) `--center "x y z"` if provided
    2) First ORI atom found in `--input_pdb` (name == 'ORI')
    3) Otherwise: error with a helpful message.

Output
------
Each ORI is written to a **separate PDB** file by appending a well-formed
HETATM line and a `TER` to the original input PDB contents.

Examples
--------
Single ORI at provided coordinate:
    python ori_token_generator.py \
      --input_pdb in.pdb \
      --output_dir out_dir \
      --center "12.0 8.5 -3.0"

Sphere on surface (center + N-1 points on surface), radius 6 Å:
    python ori_token_generator.py \
      --input_pdb in.pdb \
      --output_dir out_dir \
      --center "0 0 0" \
      --sphere --radius 6.0 --sampling_size 25 --mode surface

Sphere in volume (center + N-1 interior points), radius 5 Å:
    python ori_token_generator.py \
      --input_pdb in.pdb \
      --output_dir out_dir \
      --center "10 10 10" \
      --sphere --radius 5.0 --sampling_size 64 --mode volume

Notes
-----
• Records are appended as:
  HETATM{serial:>5}  ORI ORI {chain}{resseq:>4}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           ORI
• `chain` defaults to 'X'. You can change it with `--chain`.
• Serial & residue numbers auto-increment from `--serial_start` and `--resseq_start`.
===============================================================================
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
import math
import numpy as np

# ------------------------------- Utilities -------------------------------- #

def vprint(verbose: bool, *args, **kwargs):
    """Conditional print."""
    if verbose:
        print(*args, **kwargs)

def parse_center_from_cli(center: str | None) -> np.ndarray | None:
    """Parse `--center "x y z"` string → np.array([x,y,z]) or None."""
    if center is None:
        return None
    parts = center.split()
    if len(parts) != 3:
        raise ValueError("`--center` must have exactly three numbers: e.g., --center \"12.0 8.5 -3.0\"")
    try:
        return np.array(list(map(float, parts)), dtype=float)
    except ValueError:
        raise ValueError("`--center` must be numeric, e.g., --center \"12.0 8.5 -3.0\"")

def find_first_ori_in_pdb(pdb_path: Path, verbose: bool) -> np.ndarray | None:
    """
    Scan PDB lines for an ORI atom (name == 'ORI'). Return its coordinates or None.
    Accepts both ATOM/HETATM records. Does not validate chain/residue.
    """
    try:
        with open(pdb_path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith(("ATOM  ", "HETATM")) and line[12:16].strip().upper() == "ORI":
                    # x/y/z columns (31-38, 39-46, 47-54 in 1-based PDB; sliced below 0-based)
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    vprint(verbose, f"[detect] Found existing ORI at ({x:.3f}, {y:.3f}, {z:.3f})")
                    return np.array([x, y, z], dtype=float)
    except Exception as e:
        vprint(verbose, f"[warn] Could not parse ORI from PDB: {e}")
    return None

def format_ori_pdb_line(x: float, y: float, z: float, serial: int, chain: str, resseq: int) -> str:
    """
    Return a single well-formatted PDB HETATM line for ORI.
    PDB columns (simplified, fixed widths). Keeps 'ORI' as atom+resname.
    """
    # Columns (1-based):
    #  1-6  "HETATM"
    #  7-11 serial
    # 13-16 atom name
    # 18-20 resname
    # 22    chain
    # 23-26 resseq
    # 31-38 x, 39-46 y, 47-54 z
    # 55-60 occupancy, 61-66 tempFactor
    # 77-78 element (nonstandard "OR" would've be more PDB-like; keep "OR" from ORI)
    element = f" {chain:1}"  # 2-char element is more PDB-compliant than "ORI"
    return (
        f"HETATM{serial:5d}  ORI ORI {chain:1}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
        f"{1.00:6.2f}{1.00:6.2f}          {element:>2}\n"
    )

def write_appended_pdb(base_pdb: Path,
                       out_path: Path,
                       ori_xyz_list: list[tuple[float, float, float]],
                       chain: str = "X",
                       serial_start: int = 1,
                       resseq_start: int = 1,
                       verbose: bool = False):
    """
    Append one or more ORI HETATM lines to the *original* PDB content and add a final TER.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(base_pdb, "r", encoding="utf-8") as fin:
        base_lines = fin.readlines()

    serial = serial_start
    resseq = resseq_start

    with open(out_path, "w", encoding="utf-8") as fout:
        fout.writelines(base_lines)
        for (x, y, z) in ori_xyz_list:
            line = format_ori_pdb_line(x, y, z, serial=serial, chain=chain, resseq=resseq)
            fout.write(line)
            serial += 1
            resseq += 1
        fout.write("TER\n")

    vprint(verbose, f"[write] Wrote {len(ori_xyz_list)} ORI record(s) → {str(out_path)}")

# ------------------------- Deterministic point sets ------------------------ #

def fibonacci_surface_points(n: int) -> np.ndarray:
    """
    Deterministic, even-ish distribution of n points on the unit sphere surface.
    Returns array shape (n, 3).
    """
    assert n >= 1
    if n == 1:
        return np.array([[0.0, 0.0, 1.0]], dtype=float)  # arbitrary pole
    # "Fibonacci / golden spiral" distribution
    phi = (1 + math.sqrt(5)) / 2.0  # golden ratio
    # Evenly spaced z (exclude exactly ±1 for stability if n>1)
    z = 1 - 2 * (np.arange(n) + 0.5) / n
    theta = 2 * math.pi * ((np.arange(n) / phi) % 1.0)
    r_xy = np.sqrt(np.clip(1.0 - z**2, 0.0, 1.0))
    xyz = np.stack([r_xy * np.cos(theta), r_xy * np.sin(theta), z], axis=1)
    return xyz.astype(float)

def volume_points_fibonacci_shells(n: int, radius: float) -> np.ndarray:
    """
    Equidistant-layer interior sampling favoring outer shells.
      • Number of layers: L = min( ceil(n / TARGET_PER_LAYER), floor(n / P_MIN) ), at least 1
      • Min points per layer = P_MIN (=10)
      • Equidistant radii: r_ell = (ell / L) * radius,  ell = 1..L
      • Remainder points distributed ∝ r^2 with fair rounding (outer-heavy)
      • Per-layer z-rotation to avoid cross-layer alignment
      • Uses fibonacci_surface_points(c) for each layer
    Here, n is the number of NON-center points (center handled elsewhere).
    """
    assert n >= 1
    import math

    # --- config ---
    P_MIN = 10             # minimum points per layer
    TARGET_PER_LAYER = 20  # preferred average points per layer

    print("\n[volume_points_fibonacci_shells] ===== PLAN =====")
    print(f"n (non-center points): {n}")
    print(f"radius (R):            {radius:.3f} Å")
    print(f"P_MIN:                 {P_MIN}  (min pts/layer)")
    print(f"TARGET_PER_LAYER:      {TARGET_PER_LAYER}")

    if n == 1:
        print("[plan] n=1 → single point on the outer shell.")
        out = np.array([[0.0, 0.0, float(radius)]], dtype=float)
        print(f"[out] 1 point @ r={radius:.3f} Å\n")
        return out

    # ---- choose number of layers (few, but data-driven) ----
    L_raw = int(math.ceil(n / float(TARGET_PER_LAYER)))
    L_cap = max(1, n // P_MIN) if n >= P_MIN else 1
    L = max(1, min(L_raw, L_cap))

    print(f"[layers] L_raw (ceil(n/target)) = {L_raw}")
    print(f"[layers] L_cap (floor(n/P_MIN)) = {L_cap}")
    print(f"[layers] → Using L = {L} layer(s)")

    # ---- equidistant radii: r_ell = (ell/L) * R ----
    radii = (np.arange(1, L + 1, dtype=float) / float(L)) * float(radius)
    radii_str = ", ".join(f"{r:.3f}" for r in radii)
    print(f"[radii] Equidistant shell radii (Å): [{radii_str}]")

    # ---- base counts + proportional (r^2) remainder ----
    counts = np.full(L, min(P_MIN, n), dtype=int)  # if n<P_MIN and L=1, this is n
    base = int(counts.sum())
    print(f"[alloc] Base counts per layer (min {P_MIN}): {counts.tolist()}  (base total = {base})")

    if L > 1 and base < n:
        remainder = n - base
        print(f"[alloc] Remainder to distribute: {remainder}")
        w = radii ** 2
        wsum = float(w.sum())
        print(f"[alloc] Weights r^2: {np.array2string(w, precision=3)} (sum={wsum:.3f})")

        alloc = remainder * (w / wsum)
        print(f"[alloc] Proportional (float) adds: {np.array2string(alloc, precision=3)}")

        add_floor = np.floor(alloc).astype(int)
        counts += add_floor
        leftover = remainder - int(add_floor.sum())

        print(f"[alloc] Floor adds: {add_floor.tolist()}  → counts now {counts.tolist()}  (leftover={leftover})")

        if leftover > 0:
            frac = alloc - add_floor
            order = np.argsort(-frac)  # biggest fractional parts first
            print(f"[alloc] Fractional parts: {np.array2string(frac, precision=3)}")
            print(f"[alloc] Assign leftover {leftover} to layers (by frac): {order[:leftover].tolist()}")
            counts[order[:leftover]] += 1
    else:
        if L == 1:
            print("[alloc] Single layer → all points on outer shell.")
        else:
            print("[alloc] No remainder to distribute.")

    print(f"[alloc] FINAL per-layer counts: {counts.tolist()}")
    print(f"[check] Sum counts = {counts.sum()} (should equal n={n})")

    # ---- per-layer phase to avoid cross-layer alignment ----
    def rotate_z(xyz: np.ndarray, ang: float) -> np.ndarray:
        ca, sa = math.cos(ang), math.sin(ang)
        Rz = np.array([[ca, -sa, 0.0],
                       [sa,  ca, 0.0],
                       [0.0, 0.0, 1.0]], dtype=float)
        return xyz @ Rz.T

    shells = []
    phi = (1 + math.sqrt(5)) / 2.0
    print("[build] ----- Generating shells with Fibonacci surface points -----")
    for idx, (r, c) in enumerate(zip(radii, counts), start=1):
        if c <= 0:
            print(f"[build] Layer {idx}: skipped (c <= 0)")
            continue
        phase = 2.0 * math.pi * ((idx / phi) % 1.0)  # deterministic offset
        print(f"[build] Layer {idx}: r={r:.3f} Å, count={int(c)}, phase={phase:.3f} rad")

        layer = fibonacci_surface_points(int(c)) * float(r)
        layer = rotate_z(layer, phase)
        shells.append(layer)

    out = np.vstack(shells).astype(float) if shells else np.zeros((0, 3), dtype=float)
    print(f"[done] Built total points: {len(out)} (n requested = {n})")
    print("[volume_points_fibonacci_shells] ===== END PLAN =====\n")
    return out

# ------------------------------- Main logic -------------------------------- #

def plan_ori_coordinates(center: np.ndarray,
                         sphere: bool,
                         mode: str,
                         radius: float,
                         sampling_size: int,
                         verbose: bool) -> list[np.ndarray]:
    """
    Return a list of ORI coordinates to write.
    • If not sphere: [center]
    • If sphere & surface: [center] + (sampling_size-1) points on surface of radius R
    • If sphere & volume:  [center] + (sampling_size-1) points in volume within radius R
    The list length will be exactly 1 or sampling_size (deterministic).
    """
    if not sphere:
        vprint(verbose, "[plan] Single-ORI mode (default).")
        return [center.copy()]

    if sampling_size < 1:
        raise ValueError("`--sampling_size` must be ≥ 1 for sphere mode.")
    if radius <= 0.0:
        raise ValueError("`--radius` must be > 0 for sphere mode.")
    if mode not in {"surface", "volume"}:
        raise ValueError("`--mode` must be either 'surface' or 'volume'.")

    vprint(verbose, f"[plan] Sphere mode: mode={mode}, radius={radius}, sampling_size={sampling_size}")
    coords = [center.copy()]  # first is always the center

    m = sampling_size - 1
    if m == 0:
        vprint(verbose, "[plan] sampling_size=1 → only the center.")
        return coords

    if mode == "surface":
        surf = fibonacci_surface_points(m) * float(radius)
        # Order: keep Fibonacci order as-is (already well-distributed and deterministic).
        for p in surf:
            coords.append(center + p)
    else:  # volume
        # Deterministic interior points excluding the center (center already appended)
        vol = volume_points_fibonacci_shells(m, radius=radius)
        for p in vol:
            coords.append(center + p)

    vprint(verbose, f"[plan] Planned {len(coords)} coordinates (exact).")
    return coords

def generate_outputs(input_pdb: Path,
                     output_dir: Path,
                     coords: list[np.ndarray],
                     base_name: str,
                     chain: str,
                     serial_start: int,
                     resseq_start: int,
                     verbose: bool):
    """
    Write one file per coordinate: <basename>_ORI_<idx>.pdb
    Indexing starts at 01 for nicer human readability.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, xyz in enumerate(coords, start=1):
        out = output_dir / f"{base_name}_ORI_{i:02d}.pdb"
        write_appended_pdb(
            base_pdb=input_pdb,
            out_path=out,
            ori_xyz_list=[(float(xyz[0]), float(xyz[1]), float(xyz[2]))],
            chain=chain,
            serial_start=serial_start,
            resseq_start=resseq_start,
            verbose=verbose,
        )

# ----------------------------------- CLI ----------------------------------- #

def main():
    p = argparse.ArgumentParser(
        description="Deterministic ORI token placement: single token (default) or a sphere (surface/volume)."
    )
    p.add_argument("--input_pdb", required=True, help="Input PDB file (content will be preserved; ORI appended).")
    p.add_argument("--output_dir", required=True, help="Directory to write outputs.")
    p.add_argument("--center", type=str, default=None,
                   help='Center coordinate as "x y z". If omitted, attempts to read the first ORI from the input PDB.')
    p.add_argument("--chain", default="X", help="Chain ID to assign to ORI records (default: X).")
    p.add_argument("--serial_start", type=int, default=999, help="Starting atom serial number for ORI (default: 1).")
    p.add_argument("--resseq_start", type=int, default=1, help="Starting residue number for ORI (default: 1).")
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging.")

    # Sphere options (opt-in)
    p.add_argument("--sphere", action="store_true", help="Enable sphere mode (otherwise write a single ORI).")
    p.add_argument("--mode", choices=["surface", "volume"], default="surface",
                   help="Sphere distribution mode (surface or volume). Default: surface.")
    p.add_argument("--radius", type=float, default=5.0, help="Sphere radius in Å (default: 5.0).")
    p.add_argument("--sampling_size", type=int, default=8,
                   help="Total number of ORIs to generate in sphere mode (exact). Default: 8.")

    args = p.parse_args()

    input_pdb = Path(args.input_pdb)
    output_dir = Path(args.output_dir)
    if not input_pdb.exists():
        print(f"[ERROR] Input PDB not found: {input_pdb}", file=sys.stderr)
        sys.exit(1)

    # Determine center coordinate
    center = parse_center_from_cli(args.center)
    if center is None:
        center = find_first_ori_in_pdb(input_pdb, verbose=args.verbose)
    if center is None:
        print("[ERROR] No center coordinate provided and no ORI found in the input PDB.\n"
              "        Provide --center \"x y z\" or include an ORI atom in the input.", file=sys.stderr)
        sys.exit(1)

    # Plan coordinates (exact count)
    coords = plan_ori_coordinates(
        center=center,
        sphere=args.sphere,
        mode=args.mode,
        radius=args.radius,
        sampling_size=args.sampling_size,
        verbose=args.verbose
    )

    # Base name for outputs
    base_name = input_pdb.stem

    # Generate outputs
    generate_outputs(
        input_pdb=input_pdb,
        output_dir=output_dir,
        coords=coords,
        base_name=base_name,
        chain=args.chain,
        serial_start=args.serial_start,
        resseq_start=args.resseq_start,
        verbose=args.verbose
    )

    # ----------------------------- Summary ----------------------------- #
    print("\n================ ORI GENERATION SUMMARY ================")
    print(f"Input PDB:         {input_pdb}")
    print(f"Output directory:  {output_dir}")
    if args.sphere:
        print(f"Mode:              SPHERE ({args.mode})")
        print(f"Radius (Å):        {args.radius}")
        print(f"Sampling size:     {args.sampling_size}  (exact)")
        print(f"First ORI:         Center at ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
    else:
        print("Mode:              SINGLE ORI (default)")
        print(f"ORI coordinate:    ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
    print(f"Chain:             {args.chain}")
    print(f"Serial start:      {args.serial_start}")
    print(f"ResSeq start:      {args.resseq_start}")
    print(f"Files written:     {len(coords)}")
    print("========================================================\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
