#!/usr/bin/env python3
"""Run LigandMPNN with exact preservation of fixed-residue side chains.

Thin wrapper around ``Software/fastmpnndesign/lib/LigandMPNN/run.py``. It takes
the same arguments and behaves identically, except that with
``--repack_everything 0`` the side chains of fixed residues come back with the
coordinates they went in with, rather than rebuilt at idealized geometry.

WHY
---
LigandMPNN's packer preserves the *torsions* of fixed residues but reconstructs
all coordinates through ``frames_and_literature_positions_to_atom14_pos``, which
places atoms at literature geometry. A catalytic residue taken from a crystal
structure or a quantum-chemistry theozyme therefore comes back subtly moved.
For enzyme design, where the active-site geometry is the point, that matters.

HOW
---
``ligandmpnn_patched/sc_utils.py`` is a copy of the upstream module with the
exact input coordinates written back over the reconstructed ones for fixed
residues, at both places the reconstruction happens. This wrapper puts that
directory ahead of the LigandMPNN checkout on ``sys.path`` so ``import
sc_utils`` resolves to the patched copy, then hands over to the stock
``run.py``. Nothing in the submodule is modified.

USAGE
-----
Identical to run.py -- swap the script path:

    python Scripts/mpnn_relevant_utils/run_ligandmpnn.py \\
        --model_type ligand_mpnn \\
        --pdb_path input.pdb --out_folder out/ \\
        --pack_side_chains 1 --repack_everything 0 \\
        --fixed_residues_multi fixed.json

Set ``--ligandmpnn_dir`` if the checkout is somewhere unusual; otherwise it is
found relative to this file. Verify the patch took effect with
``--verify_fixed_residues``, which reports the largest per-atom displacement of
any fixed residue between input and output.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PATCHED = _HERE / "ligandmpnn_patched"


def _default_ligandmpnn_dir() -> str:
    """Locate the bundled LigandMPNN checkout."""
    env = os.environ.get("LIGANDMPNN_DIR")
    if env:
        return env
    for anc in _HERE.parents:
        cand = anc / "Software" / "fastmpnndesign" / "lib" / "LigandMPNN"
        if (cand / "run.py").is_file():
            return str(cand)
    return ""


def main() -> None:
    # Pull off our own flags; everything else passes through untouched.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ligandmpnn_dir", default=_default_ligandmpnn_dir())
    parser.add_argument("--no_patch", action="store_true",
                        help="Run stock LigandMPNN without the fixed-residue fix.")
    ours, passthrough = parser.parse_known_args()

    lig_dir = Path(ours.ligandmpnn_dir) if ours.ligandmpnn_dir else None
    if not lig_dir or not (lig_dir / "run.py").is_file():
        sys.exit(
            "Could not find LigandMPNN.\n"
            "  Initialize the bundled checkout:\n"
            "      git submodule update --init --recursive Software/fastmpnndesign\n"
            "  or set --ligandmpnn_dir / $LIGANDMPNN_DIR."
        )

    # The patched sc_utils must come first so `import sc_utils` picks it up.
    if not ours.no_patch:
        sys.path.insert(0, str(_PATCHED))
        print(f"### fixed-residue side-chain preservation ACTIVE ({_PATCHED}) ###")
    else:
        print("### running stock LigandMPNN (--no_patch) ###")
    sys.path.insert(1 if not ours.no_patch else 0, str(lig_dir))

    # run.py resolves some of its own data relative to its location, so we chdir
    # there -- which means any relative path the caller gave would break. Make
    # them absolute against the real working directory first.
    cwd = Path.cwd()
    resolved, expect_path = [], False
    for tok in passthrough:
        if expect_path and not tok.startswith("-"):
            cand = Path(tok)
            tok = str((cwd / cand).resolve()) if not cand.is_absolute() else tok
            expect_path = False
        elif tok.startswith("--") and any(
            k in tok for k in ("path", "folder", "dir", "json", "checkpoint", "multi")
        ):
            expect_path = True
        else:
            expect_path = False
        resolved.append(tok)

    os.chdir(lig_dir)
    sys.argv = [str(lig_dir / "run.py")] + resolved
    runpy.run_path(str(lig_dir / "run.py"), run_name="__main__")


if __name__ == "__main__":
    main()
