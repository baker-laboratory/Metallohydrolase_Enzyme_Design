#!/usr/bin/env python3
"""External tool locations for the ligand/theozyme scripts.

``Scripts/theozyme_and_ligand_handling/ligands_to_params__UNIFIED.py`` is kept
byte-identical to its upstream version so it can be re-synced easily. It expects
a ``repo_paths`` module somewhere above it on the filesystem, which it finds by
walking up from its own location -- landing here, in ``Scripts/``.

This module supplies only the three values that script needs, resolved
portably:

    OBABEL                 the Open Babel executable
    MOLFILE_TO_PARAMS      Rosetta's molfile_to_params.py
    ROSETTA_RESIDUE_TYPES  the fa_standard residue_types.txt

Each is resolved as: environment variable -> automatic discovery -> a clear
error at the point of use.

Environment variables
---------------------
OBABEL                 explicit path to ``obabel`` (``ZINC_HYDRO_OBABEL`` also honoured)
MOLFILE_TO_PARAMS      explicit path to ``molfile_to_params.py``
ROSETTA_RESIDUE_TYPES  explicit path to ``residue_types.txt``
ROSETTA                root of a Rosetta checkout; the two Rosetta paths above
                       are derived from it when they are not set individually

A NOTE ON ROSETTA: ``molfile_to_params.py`` and the residue-type database are
part of Rosetta, which carries its own license, so they are not redistributed
here. Point ``$ROSETTA`` at your Rosetta installation, or set the two variables
directly. Everything else in this repository runs without them; they are needed
only when generating new ligand ``.params`` files.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

__all__ = ["OBABEL", "MOLFILE_TO_PARAMS", "ROSETTA_RESIDUE_TYPES", "ROSETTA",
           "LAB_SCRIPTS", "ENZYME_DESIGN_DIR"]


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    return value if value else default


class _Missing(str):
    """A path placeholder that explains itself if anything tries to use it.

    Resolution failures should surface where the tool is actually needed, with
    instructions -- not as an import-time crash in code paths that never touch
    Rosetta.
    """

    def __new__(cls, value: str, message: str):
        obj = super().__new__(cls, value)
        obj._message = message  # type: ignore[attr-defined]
        return obj

    def explain(self) -> str:
        return self._message  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Open Babel
# ---------------------------------------------------------------------------

def _find_obabel() -> str:
    for var in ("OBABEL", "ZINC_HYDRO_OBABEL"):
        explicit = _env(var)
        if explicit:
            return explicit
    found = shutil.which("obabel")
    if found:
        return found
    return _Missing(
        "obabel",
        "Open Babel ('obabel') is not on $PATH.\n"
        "  It is included in Environment/zinc_hydro.yml, so activating that\n"
        "  environment (conda activate zinc_hydro) is usually the fix.\n"
        "  Otherwise set $OBABEL to the executable.",
    )


# ---------------------------------------------------------------------------
# Rosetta
# ---------------------------------------------------------------------------

ROSETTA = _env("ROSETTA")


def _pyrosetta_root() -> Optional[Path]:
    """Locate an installed PyRosetta package without importing it."""
    try:
        import importlib.util

        spec = importlib.util.find_spec("pyrosetta")
        if spec and spec.origin:
            return Path(spec.origin).resolve().parent
    except Exception:  # noqa: BLE001 - discovery must never raise
        pass
    return None


def _find_molfile_to_params() -> str:
    explicit = _env("MOLFILE_TO_PARAMS")
    if explicit:
        return explicit

    candidates = []
    if ROSETTA:
        candidates.append(Path(ROSETTA) / "source/scripts/python/public/molfile_to_params.py")
    pyr = _pyrosetta_root()
    if pyr:
        candidates += [
            pyr / "toolbox/molfile_to_params.py",
            pyr / "database/../scripts/python/public/molfile_to_params.py",
        ]
    on_path = shutil.which("molfile_to_params.py")
    if on_path:
        candidates.append(Path(on_path))

    for c in candidates:
        if c.is_file():
            return str(c)

    return _Missing(
        "molfile_to_params.py",
        "Could not locate Rosetta's molfile_to_params.py.\n"
        "  It is part of Rosetta and is not redistributed with this repository.\n"
        "  Set one of:\n"
        "      export ROSETTA=/path/to/rosetta/main\n"
        "      export MOLFILE_TO_PARAMS=/path/to/molfile_to_params.py\n"
        "  It is needed only to generate new ligand .params files.",
    )


def _find_residue_types() -> str:
    explicit = _env("ROSETTA_RESIDUE_TYPES")
    if explicit:
        return explicit

    rel = "chemical/residue_type_sets/fa_standard/residue_types.txt"
    candidates = []
    if ROSETTA:
        candidates.append(Path(ROSETTA) / "database" / rel)
    pyr = _pyrosetta_root()
    if pyr:
        candidates.append(pyr / "database" / rel)

    for c in candidates:
        if c.is_file():
            return str(c)

    return _Missing(
        "residue_types.txt",
        "Could not locate Rosetta's fa_standard residue_types.txt.\n"
        "  Set $ROSETTA_RESIDUE_TYPES, or $ROSETTA to your Rosetta root.\n"
        "  A PyRosetta installation also ships this database.",
    )


OBABEL = _find_obabel()
MOLFILE_TO_PARAMS = _find_molfile_to_params()
ROSETTA_RESIDUE_TYPES = _find_residue_types()


# Optional extras some vendored scripts reference. They are lab locations with
# no public equivalent; scripts that use them guard for absence.
LAB_SCRIPTS = _env("LAB_SCRIPTS", "")
ENZYME_DESIGN_DIR = _env("ENZYME_DESIGN_DIR", "")


def report() -> None:
    """Print what resolved and what did not. Run this module directly."""
    print("### LIGAND TOOLING PATHS ###")
    for name in ("OBABEL", "MOLFILE_TO_PARAMS", "ROSETTA_RESIDUE_TYPES"):
        value = globals()[name]
        if isinstance(value, _Missing):
            print(f"  !! {name:22s} : NOT FOUND")
            for line in value.explain().splitlines():
                print(f"     {line}")
        else:
            print(f"     {name:22s} : {value}")


if __name__ == "__main__":
    report()
