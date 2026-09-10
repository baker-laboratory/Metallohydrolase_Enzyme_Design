#!/usr/bin/env python3
"""
Environment resolution for the Metallohydrolase enzyme design repository.

WHAT THIS DOES
--------------
Resolves the external tools this repository invokes -- the python interpreter
used for helper scripts, Open Babel, and the repository root -- so that nothing
depends on a machine-specific absolute path.

Each is resolved in the same order:

    1. an explicit value passed on the command line / in a notebook cell
    2. an environment variable
    3. whatever is active right now (``sys.executable``, ``$PATH``)

No container is required: the ``zinc_hydro`` conda environment already provides
PyRosetta, Open Babel and the rest. A container remains available as an option
for clusters that prefer one. Resolution happens up front, so a bad path is
reported immediately with the fix, rather than producing a command that fails
later for an unrelated-looking reason.

USAGE
-----
From a script::

    from env_config import resolve_runner, resolve_obabel
    runner = resolve_runner(args.apptainer)      # -> ['/path/to/python'] or ['/path/to.sif']
    subprocess.run(runner + [str(helper), '-input_pdb', pdb], check=True)

From a notebook, to see what will be used before running anything::

    import env_config; env_config.print_environment_report()

ENVIRONMENT VARIABLES
---------------------
ZINC_HYDRO_SIF      Path to an Apptainer/Singularity image whose runscript is a
                    python interpreter. Build one with
                    ``apptainer build zinc_hydro.sif Environment/zinc_hydro.def``.
                    Unset (the normal case) means "use the active interpreter".
ZINC_HYDRO_OBABEL   Path to the ``obabel`` executable. Unset means "find it on
                    $PATH", which is correct inside the conda environment.
ZINC_HYDRO_REPO     Repository root. Only needed if auto-detection fails, which
                    happens when a notebook is executed from an unrelated cwd.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Sequence

__all__ = [
    "RepoLayoutError",
    "ToolNotFoundError",
    "find_repo_root",
    "resolve_runner",
    "resolve_obabel",
    "environment_report",
    "print_environment_report",
]

# Files/directories that together identify the repository root unambiguously.
_ROOT_MARKERS: Sequence[str] = ("Scripts", "Software", "Environment", "LICENSE")


class RepoLayoutError(RuntimeError):
    """Raised when the repository root cannot be located."""


class ToolNotFoundError(FileNotFoundError):
    """Raised when a required external tool cannot be resolved."""


# ---------------------------------------------------------------------------
# repository root
# ---------------------------------------------------------------------------

def find_repo_root(start: Optional[os.PathLike | str] = None) -> Path:
    """Locate the repository root without hardcoding anyone's home directory.

    Resolution order:
      1. ``$ZINC_HYDRO_REPO`` if set,
      2. this file's own location (``<repo>/Scripts/env_config.py``),
      3. an upward walk from ``start`` (default: the current directory).

    Raises
    ------
    RepoLayoutError
        If no ancestor directory looks like the repository.
    """
    env_root = os.environ.get("ZINC_HYDRO_REPO")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if not _looks_like_repo(root):
            raise RepoLayoutError(
                f"$ZINC_HYDRO_REPO is set to {root!s}, but that directory does not "
                f"look like this repository (expected to find: {', '.join(_ROOT_MARKERS)})."
            )
        return root

    # This file lives at <repo>/Scripts/env_config.py, so the parent's parent is
    # the root. This is the path that works even when the notebook's cwd has
    # been changed, which every tutorial notebook does.
    here = Path(__file__).resolve().parent.parent
    if _looks_like_repo(here):
        return here

    probe = Path(start).expanduser().resolve() if start else Path.cwd().resolve()
    for candidate in (probe, *probe.parents):
        if _looks_like_repo(candidate):
            return candidate

    raise RepoLayoutError(
        f"Could not locate the repository root starting from {probe!s}. "
        f"Set ZINC_HYDRO_REPO to the directory containing {', '.join(_ROOT_MARKERS)}."
    )


def _looks_like_repo(path: Path) -> bool:
    return all((path / marker).exists() for marker in _ROOT_MARKERS)


# ---------------------------------------------------------------------------
# python / container runner
# ---------------------------------------------------------------------------

def resolve_runner(explicit: Optional[str] = None) -> List[str]:
    """Return the argv prefix used to execute a helper python script.

    Parameters
    ----------
    explicit
        A value supplied by the caller, typically from a CLI flag or a notebook
        variable. May be:

        * ``None`` or ``""``      -- use the active interpreter (the default,
          and the right answer inside the ``zinc_hydro`` conda environment);
        * ``"python"``            -- same as above, resolved to
          ``sys.executable`` so subprocesses inherit this environment rather
          than whatever ``python`` happens to be first on ``$PATH``;
        * a path to a ``.sif``    -- an Apptainer image whose runscript is a
          python interpreter (built from ``Environment/zinc_hydro.def``);
        * any other path          -- used verbatim, e.g. another interpreter.

    Returns
    -------
    list of str
        An argv prefix. Append the script path and its arguments, then hand the
        whole list to :func:`subprocess.run`. Always a list, so callers never
        have to worry about quoting.

    Raises
    ------
    ToolNotFoundError
        If an explicit path was given but does not exist. This is the failure
        the old code turned into a silent, misleading downstream error.
    """
    candidate = explicit if explicit else os.environ.get("ZINC_HYDRO_SIF")

    if not candidate or candidate in {"python", "python3", "sys.executable"}:
        return [sys.executable]

    path = Path(candidate).expanduser()

    # A bare command name (not a path) -- resolve it on $PATH.
    if not path.is_absolute() and os.sep not in candidate:
        found = shutil.which(candidate)
        if found:
            return [found]
        raise ToolNotFoundError(
            f"Could not find {candidate!r} on $PATH.\n"
            f"  Either activate the conda environment "
            f"(conda activate zinc_hydro) and leave this unset, or pass an "
            f"absolute path."
        )

    if not path.exists():
        raise ToolNotFoundError(
            f"Container or interpreter not found: {path}\n"
            f"\n"
            f"  This repository does NOT require a container. The simplest fix is\n"
            f"  to leave this unset and run inside the conda environment:\n"
            f"\n"
            f"      conda env create -f Environment/zinc_hydro.yml\n"
            f"      conda activate zinc_hydro\n"
            f"\n"
            f"  If you do want a container, build one -- do not look for ours,\n"
            f"  the old lab paths (crispy.sif, allmighty.sif) are gone:\n"
            f"\n"
            f"      apptainer build zinc_hydro.sif Environment/zinc_hydro.def\n"
            f"      export ZINC_HYDRO_SIF=$PWD/zinc_hydro.sif\n"
        )

    if not os.access(path, os.X_OK):
        raise ToolNotFoundError(
            f"{path} exists but is not executable.\n"
            f"  Fix with:  chmod +x {path}\n"
            f"  Or invoke it explicitly:  apptainer exec {path} python ..."
        )

    return [str(path.resolve())]


# ---------------------------------------------------------------------------
# open babel
# ---------------------------------------------------------------------------

def resolve_obabel(explicit: Optional[str] = None) -> str:
    """Return a usable path to the ``obabel`` executable.

    Order: ``explicit`` -> ``$ZINC_HYDRO_OBABEL`` -> ``$PATH``.

    Raises
    ------
    ToolNotFoundError
        With installation instructions, rather than letting a subprocess fail
        with a bare ``FileNotFoundError``.
    """
    candidate = explicit if explicit else os.environ.get("ZINC_HYDRO_OBABEL")

    if candidate and candidate not in {"obabel"}:
        path = Path(candidate).expanduser()
        if path.exists() and os.access(path, os.X_OK):
            return str(path.resolve())
        found = shutil.which(candidate)
        if found:
            return found
        raise ToolNotFoundError(
            f"Open Babel not found at {candidate!r}.\n"
            f"  Leave this unset to use the copy in your conda environment, or "
            f"install one:  conda install -c conda-forge openbabel"
        )

    found = shutil.which("obabel")
    if found:
        return found

    raise ToolNotFoundError(
        "Open Babel ('obabel') is not on $PATH.\n"
        "  Install it with:  conda install -c conda-forge openbabel\n"
        "  It is already included in Environment/zinc_hydro.yml, so activating\n"
        "  that environment (conda activate zinc_hydro) is usually the fix.\n"
        "  Upstream project: https://github.com/openbabel/openbabel"
    )


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def environment_report() -> dict:
    """Collect what this module resolves, without raising on missing tools.

    Every value is either the resolved string or an ``"ERROR: ..."`` message, so
    a notebook preflight cell can show the whole picture at once instead of
    dying on the first problem.
    """

    def _try(fn, *args):
        try:
            result = fn(*args)
            return " ".join(result) if isinstance(result, list) else str(result)
        except Exception as exc:  # noqa: BLE001 - report, never raise
            return f"ERROR: {exc.__class__.__name__}: {exc}"

    report = {
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "(none active)"),
        "repo_root": _try(find_repo_root),
        "runner": _try(resolve_runner, None),
        "obabel": _try(resolve_obabel, None),
        "ZINC_HYDRO_SIF": os.environ.get("ZINC_HYDRO_SIF", "(unset -- using active interpreter)"),
        "ZINC_HYDRO_OBABEL": os.environ.get("ZINC_HYDRO_OBABEL", "(unset -- using $PATH)"),
        "ZINC_HYDRO_REPO": os.environ.get("ZINC_HYDRO_REPO", "(unset -- auto-detected)"),
    }

    for module in ("numpy", "pandas", "scipy", "matplotlib", "openpyxl", "pyrosetta"):
        try:
            mod = __import__(module)
            report[f"pkg:{module}"] = getattr(mod, "__version__", "installed")
        except Exception as exc:  # noqa: BLE001
            report[f"pkg:{module}"] = f"NOT INSTALLED ({exc.__class__.__name__})"

    return report


def print_environment_report() -> None:
    """Print :func:`environment_report` as an aligned block.

    Call this from the first cell of any notebook to see, before running
    anything, exactly which interpreter and tools will be used.
    """
    report = environment_report()
    width = max(len(k) for k in report)
    print("### ENVIRONMENT REPORT ###")
    for key, value in report.items():
        marker = "  !!" if str(value).startswith(("ERROR", "NOT INSTALLED")) else "   "
        print(f"{marker} {key.ljust(width)} : {value}")

    problems = [k for k, v in report.items() if str(v).startswith(("ERROR", "NOT INSTALLED"))]
    if problems:
        print(
            f"\n{len(problems)} problem(s) above. PyRosetta is only needed for the design "
            f"pipelines;\nthe wet lab analysis notebook needs only numpy/pandas/scipy/"
            f"matplotlib/openpyxl.\nSee Environment/README.md."
        )
    else:
        print("\nAll good.")


if __name__ == "__main__":
    print_environment_report()
