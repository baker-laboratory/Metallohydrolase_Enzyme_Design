# ─────────────────────────────────────────────────────────────────────
#  notebook_core.py
#  Central utility module for computational protein design notebooks.
#  Location: Scripts/notebook_functions/
# ─────────────────────────────────────────────────────────────────────

import os
import textwrap
from datetime import datetime
from pathlib import Path

import pandas as pd


# ═════════════════════════════════════════════════════════════════════
#  COLOR PALETTE (Sam Pellock)
# ═════════════════════════════════════════════════════════════════════

good_teal   = ( 40/255, 176/255, 193/255)
good_yellow = (250/255, 199/255,  44/255)
good_green  = (170/255, 195/255,  47/255)
good_pink   = (236/255, 114/255, 164/255)
good_peach  = (249/255, 145/255, 120/255)
good_gray   = (220/255, 220/255, 220/255)
good_blue   = ( 68/255, 153/255, 231/255)
good_red    = (228/255,  74/255,  62/255)


# ═════════════════════════════════════════════════════════════════════
#  NOTEBOOK UTILITY FUNCTIONS
# ═════════════════════════════════════════════════════════════════════

def _strip_mnt(path_str):
    """
    Strip a leading '/mnt' from an absolute path if doing so yields an
    existing directory. This handles JupyterHub mounts where '/home/user'
    is exposed as '/mnt/home/user'.
    """
    if isinstance(path_str, str) and path_str.startswith('/mnt/'):
        stripped = path_str[4:]  # drop '/mnt'
        if Path(stripped).exists():
            return stripped
    return path_str


def resolve_working_dir(override=None, strip_mnt=True):
    """
    Return the working directory path (always trailing '/').
    Uses override if provided, otherwise auto-detects the notebook's directory
    by checking (in order): __vsc_ipynb_file__ (VS Code), IPython %notebook,
    or IPython startup dir (_dh[0]).

    strip_mnt: if True (default), strip a leading '/mnt' so that JupyterHub
    paths like '/mnt/home/user/...' collapse to '/home/user/...'.
    """
    if override:
        d = str(Path(override).resolve())
    else:
        d = _find_notebook_dir()
    if strip_mnt:
        d = _strip_mnt(d)
    return d if d.endswith('/') else d + '/'


def _find_notebook_dir():
    """
    Best-effort detection of the directory containing the running .ipynb file.
    Falls back to cwd if no Jupyter/IPython context is found.
    """
    # VS Code sets this global
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None:
            vsc = ip.user_ns.get('__vsc_ipynb_file__')
            if vsc:
                return str(Path(vsc).resolve().parent)
    except Exception:
        pass

    # JupyterLab / classic notebook: IPython startup directory
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None and hasattr(ip, 'user_ns') and '_dh' in ip.user_ns:
            return str(Path(ip.user_ns['_dh'][0]).resolve())
    except Exception:
        pass

    # Fallback: cwd
    return str(Path().resolve())


def resolve_output_dir(working_dir, override=None, strip_mnt=True):
    """
    Return the output directory path (always trailing '/').
    Uses override if provided, otherwise defaults to working_dir/output/.

    strip_mnt: if True (default), strip a leading '/mnt' so that JupyterHub
    paths like '/mnt/home/user/...' collapse to '/home/user/...'.
    """
    if override:
        d = str(Path(override).resolve())
    else:
        d = f'{working_dir}output'
    if strip_mnt:
        d = _strip_mnt(d)
    return d if d.endswith('/') else d + '/'


def setup_directories(base_dir, dirs_list, export_globals=False, globals_dict=None,
                      uppercase_globals=True):
    """
    Create each subdirectory in dirs_list under base_dir and export path globals
    that ALWAYS end with a trailing path separator (e.g., '/').

    Directory names on disk keep their original case as given in dirs_list.
    Global variable names are UPPER_CASE by default (uppercase_globals=True),
    e.g. 'important_dfs' -> dir on disk: important_dfs/, variable: IMPORTANT_DFS_DIR
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    exported = {}
    for d in dirs_list:
        d_norm = str(d).strip().strip("/")

        sub = base / d_norm
        sub.mkdir(parents=True, exist_ok=True)

        var_name = d_norm.replace("/", "") + "_dir"
        if uppercase_globals:
            var_name = var_name.upper()

        path_str = str(sub)
        if not path_str.endswith(os.sep):
            path_str += os.sep

        exported[var_name] = path_str
        if export_globals and globals_dict is not None:
            globals_dict[var_name] = path_str

    return exported


def set_pandas_display(all_on=True):
    """
    Convenience toggles for Pandas display options.
    """
    if all_on:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_colwidth', None)
    else:
        pd.reset_option('display.max_columns')
        pd.reset_option('display.max_rows')
        pd.reset_option('display.max_colwidth')


def print_initialization(working_dir, output_dir, project_name=None,
                         obabel_path=None, globals_dict=None, preview=True):
    """
    Print a summary banner after notebook initialization.
    If globals_dict is provided, also lists exported *_DIR subdirectory variables.
    If preview=True (default), only shows the first 3 with a count of the rest.
    Set preview=False to print all.
    """
    print(f"{'─'*70}")
    if project_name:
        print(f"  PROJECT: {project_name}")
    print(f"  INITIALIZED: {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}")
    print(f"{'─'*70}")
    print(f"  WORKING_DIR = {working_dir}")
    print(f"  OUTPUT_DIR  = {output_dir}")
    if obabel_path and not Path(obabel_path).exists():
        print(f"  ⚠ obabel not found at: {obabel_path}")
    if globals_dict is not None:
        subdir_vars = sorted(
            k for k in globals_dict
            if k.endswith('_DIR') and k not in (
                'WORKING_DIR', 'OUTPUT_DIR', 'HOME_DIR', 'GENERAL_DIR',
                'THEOZYMES_DIR', 'PARAMS_DIR', 'CST_DIR',
                'SPECIAL_SCRIPTS_DIR', 'GIT_DIR',
            ) and isinstance(globals_dict[k], str)
        )
        if subdir_vars:
            print(f"{'─'*70}")
            show = subdir_vars[:3] if preview else subdir_vars
            print(f"  EXPORTED SUBDIRS ({len(subdir_vars)} total):")
            for k in show:
                print(f"    {k} = {globals_dict[k]}")
            if preview and len(subdir_vars) > 3:
                print(f"    ... and {len(subdir_vars) - 3} more (set preview=False to show all)")
    print(f"{'─'*70}")



# ═════════════════════════════════════════════════════════════════════
#  SLURM SUBMISSION FUNCTIONS — re-exported from slurm_submission.py
#
#  All SLURM helpers (submit_array_job, submit_array_job_legacy,
#  submit_cpu, make_af2_submit_file, etc.) live in slurm_submission.py
#  for clean separation.  They are re-exported here so existing notebook
#  code that uses `import notebook_core as nb` and calls
#  `nb.submit_array_job(...)` keeps working unchanged.
#
#  See slurm_submission.py for the function definitions and docstrings.
# ═════════════════════════════════════════════════════════════════════

from slurm_submission import (
    _walltime_to_seconds,
    submit_cpu,
    submit_array_job,
    submit_array_job_legacy,
    make_af2_submit_file,
    make_af2_submit_file_with_mem_and_optional_gpu,
)
