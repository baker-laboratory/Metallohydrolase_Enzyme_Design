# ---------------- General_NoteBook_Functions.py ----------------

### ---- IMPORTS ----
from pathlib import Path
import pandas as pd
import textwrap
import os

### ---- FUNCTIONS ----
def setup_directories(base_dir, dirs_list, export_globals=False, globals_dict=None):
    """
    Create each subdirectory in dirs_list under base_dir and export "<name>_dir"
    globals that ALWAYS end with a trailing path separator (e.g., '/').
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    exported = {}
    for d in dirs_list:
        # Normalize the subdir token (allow 'foo', 'foo/', '/foo/')
        d_norm = str(d).strip().strip("/")

        sub = base / d_norm
        sub.mkdir(parents=True, exist_ok=True)

        # Build var name: 'theozymes/' -> 'theozymes_dir', 'rfd2/configs/' -> 'rfd2configs_dir'
        var_name = d_norm.replace("/", "") + "_dir"

        # Ensure the EXPORTED path string ends with a trailing separator
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

def submit_array_job(commands, time, cores, job_name, memory, submit_file, logs_dir, num_jobs, cmds_per_job, queue):
    """
    Create a Slurm array submit script that executes lines from `commands`
    in chunks of size `cmds_per_job` across `num_jobs` array tasks.

    Parameters
    ----------
    commands : str      # path to a file; each line is a shell command
    time     : str      # e.g. "24:00:00"
    cores    : int
    job_name : str
    memory   : str      # e.g. "16G"
    submit_file : str   # path to write the submit script
    logs_dir : str      # directory prefix for stdout/stderr logs
    num_jobs : int      # array size
    cmds_per_job : int  # lines per array task
    queue : str         # "cpu", "gpu", or "gpu-bf"
    """

    # Queue-specific SBATCH extras
    queue_extras = {
        "cpu":   "",
        "gpu":   "\n#SBATCH --gres=gpu:a4000:1",
        "gpu-bf":"\n#SBATCH --constraint='A100|A4000|A5000|A6000|Titan|Quadro'\n#SBATCH --gres=gpu:1",
    }
    if queue not in queue_extras:
        raise ValueError(f"Unknown queue '{queue}'. Expected one of {list(queue_extras)}")

    extras = queue_extras[queue]

    submit_txt = textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH -J {job_name}
        #SBATCH -p {queue}
        #SBATCH -c {cores}{extras}
        #SBATCH --mem={memory}
        #SBATCH -t {time}
        #SBATCH -o {logs_dir}{job_name}_%a.stdout
        #SBATCH -e {logs_dir}{job_name}_%a.stderr
        #SBATCH -a 1-{num_jobs}

        PER_TASK={cmds_per_job}
        START_NUM=$(( ($SLURM_ARRAY_TASK_ID - 1) * $PER_TASK + 1 ))
        END_NUM=$(( $SLURM_ARRAY_TASK_ID * $PER_TASK ))
        echo This is task $SLURM_ARRAY_TASK_ID, which will do runs $START_NUM to $END_NUM
        for (( run=$START_NUM; run<=END_NUM; run++ )); do
          echo This is SLURM task $SLURM_ARRAY_TASK_ID, run number $run
          CMD=$(sed -n "${{run}}p" {commands}
        )
          echo "${{CMD}}" | bash
        done
    """)

    with open(submit_file, "w") as f:
        f.write(submit_txt)

    print(f"### SUBMIT THIS TO SLURM ###\nsbatch {submit_file}")
