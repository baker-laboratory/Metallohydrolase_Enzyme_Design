# ─────────────────────────────────────────────────────────────────────
#  slurm_submission.py
#  SLURM helpers — submission scripts for single jobs, array jobs (new),
#  the legacy array-job variant, and AF2-specific submit-script writers.
#
#  This module is imported by notebook_core.py with `from slurm_submission
#  import *`, so all functions are still accessible as `nb.<function>(...)`
#  in notebook code:
#
#      import notebook_core as nb
#      nb.submit_array_job(...)         # new flexible array submit
#      nb.submit_array_job_legacy(...)  # frozen pre-2026-05 version
#      nb.submit_cpu(...)               # single-command CPU submit
#      nb.make_af2_submit_file(...)     # AF2 array submit
#      nb.make_af2_submit_file_with_mem_and_optional_gpu(...)
#
#  All SLURM-related functions live here so they can be edited / audited
#  independently from the general notebook utilities in notebook_core.py.
# ─────────────────────────────────────────────────────────────────────

import os
import shutil
import warnings


# ═════════════════════════════════════════════════════════════════════
#  SLURM SUBMISSION FUNCTIONS
# ═════════════════════════════════════════════════════════════════════

def submit_cpu(command, time, cores, job_name, memory, submit_file):
    """
    Write a single-command SLURM CPU submit script.
    """
    submit_txt = f"""\
#!/bin/bash
#SBATCH -p cpu
#SBATCH -c {cores}
#SBATCH -t {time}
#SBATCH -J {job_name}
#SBATCH --mem={memory}
#SBATCH -o {job_name}.stdout
#SBATCH -e {job_name}.stderr

{command}
"""

    with open(submit_file, 'w') as f:
        f.write(submit_txt)

    print(f'submit this: \n{submit_file}')


def _walltime_to_seconds(walltime):
    """Parse a SLURM walltime spec ('MM', 'MM:SS', 'HH:MM:SS', 'D-HH', 'D-HH:MM',
    'D-HH:MM:SS') to total seconds.  Returns None if unparseable."""
    try:
        s = str(walltime).strip()
        if '-' in s:
            d_str, rest = s.split('-', 1)
            days = int(d_str)
            parts = rest.split(':') if rest else ['0']
            # After dash: HH | HH:MM | HH:MM:SS
            if   len(parts) == 1: h, m, sec = int(parts[0]), 0, 0
            elif len(parts) == 2: h, m, sec = int(parts[0]), int(parts[1]), 0
            elif len(parts) == 3: h, m, sec = int(parts[0]), int(parts[1]), int(parts[2])
            else: return None
        else:
            days = 0
            parts = s.split(':')
            # No dash: MM | MM:SS | HH:MM:SS
            if   len(parts) == 1: h, m, sec = 0, int(parts[0]), 0
            elif len(parts) == 2: h, m, sec = 0, int(parts[0]), int(parts[1])
            elif len(parts) == 3: h, m, sec = int(parts[0]), int(parts[1]), int(parts[2])
            else: return None
        return days * 86400 + h * 3600 + m * 60 + sec
    except (ValueError, AttributeError, TypeError):
        return None


def submit_array_job(commands, time, cpus_per_task, job_name, memory, submit_file,
                     logs_dir, num_jobs, cmds_per_job, queue,
                     *,
                     gpu_class=None,
                     constraint=None,
                     exclude_nodes=None,
                     requeue=True,
                     max_restarts=2,
                     pre_timeout_seconds=60,
                     # --- Resource layout ---
                     ntasks=1,
                     gpus_per_task=None,
                     # --- Escape hatch ---
                     extra_sbatch=None,
                     # --- Force-redo (clear skip-markers from prior attempts) ---
                     force_redo=False):
    """
    Write a SLURM array job submit script that dispatches lines from a
    commands file in chunks of cmds_per_job across num_jobs array tasks.

    CORE ARGS:
      queue        : partition name. Valid choices:
                       'cpu'          — 7d walltime, all cpu nodes
                       'cpu-bf'       — 12h walltime, backfill on cpu nodes
                       'gpu'          — 7d walltime, gpu nodes
                       'gpu-bf'       — 24h walltime, backfill on gpu nodes
                       'gpu-train'    — 7d walltime, training-tagged gpu nodes
                       'gpu-em'       — 7d walltime, EM-tagged gpu (1 node only)
                       'gpu-train-rf' — 7d walltime, RF-team training nodes (L40S)
      gpu_class    : 'small' | 'large' | 'h200'    (auto = 'small' for gpu* partitions)
      constraint   : feature string, e.g. 'A4000', 'B4000|A5000', 'Blackwell', 'UW'
      exclude_nodes: list[str] or comma-string of nodes to skip, e.g. ['g2702']

    REQUEUE / RESILIENCE:
      requeue      : auto-requeue on node-failure AND pre-timeout (capped).
      max_restarts : cap on requeues  (default 2 => up to 3 attempts)
      pre_timeout_seconds : seconds before walltime to send USR1  (default 60)

    RESOURCE LAYOUT (SLURM model: ntasks × cpus_per_task × gpus_per_task):
      cpus_per_task : CPU threads per task (i.e. per process).  3rd positional arg.
                      Maps to SLURM's --cpus-per-task / -c flag.
      ntasks        : number of parallel tasks (MPI ranks / processes).  Default 1.
      gpus_per_task : GPUs bound to each task.  Default: 1 for gpu* partitions,
                      0 for cpu. Total node GPUs = ntasks * gpus_per_task.

    ESCAPE HATCH:
      extra_sbatch : list[str] of raw '#SBATCH ...' lines appended verbatim.
                     Use for things this API does not expose: --mail-user,
                     --dependency, --nodelist, --mem-per-cpu, --nodes, --account,
                     --qos, --reservation, etc.

    SKIP-MARKER ESCAPE:
      force_redo   : if True, wipe '{logs_dir}/progress/{job_name}_*' before
                     submission, forcing all commands to re-run even if prior
                     attempts marked them done.  Use after editing the commands
                     file or when prior output is suspected partial/corrupt.

    LEGACY queue names (still accepted, emit DeprecationWarning):
      queue='gpu-b4000'    -> queue='gpu',    gpu_class='small', constraint='B4000'
      queue='gpu-bf-b4000' -> queue='gpu-bf', gpu_class='small', constraint='B4000'

    Examples:
      # Single-process, 1 GPU, default (your existing call style)
      submit_array_job(..., queue='gpu')

      # Single-process, 2 GPUs (multi-GPU DDP-via-spawn, DeepSpeed, etc)
      submit_array_job(..., queue='gpu-bf', gpu_class='large', gpus_per_task=2)

      # 4-way torchrun DDP, 8 dataloader workers per rank, 1 GPU per rank
      submit_array_job(..., queue='gpu-bf', gpu_class='large',
                       ntasks=4, cpus_per_task=8, gpus_per_task=1)

      # MPI Rosetta, 8 ranks × 4 threads, no GPUs
      submit_array_job(..., queue='cpu', ntasks=8, cpus_per_task=4)

      # Add a mail notification and dependency on an earlier job
      submit_array_job(..., extra_sbatch=[
          '#SBATCH --mail-type=FAIL',
          '#SBATCH --mail-user=woodbuse@uw.edu',
          '#SBATCH --dependency=afterok:12345',
      ])
    """
    # ---- Input validation (fail before writing anything) ----
    # job_name: no whitespace or slashes (would break SBATCH -o/-e paths)
    if not job_name or any(c.isspace() or c == '/' for c in str(job_name)):
        raise ValueError(f"job_name must be non-empty and contain no whitespace or '/' (got {job_name!r})")
    # num_jobs / cmds_per_job: must be positive integers
    try:
        num_jobs = int(num_jobs)
        cmds_per_job = int(cmds_per_job)
    except (TypeError, ValueError):
        raise ValueError(f"num_jobs and cmds_per_job must be integers (got num_jobs={num_jobs!r}, cmds_per_job={cmds_per_job!r})")
    if num_jobs < 1 or cmds_per_job < 1:
        raise ValueError(f"num_jobs and cmds_per_job must be >= 1 (got num_jobs={num_jobs}, cmds_per_job={cmds_per_job})")
    # commands file must exist & be readable; count lines (non-empty) for coverage check
    if not os.path.isfile(commands):
        raise FileNotFoundError(f"commands file not found: {commands}")
    with open(commands) as _f:
        num_cmds = sum(1 for line in _f if line.strip())
    if num_cmds == 0:
        raise ValueError(f"commands file is empty: {commands}")
    capacity = num_jobs * cmds_per_job
    if capacity < num_cmds:
        raise ValueError(
            f"Coverage error: num_jobs * cmds_per_job = {num_jobs} * {cmds_per_job} = "
            f"{capacity} < {num_cmds} commands.  Increase num_jobs or cmds_per_job."
        )
    if capacity > num_cmds + cmds_per_job:
        # More than one whole task's worth of slack — late tasks waste a scheduler slot
        empty_tasks = num_jobs - ((num_cmds + cmds_per_job - 1) // cmds_per_job)
        warnings.warn(
            f"{empty_tasks} array task(s) will have no commands to run "
            f"(num_jobs={num_jobs} but only {num_cmds} commands at {cmds_per_job}/job). "
            f"Reduce num_jobs to avoid wasted scheduler slots."
        )
    # Walltime sanity vs pre_timeout_seconds (avoid USR1-at-startup hot loops)
    walltime_s = _walltime_to_seconds(time)
    if walltime_s is None:
        warnings.warn(f"Could not parse walltime {time!r} — skipping walltime/pre_timeout sanity check")
    elif requeue and pre_timeout_seconds >= walltime_s - 30:
        raise ValueError(
            f"pre_timeout_seconds={pre_timeout_seconds}s is too large for walltime={time} "
            f"({walltime_s}s). USR1 would fire at or before job start, causing an "
            f"immediate requeue loop. Either lower pre_timeout_seconds or raise walltime."
        )
    # exclude_nodes: accept str, list/tuple of str, else raise
    if exclude_nodes is None or exclude_nodes == "":
        exclude_nodes = None
    elif isinstance(exclude_nodes, str):
        pass  # OK
    elif isinstance(exclude_nodes, (list, tuple, set)):
        bad = [n for n in exclude_nodes if not isinstance(n, str)]
        if bad:
            raise ValueError(f"exclude_nodes must be str or iterable of str (bad elements: {bad!r})")
    else:
        raise ValueError(f"exclude_nodes must be str or iterable of str (got {type(exclude_nodes).__name__})")
    # Normalize logs_dir to always end with '/' so all f-string concatenations work
    logs_dir = str(logs_dir).rstrip('/') + '/'

    # ---- Legacy queue translation ----
    legacy_map = {
        'gpu-b4000':    ('gpu',    'small', 'B4000'),
        'gpu-bf-b4000': ('gpu-bf', 'small', 'B4000'),
    }
    if queue in legacy_map:
        new_partition, legacy_class, legacy_constraint = legacy_map[queue]
        warnings.warn(
            f"queue='{queue}' is deprecated; pass queue='{new_partition}', "
            f"constraint='B4000' (gpu_class defaults to 'small') instead.",
            DeprecationWarning, stacklevel=2,
        )
        partition = new_partition
        if gpu_class is None:
            gpu_class = legacy_class
        if constraint is None:
            constraint = legacy_constraint
    else:
        partition = queue
        # Auto-default gpu_class='small' for gpu* partitions if unspecified,
        # so old call sites like queue='gpu' still get a GPU allocation.
        if partition.startswith('gpu') and gpu_class is None:
            gpu_class = 'small'

    # Partition and GRES names are site-specific. These are the ones on the
    # cluster this work ran on; anywhere else they differ, so an unrecognized
    # value warns and is passed through to SLURM rather than raising -- raising
    # here made the function unusable off-site, since it fired before submission.
    # Override the accepted sets with (comma-separated):
    #     ZINC_HYDRO_SLURM_PARTITIONS
    #     ZINC_HYDRO_SLURM_GPU_CLASSES
    _default_partitions = {'cpu', 'cpu-bf', 'gpu', 'gpu-bf', 'gpu-train',
                           'gpu-em', 'gpu-train-rf'}
    _env_parts = os.environ.get('ZINC_HYDRO_SLURM_PARTITIONS')
    valid_partitions = ({q.strip() for q in _env_parts.split(',') if q.strip()}
                        if _env_parts else _default_partitions)
    if partition not in valid_partitions:
        warnings.warn(
            f"Partition {partition!r} is not in the known set "
            f"{sorted(valid_partitions)}; passing it to SLURM unchanged. Set "
            f"ZINC_HYDRO_SLURM_PARTITIONS to silence this.",
            stacklevel=2,
        )
    _env_classes = os.environ.get('ZINC_HYDRO_SLURM_GPU_CLASSES')
    valid_gpu_classes = ({c.strip() for c in _env_classes.split(',') if c.strip()}
                         if _env_classes else {'small', 'large', 'h200'})
    if gpu_class is not None and gpu_class not in valid_gpu_classes:
        warnings.warn(
            f"gpu_class {gpu_class!r} is not in the known set "
            f"{sorted(valid_gpu_classes)}; emitting "
            f"--gres=gpu:{gpu_class}:N unchanged. Set "
            f"ZINC_HYDRO_SLURM_GPU_CLASSES to silence this.",
            stacklevel=2,
        )
    # Soften: queue='cpu' silently zero out GPU-specific knobs so one config
    # block can be reused for both cpu/gpu jobs.  Print a notice so the
    # coercion isn't invisible.
    if partition in ('cpu', 'cpu-bf'):
        softened = []
        if gpu_class is not None:
            softened.append(f"gpu_class={gpu_class!r}")
            gpu_class = None
        if gpus_per_task is not None and int(gpus_per_task) != 0:
            softened.append(f"gpus_per_task={gpus_per_task}")
            gpus_per_task = 0
        if softened:
            print(f"[notice] queue='{partition}' — ignoring GPU param(s): {', '.join(softened)}")

    # ---- Resolve resource layout ----
    try:
        cpus_per_task_int = int(cpus_per_task)
    except (TypeError, ValueError):
        raise ValueError(f"cpus_per_task must be an integer >= 1 (got {cpus_per_task!r})")
    if cpus_per_task_int < 1:
        raise ValueError(f"cpus_per_task must be >= 1 (got {cpus_per_task_int})")
    cpus_per_task = cpus_per_task_int  # canonicalize
    # gpus_per_task default depends on partition: 1 for GPU partitions, 0 for cpu
    if gpus_per_task is None:
        gpus_per_task = 1 if gpu_class else 0
    try:
        ntasks_int = int(ntasks)
        gpus_per_task_int = int(gpus_per_task)
    except (TypeError, ValueError):
        raise ValueError(f"ntasks and gpus_per_task must be integers (got ntasks={ntasks!r}, gpus_per_task={gpus_per_task!r})")
    if ntasks_int < 1:
        raise ValueError(f"ntasks must be >= 1 (got {ntasks_int})")
    if gpus_per_task_int < 0:
        raise ValueError(f"gpus_per_task must be >= 0 (got {gpus_per_task_int})")
    if gpus_per_task_int > 0 and gpu_class is None:
        raise ValueError(f"gpus_per_task={gpus_per_task_int} requested but queue='{partition}' is not a GPU partition")
    if gpu_class and gpus_per_task_int == 0:
        raise ValueError(f"gpu_class={gpu_class!r} but gpus_per_task=0 — contradictory. Set gpus_per_task>=1 or pass gpu_class=None.")
    total_gpus = ntasks_int * gpus_per_task_int if gpu_class else 0

    # ---- Build SBATCH directives ----
    sbatch_extras = []
    # If multi-task on GPUs, force --nodes=1 so --gres=gpu:CLASS:N (per-node)
    # math matches total_gpus.  Without this, SLURM may spread tasks across
    # nodes and each node would be asked for N GPUs (wrong).  Caller can
    # override via extra_sbatch for genuine multi-node MPI jobs.
    if ntasks_int > 1 and gpu_class:
        sbatch_extras.append("#SBATCH --nodes=1")
    if ntasks_int > 1:
        sbatch_extras.append(f"#SBATCH --ntasks={ntasks_int}")
    if gpu_class:
        sbatch_extras.append(f"#SBATCH --gres=gpu:{gpu_class}:{total_gpus}")
        if ntasks_int > 1:
            # For multi-task GPU jobs, also bind GPUs to tasks explicitly so each
            # MPI rank / torchrun process sees its share via CUDA_VISIBLE_DEVICES.
            sbatch_extras.append(f"#SBATCH --gpus-per-task={gpus_per_task_int}")
    if constraint:
        sbatch_extras.append(f"#SBATCH --constraint='{constraint}'")
    if exclude_nodes:
        excl = exclude_nodes if isinstance(exclude_nodes, str) else ",".join(exclude_nodes)
        sbatch_extras.append(f"#SBATCH --exclude={excl}")
    else:
        excl = None
    if requeue:
        sbatch_extras.append("#SBATCH --requeue")
        sbatch_extras.append("#SBATCH --open-mode=append")
        sbatch_extras.append(f"#SBATCH --signal=B:USR1@{pre_timeout_seconds}")
    if extra_sbatch:
        if isinstance(extra_sbatch, str):
            extra_sbatch = [extra_sbatch]
        for line in extra_sbatch:
            line = line.strip()
            if not line:
                continue
            if not line.startswith('#SBATCH'):
                line = f"#SBATCH {line}"
            sbatch_extras.append(line)
    extras = ("\n" + "\n".join(sbatch_extras)) if sbatch_extras else ""

    # ---- Build the body ----
    if requeue:
        body = f"""\
PROGRESS_DIR="{logs_dir}progress/{job_name}_${{SLURM_ARRAY_TASK_ID}}"
mkdir -p "$PROGRESS_DIR"

MAX_RESTARTS={max_restarts}
RESTART_COUNT="${{SLURM_RESTART_COUNT:-0}}"
PRE_TIMEOUT_S={pre_timeout_seconds}
TASK_START_EPOCH=$(date +%s)
echo "[JOB ${{SLURM_ARRAY_TASK_ID}}] start | attempt=$((RESTART_COUNT+1))/$((MAX_RESTARTS+1)) | host=$(hostname -s) | $(date '+%F %T')"

CURRENT_PID=""
CURRENT_RUN=""
DURATIONS=()        # seconds per successfully-completed cmd this attempt
N_RAN=0             # cmds STARTED this attempt (includes killed-in-progress)
N_OK=0              # cmds that exited 0 this attempt
N_FAILED=0          # cmds that exited non-0 this attempt
N_SKIPPED=0         # cmds skipped because marker existed from prior attempt
N_KILLED=0          # cmds terminated by USR1 mid-execution
TIMED_OUT=0         # 1 if USR1 fired before normal completion

fmt_dur() {{
    local s=$1 h m sec
    h=$((s / 3600)); m=$(((s % 3600) / 60)); sec=$((s % 60))
    if   [[ $h -gt 0 ]]; then printf '%dh%02dm' $h $m
    elif [[ $m -gt 0 ]]; then printf '%dm%02ds' $m $sec
    else                      printf '%ds' $sec
    fi
}}

print_summary() {{
    local task_elapsed=$(( $(date +%s) - TASK_START_EPOCH ))
    local n=${{#DURATIONS[@]}}
    local avg_s=0 min_s=0 max_s=0
    if [[ $n -gt 0 ]]; then
        local sum=0; min_s=${{DURATIONS[0]}}; max_s=${{DURATIONS[0]}}
        for d in "${{DURATIONS[@]}}"; do
            sum=$((sum + d))
            (( d < min_s )) && min_s=$d
            (( d > max_s )) && max_s=$d
        done
        avg_s=$((sum / n))
    fi
    local tag="OK"
    (( TIMED_OUT )) && tag="TIMEOUT"
    echo "[SUMMARY] task=${{SLURM_ARRAY_TASK_ID}} attempt=$((RESTART_COUNT+1)) status=$tag elapsed=$(fmt_dur $task_elapsed) ran=$N_RAN ok=$N_OK failed=$N_FAILED skipped=$N_SKIPPED killed=$N_KILLED avg=$(fmt_dur $avg_s) min=$(fmt_dur $min_s) max=$(fmt_dur $max_s)"
}}

on_timeout() {{
    TIMED_OUT=1
    echo "[JOB ${{SLURM_ARRAY_TASK_ID}}] USR1 received (within ${{PRE_TIMEOUT_S}}s of walltime)"
    if [[ -n "$CURRENT_PID" ]] && kill -0 "$CURRENT_PID" 2>/dev/null; then
        echo "[JOB ${{SLURM_ARRAY_TASK_ID}}] terminating in-progress run $CURRENT_RUN (pid=$CURRENT_PID, pgid=same via setsid)"
        # setsid puts the child in its own process group with pgid=pid, so
        # kill -TERM -PID signals the whole tree (apptainer + python + ...).
        kill -TERM -"$CURRENT_PID" 2>/dev/null || kill -TERM "$CURRENT_PID" 2>/dev/null || true
        # Grace window for apptainer/python to clean up, then SIGKILL the group.
        sleep 5
        kill -KILL -"$CURRENT_PID" 2>/dev/null || kill -KILL "$CURRENT_PID" 2>/dev/null || true
        N_KILLED=$((N_KILLED + 1))
    fi
    print_summary
    if [[ "$RESTART_COUNT" -lt "$MAX_RESTARTS" ]]; then
        echo "[JOB ${{SLURM_ARRAY_TASK_ID}}] requeueing (restart $RESTART_COUNT -> $((RESTART_COUNT+1)))"
        scontrol requeue "$SLURM_JOB_ID" || echo "[JOB ${{SLURM_ARRAY_TASK_ID}}] scontrol requeue FAILED"
    else
        echo "[JOB ${{SLURM_ARRAY_TASK_ID}}] restart cap hit ($MAX_RESTARTS) — not requeueing"
    fi
    exit 0
}}
trap on_timeout USR1

PER_TASK={cmds_per_job}
START_NUM=$(( (SLURM_ARRAY_TASK_ID - 1) * PER_TASK + 1 ))
END_NUM=$(( SLURM_ARRAY_TASK_ID * PER_TASK ))
echo "[JOB ${{SLURM_ARRAY_TASK_ID}}] Runs $START_NUM to $END_NUM"

for (( run=START_NUM; run<=END_NUM; run++ )); do
    marker="$PROGRESS_DIR/done_$run"
    if [[ -f "$marker" ]]; then
        echo "[SKIP]  Run $run (already done on prior attempt)"
        N_SKIPPED=$((N_SKIPPED + 1))
        continue
    fi
    CMD=$(sed -n "${{run}}p" {commands})
    if [[ -z "$CMD" ]]; then
        echo "[JOB ${{SLURM_ARRAY_TASK_ID}}] no command on line $run — finishing early"
        break
    fi
    echo "[START] Run $run | $(date '+%H:%M:%S')"
    CURRENT_RUN=$run
    cmd_start=$(date +%s)
    # Count this run as "started" BEFORE we exec, so the summary inside the
    # USR1 trap correctly reports ran>=1 if the cmd is killed mid-execution.
    N_RAN=$((N_RAN + 1))
    # setsid: child is session+process-group leader (pgid == its pid), so the
    # USR1 trap can kill the whole tree via `kill -TERM -$CURRENT_PID`.
    setsid bash -c "$CMD" &
    CURRENT_PID=$!
    wait "$CURRENT_PID"
    EXIT_CODE=$?
    CURRENT_PID=""
    cmd_elapsed=$(( $(date +%s) - cmd_start ))
    echo "[DONE]  Run $run | exit=$EXIT_CODE | dur=$(fmt_dur $cmd_elapsed) | $(date '+%H:%M:%S')"
    if [[ $EXIT_CODE -eq 0 ]]; then
        touch "$marker"
        N_OK=$((N_OK + 1))
        DURATIONS+=($cmd_elapsed)
    else
        N_FAILED=$((N_FAILED + 1))
    fi
done

echo "[JOB ${{SLURM_ARRAY_TASK_ID}}] All runs finished | $(date '+%F %T')"
print_summary
"""
    else:
        body = f"""\
TASK_START_EPOCH=$(date +%s)
DURATIONS=()
N_RAN=0; N_OK=0; N_FAILED=0

fmt_dur() {{
    local s=$1 h m sec
    h=$((s / 3600)); m=$(((s % 3600) / 60)); sec=$((s % 60))
    if   [[ $h -gt 0 ]]; then printf '%dh%02dm' $h $m
    elif [[ $m -gt 0 ]]; then printf '%dm%02ds' $m $sec
    else                      printf '%ds' $sec
    fi
}}

PER_TASK={cmds_per_job}
START_NUM=$(( ($SLURM_ARRAY_TASK_ID - 1) * $PER_TASK + 1 ))
END_NUM=$(( $SLURM_ARRAY_TASK_ID * $PER_TASK ))
echo "[JOB $SLURM_ARRAY_TASK_ID] Runs $START_NUM to $END_NUM"
for (( run=$START_NUM; run<=END_NUM; run++ )); do
  CMD=$(sed -n "${{run}}p" {commands})
  if [[ -z "$CMD" ]]; then
    echo "[JOB $SLURM_ARRAY_TASK_ID] no command on line $run — finishing early"
    break
  fi
  echo "[START] Run $run | $(date '+%H:%M:%S')"
  cmd_start=$(date +%s)
  N_RAN=$((N_RAN + 1))
  echo "${{CMD}}" | bash
  EXIT_CODE=$?
  cmd_elapsed=$(( $(date +%s) - cmd_start ))
  echo "[DONE]  Run $run | exit=$EXIT_CODE | dur=$(fmt_dur $cmd_elapsed) | $(date '+%H:%M:%S')"
  if [[ $EXIT_CODE -eq 0 ]]; then
    N_OK=$((N_OK + 1)); DURATIONS+=($cmd_elapsed)
  else
    N_FAILED=$((N_FAILED + 1))
  fi
done
task_elapsed=$(( $(date +%s) - TASK_START_EPOCH ))
n=${{#DURATIONS[@]}}
avg_s=0; min_s=0; max_s=0
if [[ $n -gt 0 ]]; then
    sum=0; min_s=${{DURATIONS[0]}}; max_s=${{DURATIONS[0]}}
    for d in "${{DURATIONS[@]}}"; do
        sum=$((sum + d))
        (( d < min_s )) && min_s=$d
        (( d > max_s )) && max_s=$d
    done
    avg_s=$((sum / n))
fi
echo "[JOB $SLURM_ARRAY_TASK_ID] All runs finished"
echo "[SUMMARY] task=$SLURM_ARRAY_TASK_ID elapsed=$(fmt_dur $task_elapsed) ran=$N_RAN ok=$N_OK failed=$N_FAILED avg=$(fmt_dur $avg_s) min=$(fmt_dur $min_s) max=$(fmt_dur $max_s)"
"""

    submit_txt = (
        "#!/bin/bash\n"
        f"#SBATCH -J {job_name}\n"
        f"#SBATCH -p {partition}\n"
        f"#SBATCH -c {cpus_per_task}{extras}\n"
        f"#SBATCH --mem={memory}\n"
        f"#SBATCH -t {time}\n"
        f"#SBATCH -o {logs_dir}{job_name}_%a.stdout\n"
        f"#SBATCH -e {logs_dir}{job_name}_%a.stderr\n"
        f"#SBATCH -a 1-{num_jobs}\n"
        "\n"
        + body
    )

    # ---- Force-redo: wipe prior skip-markers before writing the submit file ----
    if force_redo:
        progress_root = os.path.join(logs_dir, "progress")
        if os.path.isdir(progress_root):
            wiped = 0
            for entry in os.listdir(progress_root):
                if entry.startswith(f"{job_name}_"):
                    shutil.rmtree(os.path.join(progress_root, entry), ignore_errors=True)
                    wiped += 1
            print(f"[force_redo] cleared {wiped} progress dir(s) under {progress_root}")

    with open(submit_file, 'w') as f:
        f.write(submit_txt)
    # num_cmds was already counted during validation above; no need to recount.

    # ---- Reporting ----
    header = f"{'='*45} [JOB: {job_name}] {'='*45}"
    string_len = len(header)
    bits = [partition, f"{num_cmds} cmds", f"{num_jobs} jobs x {cmds_per_job}/job", time, memory, f"{cpus_per_task} cpus/task"]
    if ntasks_int > 1:           bits.append(f"ntasks={ntasks_int} (--nodes=1)" if gpu_class else f"ntasks={ntasks_int}")
    if gpu_class:                bits.append(f"gpu:{gpu_class}:{total_gpus}")
    if constraint:               bits.append(f"constraint='{constraint}'")
    if excl:                     bits.append(f"exclude={excl}")
    if requeue:                  bits.append(f"requeue<={max_restarts} (USR1@{pre_timeout_seconds}s)")
    if extra_sbatch:             bits.append(f"+{len(extra_sbatch)} extra")
    if force_redo:               bits.append("force_redo")
    print(f"\n{header}")
    print(f"| [ INFO ]: {' | '.join(bits)}")
    print(f"| [ CMDS ]: {commands}")
    print(f"| [ LOGS ]: {logs_dir}{job_name}_*.stdout")
    if requeue:
        print(f"| [ PROG ]: {logs_dir}progress/{job_name}_<task>/done_<run>")
    print(f"{'='*string_len}")
    print(f"| [SUBMIT]:")
    print(f"sbatch {submit_file}")
    print(f"{'='*string_len}")


# ============================================================================
# LEGACY: original submit_array_job (pre-2026-05-20 rewrite).  Kept verbatim
# for any notebook that depends on the old behavior (a4000/b4000 GRES, no
# requeue, no skip-markers, no timing summary, no validation, no force_redo).
#
# DO NOT USE FOR NEW WORK — use submit_array_job() above.  The legacy GRES
# names ('gpu:a4000', 'gpu:b4000') no longer exist on the cluster after the
# 2026-05 IT update and jobs submitted via this function will FAIL to schedule.
# This is preserved only to compare against, or to resurrect quickly if the
# new function has an unforeseen regression.
# ============================================================================
def submit_array_job_legacy(commands, time, cores, job_name, memory, submit_file,
                            logs_dir, num_jobs, cmds_per_job, queue):
    """
    LEGACY (pre-2026-05-20).  Same signature as the original submit_array_job.

    Write a SLURM array job submit script that dispatches lines from a
    commands file in chunks of cmds_per_job across num_jobs array tasks.

    Supported queues: cpu, gpu, gpu-bf, gpu-b4000, gpu-bf-b4000

    NOTE: GPU queues here use the OLD GRES names (gpu:a4000, gpu:b4000)
    that the cluster no longer supports.  Use submit_array_job() for new
    work — this is preserved only for reference / rollback.
    """
    queue_blocks = {
        'cpu': '',
        'gpu': '\n#SBATCH --gres=gpu:a4000:1',
        'gpu-bf': (
            "\n#SBATCH --constraint='A100|A4000|A5000|A6000|H200|L40|L40S'"
            "\n#SBATCH --exclude=g2301"
            "\n#SBATCH --gres=gpu:1"
        ),
        'gpu-b4000': '\n#SBATCH --gres=gpu:b4000:1',
        'gpu-bf-b4000': (
            "\n#SBATCH --constraint='B4000'"
            "\n#SBATCH --exclude=g2301"
            "\n#SBATCH --gres=gpu:1"
        ),
    }

    partition_map = {
        'cpu': 'cpu',
        'gpu': 'gpu',
        'gpu-bf': 'gpu-bf',
        'gpu-b4000': 'gpu',
        'gpu-bf-b4000': 'gpu-bf',
    }

    if queue not in queue_blocks:
        raise ValueError(f"Unknown queue '{queue}'. Expected one of {list(queue_blocks)}")

    partition = partition_map[queue]
    extras = queue_blocks[queue]

    submit_txt = f"""\
#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -p {partition}
#SBATCH -c {cores}{extras}
#SBATCH --mem={memory}
#SBATCH -t {time}
#SBATCH -o {logs_dir}{job_name}_%a.stdout
#SBATCH -e {logs_dir}{job_name}_%a.stderr
#SBATCH -a 1-{num_jobs}

PER_TASK={cmds_per_job}
START_NUM=$(( ($SLURM_ARRAY_TASK_ID - 1) * $PER_TASK + 1 ))
END_NUM=$(( $SLURM_ARRAY_TASK_ID * $PER_TASK ))
echo "[JOB $SLURM_ARRAY_TASK_ID] Runs $START_NUM to $END_NUM"
for (( run=$START_NUM; run<=END_NUM; run++ )); do
  CMD=$(sed -n "${{run}}p" {commands})
  echo "[START] Run $run | $(date '+%H:%M:%S')"
  echo "${{CMD}}" | bash
  EXIT_CODE=$?
  echo "[DONE]  Run $run | exit=$EXIT_CODE | $(date '+%H:%M:%S')"
done
echo "[JOB $SLURM_ARRAY_TASK_ID] All runs finished"
"""

    with open(submit_file, 'w') as f:
        f.write(submit_txt)

    with open(commands) as f:
        num_cmds = sum(1 for _ in f)

    header = f"{'='*45} [JOB: {job_name}] {'='*45}"
    string_len = len(header)
    print(f"\n{header}")
    print(f"| [ INFO ]: {queue} | {num_cmds} cmds | {num_jobs} jobs x {cmds_per_job}/job | {time} | {memory} | {cores} cores")
    print(f"| [ CMDS ]: {commands}")
    print(f"| [ LOGS ]: {logs_dir}{job_name}_*.stdout")
    print(f"{'='*string_len}")
    print(f"| [SUBMIT]:")
    print(f"sbatch {submit_file}")
    print(f"{'='*string_len}")


def make_af2_submit_file(name, cmds, job_name, time, logs_dir, cmds_per_job,
                         *, cpus_per_task=1, memory='4G', queue='cpu', **kwargs):
    """
    Write a SLURM CPU array submit script for AlphaFold2 jobs.

    Thin wrapper over submit_array_job() so AF2 jobs get the modern submit
    style instead of the old bare-bones script: per-task logs
    ('{logs_dir}{job_name}_%a.stdout/.stderr'), coverage validation,
    auto-requeue + skip-markers (re-running resumes only the undone commands
    after a preemption/timeout), and per-command timing summaries.

    The original positional signature is preserved, so existing notebook calls
    keep working unchanged:

        make_af2_submit_file(name, cmds, job_name, time, logs_dir, cmds_per_job)

    where `name` is the submit-file path to write and `cmds` is the commands
    file (one shell command per line). num_jobs is derived as
    ceil(num_cmds / cmds_per_job).

    Keyword-only extras (optional) tune resources / forward to submit_array_job():
        cpus_per_task : CPUs per task (default 1).
        memory        : --mem value (default '4G', the historical default; override as needed).
        queue         : partition (default 'cpu'; e.g. 'cpu-bf').
        **kwargs      : anything submit_array_job() accepts — constraint,
                        requeue, max_restarts, exclude_nodes, force_redo, ...

    CHANGED vs. the old version: logs now go to per-task files instead of a
    single 'log.out/log.err'; jobs auto-requeue on preemption by default (pass
    requeue=False to disable); a submission report is printed.
    """
    with open(cmds) as f:
        num_cmds = sum(1 for line in f if line.strip())
    cpj = int(cmds_per_job)
    num_jobs = max(1, (num_cmds + cpj - 1) // cpj)
    return submit_array_job(
        commands=cmds, time=time, cpus_per_task=cpus_per_task, job_name=job_name,
        memory=memory, submit_file=name, logs_dir=logs_dir, num_jobs=num_jobs,
        cmds_per_job=cpj, queue=queue, **kwargs,
    )


def make_af2_submit_file_with_mem_and_optional_gpu(name, cmds, job_name, time,
                                                    logs_dir, cmds_per_job,
                                                    memory='8G', cpu_or_gpu='gpu',
                                                    *, cpus_per_task=1, **kwargs):
    """
    Write a SLURM array submit script for AF2 with configurable memory and an
    optional GPU — now using the modern submit_array_job() style.

    The original positional signature is preserved, so existing notebook calls
    keep working unchanged:

        make_af2_submit_file_with_mem_and_optional_gpu(
            name, cmds, job_name, time, logs_dir, cmds_per_job, memory, cpu_or_gpu)

    `cpu_or_gpu` is used as the partition/queue: 'cpu' or 'gpu' (also accepts
    the backfill variants 'cpu-bf' / 'gpu-bf', etc.). For GPU jobs this now uses
    the CURRENT cluster GRES via submit_array_job's gpu_class (defaults to
    'small') instead of the retired 'gpu:a4000:1' name — so GPU jobs actually
    schedule again. num_jobs is derived as ceil(num_cmds / cmds_per_job).

    Keyword-only extras (optional) forward to submit_array_job():
        cpus_per_task : CPUs per task (default 1).
        **kwargs      : gpu_class ('small'|'large'|'h200'), constraint,
                        requeue, exclude_nodes, force_redo, extra_sbatch, ...

    CHANGED vs. the old version: the broken legacy GPU GRES ('gpu:a4000:1') is
    replaced with the supported gpu_class scheme; logs go to per-task files;
    jobs auto-requeue on preemption by default (pass requeue=False to disable).
    """
    with open(cmds) as f:
        num_cmds = sum(1 for line in f if line.strip())
    cpj = int(cmds_per_job)
    num_jobs = max(1, (num_cmds + cpj - 1) // cpj)
    return submit_array_job(
        commands=cmds, time=time, cpus_per_task=cpus_per_task, job_name=job_name,
        memory=memory, submit_file=name, logs_dir=logs_dir, num_jobs=num_jobs,
        cmds_per_job=cpj, queue=cpu_or_gpu, **kwargs,
    )
