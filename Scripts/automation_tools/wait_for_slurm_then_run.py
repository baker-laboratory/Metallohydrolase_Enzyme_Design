#!/usr/bin/env python3
"""
Wait for a Slurm job/array to leave the queue, then run local shell commands.

This is intentionally not a Slurm dependency helper: commands run on the machine
where this script is launched, so it is useful from an interactive node/session.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import shlex
import subprocess
import sys
import time
from typing import Iterable


TERMINAL_OK_STATES = {"COMPLETED"}
TERMINAL_BAD_STATES = {
    "BOOT_FAIL",
    "CANCELED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REQUEUED",
    "REVOKED",
    "SPECIAL_EXIT",
    "TIMEOUT",
}


class SlurmJobNotInQueue(RuntimeError):
    """Raised when squeue no longer has an entry for a job id."""


def log(message: str, log_file: pathlib.Path | None = None) -> None:
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def run_text_command(argv: list[str], check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check)


def squeue_rows(job_id: str) -> list[str]:
    proc = run_text_command(["squeue", "-h", "-j", job_id, "-o", "%i|%T|%M|%R"])
    if proc.returncode != 0:
        if "Invalid job id specified" in proc.stderr:
            raise SlurmJobNotInQueue(proc.stderr.strip())
        raise RuntimeError(f"squeue failed for {job_id}: {proc.stderr.strip()}")
    return [line for line in proc.stdout.splitlines() if line.strip()]


def sacct_states(job_id: str) -> list[tuple[str, str, str]]:
    proc = run_text_command(
        ["sacct", "-n", "-P", "-j", job_id, "--format=JobIDRaw,State,ExitCode"]
    )
    if proc.returncode != 0:
        return []

    states: list[tuple[str, str, str]] = []
    for line in proc.stdout.splitlines():
        fields = line.strip().split("|")
        if len(fields) < 3:
            continue
        raw_job_id, state, exit_code = fields[:3]
        if not raw_job_id:
            continue
        # Batch/extern rows are not separate array elements and can obscure the
        # useful task-level summary for arrays.
        if raw_job_id.endswith(".batch") or raw_job_id.endswith(".extern"):
            continue
        states.append((raw_job_id, state.split()[0], exit_code))
    return states


def summarize_states(states: Iterable[tuple[str, str, str]]) -> str:
    counts: dict[str, int] = {}
    for _, state, _ in states:
        counts[state] = counts.get(state, 0) + 1
    return ", ".join(f"{state}={count}" for state, count in sorted(counts.items())) or "unknown"


def wait_for_job(
    job_id: str,
    mode: str,
    poll_interval: int,
    log_file: pathlib.Path | None,
    quiet_unchanged: bool,
) -> None:
    log(f"watching Slurm job {job_id} (mode={mode}, poll={poll_interval}s)", log_file)
    last_summary = None

    while True:
        try:
            rows = squeue_rows(job_id)
        except SlurmJobNotInQueue:
            states = sacct_states(job_id)
            if states:
                break
            raise RuntimeError(
                f"squeue does not know job {job_id}, and sacct has no records for it. "
                "Check that the job id is correct."
            ) from None
        if not rows:
            break

        summary: dict[str, int] = {}
        for row in rows:
            parts = row.split("|")
            state = parts[1] if len(parts) > 1 else "UNKNOWN"
            summary[state] = summary.get(state, 0) + 1
        text = ", ".join(f"{state}={count}" for state, count in sorted(summary.items()))
        if not quiet_unchanged or text != last_summary:
            log(f"job still active: {text}", log_file)
            last_summary = text
        time.sleep(poll_interval)

    log(f"job {job_id} is no longer in squeue", log_file)

    if mode == "afterany":
        states = sacct_states(job_id)
        if states:
            log(f"sacct final states: {summarize_states(states)}", log_file)
        return

    # afterok mode: require Slurm accounting to report only COMPLETED rows.
    for _ in range(6):
        states = sacct_states(job_id)
        if states:
            break
        time.sleep(10)
    if not states:
        raise RuntimeError("afterok requested, but sacct did not return final state rows")

    bad = [(jid, state, code) for jid, state, code in states if state not in TERMINAL_OK_STATES]
    log(f"sacct final states: {summarize_states(states)}", log_file)
    if bad:
        preview = ", ".join(f"{jid}:{state}:{code}" for jid, state, code in bad[:10])
        raise RuntimeError(f"afterok requested, but some tasks did not complete: {preview}")


def _commands_from_json_text(text: str, source: str) -> list[str]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source} is not valid JSON: {exc}") from exc

    if isinstance(data, str):
        return [data]
    if not isinstance(data, list):
        raise ValueError(f"{source} must be a JSON string or list")

    commands: list[str] = []
    for index, item in enumerate(data, start=1):
        if isinstance(item, str):
            commands.append(item)
        elif isinstance(item, list) and all(isinstance(part, str) for part in item):
            # argv-style JSON is useful when a command has many literal arguments.
            commands.append(shlex.join(item))
        else:
            raise ValueError(
                f"{source}[{index}] must be a command string or a list of string argv parts"
            )
    return commands


def read_commands(
    cmds: list[str],
    cmd_files: list[pathlib.Path],
    commands_json: list[str],
    commands_json_files: list[pathlib.Path],
    commands_json_stdin: bool,
) -> list[str]:
    commands = list(cmds)
    for path in cmd_files:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                commands.append(stripped)
    for index, text in enumerate(commands_json, start=1):
        commands.extend(_commands_from_json_text(text, f"--commands-json #{index}"))
    for path in commands_json_files:
        commands.extend(_commands_from_json_text(path.read_text(encoding="utf-8"), str(path)))
    if commands_json_stdin:
        commands.extend(_commands_from_json_text(sys.stdin.read(), "stdin"))
    return commands


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    command_after_dash: list[str] = []
    if "--" in raw_argv:
        dash_index = raw_argv.index("--")
        command_after_dash = raw_argv[dash_index + 1 :]
        raw_argv = raw_argv[:dash_index]

    parser = argparse.ArgumentParser(
        description="Wait for a Slurm job/array, then run local commands sequentially."
    )
    parser.add_argument("job_id", nargs="?", help="Slurm job id or array master id, e.g. 14539905")
    parser.add_argument("--slurm-job-id", "--slurm-jobID", dest="slurm_job_id", help="Slurm job id or array master id. Overrides the positional job id if both are provided.")
    parser.add_argument("--cmd", action="append", default=[], help="Shell command to run after the job finishes. Repeatable.")
    parser.add_argument("--cmd-file", action="append", type=pathlib.Path, default=[], help="File containing one shell command per non-comment line.")
    parser.add_argument("--commands-json", action="append", default=[], help="JSON command string or list. Items can be command strings or argv lists.")
    parser.add_argument("--commands-json-file", action="append", type=pathlib.Path, default=[], help="JSON file containing a command string or list of commands.")
    parser.add_argument("--commands-json-stdin", action="store_true", help="Read JSON commands from stdin.")
    parser.add_argument("--mode", choices=["afterany", "afterok"], default="afterany", help="afterany runs regardless of final Slurm state; afterok requires all tasks COMPLETED.")
    parser.add_argument("--poll-interval", type=int, default=60, help="Seconds between squeue checks.")
    parser.add_argument("--shell", default="/bin/bash", help="Shell used for each command.")
    parser.add_argument("--continue-on-error", action="store_true", help="Keep running later commands if one command exits nonzero.")
    parser.add_argument("--dry-run", action="store_true", help="Wait and print commands, but do not execute them.")
    parser.add_argument("--quiet-unchanged", action="store_true", help="Only print Slurm status when the summary changes.")
    parser.add_argument("--log-file", type=pathlib.Path, help="Optional log file for watcher status messages.")
    args = parser.parse_args(raw_argv)

    job_id = args.slurm_job_id or args.job_id
    if not job_id:
        parser.error("provide a Slurm job id using --slurm-job-id or the positional job_id")

    commands = read_commands(
        args.cmd,
        args.cmd_file,
        args.commands_json,
        args.commands_json_file,
        args.commands_json_stdin,
    )
    if command_after_dash:
        commands.append(shlex.join(command_after_dash))

    if not commands:
        parser.error("provide at least one --cmd, --cmd-file, or command after --")

    if args.poll_interval < 5:
        parser.error("--poll-interval must be at least 5 seconds")

    wait_for_job(job_id, args.mode, args.poll_interval, args.log_file, args.quiet_unchanged)

    for index, command in enumerate(commands, start=1):
        log(f"command {index}/{len(commands)}: {command}", args.log_file)
        if args.dry_run:
            continue
        proc = subprocess.run(command, shell=True, executable=args.shell)
        if proc.returncode != 0:
            log(f"command {index} exited with code {proc.returncode}", args.log_file)
            if not args.continue_on_error:
                return proc.returncode

    log("all commands finished", args.log_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
