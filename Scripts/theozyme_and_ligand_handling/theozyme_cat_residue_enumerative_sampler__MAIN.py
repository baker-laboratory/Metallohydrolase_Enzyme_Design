#!/usr/bin/env python3
"""
SCRIPT NAME
    theozyme_cat_residue_enumerative_sampler__MAIN.py

PURPOSE
    Glob a set of PDB files (via glob patterns or explicit list) and run
    the histidine sampler (STEP1) on each in parallel.  Optionally, take
    all of those STEP1 outputs and feed them into the glu/asp sampler
    (STEP2) in a second parallel pass.

USAGE
    python theozyme_cat_residue_enumerative_sampler__MAIN.py \
        --input_pdbs pdb1.pdb pdb2.pdb '*group2_*.pdb' \
        --histidine_config '{"A1":["UO","FO","UA"],"B2":"all"}' \
        [--output_dir PATH] \
        [--allowed_tautomer_swaps_at_once 0,1,2] \
        [--classic_suffix] [--verbose_pdbs] [--nprocs N] \
        [--step2_script PATH] [--gluE_aspD_json '{"A1":"ED",...}']
"""

from pathlib import Path
import glob
import subprocess
import argparse
import json
import itertools
import os
import shlex
import sys
from multiprocessing import Pool

# Sibling helper scripts resolve next to this file, so the whole set can be
# relocated together.
_SCRIPT_DIR = Path(__file__).resolve().parent


def run_cmd(cmd):
    """Execute a shell command, streaming output."""
    subprocess.run(cmd, shell=True, check=True)


def shell_join(parts):
    return ' '.join(shlex.quote(str(part)) for part in parts)


def load_json_arg(raw, label):
    if os.path.isfile(raw):
        with open(raw) as handle:
            return json.load(handle)
    try:
        return json.loads(raw)
    except ValueError as exc:
        sys.exit(f"Could not parse {label} as JSON or JSON file path: {exc}")


def normalize_step1_allowed(histidine_config):
    cfg = load_json_arg(histidine_config, "--histidine_config")
    canonical = ['UO', 'FO', 'UA', 'FA']
    legacy_map = {'0E': 'UO', '1E': 'FO', '0D': 'UA', '1D': 'FA'}

    allowed_map = {}
    for raw_key, val in cfg.items():
        resid = str(raw_key).split(':')[0].strip()
        if val == 'all':
            allowed = list(canonical)
        else:
            values = list(val)
            allowed = []
            for item in values:
                tok = str(item).strip().upper()
                tok = legacy_map.get(tok, tok)
                if tok not in canonical:
                    sys.exit(
                        f"Invalid STEP1 choice '{item}' for {raw_key}. "
                        f"Valid choices: {canonical}."
                    )
                if tok not in allowed:
                    allowed.append(tok)
        allowed_map[resid] = allowed
    return allowed_map


def step1_suffixes(histidine_config, allowed_tautomer_swaps_at_once, classic_suffix):
    allowed_map = normalize_step1_allowed(histidine_config)
    if allowed_tautomer_swaps_at_once:
        allowed_a_counts = {
            int(x) for x in allowed_tautomer_swaps_at_once.split(',') if x != ''
        }
    else:
        allowed_a_counts = None

    suffixes = []
    keys = list(allowed_map.keys())
    for perm in itertools.product(*(allowed_map[key] for key in keys)):
        a_count = sum(1 for code in perm if code[1] == 'A')
        if allowed_a_counts is not None and a_count not in allowed_a_counts:
            continue

        if classic_suffix:
            suffix = "_".join(perm)
        else:
            flip_bits = "".join(code[0] for code in perm)
            taut_letters = "".join(code[1] for code in perm)
            suffix = f"{flip_bits}_{taut_letters}"
        suffixes.append(suffix)
    return suffixes


def expected_step1_outputs(pdb_list, args):
    suffixes = step1_suffixes(
        args.histidine_config,
        args.allowed_tautomer_swaps_at_once,
        args.classic_suffix,
    )
    outputs = []
    for pdb in pdb_list:
        outd = args.output_dir or os.path.dirname(pdb)
        base = os.path.splitext(os.path.basename(pdb))[0]
        outputs.extend(os.path.join(outd, f"{base}_{suffix}.pdb") for suffix in suffixes)
    return outputs


def normalize_step2_allowed(gluE_aspD_json):
    cfg = load_json_arg(gluE_aspD_json, "--gluE_aspD_json")
    allowed = {}
    for key, val in cfg.items():
        values = list(val) if isinstance(val, str) else list(val)
        for state in values:
            if state not in ('E', 'D'):
                sys.exit(
                    f"Invalid STEP2 choice '{state}' for {key}. Valid choices: E, D."
                )
        allowed[key] = values
    return allowed


def step2_suffixes(gluE_aspD_json):
    allowed = normalize_step2_allowed(gluE_aspD_json)
    keys = list(allowed.keys())
    return [
        ''.join(perm)
        for perm in itertools.product(*(allowed[key] for key in keys))
    ]


def expected_step2_outputs(input_pdbs, args):
    suffixes = step2_suffixes(args.gluE_aspD_json)
    outputs = []
    for pdb in input_pdbs:
        outd = args.output_dir or os.path.dirname(pdb)
        base = os.path.splitext(os.path.basename(pdb))[0]
        outputs.extend(os.path.join(outd, f"{base}_{suffix}.pdb") for suffix in suffixes)
    return outputs


def expected_step3_outputs(input_pdbs, args):
    specs = load_json_arg(args.rotation_config, "--rotation_config")
    if isinstance(specs, dict):
        specs = [specs]
    periods = [int(spec['periodicity']) for spec in specs]

    outputs = []
    for pdb in input_pdbs:
        outd = args.output_dir or os.path.dirname(pdb)
        base = os.path.splitext(os.path.basename(pdb))[0]
        for combo in itertools.product(*(range(period) for period in periods)):
            parts = []
            for period, i_step in zip(periods, combo):
                width = len(str(period))
                parts.append(f"rotP_{i_step:0{width}d}")
            suffix = '_' + '_'.join(parts)
            outputs.append(os.path.join(outd, f"{base}{suffix}.pdb"))
    return outputs


def require_expected_outputs(paths, stage_name):
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        preview = "\n".join(missing[:10])
        extra = "" if len(missing) <= 10 else f"\n... and {len(missing) - 10} more"
        sys.exit(f"{stage_name} did not create expected output(s):\n{preview}{extra}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input_pdbs', nargs='+', required=True,
                   help='Glob patterns or explicit PDB filenames')
    p.add_argument('--step1_script',
                   default=str(_SCRIPT_DIR / "theozyme_cat_residue_enumerative_sampler__STEP1_histidine_sampler.py"),
                   help='Path to the STEP1 histidine sampler')
    p.add_argument('--histidine_config', default=None,
                   help='JSON or file for STEP1; if omitted, STEP1 is skipped')
    p.add_argument('--allowed_tautomer_swaps_at_once', default=None,
                   help='STEP1: comma‑sep ints limiting tautomer-swap (A-type) ops')
    p.add_argument('--classic_suffix', action='store_true', default=False,
                   help='STEP1: use classic suffix (_UO_FA_UO)')
    p.add_argument('--verbose_pdbs', action='store_true', default=False,
                   help='Print each invoked STEP1/STEP2 command')
    p.add_argument('--output_dir', default=None,
                   help='Directory for all outputs (defaults per‐input)')
    p.add_argument('--nprocs', type=int, default=1,
                   help='Number of parallel processes')
    # STEP2 args
    p.add_argument('--step2_script',
                   default=str(_SCRIPT_DIR / "theozyme_cat_residue_enumerative_sampler__STEP2_glu_asp_sampler.py"),
                   help='Path to the STEP2 glu/asp sampler')
    p.add_argument('--gluE_aspD_json', default=None,
                   help='JSON or file for STEP2; if omitted, STEP2 is skipped')
    # STEP3 args
    p.add_argument(
        '--step3_script',
        default=str(_SCRIPT_DIR / "theozyme_cat_residue_enumerative_sampler__STEP3_residue_rotator.py"),
        help='Path to the STEP3 residue rotator script'
    )
    p.add_argument(
        '--rotation_config',
        default=None,
        help='JSON list or file for STEP3; if omitted, STEP3 is skipped'
    )

    # CLEANING ARGS 
    p.add_argument(
        '--clean_intermediates',
        action='store_true',
        default=True,
        help='If set, delete intermediate PDBs after the final STEP3 run'
    )
    p.add_argument(
        '--do_not_clean_intermediates',
        action='store_true',
        default=False,
        help='If set, delete intermediate PDBs after the final STEP3 run'
    )
    args = p.parse_args()

    ### CLEANING -> ADD HERE WHEN MORE STEPS ADDED ###
    # track which steps actually ran
    did1 = bool(args.histidine_config)
    did2 = bool(args.gluE_aspD_json)
    did3 = bool(args.rotation_config)

    # these will collect the full list of files each step produces
    step1_out = []
    step2_out = []
    step3_out = []
    ###################################################

    # 1) Expand globs
    pdb_list = []
    for pat in args.input_pdbs:
        matched = glob.glob(pat) or [pat]
        pdb_list.extend(matched)
    pdb_list = sorted(set(pdb_list))
    if not pdb_list:
        print("No PDBs found for given patterns.", file=sys.stderr)
        sys.exit(1)

    # 2) STEP1: histidine sampling
    if args.histidine_config:
        step1_out = expected_step1_outputs(pdb_list, args)
        cmds1 = []
        for pdb in pdb_list:
            cmd = [
                'python', args.step1_script,
                '--input_pdb', pdb,
                '--histidine_config', args.histidine_config
            ]
            if args.output_dir:
                cmd += ['--output_dir', args.output_dir]
            if args.allowed_tautomer_swaps_at_once:
                cmd += ['--allowed_tautomer_swaps_at_once', args.allowed_tautomer_swaps_at_once]
            if args.classic_suffix:
                cmd.append('--classic_suffix')
            if args.verbose_pdbs:
                cmd.append('--verbose_pdbs')
            cmds1.append(shell_join(cmd))

        print(f"STEP1 → Running {len(cmds1)} jobs with {args.nprocs} procs...")
        with Pool(args.nprocs) as pool:
            pool.map(run_cmd, cmds1)
        require_expected_outputs(step1_out, "STEP1")
    else:
        step1_out = pdb_list

    # 3) STEP2: glu/asp sampling
    if args.gluE_aspD_json:
        step2_out = expected_step2_outputs(step1_out, args)
        cmds2 = []
        for pdb in step1_out:
            cmd = [
                'python', args.step2_script,
                '--input_pdb', pdb,
                '--gluE_aspD_json', args.gluE_aspD_json
            ]
            if args.output_dir:
                cmd += ['--out_dir', args.output_dir]
            if args.verbose_pdbs:
                cmd.append('--verbose')
            cmds2.append(shell_join(cmd))

        print(f"STEP2 → Running {len(cmds2)} jobs with {args.nprocs} procs...")
        with Pool(args.nprocs) as pool:
            pool.map(run_cmd, cmds2)
        require_expected_outputs(step2_out, "STEP2")

    # 4) STEP3: residue rotator
    if args.rotation_config:
        # decide which PDBs to feed into STEP3:
        if args.gluE_aspD_json:
            source_list = step2_out
        elif args.histidine_config:
            source_list = step1_out
        else:
            source_list = pdb_list

        step3_out = expected_step3_outputs(source_list, args)
        cmds3 = []
        for pdb in source_list:
            cmd = [
                'python', args.step3_script,
                '--input_pdb', pdb,
                '--rotation_config', args.rotation_config
            ]
            if args.output_dir:
                cmd += ['--output_dir', args.output_dir]
            if args.verbose_pdbs:
                cmd.append('--verbose')
            cmds3.append(shell_join(cmd))

        print(f"STEP3 → Running {len(cmds3)} jobs with {args.nprocs} procs...")
        with Pool(args.nprocs) as pool:
            pool.map(run_cmd, cmds3)
        require_expected_outputs(step3_out, "STEP3")

    # ----------------------------------------------------------------------------
    #  optional cleanup of ALL intermediate files, leaving ONLY the final outputs
    # ----------------------------------------------------------------------------
    if args.clean_intermediates:
        # decide which is the “final” set
        if did3:
            final = set(step3_out)
        elif did2:
            final = set(step2_out)
        elif did1:
            final = set(step1_out)
        else:
            final = set(pdb_list)

        # union up everything that was ever generated
        all_gen = set()
        if did1: all_gen |= set(step1_out)
        if did2: all_gen |= set(step2_out)
        if did3: all_gen |= set(step3_out)

        # intermediates = everything except the finals
        to_remove = all_gen - final
        if args.do_not_clean_intermediates:
            print(f"\nNOT CLEANING UP INTERMEDIATE FILES")
        else:
            for fn in to_remove:
                try:
                    os.remove(fn)
                except OSError:
                    pass

if __name__ == '__main__':
    main()
