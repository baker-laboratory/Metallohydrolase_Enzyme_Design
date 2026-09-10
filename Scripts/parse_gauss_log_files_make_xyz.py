"""
Author: Seth M. Woodbury
Date: 2024-11-25
Description: This script processes a single Gaussian log file to:
- identify whether it is a valid transition state,
- report imaginary frequencies,
- convert the geometry to XYZ format using Open Babel, and
- optionally reorder atoms in the XYZ output based on specified ranges.

Default behavior:
- Input:  --input_gauss_log /path/to/file.log  (required)
- Output: /path/to/file_gauss_optTS.xyz  (same directory, same basename, suffix "_gauss_optTS.xyz")

Example Commands:
python parse_gauss_log_files_make_xyz.py \
    --input_gauss_log /path/to/TS_foo.log \
    --reorder_ligand_atoms_first 1-18,43-43,51-71

python parse_gauss_log_files_make_xyz.py \
    --input_gauss_log /path/to/TS_foo.log \
    --output_dir /some/other/dir \
    --output_basename TS_foo_custom

Options:
- `--input_gauss_log` (required): Path to a single Gaussian .log file.
- `--reorder_ligand_atoms_first` (optional): Comma-separated list of number ranges to reorder ligand atoms first in the XYZ output.
- `--ignore_TS_warning` (optional): If set, process files with >1 imaginary frequency, with warnings.
- `--output_path` (optional): Full path to output XYZ file. If used, `--output_dir` and `--output_basename` cannot be used.
- `--output_dir` (optional): Directory for the output XYZ file (default: input log file directory).
- `--output_basename` (optional): Basename (without extension) for the output XYZ file
  (default: <input_log_basename>_gauss_optTS).
- `--obabel_exe` (optional): Path to the obabel executable. If not provided, the script tries
  OBABEL_PATH env var and then 'obabel' on PATH.
"""

import os
import re
import subprocess
import shutil
import sys


def parse_gaussian_log(log_file):
    """
    Parse a Gaussian log file to verify transition state and extract frequencies.

    Args:
        log_file (str): Path to the Gaussian log file.

    Returns:
        dict: Contains 'is_valid_ts' (bool) and 'imaginary_freqs' (list).
    """
    frequencies = []

    with open(log_file, 'r') as file:
        for line in file:
            # Extract frequencies
            if "Frequencies" in line:
                freqs = [float(f) for f in re.findall(r"-?\d+\.\d+", line)]
                frequencies.extend(freqs)

    # Validate transition state
    imaginary_freqs = [f for f in frequencies if f < 0]
    is_valid_ts = len(imaginary_freqs) == 1

    return {
        "is_valid_ts": is_valid_ts,
        "imaginary_freqs": imaginary_freqs,
    }


def get_obabel_path(cli_override=None):
    """
    Resolve the obabel executable path using:
    1) CLI override (--obabel_exe)
    2) OBABEL_PATH environment variable
    3) 'obabel' on PATH

    If not found, print an error and exit.
    """
    # 1) CLI override wins
    if cli_override:
        return cli_override

    # 2) Environment variable
    env_path = os.environ.get("OBABEL_PATH")
    if env_path:
        return env_path

    # 3) Look for 'obabel' on PATH
    which_path = shutil.which("obabel")
    if which_path:
        return which_path

    # If we get here, nothing worked
    print("### ERROR: Could not locate an Open Babel 'obabel' executable. ###")
    print("Please either:")
    print("  - Install Open Babel and ensure 'obabel' is on your PATH, or")
    print("  - Set the OBABEL_PATH environment variable to the full path of the 'obabel' executable, or")
    print("  - Use the --obabel_exe flag to specify the executable directly.")
    print("For installation instructions and downloads, see:")
    print("  https://github.com/openbabel/openbabel")
    sys.exit(1)


def run_openbabel(log_file, output_file, obabel_exe):
    """
    Convert Gaussian log file to XYZ format using Open Babel.

    Args:
        log_file (str): Path to the Gaussian log file.
        output_file (str): Path to the output XYZ file.
        obabel_exe (str): Path to the obabel executable.

    Returns:
        bool: True if conversion is successful, False otherwise.
    """
    cmd = [obabel_exe, "-ig09", log_file, "-oxyz", "-O", output_file]
    try:
        subprocess.run(cmd, check=True)
        return True
    except FileNotFoundError:
        print(f"### ERROR: Open Babel executable not found at: {obabel_exe} ###")
        print("Please verify the path or install Open Babel from:")
        print("  https://github.com/openbabel/openbabel")
        return False
    except subprocess.CalledProcessError:
        print(f"### ERROR: Open Babel failed while processing {log_file} ###")
        print(f"Command: {cmd}")
        print("Please ensure Open Babel is correctly installed and that the input file is valid.")
        print("For more information or installation help, see:")
        print("  https://github.com/openbabel/openbabel")
        return False


def reorder_atoms(xyz_file, atom_ranges):
    """
    Reorder atoms in an XYZ file so that specified ranges come first.

    Args:
        xyz_file (str): Path to the input XYZ file.
        atom_ranges (list[str]): List of atom number ranges like ["1-18", "43-43", ...].
    """
    with open(xyz_file, 'r') as file:
        lines = file.readlines()

    if len(lines) < 2:
        print(f"### WARNING: XYZ FILE {xyz_file} LOOKS MALFORMED (FEWER THAN 2 LINES) ###")
        return

    header = lines[:2]   # First two lines are XYZ header
    atoms = lines[2:]    # Remaining lines are atomic coordinates

    # Parse ranges into a set of atom indices
    indices = []
    for r in atom_ranges:
        start, end = map(int, r.split('-'))
        indices.extend(range(start, end + 1))

    indices_set = set(indices)

    ligand_atoms = [atoms[i - 1] for i in indices if 1 <= i <= len(atoms)]
    other_atoms = [atom for i, atom in enumerate(atoms, start=1) if i not in indices_set]

    # Write reordered XYZ
    with open(xyz_file, 'w') as file:
        file.write(header[0])
        file.write(header[1])
        file.writelines(ligand_atoms + other_atoms)

    # Print ligand atom range
    if ligand_atoms:
        print(f"### LIGAND ATOM RANGE (NEW ORDER): 1 - {len(ligand_atoms)} ###")


def process_log_file(input_log, output_xyz_path, atom_reorder_ranges, ignore_TS_warning=False, obabel_exe="obabel"):
    """
    Process a single Gaussian log file.

    Args:
        input_log (str): Path to the Gaussian log file.
        output_xyz_path (str): Full path to the output XYZ file.
        atom_reorder_ranges (list[str] or None): Atom ranges for reordering in the XYZ output.
        ignore_TS_warning (bool): Whether to process files with >1 imaginary frequency.
        obabel_exe (str): Path to the obabel executable.
    """
    log_file = os.path.basename(input_log)
    print(f"### Parsing {log_file} ###")

    result = parse_gaussian_log(input_log)
    imaginary_freqs = result["imaginary_freqs"]

    proceed = result["is_valid_ts"]
    proceed_due_to_flag = False

    if len(imaginary_freqs) > 1:
        print("### WARNING: >1 IMAGINARY FREQUENCY DETECTED, PROCEED WITH CAUTION & VERIFY THAT THE OTHERS HAVE LOW MAGNITUDES ###")
        # Check if any imaginary freq is "small" (<10 cm-1 in magnitude)
        has_small_imag = any(abs(freq) < 10 for freq in imaginary_freqs)
        if has_small_imag:
            print("### KEEP CALM: At least one imaginary frequency has magnitude < 10. (Usually numerical artifact) ###")
        if ignore_TS_warning:
            print("### --ignore_TS_warning FLAG IS SET: WILL CONTINUE PROCESSING THIS FILE ###")
            proceed = True
            proceed_due_to_flag = True

    if not proceed:
        print(f"### SKIPPED {log_file}: NOT A VALID TRANSITION STATE ###\n")
        return

    # Reporting frequencies
    if len(imaginary_freqs) == 0:
        freq_str = "none found"
    else:
        freq_str = ", ".join([f"{f:.2f}" for f in imaginary_freqs])

    if result["is_valid_ts"]:
        print(f"### 1 NEGATIVE FREQUENCY CONFIRMED: {freq_str} ###")
    elif proceed_due_to_flag:
        print(f"### FILE HAS >1 IMAGINARY FREQUENCY, BUT --ignore_TS_warning WAS USED. FREQUENCIES: {freq_str} ###")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_xyz_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    success = run_openbabel(input_log, output_xyz_path, obabel_exe)
    if not success:
        print(f"### FAILED TO CREATE XYZ FILE FOR {log_file} ###\n")
        return

    print(f"### XYZ FILE CREATED: {output_xyz_path} ###")

    if atom_reorder_ranges:
        reorder_atoms(output_xyz_path, atom_reorder_ranges)
        print(f"### REORDERED XYZ FILE OVERWRITTEN: {output_xyz_path} ###\n")
    else:
        print(f"### NO REORDERING SPECIFIED ###\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process a Gaussian log file to extract an optimized TS geometry as XYZ."
    )
    parser.add_argument(
        "--input_gauss_log",
        type=str,
        required=True,
        help="Path to a single Gaussian .log file.",
    )
    parser.add_argument(
        "--reorder_ligand_atoms_first",
        type=str,
        default=None,
        help=(
            "Comma-separated list of atom ranges to reorder ligand atoms first in the XYZ "
            "output (e.g., '1-18,43-43,51-71')."
        ),
    )
    parser.add_argument(
        "--ignore_TS_warning",
        action="store_true",
        help="If set, process files with >1 imaginary frequency, with warnings.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help=(
            "Full path to the output XYZ file. If used, --output_dir and --output_basename "
            "must NOT be provided."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Directory for the output XYZ file. Default: same directory as the input "
            "Gaussian log."
        ),
    )
    parser.add_argument(
        "--output_basename",
        type=str,
        default=None,
        help=(
            "Basename (without extension) for the output XYZ file. Default: "
            "<input_log_basename>_gauss_optTS."
        ),
    )
    parser.add_argument(
        "--obabel_exe",
        type=str,
        default=None,
        help=(
            "Path to the obabel executable. If not provided, the script tries OBABEL_PATH "
            "env var and then 'obabel' on your PATH."
        ),
    )

    args = parser.parse_args()

    # Enforce mutual exclusivity
    if args.output_path and (args.output_dir or args.output_basename):
        parser.error("--output_path cannot be used together with --output_dir or --output_basename.")

    input_log = args.input_gauss_log

    # Determine output path
    if args.output_path:
        xyz_output_path = args.output_path
    else:
        # Base directory
        if args.output_dir:
            out_dir = args.output_dir
        else:
            out_dir = os.path.dirname(input_log)

        # Basename (without extension)
        if args.output_basename:
            base = args.output_basename
        else:
            base = os.path.splitext(os.path.basename(input_log))[0] + "_gauss_optTS"

        # Ensure .xyz extension
        if not base.lower().endswith(".xyz"):
            base = base + ".xyz"

        xyz_output_path = os.path.join(out_dir, base)

    atom_reorder_ranges = (
        args.reorder_ligand_atoms_first.split(",")
        if args.reorder_ligand_atoms_first
        else None
    )

    # Resolve obabel executable (CLI override > OBABEL_PATH > PATH)
    obabel_exe = get_obabel_path(cli_override=args.obabel_exe)

    process_log_file(
        input_log=input_log,
        output_xyz_path=xyz_output_path,
        atom_reorder_ranges=atom_reorder_ranges,
        ignore_TS_warning=args.ignore_TS_warning,
        obabel_exe=obabel_exe,
    )
