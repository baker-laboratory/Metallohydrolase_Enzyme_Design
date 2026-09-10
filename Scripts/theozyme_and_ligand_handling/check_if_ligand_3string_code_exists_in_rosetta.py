#!/usr/bin/env python3

def _default_residue_types():
    """Locate Rosetta's fa_standard residue_types.txt via the repo path shim."""
    import sys as _sys
    from pathlib import Path as _Path
    for _anc in _Path(__file__).resolve().parents:
        if (_anc / "repo_paths.py").is_file():
            _sys.path.insert(0, str(_anc)); break
    try:
        import repo_paths
        return str(repo_paths.ROSETTA_RESIDUE_TYPES)
    except Exception:
        import os as _os
        return _os.environ.get("ROSETTA_RESIDUE_TYPES", "")

"""
################################################################################
Script Name:
    check_if_ligand_3string_code_exists_in_rosetta.py

Purpose:
    This utility scans a Rosetta residue_types.txt file and determines whether a
    given three‑letter ligand code already exists in the Rosetta database.  It’s
    intended to prevent name collisions when generating new ligand parameter
    files for Rosetta, by catching both exact filename matches (e.g. “LAT.params”)
    and more subtle occurrences (e.g. “ LAT ” floating in comments or other lines).
    Additionally, it flags potential ambiguities if the code begins with “to” (toX)
    and warns whenever the code appears anywhere on a .params line, even if not
    directly adjacent to “.params”.

Usage:
    From the command line, run:

        python3 check_if_ligand_3string_code_exists_in_rosetta.py \
            --alphanumerical_code YOUR_CODE \
            [--txt_file_of_rosetta_chemicals /path/to/residue_types.txt]

    Required arguments:
      -c, --alphanumerical_code
          The exact three‑letter code you intend to use for your new ligand (e.g. “TAE”, “2AT”, etc.).

    Optional arguments:
      -f, --txt_file_of_rosetta_chemicals
          Path to the Rosetta residue_types.txt file.  Defaults to Rosetta’s
          standard fa_standard set:
          $ROSETTA/database/chemical/residue_type_sets/fa_standard/residue_types.txt

Behavior & Output:
    1. Prints a clearly delimited header block indicating start of scan.
    2. Emits an “optional instruction” line pointing users to the raw file path
       in case they want to verify matches manually.
    3. For each line containing your code, categorizes and prints:
       • [filename]    — exact occurrence of “CODE.params” (highest priority)
       • [standalone]  — true word‑boundary match of CODE, not part of “.params”
       • [warning]     — any line with “.params” that contains CODE anywhere,
                         even if not contiguous with “.params”
    4. If your code starts with the letters “to” (case‑insensitive), emits a
       pre‑scan warning about potential ambiguous partial matches (“toX”).
    5. After scanning, prints a clearly delimited footer block.
    6. If any matches were found, prints a multi‑line “!!! WARNING !!!” banner
       urging you to choose a different code to avoid collisions.
    7. If no matches were found, prints a friendly “Ligand code is probably okay :)”
       message to stderr and exits with code 1.

Exit Codes:
      0   — Successful scan; matches WERE found (user must choose another code)
      1   — Successful scan; NO matches found (ligand code appears safe)
      2   — Input file not found
      3   — Other I/O or unexpected error

Internal Details:
    • Uses Python’s argparse for clear CLI interface.
    • Relies on two compiled regular expressions:
        1.  standalone_pattern = r'\bCODE\b'  (true word boundaries)
        2.  filename_pattern   = r'CODE\.params\b'
    • Checks each line for “.params” before emitting the loose‑match warning.
    • Prints all warnings to stderr to distinguish them from ordinary output lines.

Example:
    $ python3 check_if_ligand_3string_code_exists_in_rosetta.py -c LAT

    printout:
    ########################################### [START] ###########################################
    [OPTIONAL INSTRUCTION]: Feel free to manually check yourself at the file below -->
    $ROSETTA/database/.../residue_types.txt

    ### --- [OFFENDING LINES LISTED BELOW] --- ###
    [filename]    Line  27: residue_types/l-caa/LAT.params
    [warning]     Line 132: residue_types/nucleic/.../A2LAT.params

    ############################################ [END] ############################################
    ###############################################################################################
    ############################### !!! WARNING WARNING WARNING !!! ###############################
    ###############################     LIGAND CODE PRE-EXISTS      ###############################
    ###############################################################################################
    [IMPORTANT INSTRUCTION]: Find a new ligand code or you will regret your life choices!!!
"""
import argparse
import re
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description="Check if a 3‑letter ligand code exists in a Rosetta residue_types.txt file."
    )
    parser.add_argument(
        "--alphanumerical_code", "-c",
        required=True,
        help="Three‑letter ligand code to search for (e.g. LAT, 2AT, etc.)"
    )
    parser.add_argument(
        "--txt_file_of_rosetta_chemicals", "-f",
        default=_default_residue_types(),
        help="Path to the Rosetta residue_types.txt file"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    code = args.alphanumerical_code
    fn   = args.txt_file_of_rosetta_chemicals
    print("########################################### [START] ###########################################")
    print(f"[OPTIONAL INSTRUCTION]: Feel free to manually check yourself at the file below -->") 
    print(f"{fn}")
    print("")
    print("### --- [OFFENDING LINES LISTED BELOW] --- ###")

    # 1) Warn if the code starts with "to" (case‑insensitive)
    if code.lower().startswith("to"):
        print(f"Warning: input code '{code}' starts with 'to'. This prefix may lead to ambiguous matches.", file=sys.stderr)

    # Compile regexes
    standalone_pattern = re.compile(rf'\b{re.escape(code)}\b')
    filename_pattern   = re.compile(rf'{re.escape(code)}\.params\b')

    found_any = False

    try:
        with open(fn, 'r') as f:
            for lineno, line in enumerate(f, start=1):
                raw = line.rstrip("\n")
                has_params_line   = ".params" in raw
                exact_in_filename = bool(filename_pattern.search(raw))
                standalone_match  = bool(standalone_pattern.search(raw)) and not exact_in_filename
                any_code_in_line  = code in raw

                # 2) Exact filename match (e.g. LAT.params)
                if exact_in_filename:
                    print(f"[filename]  Line {lineno}: {raw}")
                    found_any = True

                # 3) Standalone code (word‑boundary, not part of .params)
                if standalone_match:
                    print(f"[standalone] Line {lineno}: {raw}")
                    found_any = True

                # 4) Warning for any code occurrence on a .params line
                if has_params_line and any_code_in_line and not exact_in_filename:
                    print(f"[warning]   Line {lineno} (.params line contains '{code}' somewhere): {raw}", file=sys.stderr)
                    found_any = True

        print("")
        print("############################################ [END] ############################################")
        print("")

        if found_any:
            print("###############################################################################################")
            print("############################### !!! WARNING WARNING WARNING !!! ###############################")
            print("###############################     LIGAND CODE PRE-EXISTS      ###############################")
            print("###############################################################################################")
            print("[IMPORTANT INSTRUCTION]: Find a new ligand code or you will regret your life choices!!!")

        if not found_any:
            print(f"NO occurrences of '{code}' found in {fn}", file=sys.stderr)
            print(f"")
            print(f"Ligand code is probably okay :)")
            sys.exit(1)

    except FileNotFoundError:
        print(f"Error: file not found: {fn}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Error reading {fn}: {e}", file=sys.stderr)
        sys.exit(3)

if __name__ == "__main__":
    main()
