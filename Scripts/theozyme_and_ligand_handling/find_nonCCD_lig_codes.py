#!/usr/bin/env python3
"""
find_nonCCD_lig_codes.py

Fast parallel search for 3-character ligand codes that are NOT present
in the PDB Chemical Component Dictionary (CCD), using the RCSB ligand
endpoint as a check.

Features:
  - Arbitrary number of "classes", each with a pattern like X0? or ?9A.
  - ? = wildcard position, expanded over a configurable alphabet.
  - Parallel HTTP checks with ThreadPoolExecutor (I/O-bound).
  - Configurable batch size, max workers, targets per class, etc.

Example usage:

  # Simple: three classes, 9 ligands each, default alphabet A–Z0–9
  python find_nonCCD_lig_codes.py \
      --class CLASS_A:X0? \
      --class CLASS_B:X1? \
      --class CLASS_C:T0? \
      --target-per-class 9 \
      --max-workers 32

  # Search a larger space with limited alphabet for wildcards
  python find_nonCCD_lig_codes.py \
      --class SMALL:Q?? \
      --letters ABC123 \
      --target-per-class 20 \
      --max-workers 64 \
      --batch-size 128
"""

import argparse
import itertools
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

BASE_URL = "https://files.rcsb.org/ligands/view/{code}.cif"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Search for 3-letter ligand codes that are NOT in the CCD, in parallel."
    )
    p.add_argument(
        "--class", dest="classes", action="append", required=True,
        help=(
            "Class specification of the form NAME:PATTERN, where PATTERN is 3 characters "
            "using A–Z, 0–9 and '?' as a wildcard. "
            "Example: CLASS_A:X0? or TS:T??. "
            "Can be given multiple times."
        ),
    )
    p.add_argument(
        "--letters", default="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        help=(
            "Characters to use when expanding wildcard '?' positions. "
            "Default: A–Z and 0–9."
        ),
    )
    p.add_argument(
        "--target-per-class", type=int, default=9,
        help="Number of unused codes to find per class before stopping that class (default: 9).",
    )
    p.add_argument(
        "--max-workers", type=int, default=32,
        help="Maximum number of parallel HTTP workers (default: 32).",
    )
    p.add_argument(
        "--batch-size", type=int, default=64,
        help="How many candidate codes to test in one batch per class (default: 64).",
    )
    p.add_argument(
        "--timeout", type=float, default=4.0,
        help="HTTP timeout (seconds) for each CCD query (default: 4.0).",
    )
    p.add_argument(
        "--sleep-between-classes", type=float, default=0.0,
        help="Optional sleep (seconds) between finishing one class and starting the next.",
    )
    p.add_argument(
        "--log-every", type=int, default=500,
        help="Print a progress message every N codes checked per class (default: 500).",
    )
    p.add_argument(
        "--no-verify-ssl", action="store_true",
        help="Disable SSL verification (not recommended, but available if needed).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_class_spec(spec):
    """
    Parse NAME:PATTERN into (name, pattern_list).
    PATTERN must be exactly 3 characters and may contain '?' wildcards.
    """
    if ":" not in spec:
        raise ValueError(f"Class spec '{spec}' must be of the form NAME:PATTERN")
    name, pattern = spec.split(":", 1)
    name = name.strip()
    pattern = pattern.strip().upper()

    if len(pattern) != 3:
        raise ValueError(f"Pattern '{pattern}' in '{spec}' must be exactly 3 characters.")

    for ch in pattern:
        if not (ch.isalnum() or ch == "?"):
            raise ValueError(
                f"Invalid character '{ch}' in pattern '{pattern}'. "
                "Use A–Z, 0–9, or '?' as wildcard."
            )

    # Represent pattern as a list where '?' -> None for easy expansion
    patt_list = [None if c == "?" else c for c in pattern]
    return name, patt_list


def codes_from_pattern(pattern, letters):
    """
    Given a pattern like ['X', '0', None] and letters 'ABC...',
    generate all 3-character codes with wildcard positions expanded.
    """
    slots = []
    for ch in pattern:
        if ch is None:
            slots.append(letters)
        else:
            slots.append(ch)

    for combo in itertools.product(*slots):
        yield "".join(combo)


def code_exists_in_ccd(code, timeout=4.0, verify=True):
    """
    Return True if 'code' exists in CCD (HTTP 200), False if not (HTTP 404).
    On any request error, be conservative and treat as 'exists'.
    """
    url = BASE_URL.format(code=code)
    try:
        resp = requests.get(url, timeout=timeout, verify=verify)
        # 200 -> exists; 404 -> does not exist
        if resp.status_code == 200:
            return True
        if resp.status_code == 404:
            return False
        # Other status codes: be conservative
        print(f"[WARN] Unexpected status {resp.status_code} for code {code}; treating as 'exists'.")
        return True
    except requests.RequestException as e:
        print(f"[WARN] Request error for {code}: {e}; treating as 'exists'.")
        return True


def batched(iterable, batch_size):
    """
    Yield lists of up to batch_size items from iterable.
    """
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# ---------------------------------------------------------------------------
# Main search logic per class
# ---------------------------------------------------------------------------

def search_class(name, pattern, letters, target_per_class, max_workers,
                 batch_size, timeout, verify_ssl, log_every):
    """
    Search codes for a single class, returning a list of unused codes.

    name              - class name (string)
    pattern           - list like ['X','0',None]
    letters           - wildcard expansion alphabet
    target_per_class  - how many codes to collect
    max_workers       - max threads in pool
    batch_size        - how many codes per batch
    timeout           - HTTP timeout
    verify_ssl        - bool, passed to requests.get
    log_every         - print progress every N codes checked
    """
    print(f"[INFO] Searching for codes for {name} with pattern {pattern}...")
    codes_iter = codes_from_pattern(pattern, letters)

    found = []
    checked = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for batch in batched(codes_iter, batch_size):
            # Submit batch of codes
            future_to_code = {
                executor.submit(code_exists_in_ccd, code, timeout, verify_ssl): code
                for code in batch
            }

            for future in as_completed(future_to_code):
                code = future_to_code[future]
                exists = future.result()
                checked += 1

                if not exists:
                    found.append(code)
                    print(f"  -> {name}: found unused code {code}")
                    if len(found) >= target_per_class:
                        print(
                            f"[INFO] Done {name}: reached target {target_per_class} "
                            f"unused codes after checking {checked} candidates."
                        )
                        return found

                if log_every and checked % log_every == 0:
                    print(f"[INFO] {name}: checked {checked} codes so far, found {len(found)} unused.")

    print(
        f"[INFO] Exhausted all codes for {name} without reaching target "
        f"({len(found)} found, {target_per_class} requested)."
    )
    return found


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Parse class specs
    try:
        class_specs = [parse_class_spec(spec) for spec in args.classes]
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    letters = args.letters.upper()
    if not letters:
        print("[ERROR] --letters cannot be empty.", file=sys.stderr)
        sys.exit(1)

    # Check that letters are valid
    for ch in letters:
        if not ch.isalnum():
            print(
                f"[ERROR] Invalid character '{ch}' in --letters. Use only A–Z and 0–9.",
                file=sys.stderr,
            )
            sys.exit(1)

    verify_ssl = not args.no_verify_ssl

    print("[INFO] Configuration:")
    print(f"  Classes: {[name for name, _ in class_specs]}")
    print(f"  Letters for '?': {letters}")
    print(f"  Target per class: {args.target_per_class}")
    print(f"  Max workers: {args.max_workers}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Timeout: {args.timeout}s")
    print(f"  SSL verify: {verify_ssl}")
    print("")

    results = {}
    for i, (name, pattern) in enumerate(class_specs, start=1):
        found = search_class(
            name=name,
            pattern=pattern,
            letters=letters,
            target_per_class=args.target_per_class,
            max_workers=args.max_workers,
            batch_size=args.batch_size,
            timeout=args.timeout,
            verify_ssl=verify_ssl,
            log_every=args.log_every,
        )
        results[name] = found

        if i < len(class_specs) and args.sleep_between_classes > 0:
            time.sleep(args.sleep_between_classes)

    print("\n=== SUMMARY OF SUGGESTED CODES ===")
    for name, codes in results.items():
        if codes:
            print(f"{name}: {', '.join(codes)}")
        else:
            print(f"{name}: (none found)")


if __name__ == "__main__":
    main()
