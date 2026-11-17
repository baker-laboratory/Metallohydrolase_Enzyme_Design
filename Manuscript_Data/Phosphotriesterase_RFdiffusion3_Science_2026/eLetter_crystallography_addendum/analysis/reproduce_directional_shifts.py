#!/usr/bin/env python3
"""Reproduce the pooled sidechain shifts in Table 1 from the aligned models.

The models must already be superposed on all shared C-alpha atoms as described
in the report. This script does not predict structures or refine density.
"""

import argparse
import csv
from math import sqrt
from pathlib import Path
import sys

SIDECHAIN_ATOMS = {
    16: ('CB', 'CG', 'CD', 'CE', 'NZ', 'CX', 'OQ1', 'OQ2'),
    89: ('CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'),
    92: ('CB', 'CG', 'CD', 'OE1', 'OE2'),
    93: ('CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'),
    133: ('CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'),
    170: ('CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'),
}
ATOM_KEYS = [(residue, name) for residue, names in SIDECHAIN_ATOMS.items() for name in names]
DESIGN_CARBAMATE = {'C1': 'CX', 'O1': 'OQ1', 'O2': 'OQ2'}
EQUIVALENT_OXYGEN_PAIRS = (((16, 'OQ1'), (16, 'OQ2')), ((92, 'OE1'), (92, 'OE2')))


def read_atoms(path, chain, design=False):
    atoms = {}
    for line in path.read_text().splitlines():
        if line[:6].strip() not in ('ATOM', 'HETATM') or line[16] not in (' ', 'A'):
            continue
        name, residue, atom_chain = line[12:16].strip(), line[22:26].strip(), line[21]
        if design and atom_chain == 'B' and residue == '199' and name in DESIGN_CARBAMATE:
            key = (16, DESIGN_CARBAMATE[name])
        elif atom_chain == chain and residue.isdigit():
            key = (int(residue), name)
        else:
            continue
        if key in ATOM_KEYS:
            if key in atoms:
                raise ValueError(f'duplicate selected atom {key} in {path}')
            atoms[key] = tuple(float(line[i:i + 8]) for i in (30, 38, 46))
    if set(atoms) != set(ATOM_KEYS):
        raise ValueError(f'{path}: missing selected atoms {sorted(set(ATOM_KEYS) - set(atoms))}')
    return atoms


def difference(a, b):
    return tuple(x - y for x, y in zip(a, b))


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def assign_equivalent_oxygens(atoms, reference):
    for first, second in EQUIVALENT_OXYGEN_PAIRS:
        direct = sum(dot(difference(atoms[k], reference[k]), difference(atoms[k], reference[k]))
                     for k in (first, second))
        swapped = sum(dot(difference(atoms[k], reference[j]), difference(atoms[k], reference[j]))
                      for k, j in ((first, second), (second, first)))
        if swapped < direct:
            atoms[first], atoms[second] = atoms[second], atoms[first]
    return atoms


def calculate(aligned):
    reference = read_atoms(aligned / 'design_aligned.pdb', 'A', design=True)
    predictions = {
        'bound': assign_equivalent_oxygens(read_atoms(aligned / 'af3_bound_aligned.pdb', 'A'), reference),
        'water': assign_equivalent_oxygens(read_atoms(aligned / 'af3_water_sample0_aligned.pdb', 'A'), reference),
    }
    copies = (
        ('form1_chainA', 'form1_chainA_aligned.pdb', 'A'),
        ('form1_chainB', 'form1_chainB_aligned.pdb', 'B'),
        ('form2_chainA', 'form2_chainA_aligned.pdb', 'A'),
    )
    for label, filename, chain in copies:
        crystal = assign_equivalent_oxygens(read_atoms(aligned / filename, chain), reference)
        crystal_displacement = [difference(crystal[k], reference[k]) for k in ATOM_KEYS]
        norm2 = sum(dot(v, v) for v in crystal_displacement)
        if norm2 == 0:
            raise ValueError(f'zero design-to-crystal displacement: {label}')
        crystal_rmsd = sqrt(norm2 / len(ATOM_KEYS))
        shifts = []
        for prediction in predictions.values():
            predicted_displacement = [difference(prediction[k], reference[k]) for k in ATOM_KEYS]
            projection = sum(dot(p, c) for p, c in zip(predicted_displacement, crystal_displacement))
            shifts.append(projection / sqrt(len(ATOM_KEYS) * norm2))
        yield (label, crystal_rmsd, *shifts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aligned-dir', type=Path,
                        default=Path(__file__).resolve().parent.parent / 'structures' / 'aligned',
                        help='Directory of design-frame aligned models (default: ../structures/aligned).')
    parser.add_argument('--output', type=Path, help='CSV destination; omit to write to standard output.')
    args = parser.parse_args()
    stream = args.output.open('w', newline='') if args.output else sys.stdout
    try:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(('crystal', 'crystal_rmsd_A', 'bound_shift_A', 'water_shift_A'))
        for label, crystal_rmsd, bound, water in calculate(args.aligned_dir):
            writer.writerow((label, *(f'{v:.6f}' for v in (crystal_rmsd, bound, water))))
    finally:
        if args.output:
            stream.close()


if __name__ == '__main__':
    main()
