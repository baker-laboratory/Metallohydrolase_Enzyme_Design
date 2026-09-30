"""Map an ordered apo model to its archived pre-order transition-state complex.

Unlike a position-only alignment, sequence alignment handles terminal trimming.
Coordinates are superposed on matched C-alpha atoms; the archived ligand and
the three histidine constraint records are transferred. This is an intermediate
for relax_ordered_model.py, not a final relaxed design model.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from Bio.Align import PairwiseAligner
from Bio.SeqUtils import seq1


def residues(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("ATOM  "):
            if line[21] != "A":
                raise ValueError("Expected one protein chain, A")
            result.setdefault(int(line[22:26]), []).append(line)
    return result


def sequence(records):
    return "".join(seq1(lines[0][17:20]) for lines in records.values())


def prepare(ordered_apo, reference, output, relaxed_output, task_path, seed):
    ordered_apo, reference, output = map(Path, (ordered_apo, reference, output))
    ref_res, apo_res = residues(reference), residues(ordered_apo)
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score, aligner.mismatch_score = 2, -1
    aligner.open_gap_score, aligner.extend_gap_score = -8, -0.5
    alignment = aligner.align(sequence(ref_res), sequence(apo_res))[0]
    mapping = {}
    for ref_block, apo_block in zip(*alignment.aligned):
        for i, j in zip(range(*ref_block), range(*apo_block)):
            mapping[list(ref_res)[i]] = list(apo_res)[j]

    def ca(lines):
        line = next(line for line in lines if line[12:16].strip() == "CA")
        return np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])

    x = np.array([ca(apo_res[j]) for i, j in mapping.items()])
    y = np.array([ca(ref_res[i]) for i, j in mapping.items()])
    xc, yc = x.mean(0), y.mean(0)
    u, _, vt = np.linalg.svd((x - xc).T @ (y - yc))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    rmsd = float(np.sqrt(np.mean(np.sum(((x - xc) @ rotation + yc - y) ** 2, axis=1))))

    reference_lines = reference.read_text().splitlines()
    ligands = {line[17:20] for line in reference_lines if line.startswith("HETATM")}
    if len(ligands) != 1 or not ligands.issubset({"BRM", "BRZ", "BSM", "BSZ"}):
        raise ValueError(f"Expected one butyrate transition-state residue, found {ligands}")
    ligand = next(iter(ligands))
    remarks, catalytic_mapping = [], []
    for line in reference_lines:
        if line.startswith("REMARK 666"):
            fields = line.split()
            old = int(fields[11])
            new = mapping[old]
            if apo_res[new][0][17:20] != "HIS":
                raise ValueError("An ordered sequence changes a catalytic histidine")
            remarks.append(
                f"REMARK 666 MATCH TEMPLATE X {ligand}    0 MATCH MOTIF A HIS {new:4d}  {fields[12]}  {fields[13]}"
            )
            catalytic_mapping.append([old, new])
    if len(catalytic_mapping) != 3:
        raise ValueError("Expected three coordinating histidines")
    atoms = []
    for lines in apo_res.values():
        for line in lines:
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            xyz = (xyz - xc) @ rotation + yc
            atoms.append(line[:30] + "".join(f"{v:8.3f}" for v in xyz) + line[54:])
    for line in reference_lines:
        if line.startswith("HETATM"):
            atoms.append(line[:21] + "X" + f"{len(apo_res) + 1:4d}" + line[26:])
    atoms = [line[:6] + f"{i:5d}" + line[11:] for i, line in enumerate(atoms, 1)]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(["HEADER    RECONSTRUCTED ORDERED SEQUENCE TS COMPLEX"] + remarks + atoms + ["END"]) + "\n")
    task = dict(ligand=ligand, seed=seed, input_pdb=str(output.resolve()),
                reference_pdb=str(reference.resolve()), output_pdb=str(Path(relaxed_output).resolve()),
                catalytic_residue_mapping=catalytic_mapping, alignment_rmsd_A=rmsd)
    Path(task_path).write_text(json.dumps(task, indent=2) + "\n")
    print(f"Aligned {len(mapping)} residues; C-alpha RMSD = {rmsd:.3f} Å")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ordered-apo", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--relaxed-output", required=True)
    parser.add_argument("--task", dest="task_path", required=True)
    parser.add_argument("--seed", type=int, default=20260929)
    prepare(**vars(parser.parse_args()))
