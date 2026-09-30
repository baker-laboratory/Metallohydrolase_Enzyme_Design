"""Regression tests for PLACER aggregation, without requiring licensed PyRosetta.

The script initializes PyRosetta and starts workers at import time. Load its
actual function definitions without those CLI side effects and provide small
pose doubles for the residue identity/coordinate APIs used by aggregation.
Run with: python -m pytest Scripts/tests/test_process_placer.py
"""

import ast
import math
import os
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def load_functions():
    source = Path(__file__).resolve().parents[1] / "process_placer.py"
    tree = ast.parse(source.read_text())
    functions = ast.Module(
        body=[node for node in tree.body if isinstance(node, ast.FunctionDef)],
        type_ignores=[],
    )
    namespace = {"pd": pd, "np": np, "math": math, "os": os}
    exec(compile(functions, str(source), "exec"), namespace)
    return namespace


class Residue:
    def __init__(self, chain, number, name, coordinate):
        self.chain, self.number = chain, number
        self.name, self.coordinate = name, coordinate

    def name3(self):
        return self.name

    def seqpos(self):
        return self.position

    def natoms(self):
        return 1

    def atom_name(self, number):
        return "CA"

    def atom_is_hydrogen(self, number):
        return False

    def has(self, name):
        return name == "CA"

    def xyz(self, name):
        return np.array([self.coordinate, 0., 0.])

    def is_ligand(self):
        return self.name == "LIG"


class Pose:
    def __init__(self, residues):
        self.residues = residues
        for position, residue in enumerate(residues, 1):
            residue.position = position

    def clone(self):
        return self

    def pdb_info(self):
        return self

    def pdb2pose(self, chain, number):
        residue = self.pdb_rsd((chain, number))
        return residue.seqpos() if residue is not None else 0

    def pdb_rsd(self, identity):
        return next((r for r in self.residues if (r.chain, r.number) == identity), None)

    def residue(self, position):
        assert 1 <= position <= len(self.residues)
        return self.residues[position - 1]

    def chain(self, position):
        return self.residue(position).chain

    def size(self):
        return len(self.residues)


def atom_line(chain, number, uncertainty):
    return (f"ATOM      1  CA  HIS {chain}{number:4d}    "
            f"{0.:8.3f}{0.:8.3f}{0.:8.3f}{1.:6.2f}{uncertainty:6.2f}           C\n")


def make_case(tmp_path, top=2, missing_in_one_model=False, missing_ligand=None):
    functions = load_functions()
    ref_pose = Pose([
        Residue("A", 1, "HIS", 0.),
        Residue("A", 42, "HIS", 0.),
        Residue("B", 42, "HIS", 0.),
        Residue("B", 50, "HIS", 0.),
        Residue("Z", 1, "LIG", 0.),
    ])
    if missing_ligand == "reference":
        ref_pose.residues[-1].name = "OTHER"
    poses, models = [], []
    for j in range(3):
        residues = [Residue("A", 1, "HIS", j + .1), Residue("A", 42, "HIS", 100.)]
        lines = atom_line("A", 1, 7.) + atom_line("A", 42, 99.)
        if not (missing_in_one_model and j == 1):
            residues.append(Residue("B", 42, "HIS", j + 2.))
            lines += atom_line("B", 42, 10. * (j + 1))
        residues.append(Residue("Z", 1, "LIG", j + 1.))
        if missing_ligand == "model2" and j == 1:
            residues[-1].name = "OTHER"
        poses.append(Pose(residues))
        models.append(f"MODEL {j + 1}\n" + lines)
    (tmp_path / "design_model.pdb").write_text("ENDMDL\n".join(models) + "ENDMDL\n")

    # A similarly named design comes first; target rows retain nonzero indices
    # and deliberately disagree with PDB model order.
    scores = pd.DataFrame({
        "label": ["design_extra"] * 3 + ["design"] * 3,
        "model_idx": [1, 2, 3, 3, 1, 2],
        "kabsch": [100., 100., 100., 3., 1., 2.],
        "prmsd": [100., 100., 100., 6., 2., 4.],
        "plddt": [100., 100., 100., 99., 60., 80.],
        "plddt_pde": [100., 100., 100., 30., 10., 20.],
        "lddt": [100., 100., 100., 30., 10., 20.],
    })
    functions.update({
        "scores": scores,
        "args": SimpleNamespace(top=top, placer_pdb_path=str(tmp_path), lig_name="LIG", lig_atom=None),
        "ref_pdbs": {"design": "reference.pdb"},
        "pyr": SimpleNamespace(pose_from_file=lambda path: ref_pose),
        "design_utils": SimpleNamespace(get_matcher_residues=lambda path: {
            42: {"chain": "B", "name3": "HIS"},
            50: {"chain": "B", "name3": "HIS"},
        }),
        "load_poses": lambda models: poses,
        "get_pocket_residues": lambda pose: ([1], [1]),
        "results": {},
    })
    return functions, scores


@pytest.mark.parametrize("top", [1, 2, 3])
@pytest.mark.parametrize("missing_in_one_model", [False, True])
def test_aggregation_uses_model_identity_and_includes_second_shell(
    tmp_path, capsys, top, missing_in_one_model
):
    functions, scores = make_case(tmp_path, top, missing_in_one_model)
    queue = Queue()
    queue.put((1, "design"))
    queue.put(None)
    functions["process"](queue)
    result = functions["results"][1].loc[1]

    assert result["kabsch"] == 2.
    assert result["prmsd_avr"] == 4.
    assert result["prmsd_min"] == 2.
    assert result[f"prmsd_top{top}"] == np.mean([2., 4., 6.][:top])
    assert result[f"rmsd_prmsd_top{top}"] == np.mean([1., 2., 3.][:top])
    assert result[f"plddt_top{top}"] == np.mean([99., 80., 60.][:top])
    assert result[f"rmsd_pde_top{top}"] == np.mean([3., 2., 1.][:top])
    assert result["rmsd_ligand"] == 2.
    assert np.isfinite(result["prmsd_pnear"])
    assert result["rmsd_catres0"] == 3.
    assert result["u_catres0"] == 20.
    assert result["u_pocket"] == 7.
    assert result["rmsd_std_catres0"] == pytest.approx(
        np.std([2., 4.] if missing_in_one_model else [2., 3., 4.])
    )
    assert np.isnan(result["rmsd_catres1"])
    assert np.isnan(result["u_catres1"])
    assert "catalytic residue B50" in capsys.readouterr().out
    assert "rmsd_ligand" not in scores


@pytest.mark.parametrize("missing_ligand, message", [
    ("reference", "reference for design design"),
    ("model2", "design design, model 2"),
])
def test_missing_ligand_fails_instead_of_reusing_previous_model_position(
    tmp_path, missing_ligand, message
):
    functions, _ = make_case(tmp_path, missing_ligand=missing_ligand)
    queue = Queue()
    queue.put((1, "design"))
    queue.put(None)
    with pytest.raises(ValueError, match=message):
        functions["process"](queue)
    assert functions["results"] == {}


def test_residue_uncertainty_ignores_other_chains_and_non_atom_records():
    score = load_functions()["get_residue_score"]
    model = atom_line("B", 42, 10.) + atom_line("B", 42, 30.) + atom_line("A", 42, 99.)
    model += "REMARK B  42 is a catalytic residue\n"
    assert score(model, "B", 42) == 20.
    assert np.isnan(score(model, "B", 50))


def test_missing_atoms_do_not_raise_or_produce_zero_rmsd():
    rmsd = load_functions()["get_residue_rmsd"]
    ref = Residue("B", 42, "HIS", 0.)
    model = Residue("B", 42, "HIS", 1.)
    model.has = lambda name: False
    assert np.isnan(rmsd(ref, model))
