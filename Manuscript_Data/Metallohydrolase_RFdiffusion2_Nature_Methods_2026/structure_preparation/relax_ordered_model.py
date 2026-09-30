"""Repair an ordered-sequence TS complex with the original constrained relaxation.

Requires a licensed PyRosetta installation. Inputs are an aligned ordered apo
model with a transferred ligand, its pre-order TS reference, and a JSON residue
mapping. Three sequential FastRelax passes (two repeats each) reproduce the
protocol used to prepare the three publication hits. This reconstructs an extra
supplemental model; it does not recreate historical stochastic coordinates.
"""

import argparse
import json
from pathlib import Path

import pyrosetta as pyr
from pyrosetta import rosetta as r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="JSON input/output and mapping record")
    args = parser.parse_args()
    task_path = Path(args.task).resolve()
    task = json.loads(task_path.read_text())
    for key in ("input_pdb", "reference_pdb", "output_pdb"):
        task[key] = str((task_path.parent / task[key]).resolve())
    base = Path(__file__).resolve().parent
    ligand = task["ligand"]
    pyr.init(
        f"-extra_res_fa {base / 'params' / (ligand + '.params')} "
        f"-run:preserve_header -constant_seed -jran {task['seed']} -mute all"
    )
    pose = pyr.pose_from_file(task["input_pdb"])
    ref = pyr.pose_from_file(task["reference_pdb"])
    ordered_sequence = pose.sequence()
    # As in align_af2_with_inputs_and_copy_ligand.py --fix_catres, preserve
    # reference histidine protonation and rotamers; account for terminal trims.
    for old, new in task["catalytic_residue_mapping"]:
        # Ordered PDBs retain original residue numbers after terminal trimming;
        # Rosetta pose indices are contiguous and must be resolved separately.
        old = ref.pdb_info().pdb2pose("A", old)
        new = pose.pdb_info().pdb2pose("A", new)
        if not old or not new or pose.residue(new).name3() != "HIS":
            raise ValueError("The mapped catalytic histidine is missing")
        mut = r.protocols.simple_moves.MutateResidue(new, ref.residue(old).name())
        mut.apply(pose)
        for chi in range(1, ref.residue(old).nchi() + 1):
            pose.set_chi(chi, new, ref.chi(chi, old))

    score = pyr.get_fa_scorefxn()
    for term in ("atom_pair_constraint", "angle_constraint", "dihedral_constraint"):
        score.set_weight(r.core.scoring.score_type_from_name(term), 1.0)
    types = r.core.chemical.ChemicalManager.get_instance().residue_type_set("fa_standard")
    output = Path(task["output_pdb"])
    output.parent.mkdir(parents=True, exist_ok=True)
    for pass_index in range(3):
        constraints = r.protocols.toolbox.match_enzdes_util.EnzConstraintIO(types)
        constraints.read_enzyme_cstfile(str(base / "constraints" / f"{ligand}_3H_NO_DEPROT_RES.cst"))
        constraints.add_constraints_to_pose(pose, score, True)
        relax = r.protocols.relax.FastRelax(score, 2)
        relax.constrain_relax_to_start_coords(True)
        relax.ramp_down_constraints(False)
        factory = r.core.pack.task.TaskFactory()
        for operation in ("InitializeFromCommandline", "IncludeCurrent", "NoRepackDisulfides", "RestrictToRepacking"):
            factory.push_back(getattr(r.core.pack.task.operation, operation)())
        relax.set_task_factory(factory)
        movemap = r.core.kinematics.MoveMap()
        movemap.set_chi(True)
        movemap.set_bb(True)
        movemap.set_jump(True)
        relax.set_movemap(movemap)
        relax.apply(pose)
        if pose.sequence() != ordered_sequence:
            raise ValueError("Relaxation changed the ordered protein sequence")
        score(pose)
        intermediate = output.with_name(output.stem + f"_pass{pass_index + 1}.pdb")
        pose.dump_pdb(str(intermediate))
        # The historical workflow saved/reloaded between passes. Modern Rosetta
        # writes the metal coordination pseudobonds as LINK/CONECT records;
        # remove those serialized connections before the enzyme constraints
        # recreate them, otherwise a second pass tries to add them twice.
        pdb_text = "\n".join(
            line for line in intermediate.read_text().splitlines()
            if not line.startswith(("LINK  ", "CONECT"))
        ) + "\n"
        pose = pyr.Pose()
        r.core.import_pose.pose_from_pdbstring(pose, pdb_text)
    pose.dump_pdb(str(output))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
