# Transition-state model preparation

The 96 structures in the model archive all match the final ordered protein
sequences and include the original butyrate transition-state ligand and Zn.
BRM/BSM and BRZ/BSZ are the four original ligand residue types; zinc is atom
`ZN1` within that residue, rather than a separate `ZN` residue.

## How the historical hit models were made

The archived design notebook
`old_4muButyrate_stuff/20240402_design_campaign1/design_campaign1_zn_hydrolase.ipynb`
contains the relevant steps (cell indices below are zero-based):

1. Cell 112 aligns apo-relaxed second-round AF2 predictions against their
   ligand-containing design inputs with
   `align_af2_with_inputs_and_copy_ligand.py --fix_catres`. It transfers the
   original TS, matcher remarks and catalytic histidine rotamers/protonation.
   `STRUCTURES_FOR_AHERN_PAPER/apo_relaxed_i2_af2_withALIGNMENT/ALL_ORDERED_PDBS`
   preserves the 96 intermediate complexes.
2. Cells 189–190 select and record the final order and explicitly document
   manual sequence changes and terminal trimming before gene ordering.
   Consequently, 21 of the 96 intermediate TS models differ from their final
   ordered sequences, even though their order-design names match.
3. Cells 194–196 run `constrained_fastRELAX_simple.py` on the three chosen hits,
   then repeat on the output twice. Each pass uses two FastRelax repeats,
   ref2015 scoring, enzyme-design distance/angle/dihedral constraints weighted
   at 1.0, coordinate restraints to each pass's starting coordinates, repacking
   without sequence design, and movable backbone, side chains and jumps.
4. The final hit coordinates are in
   `STRUCTURES_FOR_AHERN_PAPER/CST_RELAXED_DESIGN_MODELS/final_structures/` and
   were renamed for the manuscript in
   `paper_hits/for_rfd2_manuscript/design_models/`. The archive uses these
   publication B11, C4 and D11 files unchanged.

## Supplementary reconstructions

For each of the 93 remaining designs, the included preparation scripts:

1. Align the original pre-order reference sequence to the final ordered apo
   sequence, accounting for residue changes and terminal deletions.
2. Superpose matched C-alpha coordinates, transfer the original TS residue,
   and map all three catalytic histidine identifiers. PDB residue numbers
   retained after trimming are converted to Rosetta pose indices explicitly.
3. Restore the reference catalytic histidine rotamers/protonation and apply
   the three-pass constrained FastRelax protocol described above.

These are new reconstructions made for this supplement. A fixed seed is recorded
for each job, but versions/platforms may change stochastic relaxation results.
The run used PyRosetta 2025.33, release `a492b89a9e4fe490a50080b9b02cb78ba1dcf14c`,
on CPU. Sources, seeds, alignments and exact final-coordinate checksums are
recorded in the input archive and model manifest. Source directories were not
modified. Sequence identity, ligand atom inventory, all three matcher histidines,
Zn distances and minimum protein–ligand heavy-atom separation are validated
before packaging; the manifest retains the numeric geometry diagnostics.

The parameters and constraints are the original files from
`STRUCTURES_FOR_AHERN_PAPER`. Some comment headers in those historical constraint
files mention phenylacetate; their actual `residue3` mappings and ligand
parameters are the butyrate BRM/BRZ/BSM/BSZ inputs used here.

## Inspect or rerun

Unzip [reconstruction_inputs.zip](reconstruction_inputs.zip). It contains the
93 ordered apo PDBs, their pre-order TS references, aligned pre-relaxation
complexes and task JSON files. JSON paths are relative to each task file.

To regenerate one aligned input with Python, NumPy and Biopython:

```bash
python align_ordered_model.py \
  --ordered-apo reconstruction_inputs/ordered_apo/A6.pdb \
  --reference reconstruction_inputs/references/A6.pdb \
  --output A6_aligned.pdb --relaxed-output A6_relaxed.pdb --task A6_task.json
```

To rerun the packaged alignment through constrained relaxation, use a Python
environment with a licensed PyRosetta installation:

```bash
python relax_ordered_model.py --task reconstruction_inputs/tasks/00_A6.json
```

The three passes are saved separately and the final PDB is written to the output
path in the task. The script strips serialized metal `LINK`/`CONECT` records
between passes before restoring enzyme constraints, so current Rosetta does not
create the same coordination pseudobonds twice. It never designs new amino
acid identities, and rejects any change in the ordered protein sequence.

Validate all packaged coordinates, sequences and ligand inventories without
installing PyRosetta:

```bash
python validate_model_archive.py
```

The manifest also reports Zn–histidine distances and the closest heavy-atom
contact. These diagnostics are not activity filters: most of the 96 ordered
designs were not hits, and some relaxed non-hit structures retain nonideal
coordination geometry. No model is silently discarded or made to appear
experimentally validated on the basis of its relaxation result.
