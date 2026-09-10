# Phosphotriesterase design with RFdiffusion3

> **Status: in preparation.** This campaign is being prepared for release
> alongside RFdiffusion3, and will be published here shortly.

De novo design of binuclear phosphotriesterases that hydrolyze paraoxon, using
**RFdiffusion3** for backbone generation. The active site is a two-zinc center
with a carboxylated lysine (KCX) bridge, taken from the best hit of the first
(RFdiffusion2) design campaign and refined by machine-learning force field
calculation.

**Notebook:** [`design_phosphotriesterase.ipynb`](design_phosphotriesterase.ipynb)

## Workflow

| Section | Stage |
|---|---|
| I–II | Theozyme post-processing: ligand naming and `.params`, catalytic-residue sampling, ORI tokens, constraint and scoring files |
| III | RFdiffusion3 backbone generation at scale via the open-source `rfd3 design` CLI, then rapid JSON-based filtering and PyRosetta scaffold analysis |
| IV | Predesign — Cartesian relaxation for geometry idealization, then comprehensive metric filtering |
| V | Sequence design with LigandMPNN, orchestrated: catalytic residues held fixed from REMARK 666, optional H-bond second-shell conservation, a temperature sweep, side-chain packing, and protonation |
| VI, IX, XIII | AlphaFold3 validation — apo, then holo, then holo after redesign |
| VII | Foldseek structural clustering |
| VIII, XI | LigandMPNN redesign of validated designs, catalytic site held fixed |
| X, XII | Alignment and deduplication |
| XIV | Apo-monomer prediction with AlphaFold2 (SuperFold) |

Most cells are **driver cells**: they assemble a command, print it, and write it
to `cmds/`. You run it in a terminal or submit it to SLURM. That keeps every
command inspectable and individually testable.

## Inputs

```
inputs/theozymes/
├── p1D1_mlff/               # MLFF (NEB transition state) calculation the theozyme derives from
├── step1_theozyme_prep/     # cropped active site, two phosphorus rotamers
├── step2_theozyme_prep_ori/ # 22 ORI-token PDBs -- the RFdiffusion3 inputs
├── params/                  # YYE ligand: .params, .mol2, conformers
├── cst_files/               # ZAPP_p1D1.cst
└── ref_from_i1/             # reference from the first design campaign
```

The ligand is `YYE` (the paraoxon transition state), with `ZN` and a KCX
post-translational modification declared to AlphaFold3.

## Setup

```bash
conda env create -f ../../Environment/zinc_hydro.yml
conda activate zinc_hydro
python ../../Scripts/env_config.py     # confirm the interpreter and tools resolve
```

The notebook locates the repository itself — there are no paths to edit. Outputs
default to `outputs/` inside this directory; point `_OUTPUT_DIR_OVERRIDE` at
scratch for a production run, since they are large.

### External prerequisites

Several stages call software that is not bundled here, either because it carries
its own license or because it is too large to vendor. Each is read from an
environment variable, so nothing is hardcoded:

| Variable | What it points at | Needed for |
|---|---|---|
| `ZINC_HYDRO_SIF` | Apptainer image, if you use one | any containerised step (unset = run with the active python) |
| `FOUNDRY_CHECKPOINT_DIRS` | extra RFdiffusion3 checkpoint directories | Section III (optional; foundry also reads `~/.foundry/checkpoints`) |
| `AF3_RUNNER`, `AF3_SIF` | AlphaFold3 entrypoint and container | Sections VI, IX, XIII |
| `LIGANDMPNN_WEIGHTS` | directory of LigandMPNN weights — **you download these** | Section V |
| `MAXIT_SIF` | maxit, for CIF→PDB conversion | AF3 output processing |
| `ROSETTA` | Rosetta root, for `molfile_to_params.py` | ligand `.params` generation |
| `FOLDSEEK_ROOT` | Foldseek checkout | Section VII (defaults to `Software/foldseek`) |

**RFdiffusion3** runs through the `rfd3 design` CLI from
[`Software/foundry`](../../Software/foundry) (branch `production`) &mdash; the same
open-source entry point as the tutorial, with no in-house dependency. Install it
once with `pip install -e "Software/foundry[rfd3]"` (or `pip install
"rc-foundry[rfd3]"`), then fetch weights with `foundry install rfd3`. Foundry
discovers checkpoints automatically, so the notebook leaves `ckpt_path` unset by
default.

### Sequence design

Section V drives
[`Scripts/mpnn_relevant_utils/design_orchestrator.py`](../../Scripts/mpnn_relevant_utils/design_orchestrator.py),
which wraps LigandMPNN with the setup and cleanup enzyme design needs. One
command per input structure:

| Stage | What it does |
|---|---|
| Before | Holds the REMARK 666 catalytic residues fixed, so there is no per-structure fixed-residues JSON to maintain. Omits Met at residue 1. Optionally conserves designable side chains that hydrogen-bond the active site, rolled per combination so the second shell varies between designs. |
| During | Sweeps temperature and batch settings in one invocation, each combination tagged so outputs never collide. |
| After | Flattens every run into one directory of packed PDBs, protonates them, restores REMARK 666, writes a REMARK 668 protonation-state block, and records where each design came from. |

Two details worth knowing:

**Protonation is not cosmetic.** LigandMPNN emits heavy atoms only — on its own
packed output the only hydrogens present are the ligand's, carried through from
the input, and the protein has none. Anything reasoning about hydrogen bonding
or tautomers downstream needs them. With a `.params` the ligand is protonated
too; without one the protein is protonated apo and the ligand block is copied
back from the input verbatim, so an unparameterized ligand costs nothing but its
own hydrogens.

**Catalytic tautomers are restored, not guessed.** A heavy-atom histidine has no
tautomer, so PyRosetta would give every catalytic HIS its default. The states
are read from the input structure and re-applied, so the geometry the theozyme
was built around survives. Declare modifications that have no hydrogens to
detect — a carboxylated lysine, say — with `--ptm A/LYS/3:KCX`.

Verified by
[`test_design_orchestrator.py`](../../Scripts/mpnn_relevant_utils/test_design_orchestrator.py),
which runs against real structures from this repository and checks every emitted
flag against LigandMPNN's own argparse. It needs no GPU and no model weights;
the protonation test skips itself when PyRosetta is unavailable.

**LigandMPNN** runs from the copy bundled with this repository, at
`Software/fastmpnndesign/lib/LigandMPNN` (a nested submodule). Initialize it
once:

```bash
git submodule update --init --recursive Software/fastmpnndesign
```

**Model weights are not bundled and must be downloaded separately.** The
repository ships the downloader that comes with LigandMPNN:

```bash
bash Software/fastmpnndesign/lib/LigandMPNN/get_model_params.sh model_weights/
export LIGANDMPNN_WEIGHTS=$PWD/model_weights
```

`model_weights/` is gitignored, and sits outside the submodules so they stay
clean.

That fetches the ProteinMPNN, LigandMPNN and SolubleMPNN checkpoints at several
training-noise levels, plus `ligandmpnn_sc_v_32_002_16.pt`, the multi-step
denoising model used for side-chain packing. Set `ligandmpnn_checkpoint` in the
notebook to the file matching your `model_type`.

> **Fixed residues keep their exact coordinates.** Commands go through
> [`Scripts/mpnn_relevant_utils/run_ligandmpnn.py`](../../Scripts/mpnn_relevant_utils/run_ligandmpnn.py)
> rather than LigandMPNN's `run.py` directly. Stock LigandMPNN preserves the
> *torsions* of fixed residues under `--repack_everything 0`, but then rebuilds
> every residue's coordinates at idealized literature geometry, so a catalytic
> residue taken from a crystal structure or a QM theozyme comes back subtly
> moved. The wrapper writes the exact input coordinates back over the
> reconstruction, at both points where it happens, and additionally makes the
> fixed residues' side-chain atoms visible to the packer as context so
> neighboring rotamers are packed around them rather than into them.
> Arguments are otherwise unchanged; pass `--no_patch` for stock behavior.
>
> Verified with
> [`Scripts/mpnn_relevant_utils/test_fixed_residue_preservation.py`](../../Scripts/mpnn_relevant_utils/test_fixed_residue_preservation.py),
> which runs stock and patched side by side on a real structure and reports the
> largest displacement of the fixed residues:
>
> ```
>   STOCK   LigandMPNN : max fixed-residue displacement = 0.2918 A
>   PATCHED            : max fixed-residue displacement = 0.0000 A
> ```
>
> The wrapper also carries two `.long()` dtype fixes that stock LigandMPNN needs
> to run `--pack_side_chains 1` at all on current PyTorch.
>
> The added context is measured by
> [`test_fixed_residue_clashes.py`](../../Scripts/mpnn_relevant_utils/test_fixed_residue_clashes.py).
> Across 96 packed structures with the three catalytic histidines held fixed:
>
> | | with context | without |
> |---|---|---|
> | clashing atom pairs | 1183 | 1253 |
> | structures with ≥1 clash | 53 / 96 | 54 / 96 |
> | mean closest approach | 2.06 Å | 1.98 Å |
>
> So it helps, by about 6 % — but it does **not** solve the problem: designed
> side chains still clash into the fixed catalytic residues in over half the
> structures. Treat MPNN-packed side chains as a starting point for Rosetta
> repacking rather than a finished model. Set
> `LIGANDMPNN_NO_FIXED_CONTEXT=1` to disable this step and reproduce the
> right-hand column.

**AlphaFold3.** The source is pinned as a submodule at
[`Software/alphafold3`](../../Software/alphafold3) (Apache-2.0). The **model
parameters are not included and cannot be redistributed**: they are requested
from Google DeepMind separately and are granted under terms that restrict use,
including a prohibition on commercial use. Obtain them yourself and point
`AF3_RUNNER` at your installation. The AF3 sections are validation steps — the
design pipeline through Section V runs without them.

**Redesign rounds (VIII and XI)** run the same orchestrator as Section V, on
structures that have already been validated by AlphaFold3. Designs are named
`<input stem>_<combo tag>_<batch>_<seq>.pdb`, which the AlphaFold3 matching
cells parse to recover the input each design came from.

New sequence design methods can also be applied at these steps &mdash; for
example [protein_chisel](https://github.com/SethWoodbury/protein_chisel).

## Related

- First campaign (RFdiffusion2, zinc esterase): [`../Metalloesterase_RFdiffusion2/`](../Metalloesterase_RFdiffusion2/)
- Method tutorial: [`../../RFdiffusion3_Tutorial/`](../../RFdiffusion3_Tutorial/)
- Experimental data: [`../../Manuscript_Data/Phosphotriesterase_RFdiffusion3/`](../../Manuscript_Data/Phosphotriesterase_RFdiffusion3/)
