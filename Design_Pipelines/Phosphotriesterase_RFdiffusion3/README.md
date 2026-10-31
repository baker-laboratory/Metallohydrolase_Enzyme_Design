# Phosphotriesterase design with RFdiffusion3

> **Repository status: driver notebook and inputs available.** Production outputs
> must be generated locally. The campaign accompanies the RFdiffusion3 study
> in *Science* (2026); see the reference below.

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

## Run

Follow the [environment guide](../../Environment/README.md) and
[software/model setup](../../Docs/software-and-models.md). RFdiffusion3 uses
foundry's `rfd3 design` CLI in a separate Python 3.12+ environment; obtain its
checkpoint with `foundry install rfd3`. LigandMPNN weights and AF3 parameters
also require separate setup.

The notebook detects the repository root. Set `_OUTPUT_DIR_OVERRIDE` to scratch
for large runs. Review generated commands: historical container launchers,
`/net` and `/home` binds, and AF3 alternatives need local adaptation. An empty
container variable does not make every command run with host Python.

| Variable | What it points at | Needed for |
|---|---|---|
| `ZINC_HYDRO_SIF` | Apptainer image, if you use one | containerized steps; review each launcher when unset |
| `FOUNDRY_CHECKPOINT_DIRS` | extra RFdiffusion3 checkpoint directories | Section III (optional; foundry also reads `~/.foundry/checkpoints`) |
| `AF3_RUNNER`, `AF3_SIF` | AlphaFold3 entrypoint and container | Sections VI, IX, XIII |
| `LIGANDMPNN_WEIGHTS` | directory of LigandMPNN weights — **you download these** | Section V |
| `MAXIT_SIF` | maxit, for CIF→PDB conversion | AF3 output processing |
| `ROSETTA` | Rosetta root, for `molfile_to_params.py` | ligand `.params` generation |
| `FOLDSEEK_ROOT` | Foldseek checkout | Section VII (defaults to `Software/foldseek`) |

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

**Protonation and tautomers.** LigandMPNN emits protein heavy atoms. The
orchestrator restores protein hydrogens and input catalytic histidine states,
which matter for hydrogen-bond analysis. With ligand `.params`, it also
protonates the ligand; otherwise it protonates the protein apo and copies the
input ligand block unchanged. Declare PTMs explicitly, e.g. `--ptm A/LYS/3:KCX`.

[Orchestrator tests](../../Scripts/mpnn_relevant_utils/test_design_orchestrator.py)
exercise repository structures and emitted flags without GPU/model weights;
the protonation test skips when PyRosetta is unavailable. Initialize the nested
LigandMPNN submodule and download its checkpoints using the
[model setup instructions](../../Docs/software-and-models.md#ligandmpnn).
Match `ligandmpnn_checkpoint` to `model_type`; side-chain packing additionally
uses `ligandmpnn_sc_v_32_002_16.pt`.

**Fixed-residue coordinates.** The
[LigandMPNN wrapper](../../Scripts/mpnn_relevant_utils/run_ligandmpnn.py)
restores exact input coordinates after reconstruction and exposes fixed side
chains to the packer as context. Stock LigandMPNN preserves torsions under
`--repack_everything 0` but rebuilds idealized coordinates. Use `--no_patch` for
stock behavior; the wrapper also supplies two `.long()` packing fixes.

> The previously reported comparison used
> [`Scripts/mpnn_relevant_utils/test_fixed_residue_preservation.py`](../../Scripts/mpnn_relevant_utils/test_fixed_residue_preservation.py),
> which runs stock and patched side by side on a real structure and reports the
> largest displacement of the fixed residues:
>
> ```
>   STOCK   LigandMPNN : max fixed-residue displacement = 0.2918 A
>   PATCHED            : max fixed-residue displacement = 0.0000 A
> ```
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
> Packed side chains still clash with catalytic residues in over half these
> structures. Treat them as starting points for Rosetta repacking. Set
> `LIGANDMPNN_NO_FIXED_CONTEXT=1` to disable this step and reproduce the
> right-hand column.

**AlphaFold3.** Source is pinned at [Software/alphafold3](../../Software/alphafold3).
Obtain parameters separately under the upstream terms and configure `AF3_RUNNER`.
Sections I–V do not require AF3; later validation rounds do.

**Redesign rounds (VIII and XI)** run the same orchestrator as Section V, on
structures that have already been validated by AlphaFold3. Designs are named
`<input stem>_<combo tag>_<batch>_<seq>.pdb`, which the AlphaFold3 matching
cells parse to recover the input each design came from.

New sequence design methods can also be applied at these steps &mdash; for
example [protein_chisel](https://github.com/SethWoodbury/protein_chisel).

## Related

- Metalloesterase campaign (RFdiffusion2): [`../Metalloesterase_RFdiffusion2/`](../Metalloesterase_RFdiffusion2/)
- Method tutorial: [`../../RFdiffusion3_Tutorial/`](../../RFdiffusion3_Tutorial/)
- Experimental data: [`../../Manuscript_Data/Phosphotriesterase_RFdiffusion3_Science_2026/`](../../Manuscript_Data/Phosphotriesterase_RFdiffusion3_Science_2026/)

## Reference

Butcher, Krishna, Mitra et al. *Atomic functional site scaffolding in biopolymers
with RFdiffusion3.* Science, eaeh2982 (2026).
[DOI](https://doi.org/10.1126/science.aeh2982).

[Citation guide](../../Docs/citations.md) · [BibTeX bibliography](../../Docs/references.bib)
