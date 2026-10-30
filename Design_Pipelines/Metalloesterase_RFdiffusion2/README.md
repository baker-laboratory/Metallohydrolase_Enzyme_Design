# Metallohydrolase Design Pipeline &mdash; Reproduction Tutorial

This directory contains the **core computational design pipeline for Design
Campaign 1** of the Nature metallohydrolase study, with its theozyme inputs and
selected intermediate outputs. The notebook takes an active-site geometry
through scaffold generation, sequence design and computational assessment to
select candidates for experimental testing.

For preparation and scaffold-generation examples, start with the
[RFdiffusion2 tutorial](../../RFdiffusion2_Tutorial/) or
[RFdiffusion3 tutorial](../../RFdiffusion3_Tutorial/).

## Pipeline Stages

The notebook [`design_zn_hydrolase.ipynb`](design_zn_hydrolase.ipynb) combines
analysis cells with commands for these stages:

1. **RFdiffusion2, filtering and inpainting** — generate motif-containing
   scaffolds, inspect structural quality, refine side chains and build loops
   with RFjoint2.
2. **ProteinMPNN** — generate sequences for the first structure-prediction round.
3. **AlphaFold2, first round** — assess global and catalytic-site agreement.
4. **FastMPNNDesign** — refine sequences and structures around the catalytic pocket.
5. **AlphaFold2, second round** — reassess candidates, relax apo structures,
   cluster sequences and prepare ligand-containing structures.
6. **PLACER** — assess protein–ligand conformational ensembles.

## Files

| Location | Contents |
|---|---|
| `inputs/invrot/`, `inputs/theozymes/` | Inverse-rotamer libraries and theozyme inputs |
| `working_dir/` | Example commands, fixed-residue specifications, cached DataFrames and RFdiffusion2 configs |
| `outputs/` | Selected scaffolds, inpainting/predesign results, sequences, AF2 predictions and PLACER inputs |

## Run

Follow the [environment guide](../../Environment/README.md) for the Linux helper
environment and [software guide](../../Docs/software-and-models.md) for model
runtimes and checkpoints. This campaign uses RFdiffusion2, ProteinMPNN/LigandMPNN,
RFjoint2, SuperFold/OpenFold and PLACER. RFdiffusion2's `setup.py` downloads
multiple containers and checkpoints; it does not install an inference runtime.

From the repository root, after setup:

```bash
conda activate zinc_hydro
jupyter lab Design_Pipelines/Metalloesterase_RFdiffusion2/design_zn_hydrolase.ipynb
```

Select the helper kernel, review output paths and execute generated model
commands in their configured runtimes. Existing intermediate files document the
campaign; a new run still requires local configuration and compute. In
particular, match the notebook's LigandMPNN checkpoint path to your download
location and obtain AF2 weights for both prediction rounds.

## References

- Kim, Woodbury, Ahern et al. *Computational design of metallohydrolases.* Nature **649**, 246–253 (2026). [DOI](https://doi.org/10.1038/s41586-025-09746-w).
- Ahern, Yim, Tischer et al. *Atom-level enzyme active site scaffolding using RFdiffusion2.* Nature Methods **23**, 96–105 (2026). [DOI](https://doi.org/10.1038/s41592-025-02975-x).

[Citation guide](../../Docs/citations.md) · [BibTeX bibliography](../../Docs/references.bib)

Questions: Seth M. Woodbury (woodbuse@uw.edu) or Donghyo Kim (donghyo@uw.edu).
