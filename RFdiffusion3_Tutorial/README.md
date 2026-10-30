# RFdiffusion3 tutorial: from theozyme to scaffold

[Open the notebook](RFdiffusion3_Tutorial_JupyterNotebook.ipynb) ·
[Workflow guide](../Docs/workflows.md) · [Phosphotriesterase campaign](../Design_Pipelines/Phosphotriesterase_RFdiffusion3/)

Prepare an active-site geometry, generate RFdiffusion3 scaffolds, filter them
and idealize their geometry. The tutorial ends before sequence design and
structure-prediction validation; those stages are in the campaign notebooks.

## Learning resources

Start with installation and simple motif scaffolding, then work through enzyme
preparation and conditioning. These resources complement the local notebook;
use this repository's [setup guide](../Docs/software-and-models.md) for its
pinned software versions, since upstream documentation can evolve.

**Official RosettaCommons written guides**

| Guide | What it covers |
|---|---|
| [Unix installation](https://rosettacommons.github.io/foundry/models/rfd3/tutorials/RFdiffusion3_installation_tutorial.html) | Environment setup, checkpoints and a verification example |
| [Introductory enzyme design](https://rosettacommons.github.io/foundry/models/rfd3/tutorials/enzyme_design_tutorial.html) → [Intermediate enzyme design](https://rosettacommons.github.io/foundry/models/rfd3/tutorials/intermediate_enzyme_design_tutorial.html) | Progress from an initial example to preparing a catalytic motif |
| [Advanced Enzyme Design with RFdiffusion3](https://rosettacommons.github.io/foundry/models/rfd3/tutorials/advanced_enzyme_design_tutorial.html) | The 1QJI metalloprotease example: theozyme preparation, ORI placement, filtering, hydrogen-bond and RASA conditioning |

Keep the [documentation index](https://rosettacommons.github.io/foundry/models/rfd3/index.html)
and [input/CLI reference](https://rosettacommons.github.io/foundry/models/rfd3/input.html)
handy when adapting selections or inference settings.

**Video walkthroughs — RosettaCommons**

1. [RFdiffusion3 installation with Foundry](https://www.youtube.com/watch?v=6rFEaV7IyV4)
2. [RFdiffusion3 Simple Motif Scaffolding Guided Tutorial](https://www.youtube.com/watch?v=bFSW4x96vEk)
3. [Enzyme design with RFdiffusion3](https://www.youtube.com/watch?v=htA3HSXCkro) — credited to Seth M. Woodbury, Maruf Hasan and Austin Seamann

## Worked examples

| Example | Source | Active-site model |
|---|---|---|
| Zinc protease | [1QJI crystal structure](https://www.rcsb.org/structure/1QJI) | Zn(II), catalytic residues and a phosphonamidate transition-state analog (`TSA`) |
| Zinc esterase | Supplied Gaussian calculation | ZETA_2 transition-state geometry (`SZD`) |

Both theozyme-preparation examples are included. The Section III configuration
and committed JSON files use **1QJI**; adapting the quantum-chemistry example
requires changing the inference groups and ligand settings. Output directories
are placeholders for new runs.

## Workflow

| Section | Work |
|---|---|
| I | Crop and format a PDB-derived active site, or extract a Gaussian geometry and convert XYZ to PDB |
| II | Add ORI positions and organize inference inputs |
| III.A | Define fixed atoms and unindexed residues; generate dialect-2 JSON inputs and `rfd3 design` commands |
| III.B–C | Filter per-structure JSON metrics, then perform PyRosetta scaffold analysis and inspect distributions |
| IV | Optionally apply constrained Cartesian relaxation and filter the idealized structures |

RFdiffusion3 adds atomwise accessible-surface-area and hydrogen-bond
conditioning, classifier-free guidance and sampler parameter sweeps. Inference
produces CIF.gz structures and JSON metrics. These notebook settings describe
the example, not universal requirements for every RFdiffusion3 task.

## Run

Follow the [environment guide](../Environment/README.md) for the Linux x86_64
`zinc_hydro` helper environment and the [software guide](../Docs/software-and-models.md)
for separate model runtimes and checkpoints. The helper environment alone does
not install an inference pipeline.

From the repository root, after setup:

```bash
conda activate zinc_hydro
python -m ipykernel install --user --name=zinc_hydro
jupyter lab RFdiffusion3_Tutorial/RFdiffusion3_Tutorial_JupyterNotebook.ipynb
```

The initialization cell detects the repository root. Review output locations,
model launchers and scheduler settings, then generate and test one command
before starting a batch. Driver cells print or write commands that you execute
in the appropriate model environment; SLURM is optional.

PyMOL supports manual cropping and inspection. Gaussian is needed for new
quantum-chemistry calculations, but the supplied calculation outputs can be
reused; ChemCraft is an optional modeling interface.

Install foundry in a separate **Python 3.12+** model environment using the
[software guide](../Docs/software-and-models.md#rfdiffusion3). The
submodule is pinned; `production` is its branch hint, not an instruction to
update it. `foundry install rfd3` downloads checkpoints to
`~/.foundry/checkpoints` by default. For another location, use
`--checkpoint-dir` and configure `FOUNDRY_CHECKPOINT_DIRS`.

GPU inference requires a compatible runtime and hardware. Hydrogen-bond
conditioning also needs HBPLUS, which is not installed automatically. See the
[pinned RFdiffusion3 README](https://github.com/RosettaCommons/foundry/blob/aad357b776e3c0d6b973080f8f8c4bcf3ed21e40/models/rfd3/README.md) and
[input specification](https://github.com/RosettaCommons/foundry/blob/aad357b776e3c0d6b973080f8f8c4bcf3ed21e40/models/rfd3/docs/input.md).

Scaffold analysis and idealization need PyRosetta and appropriate ligand inputs.
The parameterization step calls Rosetta's external `molfile_to_params.py`;
configure `ROSETTA` and check with `python Scripts/repo_paths.py`.

## Files and conventions

| Path | Contents |
|---|---|
| `theozymes/` | PDB-derived and quantum-chemistry-derived preparation examples |
| `rfd3_json/`, `cmds/`, `slurm_submit/` | Example JSONs and command/job files; regenerate for your checkout |
| `outputs/` | Placeholder directories for inference and predesign results |
| [Output processor](../Scripts/process_diffusion3_outputs.py) | PyRosetta scaffold analysis |
| [Geometry idealizer](../Scripts/scaffold_handling/idealize_rfdiffusion3_geometry.py) | Constrained Cartesian relaxation |
| [Sequence-distance filter](../Scripts/scaffold_handling/filter_pdbsDIR_by_catres_sequence_distance.py) | Catalytic-residue spacing checks |

- **Theozyme:** the catalytic geometry to scaffold, including metal ions,
  substrate/transition-state coordinates and catalytic side chains.
- **Unindexed residues:** catalytic residues whose sequence positions are chosen
  during generation; fixed-atom selections specify which coordinates to preserve.
- **ORI tokens:** pseudo-atoms that specify the desired protein center of mass.
  Inspect their placement in PyMOL; omit ORI tokens from scoring references.

## Troubleshooting

| Symptom | Check |
|---|---|
| `rfd3` or checkpoint missing | Activate the foundry runtime; check `foundry install rfd3`, `FOUNDRY_CHECKPOINT_DIRS` or the notebook's `checkpoint_path` |
| JSON validation fails | Check paths, atom/residue identifiers and selections against the pinned input specification; `ligand`, `length` and `unindex` are task-dependent |
| `obabel` missing or PyRosetta import fails | Activate the helper environment and select its kernel |
| XYZ-to-PDB conversion fails | Inspect ligand atom ranges and enable `keep_temp_files` |
| Filtering fails | Check ligand names, `.params` files and the absence of ORI tokens in scoring references |
| ORI markers are difficult to see | In PyMOL, use `show spheres` and `set sphere_scale, 0.5` |
| Geometry idealization diverges | Inspect structures and coordinate constraints; try smaller `coord_cst_stdev` or larger `coord_cst_weight` |
| SLURM jobs fail | Inspect logs and adapt resources and launchers to your cluster |

## References

- Butcher, Krishna, Mitra et al. *Atomic functional site scaffolding in biopolymers with RFdiffusion3.* Science, eaeh2982 (2026). [DOI](https://doi.org/10.1126/science.aeh2982).
- Kim, Woodbury, Ahern et al. *Computational design of metallohydrolases.* Nature **649**, 246–253 (2026). [DOI](https://doi.org/10.1038/s41586-025-09746-w).
- Ahern, Yim, Tischer et al. *Atom-level enzyme active site scaffolding using RFdiffusion2.* Nature Methods **23**, 96–105 (2026). [DOI](https://doi.org/10.1038/s41592-025-02975-x). Predecessor workflow and preparation conventions.

[Citation guide](../Docs/citations.md) · [BibTeX bibliography](../Docs/references.bib)

Questions: Seth M. Woodbury (woodbuse@uw.edu) or Donghyo Kim (donghyo@uw.edu).
