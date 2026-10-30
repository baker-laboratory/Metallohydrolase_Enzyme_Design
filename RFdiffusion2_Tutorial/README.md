# RFdiffusion2 tutorial: from theozyme to scaffold

[Open the notebook](RFdiffusion2_Tutorial_JupyterNotebook.ipynb) ·
[Workflow guide](../Docs/workflows.md) · [Full campaign](../Design_Pipelines/Metalloesterase_RFdiffusion2/)

Prepare an active-site geometry, generate RFdiffusion2 scaffolds and filter the
results. The tutorial stops at scaffold selection; sequence design, AF2 and
PLACER assessment are covered by the campaign notebook.

## Worked examples

| Example | Source | Active-site model |
|---|---|---|
| Zinc protease | [1QJI crystal structure](https://www.rcsb.org/structure/1QJI) | Zn(II), catalytic residues and a phosphonamidate transition-state analog (`TSA`) |
| Zinc esterase | Supplied Gaussian calculation | ZETA_2 transition-state geometry (`SZD`) |

Both examples include preparation and inference configuration. Start with the
structure-derived example for a first run.

## Workflow

| Section | Work |
|---|---|
| I | Crop and format a PDB-derived active site, or extract an optimized Gaussian geometry and convert XYZ to PDB; optionally split catalytic-residue subsets |
| II | Add ORI positions and organize inference inputs |
| III | Select fixed atoms and unindexed residues, generate YAML configs and commands, run inference, then inspect and filter scaffolds |

The filtering cells are present, although their historical heading still says
“Under Construction.” Filtering uses PyRosetta and a ligand `.params` file with
a matching ligand PDB for SASA calculations. The notebook generates a command
for [ligand parameterization](../Scripts/theozyme_and_ligand_handling/ligands_to_params__UNIFIED.py);
this also requires Rosetta's external `molfile_to_params.py`. Configure `ROSETTA`
and check paths with `python Scripts/repo_paths.py`.

## Run

Follow the [environment guide](../Environment/README.md) for the Linux x86_64
`zinc_hydro` helper environment and the [software guide](../Docs/software-and-models.md)
for separate model runtimes and checkpoints. The helper environment alone does
not install an inference pipeline.

From the repository root, after setup:

```bash
conda activate zinc_hydro
python -m ipykernel install --user --name=zinc_hydro
jupyter lab RFdiffusion2_Tutorial/RFdiffusion2_Tutorial_JupyterNotebook.ipynb
```

The initialization cell detects the repository root. Review output locations,
model launchers and scheduler settings, then generate and test one command
before starting a batch. Driver cells print or write commands that you execute
in the appropriate model environment; SLURM is optional.

PyMOL supports manual cropping and inspection. Gaussian is needed for new
quantum-chemistry calculations, but the supplied calculation outputs can be
reused; ChemCraft is an optional modeling interface.

RFdiffusion2's `setup.py` downloads multiple checkpoints and containers, including
`RFD_140.pt`; it does not install an inference environment. See the
[RFdiffusion2 setup notes](../Docs/software-and-models.md#rfdiffusion2).
AF2 weights are only needed if you continue into the downstream campaign.

## Files and conventions

| Path | Contents |
|---|---|
| `theozymes/from_PDB_structure/` | Original/cropped 1QJI structures, cleaned subsets and ORI inputs |
| `theozymes/from_quantum_chemistry/` | Gaussian inputs/outputs, XYZ geometry and converted ZETA_2 theozymes |
| `rfd2_configs/`, `cmds/`, `slurm_submit/` | Example configs and command/job files; regenerate for your checkout |
| `outputs/` | Deposited example scaffold outputs; review output paths before a new run |

- **Theozyme:** the catalytic geometry to scaffold, including metal ions,
  substrate/transition-state coordinates and catalytic side chains.
- **Unindexed residues:** catalytic residues whose sequence positions are chosen
  during generation; fixed-atom selections specify which coordinates to preserve.
- **ORI tokens:** pseudo-atoms that specify the desired protein center of mass.
  Inspect their placement in PyMOL; omit ORI tokens from scoring references.

## Troubleshooting

| Symptom | Check |
|---|---|
| `obabel` missing or PyRosetta import fails | Activate the helper environment and select its Jupyter kernel |
| Inference fails | Check the initialized RFdiffusion2 submodule, checkpoint, model runtime and GPU setup |
| XYZ-to-PDB conversion fails | Inspect ligand atom ranges and enable the conversion cell's `keep_temp_files` flags |
| ORI markers are difficult to see | In PyMOL, use `show spheres` and `set sphere_scale, 0.5` |
| Filtering fails | Check ligand names, `.params`/ligand PDB files and the absence of ORI tokens in the scoring reference |
| SLURM jobs fail | Inspect logs and adapt partition, resource and launcher settings to your cluster |

## References

- Kim, Woodbury, Ahern et al. *Computational design of metallohydrolases.* Nature **649**, 246–253 (2026). [DOI](https://doi.org/10.1038/s41586-025-09746-w).
- Ahern, Yim, Tischer et al. *Atom-level enzyme active site scaffolding using RFdiffusion2.* Nature Methods **23**, 96–105 (2026). [DOI](https://doi.org/10.1038/s41592-025-02975-x).

[Citation guide](../Docs/citations.md) · [BibTeX bibliography](../Docs/references.bib)

Questions: Seth Woodbury (woodbuse@uw.edu) or Donghyo Kim (donghyo@uw.edu).
