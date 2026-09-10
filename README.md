# Computational Design of Metallohydrolases

Code, data, and tutorials accompanying the publication in *Nature*:

> **Computational design of metallohydrolases**
>
> Donghyo Kim&dagger;, Seth M. Woodbury&dagger;, Woody Ahern&dagger;, Doug Tischer, Alex Kang, Emily Joyce,
> Asim K. Bera, Nikita Hanikel, Saman Salike, Rohith Krishna, Jason Yim,
> Samuel J. Pellock, Anna Lauko, Indrek Kalvet\*, Donald Hilvert\*, David Baker\*
>
> &dagger;Co-first authors &ensp; \*Corresponding authors
>
> *Nature* (2025). DOI: [10.1038/s41586-025-09746-w](https://doi.org/10.1038/s41586-025-09746-w)

This work applies **RFdiffusion2**, a generative model for de novo protein
design, to build highly active zinc-dependent hydrolases from
quantum-chemistry-defined active-site geometries (theozymes). The repository
provides everything needed to reproduce the results, learn the methodology, and
apply it to new enzyme design problems — plus a parallel tutorial using
**RFdiffusion3**, the next-generation all-atom model, and space for the
RFdiffusion3 design campaigns that follow.

Everything here runs from a fresh clone on any machine: paths are relative or
auto-detected, and no container is required. See
[Portability](#portability).

---

## Quick start

```bash
git clone --recurse-submodules https://github.com/baker-laboratory/Metallohydrolase_Enzyme_Design.git
cd Metallohydrolase_Enzyme_Design
```

Then pick the smallest environment that covers what you want to do:

**Re-plot the published wet lab data** (1 minute, no GPU, no license):

```bash
conda env create -f Environment/analysis.yml
conda activate zinc_hydro_analysis
python -m ipykernel install --user --name=zinc_hydro_analysis
jupyter lab Manuscript_Data/Metallohydrolase_Nature_2025/wetlab_data_analysis.ipynb
```

**Run the design tutorials and pipelines** (needs a GPU and a free academic
PyRosetta license):

```bash
conda env create -f Environment/zinc_hydro.yml
conda activate zinc_hydro
python -m ipykernel install --user --name=zinc_hydro
python Scripts/env_config.py          # confirm everything resolves
jupyter lab RFdiffusion2_Tutorial/RFdiffusion2_Tutorial_JupyterNotebook.ipynb
```

Full details, including containers and pixi: [`Environment/README.md`](Environment/README.md).

---

## Repository overview

| Directory | Description |
|-----------|-------------|
| [`RFdiffusion2_Tutorial/`](RFdiffusion2_Tutorial/) | Interactive tutorial for the RFdiffusion2 enzyme design workflow, with two fully worked examples |
| [`RFdiffusion3_Tutorial/`](RFdiffusion3_Tutorial/) | The same two examples using RFdiffusion3 via [foundry](https://github.com/RosettaCommons/foundry) |
| [`Design_Pipelines/`](Design_Pipelines/) | Full design campaigns at production scale, one directory per target chemistry |
| [`Manuscript_Data/`](Manuscript_Data/) | Wet lab measurements, supplementary deposits, and analysis notebooks, one directory per campaign |
| [`Scripts/`](Scripts/) | Shared Python utilities: theozyme preparation, structure processing, filtering, scoring |
| [`Software/`](Software/) | Git submodules for RFdiffusion2, RFdiffusion3 (foundry), PLACER, OpenFold, SuperFold, FastMPNNDesign, RFjoint2, plus a bundled XYZ&rarr;PDB theozyme converter |
| [`Environment/`](Environment/) | Conda, Apptainer, Docker, and pixi environment specifications |

### Campaigns

`Design_Pipelines/` and `Manuscript_Data/` are organized by campaign, using
matching directory names so the computation and the measurements stay findable
from one another.

| Campaign | Model | Chemistry | Pipeline | Data | Status |
|---|---|---|---|---|---|
| Metalloesterase | RFdiffusion2 | Zn(II) esterase, 4MU-phenylacetate | [pipeline](Design_Pipelines/Metalloesterase_RFdiffusion2/) | [data](Manuscript_Data/Metallohydrolase_Nature_2025/) | **Published** (*Nature* 2025) |
| Phosphotriesterase | RFdiffusion3 | Binuclear phosphotriesterase, paraoxon hydrolysis | [pipeline](Design_Pipelines/Phosphotriesterase_RFdiffusion3/) | [data](Manuscript_Data/Phosphotriesterase_RFdiffusion3/) | In preparation |
| Metalloprotease | RFdiffusion3 | Zn(II) metalloprotease, amide hydrolysis | [pipeline](Design_Pipelines/Metalloprotease_RFdiffusion3/) | [data](Manuscript_Data/Metalloprotease_RFdiffusion3/) | In preparation |

The two RFdiffusion3 campaigns are in preparation and will be published here
shortly, alongside the RFdiffusion3 release. The RFdiffusion3 *method* is
already fully demonstrated in
[`RFdiffusion3_Tutorial/`](RFdiffusion3_Tutorial/), which runs today.

### Structure

```
Metallohydrolase_Enzyme_Design/
|
|-- README.md
|-- LICENSE                                        # MIT License
|-- .gitmodules
|
|-- model_weights/                                 # GITIGNORED - you populate this
|                                                  #   see "Model weights" below
|
|-- Environment/
|   |-- README.md                                  # which environment to use, and why
|   |-- analysis.yml                               # wet lab analysis only (no GPU, no license)
|   |-- requirements-analysis.txt                  #   the same, for pip / uv
|   |-- zinc_hydro.yml                             # full design stack (portable, minor-version pins)
|   |-- zinc_hydro.lock.yml                        #   exact export of the environment we ran
|   |-- zinc_hydro.def                             # Apptainer/Singularity definition
|   |-- Dockerfile                                 # Docker equivalent
|   +-- pixi.toml                                  # pixi manifest (analysis + design envs)
|
|-- RFdiffusion2_Tutorial/                         # INTERACTIVE TUTORIAL (RFd2)
|   |-- RFdiffusion2_Tutorial_JupyterNotebook.ipynb
|   |-- README.md
|   |-- theozymes/                                 #   from_PDB_structure/ and from_quantum_chemistry/
|   |-- rfd2_configs/                              #   auto-generated YAML configs
|   |-- cmds/  slurm_submit/  outputs/  logs/
|
|-- RFdiffusion3_Tutorial/                         # INTERACTIVE TUTORIAL (RFd3)
|   |-- RFdiffusion3_Tutorial_JupyterNotebook.ipynb
|   |-- README.md
|   |-- theozymes/                                 #   same two worked examples
|   |-- rfd3_json/                                 #   auto-generated JSON configs (dialect 2)
|   |-- cmds/  slurm_submit/  outputs/  logs/
|
|-- Design_Pipelines/                              # PRODUCTION DESIGN CAMPAIGNS
|   |-- README.md
|   |-- Metalloesterase_RFdiffusion2/              #   published campaign
|   |   |-- design_zn_hydrolase.ipynb
|   |   +-- inputs/  outputs/
|   |-- Phosphotriesterase_RFdiffusion3/           #   in preparation
|   +-- Metalloprotease_RFdiffusion3/              #   in preparation
|
|-- Manuscript_Data/                               # EXPERIMENTAL DATA
|   |-- README.md
|   |-- Metallohydrolase_Nature_2025/
|   |   |-- wetlab_data_analysis.ipynb
|   |   |-- raw_wetlab_data/                       #   43 primary files
|   |   |-- supplemental_data/                     #   Data S1-S3
|   |   +-- wetlab_data_plots/                     #   all generated figures
|   |-- Phosphotriesterase_RFdiffusion3/           #   in preparation
|   +-- Metalloprotease_RFdiffusion3/              #   in preparation
|
|-- Scripts/                                       # SHARED UTILITIES
|   |-- env_config.py                              #   resolves interpreter/container/obabel paths
|   |-- General_NoteBook_Functions.py
|   |-- prepare_PDB_structure_into_theozyme.py
|   |-- process_RFD2_outputs.py
|   |-- process_diffusion3_outputs.py
|   |-- scaffold_handling/                         #   RFd3 scaffold filtering + idealization
|   |-- theozyme_and_ligand_handling/              #   ligand -> Rosetta .params, theozyme prep, CST files
|   |-- advanced_structure_prediction_tools/       #   AlphaFold3 input/output handling
|   |-- af2_analysis_and_tools/                    #   AlphaFold2 parsing and alignment
|   |-- design_filtering/                          #   metric monster, contacts, fpocket, shape metrics
|   |-- general_utils/                             #   REMARK 666, foldseek clustering, deduplication
|   |-- notebook_functions/                        #   shared notebook + SLURM helpers
|   |-- automation_tools/  mpnn_relevant_utils/
|   |-- repo_paths.py                              #   locates Open Babel / Rosetta tooling
|   |-- ...                                        #   ~20 further processing scripts
|   +-- enzyme_design/  scoring/
|
+-- Software/                                      # SUBMODULES + bundled tools
    |-- RFdiffusion2/  foundry/  PLACER/  openfold/
    |-- superfold/  fastmpnndesign/  proteininpainting/
    |-- foldseek/                                  #   GPL-3.0, see License below
    |-- alphafold3/                                #   params are gated; see Model weights
    +-- theozyme_XYZ_2_PDB__beta/                  #   bundled XYZ->PDB theozyme converter
```

---

## Tutorials

### RFdiffusion2 Tutorial

**Notebook:** [`RFdiffusion2_Tutorial/RFdiffusion2_Tutorial_JupyterNotebook.ipynb`](RFdiffusion2_Tutorial/)

A step-by-step tutorial covering the pipeline from active-site preparation
through scaffold generation, with two parallel worked examples throughout:

| Example | Source | Substrate | Catalytic motif |
|---------|--------|-----------|-----------------|
| Zinc protease | Crystal structure ([PDB 1QJI](https://www.rcsb.org/structure/1QJI)) | Phosphonamidate transition-state analog | HEHH / HEHHY / HEHHYM |
| Zinc esterase | DFT transition-state optimization (Gaussian) | 4-Methylumbelliferyl phosphonate acetate | HHHE |

1. **Theozyme creation** — build minimal active-site geometries from an
   experimental PDB (crop in PyMOL, format, split into residue subsets) or from
   quantum chemistry (parse Gaussian output, convert XYZ to PDB with correct
   atom naming)
2. **Theozyme post-processing** — add ORI tokens to steer the protein
   center of mass during scaffold generation
3. **Scaffold generation** — run RFdiffusion2 at scale with unindexed catalytic
   residues and fixed side-chain atoms, auto-generating configs and SLURM scripts
4. **Output filtering** — filter by radius of gyration, loop content, ligand
   burial, bond-length deviation, and catalytic residue geometry

See the [tutorial README](RFdiffusion2_Tutorial/README.md) for prerequisites and
troubleshooting.

### RFdiffusion3 Tutorial

**Notebook:** [`RFdiffusion3_Tutorial/RFdiffusion3_Tutorial_JupyterNotebook.ipynb`](RFdiffusion3_Tutorial/)

The same two worked examples, using **RFdiffusion3** via the
[foundry](https://github.com/RosettaCommons/foundry) framework. Sections I and II
(theozyme creation and post-processing) are identical to the RFdiffusion2
tutorial. What differs:

1. **JSON configs** — RFdiffusion3 uses JSON (dialect 2) rather than YAML
2. **`rfd3 design` inference** — the open-source CLI from foundry
3. **Advanced conditioning** — atomwise RASA, atomwise H-bond conditioning, and
   classifier-free guidance
4. **Two-stage filtering** — rapid JSON metrics, then PyRosetta scaffold analysis
5. **Geometry idealization** — Cartesian relaxation to correct bond geometry

> **RFdiffusion3 release status.** foundry is tracked at its `production`
> branch, currently `v0.2.0`. This tutorial runs end to end today; all of its
> helper scripts are bundled under [`Scripts/`](Scripts/).

---

## Portability

**There are no paths to edit.** Notebooks locate the repository root by walking
up from the working directory, and scripts resolve their tools at startup,
reporting clearly if a dependency is not available.

**No container is required.** The `zinc_hydro` conda environment provides
PyRosetta, Open Babel, and everything else the helper scripts need. Containers
remain available as an option: build one from
[`Environment/zinc_hydro.def`](Environment/zinc_hydro.def) and point at it with
`ZINC_HYDRO_SIF`.

Three optional environment variables control all of this:

| Variable | Default when unset | Purpose |
|---|---|---|
| `ZINC_HYDRO_SIF` | the active python interpreter | Apptainer image for helper scripts |
| `ZINC_HYDRO_OBABEL` | `obabel` from `$PATH` | Open Babel executable |
| `ZINC_HYDRO_REPO` | auto-detected | Repository root |

Verify your setup at any time:

```bash
python Scripts/env_config.py
```

---

## Software dependencies

### Conda environments

| Environment | File | For |
|---|---|---|
| `zinc_hydro_analysis` | [`Environment/analysis.yml`](Environment/analysis.yml) | Wet lab analysis. numpy, pandas, scipy, matplotlib, openpyxl, JupyterLab. No GPU, no license. |
| `zinc_hydro` | [`Environment/zinc_hydro.yml`](Environment/zinc_hydro.yml) | Everything else. |

Key packages in `zinc_hydro`:

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.10 | Runtime |
| PyTorch | 2.3.1 (CUDA 12.1) | Deep learning backend |
| PyRosetta | 2024.39 | Structure manipulation and scoring |
| OpenBabel | 3.1.1 | Chemical file format conversion |
| RDKit | 2024.03.5 | Cheminformatics |
| OpenMM | 8.1.2 | Molecular dynamics |
| JAX | 0.4.27 | Accelerated numerical computing |
| TensorFlow | 2.16.1 | OpenFold/SuperFold weight loading |
| BioPython | 1.84 | Sequence and structure utilities |
| NumPy / Pandas / SciPy / Matplotlib | — | Numerical work and plotting |
| openpyxl | 3.1 | Reading the `.xlsx` plate reader data |

PyRosetta requires a license, free for academic use:
[els2.comotion.uw.edu/product/pyrosetta](https://els2.comotion.uw.edu/product/pyrosetta).

### Git submodules (`Software/`)

| Submodule | Description | Repository | Method paper |
|-----------|-------------|------------|--------------|
| **RFdiffusion2** | Generative model for de novo backbone design with atom-level active-site scaffolding | [RosettaCommons/RFdiffusion2](https://github.com/RosettaCommons/RFdiffusion2) | Ahern et al., *Nature Methods* 23, 96&ndash;105 (2025). [doi:10.1038/s41592-025-02975-x](https://doi.org/10.1038/s41592-025-02975-x) |
| **foundry** | RFdiffusion3 &mdash; all-atom generative model, run through the `rfd3 design` CLI (tracked at `production`). Model weights are downloaded separately. | [RosettaCommons/foundry](https://github.com/RosettaCommons/foundry) | Butcher et al., *bioRxiv* (2025). [doi:10.1101/2025.09.18.676967](https://doi.org/10.1101/2025.09.18.676967) *(preprint)* |
| **PLACER** | Protein-ligand atomic conformational ensemble prediction | [baker-laboratory/PLACER](https://github.com/baker-laboratory/PLACER) | &mdash; |
| **OpenFold** | Open-source AlphaFold2 implementation | [aqlaboratory/openfold](https://github.com/aqlaboratory/openfold) | &mdash; |
| **SuperFold** | AlphaFold2-based structure prediction wrapper | [rdkibler/superfold](https://github.com/rdkibler/superfold) | &mdash; |
| **FastMPNNDesign** | LigandMPNN + Rosetta sequence design. Bundles LigandMPNN as a nested submodule &mdash; initialize with `git submodule update --init --recursive`, and download weights with its `get_model_params.sh` | [ikalvet/fastmpnndesign](https://github.com/ikalvet/fastmpnndesign) | Dauparas et al., *Nature Methods* (2025). [doi:10.1038/s41592-025-02626-1](https://doi.org/10.1038/s41592-025-02626-1) |
| **RFjoint2** | Conditional sequence and structure generation (inpainting) | [RosettaCommons/RFjoint2](https://github.com/RosettaCommons/RFjoint2) | &mdash; |
| **AlphaFold3** | Structure prediction used to validate designs. Source only &mdash; **model parameters are not included and are obtained from Google DeepMind under their own terms, which restrict commercial use** | [google-deepmind/alphafold3](https://github.com/google-deepmind/alphafold3) | Abramson et al., *Nature* **630**, 493&ndash;500 (2024) |
| **Foldseek** | Fast structural similarity search and clustering | [steineggerlab/foldseek](https://github.com/steineggerlab/foldseek) | van Kempen et al., *Nature Biotechnology* **42**, 243&ndash;246 (2024) |

All submodules are pinned to the current upstream head of their default branch
(`production` for foundry, `main` for the rest). Pins are updated deliberately,
not automatically, so a clone always reproduces a known-good combination:

```bash
git submodule update --init --recursive        # after cloning
git submodule update --remote Software/foundry # advance one, on purpose
```

Also bundled (not a submodule): `Software/theozyme_XYZ_2_PDB__beta/`, the
XYZ&rarr;PDB theozyme converter.

### External software (optional)

| Software | Used for | Required? |
|----------|----------|-----------|
| [Gaussian](https://gaussian.com/) | DFT transition-state optimization | Only for quantum-chemistry theozymes. Set `GAUSS_EXEDIR`. |
| [ChemCraft](https://www.chemcraftprog.com/) | Building initial QM geometries | Only for quantum-chemistry theozymes |
| [Rosetta](https://www.rosettacommons.org/) | `molfile_to_params.py`, used to build ligand `.params` files | Only when generating params for a new ligand. Set `$ROSETTA`. |
| SLURM | HPC job scheduling | Optional — every command can be run locally |
| AlphaFold2 weights | Structure prediction stages | See **Model weights** below |

---

## Model weights

**No model weights are committed to this repository**, and `model_weights/` is
gitignored. Fetch each set yourself. PLACER is the one exception: its weights
are committed inside its own submodule, so `--recursive` is all you need.

| Model | Fetch with | Lands in | How the code finds it |
|---|---|---|---|
| RFdiffusion2 `RFD_140.pt` (~1.3 GB) | `cd Software/RFdiffusion2 && python setup.py && cd ../..` | `Software/RFdiffusion2/rf_diffusion/model_weights/` | config default in `aa.yaml` |
| RFdiffusion3 | `foundry install rfd3` | `~/.foundry/checkpoints/` | auto-discovery; override with `FOUNDRY_CHECKPOINT_DIRS` |
| LigandMPNN / ProteinMPNN | `bash Software/fastmpnndesign/lib/LigandMPNN/get_model_params.sh model_weights/` | `model_weights/` | `$LIGANDMPNN_WEIGHTS` |
| AlphaFold2 params (~4 GB) | see the command below | `Software/superfold/alphafold_weights/params/` | `Software/superfold/alphafold_weights.pth` |
| AlphaFold3 params | [request from Google DeepMind](https://github.com/google-deepmind/alphafold3) &mdash; non-commercial terms, no public URL | your choice | `$AF3_RUNNER` |
| PLACER | nothing to do &mdash; committed in the submodule | `Software/PLACER/weights/` | submodule-relative default |

```bash
# LigandMPNN / ProteinMPNN
bash Software/fastmpnndesign/lib/LigandMPNN/get_model_params.sh model_weights/
export LIGANDMPNN_WEIGHTS=$PWD/model_weights

# AlphaFold2 params only (~4 GB). Requires aria2c: sudo apt install aria2.
# Do NOT use download_all_data.sh -- it also pulls ~2.6 TB of genetic databases
# that SuperFold, being single-sequence, never uses.
bash Software/superfold/scripts/download_alphafold_params.sh Software/superfold/alphafold_weights
realpath Software/superfold/alphafold_weights > Software/superfold/alphafold_weights.pth
```

Two similarly named variables are easy to confuse: **`LIGANDMPNN_WEIGHTS`** is
the *weights* directory, **`LIGANDMPNN_DIR`** is the *code* checkout.

> The Metalloesterase RFdiffusion2 notebook currently reads its MPNN weights
> from `Software/fastmpnndesign/lib/LigandMPNN/model_params/` rather than
> `model_weights/`. Point `get_model_params.sh` at that directory instead if you
> are running that pipeline.

---

## Scripts

| Script | Purpose |
|--------|---------|
| `env_config.py` | Resolves the interpreter, container, Open Babel, and repository root; run it directly for an environment report |
| `General_NoteBook_Functions.py` | Shared notebook utilities (directory setup, SLURM submission, display settings) |
| `prepare_PDB_structure_into_theozyme.py` | Clean and format PDB structures into theozyme inputs |
| `split_theozyme_into_subsets.py` | Split a theozyme into catalytic residue subsets |
| `add_ORI_token_to_PDB.py` | Add ORI tokens specifying the desired protein center of mass |
| `add_remark666_lines_to_pdb.py` | Add REMARK 666 metadata for catalytic residue tracking |
| `parse_gauss_log_files_make_xyz.py` | Extract optimized geometries from Gaussian logs |
| `process_RFD2_outputs.py` | Filter and score RFdiffusion2 outputs |
| `process_diffusion3_outputs.py` | Filter and score RFdiffusion3 outputs |
| `process_diffusion3_outputs__ORCHESTRATOR.py` | Chunks large output sets and runs the above on each chunk, merging the scorefiles; optional SLURM submission |
| `theozyme_and_ligand_handling/ligands_to_params__UNIFIED.py` | Generate Rosetta `.params` (and the matching `.pdb`) for a ligand taken from a theozyme; needed before any PyRosetta-based filtering |
| `repo_paths.py` | Locates Open Babel and Rosetta's `molfile_to_params.py` / residue-type database; run it directly to check what resolves |
| `scaffold_handling/filter_pdbsDIR_by_catres_sequence_distance.py` | Filter scaffolds by sequence separation between catalytic residue pairs (standard library only) |
| `scaffold_handling/idealize_rfdiffusion3_geometry.py` | Cartesian relaxation of RFdiffusion3 backbones under tight coordinate constraints, to correct bond geometry |
| `process_placer.py` | Post-process PLACER predictions |
| `setup_inpaint_from_rfd2.py` | Prepare inpainting inputs from RFdiffusion2 outputs |
| `sidechain_rmsd_and_info_af2_matching_res.py` | Validate AlphaFold2 predictions against designs |
| `experimental_data_processing_functions.py` | Kinetics fitting and plotting for the wet lab notebooks |
| `SimplePdbLib.py` / `SimpleXyzMath3.py` | PDB parsing and 3D geometry utilities |
| `enzyme_design/` | Rosetta design utilities, constraints, scoring, sequence design |

---

## Contributing and issues

Bug reports and questions are welcome at
[GitHub Issues](https://github.com/baker-laboratory/Metallohydrolase_Enzyme_Design/issues).
When reporting a problem, please include the output of:

```bash
python Scripts/env_config.py
```

which shows the interpreter, resolved tools, and installed package versions.

---

## License

[MIT License](LICENSE) — this covers original code written for this project.
Third-party code and binaries retain their upstream licenses, including copies
under `Scripts/`.

Examples of third-party components and their terms:

| Component | Origin | License |
|---|---|---|
| `Scripts/enzyme_design/DAlphaBall.gcc` | Rosetta distribution (`holes` filter) | See the [upstream source](https://github.com/RosettaCommons/rosetta/tree/main/source/external/DAlpahBall) and [Rosetta licensing](https://github.com/RosettaCommons/rosetta/blob/main/LICENSE.md); this repository's MIT license does not relicense third-party code. |
| `Scripts/mpnn_relevant_utils/ligandmpnn_patched/sc_utils.py` | [LigandMPNN](https://github.com/dauparas/LigandMPNN) | MIT (upstream copyright retained in the file header) |
| `Software/foldseek` | [steineggerlab/foldseek](https://github.com/steineggerlab/foldseek) | GPL-3.0 — a separate submodule, invoked as an external binary |

Submodules under `Software/` are governed by their own licenses. PyRosetta has a
[separate license](https://github.com/RosettaCommons/rosetta/blob/main/LICENSE.PyRosetta.md),
and AlphaFold3 parameters are subject to Google DeepMind's
[model parameter terms](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md).

---

## Citation

If you use this repository, its methods, or its data, please cite the primary
metallohydrolase paper. If you use RFdiffusion2 or RFdiffusion3 directly, please
also cite the corresponding method paper.

### Primary publication

> Kim, D., Woodbury, S.M., Ahern, W. et al. Computational design of metallohydrolases. *Nature* (2025). https://doi.org/10.1038/s41586-025-09746-w

```bibtex
@article{kim2025metallohydrolases,
  title     = {Computational design of metallohydrolases},
  author    = {Kim, Donghyo and Woodbury, Seth M. and Ahern, Woody and Tischer, Doug
               and Kang, Alex and Joyce, Emily and Bera, Asim K. and Hanikel, Nikita
               and Salike, Saman and Krishna, Rohith and Yim, Jason
               and Pellock, Samuel J. and Lauko, Anna and Kalvet, Indrek
               and Hilvert, Donald and Baker, David},
  journal   = {Nature},
  year      = {2025},
  doi       = {10.1038/s41586-025-09746-w},
  url       = {https://www.nature.com/articles/s41586-025-09746-w}
}
```

### Method papers

**RFdiffusion2** — Ahern, W., Yim, J., Tischer, D., Salike, S., Woodbury, S.M., Kim, D., Kalvet, I., Kipnis, Y., Coventry, B., Altae-Tran, H.R., Bauer, M.S., Barzilay, R., Jaakkola, T.S., Krishna, R., Baker, D. Atom-level enzyme active site scaffolding using RFdiffusion2. *Nature Methods* **23**, 96&ndash;105 (2025). https://doi.org/10.1038/s41592-025-02975-x

```bibtex
@article{ahern2025rfdiffusion2,
  title     = {Atom-level enzyme active site scaffolding using {RFdiffusion2}},
  author    = {Ahern, Woody and Yim, Jason and Tischer, Doug and Salike, Saman
               and Woodbury, Seth M. and Kim, Donghyo and Kalvet, Indrek
               and Kipnis, Yakov and Coventry, Brian and Altae-Tran, Han Raut
               and Bauer, Magnus S. and Barzilay, Regina and Jaakkola, Tommi S.
               and Krishna, Rohith and Baker, David},
  journal   = {Nature Methods},
  year      = {2025},
  volume    = {23},
  number    = {1},
  pages     = {96--105},
  doi       = {10.1038/s41592-025-02975-x},
  url       = {https://www.nature.com/articles/s41592-025-02975-x}
}
```

**RFdiffusion3** — Butcher, J., Krishna, R., Mitra, R., Brent, R.I., Li, Y., Corley, N., Kim, P.T., Funk, J., Mathis, S., Salike, S., Muraishi, A., Eisenach, H., Thompson, T.R., Chen, J., Politanska, Y., Sehgal, E., Coventry, B., Zhang, O., Qiang, B., Didi, K., Kazman, M., DiMaio, F., Baker, D. *De novo* design of all-atom biomolecular interactions with RFdiffusion3. *bioRxiv* (2025). https://doi.org/10.1101/2025.09.18.676967

> &#x26A0;&#xFE0F; *Preprint — this citation will be replaced with the peer-reviewed reference once available.*

```bibtex
@article{butcher2025rfdiffusion3,
  title     = {De novo design of all-atom biomolecular interactions with {RFdiffusion3}},
  author    = {Butcher, Jasper and Krishna, Rohith and Mitra, Raktim
               and Brent, Rafael I. and Li, Yanjing and Corley, Nathaniel
               and Kim, Paul T. and Funk, Jonathan and Mathis, Simon
               and Salike, Saman and Muraishi, Aiko and Eisenach, Helen
               and Thompson, Tuscan Rock and Chen, Jie and Politanska, Yuliya
               and Sehgal, Enisha and Coventry, Brian and Zhang, Odin
               and Qiang, Bo and Didi, Kieran and Kazman, Max
               and DiMaio, Frank and Baker, David},
  journal   = {bioRxiv},
  year      = {2025},
  doi       = {10.1101/2025.09.18.676967},
  url       = {https://www.biorxiv.org/content/10.1101/2025.09.18.676967v2},
  note      = {Preprint}
}
```

---

## Contact

- **Seth Woodbury** &mdash; woodbuse@uw.edu
- **Donghyo Kim** &mdash; donghyo@uw.edu
