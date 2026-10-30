# Computational Design of Metallohydrolases

[![License: MIT](https://img.shields.io/badge/License-MIT-3a8050?style=flat-square)](LICENSE)
[![Docs: guides](https://img.shields.io/badge/Docs-guides-336791?style=flat-square)](Docs/README.md)
[![Python for analysis: 3.10–3.12](https://img.shields.io/badge/Analysis_Python-3.10%E2%80%933.12-3776ab?style=flat-square&logo=python&logoColor=white)](Environment/README.md#cpu-manuscript-analysis)

[![Examples: notebooks](https://img.shields.io/badge/Examples-notebooks-f37726?style=flat-square&logo=jupyter&logoColor=white)](#learn-the-workflow)
[![Data: four datasets](https://img.shields.io/badge/Data-4_datasets-287d8e?style=flat-square)](Manuscript_Data/README.md)
[![Videos: RFdiffusion3](https://img.shields.io/badge/Videos-RFdiffusion3-cc3333?style=flat-square&logo=youtube&logoColor=white)](RFdiffusion3_Tutorial/README.md#learning-resources)
[![Cite: BibTeX](https://img.shields.io/badge/Cite-BibTeX-555555?style=flat-square)](Docs/references.bib)

[![Nature DOI: 10.1038/s41586-025-09746-w](https://img.shields.io/badge/DOI-10.1038%2Fs41586--025--09746--w-287d8e?style=flat-square)](https://doi.org/10.1038/s41586-025-09746-w)
[![Nature Methods DOI: 10.1038/s41592-025-02975-x](https://img.shields.io/badge/DOI-10.1038%2Fs41592--025--02975--x-466ca6?style=flat-square)](https://doi.org/10.1038/s41592-025-02975-x)
[![Science DOI: 10.1126/science.aeh2982](https://img.shields.io/badge/DOI-10.1126%2Fscience.aeh2982-7359a6?style=flat-square)](https://doi.org/10.1126/science.aeh2982)

**From catalytic geometry to designed enzymes.**

Code, tutorials, and experimental data for designing metal-dependent enzymes
with RFdiffusion2 and RFdiffusion3. The repository accompanies the
metallohydrolase study [[1]](#ref-1) and contributes enzyme-design examples and data
to the RFdiffusion2 [[2]](#ref-2) and RFdiffusion3 [[3]](#ref-3) studies.

[**Docs**](Docs/README.md) · [**Quick start**](#quick-start) ·
[**Tutorials**](#learn-the-workflow) · [**Data**](#explore-the-data) ·
[**References / BibTeX**](#citation)

<p align="center">
  <a href="Docs/assets/rfdiffusion3-highlight.mp4?raw=true">
    <img src="Docs/assets/rfdiffusion3-highlight.gif" width="960" alt="RFdiffusion3 phosphotriesterase design: a quantum-chemistry active-site model followed by an example scaffold-generation trajectory and publication credits.">
  </a>
  <br>
  <strong>RFdiffusion3: from catalytic geometry to a protein scaffold</strong>
  <br>
  Phosphotriesterase design around a quantum-chemistry theozyme.
  <br>
  <a href="Docs/assets/rfdiffusion3-highlight.mp4?raw=true">Full-resolution MP4 · 58 seconds</a> · <a href="Docs/media.md">All animations and downloads</a>
</p>

## Choose your starting point

| I want to… | Start here |
| :--- | :--- |
| Explore measurements and figures | [Experimental datasets](Manuscript_Data/README.md) |
| Reanalyze data on a CPU | [Quick start](#quick-start) |
| Learn active-site scaffolding | [RFdiffusion2 tutorial](RFdiffusion2_Tutorial/README.md) · [RFdiffusion3 tutorial](RFdiffusion3_Tutorial/README.md) |
| Adapt a design campaign | [Workflows and available drivers](Docs/workflows.md) |
| Configure a workstation or cluster | [Environments](Environment/README.md) · [Software and weights](Docs/software-and-models.md) |

## Learn the workflow

A **theozyme** specifies catalytic residues, metal ions, and substrate or
transition-state geometry. Diffusion builds a protein scaffold around this
active site; sequence design and validation select candidates for testing.

```mermaid
flowchart LR
    A[Active-site geometry] --> B[Scaffold generation]
    B --> C[Sequence design]
    C --> D[Structure validation]
    D --> E[Experimental characterization]
```

Both tutorials cover theozyme preparation from [PDB 1QJI](https://www.rcsb.org/structure/1QJI)
and the ZETA_2 quantum-chemistry example, followed by scaffold generation and
filtering. Campaign notebooks extend into sequence design and validation.

| Notebook | Model-specific steps |
| :--- | :--- |
| [RFdiffusion2](RFdiffusion2_Tutorial/RFdiffusion2_Tutorial_JupyterNotebook.ipynb) | YAML inputs, unindexed catalytic residues, fixed active-site atoms, filtering |
| [RFdiffusion3](RFdiffusion3_Tutorial/RFdiffusion3_Tutorial_JupyterNotebook.ipynb) | Foundry `rfd3 design`, JSON inputs, atom-level conditioning, filtering, geometry idealization |

**RFdiffusion3 learning resources:** [video walkthroughs and official RosettaCommons guides](RFdiffusion3_Tutorial/README.md#learning-resources),
from installation and motif scaffolding to advanced enzyme design.

<table>
  <tr>
    <th>RFdiffusion2 · Scaffold generation</th>
    <th>RFdiffusion3 · Active-site and scaffold detail</th>
  </tr>
  <tr>
    <td width="50%" align="center">
      <a href="Docs/assets/rfdiffusion2-scaffold-generation.mp4?raw=true">
        <img src="Docs/assets/rfdiffusion2-scaffold-generation.gif" width="400" alt="Complete RFdiffusion2 animation showing a protein scaffold forming around its catalytic site.">
      </a>
    </td>
    <td width="50%" align="center">
      <a href="Docs/assets/rfdiffusion3-supplementary.gif">
        <img src="Docs/assets/rfdiffusion3-supplementary.gif" width="400" alt="Supplementary RFdiffusion3 animation showing the molecular active site and surrounding protein scaffold.">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center"><a href="Docs/assets/rfdiffusion2-scaffold-generation.mp4?raw=true">Full-resolution MP4 · 5.77 seconds</a></td>
    <td align="center"><a href="Docs/assets/rfdiffusion3-supplementary.gif">Original GIF · 11 seconds</a></td>
  </tr>
</table>

## Explore the data

| Dataset | Publication | Available design material |
| :--- | :--- | :--- |
| [Metallohydrolases: phenylacetate hydrolysis](Manuscript_Data/Metallohydrolase_Nature_2026/README.md) | *Nature* [[1]](#ref-1) | [RFdiffusion2 driver and outputs](Design_Pipelines/Metalloesterase_RFdiffusion2/README.md) |
| [Additional metalloesterases: 4MU-butyrate](Manuscript_Data/Metallohydrolase_RFdiffusion2_Nature_Methods_2026/README.md) | *Nature Methods* [[2]](#ref-2) | Shared RFdiffusion2 campaign |
| [Phosphotriesterases: paraoxon hydrolysis](Manuscript_Data/Phosphotriesterase_RFdiffusion3_Science_2026/README.md) | *Science* [[3]](#ref-3) | [RFdiffusion3 driver and inputs](Design_Pipelines/Phosphotriesterase_RFdiffusion3/README.md); production outputs not deposited |
| [Metalloproteases: amide hydrolysis](Manuscript_Data/Metalloprotease_RFdiffusion3_Science_2026/README.md) | *Science* [[3]](#ref-3) | [Campaign placeholder](Design_Pipelines/Metalloprotease_RFdiffusion3/README.md); driver not deposited |

Each dataset has a `wetlab_data_analysis.py` marimo app. See
[data and reproduction](Docs/data-and-reproduction.md) for assay coverage,
figure exports, and the distinction between fresh fits and historical figures.

**Related work:** Chen *et al.* [[4]](#ref-4) reports designed metalloproteases
and applications in selective protein cleavage and caged biomodulator activation.
Its code and data have a separate [Zenodo deposit](https://doi.org/10.5281/zenodo.22654831).

## Quick start

### Analyze experimental data

```bash
git clone https://github.com/baker-laboratory/Metallohydrolase_Enzyme_Design.git
cd Metallohydrolase_Enzyme_Design
conda env create -f Environment/analysis.yml
conda activate zinc_hydro_analysis
marimo edit Manuscript_Data/Metallohydrolase_Nature_2026/wetlab_data_analysis.py
```

No GPU, model weights, or software submodules are needed. Change the dataset
path to open another app. [Other installation options](Docs/getting-started.md).

### Run design workflows

```bash
git submodule update --init --recursive
```

Follow the [environment guide](Environment/README.md): `zinc_hydro` provides
Linux/NVIDIA helper tools; inference models require separate environments and
checkpoints. After setup, inspect helper and ligand-tool resolution:

```bash
python Scripts/env_config.py
python Scripts/repo_paths.py
```

Review model launchers, output paths, container mounts, and scheduler settings
before running notebook cells or jobs. See the [workflow guide](Docs/workflows.md).

## Repository guide

| Directory | Contents |
| :--- | :--- |
| [Docs](Docs/README.md) | Setup, workflows, reproduction, software, and references |
| [Environment](Environment/README.md) | Analysis/helper specifications and container recipes |
| [RFdiffusion2 tutorial](RFdiffusion2_Tutorial/README.md) · [RFdiffusion3 tutorial](RFdiffusion3_Tutorial/README.md) | Teaching notebooks and inputs |
| [Design_Pipelines](Design_Pipelines/README.md) | Campaign drivers and computational deposits |
| [Manuscript_Data](Manuscript_Data/README.md) | Measurements, supplementary deposits, analysis apps, and figures |
| [Scripts](Scripts/) · [Software](Software/) | Shared utilities and pinned third-party tools |

[Detailed repository map](Docs/repository-map.md).

## Citation

Cite the studies corresponding to the methods and data you use.
**[Download BibTeX (.bib)](Docs/references.bib)** ·
[Citation guide](Docs/citations.md) · [CITATION.cff](CITATION.cff).

1. <a id="ref-1"></a>Kim, D., Woodbury, S. M., Ahern, W. *et al.* Computational design of metallohydrolases. *Nature* **649**, 246–253 (2026). [DOI](https://doi.org/10.1038/s41586-025-09746-w).
2. <a id="ref-2"></a>Ahern, W., Yim, J., Tischer, D. *et al.* Atom-level enzyme active site scaffolding using RFdiffusion2. *Nature Methods* **23**, 96–105 (2026). [DOI](https://doi.org/10.1038/s41592-025-02975-x).
3. <a id="ref-3"></a>Butcher, J., Krishna, R., Mitra, R. *et al.* Atomic functional site scaffolding in biopolymers with RFdiffusion3. *Science*, eaeh2982 (2026). [DOI](https://doi.org/10.1126/science.aeh2982).
4. <a id="ref-4"></a>Chen, A., Wu, K., Choi, H. *et al.* Computational design of metalloproteases. *Nature* (2026). [DOI](https://doi.org/10.1038/s41586-026-11167-2). **Related work; separate data deposit.**

## Contributing and contact

[Open an issue](https://github.com/baker-laboratory/Metallohydrolase_Enzyme_Design/issues)
with the failing command, error, and environment details.
[Contributor guide](Docs/contributing.md) · [Agent guidance](AGENTS.md).

**Scientific contacts:** [Seth Woodbury](mailto:woodbuse@uw.edu) · [Donghyo Kim](mailto:donghyo@uw.edu).

Original project code uses the **[MIT License](LICENSE)**. Third-party software
and model parameters retain their [own terms](Docs/software-and-models.md#licensing).
