# Design Pipelines

Campaign-specific computational workflows and their supporting inputs, built on
shared [`Scripts/`](../Scripts/) utilities and [`Software/`](../Software/)
submodules. The directories have different levels of deposited material: use
the table below to distinguish available drivers from planned workflows.

The [RFdiffusion2](../RFdiffusion2_Tutorial/) and
[RFdiffusion3](../RFdiffusion3_Tutorial/) tutorials teach scaffold generation.
Campaign drivers extend that work into sequence design and candidate assessment.
See the [workflow guide](../Docs/workflows.md) for stage coverage and execution
requirements.

## Campaigns

| Campaign | Model | Target chemistry | Status |
|---|---|---|---|
| [`Metalloesterase_RFdiffusion2/`](Metalloesterase_RFdiffusion2/) | RFdiffusion2 | Zn(II) esterase, 4MU-phenylacetate hydrolysis | Notebook, inputs and selected outputs available — *Nature* (2026) |
| [`Phosphotriesterase_RFdiffusion3/`](Phosphotriesterase_RFdiffusion3/) | RFdiffusion3 | Binuclear phosphotriesterase | Notebook and inputs available; production outputs must be generated — *Science* (2026) |
| [`Metalloprotease_RFdiffusion3/`](Metalloprotease_RFdiffusion3/) | RFdiffusion3 | Zn(II) metalloprotease, amide hydrolysis | Scope README and directory placeholders; campaign driver not included — *Science* (2026) |

## Layout convention

Campaign directories use the following convention; the metalloprotease
directory currently contains only its README and placeholders:

```
<Chemistry>_<Model>/
├── README.md            # what was designed, which stages ran, how to reproduce
├── <campaign>.ipynb     # the driver notebook: one cell per pipeline stage
├── inputs/              # theozymes, constraints, params, reference structures
└── outputs/             # per-stage outputs (usually gitignored when large)
```

The matching experimental data lives under
[`Manuscript_Data/`](../Manuscript_Data/) in publication-specific directories. The
[dataset index](../Manuscript_Data/README.md) maps them to these campaigns.

## Before you run anything

```bash
# from the repository root
conda env create -f Environment/zinc_hydro.yml
conda activate zinc_hydro
python Scripts/env_config.py     # confirm the interpreter and tools resolve
```

The supplied notebooks locate the repository root automatically. Review output
paths, launchers, container binds and scheduler settings before running generated
commands. The helper environment is only one part of setup: install the model
runtimes and obtain the weights needed for your chosen stages. See
[`Environment/README.md`](../Environment/README.md) and
[software and models](../Docs/software-and-models.md).

[Citation guide](../Docs/citations.md) · [BibTeX bibliography](../Docs/references.bib)
