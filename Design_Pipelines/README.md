# Design Pipelines

End-to-end design campaigns. Each subdirectory is one target chemistry run with
one generative model, self-contained apart from the shared [`Scripts/`](../Scripts/)
utilities and [`Software/`](../Software/) submodules.

These differ from the [tutorials](../RFdiffusion2_Tutorial/) in intent. A tutorial
teaches the workflow on a small worked example you can run in an afternoon. A
pipeline here is the full campaign as actually executed — the same commands, at
production scale, with the filters and thresholds that produced the reported
designs.

## Campaigns

| Campaign | Model | Target chemistry | Status |
|---|---|---|---|
| [`Metalloesterase_RFdiffusion2/`](Metalloesterase_RFdiffusion2/) | RFdiffusion2 | Zn(II) esterase, 4MU-phenylacetate hydrolysis | **Published** — Kim, Woodbury, Ahern et al., *Nature* (2025) |
| [`Phosphotriesterase_RFdiffusion3/`](Phosphotriesterase_RFdiffusion3/) | RFdiffusion3 | Binuclear phosphotriesterase | In preparation — releasing with RFdiffusion3 |
| [`Metalloprotease_RFdiffusion3/`](Metalloprotease_RFdiffusion3/) | RFdiffusion3 | Zn(II) metalloprotease, amide hydrolysis | In preparation — releasing with RFdiffusion3 |

## Layout convention

New campaigns follow the same shape, so a reader who has seen one can navigate
any of them:

```
<Chemistry>_<Model>/
├── README.md            # what was designed, which stages ran, how to reproduce
├── <campaign>.ipynb     # the driver notebook: one cell per pipeline stage
├── inputs/              # theozymes, constraints, params, reference structures
└── outputs/             # per-stage outputs (usually gitignored when large)
```

The matching experimental data lives under
[`Manuscript_Data/`](../Manuscript_Data/) with a directory of the same name, so
computation and measurement stay findable from each other without duplicating
either.

## Before you run anything

```bash
conda env create -f ../Environment/zinc_hydro.yml
conda activate zinc_hydro
python ../Scripts/env_config.py     # confirm the interpreter and tools resolve
```

The notebooks locate the repository root themselves — there are no paths to
edit. See [`Environment/README.md`](../Environment/README.md) if the report shows
a problem.
