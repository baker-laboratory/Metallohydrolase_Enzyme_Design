# Metalloprotease design with RFdiffusion3

> **Repository status: campaign driver not included.** This directory contains
> this scope README and placeholder input/output directories. The associated
> experimental data are available for the published RFdiffusion3 study in
> *Science* (2026), [doi:10.1126/science.aeh2982](https://doi.org/10.1126/science.aeh2982).

## Scope

Design of de novo metalloproteases with a mononuclear Zn(II) center for amide
bond hydrolysis, using **RFdiffusion3**. The shared
[RFdiffusion3 tutorial](../../RFdiffusion3_Tutorial/) covers theozyme preparation,
scaffold generation, filtering, and geometry idealization. Campaign-specific
execution instructions require the missing driver and inputs.

## Planned layout

```
Metalloprotease_RFdiffusion3/
├── README.md
├── design_metalloprotease.ipynb   # driver notebook
├── inputs/                        # theozymes, constraints, params
└── outputs/                       # per-stage outputs
```

The matching experimental data are available at
[`../../Manuscript_Data/Metalloprotease_RFdiffusion3_Science_2026/`](../../Manuscript_Data/Metalloprotease_RFdiffusion3_Science_2026/).

## In the meantime

The [RFdiffusion3 tutorial](../../RFdiffusion3_Tutorial/) includes two theozyme
preparation examples and a 1QJI scaffold-generation workflow through filtering
and geometry idealization. It is a starting point for adapting the method;
it does not supply this campaign's missing notebook. See the
[workflow guide](../../Docs/workflows.md) for available downstream pipelines.

## Related work

For a related study of designed zinc-dependent proteases, see Chen, A., Wu, K.,
Choi, H. *et al.*, **Computational design of metalloproteases**, *Nature* (2026),
[doi:10.1038/s41586-026-11167-2](https://doi.org/10.1038/s41586-026-11167-2).
The paper's code and data are available in its own
[Zenodo archive](https://doi.org/10.5281/zenodo.22654831).
This directory remains the placeholder for the RFdiffusion3 *Science* campaign;
the Nature study's deposit is linked as related work. See the
[citation guide](../../Docs/citations.md#related-work-computational-design-of-metalloproteases).

[BibTeX bibliography](../../Docs/references.bib)
