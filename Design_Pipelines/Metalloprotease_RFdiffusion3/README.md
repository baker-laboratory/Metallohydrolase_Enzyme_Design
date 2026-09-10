# Metalloprotease design with RFdiffusion3

> **Status: in preparation.** This campaign is being prepared for release
> alongside RFdiffusion3, and will be published here shortly.

## Scope

Design of de novo metalloproteases (mononuclear Zn(II) center, amide bond hydrolysis) using **RFdiffusion3** via the
[foundry](https://github.com/RosettaCommons/foundry) framework, following the
same staged workflow as
[`../Metalloesterase_RFdiffusion2/`](../Metalloesterase_RFdiffusion2/):

1. Theozyme preparation (quantum chemistry or crystal-structure derived)
2. ORI token placement
3. RFdiffusion3 backbone generation
4. Two-stage filtering — rapid JSON metrics, then PyRosetta scaffold analysis
5. Geometry idealization (Cartesian relaxation)
6. Sequence design (LigandMPNN + FastMPNNDesign)
7. Structure prediction and ligand placement (AF2 / PLACER)

## Planned layout

```
Metalloprotease_RFdiffusion3/
├── README.md
├── design_metalloprotease.ipynb   # driver notebook
├── inputs/                        # theozymes, constraints, params
└── outputs/                       # per-stage outputs
```

The matching experimental data will appear at
[`../../Manuscript_Data/Metalloprotease_RFdiffusion3/`](../../Manuscript_Data/Metalloprotease_RFdiffusion3/).

## In the meantime

The RFdiffusion3 workflow is already demonstrated end to end, on two worked
examples, in [`../../RFdiffusion3_Tutorial/`](../../RFdiffusion3_Tutorial/).
That tutorial is runnable today and is the right starting point for adapting the
method to your own chemistry.
