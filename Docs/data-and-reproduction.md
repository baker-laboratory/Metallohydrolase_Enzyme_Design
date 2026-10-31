# Data and reproduction

[← Documentation](README.md)

Each dataset has a [marimo](https://marimo.io/) analysis app, source measurements,
and saved figures or tables. Analysis runs on a CPU without the design stack.

| Dataset | Scope | App |
| :--- | :--- | :--- |
| [Metallohydrolases — Nature](../Manuscript_Data/Metallohydrolase_Nature_2026/) | Phenylacetate screens, kinetics, calibration, zinc dependence, turnover, biophysical assays | [Open](../Manuscript_Data/Metallohydrolase_Nature_2026/wetlab_data_analysis.py) |
| [RFdiffusion2 — Nature Methods](../Manuscript_Data/Metallohydrolase_RFdiffusion2_Nature_Methods_2026/) | Additional 4MU-butyrate assays; links to shared Nature results | [Open](../Manuscript_Data/Metallohydrolase_RFdiffusion2_Nature_Methods_2026/wetlab_data_analysis.py) |
| [RFdiffusion3 phosphotriesterases — Science](../Manuscript_Data/Phosphotriesterase_RFdiffusion3_Science_2026/) | Paraoxon screens, kinetics, figure reproduction, tagless ZAPP-1, mass spectra | [Open](../Manuscript_Data/Phosphotriesterase_RFdiffusion3_Science_2026/wetlab_data_analysis.py) |
| [RFdiffusion3 metalloproteases — Science](../Manuscript_Data/Metalloprotease_RFdiffusion3_Science_2026/) | Cis cleavage, trans fluorescence screening, G11 specificity and mass spectrum | [Open](../Manuscript_Data/Metalloprotease_RFdiffusion3_Science_2026/wetlab_data_analysis.py) |

[Citation guide](citations.md) · [BibTeX bibliography](references.bib).

## Run and export

Install the [analysis environment](getting-started.md), then run from the
repository root:

```bash
marimo edit Manuscript_Data/Metallohydrolase_Nature_2026/wetlab_data_analysis.py
```

Change the dataset path to select another app. Cells execute in dependency
order. Read export settings first: apps can write into tracked
`wetlab_data_plots/` directories. The phosphotriesterase app has an explicit
export button for tables and paper figures. Use a separate checkout when
comparing regenerated outputs with committed references.

For an executed HTML export:

```bash
marimo export html Manuscript_Data/Metallohydrolase_Nature_2026/wetlab_data_analysis.py -o analysis.html
```

Saved figures can be viewed without installation. Historical Jupyter notebooks
remain in the Nature, Nature Methods, and phosphotriesterase datasets; the
current marimo apps do not execute them.

## Interpret the results

**Calibration and fitting.** Use each dataset's analysis notes. Nature assays
use experiment-specific 4MU calibrations; phosphotriesterase
[reproduction notes](../Manuscript_Data/Phosphotriesterase_RFdiffusion3_Science_2026/REPRODUCTION.md)
explain fitting and reporting conventions. Metalloprotease screening slopes
are not complete kinetic fits for every deposited plate.

**Sequences and structures.** Use the ordered-construct sequence workbooks.
Nature models preserve design-stage coordinates; Nature Methods distinguishes
historical models from sequence-matched reconstructions. The phosphotriesterase
README explains plate, well, and scaffold numbering. Keep computational models
and experimental crystal structures identified separately.

**Figures and new runs.** Preserve labels distinguishing historical panels,
complete screens, and additional analyses. Library and font versions can affect
numerical output and rendering. Record the commit, environment, and changed
settings when comparing runs. Full computational reproduction also requires
model resources, hardware, and campaign-specific configuration; see
[workflows](workflows.md).
