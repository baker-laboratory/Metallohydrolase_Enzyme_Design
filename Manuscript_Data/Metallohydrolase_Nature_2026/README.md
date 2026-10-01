# Metallohydrolase — Nature 2026

Data accompanying **Computational design of metallohydrolases**,
Kim, Woodbury, Ahern et al., *Nature* **649**, 246–253 (2026),
[doi:10.1038/s41586-025-09746-w](https://doi.org/10.1038/s41586-025-09746-w).
The article appeared online on 3 December 2025; this directory uses the journal
issue year.

The experiments characterize RFdiffusion2-designed zinc-dependent esterases
with **4MU-phenylacetate**. The additional 4MU-butyrate experiments are available
in the [Nature Methods dataset](../Metallohydrolase_RFdiffusion2_Nature_Methods_2026/).
The computational workflow is in
[Metalloesterase_RFdiffusion2](../../Design_Pipelines/Metalloesterase_RFdiffusion2/).

## Analysis

Open [wetlab_data_analysis.py](wetlab_data_analysis.py) with `marimo edit`.
The cells recalculate from the raw measurements in dependency order. See the
[analysis environment instructions](../README.md#running-the-analysis).
No GPU or design software is required.

The notebook reads all 43 primary data files and includes:

- Section I screening plots for both design campaigns, ZETA_1 variants and knockouts.
- Calibration, Michaelis–Menten kinetics, uncatalyzed hydrolysis and zinc dependence.
- Total turnover, zinc-binding curves and circular-dichroism measurements.

Figures are saved in [wetlab_data_plots](wetlab_data_plots/); marimo displays
the plots and fitted values as it runs. The previous `.ipynb` is retained as an archive. `FIGURE_DPI` and `SAVE_EPS` in
the setup cell control figure exports.

## Supplementary data

| File | Contents |
|---|---|
| [Data S1](supplemental_data/Data_S1__4muPA_All_DFT_Theozymes__multiple.xyz) | DFT-optimized 4MU-phenylacetate theozymes |
| [Data S2](supplemental_data/Data_S2__Metallohydrolase_DNA_and_Protein_Sequences.xlsx) | Ordered protein/DNA sequences, variants and expression-vector information |
| [Data S3](supplemental_data/Data_S3__all_models.zip) | Original deposited archive of 199 design models: 192 campaign designs and seven ZETA_1 variants |
| [ZETA_2 apo](supplemental_data/experimental_structures/ZETA_2_apo_9PYJ.pdb) | Deposited crystal structure [9PYJ](https://www.rcsb.org/structure/9PYJ), 3.49 Å |
| [ZETA_2 zinc-bound](supplemental_data/experimental_structures/ZETA_2_Zn_bound_9PYL.pdb) | Deposited crystal structure [9PYL](https://www.rcsb.org/structure/9PYL), 2.07 Å |

The crystal structures are experimental observations; the design models contain
computational transition-state hypotheses. Data S3 retains the original
design-stage coordinates, which can precede final sequence changes. Use Data S2
for the ordered constructs. Individual ZETA_1 knockout coordinate models are
not part of the deposited archive.

The expression construct is `MSG` + design + `GSAWSHPQFEK`, using
**pGG-T7-cStrepII** (also called LM1369, pDT1 or pDTstrep1),
[Addgene 262810](https://www.addgene.org/262810/). The `Cloning_and_Vector`
sheet in Data S2 includes the vector sequence, cloning adapters and sequencing
primers.

## Analysis notes

The notebook uses experiment-specific 4MU calibrations and the substrate ranges
described in the Supplementary Methods. Turnover is calculated after nonlinear
fluorescence-to-concentration conversion and background subtraction; readings
near the calibration limits should be interpreted accordingly. Zinc-binding
plots use the supplied fitted curves rather than refitting binding constants.
Instrument measurements are preserved; local file-location metadata has been
removed from spreadsheet exports.

Designs are identified by source plate well (A1, A8, B9, C4, C5, F7, …).
Notebook section headings identify the manuscript names ZETA_1–ZETA_4.
