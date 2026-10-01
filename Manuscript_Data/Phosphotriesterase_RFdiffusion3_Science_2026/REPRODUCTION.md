# Analysis and figure notes

Open `wetlab_data_analysis.py` with `marimo edit`. The app fits the four
calibrations, uncatalyzed rates and all 19 enzyme/condition entries directly
from deposited acquisitions. It displays both complete screening experiments,
checks well/scaffold assignments, refits all 192 round-2 screening rates, and
repeats the tagless detector-linearity analysis. The export button writes
tables, per-design kinetics plots, complete screening plots and the historical
SI/Figure 4 reproductions. The latter remain under
`wetlab_data_plots/paper_figures/` as PDF and PNG.

`Scripts/wetlab_platereader/analysis_pte_rfd3.py` contains the reusable analysis
functions. The app does not load Jupyter cells or saved Jupyter outputs.
`wetlab_data_analysis.ipynb` is retained as the historical workflow.

The notebook checks the 18 original paper entries against
`supplemental_data/paraoxon_kinetics_ALL_DATA.xlsx`, accounting for the reference
workbook's stored precision. `kinetics_summary.csv` includes fresh fits,
fit-only errors, total errors and reported parameters. Total errors incorporate
the specified 6% enzyme-concentration uncertainty and calibration uncertainty;
the shared calibration cancels in ratios to the condition-matched uncatalyzed rate.
H89A and H170A use first-order efficiencies because separate kcat and KM are
not resolved. Their nonlinear fits remain available as diagnostics.

The marimo assay viewer uses the current fitted points, parameters and
condition-matched background rate for its lines and annotations. Its display
strings and mutant first-order estimates are recalculated; they are not copied
from the reference workbook. Reference values are used for validation and
historical manuscript reproduction only. The summary's `reported` columns
apply the manuscript formatting/reporting rules to the fresh values.

## Figure conventions

Progress curves show the complete acquisition; shaded intervals lie outside
the initial-rate fitting window. Some high-concentration traces extend beyond
the reader's linear range. Figure annotations use the reference workbook's
reporting precision. Arial matches the supplied figure typography; another
installed font may change text geometry.

Figure 4E is provided in two versions. The original reproduction retains the
original no-bicarbonate point calibration. The `__condition_matched` variant
uses the no-bicarbonate calibration for those points and error bars, consistent
with its fitted curve. Notebook fits and SI panels use condition-matched
calibration throughout.

## Complete screening views in SI typography

The [round-1](wetlab_data_plots/screen_round1_SI_style.png) and
[round-2](wetlab_data_plots/screen_round2_SI_style.png) plots use
`plot_screening_PTE_RFd3(round_number, plot_path=..., dpi=600)`. All 96 and
192 curves are drawn individually, colored by scaffold. The four round-1
and seven round-2 designs tested by kinetics are drawn last with thicker
lines and direct endpoint labels. No percentile envelope, hit filter,
rate fitting, or removal of negative-going traces is applied to these plots.

Time is in hours and ΔA₄₀₅ is relative to each well's first acquisition.
Typography, axes and endpoint leaders follow the SI presentation. Scaffold
assignments come from the deposited `all_designs` workbook sheet. All six
round-1 scaffolds and eleven round-2 scaffolds are covered, retaining the
five characterized-family colors. Round-2 workbook/model scaffold IDs 7–17
correspond to original order IDs 1–11; the app checks the assignments against
the order FASTA. An optional scaffold selector provides a closer view without
replacing the complete screens. Historical manuscript figures retain their
original artwork.

## Tagless ZAPP-1 SI

Tagless ZAPP-1 is a supplementary construct analysis in addition to the
original 18 paper kinetics entries. It uses the ZAPP-3/ZAPP-4
plate, enzyme columns 9–11, background column 12 and 23.4 µM enzyme. The notebook
and SI workbook include this result; the app also repeats the shorter-window
detector-linearity check. The tagged/tagless differences are within the stated
uncertainties.

The final tagless SI artwork, including intact mass, is deposited as
[PDF](wetlab_data_plots/paper_figures/pdf/PTE__v2SI__ZAPP1_tagless.pdf),
[PNG](wetlab_data_plots/paper_figures/png/PTE__v2SI__ZAPP1_tagless.png) and
[TeX caption](wetlab_data_plots/paper_figures/pdf/PTE__v2SI__ZAPP1_tagless.tex).
[Source tables, sequences and methods](supplemental_data/ZAPP1_tagless/)
accompany it. The export regenerates the kinetic analysis and original side
panel. The final SI artwork incorporates the separately deposited
[tagless mass-spectrum image](mass_spectrometry/ZAPP1_tagless/).

[Mass-spectrometry source images](mass_spectrometry/) also include the five
TEO/control samples, each with an overview and a zoom. These are original
instrument-rendered spectra; the paired views are separate exports of the
same acquisition.

The reference workbook identifies mutant residue labels as inferred from
catalytic-motif order; it does not provide direct sequencing confirmation of
that mapping. Labels in these figures follow the reference workbook.
