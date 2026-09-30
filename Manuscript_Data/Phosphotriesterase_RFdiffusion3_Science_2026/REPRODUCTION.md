# RFdiffusion3 phosphotriesterase reproduction

The analysis notebook runs from the deposited raw files without the original
laboratory filesystem. It preserves the original fitting method and screening
analysis, adds the tagless ZAPP-1 comparison, and calls a separate
`run_kinetics_PTE_RFd3` helper for the RFdiffusion3 figure style.

## Numerical validation

All 18 original paper fits reproduce the supplied
`paraoxon_kinetics_ALL_DATA.xlsx` at its saved precision. That workbook is
included **unchanged** in `supplemental_data/`; its fit-number columns were
already rounded (typically 0.000001 s⁻¹ for kcat, 0.01 µM for KM and
0.001 M⁻¹ s⁻¹ for efficiency). The no-bicarbonate row rescales previously rounded
numbers, so direct fresh fits need not be identical at every decimal place.
Small covariance-error differences from numerical finite differences are
allowed at 0.003% relative tolerance in addition to saved precision.

The notebook checks 234 quantities and writes
[`paper_reproduction_validation.csv`](wetlab_data_plots/paper_reproduction_validation.csv):

- 108 raw-fit parameters and fit errors across 18 designs/conditions.
- 12 first-order mutant efficiencies and uncertainties.
- Four uncatalyzed-rate values/errors and the two accepted-window counts
  (903 without bicarbonate, 838 with bicarbonate).
- 108 total-error and derived-ratio calculations from the workbook's stored,
  rounded fit inputs. The CSV identifies these separately from raw-data refits.

Errors combine the fit uncertainty, the source's assumed 6% enzyme-concentration
uncertainty and its calibration uncertainty (0.192% without bicarbonate,
0.135% with bicarbonate). Calibration cancels in ratios using the same
condition-matched uncatalyzed rate. Fresh full-precision values and fit-only/total
errors are available in `kinetics_summary.csv`; figure annotations retain the
original paper's reporting precision.

H89A (MUT2) and H170A (MUT4) retain their original first-order reporting:
separate kcat and KM are not resolved. The nonlinear fits are preserved as
CSV diagnostics, and the displayed summary and figures use the first-order
estimates instead of presenting those kcat/KM extrapolations as resolved.

## Figure reproduction

Section V reconstructs these supplied PDFs using the manuscript's figure
scripts, with portable input paths. Palettes, panel geometry, progress-curve
baselines, shaded fit-window exclusions, SEM bands, scaffold colors, titles,
callout leaders and mutant reporting are preserved.

| Figure | Output subdirectory |
|---|---|
| Round 1 kinetics and screening | `paper_figures/pdf/PTE__v2SI__kinetics_round1.pdf` |
| Round 2 kinetics and screening | `paper_figures/pdf/PTE__v2SI__kinetics_round2.pdf` |
| ZAPP-1 knockouts and multiple turnover | `paper_figures/pdf/PTE__v2SI__ZAPP1_knockouts.pdf` |
| Uncatalyzed rate and derived ratios | `paper_figures/pdf/PTE__v2SI__kuncat.pdf` |
| Figure 4 panels E, F, G and I | `paper_figures/main_fig4/` |
| Tagless ZAPP-1 comparison | `paper_figures/side/SIDE__ZAPP1_tagless.pdf` |

All nine reference PDFs were rendered with Poppler at 150 dpi and compared to
freshly generated PDFs: **identical dimensions and pixels for every figure**.
[`figure_validation.json`](wetlab_data_plots/paper_figures/figure_validation.json)
records the comparisons. Source SHA-256 hashes are recorded in
[`reproduction_sources.json`](supplemental_data/reproduction_sources.json).
PDF metadata timestamps can differ; byte-identical PDF files are not claimed.
Arial was available for this check. The original font fallback is retained, so
systems without Arial can run the notebook but may have different text geometry.
PNG previews use 150 dpi to keep the deposit compact; PDFs remain vector.

The uncatalyzed-rate and tagless PDFs predate a later source-script typography
change. Their original italic descriptive subscripts, split math text and
0.02-inch export padding are retained to reproduce the supplied PDFs exactly.
The structural, mass-spectrometry and MLFF panels in the source collection are
outside this wetlab kinetics notebook's scope.

### Historical panel E calibration inconsistency

The supplied panel E script, and its actual PDF, use the bicarbonate calibration
(0.00667808 A405/µM) for **both sets of plotted points**, while the no-bicarbonate
curve and annotations use the corrected, condition-matched calibration
(0.00633035 A405/µM). The historical panel is retained for exact comparison.
A second output, `PTE__v2MAIN__fig4E__bicarbonate_MM__condition_matched`, corrects
only the no-bicarbonate points and their error bars, increasing those by
0.00667808/0.00633035 = 1.05493. The notebook's numerical fits and SI plots
already use condition-matched calibration.

## Tagless ZAPP-1

The source notebook section 11 calls this an exploratory side analysis, outside
the original paper figure set and 18-row workbook. It is included here as an
**additional tagless comparison**, not a new designed hit. Its raw measurements
were already present in the deposited plate shared with ZAPP-3 and ZAPP-4:

- `260731_p2E3_28pt9uM...ZAPP1tagless_23pt4uM...xlsx`, rows A–F.
- Enzyme columns 9–11, background column 12; enzyme concentration 23.4 µM.
- Paraoxon 0.3–9.6 mM; +25 mM bicarbonate; fit window 300–20,000 s,
  clipped to the acquired time course.

The fresh fit gives kcat = 0.0167046 s⁻¹ (total uncertainty 0.0060916 s⁻¹),
KM = 6.93867 mM (fit uncertainty 4.59592 mM), and kcat/KM =
2.40746 M⁻¹ s⁻¹. Reported at the source's uncertainty precision, these are
0.017 ± 0.007 s⁻¹, 7 ± 5 mM and 2 ± 2 M⁻¹ s⁻¹. High-concentration traces exceed
the detector's linear range; the original side-figure sensitivity check is
rerun and printed by section V. These results do not establish a measurable
tag-removal effect within the stated uncertainties.

The sequence/SI workbook appends this result and a `reproduction_notes` sheet.
All 18 original kinetics rows and every other pre-existing sheet value are
preserved. The tagless entry references the existing ZAPP-1 design sequence;
no unprovided tagless cloning DNA sequence is invented.
`kinetics_pte_rfd3.update_tagless_supplement` can regenerate this additional row
from the notebook's `FITS['ZAPP-1 tagless']['mm']` and condition-matched `KUNCAT`.

## Preserved source caveats

The archived source workbook describes the mapping of MUT1–6 to H93A, H89A,
K16A, H170A, H133A and E92A as inferred from catalytic-motif order, rather than
confirmed from a sequencing record. The reproduced labels follow that source;
the caveat is also retained in the added SI notes.

The source detector-linearity notes are retained. The complete-acquisition
progress plots can include points beyond the reader's linear range; shaded
regions indicate the original fitting windows. These plots are reproductions
of the reported analysis, rather than a claim that all displayed absorbance
readings are quantitatively linear.
