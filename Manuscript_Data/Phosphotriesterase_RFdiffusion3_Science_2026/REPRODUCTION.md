# Analysis and figure notes

Run `wetlab_data_analysis.ipynb` in order. Sections I–III fit the calibration,
uncatalyzed rates and 19 enzyme/condition entries; section IV analyzes the two
screens. Section V writes the kinetics and background SI figures, Figure 4
panels E/F/G/I and the additional tagless comparison to
`wetlab_data_plots/paper_figures/` as PDF and PNG.

The notebook checks the 18 original paper entries against
`supplemental_data/paraoxon_kinetics_ALL_DATA.xlsx`, accounting for the reference
workbook's stored precision. `kinetics_summary.csv` includes fresh fits,
fit-only errors, total errors and reported parameters. Total errors incorporate
the specified 6% enzyme-concentration uncertainty and calibration uncertainty;
the shared calibration cancels in ratios to the condition-matched uncatalyzed rate.
H89A and H170A use first-order efficiencies because separate kcat and KM are
not resolved. Their nonlinear fits remain available as diagnostics.

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

## Additional tagless comparison

Tagless ZAPP-1 is an exploratory comparison outside the original 18 paper
kinetics entries, not an additional designed hit. It uses the ZAPP-3/ZAPP-4
plate, enzyme columns 9–11, background column 12 and 23.4 µM enzyme. The notebook
and SI workbook include this result; section V also repeats the shorter-window
detector-linearity check. The tagged/tagless differences are within the stated
uncertainties.

The reference workbook identifies mutant residue labels as inferred from
catalytic-motif order; it does not provide direct sequencing confirmation of
that mapping. Labels in these figures follow the reference workbook.
