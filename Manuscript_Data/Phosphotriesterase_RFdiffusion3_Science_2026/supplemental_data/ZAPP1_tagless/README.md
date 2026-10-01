# Tagless ZAPP-1 supplementary figure and source data

The final supplementary figure combines tagless ZAPP-1 kinetics, the tagged/tagless Michaelis–Menten curves and intact-protein mass spectrometry.

- [Final SI figure (PDF)](../../wetlab_data_plots/paper_figures/pdf/PTE__v2SI__ZAPP1_tagless.pdf), [PNG preview](../../wetlab_data_plots/paper_figures/png/PTE__v2SI__ZAPP1_tagless.png) and [LaTeX caption](../../wetlab_data_plots/paper_figures/pdf/PTE__v2SI__ZAPP1_tagless.tex).
- [Original intact-protein spectrum](../../mass_spectrometry/ZAPP1_tagless/SMW_ZAPP_tagless.png).
- [Purification and kinetic-methods excerpts](methods.tex) and [bibliography](references.bib). These LaTeX fragments use the original supplementary manuscript's preamble, cross-references and surrounding general methods.

| File | Contents |
|---|---|
| `tagless_progress_curves.csv` | Background-corrected, first-point-zeroed product concentrations for all 18 enzyme wells; time in seconds and concentrations in µM |
| `tagless_replicate_rates.csv`, `tagged_replicate_rates.csv` | Individual initial slopes and matched-background corrections for six substrate concentrations in technical triplicate |
| `tagless_MM_points.csv`, `tagged_MM_points.csv` | Concentration means, standard deviations, sample counts and enzyme-normalized rates with standard errors |
| `tagless_MM_95CI.csv`, `tagged_MM_95CI.csv` | Fitted mean response and approximate pointwise 95% confidence limits, in s⁻¹ |
| `analysis.json` | Fit parameters, uncertainty definitions, covariance matrices, sample mapping, relative input paths and intact-mass calculation |
| `tagless_ZAPP1.fasta` | 203-residue protein after intein cleavage: `SVG` + 198-residue ZAPP-1 design + `GS` |
| `expressed_ZAPP1.fasta` | 249-residue expressed His₆–Protein Select fusion before cleavage |

The raw tagged and tagless plate-reader workbooks are deposited in [`raw_wetlab_data`](../../raw_wetlab_data/). The [analysis notebook](../../wetlab_data_analysis.ipynb) fits these data and generates the earlier tagged/tagless side figure. The final assembled three-panel SI figure is deposited as supplied; its plotted numerical data are provided here. Its tagless parameters agree with the `kinetics` sheet of the [supplemental sequence and kinetics workbook](../supp_data__denovo_PTE_DNA_and_protein_sequences.xlsx).

Both preparations use 25 mM bicarbonate and 300–1800 s initial-rate fits. Enzyme concentrations are 23.4 µM (tagless) and 24.0 µM (Strep-tagged). Curves use unweighted Michaelis–Menten fits to six concentration means. The confidence bands describe uncertainty in the fitted mean response, with four residual degrees of freedom; reported kcat and kcat/Km errors include enzyme-concentration and calibration uncertainty, while these fit-only bands exclude those contributions. SI rate enhancement uses the reference workbook's stored uncatalyzed rate; the notebook's fresh fit yields the same reported rounded value.

The expected average mass after cleavage is 22,465.79 Da; the supplied spectrum labels the principal peak at 22,466.27 Da. The original spectrum image is retained separately from the annotated SI artwork.
