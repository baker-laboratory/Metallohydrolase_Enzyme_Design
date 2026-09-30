# Metallohydrolases — RFdiffusion2, Nature Methods (2026)

Additional **4MU-butyrate (4MU-B)** data for Ahern, Yim, Tischer *et al.*,
“Atom-level enzyme active site scaffolding using RFdiffusion2,”
*Nature Methods* **23**, 96–105 (2026),
[doi:10.1038/s41592-025-02975-x](https://www.nature.com/articles/s41592-025-02975-x).
The shared 4MU-phenylacetate experiments, sequences and structures are in
[Metallohydrolase_Nature_2026](../Metallohydrolase_Nature_2026/).

| File | Contents |
|---|---|
| [Analysis notebook](wetlab_data_analysis.ipynb) | Screening progress curves and kinetics for C4, B11 and D11 |
| [Experimental inputs](raw_wetlab_data/) | Screening plates, calibration and kinetics measurements |
| [Figures and result tables](wetlab_data_plots/) | Notebook outputs |
| [Supplementary sequence and kinetics workbook](supplemental_data/Data_S1__4MU_B_Ordered_Sequences.xlsx) | All 96 ordered protein/DNA sequences, expression constructs and kinetics |
| [Protein sequences: CSV](supplemental_data/ordered_sequences.csv) / [FASTA](supplemental_data/ordered_sequences.fasta) | Ordered protein sequences and model mapping |
| [Transition-state models](supplemental_data/Data_S2__4MU_B_Ordered_Models_with_Transition_States.zip) / [model index](supplemental_data/model_manifest.csv) | 96 sequence-matched computational design models containing the transition state and Zn |

Run all notebook cells with Python, NumPy, pandas, SciPy, matplotlib and
openpyxl. Section I includes the 96-well screen and additional screening
conditions; Section II uses the included 2025-04-01 calibration for the
publication kinetics. D11 is reported as a low-substrate catalytic-efficiency
estimate because separate kcat and KM values are not resolved. Screening
fluorescence is not normalized by enzyme concentration. Instrument file-location
metadata have been removed; measurement values are unchanged.

The model archive contains the three historical publication hit models
(B11, C4 and D11) and 93 supplementary reconstructions matched to the final
ordered sequences. Some reconstructed non-hit models retain nonideal zinc
coordination geometries; these computational models do not establish activity.
