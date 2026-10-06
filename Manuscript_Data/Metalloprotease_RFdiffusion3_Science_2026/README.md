# Metalloprotease — RFdiffusion3 Science 2026

Wet lab measurements for the RFdiffusion3 metalloprotease campaign, with
marimo analyses of cis cleavage, the trans fluorescence screen, G11 peptide
specificity and the G11+Zn mass spectrum.

## Data files

Source measurements are supplied as nine CSV files and five Excel workbooks.
Filenames retain their experiment identifiers.

| Measurements | Location |
|---|---|
| Cis cleavage quantification at 1 h and 18 h; selected construct sequences and properties | [`cis_screen/`](cis_screen/) |
| Trans fluorescence screening | [CSV](raw_data_files/260310_TTR_alphasyn_trans_screen.csv) and [Excel workbook](raw_data_files/260310_TTR_alphasyn_trans_screen.xlsx) |
| Kinetics plate measurements | [CSV](raw_data_files/260310_TTR_kinetics.csv) and [Excel workbook](raw_data_files/260310_TTR_kinetics.xlsx) |
| Inner-filter calibration measurements | [CSV](raw_data_files/260310_TTR_kinetics_IFE_correction.csv) |
| Zinc-condition plate measurements | [Zinc-condition plate](raw_data_files/260312_TTR_columns135_G4_10_11_columns135wZn_columns7911noZn.csv) and [Zn/EDTA plate](raw_data_files/260312_TTR_zn_dependence_column135_g4g10g11_5uMZn_noaddZn_EDTA_no_enz.csv) |
| G11 fluorescence measurements with seven short peptide reporters | [CSV](raw_data_files/260312_TTR_G11_specificity.csv) and [Excel workbook](raw_data_files/260312_TTR_G11_specificity.xlsx) |
| G11+Zn deconvoluted mass spectrum, mass and intensity values | [`AC_G11+Zn.csv`](raw_data_files/AC_G11+Zn.csv) |

## Run the analysis

From the repository root:

```bash
conda env create -f Environment/analysis.yml
conda activate zinc_hydro_analysis
marimo edit Manuscript_Data/Metalloprotease_RFdiffusion3_Science_2026/wetlab_data_analysis.py
```

The [marimo notebook](wetlab_data_analysis.py) reads the deposited CSV files
and generates the screening, specificity and mass-spectrum plots and tables in
[`wetlab_data_plots/`](wetlab_data_plots/). The cis analysis summarizes cleavage
fractions; the trans analysis displays fluorescence progress curves and
well-level slopes. The specificity analysis displays fluorescence time courses
with baseline and no-enzyme correction.

The matching design campaign is
[`Metalloprotease_RFdiffusion3`](../../Design_Pipelines/Metalloprotease_RFdiffusion3/).
