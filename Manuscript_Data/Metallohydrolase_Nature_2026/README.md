# Metallohydrolase — *Nature* 2026

Experimental data for **Computational design of metallohydrolases**
(Kim, Woodbury, Ahern et al., *Nature* **649**, 246–253, 2026 — [doi:10.1038/s41586-025-09746-w](https://doi.org/10.1038/s41586-025-09746-w)),
covering the de novo Zn(II) esterases designed with RFdiffusion2.

The journal issue is dated 1 January 2026; the article was first published
online on 3 December 2025. This folder uses the journal issue year.

The matching computational campaign is
[`../../Design_Pipelines/Metalloesterase_RFdiffusion2/`](../../Design_Pipelines/Metalloesterase_RFdiffusion2/).

## Contents

| Path | What it holds |
|---|---|
| `wetlab_data_analysis.ipynb` | Loads every raw file, fits the kinetics, produces the reported figures and statistics |
| `raw_wetlab_data/` | 43 primary files, unmodified instrument output |
| `supplemental_data/` | Deposited supplementary records (see below) |
| `wetlab_data_plots/` | Saved analysis figures, including six Section I screening plots; vector exports are opt-in |

### Raw data (`raw_wetlab_data/`)

| Measurement | Files | Format |
|---|---|---|
| Michaelis–Menten kinetics | 4-MU phenylacetate, varying enzyme and Zn(II) concentrations, 25 °C | `.xlsx` plate reader |
| Functional screening | Design campaigns 1–2, A1 variants, A1 knockouts | `.xlsx` / `.tsv` |
| Zn(II) dependence | Chelation and reconstitution series | `.xlsx` |
| Zn(II) binding | Mag-Fura-2 competition titration measurements and supplied fitted curves for A1 and six knockouts (`Zn-Hydrolase_binding_data.csv`); the notebook plots these curves rather than refitting K<sub>d</sub> | `.csv` |
| Thermal stability | CD melts 25→95 °C and post-heating recovery, 8 designs | `.csv` |
| Turnover | Total catalytic turnover with calibration ladder | `.xlsx` |
| 4MU standard curves | Calibration ladders used to convert fluorescence to product concentration (see §II.I.A) | `.xlsx` |

### Supplementary data (`supplemental_data/`)

| File | Contents |
|---|---|
| `Data_S1__4muPA_All_DFT_Theozymes__multiple.xyz` | DFT-optimized theozyme geometries, all 4MU-PA active-site variants |
| `Data_S2__Metallohydrolase_DNA_and_Protein_Sequences.xlsx` | DNA and protein sequences for every design and variant tested, plus a `Cloning_and_Vector` sheet describing the expression vector |
| `Data_S3__all_models.zip` | Original deposited archive: 192 campaign design models and seven ZETA_1 variants (199 PDBs) |
| `experimental_structures/ZETA_2_apo_9PYJ.pdb` | Unmodified deposited apo ZETA_2 crystal structure, [PDB 9PYJ](https://www.rcsb.org/structure/9PYJ), 3.49 Å resolution |
| `experimental_structures/ZETA_2_Zn_bound_9PYL.pdb` | Unmodified deposited zinc-bound ZETA_2 crystal structure, [PDB 9PYL](https://www.rcsb.org/structure/9PYL), 2.07 Å resolution |
| `experimental_structures/provenance.json` | Download URLs, PDB revisions, retrieval dates, and SHA-256 checksums |
| `notebook_validation.json` | Fresh-kernel execution, input coverage, and key numerical reproduction checks |
| `model_sequence_audit.json` / `model_sequence_differences.csv` | Comparison of deposited model sequences against the Data S2 designed-protein sequences |

These are reference deposits; the analysis notebook does not read them.
The crystal structures are experimental observations of the apo and zinc-bound
states, separate from the transition-state design hypotheses in `Data_S3`.

The model archive is byte-for-byte identical to the journal's Supplementary
Data 3. Of its 199 models, 188 exactly match the designed-protein sequence in
Data S2. Eleven campaign 1 models differ by substitutions or terminal trimming;
these inherited differences are listed in `model_sequence_differences.csv`.
The six ZETA_1 knockout sequences and measurements are included, but separate
knockout coordinate models are not present in the deposited archive. The
published files are retained unchanged so their provenance remains explicit.

Every DNA sequence in `Data_S2` is the full open reading frame as cloned: it
begins with `ATGTCAGGA` and ends with `GGTTCCGCTTGGAGCCACCCGCAGTTCGAAAAATAA`,
both contributed by the expression vector, so the expressed protein is
`MSG` + design + `GSAWSHPQFEK` (GSA linker + Strep-tag II). That vector is
**pGG-T7-cStrepII** (kanamycin, T7/lac, ccdB counter-selection), designed by
[Lucas Milles](https://www.biochem.mpg.de/milles) and deposited as
[Addgene 262810](https://www.addgene.org/262810/); it is also referred to as
LM1369, pDT1 and pDTstrep1. The `Cloning_and_Vector` sheet gives its full
sequence, annotated features, the BsaI adapters needed to order a design, and
the sequencing primers that read across an insert.

The same vector was used for the phosphotriesterase designs in
[`../Phosphotriesterase_RFdiffusion3_Science_2026/`](../Phosphotriesterase_RFdiffusion3_Science_2026/).

## Running it

See [`../README.md`](../README.md) — one small conda environment, no paths to
edit, run the first cell then go top to bottom. That file also describes how the
calibration standard curves are derived.

## Reproduction checks

Section I plots both 96-well design campaign screens, the ZETA_1 variant
screen, and the knockout screen. A fresh-kernel run reads all 43 raw files;
the executed notebook retains the figures and numerical output for viewing on
GitHub.

Two targeted corrections align the existing notebook with the published
[Supplementary Methods, §4.3](https://www.nature.com/articles/s41586-025-09746-w#MOESM1):

- ZETA_4/C5 uses 1.1–72 μM substrate, excluding the 144 μM row from its fit.
  The additional raw measurements remain available in the source spreadsheet.
- The turnover analysis fits fluorescence → product concentration explicitly.
  It previously applied the concentration → fluorescence polynomial in the
  wrong direction. Background is subtracted after nonlinear conversion in the
  additional Fig. 3e turnover plot; separate enzyme/background curves remain.

The corrected turnover endpoint is about **1,141 ± 55** at 119.8 minutes
(mean ± SD of three per-column enzyme-minus-background differences). This
reproduces the reported >1,000-turnover conclusion, rather than an exact export
of the historical panel. Early readings are below the lowest standard and
16 of 1,080 plotted enzyme readings are slightly above the highest standard;
those quartic estimates involve extrapolation. The data are not clipped to the
150 μM initial substrate concentration.

The notebook retains its original plate mappings, kinetic fitting functions,
and plot styling apart from these corrections. Zinc/buffer concentration
headings and printed rate units were also corrected. The 240906 calibration
is re-derived for comparison; its historical stored value differs by 0.29% and
is preserved. Section I.IV plots unconverted fluorescence, so that comparison
does not change the knockout screen.

The files here concern **4MU-phenylacetate**. The additional **4MU-butyrate**
screen and kinetics belong to
[`Metallohydrolase_RFdiffusion2_Nature_Methods_2026`](../Metallohydrolase_RFdiffusion2_Nature_Methods_2026/).
No butyrate files were found in this Nature dataset to move.

## Sample naming

Designs are referred to by their source plate well throughout (`A1`, `A8`, `B9`,
`C4`, `C5`, `F7`, `F11`, `H6`, `H10`, …). `A1` is the most active design and the
subject of the mutational analysis; its knockouts appear as `A1_H130A`,
`A1_N17A`, `A1_D67A`, and so on. Section headers give the manuscript names
(ZETA_1 … ZETA_4) where they apply. Cross-reference with
`supplemental_data/Data_S2` for the full sequence of any design.
