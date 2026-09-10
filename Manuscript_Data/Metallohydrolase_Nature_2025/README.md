# Metallohydrolase — *Nature* 2025

Experimental data for **Computational design of metallohydrolases**
(Kim, Woodbury, Ahern et al., *Nature*, 2025 — [doi:10.1038/s41586-025-09746-w](https://doi.org/10.1038/s41586-025-09746-w)),
covering the de novo Zn(II) esterases designed with RFdiffusion2.

The matching computational campaign is
[`../../Design_Pipelines/Metalloesterase_RFdiffusion2/`](../../Design_Pipelines/Metalloesterase_RFdiffusion2/).

## Contents

| Path | What it holds |
|---|---|
| `wetlab_data_analysis.ipynb` | Loads every raw file, fits the kinetics, produces the reported figures and statistics |
| `raw_wetlab_data/` | 41 primary files, unmodified instrument output |
| `supplemental_data/` | Deposited supplementary records (see below) |
| `wetlab_data_plots/` | Every figure the notebook writes — 65 PNGs at 150 dpi, committed; vector exports are opt-in |

### Raw data (`raw_wetlab_data/`)

| Measurement | Files | Format |
|---|---|---|
| Michaelis–Menten kinetics | 4-MU phenylacetate, varying enzyme and Zn(II) concentrations, 25 °C | `.xlsx` plate reader |
| Functional screening | Design campaigns 1–2, A1 variants, A1 knockouts | `.xlsx` / `.tsv` |
| Zn(II) dependence | Chelation and reconstitution series | `.xlsx` |
| Zn(II) binding | Mag-Fura-2 competition titration, K<sub>d</sub> for A1 and six knockouts (`Zn-Hydrolase_binding_data.csv`) | `.csv` |
| Thermal stability | CD melts 25→95 °C and post-heating recovery, 8 designs | `.csv` |
| Turnover | Total catalytic turnover with calibration ladder | `.xlsx` |
| 4MU standard curves | Calibration ladders used to convert fluorescence to product concentration (see §II.I.A) | `.xlsx` |

### Supplementary data (`supplemental_data/`)

| File | Contents |
|---|---|
| `Data_S1__4muPA_All_DFT_Theozymes__multiple.xyz` | DFT-optimized theozyme geometries, all 4MU-PA active-site variants |
| `Data_S2__Metallohydrolase_DNA_and_Protein_Sequences.xlsx` | DNA and protein sequences for every design and variant tested, plus a `Cloning_and_Vector` sheet describing the expression vector |
| `Data_S3__all_models.zip` | Structural models for all designs |

These are reference deposits; the analysis notebook does not read them.

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
[`../Phosphotriesterase_RFdiffusion3/`](../Phosphotriesterase_RFdiffusion3/).

## Running it

See [`../README.md`](../README.md) — one small conda environment, no paths to
edit, run the first cell then go top to bottom. That file also describes how the
calibration standard curves are derived.

## Sample naming

Designs are referred to by their source plate well throughout (`A1`, `A8`, `B9`,
`C4`, `C5`, `F7`, `F11`, `H6`, `H10`, …). `A1` is the most active design and the
subject of the mutational analysis; its knockouts appear as `A1_H130A`,
`A1_N17A`, `A1_D67A`, and so on. Section headers give the manuscript names
(ZETA_1 … ZETA_4) where they apply. Cross-reference with
`supplemental_data/Data_S2` for the full sequence of any design.
