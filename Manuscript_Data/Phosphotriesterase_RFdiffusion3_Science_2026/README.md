# Phosphotriesterase — RFdiffusion3 Science 2026

Experimental data for de novo binuclear Zn(II) phosphotriesterases designed with
RFdiffusion3 and assayed on paraoxon. The corresponding computational campaign
is in [Design_Pipelines/Phosphotriesterase_RFdiffusion3](../../Design_Pipelines/Phosphotriesterase_RFdiffusion3/).

## Data

| Location | Contents |
|---|---|
| `wetlab_data_analysis.ipynb` | Calibration, background rates, kinetics, screening and manuscript figures |
| `raw_wetlab_data/` | 15 files: calibration and background plates, nine kinetics plates, two screens, an order FASTA and design-assignment table |
| `supplemental_data/supp_data__denovo_PTE_design_models.zip` | 288 design models: 96 from campaign 1 and 192 from campaign 2 |
| `supplemental_data/supp_data__denovo_PTE_DNA_and_protein_sequences.xlsx` | Sequences, model identifiers, cloning information, 18 paper kinetics entries and an additional tagless ZAPP-1 comparison |
| `supplemental_data/paraoxon_kinetics_ALL_DATA.xlsx` | Reference kinetic parameters, uncertainty budget and reporting notes used by the figure routines |
| `wetlab_data_plots/` | Analysis tables and PNG figures; `paper_figures/` also contains vector PDFs |
| [Analysis notes](REPRODUCTION.md) | Fitting conventions, figure outputs and interpretation notes |

The two eluate screens cover 96 campaign-1 designs and 192 campaign-2 designs.
Kinetics use six paraoxon concentrations, three enzyme replicates and a
background measurement per concentration. Instrument measurements are unchanged;
local file metadata has been removed from the workbooks. The tagless ZAPP-1 measurements
share a plate with ZAPP-3 and ZAPP-4.

## Run the analysis

From this directory:

```bash
conda env create -f ../../Environment/analysis.yml
conda activate zinc_hydro_analysis
jupyter lab wetlab_data_analysis.ipynb
```

Run the notebook from top to bottom. It locates the repository automatically
and uses the deposited files; no GPU or external laboratory filesystem is
required. Section V generates the manuscript-style figures. Arial reproduces
the figure typography; a fallback font is used when Arial is unavailable.

The notebook uses `Scripts/wetlab_platereader/kinetics_pte_rfd3.py` for the
RFdiffusion3 plot style and the shared plate-reader fitter for numerical analysis.
PNG output is 150 dpi. Manuscript panels also produce PDF; optional per-design
EPS export is controlled by `SAVE_EPS`.

## Sequence and model identifiers

Workbook identifiers use `R<campaign> p<plate><well>`, for example `R1 p1D1`.
The `model_file` column links each design to its PDB in `all_design_models/`
inside the model archive. The five named ZAPP designs carry their manuscript
names as filename suffixes.

Archive scaffold numbers are global: campaign 1 uses 1–6 and campaign 2 uses
7–17. The campaign-2 screen uses its original 1–11 numbering; add 6 to match the
archive. Models retain catalytic-motif records, ligands and metal-coordination
annotations.

To order a design, use `eblock_order_dna_sequence`: lowercase BsaI adapters
surround the uppercase insert. The insert translates to
`designed_protein_sequence`; the expression vector supplies the terminal
sequences, giving `MSG` + design + `GSAWSHPQFEK` for the tagged protein.
Campaign-2 adapters are supplied for ordering; those designs were originally
ordered as insert-only IPDblocks.

The `cloning_and_vector` sheet contains the complete vector sequence and
cloning details for pGG-T7-cStrepII ([Addgene 262810](https://www.addgene.org/262810/)).
The `analysis_notes` sheet describes the additional tagless comparison and
relevant figure/reporting qualifications.
