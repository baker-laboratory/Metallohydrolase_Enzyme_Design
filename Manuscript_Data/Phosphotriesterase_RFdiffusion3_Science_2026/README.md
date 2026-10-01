# Phosphotriesterase — RFdiffusion3 Science 2026

Experimental data for de novo binuclear Zn(II) phosphotriesterases designed with
RFdiffusion3 and assayed on paraoxon. The corresponding computational campaign
is in [Design_Pipelines/Phosphotriesterase_RFdiffusion3](../../Design_Pipelines/Phosphotriesterase_RFdiffusion3/).

## Data

| Location | Contents |
|---|---|
| [wetlab_data_analysis.py](wetlab_data_analysis.py) | Reactive marimo analysis: raw-data calibration, background rates, kinetics, complete screens and figure export |
| `wetlab_data_analysis.ipynb` | Historical Jupyter workflow |
| `raw_wetlab_data/` | 15 files: calibration and background plates, nine kinetics plates, two screens, an order FASTA and design-assignment table |
| `supplemental_data/supp_data__denovo_PTE_design_models.zip` | 288 design models: 96 from campaign 1 and 192 from campaign 2 |
| `supplemental_data/supp_data__denovo_PTE_DNA_and_protein_sequences.xlsx` | Sequences, model identifiers, cloning information, 18 paper kinetics entries and an additional tagless ZAPP-1 comparison |
| `supplemental_data/paraoxon_kinetics_ALL_DATA.xlsx` | Reference kinetic parameters, uncertainty budget and reporting notes used by the figure routines |
| `wetlab_data_plots/` | Analysis tables and PNG figures; `paper_figures/` also contains vector PDFs |
| [Tagless ZAPP-1 SI data](supplemental_data/ZAPP1_tagless/) | Source tables, protein sequences and methods for the final tagless SI figure |
| [Mass spectrometry](mass_spectrometry/) | Tagless ZAPP-1 intact-mass image and paired overview/zoom images for the TEO samples |
| [Analysis notes](REPRODUCTION.md) | Fitting conventions, figure outputs and interpretation notes |

The two eluate screens cover 96 campaign-1 designs and 192 campaign-2 designs.
Kinetics use six paraoxon concentrations, three enzyme replicates and a
background measurement per concentration. Instrument measurements are unchanged;
local file metadata has been removed from the workbooks. The tagless ZAPP-1 measurements
share a plate with ZAPP-3 and ZAPP-4.

## Tagless ZAPP-1 and mass spectrometry

The final tagless SI figure is available as
[PNG](wetlab_data_plots/paper_figures/png/PTE__v2SI__ZAPP1_tagless.png),
[PDF](wetlab_data_plots/paper_figures/pdf/PTE__v2SI__ZAPP1_tagless.pdf) and
[TeX caption](wetlab_data_plots/paper_figures/pdf/PTE__v2SI__ZAPP1_tagless.tex).
It combines tagless kinetics, the tagged/tagless fit overlay and the
intact-mass spectrum. Its [source data](supplemental_data/ZAPP1_tagless/)
include the protein sequences and full-precision plotting tables.

The [mass-spectrometry deposit](mass_spectrometry/) retains the original
instrument-exported PNGs: one tagless spectrum and overview/zoom pairs for the
control and four TEO-treated samples. The image manifest records original
filenames and checksums.

## Run the analysis

From this directory:

```bash
conda env create -f ../../Environment/analysis.yml
conda activate zinc_hydro_analysis
marimo edit wetlab_data_analysis.py
```

The notebook runs its dependency graph automatically, using the deposited raw
files. It refits all 19 enzyme/condition entries, including tagless ZAPP-1,
and performs 234 checks against the paper reference workbook. Select an assay
or scaffold to inspect its curves. The export button regenerates the tables
and manuscript figures. No GPU or external laboratory filesystem is required.
Arial reproduces the figure typography; an installed fallback font is used
when Arial is unavailable.

The app calls ordinary Python analysis functions in
`Scripts/wetlab_platereader/analysis_pte_rfd3.py`, the RFdiffusion3 drawing
helpers in `kinetics_pte_rfd3.py`, and the shared numerical fitter. It does not
read or execute the Jupyter notebook or depend on saved notebook output.
Screening PNGs are 600 dpi; kinetics PNGs are 150 dpi. Historical manuscript
panels also produce PDF.

The current [round-1](wetlab_data_plots/screen_round1_SI_style.png) and
[round-2](wetlab_data_plots/screen_round2_SI_style.png) screening views retain
the SI typography and display **every design as an individual scaffold-colored
curve**. Kinetics-tested designs are thicker and directly labeled. Time is in
hours and ΔA₄₀₅ is referenced to each well's first acquisition. Negative-going
traces remain visible; no percentile band or hit filter replaces the data.
Historical manuscript artwork and the original Jupyter screening views remain
available alongside these complete views.

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
