# Metallohydrolases — RFdiffusion2, Nature Methods (2026)

Data for the additional **4MU-butyrate (4MU-B)** metallohydrolase designs in
Ahern, Yim, Tischer *et al.*, **Atom-level enzyme active site scaffolding using
RFdiffusion2**, *Nature Methods* **23**, 96–105 (2026),
[doi:10.1038/s41592-025-02975-x](https://www.nature.com/articles/s41592-025-02975-x).
The issue year is 2026; the article first appeared online on 3 December 2025.

That paper also presents the 4MU-phenylacetate metallohydrolase results described
in the accompanying Nature paper. Those shared experiments, sequences and
structures remain in [Metallohydrolase_Nature_2026](../Metallohydrolase_Nature_2026/).
The existing Nature notebook contains phenylacetate experiments; no butyrate
analysis cells needed to be moved out of it.

## Reproduce the experimental results

Open [wetlab_data_analysis.ipynb](wetlab_data_analysis.ipynb) and run all cells
with Python, NumPy, pandas, SciPy, matplotlib and openpyxl. Paths are resolved
from the cloned repository. PyRosetta and a GPU are not needed for wetlab analysis.

The notebook includes:

- **Section I:** the published 2024-07-10 screen, an inspection grid containing
  all 96 plate positions, and three archived 2024-07-16 screening conditions.
  The publication plot preserves the original H8/H9 exclusions and identifies
  C4, B11 and D11 as the three strongest progress curves.
- **Section II:** calibration derived from the included 2025-04-01 plate,
  progress curves, Michaelis–Menten fits for C4 and B11, and the low-substrate
  estimate for D11.

The reproduced catalytic efficiencies are:

| Design | kcat/KM (M⁻¹ s⁻¹) | Analysis |
|---|---:|---|
| C4 | 77.3802 ± 10.7543 | Michaelis–Menten |
| B11 | 12.9273 ± 1.1446 | Michaelis–Menten |
| D11 | 8.9156 ± 0.1166 | Linear low-substrate limit |

C4/B11 uncertainties follow the original notebook's propagated fit errors.
D11's uncertainty is the standard error of the zero-intercept linear fit added
here; its substrate range does not resolve separate kcat and KM values.
All 24 mean initial rates (eight substrate concentrations for each hit) are
checked against the final author notebook during execution. The 2025-04-01
calibration reproduces the published C4 value of approximately 77 M⁻¹ s⁻¹.
The superseded 2024-09-20 calibration would instead give approximately 100.

Raw inputs are unchanged copies in [raw_wetlab_data](raw_wetlab_data/).
[raw_data_manifest.csv](supplemental_data/raw_data_manifest.csv) records the
source, purpose and SHA-256 checksum of all 11 files. The exploratory C10/E10
kinetics and earlier standard curve are retained, but are not treated as
publication fits. Screening fluorescence is not normalized by enzyme
concentration and is used to find hits, not to assign catalytic efficiencies.
File names retain the original assay-preparation labels; consult the paper's
methods for final reaction concentrations.

## Sequences and transition-state structures

| File | Contents |
|---|---|
| [Data_S1__4MU_B_Ordered_Sequences.xlsx](supplemental_data/Data_S1__4MU_B_Ordered_Sequences.xlsx) | All 96 final ordered protein/DNA sequences, expression constructs, model identities and reproduced kinetics |
| [ordered_sequences.csv](supplemental_data/ordered_sequences.csv) / [FASTA](supplemental_data/ordered_sequences.fasta) | Compact protein sequences and model mapping |
| [Data_S2__4MU_B_Ordered_Models_with_Transition_States.zip](supplemental_data/Data_S2__4MU_B_Ordered_Models_with_Transition_States.zip) | One sequence-matched TS-bearing PDB for each of the 96 ordered designs |
| [model_manifest.csv](supplemental_data/model_manifest.csv) | Per-model source/stage, sequence, ligand, alignment, atom inventory, Zn coordination distances and checksum |
| [structure_preparation](structure_preparation/) | Portable preparation scripts, archived reconstruction inputs, ligand parameters and constraints |

The three publication hit models **B11, C4 and D11** are preserved byte-for-byte
from `paper_hits/for_rfd2_manuscript/design_models`. Their protein sequences match
the final order. The other 93 models are explicitly marked **reconstructions**:
they combine the final ordered apo protein sequence with its original designed
transition state and undergo the same three-pass constrained relaxation used to
prepare the publication hits. They are additional supplemental models, not
historical coordinates claimed to have been used in the paper.

The original `ALL_ORDERED_PDBS` directory contains 96 TS-bearing structures,
but 21 retain pre-order sequences, and many are ligand-transferred apo models
that had not undergone constrained holo relaxation. Copying that directory
unchanged would therefore misrepresent both sequence identity and model stage.
The reconstructed archive addresses these issues while keeping the three
historical hit structures intact. These are computational design models, not
experimentally determined crystal structures.

[validation_summary.json](supplemental_data/validation_summary.json) records the
completed checks. All 96 models match their ordered sequences and contain the
complete TS residue and Zn. No protein–ligand heavy-atom contact is below 1.5 Å
(minimum 1.9206 Å). The three original hit models have Zn–histidine distances
of 2.0092–2.1111 Å. Seven reconstructed non-hits retain at least one distance
above 3 Å: B5, B9, B10, C3, D5, F9 and H3 (maximum 4.6266 Å, F9). These
nonideal coordination geometries are flagged in the manifest; reconstruction
and relaxation do not establish catalytic activity.

## Provenance

The final experimental settings and reference outputs come from
`/home/donghyo/projects/zinc_hydrolase/enzyme_design/20240405_design_campaign1/wetlab_analysis_for_method_paper.ipynb`.
The older exploratory butyrate notebook in
`old_4muButyrate_stuff/20240402_design_campaign1/wetlab_data_order1_pub/`
was cross-checked but its superseded calibration is not used for publication fits.

The sequence source is the original `_df.csv` order table in
`old_4muButyrate_stuff/20240402_design_campaign1/Order1_SETH_68bbs_96seq_240523/john_bercow_order/`.
Protein sequences from all 96 ordered apo PDBs agree with that table.
[source_provenance.json](supplemental_data/source_provenance.json) records the
checksums of the source order table and original analysis/design notebooks.
[Structure preparation provenance](structure_preparation/README.md) traces the
original ligand-transfer and constrained-relaxation workflow and explains how
to inspect or rerun the supplementary reconstructions.
