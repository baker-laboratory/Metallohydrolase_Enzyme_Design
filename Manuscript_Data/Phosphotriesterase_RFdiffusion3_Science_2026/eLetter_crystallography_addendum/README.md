# Crystal structure of the first de novo designed phosphotriesterase ZAPP-1

This directory contains the ZAPP-1 crystallography report and the structural
data used for its comparisons with the RFdiffusion3 design and AlphaFold3.

- [Read the report](report/ZAPP1_crystallography.pdf).
- [Editable LaTeX source](report/ZAPP1_crystallography_source.zip) (Overleaf;
  set `main.tex` as the main file and compile with pdfLaTeX and BibTeX).
- `structures/design/`: RFdiffusion3 design model.
- `structures/crystals/`: coordinate and reflection files used for the analyses.
  Form 1 is the pNPP-supplemented crystal (39CY, 1.42 Å); form 2 was prepared
  without added pNPP (39CX, 2.28 Å). The supplied analysis files have not been
  independently matched to the deposited coordinate and reflection files.
- `predictions/`: the substrate-bound AF3 model and five substrate-free
  predictions with water, their available input, and confidence outputs.
  The five water models are samples from one run; the original input and
  confidence files for the bound model were not supplied.
- `analysis/`: selected design and crystal comparison metrics and a script for
  the pooled directional shifts in Table 1. The corresponding pre-aligned
  coordinates are in `structures/aligned/`.

## Analysis notes

`comparison_to_design.csv` fits each model to the design; `comparison_to_crystal.csv`
fits each model independently to each crystal. `pymol_align_ca_rmsd_A` and
`CA_trimmed_RMSD_A` use outlier rejection, whereas `ca_rmsd_A`, `CA_full_RMSD_A`,
and the sidechain/metal RMSDs use all shared Cα atoms without local refitting.
Sidechain RMSDs use the 37 catalytic heavy atoms. Internal identifiers
`el_xtal1_A/B` and `el_xtal2_A` refer to form 1 chains A/B and form 2 chain A;
`el_af3_holo` refers to the substrate-bound model and `el_af3_znwat_0` through
`_4` to the five water-containing substrate-free samples. Sample 0 is shown in
the figures. The form 2 metal-pair RMSD includes its weak second modeled site;
Table 1 reports it separately in the note rather than as a well-ordered pair.

To reproduce the pooled design-to-crystal projections from the supplied aligned
models, run `python analysis/reproduce_directional_shifts.py`. The calculation
does not rerun AF3 or perform structure refinement. The K16 density analysis
described in the report used a fixed-model scan without re-refinement; solvent
alone can also explain the density. The original scan scripts and complete
scan outputs were not supplied, so full reproduction of that scan is not
claimed here.
