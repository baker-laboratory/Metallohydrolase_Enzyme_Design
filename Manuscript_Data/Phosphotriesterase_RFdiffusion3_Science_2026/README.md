# Phosphotriesterase — RFdiffusion3 Science 2026

> **RFdiffusion3 Science 2026 manuscript — data and figure reproduction available.**
> The directory uses the author-supplied manuscript designation; it does not assert a final publication date or DOI.

Experimental data for de novo binuclear Zn(II) phosphotriesterases designed with
**RFdiffusion3**, assayed on paraoxon.

The matching computational campaign is
[`../../Design_Pipelines/Phosphotriesterase_RFdiffusion3/`](../../Design_Pipelines/Phosphotriesterase_RFdiffusion3/).

## Contents

| Path | What it holds |
|---|---|
| `wetlab_data_analysis.ipynb` | Loads every raw file, fits the kinetics, produces the reported figures and tables |
| `raw_wetlab_data/` | 15 primary files, unmodified instrument output |
| `supplemental_data/` | Deposited supplementary records, plus the unchanged paper-reference kinetics workbook |
| `REPRODUCTION.md` | Numerical and visual validation, provenance, and inherited source caveats |
| `wetlab_data_plots/` | Per-design figures, screening tables and numerical checks; `paper_figures/` contains the reconstructed SI/main panels as PDF and 150-dpi PNG |

### Raw data (`raw_wetlab_data/`)

| Measurement | Contents | Format |
|---|---|---|
| 4-nitrophenol standard curves | Calibration ladders in four buffer conditions, used to convert A<sub>405</sub> to product | `.xlsx` |
| Uncatalyzed rate | Enzyme-free paraoxon hydrolysis, 0.3–9.6 mM, ± 25 mM NaHCO<sub>3</sub> | `.xlsx` |
| Michaelis–Menten kinetics | 9 plates covering Round 1 designs, Round 2 designs and the ZAPP-1 mutants | `.xlsx` |
| Round 1 eluate screen | 96-well progress curves, 300 µM paraoxon | `.xlsx` |
| Round 2 eluate screen | 384-well progress curves, 192 designs, 600 µM paraoxon | `.csv` |
| Round 2 design table | Order FASTA (design names, scaffolds, plate positions) and the sequenced-design list used to validate the well mapping | `.fasta`, `.csv` |

### Supplementary data (`supplemental_data/`)

| File | Contents |
|---|---|
| `supp_data__denovo_PTE_design_models.zip` | Design models for all 288 ordered designs — 96 from campaign 1, 192 from campaign 2 |
| `supp_data__denovo_PTE_DNA_and_protein_sequences.xlsx` | DNA and protein sequences for every design, the ZAPP-1 mutants, 18 paper kinetics entries plus the additional tagless ZAPP-1 comparison, and the cloning scheme |
| `paraoxon_kinetics_ALL_DATA.xlsx` | Unmodified reference workbook supplied with the paper notebook: reported values, uncertainty budget and source caveats |
| `reproduction_sources.json` | SHA-256 provenance for the source workbook/notebook, raw kinetics plates, figure scripts and reference PDFs |

#### Design models (`supp_data__denovo_PTE_design_models.zip`)

Models are named
`denovo_PTE__design_campaign<N>__<index>_scaffold_<SS>____<order plate position>.pdb`,
inside `all_design_models/`. The five designs characterized kinetically carry
their manuscript name as a suffix, for example
`denovo_PTE__design_campaign1__037_scaffold_03____Plate1__D1__ZAPP1.pdb`.

**Scaffold numbering is global across campaigns.** The two campaigns used
different scaffolds but each numbered its own from 1, so campaign 2 is offset
past campaign 1's six: campaign 1 keeps `scaffold_01`–`scaffold_06`, and
campaign 2's scaffolds 1–11 become `scaffold_07`–`scaffold_17`. A scaffold
number therefore identifies one scaffold outright. Note that the Round 2 screen
in the analysis notebook reports the campaign's own 1–11 numbering, since it
reads those straight from the order FASTA — subtract 6 from an archive scaffold
number to get the notebook's.

Each file keeps its coordinates, the `REMARK 666` catalytic-motif block (and
`REMARK 665/667/668`, which document the motif and catalytic protonation states
where present), `HETNAM`, and the `LINK` records describing metal coordination.
Design-process artifacts are removed: `REMARK PDBinfo-LABEL` lines, `CONECT`
records, Rosetta pose-energy tables, and placeholder `HEADER` records. The
coordinates and `REMARK 666` blocks are byte-identical to the originals.

#### Sequences and kinetics (`supp_data__denovo_PTE_DNA_and_protein_sequences.xlsx`)

| Sheet | Contents |
|---|---|
| `overview` | What each sheet holds and how every column is defined |
| `all_designs` | All 288 designs from both campaigns: orderable eBlock fragment, insert, full ORF, designed protein (read from the deposited model), expressed protein, lengths, molecular weight and extinction coefficient |
| `ZAPP1_mutants` | The six single active-site substitutions of ZAPP-1, with the same protein-level columns |
| `kinetics`, `kuncat` | Parameters for the 18 original paper entries and a separately labeled tagless comparison, plus the condition-matched uncatalyzed rates |
| `reproduction_notes` | Tagless assay provenance, uncertainty model, detector caveat and source-reporting notes |
| `cloning_and_vector` | Golden Gate scheme, expression hosts and tags, then the full pGG-T7-cStrepII sequence, then its annotated features |

Designs are identified as `R<campaign> p<plate><well>` — `R1 p1D1` is campaign 1,
plate 1, well D1 — which is also how they are named in the kinetics sheet.

**To order any design**, take `eblock_order_dna_sequence` and order it as a
linear dsDNA fragment. It has the same form for every design: the BsaI adapters
in lowercase around the insert in uppercase, with exactly one BsaI site at each
end and none inside. That is precisely how campaign 1 was ordered; campaign 2
was ordered as IPDblocks (insert only, adapters added during synthesis), so its
adapters are supplied here for ordering rather than as-shipped.

The uppercase insert translates, in frame, to `designed_protein_sequence`
exactly — no start codon, no stop, no tag. The vector supplies both tags, so the
expressed protein is `MSG` + design + `GSAWSHPQFEK` (GSA linker + Strep-tag II),
and `orf_dna_sequence` is the complete reading frame as cloned.

Expression used BL21(DE3) *E. coli*: NEB C2527 for campaign 1, and Intact
Genomics chemically competent cells for campaign 2, which gave higher
transformation efficiency in this protocol.

The vector is **pGG-T7-cStrepII** (kanamycin, T7/lac, ccdB counter-selection),
designed by [Lucas Milles](https://www.biochem.mpg.de/milles) and deposited as
[Addgene 262810](https://www.addgene.org/262810/). It has also been referred to
as LM1369, pDT1 and pDTstrep1. Propagating the empty vector needs a
ccdB-resistant (*gyrA462*) strain; its full sequence is in the workbook.

Every row of `all_designs` carries a `model_file` column naming the design model
it corresponds to, so the two files cross-reference exactly: 288 rows, 288
models, one to one.

The sequence workbook is a reference deposit. The notebook reads the separate, unchanged `paraoxon_kinetics_ALL_DATA.xlsx` to verify refits and reproduce the paper annotations. Existing sequence-workbook rows are preserved; the tagless comparison is appended and is not counted as a new design.

## Notebook layout

| Section | Contents |
|---|---|
| **I. Calibration & background rate** | 4-nitrophenol standard curves in each buffer; uncatalyzed paraoxon hydrolysis (*k*<sub>uncat</sub>) with and without bicarbonate |
| **II. Michaelis–Menten kinetics** | One shared fitting routine, then Round 1 designs, Round 2 designs, and the ZAPP-1 active-site mutants — 18 paper fits, followed by the additional tagless comparison |
| **III. Summary** | All fits with condition-matched *k*<sub>uncat</sub> and the derived ratios |
| **IV. Eluate screening** | Round 1 96-well progress curves; Round 2 384-well screen with well mapping, mapping validation, initial rates, QC and hit calling |
| **V. Paper figures** | Four kinetics/background SI figures, Figure 4 panels E/F/G/I, and the separate tagless comparison, using the manuscript colors, geometry and labels |

Designs are named as in the manuscript with the plate identifier alongside —
`ZAPP-1 (p1D1)` — so a figure can be traced back to the raw plate-reader file
it came from.

Section II collects every fit into a `FITS` dictionary and section III builds
the summary table from it, so the table always reflects the cells actually run
rather than transcribed values. Run section II in full before section III.

## Running it

```bash
conda env create -f ../../Environment/analysis.yml
conda activate zinc_hydro_analysis
jupyter lab
```

The notebook locates the repository itself and reads raw assays from
`raw_wetlab_data/` and the reference workbook from `supplemental_data/`, so there is nothing to edit before running it. It needs
numpy, pandas, scipy, matplotlib and openpyxl — no external tools, no GPU.

Plate-reader parsing and fitting live in
[`../../Scripts/wetlab_platereader/`](../../Scripts/wetlab_platereader/)
(`kinetics.py`, `neo2_util.py`, `screening_curves.py`), vendored so the
notebook is self-contained. `kinetics_pte_rfd3.run_kinetics_PTE_RFd3` reuses the
same numerical fitter with the RFdiffusion3 presentation in `rfd3_figures/`;
the RFdiffusion2 plotting code is unchanged.

### Figure output

Figures are written to `wetlab_data_plots/` as PNG at 150 dpi — those are the
files committed here. Vector EPS is off by default because the files are large
and regenerable; set `SAVE_EPS = True` in the initialization cell, or pass
`save_eps=True` at a single call site, when you need one. Paper panels in section V always include PDF and 150-dpi PNG.
The supplied main panel E contains a calibration mismatch in its no-bicarbonate
points; both an exact historical reproduction and a separately named
`__condition_matched` correction are generated. See [REPRODUCTION.md](REPRODUCTION.md).

## Related

- Design campaign: [`../../Design_Pipelines/Phosphotriesterase_RFdiffusion3/`](../../Design_Pipelines/Phosphotriesterase_RFdiffusion3/)
- Method tutorial: [`../../RFdiffusion3_Tutorial/`](../../RFdiffusion3_Tutorial/)
- Published dataset in its finished form: [`../Metallohydrolase_Nature_2026/`](../Metallohydrolase_Nature_2026/)
