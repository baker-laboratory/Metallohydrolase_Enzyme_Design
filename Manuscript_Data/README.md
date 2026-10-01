# Manuscript Data

Experimental measurements and supplementary data, organized by publication.
The table links each publication to the corresponding computational campaign
under [`Design_Pipelines/`](../Design_Pipelines/).

| Dataset | Campaign | Status |
|---|---|---|
| [`Metallohydrolase_Nature_2026/`](Metallohydrolase_Nature_2026/) | [`Metalloesterase_RFdiffusion2`](../Design_Pipelines/Metalloesterase_RFdiffusion2/) | **Published** — Kim, Woodbury, Ahern et al., *Nature* (2026) |
| [`Metallohydrolase_RFdiffusion2_Nature_Methods_2026/`](Metallohydrolase_RFdiffusion2_Nature_Methods_2026/) | [`Metalloesterase_RFdiffusion2`](../Design_Pipelines/Metalloesterase_RFdiffusion2/) | **Published** — additional 4MU-butyrate data and shared Nature results |
| [`Phosphotriesterase_RFdiffusion3_Science_2026/`](Phosphotriesterase_RFdiffusion3_Science_2026/) | [`Phosphotriesterase_RFdiffusion3`](../Design_Pipelines/Phosphotriesterase_RFdiffusion3/) | RFdiffusion3 Science 2026 manuscript; data and figure reproduction available |
| [`Metalloprotease_RFdiffusion3_Science_2026/`](Metalloprotease_RFdiffusion3_Science_2026/) | [`Metalloprotease_RFdiffusion3`](../Design_Pipelines/Metalloprotease_RFdiffusion3/) | RFdiffusion3 Science 2026 manuscript; experimental dataset pending |

The Nature and Nature Methods directories use their 2026 journal issue year;
both articles were first published online on 3 December 2025. RFdiffusion3
directories use the manuscript designation supplied by the authors.

Each dataset directory follows the same shape:

```
<Campaign>/
├── raw_wetlab_data/          # primary instrument measurements
├── supplemental_data/        # sequences, models, and theozymes
├── wetlab_data_analysis.py  # loads raw data -> figures + reported statistics
└── wetlab_data_plots/        # every figure the notebook writes
```

---

## Running the analysis

You do **not** need the design environment. The analysis notebooks use only
marimo, numpy, pandas, scipy, matplotlib, and openpyxl.

```bash
# from the repository root
conda env create -f Environment/analysis.yml
conda activate zinc_hydro_analysis
marimo edit Manuscript_Data/Metallohydrolase_Nature_2026/wetlab_data_analysis.py
```

The marimo notebooks execute cells in dependency order. The Python source contains
the analysis; historical `.ipynb` files are retained as an archive and are not
required to run the marimo notebooks.

There are no paths to edit. The first cell locates the repository by walking up
from the notebook location, creates `wetlab_data_plots/`, and supplies the
paths used by the analysis cells. Set `ZINC_HYDRO_REPO` only if
you are running from somewhere unusual.

`openpyxl` is required — 30 of the 43 Nature raw files are `.xlsx`. It is included in
both `analysis.yml` and `zinc_hydro.yml`.

### Where the figures go

The Nature notebook writes its analysis figures to `wetlab_data_plots/` as
PNGs. Two knobs at the top of that notebook (cell 1) control this:

```python
FIGURE_DPI = 150     # raster resolution for the PNGs
SAVE_EPS   = False   # also write <name>.eps alongside <name>.png
```

PNG at 150 dpi is the default so the repository stays small — those PNGs are
committed, and are what renders on GitHub. EPS is vector and much larger, so it
is off by default; set `SAVE_EPS = True` for a full vector run, or pass
`save_eps=True` / `dpi=600` at a single call site for one figure. Every plotting
call takes `dpi=` and `save_eps=`, so a single panel can be exported at
publication resolution without changing the rest.

Optional vector exports (`.eps`, `.svg`, `.pdf`) directly in
`wetlab_data_plots/` are gitignored. The RFdiffusion3 reproduction PDFs in
`wetlab_data_plots/paper_figures/` are included in the repository.

The phosphotriesterase notebook uses its own RFdiffusion3 plotting functions to
reproduce the manuscript's SI and main-figure panels, including their colors,
labels, layouts, and PDF exports. See its dataset README for the numerical and
visual comparison results and the separately labeled tagless ZAPP1 analysis.

### Reading it without running it

Generated PNGs and result tables are available in `wetlab_data_plots/`. For a
local executed HTML copy, run `marimo export html <notebook.py> -o analysis.html`.
The archived Jupyter notebooks also retain their previous outputs.

---

## Notes on the analysis

- **Calibration.** Fluorescence is converted to product concentration using
  4-methylumbelliferone standard curves, derived in §II.I.A from the
  standard-curve plates in `raw_wetlab_data/` and defined there as named
  constants:

  | Constant | Source plate | Applied to |
  |---|---|---|
  | `SLOPE_4MU_250401` | `250401_4mu_standard_curves_..._pH8.xlsx` | the Michaelis–Menten series, and so the reported *k*<sub>cat</sub>, *K*<sub>M</sub> and *k*<sub>cat</sub>/*K*<sub>M</sub> |
  | `SLOPE_4MU_240927` | `240927_4mu_standard_curves_...LadderLEFT3_100uMLadderRIGHT3.xlsx` | §II.IV.C, A1 H130A mutant kinetics |
  | `SLOPE_4MU_240906` | `240906_a1_plus_6mutants_AND_standard_curve_...xlsx` | §I.IV, A1 knockout screen |

  Each plate carries two serial ladders fitted together as a single twelve-point
  curve, in triplicate. On the 250401 plate the ladders run across columns
  (30 µM in 1–6, 45 µM in 7–12, rows D/E/F); on the 240927 plate they run down
  the rows (80 µM in columns 1–3, 100 µM in columns 4–6), so that plate is
  transposed before fitting. §II.I.A does this explicitly.

  Instrument gain differs between sessions, so each calibration is applied only
  to the experiments recorded alongside it. To change a calibration, edit
  §II.I.A rather than the individual analysis cells.

- **Reactive execution.** Each marimo cell names its inputs explicitly; changing
  an upstream analysis setting reruns the dependent cells.

- **Nature `supplemental_data/` is reference material** — the deposited
  sequences, design models, DFT theozymes, and ZETA_2 crystal structures. The
  Nature analysis notebook does not read it. Other datasets may use their
  supplemental tables for numerical comparisons.

---

## Adding a new campaign

Create `Manuscript_Data/<Chemistry>_<Model>_<Journal>_<Year>/` with the four-part
layout above and add a row linking its computational campaign. Shared results
should link to their original dataset rather than duplicate raw files. The analysis notebook's first cell is
specific to its dataset. Copy a marimo entrypoint and update its dataset path
and analysis cells.
