# Manuscript Data

Experimental measurements and supplementary data, organized by campaign. Each
subdirectory pairs with the design campaign of the same name under
[`Design_Pipelines/`](../Design_Pipelines/).

| Dataset | Campaign | Status |
|---|---|---|
| [`Metallohydrolase_Nature_2025/`](Metallohydrolase_Nature_2025/) | [`Metalloesterase_RFdiffusion2`](../Design_Pipelines/Metalloesterase_RFdiffusion2/) | **Published** — Kim, Woodbury, Ahern et al., *Nature* (2025) |
| [`Phosphotriesterase_RFdiffusion3/`](Phosphotriesterase_RFdiffusion3/) | [`Phosphotriesterase_RFdiffusion3`](../Design_Pipelines/Phosphotriesterase_RFdiffusion3/) | In preparation |
| [`Metalloprotease_RFdiffusion3/`](Metalloprotease_RFdiffusion3/) | [`Metalloprotease_RFdiffusion3`](../Design_Pipelines/Metalloprotease_RFdiffusion3/) | In preparation |

Each dataset directory follows the same shape:

```
<Campaign>/
├── raw_wetlab_data/          # primary instrument output, unmodified
├── supplemental_data/        # sequences, models, theozymes as deposited
├── wetlab_data_analysis.ipynb  # loads raw data -> figures + reported statistics
└── wetlab_data_plots/        # every figure the notebook writes
```

---

## Running the analysis

You do **not** need the design environment. The analysis notebooks use only
numpy, pandas, scipy, matplotlib, and openpyxl.

```bash
# from the repository root
conda env create -f Environment/analysis.yml
conda activate zinc_hydro_analysis
python -m ipykernel install --user --name=zinc_hydro_analysis

jupyter lab Manuscript_Data/Metallohydrolase_Nature_2025/wetlab_data_analysis.ipynb
```

Select the **zinc_hydro_analysis** kernel, then **run the first cell before
anything else** and proceed top to bottom.

There are no paths to edit. The first cell locates the repository by walking up
from the working directory, creates `wetlab_data_plots/`, and exports the
`*_dir` variables the rest of the notebook uses. Set `ZINC_HYDRO_REPO` only if
you are running from somewhere unusual.

`openpyxl` is required — 30 of the 43 raw files are `.xlsx`. It is included in
both `analysis.yml` and `zinc_hydro.yml`.

### Where the figures go

**Every figure is written to `wetlab_data_plots/`**, as a PNG. Two knobs at the
top of the notebook (cell 1) control this:

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

Vector exports (`.eps`, `.svg`, `.pdf`) are gitignored — regenerate them when
you need them.

### Reading it without running it

The notebook ships with its cell outputs intact, so every figure and fitted
value is visible on GitHub without running anything.

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

- **Run the notebook top to bottom.** Several sections build on variables
  defined earlier, so running cells out of order can produce plots from a
  different experiment's wells.

- **`supplemental_data/` is reference material** — the deposited sequences,
  models and DFT theozymes. The analysis notebook does not read it.

---

## Adding a new campaign

Create `Manuscript_Data/<Chemistry>_<Model>/` with the four-part layout above,
add a row to the table at the top of this file, and create the matching
`Design_Pipelines/<Chemistry>_<Model>/`. The analysis notebook's first cell is
campaign-agnostic apart from its `working_dir` line — copy it and change that
one path.
