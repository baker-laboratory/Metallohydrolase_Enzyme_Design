"""Michaelis-Menten kinetics analysis for Neo2 plate reader data.

Two entry points, meant to be used sequentially in a notebook:

1. **Standard curve** (``parse_standard_curve``) — fit signal vs. known
   product concentrations to get the calibration slope (signal per uM).
   Copy the printed slope into the kinetics cell as ``signal_per_uM``.

2. **Kinetics** (``parse_kinetics``) — parse a Neo2 time-course, fit an
   initial rate for every enzyme replicate and every background replicate
   independently, subtract the per-row mean background slope from each
   enzyme replicate, then fit Michaelis-Menten to the mean rates.

Layout convention (user-specified, no CSV needed): both parsers take
``blocks``, a list of dicts describing rectangular regions of the plate.
Each block has its own rows, concentrations, and column assignments, so
you can spread one dataset across arbitrary plate regions (e.g. 16
standard-curve points split across two column groups, or an enzyme run
whose replicates and bg live in different columns).

Standard curve block keys:
    - ``rows``               list of plate rows, one per known [product]
    - ``concentrations_uM``  same length as rows
    - ``replicate_cols``     columns holding standard-curve replicate wells

Kinetics block keys:
    - ``rows``               list of plate rows, one per substrate conc
    - ``concentrations_uM``  same length as rows; concentrations[i] is [S] in rows[i]
    - ``enzyme_cols``        columns holding enzyme replicate wells
    - ``bg_cols``            columns holding buffer-only bg replicate wells
                             (may be [] to skip bg correction in that block)

Reported parameters:
    kcat        = Vmax / [E]                 (1/s)
    Km          = MM half-saturation         (uM)
    kcat/Km     = specificity constant       (1/(uM.s))
    kuncat      = slope of v_bg vs [S]       (1/s), assumes bg is first-order in [S]
    kcat/kuncat = rate enhancement           (dimensionless)

Usage::

    from wetlab.kinetics import parse_standard_curve, parse_kinetics, plot_kinetics

    # Step 1 - one-time calibration (16 concentrations across two column groups)
    signal_per_uM, _ = parse_standard_curve(
        data_path="std_curve.txt",
        blocks=[
            dict(rows=list("ABCDEFGH"),
                 concentrations_uM=[100, 50, 25, 12.5, 6.25, 3.125, 1.5625, 0.78125],
                 replicate_cols=[7, 8, 9]),
            dict(rows=list("ABCDEFGH"),
                 concentrations_uM=[0.39, 0.195, 0.098, 0.049, 0.024, 0.012, 0.006, 0.003],
                 replicate_cols=[10, 11, 12]),
        ],
    )

    # Step 2 - kinetics run (simple case: one block, one enzyme)
    df, reps_df, mm = parse_kinetics(
        data_path="kinetics.txt",
        blocks=[
            dict(rows=list("ABCDEF"),
                 concentrations_uM=[500, 250, 125, 62.5, 31.25, 15.625],
                 enzyme_cols=[1, 2, 3],
                 bg_cols=[4]),
        ],
        enzyme_uM=0.5,
        signal_per_uM=signal_per_uM,   # pasted from step 1
        time_range_seconds=(60, 900),  # optional: linear window for slope
    )
    plot_kinetics(df, reps_df, mm, time_scale="minute")
"""

# Vendored into this repository so the analysis notebooks are self-contained.
# Original author: Donghyo Kim. Kept byte-identical to upstream apart from this
# header, so it can be re-synced by replacing the file.


import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import curve_fit

from neo2_util import parse_neo2_kinetics


# Color palette (Wood/Buse "good" + rfd3_spec colors, matches lab plotting style)
_RFD3_1 = (75 / 255, 95 / 255, 170 / 255)
_RFD3_2 = (77 / 255, 140 / 255, 173 / 255)
_RFD3_3 = (79 / 255, 185 / 255, 175 / 255)
_RFD3_4 = (227 / 255, 159 / 255, 179 / 255)
_RFD3_5 = (255 / 255, 172 / 255, 183 / 255)
_GOOD_PEACH = (249 / 255, 145 / 255, 120 / 255)
_GOOD_RED = (228 / 255, 74 / 255, 62 / 255)
_GOOD_PINK = (236 / 255, 114 / 255, 164 / 255)
_PALETTE = [_RFD3_1, _RFD3_2, _RFD3_3, _RFD3_4, _RFD3_5, _GOOD_PEACH, _GOOD_RED, _GOOD_PINK]

# Module-level cache of the most recently drawn figure so ``export_plot()``
# can save it even after Jupyter's inline backend has closed it (which makes
# ``plt.gcf()`` return a fresh blank canvas).
_LAST_FIG = None


def _style_axes(ax):
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_linewidth(2.3)
    ax.spines["left"].set_linewidth(2.3)
    ax.tick_params(width=2.3, labelsize=14)
    ax.tick_params(axis="both", which="major", length=6)


def _palette_for(n):
    if n <= len(_PALETTE):
        return _PALETTE[:n]
    return list(cm.get_cmap("viridis")(np.linspace(0.15, 0.9, n)))


def _pick_font_family():
    """Return the subset of preferred Arial-compatible fonts that are actually
    installed in matplotlib's font cache. May be empty.

    Empty list means "don't touch rcParams" — the caller should leave
    matplotlib's default in place rather than set a name that will trigger
    ``findfont: Font family 'X' not found`` on every draw.
    """
    from matplotlib import font_manager
    prefs = ["Arial", "Helvetica", "Liberation Sans"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    return [name for name in prefs if name in available]


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _wells_from_grid(rows, cols):
    """Cartesian product of rows x cols into Neo2-style well IDs (e.g. 'A1')."""
    return [f"{r}{c}" for r in rows for c in cols]


# ---------------------------------------------------------------------------
# Standard curve
# ---------------------------------------------------------------------------

def parse_standard_curve(
    data_path,
    blocks,
    blank_wells=None,
    force_zero_intercept=False,
    plot=True,
):
    """Fit signal vs. known product concentration to get a calibration slope.

    The Neo2 file is expected to be either an endpoint (single read) or a
    short kinetic (few reads). Reads for the same well are averaged.

    Supports arbitrary layouts by splitting the plate into ``blocks``. Each
    block is one rectangular region with its own rows, concentrations, and
    replicate columns. Example: 16 concentrations across two column groups::

        blocks = [
            dict(rows=list("ABCDEFGH"),
                 concentrations_uM=[c1, c2, c3, c4, c5, c6, c7, c8],
                 replicate_cols=[7, 8, 9]),
            dict(rows=list("ABCDEFGH"),
                 concentrations_uM=[c9, c10, c11, c12, c13, c14, c15, c16],
                 replicate_cols=[10, 11, 12]),
        ]

    Args:
        data_path:            Neo2 ``.txt`` file with the standard curve plate.
        blocks:               List of dicts, each with keys ``rows`` (list of
                              plate rows), ``concentrations_uM`` (same length
                              as rows, one [P] per row), and ``replicate_cols``
                              (list of column ints). Wells across blocks must
                              not overlap.
        blank_wells:          Explicit list of buffer-only well IDs (e.g.
                              ``["A1","B1","C1"]``) whose mean is subtracted
                              from every signal before fitting. ``None`` to
                              skip blanking.
        force_zero_intercept: If True, fit ``signal = slope * [P]`` (no
                              intercept). Otherwise fit an affine line.
        plot:                 If True, show the fit + points.

    Returns:
        ``(slope, info_df)`` where ``slope`` is signal-per-uM (float) and
        ``info_df`` is a per-replicate DataFrame of (Well, block,
        concentration_uM, signal_raw, signal_blanked).
    """
    if not blocks:
        raise ValueError("blocks must be a non-empty list")
    for i, b in enumerate(blocks):
        missing = {"rows", "concentrations_uM", "replicate_cols"} - set(b)
        if missing:
            raise ValueError(f"blocks[{i}] missing keys: {sorted(missing)}")
        if len(b["rows"]) != len(b["concentrations_uM"]):
            raise ValueError(
                f"blocks[{i}]: rows ({len(b['rows'])}) and concentrations_uM "
                f"({len(b['concentrations_uM'])}) must have the same length"
            )
        if not b["replicate_cols"]:
            raise ValueError(f"blocks[{i}]: replicate_cols must be non-empty")

    df = parse_neo2_kinetics(data_path)
    print(
        f"Parsing [{data_path}] Measured at "
        f"[{datetime.datetime.fromtimestamp(df['datetime'].to_list()[0])}]..."
    )

    # Average reads per well (endpoint or short kinetic both collapse here)
    well_mean = df.groupby("Well", as_index=False)["value"].mean()

    # Blank subtraction: mean of explicitly listed blank wells
    blank_mean = 0.0
    if blank_wells:
        blank_signals = well_mean[well_mean["Well"].isin(blank_wells)]["value"]
        if blank_signals.empty:
            raise ValueError(f"No blank wells found in data for {blank_wells}")
        blank_mean = float(blank_signals.mean())

    # Build (Well, block, [P], signal_blanked) per replicate well
    seen = set()
    records = []
    for bi, b in enumerate(blocks):
        for row, conc in zip(b["rows"], b["concentrations_uM"]):
            for col in b["replicate_cols"]:
                well = f"{row}{col}"
                if well in seen:
                    raise ValueError(f"Well {well} appears in multiple blocks")
                seen.add(well)
                m = well_mean[well_mean["Well"] == well]
                if m.empty:
                    continue
                records.append({
                    "Well": well,
                    "block": bi,
                    "concentration_uM": conc,
                    "signal_raw": float(m["value"].values[0]),
                    "signal_blanked": float(m["value"].values[0]) - blank_mean,
                })
    info_df = pd.DataFrame(records)
    if info_df.empty:
        raise ValueError("No matching wells found for standard curve.")

    x = info_df["concentration_uM"].to_numpy(dtype=float)
    y = info_df["signal_blanked"].to_numpy(dtype=float)

    if force_zero_intercept:
        slope = float((x * y).sum() / (x * x).sum())
        intercept = 0.0
    else:
        slope, intercept = np.polyfit(x, y, 1)
        slope, intercept = float(slope), float(intercept)

    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    print(f"\nStandard curve fit:")
    print(f"  slope     = {slope:.6g}  (signal per uM)")
    print(f"  intercept = {intercept:.6g}")
    print(f"  R^2       = {r2:.4f}")
    print(f"  blank     = {blank_mean:.6g}  (subtracted before fit)")
    print(f"\nCopy into kinetics cell:  signal_per_uM = {slope:.6g}\n")

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        block_colors = ["steelblue", "darkorange", "seagreen", "purple"]
        for bi, sub in info_df.groupby("block"):
            color = block_colors[bi % len(block_colors)]
            label = f"Block {bi}" if info_df["block"].nunique() > 1 else "Replicates"
            ax.scatter(sub["concentration_uM"], sub["signal_blanked"],
                       color=color, alpha=0.7, label=label)
        xfit = np.linspace(0, max(x) * 1.05, 100)
        ax.plot(xfit, slope * xfit + intercept, color="crimson",
                label=f"Fit (R^2={r2:.3f})")
        ax.set_xlabel("[Product] (uM)")
        ax.set_ylabel("Signal (blanked)")
        ax.set_title(f"Standard curve: slope = {slope:.4g} signal/uM")
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.show()

    return slope, info_df


# ---------------------------------------------------------------------------
# Kinetics parsing
# ---------------------------------------------------------------------------

def _fit_slope(times, signals):
    """Linear fit; returns slope (signal per second). NaN if <2 points."""
    mask = np.isfinite(times) & np.isfinite(signals)
    if mask.sum() < 2:
        return float("nan")
    return float(np.polyfit(times[mask], signals[mask], 1)[0])


def _michaelis_menten(S, Vmax, Km):
    return Vmax * S / (Km + S)


def parse_kinetics(
    data_path,
    blocks,
    enzyme_uM,
    signal_per_uM,
    time_range_seconds=None,
    baseline_subtract=True,
):
    """Parse a Neo2 time-course and compute per-replicate MM kinetics.

    Supports arbitrary layouts by splitting the plate into ``blocks``. Each
    block is one rectangular region with its own rows, substrate concentrations,
    enzyme replicate columns, and background replicate columns. Background
    correction is done independently within each block (per-row mean of that
    block's bg wells). All enzyme replicates across all blocks are pooled by
    ``concentration_uM`` for the MM fit.

    Simple case (one block)::

        blocks=[dict(rows=list("ABCDEF"),
                     concentrations_uM=[500, 250, 125, 62.5, 31.25, 15.625],
                     enzyme_cols=[1, 2, 3],
                     bg_cols=[4])]

    Two-block case (concentrations split across column groups, or 2nd enzyme
    replicate on a different plate region)::

        blocks=[
            dict(rows=list("ABCDEFGH"),
                 concentrations_uM=[c1, c2, c3, c4, c5, c6, c7, c8],
                 enzyme_cols=[1, 2, 3], bg_cols=[4]),
            dict(rows=list("ABCDEFGH"),
                 concentrations_uM=[c9, c10, c11, c12, c13, c14, c15, c16],
                 enzyme_cols=[5, 6, 7], bg_cols=[8]),
        ]

    Args:
        data_path:          Neo2 ``.txt`` time-course file.
        blocks:             List of dicts, each with keys ``rows``,
                            ``concentrations_uM`` (same length as rows),
                            ``enzyme_cols``, ``bg_cols`` (may be empty list to
                            skip bg correction for that block). Wells across
                            blocks must not overlap.
        enzyme_uM:          Enzyme concentration (uM). Used for kcat = Vmax / [E].
        signal_per_uM:      Calibration slope from ``parse_standard_curve``.
                            Converts signal/s to uM/s.
        time_range_seconds: ``(t_min, t_max)`` window for the linear fit.
                            ``None`` uses all points. Strongly recommended for
                            real MM data to pick the initial-rate region.
        baseline_subtract:  Subtract the t=0 value per well before fitting the
                            slope (does not change the slope, but makes the
                            progress-curve plot start at 0).

    Returns:
        ``(df, reps_df, mm)`` where:
          - ``df`` is a long-format DataFrame of the time-course, with columns
            ``Well, time, signal, signal_baselined, block, Row, Col, role,
            concentration_uM``.
          - ``reps_df`` has one row per enzyme replicate, with columns
            ``block, Row, Col, concentration_uM, enzyme_slope_signal,
            bg_slope_signal_mean, corrected_slope_signal,
            corrected_rate_uM_per_s``.
          - ``mm`` is a dict with fitted kinetic parameters (see module docstring).
    """
    if not blocks:
        raise ValueError("blocks must be a non-empty list")
    for i, b in enumerate(blocks):
        missing = {"rows", "concentrations_uM", "enzyme_cols", "bg_cols"} - set(b)
        if missing:
            raise ValueError(f"blocks[{i}] missing keys: {sorted(missing)}")
        if len(b["rows"]) != len(b["concentrations_uM"]):
            raise ValueError(
                f"blocks[{i}]: rows ({len(b['rows'])}) and concentrations_uM "
                f"({len(b['concentrations_uM'])}) must have the same length"
            )
        if not b["enzyme_cols"]:
            raise ValueError(f"blocks[{i}]: enzyme_cols must be non-empty")
        overlap = set(b["enzyme_cols"]) & set(b["bg_cols"] or [])
        if overlap:
            raise ValueError(
                f"blocks[{i}]: enzyme_cols and bg_cols overlap: {sorted(overlap)}"
            )
    if enzyme_uM <= 0:
        raise ValueError(f"enzyme_uM must be > 0, got {enzyme_uM}")
    if signal_per_uM <= 0:
        raise ValueError(f"signal_per_uM must be > 0, got {signal_per_uM}")

    raw = parse_neo2_kinetics(data_path)
    print(
        f"Parsing [{data_path}] Measured at "
        f"[{datetime.datetime.fromtimestamp(raw['datetime'].to_list()[0])}]..."
    )

    df = raw.groupby(["Well", "time"], as_index=False).agg(signal=("value", "mean"))
    df["time"] = df["time"].astype(float) - df.groupby("Well")["time"].transform("min")

    # Build well metadata from blocks (Well is unique across all blocks)
    seen = set()
    meta_rows = []
    for bi, b in enumerate(blocks):
        row_to_conc = dict(zip(b["rows"], b["concentrations_uM"]))
        for row in b["rows"]:
            for col in b["enzyme_cols"]:
                well = f"{row}{col}"
                if well in seen:
                    raise ValueError(f"Well {well} appears in multiple blocks")
                seen.add(well)
                meta_rows.append({
                    "Well": well, "block": bi, "Row": row, "Col": int(col),
                    "role": "enzyme", "concentration_uM": float(row_to_conc[row]),
                })
            for col in (b["bg_cols"] or []):
                well = f"{row}{col}"
                if well in seen:
                    raise ValueError(f"Well {well} appears in multiple blocks")
                seen.add(well)
                meta_rows.append({
                    "Well": well, "block": bi, "Row": row, "Col": int(col),
                    "role": "background", "concentration_uM": float(row_to_conc[row]),
                })
    meta_df = pd.DataFrame(meta_rows)
    df = df.merge(meta_df, on="Well", how="inner")
    if df.empty:
        raise ValueError("None of the requested wells were found in the data file.")

    # Apply the fit window BEFORE baselining so both the slope fit and the
    # returned `df` (which plot_kinetics draws) see the same time window.
    if time_range_seconds is not None:
        t_min, t_max = time_range_seconds
        full_min, full_max = float(df["time"].min()), float(df["time"].max())
        df = df[(df["time"] >= t_min) & (df["time"] <= t_max)]
        if df.empty:
            raise ValueError(
                f"time_range_seconds={time_range_seconds} left no data. "
                f"File spans time = [{full_min:.1f}, {full_max:.1f}] s."
            )

    df = df.sort_values(["Well", "time"]).reset_index(drop=True)
    if baseline_subtract:
        df["signal_baselined"] = df["signal"] - df.groupby("Well")["signal"].transform("first")
    else:
        df["signal_baselined"] = df["signal"]

    # --- Per-well slope fit -------------------------------------------------
    slope_rows = []
    for well, g in df.groupby("Well"):
        slope_rows.append({
            "Well": well,
            "block": int(g["block"].iloc[0]),
            "Row": g["Row"].iloc[0],
            "Col": int(g["Col"].iloc[0]),
            "role": g["role"].iloc[0],
            "concentration_uM": float(g["concentration_uM"].iloc[0]),
            "slope_signal": _fit_slope(
                g["time"].to_numpy(dtype=float),
                g["signal_baselined"].to_numpy(dtype=float),
            ),
        })
    slopes = pd.DataFrame(slope_rows)

    # Per-(block, row) mean background slope (in signal/s). Correction is
    # per-block so a block with no bg_cols simply gets 0 subtracted (and its
    # enzyme rates are left uncorrected).
    bg = (
        slopes[slopes["role"] == "background"]
        .groupby(["block", "Row"], as_index=False)["slope_signal"]
        .mean()
        .rename(columns={"slope_signal": "bg_slope_signal_mean"})
    )

    enz = slopes[slopes["role"] == "enzyme"].merge(bg, on=["block", "Row"], how="left")
    enz["bg_slope_signal_mean"] = enz["bg_slope_signal_mean"].fillna(0.0)
    enz = enz.rename(columns={"slope_signal": "enzyme_slope_signal"})
    enz["corrected_slope_signal"] = enz["enzyme_slope_signal"] - enz["bg_slope_signal_mean"]
    enz["corrected_rate_uM_per_s"] = enz["corrected_slope_signal"] / signal_per_uM

    reps_df = enz[[
        "block", "Row", "Col", "concentration_uM",
        "enzyme_slope_signal", "bg_slope_signal_mean",
        "corrected_slope_signal", "corrected_rate_uM_per_s",
    ]].sort_values(
        ["concentration_uM", "block", "Col"], ascending=[False, True, True]
    ).reset_index(drop=True)

    # --- MM fit on per-concentration mean rates -----------------------------
    agg = (
        reps_df.groupby("concentration_uM", as_index=False)
        .agg(
            v_mean_uM_per_s=("corrected_rate_uM_per_s", "mean"),
            v_std_uM_per_s=("corrected_rate_uM_per_s", "std"),
            n=("corrected_rate_uM_per_s", "count"),
        )
        .sort_values("concentration_uM")
    )

    S = agg["concentration_uM"].to_numpy(dtype=float)
    v = agg["v_mean_uM_per_s"].to_numpy(dtype=float)
    p0 = [max(v.max(), 1e-9), max(np.median(S), 1e-9)]
    try:
        popt, pcov = curve_fit(
            _michaelis_menten, S, v, p0=p0,
            bounds=(0, np.inf), maxfev=10000,
        )
        Vmax, Km = float(popt[0]), float(popt[1])
        perr = np.sqrt(np.diag(pcov))
        Vmax_err, Km_err = float(perr[0]), float(perr[1])
        v_pred = _michaelis_menten(S, Vmax, Km)
        ss_res = float(np.sum((v - v_pred) ** 2))
        ss_tot = float(np.sum((v - v.mean()) ** 2))
        mm_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    except Exception as exc:
        print(f"[warn] MM fit failed: {exc}")
        Vmax = Km = Vmax_err = Km_err = mm_r2 = float("nan")

    kcat = Vmax / enzyme_uM if np.isfinite(Vmax) else float("nan")
    kcat_err = Vmax_err / enzyme_uM if np.isfinite(Vmax_err) else float("nan")
    kcat_over_Km = kcat / Km if np.isfinite(kcat) and np.isfinite(Km) and Km > 0 else float("nan")
    if np.isfinite(kcat_over_Km) and kcat > 0 and Km > 0:
        rel = np.sqrt((kcat_err / kcat) ** 2 + (Km_err / Km) ** 2)
        kcat_over_Km_err = kcat_over_Km * rel
    else:
        kcat_over_Km_err = float("nan")

    # --- kuncat: fit v_bg vs [S] (forced through zero) ---------------------
    bg_by_row = (
        slopes[slopes["role"] == "background"]
        .assign(v_bg_uM_per_s=lambda d: d["slope_signal"] / signal_per_uM)
        .groupby("concentration_uM", as_index=False)
        .agg(
            v_bg_mean=("v_bg_uM_per_s", "mean"),
            v_bg_std=("v_bg_uM_per_s", "std"),
        )
        .sort_values("concentration_uM")
    )
    if not bg_by_row.empty and (bg_by_row["concentration_uM"] > 0).any():
        Sb = bg_by_row["concentration_uM"].to_numpy(dtype=float)
        vb = bg_by_row["v_bg_mean"].to_numpy(dtype=float)
        kuncat = float((Sb * vb).sum() / (Sb * Sb).sum())
        vb_pred = kuncat * Sb
        ss_res = float(np.sum((vb - vb_pred) ** 2))
        ss_tot = float(np.sum((vb - vb.mean()) ** 2))
        kuncat_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    else:
        kuncat = float("nan")
        kuncat_r2 = float("nan")

    rate_enhancement = kcat / kuncat if np.isfinite(kcat) and np.isfinite(kuncat) and kuncat > 0 else float("nan")

    mm = {
        "Vmax_uM_per_s": Vmax,
        "Vmax_err_uM_per_s": Vmax_err,
        "Km_uM": Km,
        "Km_err_uM": Km_err,
        "kcat_per_s": kcat,
        "kcat_err_per_s": kcat_err,
        "kcat_over_Km_per_uM_per_s": kcat_over_Km,
        "kcat_over_Km_err_per_uM_per_s": kcat_over_Km_err,
        "kuncat_per_s": kuncat,
        "kcat_over_kuncat": rate_enhancement,
        "MM_R2": mm_r2,
        "kuncat_R2": kuncat_r2,
        "enzyme_uM": enzyme_uM,
        "signal_per_uM": signal_per_uM,
        "time_range_seconds": time_range_seconds,
        "_agg": agg,
        "_bg_by_row": bg_by_row,
    }

    return df, reps_df, mm


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_kinetics(df, reps_df, mm, time_scale="minute", substrate_label="[Substrate] (μM)"):
    """Plot enzyme progress curves and the Michaelis-Menten fit as two figures.

    Uses the lab palette + bold axes styling from prior work
    (Woodbury/Buse plotting convention). No background progress panel — the
    background slopes have already been subtracted upstream and would only
    add noise here.

    Args:
        df:              Long-format DataFrame from :func:`parse_kinetics`.
        reps_df:         Per-replicate DataFrame from :func:`parse_kinetics`
                         (unused for now, kept for API parity / future overlays).
        mm:              Kinetics summary dict from :func:`parse_kinetics`.
        time_scale:      ``"second"``, ``"minute"``, or ``"hour"``.
        substrate_label: X-axis label for the MM plot and legend title on the
                         progress curves. Also serves as the substrate name.
    """
    del reps_df  # currently unused; kept in signature for future extension
    scales = {"second": 1.0, "minute": 60.0, "hour": 3600.0}
    units = {"second": "s", "minute": "min", "hour": "h"}
    if time_scale not in scales:
        raise ValueError(f"time_scale must be one of {list(scales)}, got {time_scale!r}")
    tfac = scales[time_scale]
    tunit = units[time_scale]

    signal_per_uM = mm["signal_per_uM"]
    enzyme_uM = mm["enzyme_uM"]

    global _LAST_FIG
    picked = _pick_font_family()
    if picked:
        plt.rcParams["font.family"] = picked
    else:
        # No Arial-compatible font available; reset to matplotlib's default
        # so any stale value (e.g. left over from a previous version of this
        # module) doesn't keep firing findfont warnings.
        plt.rcParams["font.family"] = plt.rcParamsDefault["font.family"]
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 3.88),
        gridspec_kw={"width_ratios": [1.19, 1], "wspace": 1.1},
    )
    _LAST_FIG = fig

    # --- Progress curves (enzyme only, converted to [product] uM) -----------
    enz = df[df["role"] == "enzyme"].copy()
    # Shift time so the progress curve x-axis starts at 0 even when the
    # user trimmed the window with time_range_seconds.
    enz["t"] = (enz["time"] - enz["time"].min()) / tfac
    enz["product_uM"] = enz["signal_baselined"] / signal_per_uM

    # Descending order: highest [S] at top of legend, lowest at bottom.
    concs = sorted(enz["concentration_uM"].unique(), reverse=True)
    colors = _palette_for(len(concs))

    for c, color in zip(concs, colors):
        g = enz[enz["concentration_uM"] == c]
        stats = g.groupby("t")["product_uM"].agg(["mean", "std"]).reset_index()
        stats["std"] = stats["std"].fillna(0.0)
        ax1.plot(stats["t"], stats["mean"], color=color, label=f"{c:g}")
        ax1.fill_between(
            stats["t"],
            stats["mean"] - stats["std"],
            stats["mean"] + stats["std"],
            color=color, alpha=0.45, linewidth=0,
        )
    ax1.set_xlabel(f"Time ({tunit})", fontsize=16, fontweight="bold", labelpad=10)
    ax1.set_ylabel("[Product] (μM)", fontsize=16, fontweight="bold", labelpad=10)
    ax1.legend(
        title=substrate_label, loc="upper left", bbox_to_anchor=(1.02, 1),
        fontsize=11, title_fontsize=13, fancybox=True, borderaxespad=0.5, ncol=1,
    )
    _style_axes(ax1)

    # --- Michaelis-Menten fit -----------------------------------------------
    agg = mm["_agg"].sort_values("concentration_uM")
    S = agg["concentration_uM"].to_numpy(dtype=float)
    v_over_E = agg["v_mean_uM_per_s"].to_numpy(dtype=float) / enzyme_uM
    n = agg["n"].to_numpy(dtype=float)
    v_over_E_sem = (
        agg["v_std_uM_per_s"].fillna(0.0).to_numpy(dtype=float) / enzyme_uM
    ) / np.sqrt(np.maximum(n, 1))

    kcat = mm["kcat_per_s"]
    Km = mm["Km_uM"]
    if np.isfinite(kcat) and np.isfinite(Km):
        S_fit = np.linspace(0, max(S.max() * 1.05, 1e-12), 200)
        ax2.plot(S_fit, _michaelis_menten(S_fit, kcat, Km),
                 color="black", linewidth=2)

    ax2.scatter(S, v_over_E, color=_GOOD_RED, s=70, zorder=3)
    err = ax2.errorbar(
        S, v_over_E, yerr=v_over_E_sem,
        color="black", fmt="none", capsize=4, linewidth=2.5, zorder=2,
    )
    for cap in err[1]:
        cap.set_markeredgewidth(2)

    ax2.set_xlabel(substrate_label, fontsize=16, fontweight="bold", labelpad=10)
    ax2.set_ylabel(r"v/[E] x10$^{-3}$ (s$^{-1}$)", fontsize=16, fontweight="bold", labelpad=10)
    ax2.set_ylim(bottom=0)
    # Cap at 4 y-ticks; freeze locations then display values * 1e3 (per the
    # "x10^-3" label).
    from matplotlib.ticker import MaxNLocator
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="lower"))
    fig.canvas.draw()
    y_ticks = ax2.get_yticks()
    ax2.set_yticks(y_ticks)
    ax2.set_yticklabels([f"{t * 1e3:.2f}" for t in y_ticks])
    _style_axes(ax2)

    plt.show()

    # --- Formatted summary --------------------------------------------------
    kcat_err = mm["kcat_err_per_s"]
    Km_err = mm["Km_err_uM"]
    ke_M = mm["kcat_over_Km_per_uM_per_s"] * 1e6
    ke_M_err = mm["kcat_over_Km_err_per_uM_per_s"] * 1e6
    print(f"kcat:    {kcat:.6f} ± {kcat_err:.6f} s⁻¹")
    print(f"Km:      {Km:.2f} ± {Km_err:.2f} μM")
    print(f"kcat/Km: {ke_M:.3f} ± {ke_M_err:.3f} M⁻¹ s⁻¹")
    if np.isfinite(mm["kuncat_per_s"]):
        print(f"kuncat:  {mm['kuncat_per_s']:.3e} s⁻¹   (R² = {mm['kuncat_R2']:.3f})")
    if np.isfinite(mm["kcat_over_kuncat"]):
        print(f"kcat/kuncat: {mm['kcat_over_kuncat']:.4g}")

    return fig


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_plot(save_path, fig=None, dpi=600, transparent=True):
    """Save the current (or given) matplotlib figure to disk.

    Format is inferred from the file extension. Supported: ``.png``, ``.eps``,
    ``.pdf``, ``.svg``. Uses ``bbox_inches='tight'`` so labels/legends drawn
    outside the axes area (e.g. the substrate-legend on the progress panel)
    are included in the output rather than clipped.

    Args:
        save_path:    Output path. Extension determines format.
        fig:          Figure to save. Defaults to the current figure
                      (``plt.gcf()``), which is what ``plot_kinetics`` /
                      ``parse_standard_curve`` leave in place after they draw.
        dpi:          Resolution for raster formats (png). Ignored by vector
                      formats. Default 600 for print-quality output.
        transparent:  Save with a transparent background. Nice for
                      compositing figures onto slides / posters.

    Example::

        plot_kinetics(df, reps_df, mm)
        export_plot("kinetics_20260805.eps")
    """
    import os
    supported = {"png", "eps", "pdf", "svg"}
    ext = os.path.splitext(save_path)[1].lower().lstrip(".")
    if ext not in supported:
        raise ValueError(
            f"Unsupported extension '.{ext}'. Use one of: {sorted(supported)}"
        )
    if fig is None:
        # Prefer the cached fig from the last plot_kinetics call — Jupyter's
        # inline backend closes figures after `plt.show()`, so plt.gcf()
        # would otherwise return an empty new canvas.
        fig = _LAST_FIG if _LAST_FIG is not None else plt.gcf()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                transparent=transparent, format=ext)
    print(f"Saved figure to {save_path}")
