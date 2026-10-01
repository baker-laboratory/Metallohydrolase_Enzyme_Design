"""RFdiffusion3 phosphotriesterase wet-lab analysis from deposited raw data."""

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo

    _repo = Path(__file__).resolve().parents[2]
    _helpers = str(_repo / "Scripts" / "wetlab_platereader")
    if _helpers not in sys.path:
        sys.path.insert(0, _helpers)
    import analysis_pte_rfd3 as analysis

    return analysis, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # RFdiffusion3 phosphotriesterase analysis

    This notebook reads the deposited plate-reader acquisitions and ordered
    designs, refits the calibration, background reaction and **19
    enzyme/condition entries**, and analyzes both eluate screens. The tagless
    ZAPP-1 construct is included. Calculations do not use saved Jupyter outputs
    or previously exported fit tables.

    Every screening curve is shown individually and colored by scaffold.
    Designs subsequently tested by kinetics have bold curves and direct labels.
    Select an assay or scaffold below to inspect its measurements.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Calibration and background reaction

    Four triplicate 4-nitrophenol standards calibrate absorbance at 405 nm.
    The two legacy-buffer slopes are checked against the precision used by
    the published fits. Enzyme-free paraoxon hydrolysis is refitted over every
    original admissible time window: at least 45 minutes, starts at or after
    15 minutes, five-minute grid, and $R^2 \geq 0.90$. The free intercept
    absorbs substrate-independent baseline drift.
    """)
    return


@app.cell
def _(analysis):
    calibration = analysis.fit_calibration()
    return (calibration,)


@app.cell(hide_code=True)
def _(calibration, mo):
    mo.ui.table(calibration, selection=None, label="Calibration: absorbance per µM")
    return


@app.cell
def _(analysis):
    background = analysis.fit_background()
    return (background,)


@app.cell(hide_code=True)
def _(background, mo):
    mo.ui.table(
        [{"condition": _name, **_result} for _name, _result in background.items()],
        selection=None,
        label="Uncatalyzed rates: k and uncertainty in s⁻¹; fit-window limits in seconds",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Enzyme kinetics

    All fits use the deposited assay-specific enzyme concentration, wells,
    backgrounds and time window. Three enzyme replicates span six paraoxon
    concentrations. Progress curves retain the entire acquisition; shaded
    portions lie outside the fitted interval.

    The table keeps fit-only and total uncertainty separately. Total uncertainty
    includes the published 6% enzyme-concentration term and calibration term.
    H89A and H170A retain their first-order reporting rule because separate
    $k_{\mathrm{cat}}$ and $K_M$ are unresolved. Their nonlinear fits remain
    available as diagnostics.
    """)
    return


@app.cell
def _(analysis):
    fits = analysis.fit_kinetics()
    return (fits,)


@app.cell
def _(analysis, background, fits):
    summary, reference_checks = analysis.summarize_kinetics(fits, background)
    return reference_checks, summary


@app.cell(hide_code=True)
def _(mo, reference_checks, summary):
    mo.vstack([
        mo.md(f"**{len(summary)} fresh fits; {len(reference_checks)} reference checks passed.**"),
        mo.ui.table(summary, selection=None, page_size=20, label="Kinetic parameters and reporting status"),
    ])
    return


@app.cell
def _(fits, mo):
    assay = mo.ui.dropdown(options=list(fits), value="ZAPP-1", label="Kinetics assay")
    assay
    return (assay,)


@app.cell
def _(analysis, assay, background, fits):
    kinetics_view = analysis.kinetics_figure(assay.value, fits, background)
    kinetics_view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Eluate screening — every design

    The round-1 screen contains 96 designs on six scaffolds; round 2 contains
    192 designs on eleven scaffolds. Colors follow the scaffold identifiers in
    the deposited sequence/model workbook: **1–6** for round 1 and **7–17**
    for round 2. The original round-2 order uses 1–11; adding six gives the
    global model identifier.

    Each line is its own measured absorbance change from the first reading.
    Negative-going traces are retained. There is no percentile band or hit
    filter in these plots. Kinetics-tested designs are drawn last with thicker
    lines and direct endpoint labels. Screening eluates are not normalized for
    enzyme concentration.
    """)
    return


@app.cell
def _(analysis):
    round1_time, round1_absorbance, round1_mapping = analysis.load_screening_PTE_RFd3(1)
    round2_time, round2_absorbance, round2_mapping = analysis.load_screening_PTE_RFd3(2)
    return round1_mapping, round2_mapping


@app.cell
def _(analysis):
    round1_view, _ = analysis.plot_screening_PTE_RFd3(1, show=False)
    round1_view
    return


@app.cell
def _(analysis):
    round2_view, _ = analysis.plot_screening_PTE_RFd3(2, show=False)
    round2_view
    return


@app.cell
def _(mo, round1_mapping, round2_mapping):
    scaffold_selection = mo.ui.dropdown(
        options={
            f"Round {_round} / scaffold {_scaffold}": (_round, int(_scaffold))
            for _round, _mapping in [(1, round1_mapping), (2, round2_mapping)]
            for _scaffold in sorted(_mapping.scaffold.unique())
        },
        value="Round 1 / scaffold 1",
        label="Inspect one scaffold (the complete screens remain above)",
    )
    scaffold_selection
    return (scaffold_selection,)


@app.cell
def _(analysis, scaffold_selection):
    _round, _scaffold = scaffold_selection.value
    scaffold_view, _ = analysis.plot_screening_PTE_RFd3(_round, scaffold=_scaffold, show=False)
    scaffold_view
    return


@app.cell(hide_code=True)
def _(mo, round1_mapping, round2_mapping):
    _columns = ["design_id", "reader_well", "scaffold", "screen_scaffold", "color", "kinetics_tested"]
    mo.ui.tabs({
        "Round 1 assignments": mo.ui.table(round1_mapping[_columns], selection=None),
        "Round 2 assignments": mo.ui.table(round2_mapping[_columns], selection=None),
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Round-2 initial rates and mapping validation

    The ordered FASTA and SI workbook independently identify the same reader
    wells and scaffolds. The interleaved row-pair mapping is also checked against
    the designs sent for sequencing. Initial rates preserve the original
    2–15 minute fit window, optical-artifact flags, absorbance ceiling, background
    estimate and hit thresholds. These rules affect the rate table, not whether
    a trace is displayed above.
    """)
    return


@app.cell
def _(analysis):
    screen = analysis.load_round2_screen()
    mapping_checks = analysis.validate_round2_mapping(screen)
    screening_rates, screening_background = analysis.fit_round2_screen(screen)
    return mapping_checks, screen, screening_background, screening_rates


@app.cell(hide_code=True)
def _(mapping_checks, mo, screening_background, screening_rates):
    mo.vstack([
        mo.ui.table(mapping_checks, selection=None, label="Stamping hypotheses"),
        mo.md(
            f"**{len(screening_rates)} designs; {int(screening_rates.is_hit.sum())} original-rule hits.** "
            f"Inactive-population background: {screening_background['background_rate']:.3g} "
            "absorbance units per minute."
        ),
        mo.ui.table(screening_rates, selection=None, label="All initial rates, flags and hit calls"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tagless ZAPP-1

    Tagless ZAPP-1 uses enzyme columns 9–11 and background column 12 of the
    ZAPP-3/ZAPP-4 plate, with 23.4 µM enzyme. It is included among the 19 fits
    above. The following check repeats the tagged/tagless fits using the shorter
    detector-linearity window, retaining the original window alongside it.
    The mass spectrum is the deposited original instrument image.
    """)
    return


@app.cell
def _(analysis, background, fits):
    tagless_checks = analysis.tagless_linearity_checks(background)
    tagless_view = analysis.kinetics_figure("ZAPP-1 tagless", fits, background)
    tagless_view
    return (tagless_checks,)


@app.cell(hide_code=True)
def _(analysis, mo, tagless_checks):
    mo.vstack([
        mo.ui.table(tagless_checks, selection=None, label="Tagless/tagged detector-linearity check"),
        mo.image(analysis.DATASET / "mass_spectrometry" / "ZAPP1_tagless" / "SMW_ZAPP_tagless.png"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Export reproducible results and manuscript figures

    Export writes the newly calculated tables, all 19 kinetics plots, the two
    complete screening plots, and the original SI/Figure 4 reproductions.
    Historical manuscript artwork retains its original presentation. The
    complete scaffold-colored screening views above are the current analysis
    views. The original Figure 4E and its condition-matched-calibration variant
    are both retained.

    The final tagless SI PDF/PNG, TeX caption, source tables, sequences and
    instrument spectra are deposited alongside this notebook. The export
    rebuilds the kinetic side panel from measurements; it does not redraw the
    instrument-exported mass spectrum.
    """)
    return


@app.cell
def _(mo):
    export = mo.ui.run_button(label="Export tables and all figures")
    export
    return (export,)


@app.cell
def _(analysis, background, export, fits, mo, reference_checks, screen, screening_rates, summary):
    mo.stop(not export.value)
    export_path = analysis.export_results(
        fits, background, summary, reference_checks, screen, screening_rates,
        manuscript_figures=True,
    )
    mo.md(f"Exported to `{export_path.name}/`.")
    return


if __name__ == "__main__":
    app.run()
