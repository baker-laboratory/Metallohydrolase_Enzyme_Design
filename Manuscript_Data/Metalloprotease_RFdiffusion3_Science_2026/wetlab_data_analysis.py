"""Metalloprotease screening, peptide-reporter specificity and mass spectra."""

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    import sys
    import marimo as mo

    _scripts = str(Path(__file__).resolve().parent / "scripts")
    if _scripts not in sys.path:
        sys.path.insert(0, _scripts)
    import metalloprotease_analysis as analysis
    return analysis, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Metalloprotease screening and characterization

    Gel-derived cleaved fractions, fluorescence screening timecourses,
    G11 activity with short peptide reporters, and a deconvoluted mass spectrum.
    Calculations read the deposited CSV files. Generated plots and tables are
    written to `wetlab_data_plots/`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Cis screening

    Cleaved-fraction distributions at 1 and 18 hours for the gel lanes labeled
    `w Zn` in the quantification files. The tables give lane numbers and
    incubation time; the 1-hour table retains plate locations and identifiers,
    and the 18-hour table retains the gel number.
    """)
    return


@app.cell
def _(analysis):
    cis = analysis.run_cis_screen()
    return (cis,)


@app.cell(hide_code=True)
def _(cis, mo):
    mo.vstack([mo.as_html(_figure) for _figure in cis["figures"]])
    return


@app.cell(hide_code=True)
def _(cis, mo):
    mo.ui.table(cis["tables"]["df_all_cis_screen"], selection=None,
                page_size=20, label="Cleaved fractions by gel lane and incubation time")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Trans screening

    Fluorescence progress curves from the selected odd-numbered plate columns,
    displayed through 3,600 seconds. Each curve is referenced to its first
    reading. Wells G7, G19 and G21 are highlighted. Initial slopes are fitted
    over 900–1,500 seconds and reported in RFU per second.
    """)
    return


@app.cell
def _(analysis):
    trans = analysis.run_trans_screen()
    return (trans,)


@app.cell(hide_code=True)
def _(mo, trans):
    mo.vstack([mo.as_html(_figure) for _figure in trans["figures"]])
    return


@app.cell(hide_code=True)
def _(mo, trans):
    mo.ui.table(trans["tables"]["rates"].sort_values("initial_rate", ascending=False),
                selection=None, page_size=20, label="Initial slopes by well")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## G11 short peptide reporters

    Fluorescence timecourses for the seven short peptide reporters labeled
    Tau1, Tau2, SAA, αSyn1, αSyn2, TDP43 and TTR. Each well is referenced to
    its first reading and corrected using the matched no-protease well.
    Curves show replicate means with standard-deviation bands.
    """)
    return


@app.cell
def _(analysis):
    specificity = analysis.run_specificity()
    return (specificity,)


@app.cell(hide_code=True)
def _(mo, specificity):
    mo.as_html(specificity["figures"][-1])
    return


@app.cell(hide_code=True)
def _(mo, specificity):
    mo.ui.table(specificity["tables"]["df"], selection=None,
                page_size=20, label="Peptide-reporter fluorescence measurements")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Deconvoluted mass spectrum

    The deposited spectrum is displayed over 20,000–30,000 Da and
    55,000–60,000 Da. The table previews the numerical mass and count columns.
    """)
    return


@app.cell
def _(analysis):
    mass_spectrum = analysis.run_mass_spectrometry()
    return (mass_spectrum,)


@app.cell(hide_code=True)
def _(mass_spectrum, mo):
    mo.vstack([mo.as_html(_figure) for _figure in mass_spectrum["figures"]])
    return


@app.cell(hide_code=True)
def _(mass_spectrum, mo):
    _table = mass_spectrum["tables"]["ms_plot"]
    mo.vstack([
        mo.md(f"{len(_table):,} numerical points; previewing the first 500 rows."),
        mo.ui.table(_table.head(500), selection=None, page_size=20),
    ])
    return


if __name__ == "__main__":
    app.run()
