import marimo

__generated_with = "0.25.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # RFdiffusion2 Nature Methods: 4MU-butyrate wetlab analysis

    Screening and kinetics for the additional 4MU-butyrate metallohydrolases in Ahern *et al.*, **Nature Methods 23, 96–105 (2026)** ([paper](https://doi.org/10.1038/s41592-025-02975-x)). The shared 4MU-phenylacetate experiments are in [Metallohydrolase_Nature_2026](../Metallohydrolase_Nature_2026/).

    All required inputs are included. Publication kinetics use the 2025-04-01 calibration plate. Instrument file-location metadata have been removed from the experimental workbooks; measurement values are unchanged.
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import os, sys
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import linregress
    from scipy.optimize import curve_fit
    import marimo as mo
    _start = Path(os.environ.get('ZINC_HYDRO_REPO') or mo.notebook_location()).resolve()
    repo = next((p for p in (_start, *_start.parents) if (p / 'Scripts').is_dir() and (p / 'LICENSE').exists()), None)
    if repo is None:
        raise RuntimeError('Open this notebook inside the cloned repository, or set ZINC_HYDRO_REPO.')
    working_dir = repo / 'Manuscript_Data' / 'Metallohydrolase_RFdiffusion2_Nature_Methods_2026'
    raw_dir = working_dir / 'raw_wetlab_data'
    plots_dir = working_dir / 'wetlab_data_plots'
    plots_dir.mkdir(exist_ok=True)
    sys.path.insert(0, str(repo / 'Scripts'))
    from experimental_data_processing_functions import parse_kinetics, parse_excels, standard_curve, mm_kinetics, draw_progression_curve_v0, draw_progression_curve_seperate_v0, save_figure
    import experimental_data_processing_functions as edpf
    edpf.FIGURE_DPI, edpf.SAVE_EPS = (150, False)
    raw_wetlab_data_dir = str(raw_dir) + '/'
    _pics_dir = str(plots_dir) + '/'
    print('Inputs: raw_wetlab_data/\nFigures and result tables: wetlab_data_plots/')
    return (
        curve_fit,
        edpf,
        linregress,
        mm_kinetics,
        mo,
        np,
        os,
        parse_excels,
        parse_kinetics,
        pd,
        plots_dir,
        plt,
        raw_dir,
        save_figure,
        standard_curve,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # I. Functional screening data

    The published screening uses the **2024-07-10** plate. The original figure excludes H8 and H9 and identifies C4, B11 and D11 as the three strongest progress curves. The file name records the assay preparation; the Methods describes the final reaction mixture. Screening enzyme concentrations were not normalized, so the screen is a hit-finding assay, not a ranking of catalytic efficiencies.
    """)
    return


@app.cell
def _(parse_excels, plots_dir, plt, raw_dir, save_figure):
    screen_file = str(raw_dir / '240710_zn_hydrolase_96_order1_200uM_zinc_100uM_butyrate.xlsx')
    screen = parse_excels(screen_file, None, start_row=48)
    assert all((f'{r}{c}' in screen for r in 'ABCDEFGH' for c in range(1, 13)))
    print(f'Screen contains all 96 plate positions and {len(screen)} time points.')
    # Preserve the publication notebook's preprocessing exactly: exclude H8/H9
    # before hit selection and normalize each included trace to its first reading.
    publication_wells = [f'{r}{c}' for r in 'ABCDEFGH' for c in range(1, 13) if f'{r}{c}' not in ['H8', 'H9']]
    publication_screen = screen.loc[(screen.Time > 0) & (screen.Time < 60), ['Time', *publication_wells]].copy()
    publication_screen['Time'] -= publication_screen['Time'].iloc[0]
    publication_screen[publication_wells] -= publication_screen[publication_wells].iloc[0]
    publication_hits = publication_screen[publication_wells].iloc[-1].nlargest(3).index.tolist()
    assert publication_hits == ['C4', 'B11', 'D11']
    _fig, _ax = plt.subplots(figsize=(4, 3.5))
    for _well in publication_wells:
        if _well in publication_hits:
            _ax.plot(publication_screen.Time, publication_screen[_well], color='red')
            _ax.text(publication_screen.Time.iloc[-1], publication_screen[_well].iloc[-1], _well, color='red', ha='left', va='bottom', fontsize=8)
        else:
            _ax.plot(publication_screen.Time, publication_screen[_well], color='lightgray', alpha=0.2)
    _ax.set(xlabel='Time (minutes)', ylabel='RFU')
    left, right = _ax.get_xlim()
    _ax.set_xlim(left, right * 1.01)
    _fig.tight_layout()
    save_figure(str(plots_dir / '4MU_B_published_screen_combined'))
    plt.show()
    plt.close(_fig)
    print('Publication hits:', ', '.join(publication_hits))
    window = screen.loc[(screen.Time > 0) & (screen.Time < 60)]
    wells = [f'{r}{c}' for r in 'ABCDEFGH' for c in range(1, 13) if f'{r}{c}' not in ['H8', 'H9']]
    # Verify the publication's hit identities independently of figure rendering.
    change = window[wells].iloc[-1] - window[wells].iloc[0]
    assert list(change.nlargest(3).index) == ['C4', 'B11', 'D11']
    change.rename('change_RFU').rename_axis('well').to_csv(plots_dir / 'screen_signal_changes.csv')
    return (screen,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## I.A. All plate positions

    This grid retains H8/H9 and shows raw fluorescence for every well so the complete screen can be inspected. The publication's combined plot above preserves its original exclusions.
    """)
    return


@app.cell
def _(plots_dir, plt, save_figure, screen):
    _fig, _axes = plt.subplots(8, 12, figsize=(24, 16), sharex=True, sharey=True)
    for i, _row in enumerate('ABCDEFGH'):
        for j, _col in enumerate(range(1, 13)):
            _well = f'{_row}{_col}'
            _ax = _axes[i, j]
            _ax.plot(screen.Time, screen[_well], color='red' if _well in ['C4', 'B11', 'D11'] else 'gray', lw=1)
            _ax.set_title(_well, fontsize=8)
            _ax.tick_params(labelsize=6)
    _fig.supxlabel('Time (minutes)')
    _fig.supylabel('Fluorescence units')
    _fig.tight_layout()
    save_figure(str(plots_dir / '4MU_B_screen_all_96_wells'))
    plt.show()
    plt.close(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## I.B. Archived screening conditions (2024-07-16)

    Three additional full-plate screens are included as raw TSV exports. These are not substituted for the 2024-07-10 publication plate. File names identify the original assay conditions; the panels highlight the same three subsequently characterized hits.
    """)
    return


@app.cell
def _(pd, plots_dir, plt, raw_dir, save_figure):
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4))
    for _ax, filename in zip(_axes, sorted(raw_dir.glob('240716*butyrate.tsv'))):
        _data = pd.read_csv(filename, sep='\t')
        time_minutes = pd.to_timedelta(_data.Time).dt.total_seconds() / 60
        for _row in 'ABCDEFGH':
            for _col in range(1, 13):
                _well = f'{_row}{_col}'
                _y = pd.to_numeric(_data[_well], errors='raise')
                _ax.plot(time_minutes, _y - _y.iloc[0], color='red' if _well in ['B11', 'C4', 'D11'] else 'lightgray', lw=1)
        _ax.set_title(filename.stem.split('order1_')[-1].split('Order1_')[-1].replace('_', ' '), fontsize=9)
        _ax.set_xlabel('Time (minutes)')
    _axes[0].set_ylabel('Change in fluorescence units')
    _fig.tight_layout()
    save_figure(str(plots_dir / '4MU_B_archived_screen_conditions'))
    plt.show()
    plt.close(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # II. Kinetics

    All three publication hits were measured at 3 µM enzyme and 20 µM Zn(II), with 0.77–99 µM 4MU-butyrate. C4 and B11 use 0–200 s; D11 uses 0–300 s. The final publication notebook uses the 2025-04-01 calibration, which is re-derived below from the included plate. Progress curves and Michaelis–Menten fits use the same shared functions as the Nature metallohydrolase notebook.

    For D11, the available substrate range does not establish saturation. As stated in the paper's supplementary methods, only the low-substrate estimate of kcat/KM is meaningful; separate kcat and KM are not reported.
    """)
    return


@app.cell
def _(np, parse_kinetics, raw_dir, standard_curve):
    calibration_file = str(raw_dir / '250401_4mu_standard_curves_100uL_30uMladder_and_45uMladder_25C_5prctDMSO__pH8.xlsx')
    concentrations = [30, 15, 7.5, 3.75, 1.875, 0.9375, 45, 22.5, 11.25, 5.625, 2.8125, 1.40625]
    calibration = parse_kinetics(calibration_file, None, start_row=48)
    calibration_slopes = []
    for _row in 'DEF':
        _, _, _slope = standard_curve(calibration, _row, (1, 12), 250000, concentrations, 0, 50000)
        calibration_slopes.append(_slope)
    SLOPE_4MU_250401 = float(np.mean(calibration_slopes))
    np.testing.assert_allclose(calibration_slopes, [9.296712951159367e-10, 9.447803716354774e-10, 9.40363467934009e-10], rtol=1e-10)
    print(f'Publication calibration: {SLOPE_4MU_250401:.12g} M/FU')
    return (SLOPE_4MU_250401,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## II.A. C4
    """)
    return


@app.cell
def _(
    SLOPE_4MU_250401,
    mm_kinetics,
    np,
    os,
    parse_kinetics,
    plots_dir,
    raw_dir,
):
    # excel file output from neo2
    _excel_file = str(raw_dir / '240920_C4_OFFICIAL_kinetics_3uM___butyrate_subst_25C_4min_20uM_zn.xlsx')
    _slope = SLOPE_4MU_250401
    # Standarization slope
    _start_time = 0
    _end_time = 200
    # Time to investigate
    _substrate_name = '4MU-BA'
    _enz_conc = 3
    _rows, _cols = (list('ABCDEFGH'), [5, 6, 7, 8])
    # What substrate did you use? Important for normalization and for pretty (and accurate) plots.
    _transpose_dic = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 5: 'A', 6: 'B', 7: 'C', 8: 'D'}
    _pics_dir = str(plots_dir) + '/'
    # - what enzyme concentration did you test? (μΜ) - #
    os.makedirs(_pics_dir, exist_ok=True)
    _rename_dic = {}
    # Well information
    for _key in _transpose_dic:
        _value = _transpose_dic[_key]
    # Transpose information
        if _key in _rows:
            for _col in _cols:
                _rename_dic[f'{_key}{_col}'] = f'{_transpose_dic[_col]}{_value}'
        elif _key in _cols:
            for _row in _rows:
                _rename_dic[f'{_row}{_key}'] = f'{_value}{_transpose_dic[_row]}'
    _row_column_ranges = {'A': (1, 8)}
    _replicate_rows = ['B', 'C']
    _active_wells = []
    for _row, (_start_col, _end_col) in _row_column_ranges.items():
        for _col in range(_start_col, _end_col + 1):
            _active_wells.append(f'{_row}{_col}')
    _sub_concs = np.array([99, 49.5, 24.75, 12.38, 6.19, 3.09, 1.55, 0.77])
    _bg_row = 'D'
    # make a directory for cute pics
    _bn = os.path.basename(f'{_excel_file}').replace('.xlsx', '')
    _kinetic_data = parse_kinetics(_excel_file, _active_wells, start_row=48)
    _kinetic_data.rename(columns=_rename_dic, inplace=True)
    _v0_enz = mm_kinetics(_kinetic_data, _bn, active_wells=_active_wells, bg_well_id=_bg_row, replicate_rows=_replicate_rows, enz_conc=_enz_conc, sub_concs=_sub_concs, substrate_name=_substrate_name, pics_dir=_pics_dir, fit_type='straight_line', slope=_slope, cutoff_start_time=_start_time, cutoff_end_time=_end_time, plate_type='full', y_ax_mm_intv=0.1, x_ax_mm_intv=50, legend_label=f'[{_substrate_name}], μM', norm_zero=True)
    # specify rows and columns where measurements were taken
    # rows containing replicates
    # Create an empty list to store active wells
    # Iterate through rows/columns, append to active wells
    # - what substrate concentrations did you test in those specific wells? (μM) - #
    # which row is background?
    # plotting stuff
    results_C4 = (_kinetic_data.copy(), np.array(_v0_enz), _start_time, _end_time)
    return (results_C4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## II.B. B11
    """)
    return


@app.cell
def _(
    SLOPE_4MU_250401,
    mm_kinetics,
    np,
    os,
    parse_kinetics,
    plots_dir,
    raw_dir,
):
    # excel file output from neo2
    _excel_file = str(raw_dir / '240920_B11_OFFICIAL_kinetics_3uM___butyrate_subst_25C_4min_20uM_zn.xlsx')
    _slope = SLOPE_4MU_250401
    # Standarization slope
    _start_time = 0
    _end_time = 200
    # Time to investigate
    _substrate_name = '4MU-BA'
    _enz_conc = 3
    _rows, _cols = (list('ABCDEFGH'), [1, 2, 3, 4])
    # What substrate did you use? Important for normalization and for pretty (and accurate) plots.
    _transpose_dic = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 1: 'A', 2: 'B', 3: 'C', 4: 'D'}
    _pics_dir = str(plots_dir) + '/'
    # - what enzyme concentration did you test? (μΜ) - #
    os.makedirs(_pics_dir, exist_ok=True)
    _rename_dic = {}
    # Well information
    for _key in _transpose_dic:
        _value = _transpose_dic[_key]
    # Transpose information
        if _key in _rows:
            for _col in _cols:
                _rename_dic[f'{_key}{_col}'] = f'{_transpose_dic[_col]}{_value}'
        elif _key in _cols:
            for _row in _rows:
                _rename_dic[f'{_row}{_key}'] = f'{_value}{_transpose_dic[_row]}'
    _row_column_ranges = {'A': (1, 8)}
    _replicate_rows = ['B', 'C']
    _active_wells = []
    for _row, (_start_col, _end_col) in _row_column_ranges.items():
        for _col in range(_start_col, _end_col + 1):
            _active_wells.append(f'{_row}{_col}')
    _sub_concs = np.array([99, 49.5, 24.75, 12.38, 6.19, 3.09, 1.55, 0.77])
    _bg_row = 'D'
    # make a directory for cute pics
    _bn = os.path.basename(f'{_excel_file}').replace('.xlsx', '')
    _kinetic_data = parse_kinetics(_excel_file, _active_wells, start_row=48)
    _kinetic_data.rename(columns=_rename_dic, inplace=True)
    _v0_enz = mm_kinetics(_kinetic_data, _bn, active_wells=_active_wells, bg_well_id=_bg_row, replicate_rows=_replicate_rows, enz_conc=_enz_conc, sub_concs=_sub_concs, substrate_name=_substrate_name, pics_dir=_pics_dir, fit_type='straight_line', slope=_slope, cutoff_start_time=_start_time, cutoff_end_time=_end_time, plate_type='full', y_ax_mm_intv=0.1, x_ax_mm_intv=50, legend_label=f'[{_substrate_name}], μM', norm_zero=True)
    # specify rows and columns where measurements were taken
    # rows containing replicates
    # Create an empty list to store active wells
    # Iterate through rows/columns, append to active wells
    # - what substrate concentrations did you test in those specific wells? (μM) - #
    # which row is background?
    # plotting stuff
    results_B11 = (_kinetic_data.copy(), np.array(_v0_enz), _start_time, _end_time)
    return (results_B11,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## II.C. D11: unsaturated substrate range
    """)
    return


@app.cell
def _(
    SLOPE_4MU_250401,
    curve_fit,
    edpf,
    linregress,
    mo,
    np,
    parse_kinetics,
    pd,
    plots_dir,
    plt,
    raw_dir,
    results_B11,
    results_C4,
    save_figure,
):
    kinetic_summaries = []
    initial_rates = []
    _sub_concs = np.array([99, 49.5, 24.75, 12.38, 6.19, 3.09, 1.55, 0.77])
    file = raw_dir / '240920_D11_OFFICIAL_kinetics_3uM___butyrate_subst_25C_4min_20uM_zn.xlsx'
    data_D11 = parse_kinetics(str(file), None, start_row=48)
    rename = {f'{row}{col}': f"{'ABCD'[col - 9]}{i + 1}" for i, row in enumerate('ABCDEFGH') for col in [9, 10, 11, 12]}
    data_D11 = data_D11.rename(columns=rename)
    results_D11 = (data_D11, None, 0, 300)
    reference_rates = {'C4': [0.005421716876534271, 0.003128023464041712, 0.001754808683110724, 0.0009456295260520466, 0.0004778770195894712, 0.0002385575875186742, 0.00012291760392276586, 6.461644714452363e-05], 'B11': [0.0005709222976526836, 0.0003704268649288834, 0.0002457850864934184, 0.00014983277000763182, 8.201858059827681e-05, 4.099926602871574e-05, 2.5962860394712416e-05, 1.4154269836808468e-05], 'D11': [0.0008872122009141482, 0.0004316648904182959, 0.0002192663641355908, 0.00011444644868848663, 5.661893868771548e-05, 2.785494143699116e-05, 1.3940607363202706e-05, 6.636632906040338e-06]}
    # Reproduce all three sets of replicate initial rates and export a compact table.
    for name, (_data, helper_rates, _start, end) in {'C4': results_C4, 'B11': results_B11, 'D11': results_D11}.items():
        selected = _data.loc[(_data.Time >= _start) & (_data.Time <= end)]
        rates = []
        traces = []
        for _col, substrate in enumerate(_sub_concs, 1):
            replicate_traces = []
            for _row in 'BCA':
                _y = (selected[f'{_row}{_col}'] - selected[f'D{_col}']) * SLOPE_4MU_250401 * 1000000.0
                rate = linregress(selected.Time, _y).slope / 3.0
                rates.append(rate)
                replicate_traces.append((_y - _y.min()).to_numpy(float))
                initial_rates.append(dict(design=name, substrate_uM=substrate, replicate=_row, v_over_E_per_s=rate, start_s=_start, end_s=end))  # same replicate ordering as the original helper
            traces.append(np.array(replicate_traces))
        rates = np.array(rates).reshape(8, 3)
        means = rates.mean(axis=1)
        np.testing.assert_allclose(means, reference_rates[name], rtol=1e-10, atol=1e-14)
        if helper_rates is not None:
            np.testing.assert_allclose(means, helper_rates, rtol=1e-10)

            def mm(s, kcat, km):
                return kcat * s / (km + s)
            (kcat, km), covariance = curve_fit(mm, np.repeat(_sub_concs, 3), rates.ravel())
            kcat_sd, km_sd = np.sqrt(np.diag(covariance))
            efficiency = kcat / km * 1000000.0
            efficiency_sd = efficiency * np.sqrt((kcat_sd / kcat) ** 2 + (km_sd / km) ** 2)
            kinetic_summaries.append(dict(design=name, model='Michaelis-Menten', kcat_per_s=kcat, kcat_fit_sd=kcat_sd, KM_uM=km, KM_fit_sd=km_sd, kcat_over_KM_Minv_sinv=efficiency, efficiency_fit_sd=efficiency_sd))
        else:
            x = np.repeat(_sub_concs, 3) * 1e-06
            efficiency = float(np.dot(x, rates.ravel()) / np.dot(x, x))
            residual = rates.ravel() - efficiency * x
            efficiency_sd = float(np.sqrt(np.sum(residual ** 2) / (len(x) - 1) / np.dot(x, x)))
            kinetic_summaries.append(dict(design=name, model='low-substrate linear limit', kcat_per_s=np.nan, kcat_fit_sd=np.nan, KM_uM=np.nan, KM_fit_sd=np.nan, kcat_over_KM_Minv_sinv=efficiency, efficiency_fit_sd=efficiency_sd))
            _fig, _axes = plt.subplots(1, 2, figsize=(10, 4))
            colors = [edpf.good_purple, edpf.good_blue, edpf.good_teal, edpf.good_green, edpf.good_yellow, edpf.good_peach, edpf.good_red, edpf.good_pink]
            for substrate, traces_i, color in zip(_sub_concs, traces, colors):
                mean, sd = (traces_i.mean(axis=0), traces_i.std(axis=0))
                _axes[0].plot(selected.Time, mean, label=f'{substrate}', color=color)
                _axes[0].fill_between(selected.Time, mean - sd, mean + sd, color=color, alpha=0.45)
            _axes[0].set(xlabel='Time (s)', ylabel='[4MU] (µM)')
            _axes[0].legend(title='[4MU-B] (µM)', fontsize=7)
            _axes[1].errorbar(_sub_concs, means, yerr=rates.std(axis=1, ddof=1) / np.sqrt(3), fmt='o', color=edpf.good_red, capsize=4)
            _axes[1].plot([0, 99], efficiency * np.array([0, 99]) * 1e-06, color='black')
            _axes[1].set(xlabel='[4MU-B] (µM)', ylabel='v/[E] (s⁻¹)', title='D11: low-substrate estimate')
            _fig.tight_layout()
            save_figure(str(plots_dir / 'D11_progress_and_low_substrate_fit'))
            plt.show()
            plt.close(_fig)
    kinetics_summary = pd.DataFrame(kinetic_summaries)
    kinetics_summary.to_csv(plots_dir / 'kinetics_summary.csv', index=False)
    pd.DataFrame(initial_rates).to_csv(plots_dir / 'kinetics_initial_rates.csv', index=False)
    mo.ui.table(kinetics_summary, selection=None)
    print('Kinetic analysis complete.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## II.D. Additional experimental inputs

    The 2024-09-20 standard curve and exploratory C10/E10 experiments are included in `raw_wetlab_data/`. Publication fits use the three characterized hits and the 2025-04-01 calibration.

    The 96 ordered sequences and transition-state model archive are in `supplemental_data/`; see the [dataset overview](README.md).
    """)
    return


if __name__ == "__main__":
    app.run()
