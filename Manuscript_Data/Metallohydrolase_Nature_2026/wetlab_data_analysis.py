import marimo

__generated_with = "0.25.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Metallohydrolase Nature 2026: wetlab analysis

    Recalculate screening, calibration, kinetics, zinc dependence, turnover and circular dichroism from the deposited measurements. Cells run in dependency order. Figure export settings are below.
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import os, sys
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from scipy.stats import linregress
    _start = Path(os.environ.get("ZINC_HYDRO_REPO") or __file__).resolve()
    github_repo_dir = next(p for p in (_start, *_start.parents) if (p / "Scripts").is_dir() and (p / "LICENSE").exists())
    working_dir = github_repo_dir / "Manuscript_Data" / "Metallohydrolase_Nature_2026"
    raw_wetlab_data_dir = str(working_dir / "raw_wetlab_data") + "/"
    wetlab_data_plots_dir = str(working_dir / "wetlab_data_plots") + "/"
    Path(wetlab_data_plots_dir).mkdir(exist_ok=True)
    sys.path.insert(0, str(github_repo_dir / "Scripts"))
    from experimental_data_processing_functions import draw_progression_curve_seperate_v0, draw_progression_curve_v0, make_active_wells, make_rename_dic, make_transpose_dic, mm_kinetics, parse_kinetics, plot_CD_222nm_temperature_interval, plot_CD_spectrum_temperature_interval, plot_CD_spectrum_temperature_interval_A1, save_figure, standard_curve
    import experimental_data_processing_functions as edpf
    FIGURE_DPI = 150
    SAVE_EPS = False
    edpf.FIGURE_DPI, edpf.SAVE_EPS = FIGURE_DPI, SAVE_EPS
    print("Inputs: raw_wetlab_data/\nFigures: wetlab_data_plots/")
    return (
        FIGURE_DPI,
        SAVE_EPS,
        draw_progression_curve_seperate_v0,
        draw_progression_curve_v0,
        make_active_wells,
        make_rename_dic,
        make_transpose_dic,
        mm_kinetics,
        np,
        os,
        parse_kinetics,
        pd,
        plot_CD_222nm_temperature_interval,
        plot_CD_spectrum_temperature_interval,
        plot_CD_spectrum_temperature_interval_A1,
        plt,
        raw_wetlab_data_dir,
        save_figure,
        standard_curve,
        wetlab_data_plots_dir,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **I. FUNCTIONAL SCREENING DATA**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## I.I. Design Campaign 1 Screening Results
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### I.I.A. 96-Well Grid Plot of Individual Reaction Progress Curves
    """)
    return


@app.cell
def _(
    FIGURE_DPI,
    SAVE_EPS,
    np,
    pd,
    plt,
    raw_wetlab_data_dir,
    save_figure,
    wetlab_data_plots_dir,
):
    # Load the CSV file
    _file_path1 = f'{raw_wetlab_data_dir}240709_4mu_PA_ABCD.tsv'
    _data1 = pd.read_csv(_file_path1, sep='\t')
    _data1_rename_dic = {'Time': 'Time', 'T° 365,445': 'T° 365,445'}
    for _i, _new_row in enumerate('AABBCCDD'):
        _ori_row = 'ABCDEFGH'[_i]
        for _col in range(1, 13):
            if _i % 2 == 0:
                _data1_rename_dic[f'{_ori_row}{_col}'] = f'{_new_row}{_col}_wZN'
            else:
                _data1_rename_dic[f'{_ori_row}{_col}'] = f'{_new_row}{_col}_woZN'
    _data1.rename(columns=_data1_rename_dic, inplace=True)
    _file_path2 = f'{raw_wetlab_data_dir}240711_4mu_PA_EFGH.tsv'
    _data2 = pd.read_csv(_file_path2, sep='\t')
    _data2_rename_dic = {'Time': 'Time', 'T° 365,445': 'T° 365,445'}
    for _i, _new_row in enumerate('EEFFGGHH'):
        _ori_row = 'ABCDEFGH'[_i]
        for _col in range(1, 13):
            if _i % 2 == 0:
                _data2_rename_dic[f'{_ori_row}{_col}'] = f'{_new_row}{_col}_wZN'
            else:
                _data2_rename_dic[f'{_ori_row}{_col}'] = f'{_new_row}{_col}_woZN'
    _data2_rename_dic['E6'] = 'G6_woZN'
    _data2_rename_dic['F6'] = 'G6_wZN'
    ### PLATE RELABELING ###
    # Wells E6 and F6 hold the G6 sample pair (without and with Zn(II));
    # remap them to their correct identities before analysis.
    _data1.rename(columns=_data1_rename_dic, inplace=True)
    _data2.rename(columns=_data2_rename_dic, inplace=True)
    ### END PLATE RELABELING ###
    data = pd.merge(_data1, _data2, on='Time', how='inner')
    data['Time'] = pd.to_timedelta(data['Time'])
    data['Time'] = data['Time'].dt.total_seconds() / 60
    for _letter in 'ABCDEFGH':
        for _num in range(1, 13):
            data[f'{_letter}{_num}_wZN'] = data[f'{_letter}{_num}_wZN'] - np.min(data[f'{_letter}{_num}_wZN'].to_list())
    # Convert 'Time' to a more usable format (assuming it's in hh:mm:ss)
            data[f'{_letter}{_num}_woZN'] = data[f'{_letter}{_num}_woZN'] - np.min(data[f'{_letter}{_num}_woZN'].to_list())
    _pairs = []  # Convert time to minutes
    for _letter in 'ABCDEFGH':
    # Standardization and Normalization
        for _num in range(1, 13):
            _pairs.append((f'{_letter}{_num}_wZN', f'E5_wZN'))
    _y_min = data[[col for pair in _pairs for col in pair]].min().min()  #data[f'{letter}{num}_wZN'] = data[f'{letter}{num}_wZN'] - data[f'{blank_well}_wZN']
    _y_max = data[[col for pair in _pairs for col in pair]].max().max()
    num_plots = len(_pairs)  #data[f'{letter}{num}_woZN'] = data[f'{letter}{num}_woZN'] - data[f'{blank_well}_woZN']
    _cols = 12
    _rows = (num_plots + _cols - 1) // _cols
    _fig, _axes = plt.subplots(_rows, _cols, figsize=(4 * _cols, 4 * _rows))
    # Define pairs for plotting
    _axes = _axes.flatten()
    _custom_titles = {'B6': 'PC1', 'B7': 'PC2', 'D7': 'Blank (rxn buffer only)', 'E5': 'Blank (rxn buffer only)', 'F8': 'C4', 'G4': 'PC1', 'G6': 'PC2', 'H2': 'Blank (rxn buffer only)'}
    for _i, (col1, col2) in enumerate(_pairs):
        ax = _axes[_i]
        ax.plot(data['Time'], data[col1], color='red')
    # Find global min and max values for normalization
        ax.plot(data['Time'], data[col2], color='black')
        ax.set_ylim(_y_min, _y_max)
    for j in range(_i + 1, len(_axes)):
    # Set up the plot grid
        _fig.delaxes(_axes[j])
    cute_pics_dir = wetlab_data_plots_dir
    plt.tight_layout()
    save_figure(f'{cute_pics_dir}240709_4mu_PA_screening', dpi=FIGURE_DPI, save_eps=SAVE_EPS)
    # Custom titles and spine colors
    # Plot each pair
    # Remove any empty subplots
    plt.show()  # Plot the data  # Set axis labels and title  #ax.set_xlabel('Time (minutes)')  #ax.set_ylabel('Fluorescence Units')  # PNG always; EPS only if SAVE_EPS. Raise dpi here for one figure.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### I.I.B. Combined Plot of Reaction Progress Curves
    """)
    return


@app.cell
def _(
    FIGURE_DPI,
    SAVE_EPS,
    np,
    pd,
    plt,
    raw_wetlab_data_dir,
    save_figure,
    wetlab_data_plots_dir,
):
    _file_path1 = f'{raw_wetlab_data_dir}240709_4mu_PA_ABCD.tsv'
    _data1 = pd.read_csv(_file_path1, sep='\t')
    _data1_rename_dic = {'Time': 'Time', 'T° 365,445': 'T° 365,445'}
    for _i, _new_row in enumerate('AABBCCDD'):
        _ori_row = 'ABCDEFGH'[_i]
        for _col in range(1, 13):
            if _i % 2 == 0:
                _data1_rename_dic[f'{_ori_row}{_col}'] = f'{_new_row}{_col}_wZN'
            else:
                _data1_rename_dic[f'{_ori_row}{_col}'] = f'{_new_row}{_col}_woZN'
    _data1.rename(columns=_data1_rename_dic, inplace=True)
    _file_path2 = f'{raw_wetlab_data_dir}240711_4mu_PA_EFGH.tsv'
    _data2 = pd.read_csv(_file_path2, sep='\t')
    _data2_rename_dic = {'Time': 'Time', 'T° 365,445': 'T° 365,445'}
    for _i, _new_row in enumerate('EEFFGGHH'):
        _ori_row = 'ABCDEFGH'[_i]
        for _col in range(1, 13):
            if _i % 2 == 0:
                _data2_rename_dic[f'{_ori_row}{_col}'] = f'{_new_row}{_col}_wZN'
            else:
                _data2_rename_dic[f'{_ori_row}{_col}'] = f'{_new_row}{_col}_woZN'
    _data2_rename_dic['E6'] = 'G6_woZN'
    _data2_rename_dic['F6'] = 'G6_wZN'
    _data1.rename(columns=_data1_rename_dic, inplace=True)
    _data2.rename(columns=_data2_rename_dic, inplace=True)
    data_1 = pd.merge(_data1, _data2, on='Time', how='inner')
    data_1['Time'] = pd.to_timedelta(data_1['Time'])
    data_1['Time'] = data_1['Time'].dt.total_seconds() / 60
    for _letter in 'ABCDEFGH':
        for _num in range(1, 13):
            data_1[f'{_letter}{_num}_wZN'] = data_1[f'{_letter}{_num}_wZN'] - np.min(data_1[f'{_letter}{_num}_wZN'].to_list())
            data_1[f'{_letter}{_num}_woZN'] = data_1[f'{_letter}{_num}_woZN'] - np.min(data_1[f'{_letter}{_num}_woZN'].to_list())
    _pairs = []
    for _letter in 'ABCDEFGH':
        for _num in range(1, 13):
            _pairs.append((f'{_letter}{_num}_wZN', f'E5_wZN'))
    _y_min = data_1[[col for pair in _pairs for col in pair]].min().min()
    _y_max = data_1[[col for pair in _pairs for col in pair]].max().max()
    plt.figure(figsize=(6, 5.5))
    _custom_titles = {'B6': 'PC1', 'B7': 'PC2', 'D7': 'Blank (rxn buffer only)', 'E5': 'Blank (rxn buffer only)', 'F8': 'C4', 'G4': 'PC1', 'G6': 'PC2', 'H2': 'Blank (rxn buffer only)'}
    for _col in range(1, 13):
        for _row in 'ABCDEFGH':
            _well = f'{_row}{_col}'
            if _well in ['B6', 'B7', 'D7', 'F8', 'G4', 'G6', 'H2']:
                continue
            if _well in ['A1', 'A8', 'B9', 'C4', 'F7']:
                plt.plot(data_1['Time'], data_1[_well + '_wZN'], label=_well, color='red')
            elif _well in ['A2', 'A5', 'D9', 'E3', 'E8', 'F2', 'F3', 'G7', 'H5']:
                plt.plot(data_1['Time'], data_1[_well + '_wZN'], label=_well, color='pink')
            elif _well in ['H7', 'H8']:
                plt.plot(data_1['Time'], data_1[_well + '_wZN'], label=_well, color='black')
            elif not _well in ['A1', 'A8', 'B9', 'C4', 'F7', 'E5']:
                plt.plot(data_1['Time'], data_1[_well + '_wZN'], label=_well, color='gray')
            elif _well == 'E5':
                plt.plot(data_1['Time'], data_1[_well + '_wZN'], label=_well, color='black', linestyle='dashed')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Fluorescence Units')
    plt.tight_layout()
    save_figure(f'{wetlab_data_plots_dir}241006_4muPA_integrated_screening_results', dpi=FIGURE_DPI, save_eps=SAVE_EPS)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## I.II. Design Campaign 2 Screening Results
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### I.II.A. 96-Well Grid Plot of Individual Reaction Progress Curves
    """)
    return


@app.cell
def _(
    FIGURE_DPI,
    SAVE_EPS,
    draw_progression_curve_seperate_v0,
    raw_wetlab_data_dir,
    wetlab_data_plots_dir,
):
    ### Input
    _excel_file = f'{raw_wetlab_data_dir}250307_singleZn_order1_cowboy_elutantScreen_100uM_4MUPA_and_100uM_suppZn_25C.xlsx'
    _eps_out = f'{wetlab_data_plots_dir}250307_screening_result_100uM_4MUPA_100uM_suppZn_25C_separated'
    _rows, _columns = (list('ABCDEFGH'), list(range(1, 13)))
    bg = 'A5'
    _time_start = 0
    _time_end = 60
    # Scaffold1: ["F2", "F8", "F11", "H1", "H10"]
    # Scaffold2: ["B5", "B10", "D9", "F12", "H6"]
    # Scaffold3: ["C5"]
    draw_progression_curve_seperate_v0(_excel_file, _rows, _columns, bg, _time_start, _time_end, start_row=48, reverse=False, save=_eps_out, dpi=FIGURE_DPI, save_eps=SAVE_EPS)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### I.II.B. Combined Plot of Reaction Progress Curves
    """)
    return


@app.cell
def _(
    FIGURE_DPI,
    SAVE_EPS,
    draw_progression_curve_v0,
    raw_wetlab_data_dir,
    wetlab_data_plots_dir,
):
    ### Input
    _excel_file = f'{raw_wetlab_data_dir}250307_singleZn_order1_cowboy_elutantScreen_100uM_4MUPA_and_100uM_suppZn_25C.xlsx'
    _eps_out = f'{wetlab_data_plots_dir}250307_screening_result_100uM_4MUPA_100uM_suppZn_25C'
    _rows, _columns = (list('ABCDEFGH'), list(range(1, 13)))
    hit_definition_num = 13
    blank = []
    _time_start = 0
    _time_end = 60
    ignore = ['A5', 'A8', 'B6', 'E8', 'F4', 'G2', 'G7', 'G10', 'G11', 'H7', 'H9']
    # Scaffold1: ["F2", "F8", "F11", "H1", "H10"]
    # Scaffold2: ["B5", "B10", "D9", "F12", "H6"]
    # Scaffold3: ["C5"]
    draw_progression_curve_v0(_excel_file, _rows, _columns, hit_definition_num, blank, _time_start, _time_end, label=False, start_row=48, ignore=ignore, reverse=False, save=_eps_out, dpi=FIGURE_DPI, save_eps=SAVE_EPS)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## I.III. ZETA_1 (A1) Variant Screening Results
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### I.III.A. Combined Plot of Reaction Progress Curves
    """)
    return


@app.cell
def _(
    FIGURE_DPI,
    SAVE_EPS,
    pd,
    plt,
    raw_wetlab_data_dir,
    save_figure,
    wetlab_data_plots_dir,
):
    _file_path1 = f'{raw_wetlab_data_dir}240826_zn_hydrolase_donghyo_knockouts_AND_variants_200uM_zn_25uM_PA_AND_100uM_PA.xlsx'
    _data1 = pd.read_excel(_file_path1, header=48, skipfooter=36)
    data_2 = _data1.copy()
    data_2 = data_2.dropna(axis=1, how='all').dropna(axis=0, how='all')
    data_2.reset_index(drop=True, inplace=True)
    data_2['Time'] = pd.to_datetime(data_2['Time'], format='%H:%M:%S', errors='coerce')
    data_2['Time'] = (data_2['Time'] - pd.Timestamp('1900-01-01 00:00:00')).dt.total_seconds() / 60
    plt.figure(figsize=(6, 5.5))
    active_wells = ['A1', 'E11', 'E12', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7']
    for _well in active_wells:
        data_2[_well] = data_2[_well] - data_2[_well].to_list()[0]
    for _i, _well in enumerate(active_wells):
        if _i == 0:
            plt.plot(data_2['Time'], data_2[_well], label=_well)
        elif _i < 8:
            plt.plot(data_2['Time'], data_2[_well], label=f'A1_var{_i}')
        elif _i == 8:
            plt.plot(data_2['Time'], data_2[_well], label=f'A1_par')
        else:
            plt.plot(data_2['Time'], data_2[_well], label='Blank', linestyle='dashed', color='black')
        plt.xlabel('Time (minutes)')
        xlim = plt.xlim()
        plt.xlim(xlim[0], 16)
        plt.ylabel('Fluorescence Units')
    plt.tight_layout()
    plt.legend()
    save_figure(f'{wetlab_data_plots_dir}240826_A1_variations_progression_curves', dpi=FIGURE_DPI, save_eps=SAVE_EPS)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## I.IV. ZETA_1 (A1) Mutant (Knockout) Screening Results
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### I.IV.A. Combined Plot of Reaction Progress Curves
    """)
    return


@app.cell
def _(
    FIGURE_DPI,
    SAVE_EPS,
    np,
    parse_kinetics,
    plt,
    raw_wetlab_data_dir,
    save_figure,
    wetlab_data_plots_dir,
):
    _excel_file = f'{raw_wetlab_data_dir}240906_a1_plus_6mutants_AND_standard_curve___1pt8_uM_4muPA_100nM_enz_25C_1hr.xlsx'
    _row_column_ranges = {'A': (1, 7), 'B': (1, 7), 'C': (1, 7), 'D': (1, 7), 'E': (1, 7), 'F': (1, 7), 'G': (1, 7)}
    replicate_dic = {1: [2, 3]}
    _background_col = 4
    active_wells_1 = []
    for _row, (_start_col, _end_col) in _row_column_ranges.items():
        for _col in range(_start_col, _end_col + 1):
            active_wells_1.append(f'{_row}{_col}')
    _slope = np.average([6.857301209337525e-10, 7.101340604738747e-10, 7.18193078001004e-10])
    screening_data = parse_kinetics(_excel_file, active_wells_1, start_row=48)
    '\n# Standardize\nfor well in active_wells:\n    fu = screening_data[well].values\n    product_concentration = fu * slope * 1000000\n    screening_data[well] = product_concentration\n'
    for _well in active_wells_1:
        screening_data[_well] = screening_data[_well] - np.min(screening_data[_well])
    "\n# Normalize\nfor well in active_wells:\n    row = well[0]\n    screening_data[well] = np.array(screening_data[well]) - np.array(screening_data[f'{row}{background_col}'])\n"
    screening_data['Time'] = screening_data['Time'] / 60
    plt.figure(figsize=(6, 5.5))
    for _representive_col in replicate_dic:
        _replicates = [_representive_col]
        _replicates.extend(replicate_dic[_representive_col])
        for _row, _name in zip(['A', 'B', 'C', 'D', 'E', 'F', 'G'], ['WT', 'H118A;H130A;H134A', 'H130A', 'H134A', 'H118A', 'N17A', 'D67A']):
            _replicates_well = [f'{_row}{col}' for col in _replicates]
            _avg_product_concentration = np.array(screening_data[_replicates_well].mean(axis=1).tolist())
            _std_product_concentration = np.array(screening_data[_replicates_well].std(axis=1).tolist())
            plt.plot(screening_data['Time'].values, _avg_product_concentration, label=_name)
            plt.fill_between(screening_data['Time'].values, _avg_product_concentration - _std_product_concentration, _avg_product_concentration + _std_product_concentration, alpha=0.45)
        plt.plot(screening_data['Time'].values, screening_data[f'{_row}{_background_col}'], linestyle='dashed', label='background')
        plt.ylabel('Fluorescence Units')
        plt.xlabel('Time (min)')
    plt.legend()
    plt.tight_layout()
    save_figure(f'{wetlab_data_plots_dir}A1_KO_experiment_progression_curves', dpi=FIGURE_DPI, save_eps=SAVE_EPS)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **II. KINETIC MEASUREMENT DATA**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## II.I. Standard Curves & Background Reaction Rate (*k*<sub>uncat</sub>)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### II.I.A. 4-Methylumbelliferone (4MU) Standard Curve

    Every kinetic measurement in this notebook is recorded as raw fluorescence units
    (FU). Converting FU to product concentration requires a calibration slope, in
    M per FU, measured from a dilution series of free 4-methylumbelliferone read on
    the same instrument under the same gain settings.

    The cell below derives the three calibration slopes this notebook uses, each
    from its own standard-curve plate in `raw_wetlab_data/`:

    | Constant | Plate | Applied to |
    |---|---|---|
    | `SLOPE_4MU_250401` | `250401_4mu_standard_curves_..._pH8` | the Michaelis-Menten series (Sections II and IV) |
    | `SLOPE_4MU_240927` | `240927_4mu_standard_curves_...LadderLEFT3_100uMLadderRIGHT3` | Section II.IV.C, A1 H130A |
    | `SLOPE_4MU_240906` | `240906_a1_plus_6mutants_AND_standard_curve_...` | Section I.IV, A1 knockout screen |

    Each plate carries two serial dilution ladders, fitted together as a single
    twelve-point curve and repeated in triplicate; the three replicate slopes are
    averaged. On the 250401 plate the ladders run across columns (30 uM in 1-6,
    45 uM in 7-12, rows D/E/F). On the 240927 plate they run down the rows (80 uM in
    plate columns 1-3, 100 uM in columns 4-6), so that plate is transposed before
    fitting.

    Instrument gain differs between sessions, so each calibration is applied only to
    the experiments recorded alongside it. Defining them here means a calibration is
    changed in one place rather than in every analysis cell.
    """)
    return


@app.cell
def _(np, parse_kinetics, raw_wetlab_data_dir, standard_curve):
    standard_curve_file = f'{raw_wetlab_data_dir}240906_a1_plus_6mutants_AND_standard_curve___1pt8_uM_4muPA_100nM_enz_25C_1hr.xlsx'
    standard_rows = 'ABC'
    standard_col_range = (1, 7)
    pro_concs = [2, 1.5, 1, 0.75, 0.5, 0.375, 0.1875]
    Max_FU = 100000
    cutoff_start_time = 0
    cutoff_end_time = 50000
    _rows, _cols = (list('ABCDEFG'), [1, 2, 3, 4, 5, 6, 7])
    _transpose_dic = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 5: 'A', 6: 'B', 7: 'C', 1: 'D', 2: 'E', 3: 'F', 4: 'G'}
    _rename_dic = {}
    for _key, _value in _transpose_dic.items():
        if _key in _rows:
            for _col in _cols:
                _rename_dic[f'{_key}{_col}'] = f'{_transpose_dic[_col]}{_value}'
        elif _key in _cols:
            for _row in _rows:
                _rename_dic[f'{_row}{_key}'] = f'{_value}{_transpose_dic[_row]}'
    active_wells_2 = [f'{r}{c}' for r in standard_rows for c in range(standard_col_range[0], standard_col_range[1] + 1)]
    standard_data = parse_kinetics(standard_curve_file, active_wells_2, start_row=48)
    standard_data = standard_data.rename(columns=_rename_dic)
    derived_slopes = []
    for standard_row in standard_rows:
        print(f'--- replicate row {standard_row} ---')
        _, _, _slope = standard_curve(standard_data, standard_row, standard_col_range, Max_FU, pro_concs, cutoff_start_time, cutoff_end_time)
        derived_slopes.append(_slope)
    SLOPE_4MU_DERIVED = np.average(derived_slopes)
    print(f'\nDerived from the shipped plate : {SLOPE_4MU_DERIVED:.6e} M / FU')
    slope_250401_file = f'{raw_wetlab_data_dir}250401_4mu_standard_curves_100uL_30uMladder_and_45uMladder_25C_5prctDMSO__pH8.xlsx'
    slope_250401_rows = 'DEF'
    slope_250401_cols = (1, 12)
    slope_250401_pro_concs = [30, 15, 7.5, 3.75, 1.875, 0.9375, 45, 22.5, 11.25, 5.625, 2.8125, 1.40625]
    slope_250401_max_fu = 250000
    _active = [f'{r}{c}' for r in slope_250401_rows for c in range(slope_250401_cols[0], slope_250401_cols[1] + 1)]
    _kd_250401 = parse_kinetics(slope_250401_file, _active, start_row=48)
    _slopes_250401 = []
    for _row in slope_250401_rows:
        print(f'--- 250401 replicate row {_row} ---')
        _, _, _s = standard_curve(_kd_250401, _row, slope_250401_cols, slope_250401_max_fu, slope_250401_pro_concs, cutoff_start_time, cutoff_end_time)
        _slopes_250401.append(_s)
    SLOPE_4MU_250401 = np.average(_slopes_250401)
    print(f'\nSLOPE_4MU_250401 (derived) = {SLOPE_4MU_250401:.6e} M / FU')
    slope_240927_file = f'{raw_wetlab_data_dir}240927_4mu_standard_curves_25C_5dmso_95akta_80uMLadderLEFT3_100uMLadderRIGHT3.xlsx'
    slope_240927_pro_concs = [80, 40, 20, 10, 5, 2.5, 100, 50, 25, 12.5, 6.25, 3.125]
    slope_240927_max_fu = 100000
    _tp = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 1: 'A', 2: 'B', 3: 'C'}
    _rename = {}
    for _k, _v in _tp.items():
        if _k in list('ABCDEF'):
            for _c in (1, 2, 3):
                _rename[f'{_k}{_c}'] = f'{_tp[_c]}{_v}'
        elif _k in (1, 2, 3):
            for _r in 'ABCDEF':
                _rename[f'{_r}{_k}'] = f'{_v}{_tp[_r]}'
    for _c in (4, 5, 6):
        for _r in 'ABCDEF':
            _rename[f'{_r}{_c}'] = f"{'ABC'[_c - 4]}{_tp[_r] + 6}"
    _active = [f'{r}{c}' for r in 'ABC' for c in range(1, 13)]
    _kd_240927 = parse_kinetics(slope_240927_file, _active, start_row=48).rename(columns=_rename)
    _slopes_240927 = []
    for _row in 'ABC':
        print(f'--- 240927 replicate {_row} ---')
        _, _, _s = standard_curve(_kd_240927, _row, (1, 12), slope_240927_max_fu, slope_240927_pro_concs, cutoff_start_time, cutoff_end_time)
        _slopes_240927.append(_s)
    SLOPE_4MU_240927 = np.average(_slopes_240927)
    print(f'\nSLOPE_4MU_240927 (derived) = {SLOPE_4MU_240927:.6e} M / FU')
    SLOPE_4MU_240906 = np.average([6.857301209337525e-10, 7.101340604738747e-10, 7.18193078001004e-10])
    print(f'Re-derived vs deposited 240906  : {SLOPE_4MU_DERIVED:.6e} vs {SLOPE_4MU_240906:.6e} ({100 * abs(SLOPE_4MU_DERIVED - SLOPE_4MU_240906) / SLOPE_4MU_240906:.2f} % difference)')
    print(f'Main kinetics calibration       : {SLOPE_4MU_250401:.6e} M / FU')
    print(f'240927 calibration (II.IV.C)    : {SLOPE_4MU_240927:.6e} M / FU')
    return SLOPE_4MU_240927, SLOPE_4MU_250401


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### II.I.B. Background Reaction Rate for 4MU-PA Hydrolysis (*k*<sub>uncat</sub>)
    """)
    return


@app.cell
def _(
    FIGURE_DPI,
    SAVE_EPS,
    SLOPE_4MU_250401,
    make_active_wells,
    make_rename_dic,
    make_transpose_dic,
    mm_kinetics,
    np,
    os,
    parse_kinetics,
    raw_wetlab_data_dir,
    wetlab_data_plots_dir,
):
    _excel_file = f'{raw_wetlab_data_dir}250415_BLANK_kuncat_experiment__phenylacetate_subst_25C_1hour_no_zn_144uMsubstLadder.xlsx'
    _pics_dir = wetlab_data_plots_dir
    _slope = SLOPE_4MU_250401
    _enz_conc = 1000000.0
    _rows, _cols = (list('ABCDEFGH'), [1, 2, 3, 4])
    replicate_dic_1 = {1: [2, 3]}
    _bg_col = 4
    _sub_concs = np.array([72, 36, 18, 9, 4.5, 2.3, 1.1])
    _col_range_to_analyze = (2, 8)
    _start_time = 30
    _end_time = 500
    _substrate_name = '4MU-PA'
    _transpose_dic = make_transpose_dic(_rows, _cols)
    _rename_dic = make_rename_dic(_transpose_dic, _rows, _cols)
    _represent_row = 'A'
    _replicate_rows = ['ABCDEFGHIJKL'[_cols.index(rep)] for rep in list(replicate_dic_1.values())[0]]
    _bg_row = 'ABCDEFGHIJKL'[_cols.index(_bg_col)]
    _row_column_ranges = {_represent_row: _col_range_to_analyze}
    active_wells_3 = make_active_wells(_row_column_ranges)
    _bn = os.path.basename(f'{_excel_file}').replace('.xlsx', '')
    _kinetic_data = parse_kinetics(_excel_file, active_wells_3, start_row=48)
    _kinetic_data.rename(columns=_rename_dic, inplace=True)
    _v0_enz = mm_kinetics(_kinetic_data, _bn, _rows, _cols, active_wells_3, _bg_row, _enz_conc, _sub_concs, _substrate_name, _pics_dir, _replicate_rows, fit_type='straight_line', slope=_slope, cutoff_start_time=_start_time, cutoff_end_time=_end_time, plate_type='full', y_ax_mm_intv=0.1, x_ax_mm_intv=50, legend_label=f'[{_substrate_name}], μM', norm_zero=True, silent=True, ylabel='vx10$^{-3}$ (s$^{-1}$)', dpi=FIGURE_DPI, save_eps=SAVE_EPS)
    v0_list, sub_conc_list = ([], [])
    for _i, sub_conc in enumerate(_sub_concs):
        v0_list.append(_v0_enz[_i] * 1e-06 * _enz_conc)
        sub_conc_list.append(sub_conc * 1e-06)
    _coefficients = np.polyfit(sub_conc_list, v0_list, 1)
    print(f'kuncat: {_coefficients[0]}')
    print(f'coefficients: {_coefficients}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## II.II. Michaelis-Menten Kinetics for Top 5 Design Campaign 1 Hits (5 Unique RFdiffusion2 Scaffolds)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### II.II.A. 241008 ZETA_1 (A1) Kinetics ([E]<sub>0</sub> = 100nM; [Zn(II)] = 4μM; Temp = 25C)
    """)
    return


@app.cell
def _(
    FIGURE_DPI,
    SAVE_EPS,
    SLOPE_4MU_250401,
    make_active_wells,
    make_rename_dic,
    make_transpose_dic,
    mm_kinetics,
    np,
    os,
    parse_kinetics,
    raw_wetlab_data_dir,
    wetlab_data_plots_dir,
):
    _excel_file = f'{raw_wetlab_data_dir}241008_A1_FINAL_kinetics_100nM___phenylacetate_subst_25C_5min_40x_zn_144uMsubstLadder.xlsx'
    _pics_dir = wetlab_data_plots_dir
    _slope = SLOPE_4MU_250401
    _enz_conc = 0.1
    _rows, _cols = (list('ABCDEFGH'), [1, 2, 3, 4])
    replicate_dic_2 = {1: [2, 3]}
    _bg_col = 4
    _sub_concs = np.array([72, 36, 18, 9, 4.5, 2.3, 1.1])
    _col_range_to_analyze = (2, 8)
    _start_time = 0
    _end_time = 30
    _substrate_name = '4MU-PA'
    _transpose_dic = make_transpose_dic(_rows, _cols)
    _rename_dic = make_rename_dic(_transpose_dic, _rows, _cols)
    _represent_row = 'A'
    _replicate_rows = ['ABCDEFGHIJKL'[_cols.index(rep)] for rep in list(replicate_dic_2.values())[0]]
    _bg_row = 'ABCDEFGHIJKL'[_cols.index(_bg_col)]
    _row_column_ranges = {_represent_row: _col_range_to_analyze}
    active_wells_4 = make_active_wells(_row_column_ranges)
    _bn = os.path.basename(f'{_excel_file}').replace('.xlsx', '')
    _kinetic_data = parse_kinetics(_excel_file, active_wells_4, start_row=48)
    _kinetic_data.rename(columns=_rename_dic, inplace=True)
    _v0_enz = mm_kinetics(_kinetic_data, _bn, _rows, _cols, active_wells_4, _bg_row, _enz_conc, _sub_concs, _substrate_name, _pics_dir, _replicate_rows, fit_type='straight_line', slope=_slope, cutoff_start_time=_start_time, cutoff_end_time=_end_time, plate_type='full', y_ax_mm_intv=0.1, x_ax_mm_intv=50, legend_label=f'[{_substrate_name}], μM', norm_zero=True, dpi=FIGURE_DPI, save_eps=SAVE_EPS)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### II.II.B. 240830 A8 Kinetics ([E]<sub>0</sub> = 2μM; [Zn(II)] = 20μM; Temp = 25C)
    """)
    return


@app.cell
def _(
    FIGURE_DPI,
    SAVE_EPS,
    SLOPE_4MU_250401,
    make_active_wells,
    make_rename_dic,
    make_transpose_dic,
    mm_kinetics,
    np,
    os,
    parse_kinetics,
    raw_wetlab_data_dir,
    wetlab_data_plots_dir,
):
    _excel_file = f'{raw_wetlab_data_dir}240830_A8_OFFICIAL_kinetics_2uM___phenylacetate_subst_25C_4min_10x_zn.xlsx'
    _pics_dir = wetlab_data_plots_dir
    _slope = SLOPE_4MU_250401
    _enz_conc = 2.0
    _rows, _cols = (list('ABCDEFGH'), [1, 2, 3, 4])
    replicate_dic_3 = {1: [2, 3]}
    _bg_col = 4
    _sub_concs = np.array([144, 72, 36, 18, 9, 4.5, 2.3, 1.1])
    _col_range_to_analyze = (1, 8)
    _start_time = 0
    _end_time = 240
    _substrate_name = '4MU-PA'
    _transpose_dic = make_transpose_dic(_rows, _cols)
    _rename_dic = make_rename_dic(_transpose_dic, _rows, _cols)
    _represent_row = 'A'
    _replicate_rows = ['ABCDEFGHIJKL'[_cols.index(rep)] for rep in list(replicate_dic_3.values())[0]]
    _bg_row = 'ABCDEFGHIJKL'[_cols.index(_bg_col)]
    _row_column_ranges = {_represent_row: _col_range_to_analyze}
    active_wells_5 = make_active_wells(_row_column_ranges)
    _bn = os.path.basename(f'{_excel_file}').replace('.xlsx', '')
    _kinetic_data = parse_kinetics(_excel_file, active_wells_5, start_row=48)
    _kinetic_data.rename(columns=_rename_dic, inplace=True)
    _v0_enz = mm_kinetics(_kinetic_data, _bn, _rows, _cols, active_wells_5, _bg_row, _enz_conc, _sub_concs, _substrate_name, _pics_dir, _replicate_rows, fit_type='straight_line', slope=_slope, cutoff_start_time=_start_time, cutoff_end_time=_end_time, plate_type='full', y_ax_mm_intv=0.1, x_ax_mm_intv=50, legend_label=f'[{_substrate_name}], μM', norm_zero=True, dpi=FIGURE_DPI, save_eps=SAVE_EPS)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### II.II.C. 240830 B9 Kinetics ([E]<sub>0</sub> = 2μM; [Zn(II)] = 20μM; Temp = 25C)
    """)
    return


@app.cell
def _(
    FIGURE_DPI,
    SAVE_EPS,
    SLOPE_4MU_250401,
    make_active_wells,
    make_rename_dic,
    make_transpose_dic,
    mm_kinetics,
    np,
    os,
    parse_kinetics,
    raw_wetlab_data_dir,
    wetlab_data_plots_dir,
):
    _excel_file = f'{raw_wetlab_data_dir}240830_B9_OFFICIAL_kinetics_2uM___phenylacetate_subst_25C_4min_10x_zn.xlsx'
    _pics_dir = wetlab_data_plots_dir
    _slope = SLOPE_4MU_250401
    _enz_conc = 2.0
    _rows, _cols = (list('ABCDEFGH'), [5, 6, 7, 8])
    replicate_dic_4 = {5: [6, 7]}
    _bg_col = 8
    _sub_concs = np.array([144, 72, 36, 18, 9, 4.5, 2.3, 1.1])
    _col_range_to_analyze = (1, 8)
    _start_time = 0
    _end_time = 240
    _substrate_name = '4MU-PA'
    _transpose_dic = make_transpose_dic(_rows, _cols)
    _rename_dic = make_rename_dic(_transpose_dic, _rows, _cols)
    _represent_row = 'A'
    _replicate_rows = ['ABCDEFGHIJKL'[_cols.index(rep)] for rep in list(replicate_dic_4.values())[0]]
    _bg_row = 'ABCDEFGHIJKL'[_cols.index(_bg_col)]
    _row_column_ranges = {_represent_row: _col_range_to_analyze}
    active_wells_6 = make_active_wells(_row_column_ranges)
    _bn = os.path.basename(f'{_excel_file}').replace('.xlsx', '')
    _kinetic_data = parse_kinetics(_excel_file, active_wells_6, start_row=48)
    _kinetic_data.rename(columns=_rename_dic, inplace=True)
    _v0_enz = mm_kinetics(_kinetic_data, _bn, _rows, _cols, active_wells_6, _bg_row, _enz_conc, _sub_concs, _substrate_name, _pics_dir, _replicate_rows, fit_type='straight_line', slope=_slope, cutoff_start_time=_start_time, cutoff_end_time=_end_time, plate_type='full', y_ax_mm_intv=0.1, x_ax_mm_intv=50, legend_label=f'[{_substrate_name}], μM', norm_zero=True, dpi=FIGURE_DPI, save_eps=SAVE_EPS)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### II.II.D. 240927 C4 Kinetics ([E]<sub>0</sub> = 3μM; [Zn(II)] = 10μM; Temp = 25C)
    """)
    return


@app.cell
def _(
    FIGURE_DPI,
    SAVE_EPS,
    SLOPE_4MU_250401,
    make_active_wells,
    make_rename_dic,
    make_transpose_dic,
    mm_kinetics,
    np,
    os,
    parse_kinetics,
    raw_wetlab_data_dir,
    wetlab_data_plots_dir,
):
    _excel_file = f'{raw_wetlab_data_dir}240927_C4_FINAL_kinetics_3uM___phenylacetate_subst_25C_5min_10uM_zn_144uMsubstLadder.xlsx'
    _pics_dir = wetlab_data_plots_dir
    _slope = SLOPE_4MU_250401
    _enz_conc = 3.0
    _rows, _cols = (list('ABCDEFGH'), [9, 10, 11, 12])
    replicate_dic_5 = {9: [10, 11]}
    _bg_col = 12
    _sub_concs = np.array([144, 72, 36, 18, 9, 4.5, 2.3, 1.1])
    _col_range_to_analyze = (1, 8)
    _start_time = 0
    _end_time = 600
    _substrate_name = '4MU-PA'
    _transpose_dic = make_transpose_dic(_rows, _cols)
    _rename_dic = make_rename_dic(_transpose_dic, _rows, _cols)
    _represent_row = 'A'
    _replicate_rows = ['ABCDEFGHIJKL'[_cols.index(rep)] for rep in list(replicate_dic_5.values())[0]]
    _bg_row = 'ABCDEFGHIJKL'[_cols.index(_bg_col)]
    _row_column_ranges = {_represent_row: _col_range_to_analyze}
    active_wells_7 = make_active_wells(_row_column_ranges)
    _bn = os.path.basename(f'{_excel_file}').replace('.xlsx', '')
    _kinetic_data = parse_kinetics(_excel_file, active_wells_7, start_row=48)
    _kinetic_data.rename(columns=_rename_dic, inplace=True)
    _v0_enz = mm_kinetics(_kinetic_data, _bn, _rows, _cols, active_wells_7, _bg_row, _enz_conc, _sub_concs, _substrate_name, _pics_dir, _replicate_rows, fit_type='straight_line', slope=_slope, cutoff_start_time=_start_time, cutoff_end_time=_end_time, plate_type='full', y_ax_mm_intv=0.1, x_ax_mm_intv=50, legend_label=f'[{_substrate_name}], μM', norm_zero=True, dpi=FIGURE_DPI, save_eps=SAVE_EPS)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### II.II.E. 240830 F7 Kinetics ([E]<sub>0</sub> = 2μM; [Zn(II)] = 20μM; Temp = 25C)
    """)
    return


@app.cell
def _(
    FIGURE_DPI,
    SAVE_EPS,
    SLOPE_4MU_250401,
    make_active_wells,
    make_rename_dic,
    make_transpose_dic,
    mm_kinetics,
    np,
    os,
    parse_kinetics,
    raw_wetlab_data_dir,
    wetlab_data_plots_dir,
):
    _excel_file = f'{raw_wetlab_data_dir}240830_F7_OFFICIAL_kinetics_2uM___phenylacetate_subst_25C_4min_10x_zn.xlsx'
    _pics_dir = wetlab_data_plots_dir
    _slope = SLOPE_4MU_250401
    _enz_conc = 2.0
    _rows, _cols = (list('ABCDEFGH'), [9, 10, 11, 12])
    replicate_dic_6 = {9: [10, 11]}
    _bg_col = 12
    _sub_concs = np.array([144, 72, 36, 18, 9, 4.5, 2.3, 1.1])
    _col_range_to_analyze = (1, 8)
    _start_time = 0
    _end_time = 240
    _substrate_name = '4MU-PA'
    _transpose_dic = make_transpose_dic(_rows, _cols)
    _rename_dic = make_rename_dic(_transpose_dic, _rows, _cols)
    _represent_row = 'A'
    _replicate_rows = ['ABCDEFGHIJKL'[_cols.index(rep)] for rep in list(replicate_dic_6.values())[0]]
    _bg_row = 'ABCDEFGHIJKL'[_cols.index(_bg_col)]
    _row_column_ranges = {_represent_row: _col_range_to_analyze}
    active_wells_8 = make_active_wells(_row_column_ranges)
    _bn = os.path.basename(f'{_excel_file}').replace('.xlsx', '')
    _kinetic_data = parse_kinetics(_excel_file, active_wells_8, start_row=48)
    _kinetic_data.rename(columns=_rename_dic, inplace=True)
    _v0_enz = mm_kinetics(_kinetic_data, _bn, _rows, _cols, active_wells_8, _bg_row, _enz_conc, _sub_concs, _substrate_name, _pics_dir, _replicate_rows, fit_type='straight_line', slope=_slope, cutoff_start_time=_start_time, cutoff_end_time=_end_time, plate_type='full', y_ax_mm_intv=0.1, x_ax_mm_intv=50, legend_label=f'[{_substrate_name}], μM', norm_zero=True, dpi=FIGURE_DPI, save_eps=SAVE_EPS)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## II.III. Michaelis-Menten Kinetics for Top 11 Design Campaign 2 Hits (3 Unique RFdiffusion2 Scaffolds)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### II.III.A. 250325 ZETA_2 (H10) Kinetics ([E]<sub>0</sub> = 100nM; [Zn(II)] = 1μM; Temp = 25C)
    """)
    return


@app.cell
def _(
    FIGURE_DPI,
    SAVE_EPS,
    SLOPE_4MU_250401,
    make_active_wells,
    make_rename_dic,
    make_transpose_dic,
    mm_kinetics,
    np,
    os,
    parse_kinetics,
    raw_wetlab_data_dir,
    wetlab_data_plots_dir,
):
    _excel_file = f'{raw_wetlab_data_dir}250325_H10_1xZn_kinetics_100nM___phenylacetate_subst_25C_5min_10x_zn_144uMsubstLadder.xlsx'
    _pics_dir = wetlab_data_plots_dir
    _slope = SLOPE_4MU_250401
    _enz_conc = 0.1
    _rows, _cols = (list('ABCDEFGH'), [1, 2, 3, 4])
    replicate_dic_7 = {1: [2, 3]}
    _bg_col = 4
    _sub_concs = np.array([72, 36, 18, 9, 4.5, 2.3, 1.1])
    _col_range_to_analyze = (2, 8)
    _start_time = 0
    _end_time = 30
    _substrate_name = '4MU-PA'
    _transpose_dic = make_transpose_dic(_rows, _cols)
    _rename_dic = make_rename_dic(_transpose_dic, _rows, _cols)
    _represent_row = 'A'
    _replicate_rows = ['ABCDEFGHIJKL'[_cols.index(rep)] for rep in list(replicate_dic_7.values())[0]]
    _bg_row = 'ABCDEFGHIJKL'[_cols.index(_bg_col)]
    _row_column_ranges = {_represent_row: _col_range_to_analyze}
    active_wells_9 = make_active_wells(_row_column_ranges)
    _bn = os.path.basename(f'{_excel_file}').replace('.xlsx', '')
    _kinetic_data = parse_kinetics(_excel_file, active_wells_9, start_row=48)
    _kinetic_data.rename(columns=_rename_dic, inplace=True)
    _v0_enz = mm_kinetics(_kinetic_data, _bn, _rows, _cols, active_wells_9, _bg_row, _enz_conc, _sub_concs, _substrate_name, _pics_dir, _replicate_rows, fit_type='straight_line', slope=_slope, cutoff_start_time=_start_time, cutoff_end_time=_end_time, plate_type='full', y_ax_mm_intv=0.1, x_ax_mm_intv=50, legend_label=f'[{_substrate_name}], μM', norm_zero=True, dpi=FIGURE_DPI, save_eps=SAVE_EPS)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### II.III.B. 250318 F11 Kinetics ([E]<sub>0</sub> = 100nM; [Zn(II)] = 1μM; Temp = 25C)
    """)
    return


@app.cell
def _(
    FIGURE_DPI,
    SAVE_EPS,
    SLOPE_4MU_250401,
    make_active_wells,
    make_rename_dic,
    make_transpose_dic,
    mm_kinetics,
    np,
    os,
    parse_kinetics,
    raw_wetlab_data_dir,
    wetlab_data_plots_dir,
):
    _excel_file = f'{raw_wetlab_data_dir}250318_F11_1xZn_kinetics_100nM___phenylacetate_subst_25C_5min_10x_zn_144uMsubstLadder.xlsx'
    _pics_dir = wetlab_data_plots_dir
    _slope = SLOPE_4MU_250401
    _enz_conc = 0.1
    _rows, _cols = (list('ABCDEFGH'), [5, 6, 7, 8])
    replicate_dic_8 = {5: [7]}
    _bg_col = 8
    _sub_concs = np.array([72, 36, 18, 9, 4.5, 2.3, 1.1])
    _col_range_to_analyze = (2, 8)
    _start_time = 0
    _end_time = 30
    _substrate_name = '4MU-PA'
    _transpose_dic = make_transpose_dic(_rows, _cols)
    _rename_dic = make_rename_dic(_transpose_dic, _rows, _cols)
    _represent_row = 'A'
    _replicate_rows = ['ABCDEFGHIJKL'[_cols.index(rep)] for rep in list(replicate_dic_8.values())[0]]
    _bg_row = 'ABCDEFGHIJKL'[_cols.index(_bg_col)]
    _row_column_ranges = {_represent_row: _col_range_to_analyze}
    active_wells_10 = make_active_wells(_row_column_ranges)
    _bn = os.path.basename(f'{_excel_file}').replace('.xlsx', '')
    _kinetic_data = parse_kinetics(_excel_file, active_wells_10, start_row=48)
    _kinetic_data.rename(columns=_rename_dic, inplace=True)
    _v0_enz = mm_kinetics(_kinetic_data, _bn, _rows, _cols, active_wells_10, _bg_row, _enz_conc, _sub_concs, _substrate_name, _pics_dir, _replicate_rows, fit_type='straight_line', slope=_slope, cutoff_start_time=_start_time, cutoff_end_time=_end_time, plate_type='full', y_ax_mm_intv=0.1, x_ax_mm_intv=50, legend_label=f'[{_substrate_name}], μM', norm_zero=True, dpi=FIGURE_DPI, save_eps=SAVE_EPS)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### II.III.C. 250327 H1 Kinetics ([E]<sub>0</sub> = 100nM; [Zn(II)] = 1μM; Temp = 25C)
    """)
    return


@app.cell
def _(
    FIGURE_DPI,
    SAVE_EPS,
    SLOPE_4MU_250401,
    make_active_wells,
    make_rename_dic,
    make_transpose_dic,
    mm_kinetics,
    np,
    os,
    parse_kinetics,
    raw_wetlab_data_dir,
    wetlab_data_plots_dir,
):
    _excel_file = f'{raw_wetlab_data_dir}250327_H1_1xZn_kinetics_100nM___phenylacetate_subst_25C_5min_10x_zn_144uMsubstLadder.xlsx'
    _pics_dir = wetlab_data_plots_dir
    _slope = SLOPE_4MU_250401
    _enz_conc = 0.1
    _rows, _cols = (list('ABCDEFGH'), [1, 2, 3, 4])
    replicate_dic_9 = {1: [2, 3]}
    _bg_col = 4
    _sub_concs = np.array([72, 36, 18, 9, 4.5, 2.3, 1.1])
    _col_range_to_analyze = (2, 8)
    _start_time = 0
    _end_time = 1200
    _substrate_name = '4MU-PA'
    _transpose_dic = make_transpose_dic(_rows, _cols)
    _rename_dic = make_rename_dic(_transpose_dic, _rows, _cols)
    _represent_row = 'A'
    _replicate_rows = ['ABCDEFGHIJKL'[_cols.index(rep)] for rep in list(replicate_dic_9.values())[0]]
    _bg_row = 'ABCDEFGHIJKL'[_cols.index(_bg_col)]
    _row_column_ranges = {_represent_row: _col_range_to_analyze}
    active_wells_11 = make_active_wells(_row_column_ranges)
    _bn = os.path.basename(f'{_excel_file}').replace('.xlsx', '')
    _kinetic_data = parse_kinetics(_excel_file, active_wells_11, start_row=48)
    _kinetic_data.rename(columns=_rename_dic, inplace=True)
    _v0_enz = mm_kinetics(_kinetic_data, _bn, _rows, _cols, active_wells_11, _bg_row, _enz_conc, _sub_concs, _substrate_name, _pics_dir, _replicate_rows, fit_type='straight_line', slope=_slope, cutoff_start_time=_start_time, cutoff_end_time=_end_time, plate_type='full', y_ax_mm_intv=0.1, x_ax_mm_intv=50, legend_label=f'[{_substrate_name}], μM', norm_zero=True, dpi=FIGURE_DPI, save_eps=SAVE_EPS)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### II.III.D. 250327 F8 Kinetics ([E]<sub>0</sub> = 100nM; [Zn(II)] = 1μM; Temp = 25C)
    """)
    return


@app.cell
def _(
    FIGURE_DPI,
    SAVE_EPS,
    SLOPE_4MU_250401,
    make_active_wells,
    make_rename_dic,
    make_transpose_dic,
    mm_kinetics,
    np,
    os,
    parse_kinetics,
    raw_wetlab_data_dir,
    wetlab_data_plots_dir,
):
    _excel_file = f'{raw_wetlab_data_dir}250327_F8_1xZn_kinetics_100nM___phenylacetate_subst_25C_5min_10x_zn_144uMsubstLadder.xlsx'
    _pics_dir = wetlab_data_plots_dir
    _slope = SLOPE_4MU_250401
    _enz_conc = 0.1
    _rows, _cols = (list('ABCDEFGH'), [5, 6, 7, 8])
    replicate_dic_10 = {5: [6, 7]}
    _bg_col = 8
    _sub_concs = np.array([72, 36, 18, 9, 4.5, 2.3, 1.1])
    _col_range_to_analyze = (2, 8)
    _start_time = 0
    _end_time = 1200
    _substrate_name = '4MU-PA'
    _transpose_dic = make_transpose_dic(_rows, _cols)
    _rename_dic = make_rename_dic(_transpose_dic, _rows, _cols)
    _represent_row = 'A'
    _replicate_rows = ['ABCDEFGHIJKL'[_cols.index(rep)] for rep in list(replicate_dic_10.values())[0]]
    _bg_row = 'ABCDEFGHIJKL'[_cols.index(_bg_col)]
    _row_column_ranges = {_represent_row: _col_range_to_analyze}
    active_wells_12 = make_active_wells(_row_column_ranges)
    _bn = os.path.basename(f'{_excel_file}').replace('.xlsx', '')
    _kinetic_data = parse_kinetics(_excel_file, active_wells_12, start_row=48)
    _kinetic_data.rename(columns=_rename_dic, inplace=True)
    _v0_enz = mm_kinetics(_kinetic_data, _bn, _rows, _cols, active_wells_12, _bg_row, _enz_conc, _sub_concs, _substrate_name, _pics_dir, _replicate_rows, fit_type='straight_line', slope=_slope, cutoff_start_time=_start_time, cutoff_end_time=_end_time, plate_type='full', y_ax_mm_intv=0.1, x_ax_mm_intv=50, legend_label=f'[{_substrate_name}], μM', norm_zero=True, dpi=FIGURE_DPI, save_eps=SAVE_EPS)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### II.III.E. 250327 F2 Kinetics ([E]<sub>0</sub> = 100nM; [Zn(II)] = 1μM; Temp = 25C)
    """)
    return


@app.cell
def _(
    FIGURE_DPI,
    SAVE_EPS,
    SLOPE_4MU_250401,
    make_active_wells,
    make_rename_dic,
    make_transpose_dic,
    mm_kinetics,
    np,
    os,
    parse_kinetics,
    raw_wetlab_data_dir,
    wetlab_data_plots_dir,
):
    _excel_file = f'{raw_wetlab_data_dir}250327_F2_1xZn_kinetics_100nM___phenylacetate_subst_25C_5min_10x_zn_144uMsubstLadder.xlsx'
    _pics_dir = wetlab_data_plots_dir
    _slope = SLOPE_4MU_250401
    _enz_conc = 0.1
    _rows, _cols = (list('ABCDEFGH'), [5, 6, 7, 8])
    replicate_dic_11 = {5: [6, 7]}
    _bg_col = 8
    _sub_concs = np.array([72, 36, 18, 9, 4.5, 2.3, 1.1])
    _col_range_to_analyze = (2, 8)
    _start_time = 0
    _end_time = 1200
    _substrate_name = '4MU-PA'
    _transpose_dic = make_transpose_dic(_rows, _cols)
    _rename_dic = make_rename_dic(_transpose_dic, _rows, _cols)
    _represent_row = 'A'
    _replicate_rows = ['ABCDEFGHIJKL'[_cols.index(rep)] for rep in list(replicate_dic_11.values())[0]]
    _bg_row = 'ABCDEFGHIJKL'[_cols.index(_bg_col)]
    _row_column_ranges = {_represent_row: _col_range_to_analyze}
    active_wells_13 = make_active_wells(_row_column_ranges)
    _bn = os.path.basename(f'{_excel_file}').replace('.xlsx', '')
    _kinetic_data = parse_kinetics(_excel_file, active_wells_13, start_row=48)
    _kinetic_data.rename(columns=_rename_dic, inplace=True)
    _v0_enz = mm_kinetics(_kinetic_data, _bn, _rows, _cols, active_wells_13, _bg_row, _enz_conc, _sub_concs, _substrate_name, _pics_dir, _replicate_rows, fit_type='straight_line', slope=_slope, cutoff_start_time=_start_time, cutoff_end_time=_end_time, plate_type='full', y_ax_mm_intv=0.1, x_ax_mm_intv=50, legend_label=f'[{_substrate_name}], μM', norm_zero=True, dpi=FIGURE_DPI, save_eps=SAVE_EPS)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### II.III.F. 250327 F7 Kinetics ([E]<sub>0</sub> = 500nM; [Zn(II)] = 5μM; Temp = 25C) (BONUS DATA)
    """)
    return


@app.cell
def _(
    FIGURE_DPI,
    SAVE_EPS,
    SLOPE_4MU_250401,
    make_active_wells,
    make_rename_dic,
    make_transpose_dic,
    mm_kinetics,
    np,
    os,
    parse_kinetics,
    raw_wetlab_data_dir,
    wetlab_data_plots_dir,
):
    _excel_file = f'{raw_wetlab_data_dir}250327_F7_1xZn_kinetics_500nM___phenylacetate_subst_25C_5min_10x_zn_144uMsubstLadder.xlsx'
    _pics_dir = wetlab_data_plots_dir
    _slope = SLOPE_4MU_250401
    _enz_conc = 0.5
    _rows, _cols = (list('ABCDEFGH'), [5, 6, 7, 8])
    replicate_dic_12 = {5: [6, 7]}
    _bg_col = 8
    _sub_concs = np.array([72, 36, 18, 9, 4.5, 2.3, 1.1])
    _col_range_to_analyze = (2, 8)
    _start_time = 0
    _end_time = 1200
    _substrate_name = '4MU-PA'
    _transpose_dic = make_transpose_dic(_rows, _cols)
    _rename_dic = make_rename_dic(_transpose_dic, _rows, _cols)
    _represent_row = 'A'
    _replicate_rows = ['ABCDEFGHIJKL'[_cols.index(rep)] for rep in list(replicate_dic_12.values())[0]]
    _bg_row = 'ABCDEFGHIJKL'[_cols.index(_bg_col)]
    _row_column_ranges = {_represent_row: _col_range_to_analyze}
    active_wells_14 = make_active_wells(_row_column_ranges)
    _bn = os.path.basename(f'{_excel_file}').replace('.xlsx', '')
    _kinetic_data = parse_kinetics(_excel_file, active_wells_14, start_row=48)
    _kinetic_data.rename(columns=_rename_dic, inplace=True)
    _v0_enz = mm_kinetics(_kinetic_data, _bn, _rows, _cols, active_wells_14, _bg_row, _enz_conc, _sub_concs, _substrate_name, _pics_dir, _replicate_rows, fit_type='straight_line', slope=_slope, cutoff_start_time=_start_time, cutoff_end_time=_end_time, plate_type='full', y_ax_mm_intv=0.1, x_ax_mm_intv=50, legend_label=f'[{_substrate_name}], μM', norm_zero=True, dpi=FIGURE_DPI, save_eps=SAVE_EPS)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### II.III.G. 250318 ZETA_3 (H6) Kinetics ([E]<sub>0</sub> = 100nM; [Zn(II)] = 1μM; Temp = 25C)
    """)
    return


@app.cell
def _(
    FIGURE_DPI,
    SAVE_EPS,
    SLOPE_4MU_250401,
    make_active_wells,
    make_rename_dic,
    make_transpose_dic,
    mm_kinetics,
    np,
    os,
    parse_kinetics,
    raw_wetlab_data_dir,
    wetlab_data_plots_dir,
):
    _excel_file = f'{raw_wetlab_data_dir}250318_H6_1xZn_kinetics_100nM___phenylacetate_subst_25C_5min_10x_zn_144uMsubstLadder.xlsx'
    _pics_dir = wetlab_data_plots_dir
    _slope = SLOPE_4MU_250401
    _enz_conc = 0.1
    _rows, _cols = (list('ABCDEFGH'), [5, 6, 7, 8])
    replicate_dic_13 = {5: [6, 7]}
    _bg_col = 8
    _sub_concs = np.array([72, 36, 18, 9, 4.5, 2.3, 1.1])
    _col_range_to_analyze = (2, 8)
    _start_time = 0
    _end_time = 1200
    _substrate_name = '4MU-PA'
    _transpose_dic = make_transpose_dic(_rows, _cols)
    _rename_dic = make_rename_dic(_transpose_dic, _rows, _cols)
    _represent_row = 'A'
    _replicate_rows = ['ABCDEFGHIJKL'[_cols.index(rep)] for rep in list(replicate_dic_13.values())[0]]
    _bg_row = 'ABCDEFGHIJKL'[_cols.index(_bg_col)]
    _row_column_ranges = {_represent_row: _col_range_to_analyze}
    active_wells_15 = make_active_wells(_row_column_ranges)
    _bn = os.path.basename(f'{_excel_file}').replace('.xlsx', '')
    _kinetic_data = parse_kinetics(_excel_file, active_wells_15, start_row=48)
    _kinetic_data.rename(columns=_rename_dic, inplace=True)
    _v0_enz = mm_kinetics(_kinetic_data, _bn, _rows, _cols, active_wells_15, _bg_row, _enz_conc, _sub_concs, _substrate_name, _pics_dir, _replicate_rows, fit_type='straight_line', slope=_slope, cutoff_start_time=_start_time, cutoff_end_time=_end_time, plate_type='full', y_ax_mm_intv=0.1, x_ax_mm_intv=50, legend_label=f'[{_substrate_name}], μM', norm_zero=True, dpi=FIGURE_DPI, save_eps=SAVE_EPS)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### II.III.H. 250318 B10 Kinetics ([E]<sub>0</sub> = 100nM; [Zn(II)] = 1μM; Temp = 25C)
    """)
    return


@app.cell
def _(
    FIGURE_DPI,
    SAVE_EPS,
    SLOPE_4MU_250401,
    make_active_wells,
    make_rename_dic,
    make_transpose_dic,
    mm_kinetics,
    np,
    os,
    parse_kinetics,
    raw_wetlab_data_dir,
    wetlab_data_plots_dir,
):
    _excel_file = f'{raw_wetlab_data_dir}250318_B10_1xZn_kinetics_100nM___phenylacetate_subst_25C_5min_10x_zn_144uMsubstLadder.xlsx'
    _pics_dir = wetlab_data_plots_dir
    _slope = SLOPE_4MU_250401
    _enz_conc = 0.1
    _rows, _cols = (list('ABCDEFGH'), [5, 6, 7, 8])
    replicate_dic_14 = {5: [6, 7]}
    _bg_col = 8
    _sub_concs = np.array([72, 36, 18, 9, 4.5, 2.3, 1.1])
    _col_range_to_analyze = (2, 8)
    _start_time = 0
    _end_time = 600
    _substrate_name = '4MU-PA'
    _transpose_dic = make_transpose_dic(_rows, _cols)
    _rename_dic = make_rename_dic(_transpose_dic, _rows, _cols)
    _represent_row = 'A'
    _replicate_rows = ['ABCDEFGHIJKL'[_cols.index(rep)] for rep in list(replicate_dic_14.values())[0]]
    _bg_row = 'ABCDEFGHIJKL'[_cols.index(_bg_col)]
    _row_column_ranges = {_represent_row: _col_range_to_analyze}
    active_wells_16 = make_active_wells(_row_column_ranges)
    _bn = os.path.basename(f'{_excel_file}').replace('.xlsx', '')
    _kinetic_data = parse_kinetics(_excel_file, active_wells_16, start_row=48)
    _kinetic_data.rename(columns=_rename_dic, inplace=True)
    _v0_enz = mm_kinetics(_kinetic_data, _bn, _rows, _cols, active_wells_16, _bg_row, _enz_conc, _sub_concs, _substrate_name, _pics_dir, _replicate_rows, fit_type='straight_line', slope=_slope, cutoff_start_time=_start_time, cutoff_end_time=_end_time, plate_type='full', y_ax_mm_intv=0.1, x_ax_mm_intv=50, legend_label=f'[{_substrate_name}], μM', norm_zero=True, dpi=FIGURE_DPI, save_eps=SAVE_EPS)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### II.III.I. 250327 B5 Kinetics ([E]<sub>0</sub> = 500nM; [Zn(II)] = 5μM; Temp = 25C)
    """)
    return


@app.cell
def _(
    FIGURE_DPI,
    SAVE_EPS,
    SLOPE_4MU_250401,
    make_active_wells,
    make_rename_dic,
    make_transpose_dic,
    mm_kinetics,
    np,
    os,
    parse_kinetics,
    raw_wetlab_data_dir,
    wetlab_data_plots_dir,
):
    _excel_file = f'{raw_wetlab_data_dir}250327_B5_1xZn_kinetics_500nM___phenylacetate_subst_25C_5min_10x_zn_144uMsubstLadder.xlsx'
    _pics_dir = wetlab_data_plots_dir
    _slope = SLOPE_4MU_250401
    _enz_conc = 0.5
    _rows, _cols = (list('ABCDEFGH'), [9, 10, 11, 12])
    replicate_dic_15 = {9: [10, 11]}
    _bg_col = 12
    _sub_concs = np.array([72, 36, 18, 9, 4.5, 2.3, 1.1])
    _col_range_to_analyze = (2, 8)
    _start_time = 30
    _end_time = 150
    _substrate_name = '4MU-PA'
    _transpose_dic = make_transpose_dic(_rows, _cols)
    _rename_dic = make_rename_dic(_transpose_dic, _rows, _cols)
    _represent_row = 'A'
    _replicate_rows = ['ABCDEFGHIJKL'[_cols.index(rep)] for rep in list(replicate_dic_15.values())[0]]
    _bg_row = 'ABCDEFGHIJKL'[_cols.index(_bg_col)]
    _row_column_ranges = {_represent_row: _col_range_to_analyze}
    active_wells_17 = make_active_wells(_row_column_ranges)
    _bn = os.path.basename(f'{_excel_file}').replace('.xlsx', '')
    _kinetic_data = parse_kinetics(_excel_file, active_wells_17, start_row=48)
    _kinetic_data.rename(columns=_rename_dic, inplace=True)
    _v0_enz = mm_kinetics(_kinetic_data, _bn, _rows, _cols, active_wells_17, _bg_row, _enz_conc, _sub_concs, _substrate_name, _pics_dir, _replicate_rows, fit_type='straight_line', slope=_slope, cutoff_start_time=_start_time, cutoff_end_time=_end_time, plate_type='full', y_ax_mm_intv=0.1, x_ax_mm_intv=50, legend_label=f'[{_substrate_name}], μM', norm_zero=True, dpi=FIGURE_DPI, save_eps=SAVE_EPS)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### II.III.J. 250327 D9 Kinetics ([E]<sub>0</sub> = 500nM; [Zn(II)] = 5μM; Temp = 25C)
    """)
    return


@app.cell
def _(
    FIGURE_DPI,
    SAVE_EPS,
    SLOPE_4MU_250401,
    make_active_wells,
    make_rename_dic,
    make_transpose_dic,
    mm_kinetics,
    np,
    os,
    parse_kinetics,
    raw_wetlab_data_dir,
    wetlab_data_plots_dir,
):
    _excel_file = f'{raw_wetlab_data_dir}250327_D9_1xZn_kinetics_500nM___phenylacetate_subst_25C_5min_10x_zn_144uMsubstLadder.xlsx'
    _pics_dir = wetlab_data_plots_dir
    _slope = SLOPE_4MU_250401
    _enz_conc = 0.5
    _rows, _cols = (list('ABCDEFGH'), [9, 10, 11, 12])
    replicate_dic_16 = {9: [10, 11]}
    _bg_col = 12
    _sub_concs = np.array([72, 36, 18, 9, 4.5, 2.3, 1.1])
    _col_range_to_analyze = (2, 8)
    _start_time = 0
    _end_time = 120
    _substrate_name = '4MU-PA'
    _transpose_dic = make_transpose_dic(_rows, _cols)
    _rename_dic = make_rename_dic(_transpose_dic, _rows, _cols)
    _represent_row = 'A'
    _replicate_rows = ['ABCDEFGHIJKL'[_cols.index(rep)] for rep in list(replicate_dic_16.values())[0]]
    _bg_row = 'ABCDEFGHIJKL'[_cols.index(_bg_col)]
    _row_column_ranges = {_represent_row: _col_range_to_analyze}
    active_wells_18 = make_active_wells(_row_column_ranges)
    _bn = os.path.basename(f'{_excel_file}').replace('.xlsx', '')
    _kinetic_data = parse_kinetics(_excel_file, active_wells_18, start_row=48)
    _kinetic_data.rename(columns=_rename_dic, inplace=True)
    _v0_enz = mm_kinetics(_kinetic_data, _bn, _rows, _cols, active_wells_18, _bg_row, _enz_conc, _sub_concs, _substrate_name, _pics_dir, _replicate_rows, fit_type='straight_line', slope=_slope, cutoff_start_time=_start_time, cutoff_end_time=_end_time, plate_type='full', y_ax_mm_intv=0.1, x_ax_mm_intv=50, legend_label=f'[{_substrate_name}], μM', norm_zero=True, dpi=FIGURE_DPI, save_eps=SAVE_EPS)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### II.III.K. 250327 F12 Kinetics ([E]<sub>0</sub> = 500nM; [Zn(II)] = 5μM; Temp = 25C)
    """)
    return


@app.cell
def _(
    FIGURE_DPI,
    SAVE_EPS,
    SLOPE_4MU_250401,
    make_active_wells,
    make_rename_dic,
    make_transpose_dic,
    mm_kinetics,
    np,
    os,
    parse_kinetics,
    raw_wetlab_data_dir,
    wetlab_data_plots_dir,
):
    _excel_file = f'{raw_wetlab_data_dir}250327_F12_1xZn_kinetics_500nM___phenylacetate_subst_25C_5min_10x_zn_144uMsubstLadder.xlsx'
    _pics_dir = wetlab_data_plots_dir
    _slope = SLOPE_4MU_250401
    _enz_conc = 0.5
    _rows, _cols = (list('ABCDEFGH'), [9, 10, 11, 12])
    replicate_dic_17 = {9: [10, 11]}
    _bg_col = 12
    _sub_concs = np.array([72, 36, 18, 9, 4.5, 2.3, 1.1])
    _col_range_to_analyze = (2, 8)
    _start_time = 0
    _end_time = 1200
    _substrate_name = '4MU-PA'
    _transpose_dic = make_transpose_dic(_rows, _cols)
    _rename_dic = make_rename_dic(_transpose_dic, _rows, _cols)
    _represent_row = 'A'
    _replicate_rows = ['ABCDEFGHIJKL'[_cols.index(rep)] for rep in list(replicate_dic_17.values())[0]]
    _bg_row = 'ABCDEFGHIJKL'[_cols.index(_bg_col)]
    _row_column_ranges = {_represent_row: _col_range_to_analyze}
    active_wells_19 = make_active_wells(_row_column_ranges)
    _bn = os.path.basename(f'{_excel_file}').replace('.xlsx', '')
    _kinetic_data = parse_kinetics(_excel_file, active_wells_19, start_row=48)
    _kinetic_data.rename(columns=_rename_dic, inplace=True)
    _v0_enz = mm_kinetics(_kinetic_data, _bn, _rows, _cols, active_wells_19, _bg_row, _enz_conc, _sub_concs, _substrate_name, _pics_dir, _replicate_rows, fit_type='straight_line', slope=_slope, cutoff_start_time=_start_time, cutoff_end_time=_end_time, plate_type='full', y_ax_mm_intv=0.1, x_ax_mm_intv=50, legend_label=f'[{_substrate_name}], μM', norm_zero=True, dpi=FIGURE_DPI, save_eps=SAVE_EPS)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### II.III.L. 250327 C7 Kinetics ([E]<sub>0</sub> = 500nM; [Zn(II)] = 5μM; Temp = 25C) (BONUS DATA)
    """)
    return


@app.cell
def _(
    FIGURE_DPI,
    SAVE_EPS,
    SLOPE_4MU_250401,
    make_active_wells,
    make_rename_dic,
    make_transpose_dic,
    mm_kinetics,
    np,
    os,
    parse_kinetics,
    raw_wetlab_data_dir,
    wetlab_data_plots_dir,
):
    _excel_file = f'{raw_wetlab_data_dir}250327_C7_1xZn_kinetics_500nM___phenylacetate_subst_25C_5min_10x_zn_144uMsubstLadder.xlsx'
    _pics_dir = wetlab_data_plots_dir
    _slope = SLOPE_4MU_250401
    _enz_conc = 0.5
    _rows, _cols = (list('ABCDEFGH'), [9, 10, 11, 12])
    replicate_dic_18 = {9: [10, 11]}
    _bg_col = 12
    _sub_concs = np.array([72, 36, 18, 9, 4.5, 2.3, 1.1])
    _col_range_to_analyze = (2, 8)
    _start_time = 0
    _end_time = 1200
    _substrate_name = '4MU-PA'
    _transpose_dic = make_transpose_dic(_rows, _cols)
    _rename_dic = make_rename_dic(_transpose_dic, _rows, _cols)
    _represent_row = 'A'
    _replicate_rows = ['ABCDEFGHIJKL'[_cols.index(rep)] for rep in list(replicate_dic_18.values())[0]]
    _bg_row = 'ABCDEFGHIJKL'[_cols.index(_bg_col)]
    _row_column_ranges = {_represent_row: _col_range_to_analyze}
    active_wells_20 = make_active_wells(_row_column_ranges)
    _bn = os.path.basename(f'{_excel_file}').replace('.xlsx', '')
    _kinetic_data = parse_kinetics(_excel_file, active_wells_20, start_row=48)
    _kinetic_data.rename(columns=_rename_dic, inplace=True)
    _v0_enz = mm_kinetics(_kinetic_data, _bn, _rows, _cols, active_wells_20, _bg_row, _enz_conc, _sub_concs, _substrate_name, _pics_dir, _replicate_rows, fit_type='straight_line', slope=_slope, cutoff_start_time=_start_time, cutoff_end_time=_end_time, plate_type='full', y_ax_mm_intv=0.1, x_ax_mm_intv=50, legend_label=f'[{_substrate_name}], μM', norm_zero=True, dpi=FIGURE_DPI, save_eps=SAVE_EPS)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### II.III.M. 250327 ZETA_4 (C5) Kinetics ([E]<sub>0</sub> = 500nM; [Zn(II)] = 5μM; Temp = 25C)

    The fitted substrate range is 1.1–72 μM, matching Supplementary Methods §4.3;
    the 144 μM wells remain in the raw data but are excluded from this published fit.
    """)
    return


@app.cell
def _(
    FIGURE_DPI,
    SAVE_EPS,
    SLOPE_4MU_250401,
    make_active_wells,
    make_rename_dic,
    make_transpose_dic,
    mm_kinetics,
    np,
    os,
    parse_kinetics,
    raw_wetlab_data_dir,
    wetlab_data_plots_dir,
):
    _excel_file = f'{raw_wetlab_data_dir}250327_C5_1xZn_kinetics_500nM___phenylacetate_subst_25C_5min_10x_zn_144uMsubstLadder.xlsx'
    _pics_dir = wetlab_data_plots_dir
    _slope = SLOPE_4MU_250401
    _enz_conc = 0.5
    _rows, _cols = (list('ABCDEFGH'), [1, 2, 3, 4])
    replicate_dic_19 = {1: [2, 3]}
    _bg_col = 4
    _sub_concs = np.array([72, 36, 18, 9, 4.5, 2.3, 1.1])
    _col_range_to_analyze = (2, 8)
    _start_time = 0
    _end_time = 1200
    _substrate_name = '4MU-PA'
    _transpose_dic = make_transpose_dic(_rows, _cols)
    _rename_dic = make_rename_dic(_transpose_dic, _rows, _cols)
    _represent_row = 'A'
    _replicate_rows = ['ABCDEFGHIJKL'[_cols.index(rep)] for rep in list(replicate_dic_19.values())[0]]
    _bg_row = 'ABCDEFGHIJKL'[_cols.index(_bg_col)]
    _row_column_ranges = {_represent_row: _col_range_to_analyze}
    active_wells_21 = make_active_wells(_row_column_ranges)
    _bn = os.path.basename(f'{_excel_file}').replace('.xlsx', '')
    _kinetic_data = parse_kinetics(_excel_file, active_wells_21, start_row=48)
    _kinetic_data.rename(columns=_rename_dic, inplace=True)
    _v0_enz = mm_kinetics(_kinetic_data, _bn, _rows, _cols, active_wells_21, _bg_row, _enz_conc, _sub_concs, _substrate_name, _pics_dir, _replicate_rows, fit_type='straight_line', slope=_slope, cutoff_start_time=_start_time, cutoff_end_time=_end_time, plate_type='full', y_ax_mm_intv=0.1, x_ax_mm_intv=50, legend_label=f'[{_substrate_name}], μM', norm_zero=True, dpi=FIGURE_DPI, save_eps=SAVE_EPS)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## II.IV. Michaelis-Menten Kinetics for ZETA_1 Mutants (Knockouts) with Activity
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### II.IV.A. 241008 ZETA_1 (A1; Design Campaign 1) **D67A Mutant** Kinetics ([E]<sub>0</sub> = 100nM; [Zn(II)] = 4μM; Temp = 25C)
    """)
    return


@app.cell
def _(
    FIGURE_DPI,
    SAVE_EPS,
    SLOPE_4MU_250401,
    mm_kinetics,
    np,
    os,
    parse_kinetics,
    raw_wetlab_data_dir,
    wetlab_data_plots_dir,
):
    _excel_file = f'{raw_wetlab_data_dir}241008_A1_D67A_FINAL_kinetics_100nM___phenylacetate_subst_25C_5min_40x_zn_144uMsubstLadder.xlsx'
    _slope = SLOPE_4MU_250401
    _start_time = 0
    _end_time = 30
    _substrate_name = '4MU-PA'
    _enz_conc = 0.1
    _rows, _cols = (list('ABCDEFGH'), [9, 10, 11, 12])
    _transpose_dic = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 9: 'A', 10: 'B', 11: 'C', 12: 'D'}
    _pics_dir = wetlab_data_plots_dir
    _rename_dic = {}
    for _key in _transpose_dic:
        _value = _transpose_dic[_key]
        if _key in _rows:
            for _col in _cols:
                _rename_dic[f'{_key}{_col}'] = f'{_transpose_dic[_col]}{_value}'
        elif _key in _cols:
            for _row in _rows:
                _rename_dic[f'{_row}{_key}'] = f'{_value}{_transpose_dic[_row]}'
    _row_column_ranges = {'A': (2, 8)}
    _replicate_rows = ['B', 'C']
    active_wells_22 = []
    for _row, (_start_col, _end_col) in _row_column_ranges.items():
        for _col in range(_start_col, _end_col + 1):
            active_wells_22.append(f'{_row}{_col}')
    _sub_concs = np.array([72, 36, 18, 9, 4.5, 2.3, 1.1])
    _bg_row = 'D'
    _bn = os.path.basename(f'{_excel_file}').replace('.xlsx', '')
    _kinetic_data = parse_kinetics(_excel_file, active_wells_22, start_row=48)
    _kinetic_data.rename(columns=_rename_dic, inplace=True)
    mm_kinetics(_kinetic_data, _bn, None, None, active_wells_22, _bg_row, _enz_conc, _sub_concs, _substrate_name, _pics_dir, _replicate_rows, fit_type='straight_line', slope=_slope, cutoff_start_time=_start_time, cutoff_end_time=_end_time, plate_type='full', y_ax_mm_intv=0.1, x_ax_mm_intv=50, legend_label=f'[{_substrate_name}], μM', norm_zero=True, dpi=FIGURE_DPI, save_eps=SAVE_EPS)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### II.IV.B. 240927 ZETA_1 (A1; Design Campaign 1) **N17A Mutant** Kinetics ([E]<sub>0</sub> = 500nM; [Zn(II)] = 20μM; Temp = 25C)
    """)
    return


@app.cell
def _(
    FIGURE_DPI,
    SAVE_EPS,
    SLOPE_4MU_250401,
    mm_kinetics,
    np,
    os,
    parse_kinetics,
    raw_wetlab_data_dir,
    wetlab_data_plots_dir,
):
    _excel_file = f'{raw_wetlab_data_dir}240927_A1_N17A_FINAL_kinetics_500nM___phenylacetate_subst_25C_5min_40x_zn_144uMsubstLadder.xlsx'
    _slope = SLOPE_4MU_250401
    _start_time = 0
    _end_time = 60
    _substrate_name = '4MU-PA'
    _enz_conc = 0.5
    _rows, _cols = (list('ABCDEFGH'), [5, 6, 7, 8])
    _transpose_dic = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 5: 'A', 6: 'B', 7: 'C', 8: 'D'}
    _pics_dir = wetlab_data_plots_dir
    _rename_dic = {}
    for _key in _transpose_dic:
        _value = _transpose_dic[_key]
        if _key in _rows:
            for _col in _cols:
                _rename_dic[f'{_key}{_col}'] = f'{_transpose_dic[_col]}{_value}'
        elif _key in _cols:
            for _row in _rows:
                _rename_dic[f'{_row}{_key}'] = f'{_value}{_transpose_dic[_row]}'
    _row_column_ranges = {'A': (2, 8)}
    _replicate_rows = ['B', 'C']
    active_wells_23 = []
    for _row, (_start_col, _end_col) in _row_column_ranges.items():
        for _col in range(_start_col, _end_col + 1):
            active_wells_23.append(f'{_row}{_col}')
    _sub_concs = np.array([72, 36, 18, 9, 4.5, 2.3, 1.1])
    _bg_row = 'D'
    _bn = os.path.basename(f'{_excel_file}').replace('.xlsx', '')
    _kinetic_data = parse_kinetics(_excel_file, active_wells_23, start_row=48)
    _kinetic_data.rename(columns=_rename_dic, inplace=True)
    mm_kinetics(_kinetic_data, _bn, None, None, active_wells_23, _bg_row, _enz_conc, _sub_concs, _substrate_name, _pics_dir, _replicate_rows, fit_type='straight_line', slope=_slope, cutoff_start_time=_start_time, cutoff_end_time=_end_time, plate_type='full', y_ax_mm_intv=0.01, x_ax_mm_intv=50, legend_label=f'[{_substrate_name}], μM', norm_zero=True, dpi=FIGURE_DPI, save_eps=SAVE_EPS)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### II.IV.C. 240927 ZETA_1 (A1; Design Campaign 1) **H130A Mutant** Kinetics ([E]<sub>0</sub> = 500nM; [Zn(II)] = 20μM; Temp = 25C)
    """)
    return


@app.cell
def _(
    FIGURE_DPI,
    SAVE_EPS,
    SLOPE_4MU_240927,
    SLOPE_4MU_250401,
    mm_kinetics,
    np,
    os,
    parse_kinetics,
    raw_wetlab_data_dir,
    wetlab_data_plots_dir,
):
    _excel_file = f'{raw_wetlab_data_dir}240927_A1_H130A_FINAL_kinetics_500nM___phenylacetate_subst_25C_5min_40x_zn_144uMsubstLadder.xlsx'
    _slope = SLOPE_4MU_240927
    _slope = SLOPE_4MU_250401
    _start_time = 0
    _end_time = 60
    _substrate_name = '4MU-PA'
    _enz_conc = 0.5
    _rows, _cols = (list('ABCDEFGH'), [5, 6, 7, 8])
    _transpose_dic = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 5: 'A', 6: 'B', 7: 'C', 8: 'D'}
    _pics_dir = wetlab_data_plots_dir
    _rename_dic = {}
    for _key in _transpose_dic:
        _value = _transpose_dic[_key]
        if _key in _rows:
            for _col in _cols:
                _rename_dic[f'{_key}{_col}'] = f'{_transpose_dic[_col]}{_value}'
        elif _key in _cols:
            for _row in _rows:
                _rename_dic[f'{_row}{_key}'] = f'{_value}{_transpose_dic[_row]}'
    _row_column_ranges = {'A': (2, 8)}
    _replicate_rows = ['B', 'C']
    active_wells_24 = []
    for _row, (_start_col, _end_col) in _row_column_ranges.items():
        for _col in range(_start_col, _end_col + 1):
            active_wells_24.append(f'{_row}{_col}')
    _sub_concs = np.array([72, 36, 18, 9, 4.5, 2.3, 1.1])
    _bg_row = 'D'
    _bn = os.path.basename(f'{_excel_file}').replace('.xlsx', '')
    _kinetic_data = parse_kinetics(_excel_file, active_wells_24, start_row=48)
    _kinetic_data.rename(columns=_rename_dic, inplace=True)
    mm_kinetics(_kinetic_data, _bn, None, None, active_wells_24, _bg_row, _enz_conc, _sub_concs, _substrate_name, _pics_dir, _replicate_rows, fit_type='straight_line', slope=_slope, cutoff_start_time=_start_time, cutoff_end_time=_end_time, plate_type='full', y_ax_mm_intv=0.01, x_ax_mm_intv=50, legend_label=f'[{_substrate_name}], μM', norm_zero=True, dpi=FIGURE_DPI, save_eps=SAVE_EPS)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **III. Zn(II)-DEPENDENCE DATA**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## III.I. Zn(II)-Dependent Activity Screening of ZETA_1
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### III.I.A. Reaction Progress Curves for 1.) Zn(II)-Free Enzyme, Followed by 2.) Readdition of Zn(II) to Reactions
    """)
    return


@app.cell
def _(
    FIGURE_DPI,
    SAVE_EPS,
    np,
    parse_kinetics,
    plt,
    raw_wetlab_data_dir,
    save_figure,
    wetlab_data_plots_dir,
):
    excel_file1 = f'{raw_wetlab_data_dir}240904_zn_chelation_72uM_18uM_1pt8uM_4mu_PositiveTopRow_ChelatedBottomRow_30min_25C_pt1_BEFORE_100uM_zn.xlsx'
    excel_file2 = f'{raw_wetlab_data_dir}240904_zn_chelation_72uM_18uM_1pt8uM_4mu_PositiveTopRow_ChelatedBottomRow_30min_25C_pt1_AFTER_100uM_zn_2min_delay.xlsx'
    _row_column_ranges = {'A': (1, 12), 'B': (1, 12)}
    replicate_dic_20 = {9: [10, 11]}
    _background_col = 12
    active_wells_25 = []
    for _row, (_start_col, _end_col) in _row_column_ranges.items():
        for _col in range(_start_col, _end_col + 1):
            active_wells_25.append(f'{_row}{_col}')
    screening_data_before_zn = parse_kinetics(excel_file1, active_wells_25, start_row=48)
    screening_data_after_zn = parse_kinetics(excel_file2, active_wells_25, start_row=48)
    '\nfor i, df in enumerate([screening_data_before_zn, screening_data_after_zn]):\n    for well in active_wells:\n        fu = df[well].values\n        product_concentration = fu * slope * 1000000\n        df[well] = product_concentration\n'
    for _well in active_wells_25:
        screening_data_after_zn[_well] = screening_data_after_zn[_well] - np.min(screening_data_before_zn[_well])
        screening_data_before_zn[_well] = screening_data_before_zn[_well] - np.min(screening_data_before_zn[_well])
    for _i, df in enumerate([screening_data_before_zn, screening_data_after_zn]):
        df['Time'] = df['Time'] / 60
    screening_data_after_zn['Time'] = screening_data_after_zn['Time'] + 32
    _fig, _axes = plt.subplots(1, 2, figsize=(6, 3))
    for _representive_col in replicate_dic_20:
        _replicates = [_representive_col]
        _replicates.extend(replicate_dic_20[_representive_col])
        for _row, _name, color in zip(['A', 'B'], ['WT', 'Chelated'], ['black', 'red']):
            _replicates_well = [f'{_row}{col}' for col in _replicates]
            avg_product_concentration_before_zn = np.array(screening_data_before_zn[_replicates_well].mean(axis=1).tolist())
            std_product_concentration_before_zn = np.array(screening_data_before_zn[_replicates_well].std(axis=1).tolist())
            _axes[0].plot(screening_data_before_zn['Time'].values, avg_product_concentration_before_zn, color=color)
            _axes[0].fill_between(screening_data_before_zn['Time'].values, avg_product_concentration_before_zn - std_product_concentration_before_zn, avg_product_concentration_before_zn + std_product_concentration_before_zn, color=color, alpha=0.45)
            avg_product_concentration_after_zn = np.array(screening_data_after_zn[_replicates_well].mean(axis=1).tolist())
            std_product_concentration_after_zn = np.array(screening_data_after_zn[_replicates_well].std(axis=1).tolist())
            _axes[1].plot(screening_data_after_zn['Time'].values, avg_product_concentration_after_zn, color=color)
            if color == 'red':
                zn_annot_time = screening_data_after_zn['Time'].values
                zn_annot_curve = avg_product_concentration_after_zn
            _axes[1].fill_between(screening_data_after_zn['Time'].values, avg_product_concentration_after_zn - std_product_concentration_after_zn, avg_product_concentration_after_zn + std_product_concentration_after_zn, color=color, alpha=0.45)
            _axes[0].plot(screening_data_before_zn['Time'].values, screening_data_before_zn[f'{_row}{_background_col}'], color=color, linestyle='dashed')
            _axes[1].plot(screening_data_after_zn['Time'].values, screening_data_after_zn[f'{_row}{_background_col}'], color=color, linestyle='dashed')
        _axes[0].set_ylim(_axes[0].get_ylim()[0], _axes[1].get_ylim()[1])
        _axes[1].set_ylim(_axes[0].get_ylim()[0], _axes[1].get_ylim()[1])
        _axes[1].set_yticklabels([])
        _x0, _y0 = (zn_annot_time[0], zn_annot_curve[0])
        _xlo, _xhi = _axes[1].get_xlim()
        _ylo, _yhi = _axes[1].get_ylim()
        _annot_grey = '0.35'
        _axes[1].annotate('Zn(II)\naddition', xy=(_x0, _y0), xytext=(_x0 + 0.04 * (_xhi - _xlo), _ylo + 0.58 * (_yhi - _ylo)), color=_annot_grey, fontsize=7, ha='left', va='bottom', linespacing=1.2, arrowprops=dict(arrowstyle='->', color=_annot_grey, lw=0.8, shrinkA=3, shrinkB=3))
        _axes[0].set_ylabel('Fluorescence Units')
    plt.tight_layout()
    save_figure(f'{wetlab_data_plots_dir}zinc_chelating_progress_curves_1pt8uM_4MUPA', dpi=FIGURE_DPI, save_eps=SAVE_EPS)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## III.II. Dissociation Constant (K<sub>d</sub>) Measurement for Zn(II) Binding to ZETA_1 + Mutants (Knockouts)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### III.II.A. Load Data and Plot (Non-Normalized)
    """)
    return


@app.cell
def _(pd, raw_wetlab_data_dir):
    file_path = f'{raw_wetlab_data_dir}Zn-Hydrolase_binding_data.csv'
    data_3 = pd.read_csv(file_path, delimiter=',', dtype=float, na_values=['', ' '])
    return (data_3,)


@app.cell
def _(FIGURE_DPI, SAVE_EPS, data_3, plt, save_figure, wetlab_data_plots_dir):
    #Plot not normalized data
    plt.figure()
    #Dye (Mag-Fura-2)
    plt.plot(data_3.iloc[:, 0], data_3.iloc[:, 1], color='black')  # open a fresh figure so this cell does not draw onto a previous one
    plt.scatter(data_3.iloc[:, 2], data_3.iloc[:, 3], color='black', label='Mag-Fura-2')
    plt.errorbar(data_3.iloc[:, 2], data_3.iloc[:, 3], yerr=data_3.iloc[:, 4], fmt='o', capsize=5, color='black')
    plt.plot(data_3.iloc[:, 5], data_3.iloc[:, 6], color='green')
    plt.scatter(data_3.iloc[:, 7], data_3.iloc[:, 8], color='green', label='A1')
    #A1
    plt.errorbar(data_3.iloc[:, 7], data_3.iloc[:, 8], yerr=data_3.iloc[:, 9], fmt='o', capsize=5, color='green')
    plt.plot(data_3.iloc[:, 10], data_3.iloc[:, 11], color='orange')
    plt.scatter(data_3.iloc[:, 12], data_3.iloc[:, 13], color='orange', label='MUT1')
    plt.errorbar(data_3.iloc[:, 12], data_3.iloc[:, 13], yerr=data_3.iloc[:, 14], fmt='o', capsize=5, color='orange')
    #MUT1
    plt.plot(data_3.iloc[:, 15], data_3.iloc[:, 16], color='purple')
    plt.scatter(data_3.iloc[:, 17], data_3.iloc[:, 18], color='purple', label='MUT2')
    plt.errorbar(data_3.iloc[:, 17], data_3.iloc[:, 18], yerr=data_3.iloc[:, 19], fmt='o', capsize=5, color='purple')
    plt.plot(data_3.iloc[:, 20], data_3.iloc[:, 21], color='blue')
    #MUT2
    plt.scatter(data_3.iloc[:, 22], data_3.iloc[:, 23], color='blue', label='MUT3')
    plt.errorbar(data_3.iloc[:, 22], data_3.iloc[:, 23], yerr=data_3.iloc[:, 24], fmt='o', capsize=5, color='blue')
    plt.plot(data_3.iloc[:, 25], data_3.iloc[:, 26], color='pink')
    plt.scatter(data_3.iloc[:, 27], data_3.iloc[:, 28], color='pink', label='MUT4')
    #MUT3
    plt.errorbar(data_3.iloc[:, 27], data_3.iloc[:, 28], yerr=data_3.iloc[:, 29], fmt='o', capsize=5, color='pink')
    plt.plot(data_3.iloc[:, 30], data_3.iloc[:, 31], color='red')
    plt.scatter(data_3.iloc[:, 32], data_3.iloc[:, 33], color='red', label='MUT5')
    plt.errorbar(data_3.iloc[:, 32], data_3.iloc[:, 33], yerr=data_3.iloc[:, 34], fmt='o', capsize=5, color='red')
    #MUT4
    plt.plot(data_3.iloc[:, 35], data_3.iloc[:, 36], color='lightblue')
    plt.scatter(data_3.iloc[:, 37], data_3.iloc[:, 38], color='lightblue', label='MUT6')
    plt.errorbar(data_3.iloc[:, 37], data_3.iloc[:, 38], yerr=data_3.iloc[:, 39], fmt='o', capsize=5, color='lightblue')
    plt.xlabel('Total metal ion concentration (uM)')
    #MUT5
    plt.ylabel('Absorbance ratio')
    plt.title('Metal Binding Isotherm')
    plt.legend()
    save_figure(f'{wetlab_data_plots_dir}zinc_affinity_absorbance_raw_isotherm', dpi=FIGURE_DPI, save_eps=SAVE_EPS)
    #MUT6
    plt.show()  # PNG always; EPS only if SAVE_EPS. Raise dpi here for one figure.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### III.II.B. Normalize Data and Plot
    """)
    return


@app.cell
def _(data_3, np):
    #Data normalization
    dye_fit_normalized = data_3.iloc[:, 1] / np.max(np.abs(data_3.iloc[:, 1]))
    #Dye
    dye_data_normalized = data_3.iloc[:, 3] / np.max(np.abs(data_3.iloc[:, 1]))
    dye_error_normalized = data_3.iloc[:, 4] / np.max(np.abs(data_3.iloc[:, 1]))
    A1_fit_normalized = data_3.iloc[:, 6] / np.max(np.abs(data_3.iloc[:, 6]))
    A1_data_normalized = data_3.iloc[:, 8] / np.max(np.abs(data_3.iloc[:, 6]))
    #A1
    A1_error_normalized = data_3.iloc[:, 9] / np.max(np.abs(data_3.iloc[:, 6]))
    MUT1_fit_normalized = data_3.iloc[:, 11] / np.max(np.abs(data_3.iloc[:, 11]))
    MUT1_data_normalized = data_3.iloc[:, 13] / np.max(np.abs(data_3.iloc[:, 11]))
    MUT1_error_normalized = data_3.iloc[:, 14] / np.max(np.abs(data_3.iloc[:, 11]))
    #MUT1
    MUT2_fit_normalized = data_3.iloc[:, 16] / np.max(np.abs(data_3.iloc[:, 16]))
    MUT2_data_normalized = data_3.iloc[:, 18] / np.max(np.abs(data_3.iloc[:, 16]))
    MUT2_error_normalized = data_3.iloc[:, 19] / np.max(np.abs(data_3.iloc[:, 16]))
    MUT3_fit_normalized = data_3.iloc[:, 21] / np.max(np.abs(data_3.iloc[:, 21]))
    #MUT2
    MUT3_data_normalized = data_3.iloc[:, 23] / np.max(np.abs(data_3.iloc[:, 21]))
    MUT3_error_normalized = data_3.iloc[:, 24] / np.max(np.abs(data_3.iloc[:, 21]))
    MUT4_fit_normalized = data_3.iloc[:, 26] / np.max(np.abs(data_3.iloc[:, 26]))
    MUT4_data_normalized = data_3.iloc[:, 28] / np.max(np.abs(data_3.iloc[:, 26]))
    #MUT3
    MUT4_error_normalized = data_3.iloc[:, 29] / np.max(np.abs(data_3.iloc[:, 26]))
    MUT5_fit_normalized = data_3.iloc[:, 31] / np.max(np.abs(data_3.iloc[:, 31]))
    MUT5_data_normalized = data_3.iloc[:, 33] / np.max(np.abs(data_3.iloc[:, 31]))
    MUT5_error_normalized = data_3.iloc[:, 34] / np.max(np.abs(data_3.iloc[:, 31]))
    #MUT4
    MUT6_fit_normalized = data_3.iloc[:, 36] / np.max(np.abs(data_3.iloc[:, 36]))
    MUT6_data_normalized = data_3.iloc[:, 38] / np.max(np.abs(data_3.iloc[:, 36]))
    #MUT5
    #MUT6
    MUT6_error_normalized = data_3.iloc[:, 39] / np.max(np.abs(data_3.iloc[:, 36]))
    return (
        A1_data_normalized,
        A1_error_normalized,
        A1_fit_normalized,
        MUT1_data_normalized,
        MUT1_error_normalized,
        MUT1_fit_normalized,
        MUT2_data_normalized,
        MUT2_error_normalized,
        MUT2_fit_normalized,
        MUT3_data_normalized,
        MUT3_error_normalized,
        MUT3_fit_normalized,
        MUT4_data_normalized,
        MUT4_error_normalized,
        MUT4_fit_normalized,
        MUT5_data_normalized,
        MUT5_error_normalized,
        MUT5_fit_normalized,
        MUT6_data_normalized,
        MUT6_error_normalized,
        MUT6_fit_normalized,
        dye_data_normalized,
        dye_error_normalized,
        dye_fit_normalized,
    )


@app.cell
def _(
    A1_data_normalized,
    A1_error_normalized,
    A1_fit_normalized,
    FIGURE_DPI,
    MUT1_data_normalized,
    MUT1_error_normalized,
    MUT1_fit_normalized,
    MUT2_data_normalized,
    MUT2_error_normalized,
    MUT2_fit_normalized,
    MUT3_data_normalized,
    MUT3_error_normalized,
    MUT3_fit_normalized,
    MUT4_data_normalized,
    MUT4_error_normalized,
    MUT4_fit_normalized,
    MUT5_data_normalized,
    MUT5_error_normalized,
    MUT5_fit_normalized,
    MUT6_data_normalized,
    MUT6_error_normalized,
    MUT6_fit_normalized,
    SAVE_EPS,
    data_3,
    dye_data_normalized,
    dye_error_normalized,
    dye_fit_normalized,
    plt,
    save_figure,
    wetlab_data_plots_dir,
):
    #Plot normalized data
    plt.figure()
    #Dye (Mag-Fura-2)
    plt.plot(data_3.iloc[:, 0], dye_fit_normalized, color='black')  # open a fresh figure so this cell does not draw onto a previous one
    plt.scatter(data_3.iloc[:, 2], dye_data_normalized, color='black', label='Mag-Fura-2')
    plt.errorbar(data_3.iloc[:, 2], dye_data_normalized, yerr=dye_error_normalized, fmt='o', capsize=5, color='black')
    plt.plot(data_3.iloc[:, 5], A1_fit_normalized, color='green')
    plt.scatter(data_3.iloc[:, 7], A1_data_normalized, color='green', label='A1')
    #A1
    plt.errorbar(data_3.iloc[:, 7], A1_data_normalized, yerr=A1_error_normalized, fmt='o', capsize=5, color='green')
    plt.plot(data_3.iloc[:, 10], MUT1_fit_normalized, color='orange')
    plt.scatter(data_3.iloc[:, 12], MUT1_data_normalized, color='orange', label='MUT1')
    plt.errorbar(data_3.iloc[:, 12], MUT1_data_normalized, yerr=MUT1_error_normalized, fmt='o', capsize=5, color='orange')
    #MUT1
    plt.plot(data_3.iloc[:, 15], MUT2_fit_normalized, color='purple')
    plt.scatter(data_3.iloc[:, 17], MUT2_data_normalized, color='purple', label='MUT2')
    plt.errorbar(data_3.iloc[:, 17], MUT2_data_normalized, yerr=MUT2_error_normalized, fmt='o', capsize=5, color='purple')
    plt.plot(data_3.iloc[:, 20], MUT3_fit_normalized, color='blue')
    #MUT2
    plt.scatter(data_3.iloc[:, 22], MUT3_data_normalized, color='blue', label='MUT3')
    plt.errorbar(data_3.iloc[:, 22], MUT3_data_normalized, yerr=MUT3_error_normalized, fmt='o', capsize=5, color='blue')
    plt.plot(data_3.iloc[:, 25], MUT4_fit_normalized, color='pink')
    plt.scatter(data_3.iloc[:, 27], MUT4_data_normalized, color='pink', label='MUT4')
    #MUT3
    plt.errorbar(data_3.iloc[:, 27], MUT4_data_normalized, yerr=MUT4_error_normalized, fmt='o', capsize=5, color='pink')
    plt.plot(data_3.iloc[:, 30], MUT5_fit_normalized, color='red')
    plt.scatter(data_3.iloc[:, 32], MUT5_data_normalized, color='red', label='MUT5')
    plt.errorbar(data_3.iloc[:, 32], MUT5_data_normalized, yerr=MUT5_error_normalized, fmt='o', capsize=5, color='red')
    #MUT4
    plt.plot(data_3.iloc[:, 35], MUT6_fit_normalized, color='lightblue')
    plt.scatter(data_3.iloc[:, 37], MUT6_data_normalized, color='lightblue', label='MUT6')
    plt.errorbar(data_3.iloc[:, 37], MUT6_data_normalized, yerr=MUT6_error_normalized, fmt='o', capsize=5, color='lightblue')
    plt.xlabel('Total metal ion concentration (uM)', fontsize=14)
    #MUT5
    plt.ylabel('Absorbance ratio', fontsize=14)
    plt.title('A1 and Knockouts', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    #MUT6
    plt.legend(fontsize=12)
    save_figure(f'{wetlab_data_plots_dir}zinc_affinity_absorbance_ratio_plot', dpi=FIGURE_DPI, save_eps=SAVE_EPS)
    plt.show()  # PNG always; EPS only if SAVE_EPS. Raise dpi here for one figure.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **IV. TOTAL CATALYTIC TURNOVER DATA**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## IV.I. Total Turnover Experiment for ZETA_1
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### IV.I.A. Progress Curves for the Total Turnover Experiment (Enzyme-Catalyzed, Background, & 4MU Product Standards)
    """)
    return


@app.cell
def _(
    FIGURE_DPI,
    SAVE_EPS,
    np,
    parse_kinetics,
    plt,
    raw_wetlab_data_dir,
    save_figure,
    wetlab_data_plots_dir,
):
    _excel_file = f'{raw_wetlab_data_dir}241008_a1_turnover_100nM_enz_AND_150uM_4muPA_40x_ZINC_blank2ndROW_ladder150uMdown.xlsx'
    _row_column_ranges = {'A': (5, 7), 'B': (5, 7), 'C': (5, 7), 'D': (5, 7), 'E': (5, 7), 'F': (5, 7), 'G': (5, 7), 'H': (5, 7)}
    enzyme_row = 'A'
    replicate_dic_21 = {5: [6, 7]}
    background_row = 'B'
    _standard_wells = [['150uM', ['C5', 'C6', 'C7']], ['75uM', ['D5', 'D6', 'D7']], ['37.5uM', ['E5', 'E6', 'E7']], ['18.75uM', ['F5', 'F6', 'F7']], ['9.375uM', ['G5', 'G6', 'G7']], ['4.6875uM', ['H5', 'H6', 'H7']]]
    max_time = 150
    active_wells_26 = []
    for _row, (_start_col, _end_col) in _row_column_ranges.items():
        for _col in range(_start_col, _end_col + 1):
            active_wells_26.append(f'{_row}{_col}')
    screening_data_1 = parse_kinetics(_excel_file, active_wells_26, start_row=48)
    screening_data_1['Time'] = screening_data_1['Time'] / 60
    screening_data_1 = screening_data_1[screening_data_1['Time'] < max_time]
    plt.figure(figsize=(6, 5.5))
    for _representive_col in replicate_dic_21:
        _replicates = [_representive_col]
        _replicates.extend(replicate_dic_21[_representive_col])
        _replicates_well = [f'{enzyme_row}{col}' for col in _replicates]
        _avg_product_concentration = np.array(screening_data_1[_replicates_well].mean(axis=1).tolist())
        _std_product_concentration = np.array(screening_data_1[_replicates_well].std(axis=1).tolist())
        plt.plot(screening_data_1['Time'].values, _avg_product_concentration, label='A1')
        plt.fill_between(screening_data_1['Time'].values, _avg_product_concentration - _std_product_concentration, _avg_product_concentration + _std_product_concentration, alpha=0.45)
        _replicates_well = [f'{background_row}{col}' for col in _replicates]
        _avg_product_concentration = np.array(screening_data_1[_replicates_well].mean(axis=1).tolist())
        _std_product_concentration = np.array(screening_data_1[_replicates_well].std(axis=1).tolist())
        plt.plot(screening_data_1['Time'].values, _avg_product_concentration, linestyle='dashed', label='background')
        plt.fill_between(screening_data_1['Time'].values, _avg_product_concentration - _std_product_concentration, _avg_product_concentration + _std_product_concentration, alpha=0.45)
        for product_label, _replicates_well in _standard_wells:
            _avg_product_concentration = np.array(screening_data_1[_replicates_well].mean(axis=1).tolist())
            _std_product_concentration = np.array(screening_data_1[_replicates_well].std(axis=1).tolist())
            plt.plot(screening_data_1['Time'].values, _avg_product_concentration, label=product_label)
            plt.fill_between(screening_data_1['Time'].values, _avg_product_concentration - _std_product_concentration, _avg_product_concentration + _std_product_concentration, alpha=0.45)
        plt.ylabel('Fluorescence Units')
        plt.xlabel('Time (min)')
    plt.legend()
    plt.tight_layout()
    save_figure(f'{wetlab_data_plots_dir}241009_A1_turnover_experiment', dpi=FIGURE_DPI, save_eps=SAVE_EPS)
    plt.show()
    return (
        active_wells_26,
        background_row,
        enzyme_row,
        replicate_dic_21,
        screening_data_1,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### IV.I.B. Product Fluorescence Standard Curve
    """)
    return


@app.cell
def _(
    FIGURE_DPI,
    SAVE_EPS,
    np,
    plt,
    save_figure,
    screening_data_1,
    wetlab_data_plots_dir,
):
    from scipy.stats import pearsonr
    calibration_x = []
    calibration_y = []
    calibration_y_std = []
    _enzyme_concentration = 0.1
    _standard_wells = [['150uM', ['C5', 'C6', 'C7']], ['75uM', ['D5', 'D6', 'D7']], ['37.5uM', ['E5', 'E6', 'E7']], ['18.75uM', ['F5', 'F6', 'F7']], ['9.375uM', ['G5', 'G6', 'G7']], ['4.6875uM', ['H5', 'H6', 'H7']]]
    for product_concentration, well_list in _standard_wells:
        calibration_x.append(float(product_concentration[:-2]))
        calibration_y.append(np.mean(np.array(screening_data_1[well_list].mean(axis=0).tolist())))
        calibration_y_std.append(np.std(np.array(screening_data_1[well_list].mean(axis=0).tolist())))
    plt.figure()
    plt.errorbar(calibration_x, calibration_y, yerr=calibration_y_std, fmt='o', capsize=5)
    "\nx_fit = np.linspace(min(calibration_x), max(calibration_x), 100)\ny_fit = poly(x_fit)\nplt.plot(y_fit, x_fit, color='red', label='Fitted Line')\n"
    _coefficients = np.polyfit(calibration_x, calibration_y, 4)
    poly = np.poly1d(_coefficients)
    fluorescence_to_4mu = np.poly1d(np.polyfit(calibration_y, calibration_x, 4))
    x_fit = np.linspace(min(calibration_x), max(calibration_x), 100)
    y_fit = poly(x_fit)
    plt.plot(x_fit, y_fit, color='red', label='Fitted Line')
    calibration_residuals = np.asarray(calibration_y) - poly(calibration_x)
    calibration_r_squared = 1 - np.sum(calibration_residuals ** 2) / np.sum((np.asarray(calibration_y) - np.mean(calibration_y)) ** 2)
    print('Calibration R-squared:', calibration_r_squared)
    plt.ylabel('Fluorescence Units')
    plt.xlabel('[Product], uM')
    save_figure(f'{wetlab_data_plots_dir}241009_calibration_curve_for_turnover_number_experiment', dpi=FIGURE_DPI, save_eps=SAVE_EPS)
    plt.show()
    return (fluorescence_to_4mu,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### IV.I.C. Turnover Number Plot

    Convert fluorescence to 4MU concentration before dividing by the enzyme
    concentration. The first plot retains enzyme and background traces separately;
    the second subtracts the corresponding background replicate after conversion,
    as in Fig. 3e and Supplementary Methods §4.3. The quartic calibration is specific
    to this plate; values near its limits are estimates, not an extension of the
    instrument's validated range.
    """)
    return


@app.cell
def _(
    FIGURE_DPI,
    SAVE_EPS,
    active_wells_26,
    background_row,
    enzyme_row,
    fluorescence_to_4mu,
    np,
    pd,
    plt,
    replicate_dic_21,
    save_figure,
    screening_data_1,
    wetlab_data_plots_dir,
):
    _enzyme_concentration = 0.1
    converted_screening_data = pd.DataFrame()
    converted_screening_data['Time'] = screening_data_1['Time']
    for _well in active_wells_26:
        converted_screening_data[_well] = fluorescence_to_4mu(screening_data_1[_well].to_numpy(dtype=float)) / _enzyme_concentration
    converted_screening_data = converted_screening_data[converted_screening_data['Time'] < 120]
    plt.figure(figsize=(6, 5.5))
    for _representive_col in replicate_dic_21:
        _replicates = [_representive_col]
        _replicates.extend(replicate_dic_21[_representive_col])
        _replicates_well = [f'{enzyme_row}{col}' for col in _replicates]
        bg_replicates_well = [f'{background_row}{col}' for col in _replicates]
        _avg_product_concentration = np.array(converted_screening_data[_replicates_well].mean(axis=1).tolist())
        _std_product_concentration = np.array(converted_screening_data[_replicates_well].std(axis=1).tolist())
        plt.plot(converted_screening_data['Time'].values, _avg_product_concentration, label='A1', color='black')
        plt.fill_between(converted_screening_data['Time'].values, _avg_product_concentration - _std_product_concentration, _avg_product_concentration + _std_product_concentration, alpha=0.45, color='gray')
        _avg_product_concentration = np.array(converted_screening_data[bg_replicates_well].mean(axis=1).tolist())
        _std_product_concentration = np.array(converted_screening_data[bg_replicates_well].std(axis=1).tolist())
        plt.plot(converted_screening_data['Time'].values, _avg_product_concentration, linestyle='dashed', label='background', color='black')
        plt.fill_between(converted_screening_data['Time'].values, _avg_product_concentration - _std_product_concentration, _avg_product_concentration + _std_product_concentration, alpha=0.45, color='gray')
        plt.ylabel('Turnover Number')
        plt.xlabel('Time (min)')
    plt.xlim(-2, 122)
    plt.tight_layout()
    save_figure(f'{wetlab_data_plots_dir}241009_A1_turnover_experiment_y_axis_turnover_not_zeroing', dpi=FIGURE_DPI, save_eps=SAVE_EPS)
    plt.show()
    enzyme_wells = [f'{enzyme_row}{col}' for col in _replicates]
    background_wells = [f'{background_row}{col}' for col in _replicates]
    net_turnover = converted_screening_data[enzyme_wells].to_numpy() - converted_screening_data[background_wells].to_numpy()
    net_turnover_mean = net_turnover.mean(axis=1)
    net_turnover_sd = net_turnover.std(axis=1, ddof=1)
    plt.figure(figsize=(6, 5.5))
    plt.plot(converted_screening_data['Time'], net_turnover_mean, color='black')
    plt.fill_between(converted_screening_data['Time'], net_turnover_mean - net_turnover_sd, net_turnover_mean + net_turnover_sd, color='gray', alpha=0.45)
    plt.xlabel('Time (min)')
    plt.ylabel('Background-subtracted turnover number')
    plt.xlim(-2, 122)
    plt.tight_layout()
    save_figure(f'{wetlab_data_plots_dir}241009_A1_turnover_background_subtracted', dpi=FIGURE_DPI, save_eps=SAVE_EPS)
    plt.show()
    print(f"Net turnover at {converted_screening_data['Time'].iloc[-1]:.1f} min: {net_turnover_mean[-1]:.0f} ± {net_turnover_sd[-1]:.0f} (mean ± SD, n=3)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **V. BIOPHYSICAL CHARACTERIZATION DATA**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## V.I. Circular Dichroism (CD) Spectra for Best Design from Each Unique Scaffold Hit
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### V.I.A. A1 ([Enz]: 0.5mg/ml (27.5 uM), 25mM Tris, 25mM NaCl, Temp: 25C)
    """)
    return


@app.cell
def _(
    FIGURE_DPI,
    SAVE_EPS,
    os,
    plot_CD_spectrum_temperature_interval_A1,
    raw_wetlab_data_dir,
    wetlab_data_plots_dir,
):
    ### Input ###
    _CD_result_path = raw_wetlab_data_dir
    _file_name = 'A1_25mMtris_25mMnacl_ph8_noAddedZn__25C_to_95C_1nmBand_Spectrum.csv'
    file_name2 = 'A1_25mMtris_25mMnacl_ph8_25C___recoveryAFTER95C-x-1.csv'
    _pathlength = 1
    _protein_concentration = 27.5 * 1e-06  # Unit: mm
    _protein_length = 162
    _save_eps_path = os.path.join(wetlab_data_plots_dir, '250326_CD_temp_interval_A1_25mMtris_25mMnacl_ph8_noAddedZn')
    _wavelength_min = 200
    ### Output ###
    _window_size = 5
    _colors = ['#371043', '#36296b', '#2b487b', '#246c7b', '#1f9375', '#44b759', '#95c93e', '#fbe51c']
    ## Parameter ###
    plot_CD_spectrum_temperature_interval_A1(os.path.join(_CD_result_path, _file_name), os.path.join(_CD_result_path, file_name2), _protein_concentration, _protein_length, pathlength=_pathlength, wavelength_min=_wavelength_min, window_size=_window_size, colors=_colors, save=_save_eps_path, dpi=FIGURE_DPI, save_eps=SAVE_EPS)
    return


@app.cell
def _(
    FIGURE_DPI,
    SAVE_EPS,
    os,
    plot_CD_222nm_temperature_interval,
    raw_wetlab_data_dir,
    wetlab_data_plots_dir,
):
    ### Input ###
    _CD_result_path = raw_wetlab_data_dir
    _file_name = 'A1_25mMtris_25mMnacl_ph8_noAddedZn__25C_to_95C_1nmBand_222nmMeasurement.csv'
    _pathlength = 1
    _protein_concentration = 27.5 * 1e-06  # Unit: mm
    _protein_length = 162
    _save_eps_path = os.path.join(wetlab_data_plots_dir, '250326_CD_temp_interval_222nm_A1_25mMtris_25mMnacl_ph8_noAddedZn')
    ### Output ###
    ## Parameter ###
    plot_CD_222nm_temperature_interval(os.path.join(_CD_result_path, _file_name), _protein_concentration, _protein_length, window_size=3, pathlength=_pathlength, save=_save_eps_path, dpi=FIGURE_DPI, save_eps=SAVE_EPS)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### V.I.B. 250328 A8 ([Enz]: 19.3 uM, 25mM Tris, 25mM NaCl, Temp: 25C)
    """)
    return


@app.cell
def _(
    FIGURE_DPI,
    SAVE_EPS,
    os,
    plot_CD_spectrum_temperature_interval,
    raw_wetlab_data_dir,
    wetlab_data_plots_dir,
):
    ### Input ###
    _CD_result_path = raw_wetlab_data_dir
    _file_name = 'A8_25mMtris_25mMnacl_ph8_noAddedZn__25C_to_95C_1nmBand_Spectrum.csv'
    _pathlength = 1
    _protein_concentration = 19.3 * 1e-06  # Unit: mm
    _protein_length = 220
    _save_eps_path = os.path.join(wetlab_data_plots_dir, '250328_CD_temp_interval_A8_25mMtris_25mMnacl_ph8_noAddedZn')
    _wavelength_min = 200
    ### Output ###
    _window_size = 5
    _colors = ['#371043', '#36296b', '#2b487b', '#246c7b', '#1f9375', '#44b759', '#95c93e', '#fbe51c']
    ## Parameter ###
    plot_CD_spectrum_temperature_interval(os.path.join(_CD_result_path, _file_name), _protein_concentration, _protein_length, pathlength=_pathlength, wavelength_min=_wavelength_min, window_size=_window_size, colors=_colors, save=_save_eps_path, dpi=FIGURE_DPI, save_eps=SAVE_EPS)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### V.I.C. 250328 B9 ([Enz]: 22.4 uM, 25mM Tris, 25mM NaCl, Temp: 25C)
    """)
    return


@app.cell
def _(
    FIGURE_DPI,
    SAVE_EPS,
    os,
    plot_CD_spectrum_temperature_interval,
    raw_wetlab_data_dir,
    wetlab_data_plots_dir,
):
    ### Input ###
    _CD_result_path = raw_wetlab_data_dir
    _file_name = 'B9_25mMtris_25mMnacl_ph8_noAddedZn__25C_to_95C_1nmBand_Spectrum.csv'
    _pathlength = 1
    _protein_concentration = 22.4 * 1e-06  # Unit: mm
    _protein_length = 182
    _save_eps_path = os.path.join(wetlab_data_plots_dir, '250328_CD_temp_interval_B9_25mMtris_25mMnacl_ph8_noAddedZn')
    _wavelength_min = 200
    ### Output ###
    _window_size = 5
    _colors = ['#371043', '#36296b', '#2b487b', '#246c7b', '#1f9375', '#44b759', '#95c93e', '#fbe51c']
    ## Parameter ###
    plot_CD_spectrum_temperature_interval(os.path.join(_CD_result_path, _file_name), _protein_concentration, _protein_length, pathlength=_pathlength, wavelength_min=_wavelength_min, window_size=_window_size, colors=_colors, save=_save_eps_path, dpi=FIGURE_DPI, save_eps=SAVE_EPS)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### V.I.D. 250328 C4 ([Enz]: 25.7 uM, 25mM Tris, 25mM NaCl, Temp: 25C)
    """)
    return


@app.cell
def _(
    FIGURE_DPI,
    SAVE_EPS,
    os,
    plot_CD_spectrum_temperature_interval,
    raw_wetlab_data_dir,
    wetlab_data_plots_dir,
):
    ### Input ###
    _CD_result_path = raw_wetlab_data_dir
    _file_name = 'C4pa_25mMtris_25mMnacl_ph8_noAddedZn__25C_to_95C_1nmBand_Spectrum.csv'
    _pathlength = 1
    _protein_concentration = 25.7 * 1e-06  # Unit: mm
    _protein_length = 168
    _save_eps_path = os.path.join(wetlab_data_plots_dir, '250328_CD_temp_interval_C4_25mMtris_25mMnacl_ph8_noAddedZn')
    _wavelength_min = 200
    ### Output ###
    _window_size = 5
    _colors = ['#371043', '#36296b', '#2b487b', '#246c7b', '#1f9375', '#44b759', '#95c93e', '#fbe51c']
    ## Parameter ###
    plot_CD_spectrum_temperature_interval(os.path.join(_CD_result_path, _file_name), _protein_concentration, _protein_length, pathlength=_pathlength, wavelength_min=_wavelength_min, window_size=_window_size, colors=_colors, save=_save_eps_path, dpi=FIGURE_DPI, save_eps=SAVE_EPS)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### V.I.E. 250328 F7 ([Enz]: 28.1 uM, 25mM Tris, 25mM NaCl, Temp: 25C)
    """)
    return


@app.cell
def _(
    FIGURE_DPI,
    SAVE_EPS,
    os,
    plot_CD_spectrum_temperature_interval,
    raw_wetlab_data_dir,
    wetlab_data_plots_dir,
):
    ### Input ###
    _CD_result_path = raw_wetlab_data_dir
    _file_name = 'F7_25mMtris_25mMnacl_ph8_noAddedZn__25C_to_95C_1nmBand_Spectrum.csv'
    _pathlength = 1
    _protein_concentration = 28.1 * 1e-06  # Unit: mm
    _protein_length = 185
    _save_eps_path = os.path.join(wetlab_data_plots_dir, '250328_CD_temp_interval_F7_25mMtris_25mMnacl_ph8_noAddedZn')
    _wavelength_min = 200
    ### Output ###
    _window_size = 5
    _colors = ['#371043', '#36296b', '#2b487b', '#246c7b', '#1f9375', '#44b759', '#95c93e', '#fbe51c']
    ## Parameter ###
    plot_CD_spectrum_temperature_interval(os.path.join(_CD_result_path, _file_name), _protein_concentration, _protein_length, pathlength=_pathlength, wavelength_min=_wavelength_min, window_size=_window_size, colors=_colors, save=_save_eps_path, dpi=FIGURE_DPI, save_eps=SAVE_EPS)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### V.I.F. 250328 H10 ([Enz]: 33.1 uM, 25mM Tris, 25mM NaCl, Temp: 25C)
    """)
    return


@app.cell
def _(
    FIGURE_DPI,
    SAVE_EPS,
    os,
    plot_CD_spectrum_temperature_interval,
    raw_wetlab_data_dir,
    wetlab_data_plots_dir,
):
    ### Input ###
    _CD_result_path = raw_wetlab_data_dir
    _file_name = 'H10_25mMtris_25mMnacl_ph8_noAddedZn__25C_to_95C_1nmBand_Spectrum.csv'
    _pathlength = 1
    _protein_concentration = 33.1 * 1e-06  # Unit: mm
    _protein_length = 138
    _save_eps_path = os.path.join(wetlab_data_plots_dir, '250328_CD_temp_interval_H10_25mMtris_25mMnacl_ph8_noAddedZn')
    _wavelength_min = 200
    ### Output ###
    _window_size = 5
    _colors = ['#371043', '#36296b', '#2b487b', '#246c7b', '#1f9375', '#44b759', '#95c93e', '#fbe51c']
    ## Parameter ###cc
    plot_CD_spectrum_temperature_interval(os.path.join(_CD_result_path, _file_name), _protein_concentration, _protein_length, pathlength=_pathlength, wavelength_min=_wavelength_min, window_size=_window_size, colors=_colors, save=_save_eps_path, dpi=FIGURE_DPI, save_eps=SAVE_EPS)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### V.I.G. 250328 H6 ([Enz]: 39.4 uM, 25mM Tris, 25mM NaCl, Temp: 25C)
    """)
    return


@app.cell
def _(
    FIGURE_DPI,
    SAVE_EPS,
    os,
    plot_CD_spectrum_temperature_interval,
    raw_wetlab_data_dir,
    wetlab_data_plots_dir,
):
    ### Input ###
    _CD_result_path = raw_wetlab_data_dir
    _file_name = 'H6_25mMtris_25mMnacl_ph8_noAddedZn__25C_to_95C_1nmBand_Spectrum.csv'
    _pathlength = 1
    _protein_concentration = 39.4 * 1e-06  # Unit: mm
    _protein_length = 184
    _save_eps_path = os.path.join(wetlab_data_plots_dir, '250328_CD_temp_interval_H6_25mMtris_25mMnacl_ph8_noAddedZn')
    _wavelength_min = 200
    ### Output ###
    _window_size = 5
    _colors = ['#371043', '#36296b', '#2b487b', '#246c7b', '#1f9375', '#44b759', '#95c93e', '#fbe51c']
    ## Parameter ###
    plot_CD_spectrum_temperature_interval(os.path.join(_CD_result_path, _file_name), _protein_concentration, _protein_length, pathlength=_pathlength, wavelength_min=_wavelength_min, window_size=_window_size, colors=_colors, save=_save_eps_path, dpi=FIGURE_DPI, save_eps=SAVE_EPS)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### V.I.H. 250416 C5 ([Enz]: 16.5 uM, 25mM Tris, 25mM NaCl, Temp: 25C)
    """)
    return


@app.cell
def _(
    FIGURE_DPI,
    SAVE_EPS,
    os,
    plot_CD_spectrum_temperature_interval,
    raw_wetlab_data_dir,
    wetlab_data_plots_dir,
):
    ### Input ###
    _CD_result_path = raw_wetlab_data_dir
    _file_name = 'C5_25mMtris_25mMnacl_ph8_noAddedZn__25C_to_95C_1nmBand_Spectrum.csv'
    _pathlength = 1
    _protein_concentration = 16.5 * 1e-06  # Unit: mm
    _protein_length = 192
    _save_eps_path = os.path.join(wetlab_data_plots_dir, '250416_CD_temp_interval_C5_25mMtris_25mMnacl_ph8_noAddedZn')
    _wavelength_min = 200
    ### Output ###
    _window_size = 5
    _colors = ['#371043', '#36296b', '#2b487b', '#246c7b', '#1f9375', '#44b759', '#95c93e', '#fbe51c']
    ## Parameter ###
    plot_CD_spectrum_temperature_interval(os.path.join(_CD_result_path, _file_name), _protein_concentration, _protein_length, pathlength=_pathlength, wavelength_min=_wavelength_min, window_size=_window_size, colors=_colors, save=_save_eps_path, dpi=FIGURE_DPI, save_eps=SAVE_EPS)
    return


if __name__ == "__main__":
    app.run()
