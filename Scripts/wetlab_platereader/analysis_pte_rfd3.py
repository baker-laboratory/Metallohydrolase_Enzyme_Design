"""Reproducible RFdiffusion3 paraoxon analysis from deposited measurements.

The marimo app and Python callers use these ordinary functions. No Jupyter
notebook, saved cell output, or precomputed fit table is read to fit the data.
Fitting windows, calibration constants and uncertainty budgets match the
historical analysis; the reference workbook is used only for reporting/checks.
"""
from pathlib import Path
import os
from contextlib import redirect_stdout
from functools import wraps
from io import StringIO
import re
import numpy as np
import pandas as pd
from scipy import stats
from kinetics import parse_standard_curve
from neo2_util import parse_neo2_kinetics
from kinetics_pte_rfd3 import (
    SPU_BIC, SPU_NOBIC, paper_style, result_row, validate_reference,
    reproduce_paper_figures, run_kinetics_PTE_RFd3,
    load_screening_PTE_RFd3, plot_screening_PTE_RFd3,
)

DATASET = Path(__file__).resolve().parents[2] / "Manuscript_Data" / "Phosphotriesterase_RFdiffusion3_Science_2026"
RAW = DATASET / "raw_wetlab_data"
PLOTS = DATASET / "wetlab_data_plots"



def _quiet_reader(function):
    """Keep verbose legacy reader messages out of reactive plot/table cells."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        with redirect_stdout(StringIO()):
            return function(*args, **kwargs)
    return wrapped


def raw_file(prefix):
    """Resolve one deposited acquisition, independent of the working directory."""
    matches = sorted(RAW.glob(prefix + "*"))
    if len(matches) != 1:
        raise ValueError(f"Expected one acquisition with prefix {prefix!r}, found {len(matches)}")
    return str(matches[0])


@_quiet_reader
def fit_calibration():
    """Fit all four 4-nitrophenol standards from the raw plate."""
    concentrations = [200, 100, 50, 25, 12.5, 6.25, 3.175, 1.5875]
    conditions = [("legacy buffer, no NaHCO3", [1, 2, 3]),
                  ("legacy buffer + 25 mM NaHCO3", [4, 5, 6]),
                  ("modern buffer, no NaHCO3", [7, 8, 9]),
                  ("modern buffer + 25 mM NaHCO3", [10, 11, 12])]
    rows = []
    for name, columns in conditions:
        slope, _ = parse_standard_curve(
            data_path=raw_file("260225_StdCurve_200uMserials_vert__80uL"),
            blocks=[dict(rows=list("ABCDEFGH"), concentrations_uM=concentrations,
                         replicate_cols=columns)], force_zero_intercept=False, plot=False)
        rows.append({"condition": name, "signal_per_uM": slope})
    result = pd.DataFrame(rows)
    # Preserve the calibration precision used by the published kinetic fits.
    np.testing.assert_allclose(result.signal_per_uM.iloc[:2], [SPU_NOBIC, SPU_BIC],
                               rtol=0, atol=0.5e-8)
    return result



def fit_background():
    """Refit the background reaction over all original admissible windows."""
    kuncat_file = raw_file('260807_kuncat_legacy_buffer')
    ROWS_K = list("ABCDEF")                                   # one row per [paraoxon]
    CONC_K = np.array([9600, 4800, 2400, 1200, 600, 300], float)   # uM
    KCFG = {"no NaHCO3":     dict(cols=[1, 2, 3], spu=SPU_NOBIC),
            "+25 mM NaHCO3": dict(cols=[4, 5, 6], spu=SPU_BIC)}

    ### LOAD AND RESHAPE THE TIME COURSE ###
    _raw_kuncat = parse_neo2_kinetics(kuncat_file)
    _d = _raw_kuncat.groupby(["Well", "time"], as_index=False).agg(signal=("value", "mean"))
    _d["time"] = _d["time"].astype(float) - _d.groupby("Well")["time"].transform("min")
    KPIV = _d.pivot(index="time", columns="Well", values="signal").sort_index()
    KT = KPIV.index.to_numpy(float)

    ### FITTING ###
    def kuncat_block(condition, lo, hi):
        """Mean v0 per [S] over one fit window, plus the free-intercept fit."""
        cfg = KCFG[condition]
        m = (KT >= lo) & (KT <= hi)
        tt = KT[m]
        reps = [np.polyfit(tt, np.column_stack([KPIV[f"{r}{c}"].to_numpy(float)[m]
                for c in cfg["cols"]]), 1)[0] / cfg["spu"] for r in ROWS_K]
        V = np.array([x.mean() for x in reps])
        E = np.array([x.std(ddof=1) / np.sqrt(len(x)) for x in reps])
        k, c = np.polyfit(CONC_K, V, 1)
        pred = k * CONC_K + c
        ss = np.sum((V - V.mean()) ** 2)
        return k, c, (1 - np.sum((V - pred) ** 2) / ss if ss > 0 else np.nan), V, E

    def kuncat_report(condition, grid=300., min_duration=2700., min_start=900.):
        """Median k_uncat over every admissible fit window, with combined error."""
        accepted = []
        for lo in np.arange(min_start, KT.max() - min_duration + 1, grid):
            for hi in np.arange(lo + min_duration, KT.max() + 1, grid):
                k, c, r2, V, E = kuncat_block(condition, lo, hi)
                if r2 >= 0.90:
                    # leave-one-out spread across the six substrate concentrations
                    jk = np.array([np.polyfit(np.delete(CONC_K, i), np.delete(V, i), 1)[0]
                                   for i in range(len(CONC_K))])
                    accepted.append((lo, hi, k, np.sqrt(5/6 * np.sum((jk - jk.mean()) ** 2))))
        A = np.array(accepted)
        k = A[:, 2]
        median = np.median(k)
        p16, p84 = np.percentile(k, [16, 84])
        stat = np.median(A[:, 3])
        err = max(np.hypot(median - p16, stat), np.hypot(p84 - median, stat))
        best = A[np.argmin(np.abs(k - median))]        # the window closest to the median
        return dict(k=median, err=err, n=len(A), lo=best[0], hi=best[1])

    ### EXECUTION ###
    KUNCAT = {c: kuncat_report(c) for c in KCFG}

    return KUNCAT



KINETICS_SPECS = [{'name': 'ZAPP-1',
  'file_prefix': '260804_ZAPP1strp_24uM',
  'enzyme_uM': 24.0,
  'enzyme_cols': [1, 2, 3],
  'bg_cols': [4],
  'scaffold': 'PET_i1',
  'plate_id': 'p1D1',
  'plot_name': 'kinetics_ZAPP1_p1D1'},
 {'name': 'ZAPP-1 (no NaHCO3)',
  'file_prefix': '260225_p1D1_30pt9uM',
  'enzyme_uM': 37.3,
  'enzyme_cols': [1, 2, 3],
  'bg_cols': [4],
  'condition': 'no NaHCO3',
  'scaffold': 'PET_i1',
  'plate_id': 'p1D1',
  'plot_name': 'kinetics_ZAPP1_p1D1_no_bicarbonate'},
 {'name': 'R1 p1D7',
  'file_prefix': '260224_p1D7_21pt3uM',
  'enzyme_uM': 21.3,
  'enzyme_cols': [1, 2, 3],
  'bg_cols': [4],
  'scaffold': 'PET_i1',
  'plate_id': 'p1D7',
  'plot_name': 'kinetics_R1_p1D7'},
 {'name': 'R1 p1D8',
  'file_prefix': '260224_p1D1_26pt2uM',
  'enzyme_uM': 26.1,
  'enzyme_cols': [5, 6, 7],
  'bg_cols': [8],
  'scaffold': 'PET_i1',
  'plate_id': 'p1D8',
  'plot_name': 'kinetics_R1_p1D8'},
 {'name': 'R1 p1E10',
  'file_prefix': '260224_p1D7_21pt3uM',
  'enzyme_uM': 15.3,
  'enzyme_cols': [5, 6, 7],
  'bg_cols': [8],
  'scaffold': 'PET_i1',
  'plate_id': 'p1E10',
  'plot_name': 'kinetics_R1_p1E10'},
 {'name': 'ZAPP-2',
  'file_prefix': '260731_p1F6_34pt8uM',
  'enzyme_uM': 26.6,
  'enzyme_cols': [5, 6, 7],
  'bg_cols': [8],
  'time_range_seconds': (300, 660),
  'scaffold': 'PTE_i2_s3',
  'plate_id': 'p2C4',
  'plot_name': 'kinetics_ZAPP2_p2C4'},
 {'name': 'R2 p2B12',
  'file_prefix': '260731_p1F6_34pt8uM',
  'enzyme_uM': 10.3,
  'enzyme_cols': [9, 10, 11],
  'bg_cols': [12],
  'time_range_seconds': (300, 660),
  'scaffold': 'PTE_i2_s3',
  'plate_id': 'p2B12',
  'plot_name': 'kinetics_R2_p2B12'},
 {'name': 'ZAPP-3',
  'file_prefix': '260731_p2E3_28pt9uM',
  'enzyme_uM': 27.5,
  'enzyme_cols': [5, 6, 7],
  'bg_cols': [8],
  'scaffold': 'PTE_i2_s7',
  'plate_id': 'p2G1',
  'plot_name': 'kinetics_ZAPP3_p2G1'},
 {'name': 'ZAPP-4',
  'file_prefix': '260731_p2E3_28pt9uM',
  'enzyme_uM': 28.9,
  'enzyme_cols': [1, 2, 3],
  'bg_cols': [4],
  'scaffold': 'PTE_i2_s6',
  'plate_id': 'p2E3',
  'plot_name': 'kinetics_ZAPP4_p2E3'},
 {'name': 'ZAPP-5',
  'file_prefix': '260731_p1F6_34pt8uM',
  'enzyme_uM': 34.8,
  'enzyme_cols': [1, 2, 3],
  'bg_cols': [4],
  'time_range_seconds': (300, 660),
  'scaffold': 'PTE_i2_s1',
  'plate_id': 'p1F6',
  'plot_name': 'kinetics_ZAPP5_p1F6'},
 {'name': 'R2 p1D9',
  'file_prefix': '260804_ZAPP1strp_24uM',
  'enzyme_uM': 44.6,
  'enzyme_cols': [5, 6, 7],
  'bg_cols': [8],
  'scaffold': 'PTE_i2_s1',
  'plate_id': 'p1D9',
  'plot_name': 'kinetics_R2_p1D9'},
 {'name': 'R2 p1H4',
  'file_prefix': '260804_ZAPP1strp_24uM',
  'enzyme_uM': 39.1,
  'enzyme_cols': [9, 10, 11],
  'bg_cols': [12],
  'scaffold': 'PTE_i2_s1',
  'plate_id': 'p1H4',
  'plot_name': 'kinetics_R2_p1H4'},
 {'name': 'ZAPP-1 MUT1',
  'file_prefix': '260804_p2H3_4pt6uM',
  'enzyme_uM': 32.7,
  'enzyme_cols': [5, 6, 7],
  'bg_cols': [8],
  'scaffold': 'PET_i1',
  'plate_id': 'p1D1',
  'plot_name': 'kinetics_ZAPP1_MUT1'},
 {'name': 'ZAPP-1 MUT2',
  'file_prefix': '260804_p2H3_4pt6uM',
  'enzyme_uM': 25.8,
  'enzyme_cols': [9, 10, 11],
  'bg_cols': [12],
  'scaffold': 'PET_i1',
  'plate_id': 'p1D1',
  'plot_name': 'kinetics_ZAPP1_MUT2'},
 {'name': 'ZAPP-1 MUT3',
  'file_prefix': '260804_ZAPPmut3_28pt1uM',
  'enzyme_uM': 28.1,
  'enzyme_cols': [1, 2, 3],
  'bg_cols': [4],
  'scaffold': 'PET_i1',
  'plate_id': 'p1D1',
  'plot_name': 'kinetics_ZAPP1_MUT3'},
 {'name': 'ZAPP-1 MUT4',
  'file_prefix': '260804_ZAPPmut3_28pt1uM',
  'enzyme_uM': 18.7,
  'enzyme_cols': [5, 6, 7],
  'bg_cols': [8],
  'scaffold': 'PET_i1',
  'plate_id': 'p1D1',
  'plot_name': 'kinetics_ZAPP1_MUT4'},
 {'name': 'ZAPP-1 MUT5',
  'file_prefix': '260804_ZAPPmut3_28pt1uM',
  'enzyme_uM': 39.5,
  'enzyme_cols': [9, 10, 11],
  'bg_cols': [12],
  'scaffold': 'PET_i1',
  'plate_id': 'p1D1',
  'plot_name': 'kinetics_ZAPP1_MUT5'},
 {'name': 'ZAPP-1 MUT6',
  'file_prefix': '260804_ZAPPmut6_23pt2uM',
  'enzyme_uM': 23.2,
  'enzyme_cols': [1, 2, 3],
  'bg_cols': [4],
  'scaffold': 'PET_i1',
  'plate_id': 'p1D1',
  'plot_name': 'kinetics_ZAPP1_MUT6'},
 {'name': 'ZAPP-1 tagless',
  'file_prefix': '260731_p2E3_28pt9uM',
  'enzyme_uM': 23.4,
  'enzyme_cols': [9, 10, 11],
  'bg_cols': [12],
  'scaffold': 'PET_i1',
  'plate_id': 'p1D1',
  'plot_name': 'kinetics_ZAPP1_tagless'}]



@_quiet_reader
def fit_kinetics():
    """Fresh fits for all 19 enzyme/condition entries, including tagless ZAPP-1."""
    fits = {}
    for config in KINETICS_SPECS:
        spec = dict(config)
        name = spec.pop("name")
        path = raw_file(spec.pop("file_prefix"))
        scaffold, plate = spec.pop("scaffold"), spec.pop("plate_id")
        spec.pop("plot_name")
        condition = spec.get("condition", "+25 mM NaHCO3")
        mm = run_kinetics_PTE_RFd3(name, data_path=path, show=False, **spec)
        fits[name] = dict(mm=mm, condition=condition, enzyme_uM=spec["enzyme_uM"],
                          scaffold=scaffold, plate_id=plate)
    return fits


def fresh_result_row(name, fit, background):
    """Calculate display parameters from the current fit and matched background.

    The two unresolved mutants retain the established first-order reporting
    rule; their slopes and uncertainties are recalculated from raw fit points.
    No parameter or formatted value is taken from the reference workbook.
    """
    mm = fit['mm']
    row = result_row(name, mm, fit['condition'], background[fit['condition']])
    row['class'] = ('mutant' if 'MUT' in name else
                    'tagless comparison' if name == 'ZAPP-1 tagless' else 'enzyme')
    row['reporting model'] = 'Michaelis-Menten'
    if 'MUT' in name:
        agg = mm['_agg']
        substrate = agg['concentration_uM'].to_numpy(float)
        rate = agg['v_mean_uM_per_s'].to_numpy(float) / fit['enzyme_uM']
        slope = float(substrate @ rate / (substrate @ substrate))
        se = np.sqrt(np.sum((rate-slope*substrate)**2) /
                     (len(substrate)-1) / (substrate @ substrate))
        row['kcat/Km chord (M-1 s-1)'] = slope * 1e6
        row['kcat/Km chord sd'] = np.hypot(se, .06*slope) * 1e6
        row['kcat/Km chord (reported)'] = paper_style().plain(
            row['kcat/Km chord (M-1 s-1)'], row['kcat/Km chord sd'])
    if name in ('ZAPP-1 MUT2', 'ZAPP-1 MUT4'):
        row['kcat (reported)'] = 'n.d.'
        row['Km (reported, mM)'] = 'n.d.'
        row['kcat/Km (reported)'] = row['kcat/Km chord (reported)']
        row['reporting model'] = 'first-order; kcat and KM not resolved'
    return row


@_quiet_reader
def kinetics_figure(name, fits, background):
    """Draw current fit points/parameters, with no reference-table overlay."""
    import matplotlib.pyplot as plt
    from kinetics_pte_rfd3 import _standalone_legend

    config = next(s for s in KINETICS_SPECS if s['name'] == name)
    fit = fits[name]
    parameters = fresh_result_row(name, fit, background)
    sp = dict(path=raw_file(config['file_prefix']), E=fit['enzyme_uM'],
              cols=config['enzyme_cols'], bg=config['bg_cols'],
              tr=config.get('time_range_seconds', (300, 20000)))
    spu = SPU_NOBIC if fit['condition'] == 'no NaHCO3' else SPU_BIC
    aggregate = fit['mm']['_agg'].sort_values('concentration_uM')
    points = (aggregate['concentration_uM'].to_numpy(float),
              aggregate['v_mean_uM_per_s'].to_numpy(float) / fit['enzyme_uM'],
              aggregate['v_std_uM_per_s'].fillna(0).to_numpy(float) /
              fit['enzyme_uM'] / np.sqrt(np.maximum(aggregate['n'].to_numpy(float), 1)))
    paper = paper_style()
    with plt.rc_context({**paper.STYLE, 'mathtext.cal': paper.fam}):
        figure, axes = plt.subplots(1, 2, figsize=(6.4, 2.65))
        figure.subplots_adjust(left=.10, right=.77, bottom=.23, top=.70, wspace=.55)
        paper.draw_progress(axes[0], sp, spu)
        lines = paper.draw_mm(axes[1], name, sp, spu, parameters=parameters, points=points)
        figure.text(.43, .93, name, ha='center', fontsize=9,
                    color=paper.famcol(name)[1], fontweight='bold')
        figure.text(.43, .76, "\n".join(lines), ha='center', fontsize=6.5)
        _standalone_legend(figure, paper)
    return figure



def summarize_kinetics(fits, background):
    """Keep raw fits, propagated errors and manuscript reporting rules together."""
    FITS, KUNCAT = fits, background
    rows = []
    for name, f in FITS.items():
        mm = f['mm']
        rows.append({
            "design":            name,
            "plate ID":          f.get('plate_id'),
            "scaffold":          f.get('scaffold'),
            "condition":         f['condition'],
            "[E]0 (uM)":         f['enzyme_uM'],
            "kcat (s-1)":        mm['kcat_per_s'],
            "kcat sd":           mm['kcat_err_per_s'],
            "Km (uM)":           mm['Km_uM'],
            "Km sd":             mm['Km_err_uM'],
            "kcat/Km (M-1 s-1)": mm['kcat_over_Km_per_uM_per_s'] * 1e6,
            "kcat/Km sd":        mm['kcat_over_Km_err_per_uM_per_s'] * 1e6,
        })
    summary_df = pd.DataFrame(rows)

    ### CONDITION-MATCHED UNCATALYZED RATE ###
    summary_df["kuncat (s-1)"]  = summary_df["condition"].map(lambda c: KUNCAT[c]["k"])
    summary_df["kuncat sd"]     = summary_df["condition"].map(lambda c: KUNCAT[c]["err"])

    ### DERIVED RATIOS (relative errors added in quadrature) ###
    def _rel(a, sa, b, sb):
        return np.hypot(sa / a, sb / b)

    summary_df["kcat/kuncat"] = summary_df["kcat (s-1)"] / summary_df["kuncat (s-1)"]
    summary_df["kcat/kuncat sd"] = summary_df["kcat/kuncat"] * _rel(
        summary_df["kcat (s-1)"], summary_df["kcat sd"],
        summary_df["kuncat (s-1)"], summary_df["kuncat sd"])
    summary_df["(kcat/Km)/kuncat (M-1)"] = summary_df["kcat/Km (M-1 s-1)"] / summary_df["kuncat (s-1)"]
    summary_df["(kcat/Km)/kuncat sd"] = summary_df["(kcat/Km)/kuncat (M-1)"] * _rel(
        summary_df["kcat/Km (M-1 s-1)"], summary_df["kcat/Km sd"],
        summary_df["kuncat (s-1)"], summary_df["kuncat sd"])

    ### PAPER UNCERTAINTY BUDGET AND REPORTING STATUS ###
    # Keep fit-only errors above, and expose total errors separately. The original
    # paper table stores rounded inputs; freshly computed values are kept full precision.
    for _i, _row in summary_df.iterrows():
        _name = _row['design']
        _full = fresh_result_row(_name, FITS[_name], KUNCAT)
        for _col in ['kcat sd (total)', 'kcat/Km sd (total)', 'kcat/kuncat sd',
                     '(kcat/Km)/kuncat sd (M-1)', 'kcat (reported)',
                     'Km (reported, mM)', 'kcat/Km (reported)', 'reporting model']:
            summary_df.loc[_i, _col] = _full[_col]
        for _col in ['kcat/Km chord (M-1 s-1)', 'kcat/Km chord sd',
                     'kcat/Km chord (reported)']:
            if _col in _full:
                summary_df.loc[_i, _col] = _full[_col]
    summary_df['(kcat/Km)/kuncat sd'] = summary_df['(kcat/Km)/kuncat sd (M-1)']
    validation_df = validate_reference(FITS, KUNCAT)


    return summary_df, validation_df



def load_round2_screen():
    """Load raw traces and independently validate the ordered-design mapping."""
    EXPECTED_N_DESIGNS = EXPECTED_N_READER_WELLS = 192
    platereader_csv      = raw_file('260525_i3smw')
    order_fasta          = raw_file('260514__ARuder')
    design_decisions_csv = raw_file('design_decisions')

    expected_n_designs      = EXPECTED_N_DESIGNS
    expected_n_reader_wells = EXPECTED_N_READER_WELLS

    #####################
    ### HELPER LOGIC  ###
    #####################

    DESIGN_NAME_RE = re.compile(
        r'(?P<design_id>ZAPP_i3_AR_pte_(?P<design_index>\d+))'
        r'_scaffold_(?P<scaffold>\d+)'
        r'____Plate(?P<plate_number>\d)__(?P<well_position>[A-H]\d{1,2})'
        r'____(?P<parent>.+?)(?:\.pdb)?$'
    )

    def parse_design_name(name):
        """Pull design id, scaffold and plate position out of an ordered design name."""
        match = DESIGN_NAME_RE.match(str(name).strip())
        if match is None:
            raise ValueError(f'Could not parse ordered design name: {name}')
        fields = match.groupdict()
        return {
            'design_id':     fields['design_id'],
            'design_index':  int(fields['design_index']),
            'scaffold':      int(fields['scaffold']),
            'plate_number':  int(fields['plate_number']),
            'well_position': fields['well_position'].upper(),
            'parent':        fields['parent'],
        }

    def load_design_table(fasta_path):
        """Build the ordered-design table from the FASTA headers sent to the vendor."""
        headers = [line[1:].strip() for line in Path(fasta_path).read_text().splitlines()
                   if line.startswith('>')]
        if len(headers) != expected_n_designs:
            raise ValueError(f'Expected {expected_n_designs} FASTA records, found {len(headers)}.')

        design_df = pd.DataFrame([parse_design_name(h) for h in headers])
        design_df['source_well'] = ('P' + design_df['plate_number'].astype(str)
                                    + design_df['well_position'])
        # the ORI_* token identifies the diffusion parent that defines each scaffold family
        design_df['ori_family'] = design_df['parent'].str.extract(r'(ORI_\d+_C\d+_i_\d+_model_\d+)')

        if design_df['design_id'].duplicated().any():
            raise ValueError('Duplicate design IDs in the order FASTA.')
        if design_df.duplicated(['plate_number', 'well_position']).any():
            raise ValueError('Duplicate plate/well positions in the order FASTA.')
        if not design_df['plate_number'].isin([1, 2]).all():
            raise ValueError('Found a plate number outside {1, 2}.')
        if not design_df['well_position'].str.fullmatch(r'[A-H](?:[1-9]|1[0-2])').all():
            raise ValueError('Found an invalid 96-well position.')

        # each scaffold must correspond to exactly one diffusion parent
        family_counts = design_df.groupby('scaffold')['ori_family'].nunique()
        if (family_counts != 1).any():
            raise ValueError('A scaffold maps to more than one ORI parent family.')
        return design_df

    def map_reader_well_to_plate_well(reader_well):
        """384-well reader position -> (source plate number, 96-well position)."""
        match = re.fullmatch(r'([A-H])(\d{1,2})', str(reader_well))
        if match is None:
            raise ValueError(f'Invalid reader well label: {reader_well}')

        reader_row = match.group(1)
        reader_col = int(match.group(2))
        if reader_col < 1 or reader_col > 24:
            raise ValueError(f'Reader well column outside 1-24: {reader_well}')

        reader_row_idx = 'ABCDEFGH'.index(reader_row)
        plate_number = 1 if reader_row_idx < 4 else 2
        row_pairs = [('A', 'B'), ('C', 'D'), ('E', 'F'), ('G', 'H')]
        row_pair = row_pairs[reader_row_idx % 4]
        well_row = row_pair[0] if reader_col % 2 == 1 else row_pair[1]
        well_col = (reader_col + 1) // 2
        return plate_number, f'{well_row}{well_col}'

    def get_reader_well_columns(df):
        well_cols = [col for col in df.columns if re.fullmatch(r'[A-H](?:[1-9]|1\d|2[0-4])', str(col))]
        if len(well_cols) != expected_n_reader_wells:
            raise ValueError(f'Expected {expected_n_reader_wells} reader wells, found {len(well_cols)}.')
        return well_cols

    def build_reader_map(well_cols):
        """Reader-well -> plate/well table, with the mapping's own unit tests."""
        example_expectations = {
            'A1':  (1, 'A1'),
            'A2':  (1, 'B1'),
            'B5':  (1, 'C3'),
            'E4':  (2, 'B2'),
            'H24': (2, 'H12'),
        }
        for reader_well, expected in example_expectations.items():
            observed = map_reader_well_to_plate_well(reader_well)
            if observed != expected:
                raise AssertionError(f'{reader_well} mapped to {observed}, expected {expected}.')

        reader_map_df = pd.DataFrame([
            {'reader_well': w, **dict(zip(['plate_number', 'well_position'],
                                          map_reader_well_to_plate_well(w)))}
            for w in well_cols
        ])
        if reader_map_df.duplicated(['plate_number', 'well_position']).any():
            raise ValueError('Reader-well mapping produced duplicate 96-well positions.')
        return reader_map_df

    def load_reader_traces(platereader_csv):
        """Return (time_min, raw absorbance frame indexed by reader well)."""
        reader_df = pd.read_csv(platereader_csv)
        well_cols = get_reader_well_columns(reader_df)

        raw_time = pd.to_numeric(reader_df['Time'], errors='coerce')
        if raw_time.isna().any():
            raise ValueError('Plate-reader Time column contains non-numeric values.')
        # the reader writes elapsed time as a fraction of a day
        time_min = (raw_time - raw_time.iloc[0]) * 24 * 60

        abs_df = reader_df[well_cols].apply(pd.to_numeric, errors='coerce')
        if abs_df.isna().any().any():
            raise ValueError('Plate-reader absorbance block contains non-numeric values.')
        return time_min.to_numpy(dtype=float), abs_df, well_cols

    #################
    ### LOAD DATA ###
    #################

    design_df = load_design_table(order_fasta)
    time_min, abs_df, reader_well_cols = load_reader_traces(platereader_csv)
    reader_map_df = build_reader_map(reader_well_cols)

    well_df = reader_map_df.merge(design_df, on=['plate_number', 'well_position'],
                                  how='left', validate='one_to_one')
    if well_df['design_id'].isna().any():
        display(well_df[well_df['design_id'].isna()])
        raise ValueError('Some reader wells did not receive a design assignment.')

    n_timepoints = len(time_min)
    read_interval_min = float(np.median(np.diff(time_min)))


    # Cross-check the FASTA-derived mapping against the SI sequence/model workbook.
    _, _, deposited = load_screening_PTE_RFd3(2)
    check = well_df.merge(deposited[['reader_well', 'scaffold', 'order_plate', 'well']],
                          on='reader_well', validate='one_to_one', suffixes=('', '_global'))
    assert (check.scaffold + 6 == check.scaffold_global).all()
    assert (check.plate_number == check.order_plate).all()
    assert (check.well_position == check.well).all()
    return dict(time_min=time_min, abs_df=abs_df, reader_well_cols=reader_well_cols,
                well_df=well_df, design_df=design_df, design_decisions_csv=design_decisions_csv)



def validate_round2_mapping(screen):
    """Compare both stamping hypotheses against the sequencing decisions."""
    time_min, abs_df = screen['time_min'], screen['abs_df']
    reader_well_cols, well_df = screen['reader_well_cols'], screen['well_df']
    design_decisions_csv = screen['design_decisions_csv']
    validation_fit_start_min, validation_fit_end_min = 2., 15.
    def quick_ols_rate(y, t, start_min, end_min):
        """Plain OLS slope over a time window - used only for mapping validation."""
        mask = (t >= start_min) & (t <= end_min)
        return float(np.polyfit(t[mask], y[mask], 1)[0])

    def column_halves_mapping(reader_well):
        """Alternative stamping hypothesis: plate 1 = columns 1-12, plate 2 = columns 13-24."""
        row, col = re.fullmatch(r'([A-H])(\d+)', reader_well).groups()
        col = int(col)
        plate = 1 if col <= 12 else 2
        return f'P{plate}{row}{col if col <= 12 else col - 12}'

    decisions_df = pd.read_csv(design_decisions_csv)
    sequenced_wells = sorted(set(decisions_df['design']))

    provisional_rate = pd.Series(
        {w: quick_ols_rate(abs_df[w].to_numpy(float), time_min,
                           validation_fit_start_min, validation_fit_end_min)
         for w in reader_well_cols}
    )

    validation_rows = []
    for label, source_lookup in [
        ('interleaved row-pairs (used here)',
         dict(zip(well_df['reader_well'], well_df['source_well']))),
        ('column halves (alternative)',
         {w: column_halves_mapping(w) for w in reader_well_cols}),
    ]:
        ranked = (pd.DataFrame({'reader_well': list(source_lookup),
                                'source_well': list(source_lookup.values())})
                  .assign(rate=lambda d: provisional_rate[d['reader_well']].to_numpy())
                  .assign(rank=lambda d: d['rate'].rank(ascending=False).astype(int))
                  .set_index('source_well'))
        ranks = ranked.loc[sequenced_wells, 'rank'].sort_values()
        validation_rows.append({
            'mapping': label,
            'ranks_of_sequenced_designs': list(ranks.values),
            'n_in_top_10': int((ranks <= 10).sum()),
            'n_in_top_20': int((ranks <= 20).sum()),
        })

    validation_df = pd.DataFrame(validation_rows)
    print(f'Designs sent for sequencing (n={len(sequenced_wells)}): {sequenced_wells}')
    n_top = validation_df.loc[0, 'n_in_top_10']
    # probability of >= n_top of 8 randomly chosen designs landing in the top 10 of 192
    p_chance = stats.hypergeom.sf(n_top - 1, 192, 10, len(sequenced_wells))
    print(f'Interleaved mapping puts {n_top}/{len(sequenced_wells)} sequenced designs in the '
          f'top 10 (p = {p_chance:.2e} by chance).')
    if validation_df.loc[0, 'n_in_top_20'] <= validation_df.loc[1, 'n_in_top_20']:
        raise AssertionError('The interleaved mapping is not better supported than the alternative.')
    print('Mapping confirmed.')
    return validation_df



def fit_round2_screen(screen):
    """Retain the original initial-rate, artifact, background and hit rules."""
    FIT_START_MIN, FIT_END_MIN = 2., 15.
    ABS_LINEAR_CEILING, DROP_ABS_TOL, MIN_FIT_POINTS = 2., .010, 8
    HIT_Z_THRESHOLD, HIT_FOLD_THRESHOLD = 5., 3.
    time_min, abs_df = screen['time_min'], screen['abs_df']
    reader_well_cols, well_df = screen['reader_well_cols'], screen['well_df']
    fit_start_min      = FIT_START_MIN
    fit_end_min        = FIT_END_MIN
    abs_linear_ceiling = ABS_LINEAR_CEILING
    drop_abs_tol       = DROP_ABS_TOL
    min_fit_points     = MIN_FIT_POINTS
    hit_z_threshold    = HIT_Z_THRESHOLD
    hit_fold_threshold = HIT_FOLD_THRESHOLD

    ### OUTPUTS ###


    #####################
    ### HELPER LOGIC  ###
    #####################

    def robust_sd(values):
        """MAD-based standard deviation estimate."""
        values = np.asarray(values, dtype=float)
        return 1.4826 * float(np.median(np.abs(values - np.median(values))))

    def find_drop_artifacts(y, tol):
        """Indices of reads where absorbance falls by more than `tol` in a single step."""
        return np.flatnonzero(np.diff(y) < -tol) + 1

    def longest_clean_segment(n_points, break_indices, window_mask):
        """Longest run of in-window reads uninterrupted by an artifact."""
        bounds = [0, *sorted(break_indices), n_points]
        best = None
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            seg = np.zeros(n_points, dtype=bool)
            seg[lo:hi] = True
            seg &= window_mask
            if best is None or seg.sum() > best.sum():
                best = seg
        return best

    def fit_initial_rate(y, t):
        """OLS initial rate with artifact-aware segment selection.

        Returns the slope in absorbance units per minute plus the diagnostics needed to
        decide whether to trust it.
        """
        window_mask = (t >= fit_start_min) & (t <= fit_end_min)
        drops = find_drop_artifacts(y, drop_abs_tol)

        fit_mask = window_mask
        segmented = False
        if len(drops) and window_mask[drops].any():
            fit_mask = longest_clean_segment(len(y), drops, window_mask)
            segmented = True

        if fit_mask.sum() < min_fit_points:
            # fall back to the full window rather than fitting a handful of reads
            fit_mask = window_mask
            segmented = False

        x, yy = t[fit_mask], y[fit_mask]
        slope, intercept = np.polyfit(x, yy, 1)
        predicted = slope * x + intercept
        ss_res = float(np.sum((yy - predicted) ** 2))
        ss_tot = float(np.sum((yy - yy.mean()) ** 2))

        return {
            'rate_abs_per_min':  float(slope),
            'fit_intercept_abs': float(intercept),
            'fit_r2':            np.nan if ss_tot == 0 else 1 - ss_res / ss_tot,
            'fit_residual_sd':   float(np.sqrt(ss_res / max(len(x) - 2, 1))),
            'fit_n_points':      int(fit_mask.sum()),
            'fit_first_min':     float(x.min()),
            'fit_last_min':      float(x.max()),
            'fit_segmented':     segmented,
            'initial_abs':       float(y[0]),
            'final_abs':         float(y[-1]),
            'max_abs':           float(y.max()),
            'max_abs_in_window': float(y[window_mask].max()),
            'n_drop_artifacts':  int(len(drops)),
            'first_drop_min':    float(t[drops[0]]) if len(drops) else np.nan,
            'largest_drop_abs':  float(np.diff(y).min()),
        }

    #################
    ### LOAD DATA ###
    #################

    rate_records = []
    for reader_well in reader_well_cols:
        y = abs_df[reader_well].to_numpy(dtype=float)
        rate_records.append({'reader_well': reader_well, **fit_initial_rate(y, time_min)})

    rates_df = well_df.merge(pd.DataFrame(rate_records), on='reader_well',
                             how='left', validate='one_to_one')

    # >> QC flags
    rates_df['flag_optical_artifact'] = rates_df['n_drop_artifacts'] > 0
    rates_df['flag_above_linear_range'] = rates_df['max_abs'] > abs_linear_ceiling
    rates_df['flag_saturated_in_window'] = rates_df['max_abs_in_window'] > abs_linear_ceiling
    rates_df['qc_pass'] = ~(rates_df['flag_optical_artifact'] | rates_df['flag_saturated_in_window'])

    # >> background: iteratively trim the active tail to isolate the inactive population
    inactive = rates_df.loc[rates_df['qc_pass'], 'rate_abs_per_min'].to_numpy(float)
    for _ in range(50):
        center, spread = np.median(inactive), robust_sd(inactive)
        keep = np.abs(inactive - center) <= 3 * spread
        if keep.all():
            break
        inactive = inactive[keep]

    background_rate = float(np.median(inactive))
    background_sd   = robust_sd(inactive)
    n_inactive      = int(len(inactive))

    rates_df['rate_corrected_abs_per_min'] = rates_df['rate_abs_per_min'] - background_rate
    rates_df['robust_z'] = (rates_df['rate_abs_per_min'] - background_rate) / background_sd
    rates_df['fold_over_background'] = rates_df['rate_abs_per_min'] / background_rate

    rates_df['is_hit'] = (
        (rates_df['robust_z'] >= hit_z_threshold) &
        (rates_df['fold_over_background'] >= hit_fold_threshold) &
        rates_df['qc_pass']
    )
    rates_df['activity_rank'] = rates_df['rate_abs_per_min'].rank(ascending=False, method='min').astype(int)
    rates_df = rates_df.sort_values('activity_rank').reset_index(drop=True)

    limit_of_detection = 3 * background_sd


    return rates_df, dict(background_rate=background_rate, background_sd=background_sd,
                          n_inactive=n_inactive, limit_of_detection=limit_of_detection)



@_quiet_reader
def tagless_linearity_checks(background):
    """Refit tagged/tagless assays using the original detector-linearity check."""
    from rfd3_figures import side_ZAPP1_tagless as tagless
    paper = paper_style()
    rows = []
    for label, spec in [("tagless", paper.spec(7)), ("tagged", paper.spec(9))]:
        complete = tagless.fit(spec)
        piv = paper.traces(spec)
        time = piv.index.to_numpy(float)
        signals = np.column_stack([piv[f"{r}{c}"].to_numpy(float)
                                   for r in "ABCDEF" for c in spec["cols"]])
        bad = (signals > 2.5) | ~np.isfinite(signals)
        indices = np.flatnonzero(bad.any(axis=1))
        end = float(time[indices[0]]) if len(indices) else float(time[-1])
        linear = tagless.fit(dict(spec, tr=(spec["tr"][0], end)))
        for window, values, hi in [("original", complete, spec["tr"][1]),
                                    ("detector-linearity check", linear, end)]:
            values = dict(values)
            ku = background['+25 mM NaHCO3']
            values['rat'] = values['kcat'] / ku['k']
            # The shared calibration term cancels in the ratio uncertainty.
            fractional_fit_variance = (values['kcat_sd']/values['kcat'])**2 - tagless.CAL**2
            values['rat_sd'] = values['rat'] * np.sqrt(
                fractional_fit_variance + (ku['err']/ku['k'])**2)
            rows.append(dict(construct=label, fit_window=window, start_s=spec["tr"][0],
                             end_s=hi, **values))
    return pd.DataFrame(rows)


def export_results(fits, background, summary, validation, screen, rates, *, manuscript_figures=False):
    """Write fresh tables/plots; optionally rebuild the historical manuscript pages."""
    PLOTS.mkdir(exist_ok=True)
    summary.to_csv(PLOTS / "kinetics_summary.csv", index=False, float_format="%.6g")
    validation.to_csv(PLOTS / "paper_reproduction_validation.csv", index=False)
    rates.to_csv(PLOTS / "round2_screen_initial_rates.csv", index=False)
    hit_columns = ['activity_rank', 'design_id', 'scaffold', 'plate_number', 'well_position',
                   'source_well', 'reader_well', 'rate_abs_per_min', 'rate_corrected_abs_per_min',
                   'robust_z', 'fold_over_background', 'fit_r2', 'fit_residual_sd', 'fit_n_points',
                   'initial_abs', 'final_abs', 'max_abs', 'qc_pass', 'flag_optical_artifact',
                   'flag_above_linear_range', 'fit_segmented', 'is_hit', 'parent']
    rates.loc[rates.is_hit, hit_columns].sort_values('activity_rank').to_csv(
        PLOTS / "round2_screen_hits.csv", index=False)
    for round_number in (1, 2):
        plot_screening_PTE_RFd3(round_number, show=False,
            plot_path=PLOTS / f"screen_round{round_number}_SI_style.png")
    for spec in KINETICS_SPECS:
        fig = kinetics_figure(spec['name'], fits, background)
        fig.savefig(PLOTS / (spec['plot_name'] + '.png'), dpi=150, bbox_inches='tight')
        import matplotlib.pyplot as plt
        plt.close(fig)
    tagless_linearity_checks(background).to_csv(PLOTS / 'tagless_linearity_checks.csv', index=False)
    if manuscript_figures:
        namespace = dict(screen, FITS=fits, KUNCAT=background, rates_df=rates,
                         BACKGROUND_TRACE_COLOR='#b8b8b8')
        reproduce_paper_figures(namespace)
    return PLOTS
