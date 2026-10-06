"""Metalloprotease screening, peptide-reporter specificity and mass spectra."""
from pathlib import Path
from contextlib import redirect_stdout
from functools import wraps
from io import StringIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / 'raw_data_files'

def time_to_seconds(t):
    """Convert HH:MM:SS to elapsed seconds."""
    h, m, s = [int(x) for x in t.split(':')]
    return h * 3600 + m * 60 + s

def parse_channel(raw_df, channel_label):
    """
    Parse one fluorescence channel block (e.g. '485,528' or '485,610')
    """
    records = []
    channel_rows = raw_df.index[raw_df[0] == channel_label]
    for start_idx in channel_rows:
        header_idx = start_idx + 2
        header = raw_df.iloc[header_idx].tolist()
        well_cols = header[3:]
        well_col_indices = list(range(3, 3 + len(well_cols)))
        data_start = header_idx + 1
        for i in range(data_start, len(raw_df)):
            row = raw_df.iloc[i]
            if pd.isna(row[1]) or row[0] in ['485,528', '485,610']:
                break
            time_str = row[1]
            time_sec = time_to_seconds(time_str)
            for well, col_idx in zip(well_cols, well_col_indices):
                value = row[col_idx]
                if pd.isna(value):
                    continue
                records.append({'well_name': well, 'time': time_sec, channel_label: float(value)})
    return pd.DataFrame(records)

def load_channels(path):
    """Load both fluorescence channels from the plate-reader CSV."""
    raw = pd.read_csv(path, header=None, dtype=str)
    df_gfp = parse_channel(raw, '485,528').rename(columns={'485,528': 'sfGFP_RFU'})
    df_mscarlet = parse_channel(raw, '485,610').rename(columns={'485,610': 'mscarlet_RFU'})
    df = pd.merge(df_gfp, df_mscarlet, on=['well_name', 'time'], how='outer').sort_values(['well_name', 'time']).reset_index(drop=True)
    return (df_gfp, df_mscarlet, df)

def captured_analysis(function):
    """Return recalculated figures and tables for the notebook."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        stdout = StringIO()
        with plt.rc_context(), redirect_stdout(stdout):
            result = function(*args, **kwargs)
        result['stdout'] = stdout.getvalue()
        if function.__name__ == 'run_cis_screen':
            result['tables'] = {name: table.rename(columns={'design': 'lane'})
                                for name, table in result['tables'].items()}
        filenames = TABLE_EXPORTS.get(function.__name__, {})
        for name, filename in filenames.items():
            result['tables'][name].to_csv(ROOT / 'wetlab_data_plots' / filename, index=False)
        return result
    return wrapped

@captured_analysis
def run_cis_screen():
    """Calculate cleaved-fraction distributions by incubation time."""
    _outputs = ROOT / 'wetlab_data_plots'
    _outputs.mkdir(parents=True, exist_ok=True)
    _figures = []

    def _capture_current():
        _figure = plt.gcf()
        _figures.append(_figure)
        plt.close(_figure)
    import re
    import pandas as pd

    def parse_gel_csv(csv_path):
        records = []
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
        gel_num = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            gel_header = re.match('gel\\s*(\\d+),', line, flags=re.IGNORECASE)
            if gel_header:
                gel_num = int(gel_header.group(1))
                continue
            row = [c.strip() for c in line.split(',')]
            try:
                design_num = int(row[0])
            except:
                continue
            if 'w Zn' not in row:
                continue
            if len(row) <= 8:
                continue
            value = row[8].replace('%', '').strip()
            if value in ('', '#DIV/0!'):
                continue
            try:
                digested_fraction = float(value) / 100.0
            except:
                continue
            records.append({'gel': gel_num, 'design': design_num, 'digested_fraction': digested_fraction})
        return pd.DataFrame(records)
    import pandas as pd

    def parse_gel_csv_1h(csv_path):
        df = pd.read_csv(csv_path)
        condition_col = df.columns[df.columns.get_loc('digested fraction') + 1]
        df = df[df[condition_col].astype(str).str.strip().eq('w Zn')].copy()
        df['digested_fraction'] = df['digested fraction'].astype(str).str.replace('%', '', regex=False).replace('', pd.NA).astype(float) / 100.0
        out = df.rename(columns={'gel': 'design'})[['design', 'plate_location', 'id', 'digested_fraction']].copy()
        return out
    gel_csv_1h = ROOT / 'cis_screen' / '1h_quantification.csv'
    gel_csv_18h = ROOT / 'cis_screen' / '18h_quantification.csv'
    df_1h = parse_gel_csv_1h(gel_csv_1h)
    df_18h = parse_gel_csv(gel_csv_18h)
    df_1h['incubation_time'] = 1
    df_18h['incubation_time'] = 18
    df_all_cis_screen = pd.concat([df_1h, df_18h], ignore_index=True)
    len(df_18h[df_18h['digested_fraction'] > 0.1])
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.histplot(df_all_cis_screen, x='digested_fraction', hue='incubation_time', ax=ax, binwidth=0.05, multiple='dodge', palette='Greys')
    ax.set_xlabel('cleaved fraction')
    ax.set_ylabel('count')
    legend = ax.get_legend()
    if legend is not None:
        legend.set_title('incubation time (h)')
    plt.tight_layout()
    plt.savefig(_outputs / 'cis_screen_histogram.png', dpi=300, bbox_inches='tight')
    for _number in list(plt.get_fignums()):
        _figure = plt.figure(_number)
        if _figure not in _figures:
            _figures.append(_figure)
        plt.close(_figure)
    _namespace = locals()
    _tables = {name: _namespace[name].copy(deep=True) for name in ['df_1h', 'df_18h', 'df_all_cis_screen'] if name in _namespace and isinstance(_namespace[name], pd.DataFrame)}
    return dict(figures=_figures, tables=_tables, output_directory=_outputs)

@captured_analysis
def run_trans_screen():
    """Calculate fluorescence timecourses and initial slopes by well."""
    _outputs = ROOT / 'wetlab_data_plots'
    _outputs.mkdir(parents=True, exist_ok=True)
    _figures = []

    def _capture_current():
        _figure = plt.gcf()
        _figures.append(_figure)
        plt.close(_figure)
    df_gfp, df_mscarlet, df = load_channels(RAW_DIR / '260310_TTR_alphasyn_trans_screen.csv')
    import pandas as pd
    df_gfp['col_number'] = df_gfp['well_name'].str.extract('(\\d+)').astype(int)
    df_gfp_odd = df_gfp[df_gfp['col_number'] % 2 == 1].copy()
    unique_wells = sorted(df_gfp_odd['well_name'].unique(), key=lambda x: int(x[1:]))
    well_to_number = {well: i + 1 for i, well in enumerate(unique_wells)}
    df_gfp_odd['name_by_number'] = df_gfp_odd['well_name'].map(well_to_number)
    df_gfp_odd = df_gfp_odd.drop(columns='col_number')
    print(df_gfp_odd.head())
    import numpy as np
    import pandas as pd

    def calculate_initial_rate(df, start_time, end_time):
        """
        Calculate initial reaction rate (slope of sfGFP_RFU vs time)
        for each well using data between start_time and end_time (seconds).

        Returns a dataframe with slope and intercept.
        """
        results = []
        for well, group in df.groupby('well_name'):
            group = group.sort_values('time')
            fit = group[(group['time'] >= start_time) & (group['time'] <= end_time)]
            if len(fit) < 2:
                continue
            x = fit['time'].values
            y = fit['sfGFP_RFU'].values
            slope, intercept = np.polyfit(x, y, 1)
            results.append({'well_name': well, 'initial_rate': slope, 'intercept': intercept, 'n_points': len(fit)})
        return pd.DataFrame(results)
    rates = calculate_initial_rate(df_gfp_odd, start_time=900, end_time=1500)
    rates.sort_values(by='initial_rate', ascending=False)
    import matplotlib.pyplot as plt
    df_plot = df_gfp_odd[~df_gfp_odd['well_name'].isin([])].copy()
    df_plot = df_plot[df_plot['time'] <= 3600]
    df_plot['rfu_corrected'] = df_plot['sfGFP_RFU'] - df_plot.groupby('well_name')['sfGFP_RFU'].transform('first')
    plt.figure(figsize=(5, 4))
    for well, df_well in df_plot.groupby('well_name'):
        if well in ['G7', 'G19', 'G21']:
            color = 'red'
            linewidth = 2.5
            zorder = 3
        else:
            color = 'gray'
            linewidth = 1.5
            zorder = 1
        plt.plot(df_well['time'], df_well['rfu_corrected'], color=color, linewidth=linewidth, zorder=zorder)
    plt.xlabel('Time (s)')
    plt.ylabel('sfGFP RFU')
    plt.title('sfGFP Progress Curves ')
    plt.tight_layout()
    plt.savefig(_outputs / 'trans_screen.png', dpi=300)
    _capture_current()
    for _number in list(plt.get_fignums()):
        _figure = plt.figure(_number)
        if _figure not in _figures:
            _figures.append(_figure)
        plt.close(_figure)
    _namespace = locals()
    _tables = {name: _namespace[name].copy(deep=True) for name in ['rates', 'df_plot'] if name in _namespace and isinstance(_namespace[name], pd.DataFrame)}
    return dict(figures=_figures, tables=_tables, output_directory=_outputs)

@captured_analysis
def run_specificity():
    """Plot G11 fluorescence timecourses for the short peptide reporters."""
    _outputs = ROOT / 'wetlab_data_plots'
    _outputs.mkdir(parents=True, exist_ok=True)
    _figures = []

    def _capture_current():
        _figure = plt.gcf()
        _figures.append(_figure)
        plt.close(_figure)
    df_gfp, df_mscarlet, df = load_channels(RAW_DIR / '260312_TTR_G11_specificity.csv')
    print(df_gfp)
    df = df_gfp.copy()
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    substrates = ['Tau1', 'Tau2', 'SAA', 'αSyn1', 'αSyn2', 'TDP43', 'TTR']
    seqs = ['GGSVQIVYKP', 'PGGGKVQIIN', 'SSRSFFSFLG', 'GGAVVTGVTA', 'VVHGVATVAE', 'ALQSSWGMMG', 'PAINVAVHV']
    substrate_map = dict(zip(substrates, seqs))
    cols = [1, 3, 5, 7, 9, 11, 13]
    prot_rows = ['F', 'G', 'H']
    noprot_rows = ['I', 'J', 'K']
    df = df_gfp.copy()
    df['row'] = df['well_name'].str[0]
    df['col'] = df['well_name'].str[1:].astype(int)
    df = df[df['col'].isin(cols)]
    df = df[df['row'].isin(prot_rows + noprot_rows)]
    col_to_sub = dict(zip(cols, substrates))
    df['substrate'] = df['col'].map(col_to_sub)
    df['protease'] = df['row'].isin(prot_rows).astype(int)
    rep_map = {'F': 1, 'G': 2, 'H': 3, 'I': 1, 'J': 2, 'K': 3}
    df['replicate'] = df['row'].map(rep_map)
    t0 = df[df['time'] == 0].set_index('well_name')['sfGFP_RFU']
    df['RFU_t0sub'] = df['sfGFP_RFU'] - df['well_name'].map(t0)
    pivot = df.pivot_table(index=['time', 'substrate', 'replicate'], columns='protease', values='RFU_t0sub').reset_index()
    pivot.columns.name = None
    pivot = pivot.rename(columns={0: 'noProt', 1: 'Prot'})
    pivot['RFU_final'] = pivot['Prot'] - pivot['noProt']
    paper_navaho = (255 / 255, 224 / 255, 172 / 255)
    paper_melon = (255 / 255, 198 / 255, 178 / 255)
    paper_pink = (255 / 255, 172 / 255, 183 / 255)
    paper_purple = (213 / 255, 154 / 255, 181 / 255)
    paper_lightblue = (149 / 255, 150 / 255, 198 / 255)
    paper_blue = (102 / 255, 134 / 255, 197 / 255)
    paper_teal = (0.31, 0.725, 0.686)
    color_map = {'Tau1': paper_navaho, 'Tau2': paper_melon, 'SAA': paper_pink, 'αSyn1': paper_purple, 'αSyn2': paper_lightblue, 'TDP43': paper_blue, 'TTR': paper_teal}
    fig, ax = plt.subplots(figsize=(5, 4))
    for sub in substrates:
        d = pivot[pivot['substrate'] == sub]
        stats = d.groupby('time')['RFU_final'].agg(['mean', 'std']).reset_index()
        color = color_map[sub]
        ax.plot(stats['time'], stats['mean'], linewidth=2.6, color=color, label=sub)
        ax.fill_between(stats['time'], stats['mean'] - stats['std'], stats['mean'] + stats['std'], color=color, alpha=0.2, linewidth=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Relative Fluorescence Unit')
    ax.legend(frameon=False, fontsize=9, ncol=2)
    plt.tight_layout()
    plt.savefig(_outputs / 'specificity_timecourses.png', dpi=300, bbox_inches='tight')
    _capture_current()
    import metalloprotease_plot_style as pu
    pu.apply_style()
    from metalloprotease_plot_style import styling, init_fig, apply_style
    apply_style()
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    substrates = ['Tau1', 'Tau2', 'SAA', 'αSyn1', 'αSyn2', 'TDP43', 'TTR']
    seqs = ['GGSVQIVYKP', 'PGGGKVQIIN', 'SSRSFFSFLG', 'GGAVVTGVTA', 'VVHGVATVAE', 'ALQSSWGMMG', 'PAINVAVHV']
    substrate_map = dict(zip(substrates, seqs))
    cols = [1, 3, 5, 7, 9, 11, 13]
    prot_rows = ['F', 'G', 'H']
    noprot_rows = ['I', 'J', 'K']
    df = df_gfp.copy()
    df['row'] = df['well_name'].str[0]
    df['col'] = df['well_name'].str[1:].astype(int)
    df = df[df['col'].isin(cols)]
    df = df[df['row'].isin(prot_rows + noprot_rows)]
    col_to_sub = dict(zip(cols, substrates))
    df['substrate'] = df['col'].map(col_to_sub)
    df['protease'] = df['row'].isin(prot_rows).astype(int)
    rep_map = {'F': 1, 'G': 2, 'H': 3, 'I': 1, 'J': 2, 'K': 3}
    df['replicate'] = df['row'].map(rep_map)
    t0 = df[df['time'] == 0].set_index('well_name')['sfGFP_RFU']
    df['RFU_t0sub'] = df['sfGFP_RFU'] - df['well_name'].map(t0)
    pivot = df.pivot_table(index=['time', 'substrate', 'replicate'], columns='protease', values='RFU_t0sub').reset_index()
    pivot.columns.name = None
    pivot = pivot.rename(columns={0: 'noProt', 1: 'Prot'})
    pivot['RFU_final'] = pivot['Prot'] - pivot['noProt']
    paper_teal = (0.31, 0.725, 0.686)
    paper_navaho = (255 / 255, 224 / 255, 172 / 255)
    paper_melon = (255 / 255, 198 / 255, 178 / 255)
    paper_pink = (255 / 255, 172 / 255, 183 / 255)
    paper_purple = (213 / 255, 154 / 255, 181 / 255)
    paper_lightblue = (149 / 255, 150 / 255, 198 / 255)
    paper_blue = (102 / 255, 134 / 255, 197 / 255)
    color_map = {'Tau1': paper_navaho, 'Tau2': paper_melon, 'SAA': paper_pink, 'αSyn1': paper_purple, 'αSyn2': paper_lightblue, 'TDP43': paper_blue, 'TTR': paper_teal}
    cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(9 * cm, 9 * 4 / 5 * cm), dpi=300)
    styling.despline(ax)
    styling.set_spine_width(ax)
    for sub in substrates:
        d = pivot[pivot['substrate'] == sub]
        stats = d.groupby('time')['RFU_final'].agg(['mean', 'std']).reset_index()
        color = color_map[sub]
        ax.plot(stats['time'], stats['mean'], linewidth=2.6, color=color, label=sub)
        ax.fill_between(stats['time'], stats['mean'] - stats['std'], stats['mean'] + stats['std'], color=color, alpha=0.2, linewidth=0)
    styling.despline(ax)
    styling.set_spine_width(ax, width=1.0)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Relative Fluorescence Unit')
    ax.legend(frameon=False, fontsize=9, ncol=2, loc='upper right', bbox_to_anchor=(1, 0.52), columnspacing=0.8)
    plt.tight_layout()
    plt.savefig(_outputs / 'specificity_timecourses.png', dpi=300, bbox_inches='tight')
    _capture_current()
    for _number in list(plt.get_fignums()):
        _figure = plt.figure(_number)
        if _figure not in _figures:
            _figures.append(_figure)
        plt.close(_figure)
    _namespace = locals()
    _tables = {name: _namespace[name].copy(deep=True) for name in ['df'] if name in _namespace and isinstance(_namespace[name], pd.DataFrame)}
    return dict(figures=_figures, tables=_tables, output_directory=_outputs)

@captured_analysis
def run_mass_spectrometry():
    """Plot two regions of the deconvoluted mass spectrum."""
    _outputs = ROOT / 'wetlab_data_plots'
    _outputs.mkdir(parents=True, exist_ok=True)
    _figures = []

    def _capture_current():
        _figure = plt.gcf()
        _figures.append(_figure)
        plt.close(_figure)
    import pandas as pd
    import matplotlib.pyplot as plt
    ms_data_path = RAW_DIR / 'AC_G11+Zn.csv'
    ms_data = pd.read_csv(ms_data_path)
    ms_data
    ms_data.columns = ['X(Daltons)', 'Y(Counts)']
    ms_data = ms_data.iloc[1:]
    ms_data
    from metalloprotease_plot_style import styling, apply_style
    apply_style()
    ms_plot = ms_data.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
    ms_plot
    import matplotlib.ticker as mticker
    cm = 1 / 2.54
    fig, axes = plt.subplots(1, 2, figsize=(9 * cm, 9 * 4 / 5 * cm), dpi=300, sharey=False)
    regions = [(20000, 30000), (55000, 60000)]
    paper_teal = (0.31, 0.725, 0.686)
    for ax, (lo, hi) in zip(axes, regions):
        d = ms_plot[(ms_plot['X(Daltons)'] >= lo) & (ms_plot['X(Daltons)'] <= hi)].iloc[::5]
        ax.plot(d['X(Daltons)'] / 1000, d['Y(Counts)'], color=paper_teal, linewidth=1.8)
        styling.despline(ax)
        styling.set_spine_width(ax, width=1.0)
        ax.set_xlim(lo / 1000, hi / 1000)
        ax.set_xlabel('Mass (kDa)')
        ax.set_ylim(bottom=0)
        formatter = mticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(formatter)
    axes[0].set_ylabel('Counts')
    axes[1].set_ylabel('Counts')
    plt.tight_layout()
    plt.savefig(_outputs / 'ms_two_regions.png', dpi=300, bbox_inches='tight')
    _capture_current()
    for _number in list(plt.get_fignums()):
        _figure = plt.figure(_number)
        if _figure not in _figures:
            _figures.append(_figure)
        plt.close(_figure)
    _namespace = locals()
    _tables = {name: _namespace[name].copy(deep=True) for name in ['ms_plot'] if name in _namespace and isinstance(_namespace[name], pd.DataFrame)}
    return dict(figures=_figures, tables=_tables, output_directory=_outputs)


TABLE_EXPORTS = {
    'run_cis_screen': {
        'df_1h': 'cis_screen_1h.csv',
        'df_18h': 'cis_screen_18h.csv',
        'df_all_cis_screen': 'cis_screen_cleaved_fractions.csv',
    },
    'run_trans_screen': {
        'rates': 'trans_screen_initial_rates.csv',
        'df_plot': 'trans_screen_progress_curves.csv',
    },
    'run_specificity': {'df': 'G11_specificity_timecourses.csv'},
}
