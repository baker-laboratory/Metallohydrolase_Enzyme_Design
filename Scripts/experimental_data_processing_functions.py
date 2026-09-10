import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
import os

# Color constants
good_purple = (141/255, 101/255, 184/255)
good_blue   = (68/255, 153/255, 231/255)
good_teal   = (40/255, 176/255, 193/255)
good_yellow = (250/255, 199/255, 44/255)
good_green  = (170/255, 195/255, 47/255)
good_pink   = (236/255, 114/255, 164/255)
good_peach  = (249/255, 145/255, 120/255)
good_red    = (228/255, 74/255, 62/255)
good_gray   = (220/255, 220/255, 220/255)


def _time_column_to_seconds(col):
    """Normalize a plate reader 'Time' column to seconds.

    The Neo2 exports this column in one of two ways depending on how the sheet
    was saved: as clock values (``datetime.time``/``"H:MM:SS"`` strings), or as
    Excel serial day fractions (float, e.g. 4.63e-05 == 4 s). Both are handled.
    """
    if pd.api.types.is_numeric_dtype(col):
        return col.astype(float) * 86400.0          # days -> seconds

    converted = pd.to_datetime(col, format='%H:%M:%S', errors='coerce')
    if converted.isna().all():
        converted = pd.to_datetime(col, errors='coerce')   # let pandas infer
    if converted.isna().all():
        raise ValueError(
            "Could not interpret the 'Time' column: it is neither numeric "
            "(Excel serial days) nor a recognizable clock format."
        )
    t = converted.dt.time
    return t.apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)


###############################################################################
# Figure output
###############################################################################

# Defaults for every figure this module and the analysis notebooks write.
# PNG at a modest DPI is the default so the repository stays small; EPS is
# vector and much larger, so it is opt-in. Override globally by reassigning
# these, or per figure by passing dpi= / save_eps= to save_figure().
FIGURE_DPI: int = 150
SAVE_EPS: bool = False


def save_figure(path_stem, dpi=None, save_eps=None, bbox_inches="tight", **kwargs):
    """Save the current matplotlib figure as PNG, and optionally also as EPS.

    Parameters
    ----------
    path_stem
        Output path. Any ``.png`` / ``.eps`` / ``.svg`` extension is stripped,
        so callers may pass either a stem or a legacy ``*.eps`` path.
    dpi
        Raster resolution for the PNG. Defaults to the module-level
        ``FIGURE_DPI`` (150). Raise it for a figure that needs more detail.
    save_eps
        Also write ``<stem>.eps``. Defaults to the module-level ``SAVE_EPS``
        (False). EPS carries no alpha channel, so matplotlib flattens any
        transparency and warns; that is expected and harmless.

    Returns
    -------
    list of str
        The paths actually written.
    """
    dpi = FIGURE_DPI if dpi is None else dpi
    save_eps = SAVE_EPS if save_eps is None else save_eps

    stem = str(path_stem)
    for ext in (".png", ".eps", ".svg", ".pdf"):
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
            break

    parent = os.path.dirname(stem)
    if parent:
        os.makedirs(parent, exist_ok=True)

    written = []
    png = f"{stem}.png"
    plt.savefig(png, dpi=dpi, bbox_inches=bbox_inches, format="png", **kwargs)
    written.append(png)

    if save_eps:
        eps = f"{stem}.eps"
        # EPS is vector: dpi only affects any rasterised sub-artists.
        plt.savefig(eps, dpi=dpi, bbox_inches=bbox_inches, format="eps", **kwargs)
        written.append(eps)

    return written


###############################################################################
# Michaelis-Menten enzyme kinetics functions
###############################################################################

def parse_kinetics(excel_file, active_wells, start_row=30, skipfooter=35):
    num_samples = len(active_wells) if active_wells is not None else 0

    bn = os.path.basename(excel_file).replace('.xlsx','')

    df = pd.read_excel(excel_file, header=start_row, skipfooter=skipfooter, engine='openpyxl')
    df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
    df.reset_index(drop=True, inplace=True)

    if 'Time' not in df.columns:
        print(df.head(3))
        raise ValueError("Please vary start_row. Time is not observed in the column.")

    df['Time'] = _time_column_to_seconds(df['Time'])
    df_final = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
    df_final.head()
    data = df_final
    return data


def standard_curve(data, standard_row, standard_col, Max_FU, pro_concs, cutoff_start_time, cutoff_end_time, show_plot=True):
    if cutoff_start_time and cutoff_end_time:
        data = data[(data['Time'] >= cutoff_start_time) & (data['Time'] <= cutoff_end_time)]
    elif cutoff_start_time:
        data = data[data['Time'] >= cutoff_start_time]
    elif cutoff_end_time:
        data = data[data['Time'] <= cutoff_end_time]

    if show_plot == True:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes = axes.flatten()

    good_colors = [good_purple, good_blue, good_teal, good_green, good_yellow, good_peach, good_red, good_pink]

    product_concentration, avr_FUs = [], []
    for index, i in enumerate(range(standard_col[0], standard_col[1]+1)):
        avr_FU = np.mean(data[f"{standard_row}{i}"])
        current_concentration = pro_concs[index]

        product_concentration.append(current_concentration*1e-6)
        avr_FUs.append(avr_FU)

        if show_plot == True:
            if len(good_colors) < index:
                axes[0].plot(data['Time'], data[f"{standard_row}{i}"], label=f"{current_concentration}uM")
            else:
                axes[0].plot(data['Time'], data[f"{standard_row}{i}"], label=f"{current_concentration}uM", color=good_colors[index-1])
            axes[0].plot([data['Time'].iloc[0], data['Time'].iloc[-1]], [avr_FU, avr_FU], "--", color="black")

            axes[0].set_xlabel('Time (seconds)')
            axes[0].set_ylabel('Fluorescence Unit')

    avr_FUs_filtered, product_concentration_filtered = [], []
    for i in range(len(avr_FUs)):
        cur_avr_FU, cur_product_concentration = avr_FUs[i], product_concentration[i]
        if cur_avr_FU < Max_FU:
            avr_FUs_filtered.append(cur_avr_FU)
            product_concentration_filtered.append(cur_product_concentration)

    slope, intercept, r_value, p_value, std_err = linregress(avr_FUs_filtered, product_concentration_filtered)
    print(f"Slope: {slope}, Intercept: {intercept}")

    if show_plot == True:
        axes[1].scatter(avr_FUs, np.array(product_concentration_filtered) * 1e6)

    fit_line_x = np.linspace(min(avr_FUs_filtered), max(avr_FUs_filtered), 100)
    fit_line_y = (slope * fit_line_x + intercept) * 1e6

    if show_plot == True:
        axes[1].plot(fit_line_x, fit_line_y, '--', color='black', label='Linear Fit')
        axes[1].set_xlabel('Fluorescence Unit')
        axes[1].set_ylabel('[4mu] uM')
        plt.tight_layout()

    if show_plot == True:
        plt.show()

    return avr_FUs_filtered, product_concentration_filtered, slope


def michaelis_menten(s, Vmax, Km):
    return (Vmax * s) / (Km + s)


def make_transpose_dic(rows, cols):
    transpose_dic = {}
    for i, row in enumerate(rows):
        transpose_dic[row] = i+1
    for i, col in enumerate(cols):
        transpose_dic[col] = "ABCDEFGHIJKL"[i]
    return transpose_dic


def make_rename_dic(transpose_dic, rows, cols):
    rename_dic = {}
    for key in transpose_dic:
        value = transpose_dic[key]
        if key in rows:
            for col in cols:
                rename_dic[f"{key}{col}"] = f"{transpose_dic[col]}{value}"
        elif key in cols:
            for row in rows:
                rename_dic[f"{row}{key}"] = f"{value}{transpose_dic[row]}"
    return rename_dic


def make_active_wells(row_column_ranges):
    active_wells = []
    for row, (start_col, end_col) in row_column_ranges.items():
        for col in range(start_col, end_col + 1):
            active_wells.append(f"{row}{col}")
    return active_wells


def mm_kinetics(data, bn, rows=None, cols=None, active_wells=None, bg_well_id=None, enz_conc=None, sub_concs=None, substrate_name=None, pics_dir=None, replicate_rows=None, fit_type=None, slope=None,
                cutoff_start_time=None, cutoff_end_time=None,
                plate_type='full', y_ax_mm_intv=0.05, x_ax_mm_intv=50, norm_zero=True, legend_label='', silent=False, ylabel=None, dpi=None, save_eps=None):
    '''
    Fit and plot kinetic data with the Michaelis-Menten kinetic model

    Inputs:
    data: Kinetic data in a dataframe
    bn: basename of file
    rows: Row labels for plate layout (optional)
    cols: Column labels for plate layout (optional)
    active_wells: List of active well IDs
    bg_well_id: row ID of background rxn wells
    enz_conc: Enzyme concentration
    sub_concs: List of substrate concentrations
    substrate_name: Name of the substrate
    pics_dir: Directory to save pictures
    replicate_rows: List of replicate rows
    fit_type: Type of fit ('straight_line' or 'slow_binding_model')
    slope: Standard curve slope for FU-to-concentration conversion
    cutoff_start_time: Time cutoff start for data
    cutoff_end_time: Time cutoff end for data
    plate_type: Type of plate ('half' or 'full')

    Output:
    Plots, kcat, Km, and kcat/Km
    '''

    sub_conc_list = []
    good_colors = [good_purple, good_blue, good_teal, good_green, good_yellow, good_peach, good_red, good_pink]
    data = data.dropna()  # reassigned rather than inplace, for pandas copy-on-write

    # Truncate data for straight-line fit
    if cutoff_start_time and cutoff_end_time:
        data = data[(data['Time'] >= cutoff_start_time) & (data['Time'] <= cutoff_end_time)]
    elif cutoff_start_time:
        data = data[data['Time'] >= cutoff_start_time]
    elif cutoff_end_time:
        data = data[data['Time'] <= cutoff_end_time]

    time_points = data['Time'].values

    # initialize arrays to store fluorescence values for each replicate
    fluorescence_values = np.zeros((len(active_wells), len(time_points), len(replicate_rows) + 1))

    # process data from each well
    for i, well in enumerate(active_wells):
        bg_well = well.replace(active_wells[0][0], bg_well_id)
        bg_fluorescence = data[bg_well].values

        replicates = [well.replace(active_wells[0][0], rep_row) for rep_row in replicate_rows]
        replicates.append(well)

        for j, rep in enumerate(replicates):
            raw_fluorescence = data[rep].values
            corrected_fluorescence = raw_fluorescence - bg_fluorescence

            if plate_type == 'half':
                raise ValueError

            elif plate_type == 'full':
                product_concentration = (corrected_fluorescence) * slope * 1000000

            if norm_zero == True:
                normalized_product_concentration = product_concentration - np.min(product_concentration)
                normalized_product_concentration = normalized_product_concentration.astype(float)
                fluorescence_values[i, :, j] = normalized_product_concentration
                sub_conc_list.append(sub_concs[i])

            else:
                normalized_product_concentration = product_concentration
                normalized_product_concentration = normalized_product_concentration.astype(float)
                fluorescence_values[i, :, j] = normalized_product_concentration
                sub_conc_list.append(sub_concs[i])

    # Fit the data using the selected fit type
    velocities = []
    for i, well in enumerate(active_wells):
        bg_well = well.replace(active_wells[0][0], bg_well_id)
        bg_fluorescence = data[bg_well].values

        replicates = [well.replace(active_wells[0][0], rep_row) for rep_row in replicate_rows]
        replicates.append(well)

        for rep in replicates:
            raw_fluorescence = data[rep].values
            corrected_fluorescence = raw_fluorescence - bg_fluorescence

            if plate_type == 'half':
                raise ValueError
            elif plate_type == 'full':
                product_concentration = (corrected_fluorescence) * slope * 1000000

            if norm_zero == True:
                normalized_product_concentration = product_concentration - np.min(product_concentration)
                normalized_product_concentration = normalized_product_concentration.astype(float)

            else:
                normalized_product_concentration = product_concentration
                normalized_product_concentration = normalized_product_concentration.astype(float)

            if fit_type == 'straight_line':
                m, _, _, _, _ = linregress(x=time_points, y=normalized_product_concentration)
                velocities.append(m / enz_conc)

            avg_fluorescence = np.mean(fluorescence_values, axis=2)
            std_fluorescence = np.std(fluorescence_values, axis=2)

    # Plot the average fluorescence with shaded regions representing standard deviation
    fig, ax = plt.subplots(figsize=(5, 4))
    for i, sub_conc in enumerate(sub_concs):
        plt.plot(time_points, avg_fluorescence[i], label=f'{sub_conc}', color=good_colors[i])
        plt.fill_between(time_points, avg_fluorescence[i] - std_fluorescence[i], avg_fluorescence[i] + std_fluorescence[i],
                         color=good_colors[i], alpha=0.45)

    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')

    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2.3)
    ax.spines['left'].set_linewidth(2.3)
    ax.tick_params(width=2.3)
    ax.tick_params(axis='both', which='major', length=6)

    plt.xlabel('Time, seconds', fontsize=15, fontweight='bold', labelpad=10)
    plt.ylabel('[4MU], \u03bcM', fontsize=15, fontweight='bold', labelpad=10)

    plt.legend(title=f"{legend_label}", loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='medium', fancybox=True, borderaxespad=0.5, ncol=1)
    save_figure(f'{pics_dir}{bn}_prog_curves', dpi=dpi, save_eps=save_eps)
    plt.show()

    ##################################################################################################################

    # Do the Michaelis-Menten fit
    params, covariance = curve_fit(michaelis_menten, sub_conc_list, velocities)

    kcat_optimized, Km_optimized = params
    kcat_stdev, Km_stdev = np.sqrt(np.diag(covariance))

    rel_kcat_unc = kcat_stdev / kcat_optimized
    rel_Km_unc = Km_stdev / Km_optimized
    kcat_Km_unc = kcat_optimized / Km_optimized * np.sqrt(rel_kcat_unc**2 + rel_Km_unc**2)

    avg_velocities = []
    for i in range(0, len(velocities), len(replicate_rows) + 1):
        avg_velocities.append(np.mean(velocities[i:i+len(replicate_rows)+1]))

    avg_velocities = []
    std_errors = []

    for i in range(0, len(velocities), len(replicate_rows) + 1):
        avg_velocity = np.mean(velocities[i:i+len(replicate_rows)+1])
        std_error = np.std(velocities[i:i+len(replicate_rows)+1], ddof=1) / np.sqrt(len(replicate_rows) + 1)
        avg_velocities.append(avg_velocity)
        std_errors.append(std_error)

    fig, ax = plt.subplots(figsize=(4, 4))

    x_values = np.linspace(0, max(sub_concs), 100)
    y_values = michaelis_menten(x_values, kcat_optimized, Km_optimized)
    ax.plot(x_values, y_values, color='black', linewidth=2)

    max_y_value = max(y_values)
    max_x_value = max(sub_concs)

    plt.xlabel(f'{legend_label}', fontsize=15, fontweight='bold', labelpad=10)
    if not ylabel:
        plt.ylabel('v/[E] x10$^{-3}$ (s$^{-1}$)', fontsize=15, fontweight='bold', labelpad=10)
    else:
        plt.ylabel(ylabel, fontsize=15, fontweight='bold', labelpad=10)

    ax.scatter(sub_concs, avg_velocities, color=good_red, s=100)
    errorbar = ax.errorbar(sub_concs, avg_velocities, yerr=std_errors, color='black', fmt='none', capsize=4, linewidth=2.5)

    for cap in errorbar[1]:
        cap.set_markeredgewidth(2)

    y_ticks = ax.get_yticks()
    y_ticklabels = [f"{tick * 1e3:.1f}" for tick in y_ticks]
    # Pin the locator before replacing the labels, otherwise matplotlib warns
    # 'FixedFormatter should only be used together with FixedLocator'.
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticklabels)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2.3)
    ax.spines['left'].set_linewidth(2.3)
    ax.tick_params(width=2.3)
    ax.tick_params(axis='both', which='major', length=6)

    save_figure(f'{pics_dir}{bn}_mm_fit_{fit_type}', dpi=dpi, save_eps=save_eps)

    plt.show()

    if not silent:
        print(f"kcat:{kcat_optimized:.6f}\u00b1{kcat_stdev:.6f}s\u207b\u00b9")
        print(f"Km:{Km_optimized:.2f}\u00b1{Km_stdev:.2f}\u03bcM")
        print(f"kcat/Km: {(kcat_optimized/Km_optimized)*1000000:.2f} \u00b1 {kcat_Km_unc*1000000:.2f} M\u207b\u00b9 s\u207b\u00b9")
        print()

        for i, sub_conc in enumerate(sub_concs):
            print(f"[SUBST] = {sub_conc}uM, v0/[E] = {avg_velocities[i]*1e-6}")

    return avg_velocities


###############################################################################
# Plate reader data processing and screening functions
###############################################################################

def parse_excels(excel_file, active_wells, start_row=30, skipfooter=35):
    num_samples = len(active_wells) if active_wells is not None else 0

    bn = os.path.basename(excel_file).replace('.xlsx','')

    df = pd.read_excel(excel_file, header=start_row, skipfooter=skipfooter, engine='openpyxl')
    df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
    df.reset_index(drop=True, inplace=True)

    if 'Time' not in df.columns:
        print(df.head(3))
        raise ValueError("Please vary start_row. Time is not observed in the column.")

    # Convert the time column to minutes
    df['Time'] = _time_column_to_seconds(df['Time'])
    df['Time'] = df['Time'] / 60
    df_final = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
    df_final.head()
    data = df_final
    return data


def trim_time_end(df, time_cutoff_hours=6):
    df = df[df["Time"] < time_cutoff_hours].copy()
    return df


def trim_time_start(df, time_cutoff_hours=1):
    df = df[df["Time"] > time_cutoff_hours].copy()
    df.loc[:, "Time"] -= df["Time"].iloc[0]
    return df


def normalize_to_first_row(df, active_wells):
    for well in active_wells:
        df[well] -= df[well].iloc[0]
    return df


def normalize_to_min(df, active_wells):
    for well in active_wells:
        df[well] -= min(df[well])
    return df


def draw_progression_curve_v0(excel_file, rows, columns, hit_definition_num, blank, time_start, time_end, label=True, start_row=30, ignore=None, reverse=True, save=None, dpi=None, save_eps=None):
    # Active well define
    active_wells = []
    for row in rows:
        for column in columns:
            if ignore:
                if f"{row}{column}" in ignore:
                    continue
            active_wells.append(f"{row}{column}")

    # Data preprocessing
    df = parse_excels(excel_file, active_wells, start_row=start_row)
    df = trim_time_end(df, time_cutoff_hours=time_end)
    df = trim_time_start(df, time_cutoff_hours=time_start)

    # Hit define
    if reverse:
        possible_hits = df.iloc[-1].nsmallest(hit_definition_num).index
    else:
        possible_hits = df.iloc[-1].nlargest(hit_definition_num).index
    print(f"Hits are: {', '.join(possible_hits)}")

    # Plotting
    fig, ax = plt.subplots(figsize=(4, 3.5))
    for well in active_wells:
        if well in blank:
            ax.plot(df['Time'], df[well], label=well, color='black', linestyle='dashed')
        elif well in possible_hits:
            ax.plot(df['Time'], df[well], label=well, color='red')
            x_pos = df['Time'].iloc[-1]
            y_pos = df[well].iloc[-1]
            if label:
                ax.text(x_pos+(x_pos*0.0), y_pos, well, color='red', ha='left', va='bottom', fontsize=8)
        else:
            ax.plot(df['Time'], df[well], label=well, color='lightgray', alpha=0.2)

    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('RFU')
    previous_xlim = ax.get_xlim()
    ax.set_xlim(previous_xlim[0], (previous_xlim[1]*1.01))

    plt.tight_layout()
    if save:
        save_figure(save, dpi=dpi, save_eps=save_eps)
    plt.show()


def draw_progression_curve_seperate_v0(excel_file, rows, columns, bg, time_start, time_end, start_row=48, reverse=False, save=None, dpi=None, save_eps=None):
    # Active well define
    active_wells, pairs = [], []
    for row in rows:
        for column in columns:
            active_wells.append(f"{row}{column}")
            pairs.append((f'{row}{column}', bg))

    # Data preprocessing
    df = parse_excels(excel_file, active_wells, start_row=start_row)
    df = trim_time_end(df, time_cutoff_hours=time_end)
    df = trim_time_start(df, time_cutoff_hours=time_start)

    # Find global min and max values for normalization
    y_min = df[[col for pair in pairs for col in pair]].min().min()
    y_max = df[[col for pair in pairs for col in pair]].max().max()

    # Set up the plot grid
    num_plots = len(pairs)
    cols = 12
    rows = (num_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten()

    # Plot each pair
    for i, (col1, col2) in enumerate(pairs):
        ax = axes[i]
        ax.plot(df['Time'], df[col1], color='red')
        ax.plot(df['Time'], df[col2], color='black')
        ax.set_ylim(y_min, y_max)

    # Remove any empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if save:
        save_figure(save, dpi=dpi, save_eps=save_eps)
    plt.show()


###############################################################################
# Circular dichroism (CD) spectroscopy functions
###############################################################################

def smooth_dataframe(df, window_size):
    df['deg'] = df['deg'].rolling(window=window_size, center=True, min_periods=1).mean()
    df['MRE'] = df['MRE'].rolling(window=window_size, center=True, min_periods=1).mean()
    return df


def plot_CD_spectrum_temperature_interval(file_path, protein_concentration, protein_length, pathlength=1, wavelength_min=190, window_size=1, colors=None, save=None, dpi=None, save_eps=None):
    f = open(file_path)
    data = f.read()
    f.close()

    data = data.split("Channel 1\n")[1].split("Channel 2\n")[0].strip()
    data = [el.split(",") for el in data.split("\n")]
    cols = ["Wavelength"]
    cols.extend(data[0][1:])

    df = pd.DataFrame(data[1:], columns=cols)
    df["Wavelength"] = df["Wavelength"].astype(float)
    df = df[df["Wavelength"] > wavelength_min]

    fig, axes = plt.subplots(2, 1, figsize=(6, 4))

    for i, col in enumerate(cols[1:]):
        df[col] = df[col].astype(float)
        tmp_df = df[["Wavelength", col]].copy()
        tmp_df.columns = ["Wavelength", "deg"]
        tmp_df[f"MRE"] = tmp_df[f"deg"] / (pathlength * protein_concentration * protein_length)
        tmp_df = smooth_dataframe(tmp_df, window_size)

        if colors:
            axes[0].plot(tmp_df["Wavelength"].to_list(), tmp_df["MRE"].to_list(), label=int(float(col)), color=colors[i])
            axes[1].plot(tmp_df["Wavelength"].to_list(), tmp_df["deg"].to_list(), color=colors[i])
        else:
            axes[0].plot(tmp_df["Wavelength"].to_list(), tmp_df["MRE"].to_list(), label=int(float(col)))
            axes[1].plot(tmp_df["Wavelength"].to_list(), tmp_df["deg"].to_list())

        axes[0].set_ylabel("MRE")
        axes[0].set_xticks([])
        axes[0].set_xticklabels([])
        axes[1].set_ylabel("deg")
        axes[1].set_xlabel("Wavelength")

    fig.legend(loc='center left', bbox_to_anchor=(1.0, 0.35))
    plt.tight_layout()

    if save:
        save_figure(save, dpi=dpi, save_eps=save_eps)

    plt.show()


def plot_CD_spectrum_temperature_interval_A1(file_path, file_path2, protein_concentration, protein_length, pathlength=1, wavelength_min=190, window_size=1, colors=None, save=None, dpi=None, save_eps=None):
    f = open(file_path)
    data = f.read()
    f.close()

    data = data.split("Channel 1\n")[1].split("Channel 2\n")[0].strip()
    data = [el.split(",") for el in data.split("\n")]
    cols = ["Wavelength"]
    cols.extend(data[0][1:])

    df = pd.DataFrame(data[1:], columns=cols)
    df["Wavelength"] = df["Wavelength"].astype(float)
    df = df[df["Wavelength"] > wavelength_min]

    fig, axes = plt.subplots(2, 1, figsize=(6, 4))

    for i, col in enumerate(cols[1:]):
        df[col] = df[col].astype(float)
        tmp_df = df[["Wavelength", col]].copy()
        tmp_df.columns = ["Wavelength", "deg"]
        tmp_df[f"MRE"] = tmp_df[f"deg"] / (pathlength * protein_concentration * protein_length)
        tmp_df = smooth_dataframe(tmp_df, window_size)

        if colors:
            axes[0].plot(tmp_df["Wavelength"].to_list(), tmp_df["MRE"].to_list(), label=int(float(col)), color=colors[i])
            axes[1].plot(tmp_df["Wavelength"].to_list(), tmp_df["deg"].to_list(), color=colors[i])
        else:
            axes[0].plot(tmp_df["Wavelength"].to_list(), tmp_df["MRE"].to_list(), label=int(float(col)))
            axes[1].plot(tmp_df["Wavelength"].to_list(), tmp_df["deg"].to_list())

        axes[0].set_ylabel("MRE")
        axes[0].set_xticks([])
        axes[0].set_xticklabels([])
        axes[1].set_ylabel("deg")
        axes[1].set_xlabel("Wavelength")

    f = open(file_path2)
    data = f.read()
    f.close()

    data = data.split("XYDATA\n")[1].split("\n##### Extended Information")[0].strip()
    data = [el.split(",") for el in data.split("\n")]
    cols = ["Wavelength", "deg", "HT"]

    df2 = pd.DataFrame(data, columns=cols)
    df2["Wavelength"] = df2["Wavelength"].astype(float)
    df2["deg"] = df2["deg"].astype(float)
    df2[f"MRE"] = df2[f"deg"] / (pathlength * protein_concentration * protein_length)
    df2 = smooth_dataframe(df2, window_size)

    axes[0].plot(df2["Wavelength"].to_list(), df2["MRE"].to_list(), label="95C -> 25C", color='gray')
    axes[1].plot(df2["Wavelength"].to_list(), df2["deg"].to_list(), color='gray')

    fig.legend(loc='center left', bbox_to_anchor=(1.0, 0.35))
    plt.tight_layout()

    if save:
        save_figure(save, dpi=dpi, save_eps=save_eps)

    plt.show()


def plot_CD_222nm_temperature_interval(file_path, protein_concentration, protein_length, window_size=1, pathlength=1, save=None, dpi=None, save_eps=None):
    f = open(file_path)
    data = f.read()
    f.close()

    data = data.split("XYDATA\n")[1].split("\n##### Extended Information")[0].strip()
    data = [el.split(",") for el in data.split("\n")]
    cols = ["temperature", "deg", "HT"]

    df = pd.DataFrame(data, columns=cols)
    df["temperature"] = df["temperature"].astype(float)
    df["deg"] = df["deg"].astype(float)

    fig, axes = plt.subplots(2, 1, figsize=(6, 4))

    df[f"MRE"] = df[f"deg"] / (pathlength * protein_concentration * protein_length)
    df = smooth_dataframe(df, window_size)

    axes[0].plot(df["temperature"].to_list(), df["MRE"].to_list())
    axes[1].plot(df["temperature"].to_list(), df["deg"].to_list())

    axes[0].set_ylabel("MRE")
    axes[0].set_xticks([])
    axes[0].set_xticklabels([])
    axes[1].set_ylabel("deg")
    axes[1].set_xlabel("temperature")

    fig.legend(loc='center left', bbox_to_anchor=(1.0, 0.35))
    plt.tight_layout()

    if save:
        save_figure(save, dpi=dpi, save_eps=save_eps)

    plt.show()
