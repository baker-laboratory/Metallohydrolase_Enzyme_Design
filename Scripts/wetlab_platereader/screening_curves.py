"""Reaction progress curves for plate-reader eluate screens.

A screen runs one designed protein per well and follows product formation for
an hour or so. What matters is the shape of each well's progress curve, so the
helpers here parse a Neo2 export into a wide time x well table, trim it to a
usable window, normalize each trace to its own starting point, and draw the
whole plate on one axis with the fastest wells labeled.

    draw_progression_curve_v0(...)            one well per design
    draw_progression_curve_v0_replicate(...)  wells grouped into replicates

Vendored into this repository so the analysis notebooks are self-contained.
Original author: Donghyo Kim. One change from the original: the output format
now follows the file extension instead of always being EPS.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Raster resolution for saved progress-curve panels. The notebooks set this to
# match their own FIGURE_DPI so every figure in the repository is consistent.
SAVE_DPI = 150

def parse_excels(excel_path):
    xls = pd.ExcelFile(excel_path)
    sheet_name = xls.sheet_names[0]
    df_raw = pd.read_excel(xls, sheet_name=sheet_name, header=None)

    time_header_rows = df_raw.index[df_raw.iloc[:, 1] == "Time"].tolist()
    results_row = df_raw.index[df_raw.iloc[:, 0].astype(str).str.contains("Results", na=False)].tolist()
    end_row_limit = results_row[-1] if results_row else df_raw.shape[0]

    melted_blocks = []

    for i, header_row in enumerate(time_header_rows):
        start_row = header_row + 1
        end_row = time_header_rows[i+1] if i+1 < len(time_header_rows) else end_row_limit

        column_names = df_raw.iloc[header_row].tolist()
        data_block = df_raw.iloc[start_row:end_row]
        data_block.columns = column_names
        data_block = data_block.dropna(subset=["Time"]).reset_index(drop=True)

        melted = data_block.melt(id_vars="Time", var_name="Well", value_name="Fluorescence")
        melted = melted[~melted["Fluorescence"].isna()]
        melted = melted[~melted["Well"].str.contains("T°", na=False)]
        melted = melted[pd.to_numeric(melted["Fluorescence"], errors="coerce").notna()]
        melted_blocks.append(melted)

    if not melted_blocks:
        raise ValueError("No valid data blocks found.")

    # Combine all long-format blocks
    df_long = pd.concat(melted_blocks, ignore_index=True)

    # Convert Time to hours
    df_long["Time"] = pd.to_datetime(df_long["Time"], format="%H:%M:%S", errors="coerce").dt.time
    df_long = df_long.dropna(subset=["Time"])
    df_long["Time"] = df_long["Time"].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second) / 3600
    df_long = df_long[df_long["Time"] != 0]

    # Average replicate fluorescence values for the same (Time, Well)
    df_grouped = df_long.groupby(["Time", "Well"], as_index=False)["Fluorescence"].mean()

    # Pivot to wide format: Time, A1, A2, ..., P24
    df_wide = df_grouped.pivot(index="Time", columns="Well", values="Fluorescence").reset_index()

    return df_wide

def trim_time_end(df, time_cutoff_hours = 6):
    df = df[df["Time"] < time_cutoff_hours].copy()
    return df
    
def trim_time_start(df, time_cutoff_hours = 1):
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


def draw_progression_curve_v0(excel_file, rows, columns, hit_definition_num, blank, time_start, time_end, label=True, start_row=30, ignore=None, rename_dic=None, norm=True, save_path=None): 
    # Active well define
    active_wells = []
    for row in rows:
        for column in columns:
            if ignore:
                if f"{row}{column}" in ignore:
                    continue
            active_wells.append(f"{row}{column}")
    
    # Data preprocessing
    df = parse_excels(excel_file) 
    df = trim_time_end(df, time_cutoff_hours = time_end)
    df = trim_time_start(df, time_cutoff_hours = time_start)
    #df = normalize_to_first_row(df, active_wells)
    if norm:
        df = normalize_to_min(df, active_wells)
    df = df[["Time"]+active_wells]
    
    if rename_dic:
        df.rename(columns=rename_dic, inplace=True)
        active_wells = [rename_dic[well] for well in active_wells]
                                                   
    # Hit define
    numeric_values = pd.to_numeric(df.iloc[-1][active_wells], errors='coerce')
    possible_hits = numeric_values.nlargest(hit_definition_num).index
    #possible_hits = numeric_values.nsmallest(hit_definition_num).index
    print (f"Hits are: {', '.join(possible_hits)}")
    
    # Plotting
    fig, ax = plt.subplots(figsize=(4, 4))
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
        
    # Set axis labels and title
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Absorbance change')
    #ax.set_ylim(0, 0.05)
    previous_xlim = ax.get_xlim()
    ax.set_xlim(previous_xlim[0], (previous_xlim[1]*1.01))
        
    plt.tight_layout()
    if save_path is not None:
        # Format follows the extension. Upstream hardcoded format="eps", which
        # wrote EPS bytes into whatever filename it was given.
        plt.savefig(save_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.show()


def draw_progression_curve_v0_replicate(excel_file, replicate_dic, time_start, time_end, label=True, start_row=30, blank=None, figsize=(4,4), norm=False, bg=False, save_path=None): 
    # Active well define
    active_wells = []
    for sample_id in replicate_dic:
        for well in replicate_dic[sample_id]:
            active_wells.append(well)
    
    # Data preprocessing
    df = parse_excels(excel_file)
    df = trim_time_end(df, time_cutoff_hours = time_end)
    df = trim_time_start(df, time_cutoff_hours = time_start)
    #df = normalize_to_first_row(df, active_wells)
    df = normalize_to_min(df, active_wells)
    
    if norm:
        if bg == False:
            raise ValueError (f"To normalize progression curve using protein concentration, background activity must be provided.")
            
        bg_activity = df[replicate_dic[bg]].mean(axis=1)
        
        for sample_id in replicate_dic:
            if sample_id == bg:
                continue
            for well in replicate_dic[sample_id]:
                df[well] = df[well] - bg_activity
                df[well] = df[well] / norm[sample_id]
    
    # Plotting
    fig, ax = plt.subplots(figsize=figsize)
    for sample_id in replicate_dic:
        if sample_id == bg:
            continue
        data = []
        for well in replicate_dic[sample_id]:
            data.append(f"df['{well}'].to_list()")
        avg_data = [np.mean(el) for el in eval(f"zip({','.join(data)})")]
        std_data = [np.std(el) for el in eval(f"zip({','.join(data)})")]

        if not sample_id == blank:
            plt.plot(df["Time"], avg_data, label=f'{sample_id}')
            plt.fill_between(df["Time"], np.array(avg_data) - np.array(std_data), np.array(avg_data) + np.array(std_data), alpha=0.45)
        else:
            plt.plot(df["Time"], avg_data, label=f'{sample_id}', color="black")
            plt.fill_between(df["Time"], np.array(avg_data) - np.array(std_data), np.array(avg_data) + np.array(std_data), alpha=0.45, color="black")

        if label:
            x_pos = df["Time"].to_list()[-1]
            y_pos = max(avg_data)
            ax.text(x_pos+(x_pos*0.0), y_pos, sample_id, ha='left', va='bottom', fontsize=8)
        
    # Set axis labels and title
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('OD405')
    previous_xlim = ax.get_xlim()
    ax.set_xlim(previous_xlim[0], (previous_xlim[1]*1.03))
        
    ax.legend(bbox_to_anchor=(1.3, 1), loc='upper left', borderaxespad=0.)
    #plt.tight_layout()
    if save_path is not None:
        # Format follows the extension. Upstream hardcoded format="eps", which
        # wrote EPS bytes into whatever filename it was given.
        plt.savefig(save_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.show()
