# Vendored into this repository so the analysis notebooks are self-contained.
# Original author: Donghyo Kim. Kept byte-identical to upstream apart from this
# header, so it can be re-synced by replacing the file.

import re
import pandas as pd
import datetime

##############################
# Utility functions and imports
##############################

_WELL_RE = re.compile(r"^[A-P]\d{1,2}$")


def _parse_neo2_xlsx(filepath):
    """Parse a Neo2 kinetics ``.xlsx`` export into the same tidy frame as the TSV parser.

    Layout: a metadata block at the top (labels in col 0), then one or more
    data blocks whose header row has ``"Time"`` in col 1, ``"T° <read>"`` in
    col 2, and well IDs (``A1``..``H12``) in col 3+. Times in the body are
    ``datetime.time`` objects.
    """
    raw = pd.read_excel(filepath, sheet_name=0, header=None)

    def _find(label):
        m = raw[raw.iloc[:, 0].astype(str).str.strip() == label]
        return None if m.empty else m.iloc[0, 1]

    serial_number = str(_find("Reader Serial Number:"))
    date_val = _find("Date")
    time_val = _find("Time")
    if date_val is None or time_val is None:
        raise ValueError(f"{filepath}: missing Date/Time metadata rows")
    d = date_val.date() if hasattr(date_val, "date") else pd.to_datetime(str(date_val)).date()
    t = time_val if isinstance(time_val, datetime.time) else pd.to_datetime(str(time_val)).time()
    stamp = datetime.datetime.combine(d, t).timestamp()

    hdr_matches = raw[raw.iloc[:, 1].astype(str).str.strip() == "Time"]
    if hdr_matches.empty:
        raise ValueError(f"{filepath}: could not find data header row (col 1 == 'Time')")

    frames = []
    for hdr_idx in hdr_matches.index:
        header = raw.iloc[hdr_idx].tolist()
        body = raw.iloc[hdr_idx + 1:].copy()
        body.columns = header
        # Body ends at the first row without a valid Time value
        body = body[body["Time"].apply(lambda x: isinstance(x, datetime.time))]
        if body.empty:
            continue
        well_cols = [c for c in body.columns if isinstance(c, str) and _WELL_RE.match(c)]
        if not well_cols:
            continue
        body = body[["Time"] + well_cols]
        body["Time"] = body["Time"].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)
        tidy = body.melt(id_vars="Time", var_name="Well", value_name="value")
        tidy = tidy.rename(columns={"Time": "time"})
        tidy["value"] = pd.to_numeric(tidy["value"], errors="coerce")
        tidy = tidy.dropna(subset=["value"])
        frames.append(tidy)

    if not frames:
        raise ValueError(f"{filepath}: found header row(s) but no numeric well data below")
    df = pd.concat(frames, ignore_index=True)
    df["datetime"] = stamp
    df["serial_number"] = serial_number
    return df


def parse_neo2_kinetics(filepath):
    """
    Parse a TSV file from neo2 with kinetics data.

    This function reads a TSV file containing kinetics data, processes it, and returns a tidy DataFrame.
    Optionally, it can also parse metadata from the filename.

    Parameters:
    -----------
    filepath : str
        The path to the TSV file to be parsed.

    Returns:
    --------
    pandas.DataFrame
        A tidy DataFrame containing the parsed kinetics data and optional metadata.

    The DataFrame contains the following columns:
    - time: Time in seconds.
    - well: Well identifier.
    - value: Kinetics measurement value.
    - serial_number: Instrument serial number.

    Notes:
    ------
    - The function expects the TSV file to be encoded in 'cp1252' and delimited by tabs.
    - The function will drop any rows with missing values in the 'value' column.
    - The function will convert the 'time' column from HH:MM:SS format to seconds.
    - The data is parsed starting from the line starting with `Time\tT°` and is expected to have a 'Well' column.
    - ``.xlsx`` exports are also supported (dispatched to :func:`_parse_neo2_xlsx`).
    """

    if filepath.lower().endswith(".xlsx"):
        return _parse_neo2_xlsx(filepath)

    ## parse instrument serial number
    df = pd.read_csv(
        filepath,
        encoding="cp1252",
        delimiter="\t",
        nrows=20,
    )
    serial_number = df[df.iloc[:, 0] == "Reader Serial Number:"]
    serial_number = serial_number.iloc[:, 1].values[0]

    # find the line number including data
    f = open(filepath, encoding="cp1252")
    lines = f.readlines()
    f.close()
    start_row = [line[:7] for line in lines].index("Time\tT°")

    _date = df[df.iloc[:, 0] == "Date"]
    _time = df[df.iloc[:, 0] == "Time"]
    _date_time = _date.iloc[:, 1].values[0] + " " + _time.iloc[:, 1].values[0]

    ## parse kinetics data
    df = pd.read_csv(
        filepath,
        encoding="cp1252",
        delimiter="\t",
        skiprows=range(1, start_row),
    )

    df = df.drop(df.columns[1], axis=1)
    df.rename(columns={"Time": "time"}, inplace=True)

    # time conversion
    df["time"] = pd.to_datetime(df["time"], format="%H:%M:%S", errors="coerce").dt.time

    # Convert the time column to seconds
    df["time"] = df["time"].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)

    # make dataframe tidy
    df = df.melt(id_vars=["time"], var_name="Well", value_name="value")
    df.dropna(inplace=True)

    # ensure value is numeric
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # add date time
    df["datetime"] = datetime.datetime.strptime(_date_time, "%m/%d/%Y %I:%M:%S %p").timestamp()

    # add info
    df["serial_number"] = serial_number

    return df