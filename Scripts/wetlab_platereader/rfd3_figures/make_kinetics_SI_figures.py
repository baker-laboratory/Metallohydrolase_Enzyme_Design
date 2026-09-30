# Adapted from FOR_RFdiffusion3_paper/scripts/make_kinetics_SI_figures.py (2026-09-29).
# Paper geometry, colors and labels are retained; paths resolve inside this repository.
"""SI kinetics figures: Round 1 (+screen), ZAPP-1 knockouts, Round 2 (+screen).

Panel pairs are [progress curves | Michaelis-Menten fit].  Every number annotated on a
panel is read from paraoxon_kinetics_ALL_DATA.xlsx, so the figures and the table can
never disagree.  Raw traces are re-parsed from the plate reader files for plotting only.
"""
import os, sys, json, re
import numpy as np, pandas as pd, matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, MultipleLocator
from neo2_util import parse_neo2_kinetics
from kinetics import parse_kinetics

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[3] / "Manuscript_Data" / "Phosphotriesterase_RFdiffusion3_Science_2026")
XLSX   = os.path.join(ROOT, "supplemental_data", "paraoxon_kinetics_ALL_DATA.xlsx")
FIGDIR = os.path.join(ROOT, "wetlab_data_plots", "paper_figures")
RAW = os.path.join(ROOT, "raw_wetlab_data")
SCREEN_R1 = next(str(p) for p in __import__("pathlib").Path(RAW).glob("250919_order1_plate1*.xlsx"))
SCREEN_R2 = next(str(p) for p in __import__("pathlib").Path(RAW).glob("260525_i3smw*.csv"))

CONC  = np.array([9600, 4800, 2400, 1200, 600, 300], float)
RAMP  = {9600:"#662d88", 4800:"#79459c", 2400:"#8d5daf",
         1200:"#a274c2", 600:"#b78cd6", 300:"#cca5ea"}
RED, INK, INK2, GRAY = "#c2521f", "#0b0b0b", "#52514e", "#b8b7b2"
TICKFS, LABFS = 6.5, 7.5        # tick numbers / axis labels
E_REL = 0.06                    # rel. sd on [E]0, same budget as the xlsx

# scaffold-family colors, sampled from the main figure (panel I of enzyme_experiments.pdf).
# FAM[design] -> (marker/fit color, darker variant that is legible as text)
TEAL, NAVY, MBLUE, PINK, PEACH = "#4fb9af", "#4b5faa", "#6686c5", "#f0a4b3", "#ffc6b2"
FAM = {
 "ZAPP-1":(TEAL,"#2e8b82"), "ZAPP-1 (no NaHCO3)":(TEAL,"#2e8b82"),
 "R1 p1D7":(TEAL,"#2e8b82"), "R1 p1D8":(TEAL,"#2e8b82"), "R1 p1E10":(TEAL,"#2e8b82"),
 "ZAPP-2":(NAVY,"#3c4d8c"),  "R2 p2B12":(NAVY,"#3c4d8c"),
 "ZAPP-3":(MBLUE,"#4a68a8"),
 "ZAPP-4":(PINK,"#c9647a"),
 "ZAPP-5":(PEACH,"#cf7c52"), "R2 p1D9":(PEACH,"#cf7c52"), "R2 p1H4":(PEACH,"#cf7c52"),
}
for _m in range(1, 7):
    FAM[f"ZAPP-1 MUT{_m}"] = (TEAL, "#2e8b82")
famcol = lambda n: FAM.get(n, (RED, INK))

fam = next((f for f in ["Arial","Helvetica","Liberation Sans","DejaVu Sans"]
            if f in {x.name for x in font_manager.fontManager.ttflist}), "sans-serif")
STYLE = {
    "font.family": fam, "font.size": 7,
    "axes.linewidth": 0.8, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.top": False, "ytick.right": False, "xtick.bottom": True, "ytick.left": True,
    "axes.spines.top": True, "axes.spines.right": True,
    "axes.spines.bottom": True, "axes.spines.left": True,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.minor.width": 0.7, "ytick.minor.width": 0.7,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
    "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "custom",
    "mathtext.rm": fam, "mathtext.it": f"{fam}:italic", "mathtext.bf": f"{fam}:bold",
    "mathtext.bfit": f"{fam}:italic:bold",
}
plt.rcParams.update(STYLE)

# ── registry: pull file / wells / [E] straight from the source notebook ──────
_REGISTRY = json.loads(__import__("pathlib").Path(__file__).with_name("plate_registry.json").read_text())
def spec(cell, cols_override=None):
    sp = dict(_REGISTRY[str(cell)])
    sp["path"] = os.path.join(RAW, sp.pop("file"))
    if cols_override is not None:
        sp["cols"] = cols_override
    return sp

#   xlsx name,             title,                        subtitle,               cell, cols
ROUND1 = [
 ("ZAPP-1",             "ZAPP-1",                      "R1 p1D1",                 9, None),
 ("ZAPP-1 (no NaHCO3)", "ZAPP-1, no NaHCO$_3$",        "R1 p1D1",                 5, None),
 ("R1 p1D7",            "R1 p1D7",                     "ZAPP-1 scaffold",        12, None),
 ("R1 p1D8",            "R1 p1D8",                     "ZAPP-1 scaffold",        14, None),
 ("R1 p1E10",           "R1 p1E10",                    "ZAPP-1 scaffold",        16, None)]
MUTANTS = [
 ("ZAPP-1 MUT1","ZAPP-1 H93A", "Mutant 1",19,None),
 ("ZAPP-1 MUT2","ZAPP-1 H89A", "Mutant 2",21,None),
 ("ZAPP-1 MUT3","ZAPP-1 K16A", "Mutant 3",23,None),
 ("ZAPP-1 MUT4","ZAPP-1 H170A","Mutant 4",25,None),
 ("ZAPP-1 MUT5","ZAPP-1 H133A","Mutant 5",27,None),
 ("ZAPP-1 MUT6","ZAPP-1 E92A", "Mutant 6",29,None)]
# grouped by scaffold: s3 (ZAPP-2, p2B12) | s7 (ZAPP-3) | s6 (ZAPP-4) | s1 (ZAPP-5, p1D9, p1H4)
ROUND2 = [
 ("ZAPP-2",   "ZAPP-2",   "R2 p2C4",           46, None),
 ("R2 p2B12", "R2 p2B12", "ZAPP-2 scaffold",   48, None),
 ("ZAPP-3",   "ZAPP-3",   "R2 p2G1",           54, [5,6,7]),   # corrected wells
 ("ZAPP-4",   "ZAPP-4",   "R2 p2E3",           51, None),
 ("ZAPP-5",   "ZAPP-5",   "R2 p1F6",           32, None),
 ("R2 p1D9",  "R2 p1D9",  "ZAPP-5 scaffold",   38, None),
 ("R2 p1H4",  "R2 p1H4",  "ZAPP-5 scaffold",   40, None)]     # was mislabelled ZAPP-2

TAB = pd.read_excel(XLSX, sheet_name="kinetics").set_index("design")

# ── data helpers ────────────────────────────────────────────────────────────
_cache = {}
def traces(sp):
    key = (sp["path"], tuple(sp["cols"]))
    if key not in _cache:
        raw = parse_neo2_kinetics(sp["path"])
        d = raw.groupby(["Well","time"], as_index=False).agg(v=("value","mean"))
        d["time"] = d["time"].astype(float) - d.groupby("Well")["time"].transform("min")
        _cache[key] = d.pivot(index="time", columns="Well", values="v").sort_index()
    return _cache[key]

def mm_points(sp, spu):
    _, _, mm = parse_kinetics(data_path=sp["path"],
        blocks=[dict(rows=list("ABCDEF"), concentrations_uM=list(CONC),
                     enzyme_cols=sp["cols"], bg_cols=sp["bg"])],
        enzyme_uM=sp["E"], signal_per_uM=spu,
        time_range_seconds=sp["tr"], baseline_subtract=True)
    a = mm["_agg"].sort_values("concentration_uM")
    n = a["n"].to_numpy(float)
    return (a["concentration_uM"].to_numpy(float),
            a["v_mean_uM_per_s"].to_numpy(float)/sp["E"],
            a["v_std_uM_per_s"].fillna(0).to_numpy(float)/sp["E"]/np.sqrt(np.maximum(n,1)))

def style(ax, minor_x=4, minor_y=4):
    for s_ in ax.spines.values(): s_.set_visible(True)
    ax.xaxis.set_minor_locator(AutoMinorLocator(minor_x))
    ax.yaxis.set_minor_locator(AutoMinorLocator(minor_y))
    ax.tick_params(which="both", top=False, right=False)
    ax.tick_params(which="both", labelsize=TICKFS)   # one tick size for every panel

def panel_letter(ax, L, x=-0.34, y=1.06):
    ax.text(x, y, L, transform=ax.transAxes, fontsize=12, fontweight="normal",
            va="bottom", ha="left", color=INK)

# ── the two panels of a design ──────────────────────────────────────────────
def draw_progress(ax, sp, spu, tmax_min=None):
    """Whole acquisition on the real clock; the shaded span is the window that was
    actually fit (short for the fast designs, whose curves bend early)."""
    piv = traces(sp); t = piv.index.to_numpy(float)
    lo, hi = sp["tr"]                       # gray out what was NOT fitted
    if lo > t.min(): ax.axvspan(t.min()/60.0, lo/60.0, color=GRAY, alpha=0.22, lw=0, zorder=0)
    if hi < t.max(): ax.axvspan(hi/60.0, t.max()/60.0, color=GRAY, alpha=0.22, lw=0, zorder=0)
    i0 = int(np.searchsorted(t, lo))            # first sample inside the fit window
    win = (t >= lo) & (t <= min(hi, t.max()))
    ylo, yhi = 0.0, 0.0
    for c, r in zip(CONC, list("ABCDEF")):
        y = np.column_stack([piv[f"{r}{k}"].to_numpy(float) for k in sp["cols"]])/spu
        # Baseline = each well's minimum up to the start of the fit window.  For a
        # clean monotonic well that is simply its t=0 read, so product starts at zero;
        # for a well whose first reads are a bubble (p1D9 C7: 0.914 then 0.209) it
        # returns the true baseline instead of the artifact.
        y = y - np.nanmin(y[:i0+1], axis=0)
        mu = np.nanmean(y, 1); sd = np.nanstd(y, 1, ddof=1)/np.sqrt(y.shape[1])
        tt = t/60.0
        ax.fill_between(tt, mu-sd, mu+sd, color=RAMP[c], alpha=0.30, lw=0)
        ax.plot(tt, mu, color=RAMP[c], lw=1.0, solid_capstyle="round")
        ylo = min(ylo, np.nanmin(mu-sd)); yhi = max(yhi, np.nanmax(mu+sd))
    rng = yhi - ylo                             # scale to the whole acquisition
    ax.set_ylim(ylo - 0.06*rng, yhi + 0.19*rng) # extra headroom so the [E]0 box clears
    ax.text(0.058, 0.930, "[E]$_0$ = %s µM" % plain(sp["E"], sp["E"]*E_REL),
            transform=ax.transAxes, va="top", ha="left", fontsize=5.6,
            color=INK, zorder=7,
            bbox=dict(boxstyle="square,pad=0.30", fc="white", ec=INK2, lw=0.4))
    ax.set_xlabel("Time (min)", fontsize=LABFS, labelpad=2)
    ax.set_ylabel("[4-nitrophenol] (µM)", fontsize=LABFS, labelpad=2)
    ax.set_xlim(0, None); style(ax)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=4)); ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=4))

def draw_mm(ax, name, sp, spu):
    row = TAB.loc[name]
    S, v, e = mm_points(sp, spu)
    kcat, Km = row["kcat (s-1)"], row["Km (uM)"]
    unresolved = str(row["kcat (reported)"]).strip() == "n.d."
    sc = 1e-3                                    # every kinetics panel uses 10^-3 s^-1
    col = famcol(name)[0]
    ax.plot(S/1000, v/sc, "o", ms=3.6, mfc=col, mec="black", mew=0.5, ls="none", zorder=3)
    ax.errorbar(S/1000, v/sc, yerr=e/sc, fmt="none", ecolor="black", elinewidth=0.9,
                capsize=2.0, capthick=0.9, zorder=5)          # always on top of the marker
    xf = np.linspace(0, 10500, 200)
    if unresolved:                                   # first-order regime only
        k = row["kcat/Km chord (M-1 s-1)"]/1e6
        ax.plot(xf/1000, k*xf/sc, color=famcol(name)[1], lw=1.1, zorder=2)
    else:
        ax.plot(xf/1000, (kcat*xf/(Km+xf))/sc, color=famcol(name)[1], lw=1.1, zorder=2)
    ax.set_xlabel("[Paraoxon] (mM)", fontsize=LABFS, labelpad=2)
    ax.set_ylabel("$v_0$/[E]$_0$ (10$^{-3}$ s$^{-1}$)", fontsize=LABFS, labelpad=2)
    ax.set_xlim(0, 10.5)
    ymin = min(0.0, float(((v-e)/sc).min())*1.20)
    ax.set_ylim(ymin, None)
    # 0 / 4.8 / 9.6 labelled, 2.4 and 7.2 as minor ticks -- five labels collide
    # in the narrow panels of the knockout figure.
    ax.set_xticks([0, 4.8, 9.6]); style(ax)
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=4))
    ax.text(0.055, 0.945, "$n$ = 3", transform=ax.transAxes, ha="left",
            va="top", fontsize=6.0, color=INK2)

    return param_lines(row, unresolved)

def _round_up_1sf(e):
    """Uncertainty to 1 s.f., rounded UP; 9 promotes a decade. Mirrors the xlsx rule."""
    from math import floor, log10, ceil
    d = floor(log10(abs(e))); lead = ceil(abs(e)/10**d - 1e-9)
    if lead >= 9: lead, d = 1, d + 1
    return lead*10.0**d, -d

def plain(v, e, comma=False):
    """'value ± error' as a plain decimal, truncated at the error's last digit."""
    er, dec = _round_up_1sf(e)
    if dec > 0:
        return f"{round(v,dec):.{dec}f} ± {er:.{dec}f}"
    f = ",.0f" if comma else ".0f"
    return f"{round(v,dec):{f}} ± {er:{f}}"

def param_lines(row, unresolved):
    """Parameter strings in the agreed order: kcat, Km, kcat/Km, kcat/kuncat."""
    if unresolved:
        return ["$k_\\mathrm{cat}$, $K_\\mathrm{M}$ not resolved (saturation not approached)",
                f"$k_\\mathrm{{cat}}$/$K_\\mathrm{{M}}$ = {row['kcat/Km chord (reported)']} M$^{{-1}}$s$^{{-1}}$"
                "   (first-order fit)"]
    kc = plain(row["kcat (s-1)"], row["kcat sd (total)"])
    ke = plain(row["kcat/kuncat"], row["kcat/kuncat sd"], comma=True)
    return [f"$k_\\mathrm{{cat}}$ = {kc} s$^{{-1}}$    $K_\\mathrm{{M}}$ = {row['Km (reported, mM)']} mM",
            f"$k_\\mathrm{{cat}}$/$K_\\mathrm{{M}}$ = {row['kcat/Km (reported)']} M$^{{-1}}$s$^{{-1}}$"
            f"    $k_\\mathrm{{cat}}$/$k_\\mathrm{{uncat}}$ = {ke}"]

def substrate_legend(fig, bbox):
    h = [Line2D([], [], color=RAMP[c], lw=1.8) for c in CONC]
    lg = fig.legend(h, [f"{c:,.0f}" for c in CONC], title="[Paraoxon] (µM)",
                    loc="upper left", bbox_to_anchor=bbox, frameon=False,
                    fontsize=6, title_fontsize=6.5, handlelength=1.3, labelspacing=0.30)
    lg._legend_box.align = "left"
    fig.add_artist(lg)
    # the progress curves show the whole acquisition; say which part was fit
    lg2 = fig.legend([Patch(facecolor=GRAY, alpha=0.22, lw=0)], ["outside the\ninitial-velocity\nfit window"],
                     loc="upper left", bbox_to_anchor=(bbox[0], bbox[1] - 0.128),
                     frameon=False, fontsize=6, handlelength=1.3, handleheight=1.1,
                     labelspacing=0.30)
    lg2._legend_box.align = "left"

def place_labels(ax, ends, x_end, fs=5.6, dxfrac=0.035):
    """Right-edge labels pushed apart in DATA units so they never overlap.

    A displaced label gets an elbow leader -- short horizontal off the trace, a
    vertical riser in its own column, then a short horizontal into the text.  A
    straight diagonal leader cuts across whatever label sits between the two.
    """
    ax.figure.canvas.draw()
    ylo, yhi = ax.get_ylim(); xlo, xhi = ax.get_xlim()
    dy = 0.105*(yhi-ylo); dx = dxfrac*(xhi-xlo)
    ends = sorted(ends, key=lambda t: -t[0]); ys = []
    for i, (y, _, _) in enumerate(ends):
        ys.append(y if i == 0 else min(y, ys[-1]-dy))
    for i, ((y, lab, col), yy) in enumerate(zip(ends, ys)):
        if abs(yy - y) > 1e-9:                     # risers staggered so they never overlap
            xk = x_end + dx*(0.22 + 0.16*(i % 3))
            ax.plot([x_end, xk, xk, x_end + dx*0.93], [y, y, yy, yy],
                    color=col, lw=0.5, clip_on=False, zorder=6,
                    solid_capstyle="butt", solid_joinstyle="miter")
        ax.text(x_end + dx, yy, lab, fontsize=fs, color=col, va="center",
                ha="left", fontweight="bold", clip_on=False, zorder=7)

# ── screening panels ────────────────────────────────────────────────────────
def screen_round1(ax, hits):
    piv = parse_neo2_kinetics(SCREEN_R1).groupby(["Well","time"], as_index=False)\
            .agg(v=("value","mean")).pivot(index="time", columns="Well", values="v").sort_index()
    t = (piv.index.to_numpy(float) - piv.index.min())/3600.0
    Y = piv.to_numpy(float); Y = Y - Y[0]
    other = [i for i,w in enumerate(piv.columns) if w not in hits]
    ax.fill_between(t, np.nanpercentile(Y[:,other],5,axis=1),
                    np.nanpercentile(Y[:,other],95,axis=1), color=GRAY, alpha=0.55, lw=0)
    ends = []
    for w, (lab, dname) in hits.items():
        if w in piv.columns:
            c = famcol(dname)[1]
            y = Y[:, list(piv.columns).index(w)]
            ax.plot(t, y, color=c, lw=1.1); ends.append((y[-1], lab, c))
    ax.set_xlabel("Time (h)", fontsize=LABFS, labelpad=2)
    ax.set_ylabel("$\\Delta$A$_{405}$", fontsize=LABFS, labelpad=2)
    ax.set_xlim(0, 1.24); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); style(ax)
    ax.set_ylim(top=ax.get_ylim()[1]*1.22)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=4))
    place_labels(ax, ends, t[-1])

def screen_round2(ax, hits):
    """i3 384-well eluate screen. Time is in days in the CSV; reader wells from the
    exported initial-rate table (source well P2C4 etc. -> reader well)."""
    csv = os.path.join(os.path.dirname(SCREEN_R2),
                       os.path.basename(SCREEN_R2).replace(".xlsx", ".csv"))
    d = pd.read_csv(csv)
    wells = [c for c in d.columns if re.match(r"^[A-P]\d{1,2}$", str(c))]
    t = d["Time"].to_numpy(float) * 24.0                     # days -> hours
    t = t - t[0]
    Y = d[wells].apply(pd.to_numeric, errors="coerce").to_numpy(float); Y = Y - Y[0]
    hi = set(hits)
    other = [i for i, w in enumerate(wells) if w not in hi]
    ax.fill_between(t, np.nanpercentile(Y[:, other], 5, axis=1),
                    np.nanpercentile(Y[:, other], 95, axis=1),
                    color=GRAY, alpha=0.55, lw=0)
    ends = []
    for w, (lab, dname) in hits.items():
        if w in wells:
            c = famcol(dname)[1]
            y = Y[:, wells.index(w)]
            ax.plot(t, y, color=c, lw=1.1); ends.append((y[-1], lab, c))
    ax.set_xlabel("Time (h)", fontsize=LABFS, labelpad=2)
    ax.set_ylabel("$\\Delta$A$_{405}$", fontsize=LABFS, labelpad=2)
    ax.set_xlim(0, t.max() * 1.38); style(ax)
    ax.set_ylim(top=ax.get_ylim()[1]*1.20)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=4))
    place_labels(ax, ends, t[-1], fs=5.6)

def turnover_panel(ax, _unused=None):
    """ZAPP-1 product/[E]0 vs time: shows it turns over many times, unlike the knockouts."""
    sp = spec(9); E = sp["E"]                                  # ZAPP-1, [E]0 = 24.0 uM
    piv = traces(sp); t = piv.index.to_numpy(float)
    top = 0
    for c, r in zip(CONC, list("ABCDEF")):                     # full 30 min run
        y = np.column_stack([piv[f"{r}{k}"].to_numpy(float) for k in sp["cols"]])/SPU_BIC
        # same robust baseline as the progress curves; product counted from mixing
        i0 = int(np.searchsorted(t, sp["tr"][0]))
        y = (y - np.nanmin(y[:i0+1], axis=0))/E                # -> turnovers
        mu = np.nanmean(y, 1); sd = np.nanstd(y, 1, ddof=1)/np.sqrt(y.shape[1])
        tt = t/60.0
        ax.fill_between(tt, mu-sd, mu+sd, color=RAMP[c], alpha=0.30, lw=0)
        ax.plot(tt, mu, color=RAMP[c], lw=1.0, solid_capstyle="round")
        top = max(top, np.nanmax(mu))
    ax.axhline(1.0, color=INK, lw=0.8, ls=(0,(3,2)), zorder=4)
    ax.annotate("1 turnover", xy=(0.30, 1.0), xytext=(0.30, 1.0 + 0.045*top),
                fontsize=5.8, color=INK, va="bottom")
    ax.text(0.058, 0.930, "[E]$_0$ = %s µM" % plain(sp["E"], sp["E"]*E_REL),
            transform=ax.transAxes, va="top", ha="left", fontsize=5.6,
            color=INK, zorder=7,
            bbox=dict(boxstyle="square,pad=0.30", fc="white", ec=INK2, lw=0.4))
    ax.set_xlabel("Time (min)", fontsize=LABFS, labelpad=2)
    ax.set_ylabel("Turnovers  ([P] / [E]$_0$)", fontsize=LABFS, labelpad=2)
    ax.set_xlim(0, None); ax.set_ylim(0, top*1.18); style(ax)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=4))
    ax.xaxis.set_major_locator(MultipleLocator(5))      # every 5 min
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))

# ── figure builders ─────────────────────────────────────────────────────────
def build(designs, outname, screen=None, screen_hits=None, spu_map=None,
          ncol_pairs=2, screen_title='Eluate screen', screen_sub=None,
          figw=7.0, wspace=0.72, right=0.840, legx=0.855):
    """figw / wspace / right / legx are exposed so a variant layout can widen the
    panels without touching the approved defaults."""
    n = len(designs)
    rows = int(np.ceil((n + (1 if screen else 0)) / ncol_pairs))
    FH = 2.02*rows + 0.34
    fig = plt.figure(figsize=(figw, FH))
    yo = lambda inch: inch / FH          # inches -> figure fraction
    gs = fig.add_gridspec(rows, 2*ncol_pairs, hspace=1.10, wspace=wspace,
                          left=0.082, right=right, top=0.905, bottom=0.072)
    letters = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    slot = 0
    if screen:
        ax = fig.add_subplot(gs[0, 0:2]); screen(ax, screen_hits)
        fig.canvas.draw(); pp = ax.get_position()
        fig.text((pp.x0+pp.x1)/2, pp.y1 + yo(0.325), screen_title,
                 ha="center", va="bottom", fontsize=7.8, color=INK, fontweight="bold")
        if screen_sub:
            fig.text((pp.x0+pp.x1)/2, pp.y1 + yo(0.05), screen_sub, ha="center",
                     va="bottom", fontsize=6.1, color=INK, linespacing=0.94)
        fig.text(pp.x0 - 0.052, pp.y1 + yo(0.325), next(letters), ha="left", va="bottom",
                 fontsize=12, color=INK)
        slot = 1
    for (name, title, sub, cell, cols) in designs:
        r, c = divmod(slot, ncol_pairs); slot += 1
        sp = spec(cell, cols); spu = spu_map(name)
        a1 = fig.add_subplot(gs[r, 2*c]); a2 = fig.add_subplot(gs[r, 2*c+1])
        draw_progress(a1, sp, spu); lines = draw_mm(a2, name, sp, spu)
        fig.canvas.draw()
        p1 = a1.get_position(); p2 = a2.get_position()
        xc = (p1.x0 + p2.x1)/2; ytop = p1.y1
        fig.text(xc, ytop + yo(0.325), f"{title}  ({sub})", ha="center", va="bottom",
                 fontsize=7.8, color=famcol(name)[1], fontweight="bold")
        fig.text(xc, ytop + yo(0.05), "\n".join(lines), ha="center", va="bottom",
                 fontsize=6.1, color=INK, linespacing=0.94)
        fig.text(p1.x0 - 0.052, ytop + yo(0.325), next(letters), ha="left", va="bottom",
                 fontsize=12, color=INK)
    substrate_legend(fig, (legx, 0.93))
    for ext, dpi in [("pdf",None),("png",150)]:
        d = os.path.join(FIGDIR, ext); os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, f"{outname}.{ext}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {outname}  ({rows} rows, {n} designs)")

SPU_NOBIC, SPU_BIC = 0.00633035, 0.00667808
spu_for = lambda n: SPU_NOBIC if n == "ZAPP-1 (no NaHCO3)" else SPU_BIC

def build_all():
    print("building SI kinetics figures ...")
    build(ROUND1, "PTE__v2SI__kinetics_round1", screen=screen_round1,
          screen_hits={"D1":("ZAPP-1","ZAPP-1"),"D7":("R1 p1D7","R1 p1D7"),
                       "D8":("R1 p1D8","R1 p1D8"),"E10":("R1 p1E10","R1 p1E10")},
          spu_map=spu_for, screen_title="Round 1 eluate screen  ($\\mathbfit{n}$ = 96 designs)",
          screen_sub="300 µM paraoxon · 1% MeOH · 200 µM ZnSO$_4$ · 25 °C, 1 h\n"
                     "gray = all other uncharacterized designs")
    build(MUTANTS, "PTE__v2SI__ZAPP1_knockouts", screen=turnover_panel, screen_hits=None,
          spu_map=spu_for, screen_title="ZAPP-1 multiple turnover",
          screen_sub="ZAPP-1, 30 min acquisition\n"
                     "~15 turnovers at 9.6 mM; >1 turnover at "
                     "[S]$_0$ $\\geq$ 600 µM")
    build(ROUND2, "PTE__v2SI__kinetics_round2", screen=screen_round2,
          screen_hits={"F7":("ZAPP-2","ZAPP-2"),"H1":("ZAPP-3","ZAPP-3"),
                       "G5":("ZAPP-4","ZAPP-4"),"C12":("ZAPP-5","ZAPP-5"),
                       "E24":("R2 p2B12","R2 p2B12"),"B18":("R2 p1D9","R2 p1D9"),
                       "D8":("R2 p1H4","R2 p1H4")}, spu_map=spu_for,
          screen_title="Round 2 eluate screen  ($\\mathbfit{n}$ = 192 designs)",
          screen_sub="600 µM paraoxon · 1% MeOH · 25 mM NaHCO$_3$ (final) · 25 °C, 1 h\n"
                     "gray = all other uncharacterized designs")
    print("done -> paper_figures/{pdf,png}/PTE__v2SI__*")
