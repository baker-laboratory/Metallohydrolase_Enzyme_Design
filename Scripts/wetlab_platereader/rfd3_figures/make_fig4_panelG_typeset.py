# Adapted from FOR_RFdiffusion3_paper/scripts/make_fig4_panelG_typeset.py (2026-09-29).
# Paper geometry, colors and labels are retained; paths resolve inside this repository.
"""Figure 4 panel G, redrawn: type matched to panels E and F, colors corrected.

WHY THIS EXISTS
Panel G is the round 2 eluate screen, drawn by section 7 of
paraoxon_kinetics_FOR_PAPER.ipynb (cell 70) at 8 pt axis titles and 7 pt ticks
and legend, then placed in the composite at about 77% -- its axes box is 43.1 mm
tall as drawn and 33.0 mm on the page -- so its type lands near 5-6 pt against
7.5 pt in panels E and F. It cannot be restyled in place either: its tick numbers
and legend are OUTLINED PATHS, not type ("Scaffold" appears zero times in the SVG
source). So it is redrawn here from the same data.

WHAT CHANGED BESIDES THE TYPE

1. THE SCAFFOLD COLORS WERE WRONG. Cell 68 hardcodes

       SCAFFOLD_TO_ZAPP = {1: 2, 3: 3, 6: 4, 7: 5}

   but the screen data names the designs directly, through the plate and well
   each highlighted trace comes from:

       scaffold 3   p2C4 (ZAPP-2), p2B12         -> the ZAPP-2 family
       scaffold 1   p1F6 (ZAPP-5), p1D9, p1H4    -> the ZAPP-5 family
       scaffold 7   p2G1 (ZAPP-3)
       scaffold 6   p2E3 (ZAPP-4)

   That is exactly the family assignment in Table S1 (make_SI_tables.py, dict D),
   and it is self-consistent: the two scaffold-3 designs are the two ZAPP-2
   family members, the three scaffold-1 designs are the three ZAPP-5 family
   members. The hardcoded map swaps ZAPP-2 with ZAPP-5 and misassigns ZAPP-3, so
   in the printed panel the ZAPP-2 curves carry ZAPP-3's color and so on. Here
   the map is DERIVED from the wells and checked against the table, and the
   script stops rather than guessing if the two ever disagree.

2. THE DESIGN CALLOUTS ARE DRAWN, not added by hand afterwards, so they cannot
   drift from the curves they name. The hand-added ones in the composite carry
   two typos: ZAPP-3 is p2G1, not p1G1, and ZAPP-4 is p2E3, not p1E3.

3. Non-bold axis titles, no top or right spine, ticks pointing out -- as panels
   E and F have them, and as panel G had before cell 70 began setting its titles
   bold.

Data, traces, highlighted designs and axis ranges are untouched. The caller
passes the namespace from the deposited analysis notebook after screening.

    python scripts/make_fig4_panelG_typeset.py
"""
import contextlib
import io
import json
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[3] / "Manuscript_Data" / "Phosphotriesterase_RFdiffusion3_Science_2026")
OUT = os.path.join(ROOT, "wetlab_data_plots", "paper_figures", "main_fig4")
STEM = "PTE__v2MAIN__fig4G__screen_progress"

N_HIGHLIGHT = 7                       # as cell 70; drops the least active ZAPP-2

# Type, on the page. Axis titles and ticks match panels E and F; the legend and
# the callouts stay smaller, being annotation rather than axis type.
PT, PT_ANNOT = 7.5, 6.0

# The printed panel's axes box is 49.4 x 33.0 mm with its right edge at
# x = 206.67 mm, which leaves 3.3 mm to the page edge -- the hand-added callouts
# were laid over the plot itself. Drawn callouts need their own column, so the
# plot is narrowed to 37.5 mm and the freed space carries the leaders and the
# labels: about 2 mm of leader zone and 9.8 mm of text, ending at 208.2 mm on a
# 210 mm page.
AXES_MM = (37.5, 33.0)
AXES_XY_MM = (158.73, 50.00)          # top-left corner of the axes on the page
                                      # (1.5 mm right and 1.2 mm up from where the
                                      #  printed panel had it, to clear panel F's
                                      #  axis label and panel I's headers)

# Family colors, from Jasper's enzyme_experiments.ipynb (DESIGN_COLOR) -- the
# same palette panels F and I use.
FAMILY_COLOR = {"ZAPP-2": "#4B5FAA", "ZAPP-3": "#6686C5",
                "ZAPP-4": "#F0A4B3", "ZAPP-5": "#FFC6B2"}

# Plate well -> (display name, family), from Table S1.
WELLS = {"p2C4": ("ZAPP-2", "ZAPP-2"), "p2B12": ("R2 p2B12", "ZAPP-2"),
         "p2G1": ("ZAPP-3", "ZAPP-3"), "p2E3": ("ZAPP-4", "ZAPP-4"),
         "p1F6": ("ZAPP-5", "ZAPP-5"), "p1D9": ("R2 p1D9", "ZAPP-5"),
         "p1H4": ("R2 p1H4", "ZAPP-5")}



def highlights(ns):
    """The seven highlighted traces, named from their own plate wells."""
    rates = ns["rates_df"].copy()
    trace = ns["baseline_subtracted_trace"]
    rates["peak_delta_abs"] = [float(trace(w).max()) for w in rates["reader_well"]]
    top = rates.nlargest(N_HIGHLIGHT, "peak_delta_abs").copy()
    top["well"] = ["p%d%s" % (int(r.plate_number), r.well_position)
                   for r in top.itertuples()]
    unknown = sorted(set(top["well"]) - set(WELLS))
    if unknown:
        raise SystemExit("highlighted wells absent from Table S1: %s" % unknown)
    top["label"] = [WELLS[w][0] for w in top["well"]]
    top["family"] = [WELLS[w][1] for w in top["well"]]

    # one scaffold is one design series, so every design on it shares a family
    for scaffold, grp in top.groupby("scaffold"):
        fams = sorted(set(grp["family"]))
        if len(fams) > 1:
            raise SystemExit("scaffold %s spans families %s" % (scaffold, fams))
    derived = {int(s): sorted(set(g["family"]))[0] for s, g in top.groupby("scaffold")}
    hard = {int(s): "ZAPP-%d" % z for s, z in ns["SCAFFOLD_TO_ZAPP"].items()}
    print("scaffold -> family, derived from the plate wells: %s" % derived)
    if hard != derived:
        print("  notebook cell 68 hardcodes %s -- corrected here" % hard)
    return top.sort_values("peak_delta_abs", ascending=False)


def build(ns):
    top = highlights(ns)
    trace = ns["baseline_subtracted_trace"]
    t = ns["time_min"]

    fam = next((f for f in ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
                if f in {x.name for x in font_manager.fontManager.ttflist}), "sans-serif")
    plt.rcParams.update({
        "font.family": fam, "font.size": PT,
        "axes.labelsize": PT, "xtick.labelsize": PT, "ytick.labelsize": PT,
        "axes.linewidth": 0.8, "axes.edgecolor": "black", "axes.labelcolor": "black",
        "text.color": "black", "xtick.color": "black", "ytick.color": "black",
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.top": False, "ytick.right": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.minor.width": 0.7, "ytick.minor.width": 0.7,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        # the replayed cells leave savefig.bbox on "tight"; passing
        # bbox_inches=None to savefig does NOT override that (print_figure falls
        # back to the rcParam when the argument is None), and a trimmed canvas
        # would put the axes at an unknown offset, breaking the placement below
        "savefig.bbox": None, "savefig.pad_inches": 0.0,
        "mathtext.fontset": "custom", "mathtext.rm": fam,
        "mathtext.it": f"{fam}:italic", "mathtext.bf": f"{fam}:bold",
    })

    # Margins around the axes, in mm. The top one clears the first callout,
    # which is centred on a curve ending at the top of the plot; the right one
    # holds the leader zone plus the widest callout; the bottom one is kept
    # tight because panel I's "Round 1 / Round 2" rules sit at y = 93.5 mm and a
    # deeper margin drops "Time (min)" straight onto them. It still has to clear
    # the canvas edge by a millimetre: the panel is inlined into the composite
    # as a nested <svg>, which clips anything past its viewport.
    ml, mr, mb, mt = 11.0, 13.0, 9.0, 4.3
    W, H = AXES_MM[0] + ml + mr, AXES_MM[1] + mb + mt
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    ax = fig.add_axes([ml / W, mb / H, AXES_MM[0] / W, AXES_MM[1] / H])

    hot = set(top["reader_well"])
    for w in ns["reader_well_cols"]:
        if w not in hot:
            ax.plot(t, trace(w), color=ns["BACKGROUND_TRACE_FLAT"], lw=0.35,
                    solid_capstyle="round", zorder=1)
    for r in top.sort_values("peak_delta_abs").itertuples():
        ax.plot(t, trace(r.reader_well), color=FAMILY_COLOR[r.family], lw=1.4,
                solid_capstyle="round", zorder=3)

    ax.set_xlim(0, t[-1]); ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xlabel("Time (min)", fontsize=PT, labelpad=2)
    ax.set_ylabel("$\\Delta A_\\mathrm{405nm}$", fontsize=PT, labelpad=2)

    ax.legend(handles=[Line2D([], [], color=FAMILY_COLOR[f], lw=1.4,
                              label="%s Scaffold" % f)
                       for f in sorted(set(top["family"]))],
              loc="upper left", fontsize=PT_ANNOT, handlelength=1.5,
              labelspacing=0.28, borderpad=0.22, handletextpad=0.5,
              borderaxespad=0.35, frameon=False)

    # Callouts, placed the way the SI screen panels do it (place_labels in
    # scripts/make_kinetics_SI_figures.py): a displaced label gets an ELBOW
    # leader -- short horizontal off the trace, a vertical riser in its own
    # staggered column, then a short horizontal into the text. A straight
    # diagonal, which is what this panel had, cuts across whatever label happens
    # to sit between the two; that is what put the ZAPP-3 leader through
    # "(R2 p2E3)".
    #
    # Spacing is per label rather than one constant, because a two-line callout
    # is twice the height of "R2 p1H4" and the four least active designs finish
    # within 0.15 absorbance units of each other.
    ends = sorted(((r.label, r.family, r.well, float(trace(r.reader_well)[-1]))
                   for r in top.itertuples()), key=lambda e: -e[3])
    texts = [(lab if lab.startswith("R2") else "%s\n(R2 %s)" % (lab, well))
             for lab, _f, well, _y in ends]
    lo, hi = ax.get_ylim()
    mm_per_unit = AXES_MM[1] / (hi - lo)
    heights = [(2 if "\n" in s else 1) * 1.2 * PT_ANNOT / 72 * 25.4 / mm_per_unit
               for s in texts]
    pad = 0.16 * heights[0]

    ys, prev, prev_h = [], None, 0.0
    for (_lab, _f, _w, y_end), h in zip(ends, heights):
        y = y_end if prev is None else min(y_end, prev - (h + prev_h) / 2 - pad)
        ys.append(y); prev, prev_h = y, h
    drop = (lo + heights[-1] / 2) - ys[-1]
    if drop > 0:                                   # lift the stack back inside
        ys = [y + drop for y in ys]

    x_end = t[-1]
    dx = 0.055 * (x_end - ax.get_xlim()[0])        # width of the leader zone
    placed = []
    for i, ((lab, famly, well, y_end), text, y) in enumerate(zip(ends, texts, ys)):
        col = FAMILY_COLOR[famly]
        if abs(y - y_end) > 1e-9:                  # risers staggered, never overlapping
            xk = x_end + dx * (0.22 + 0.16 * (i % 3))
            ax.plot([x_end, xk, xk, x_end + dx * 0.93], [y_end, y_end, y, y],
                    color=col, lw=0.5, ls=(0, (2.2, 1.4)), clip_on=False,
                    zorder=6, dash_capstyle="butt", solid_joinstyle="miter")
        # multialignment centres the two lines on each other -- the design name
        # sits over its well -- while ha="left" keeps every callout starting in
        # the same column
        ax.text(x_end + dx, y, text, fontsize=PT_ANNOT, color=col, va="center",
                ha="left", multialignment="center", clip_on=False, zorder=7,
                linespacing=1.2)
        placed.append((lab, well, famly, y_end))

    os.makedirs(OUT, exist_ok=True)
    for fmt in ("pdf", "png"):
        p = os.path.join(OUT, "%s.%s" % (STEM, fmt))
        # bbox_inches=None, NOT the "tight" the replayed notebook cells set in
        # rcParams: the canvas has to stay exactly the size computed above, or
        # the axes no longer sit at a known offset and the placement below lies
        fig.savefig(p, dpi=150, facecolor="white", bbox_inches=None)
        print("wrote " + os.path.relpath(p, ROOT))

    aw, ah = ax.get_window_extent().transformed(
        fig.dpi_scale_trans.inverted()).size * 25.4
    print("axes box %.1f x %.1f mm (target %.1f x %.1f)" % (aw, ah, *AXES_MM))
    print("type: axis titles and ticks %.1f pt, legend and callouts %.1f pt"
          % (PT, PT_ANNOT))
    print("canvas %.2f x %.2f mm; axes inset %.2f mm from the left, %.2f from the top"
          % (W, H, ml, mt))
    print("place at X = %.2f mm, Y = %.2f mm (Inkscape, 100%%)"
          % (AXES_XY_MM[0] - ml, AXES_XY_MM[1] - mt))
    plt.close(fig)
    print("\nhighlighted traces, most active first:")
    for label, well, family, y in placed:
        print("   %-9s %-6s %-7s ends at %.2f" % (label, well, family, y))
