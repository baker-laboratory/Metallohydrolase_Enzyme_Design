# Adapted from FOR_RFdiffusion3_paper/scripts/make_panelE_bicarbonate.py (2026-09-29).
# Paper geometry, colors and labels are retained; paths resolve inside this repository.
"""Figure 4 panel E (ZAPP-1 +/- bicarbonate) regenerated with live text.

WHY THIS EXISTS
The panel E currently in enzyme_experiments.svg has its parameter annotations as
OUTLINED TEXT -- "Bicarbonate" and "uncat" appear zero times in the 12 MB of SVG
source, and none of the 8 embedded rasters is this panel. So the numbers cannot
be corrected by editing either the SVG or the PDF. The nearest live-text source,
enzyme_experiments_full.svg (2026-08-07), is an older three-line revision with no
k_cat/k_uncat row, and the generating cell (enzyme_experiments.ipynb cell 7) never
drew the annotations at all and is still on a 0-5 mM axis.

This script rebuilds the panel from the same raw plate-reader files, with the
annotations as real text objects, so the next correction is an edit rather than a
retype.

WHAT CHANGED, AND WHY
The printed figure quoted the FIT-ONLY standard error at 1-2 significant figures.
The master xlsx, Table S1, and the SI Methods sentence all use the TOTAL error:
regression SE combined in quadrature with 6% on [E]_o and the calibration, then
rounded UP to one significant figure with the value truncated at that digit. Six
of the eight numbers differed. Every value below is computed here from the xlsx
raw columns using that rule, so the panel and Table S1 cannot drift apart again.

    quantity              was (figure)        now (xlsx / Table S1)
    +bic K_M              7.8 +/- 1.8 mM      8 +/- 2 mM
    +bic k_cat            0.015 +/- 0.002     0.015 +/- 0.003
    +bic k_cat/K_M        2.0 +/- 0.5         2.0 +/- 0.6
    +bic k_cat/k_uncat    (3.1 +/- 0.9)e5     (3 +/- 1)e5
    -bic K_M              2.6 +/- 0.7 mM      2.6 +/- 0.7 mM   (unchanged)
    -bic k_cat            0.0013 +/- 0.0001   0.0013 +/- 0.0002
    -bic k_cat/K_M        0.52 +/- 0.14       0.5 +/- 0.2
    -bic k_cat/k_uncat    (2.5 +/- 0.3)e4     (2.5 +/- 0.4)e4

Colors are sampled from the printed figure: markers and the "+ Bicarbonate"
heading #4fb9af, the minus-condition markers #ffe0ac, its heading #d9962b.
"""
import os, sys, importlib.util
from math import floor, log10, ceil
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[3] / "Manuscript_Data" / "Phosphotriesterase_RFdiffusion3_Science_2026")
OUT  = os.path.join(ROOT, "wetlab_data_plots", "paper_figures", "main_fig4")
STEM = "PTE__v2MAIN__fig4E__bicarbonate_MM"

from . import make_kinetics_SI_figures as K

TEAL, TAN   = "#4fb9af", "#ffe0ac"       # marker fills, sampled from the PDF
HDR_P, HDR_M = "#4fb9af", "#d9962b"      # the two heading colors
INK = "#0b0b0b"
SPU = 0.00667808                         # signal per uM, as in the source notebook

CURVE_P, CURVE_M = "#2e8b82", "#c8851c"   # dark variants for the fit lines
CONDS = [("ZAPP-1",             9, "+ Bicarbonate", TEAL, CURVE_P),
         ("ZAPP-1 (no NaHCO3)", 5, "- Bicarbonate", TAN,  CURVE_M)]


def _r1(e):
    """Uncertainty to 1 s.f., rounded UP; a leading 9 promotes a decade."""
    d = floor(log10(abs(e))); lead = ceil(abs(e)/10**d - 1e-9)
    if lead >= 9: lead, d = 1, d + 1
    return lead*10.0**d, -d


def pm(v, e):
    """'value +/- error', value truncated at the error's last digit."""
    er, dec = _r1(e)
    if dec > 0: return f"{round(v,dec):.{dec}f} ± {er:.{dec}f}"
    return f"{round(v,dec):.0f} ± {er:.0f}"


def sci(v, e):
    """'(a +/- b) x 10^n' with the mantissa carrying the 1 s.f. error rule."""
    n = floor(log10(abs(v)))
    return f"({pm(v/10**n, e/10**n)}) $\\times$ 10$^{{{n}}}$"


def param_lines(name):
    """The four annotation rows, computed from the xlsx raw columns.

    Symbols are set FULLY ITALIC -- k, cat, K, M, uncat -- to match the rest of
    the manuscript. Note this is not the IUPAC/IUBMB convention, which italicizes
    only the quantity symbol and sets a descriptive subscript roman
    (italic k, roman "cat"; italic K, roman "M" for Michaelis). Internal
    consistency with the other figures and the main text wins here. To switch to
    the standard later, wrap each subscript in \\mathrm{} in this one function and
    in scripts/make_SI_tables.py; nothing else needs touching.

    Order is k_cat, then K_M, then the two ratios.
    """
    r = K.TAB.loc[name]
    return [
        f"$k_\\mathrm{{cat}}$ = {pm(r['kcat (s-1)'], r['kcat sd (total)'])} s$^{{-1}}$",
        f"$K_\\mathrm{{M}}$ = {pm(r['Km (uM)']/1000.0, r['Km sd']/1000.0)} mM",
        f"$k_\\mathrm{{cat}}/K_\\mathrm{{M}}$ = "
        f"{pm(r['kcat/Km (M-1 s-1)'], r['kcat/Km sd (total)'])} M$^{{-1}}$ s$^{{-1}}$",
        f"$k_\\mathrm{{cat}}/k_\\mathrm{{uncat}}$ = {sci(r['kcat/kuncat'], r['kcat/kuncat sd'])}",
    ]


def build(figw=3.05, figh=2.05, base=5.6, axfs=None, dropin=None, correct_calibration=False):
    stem = STEM + ("__condition_matched" if correct_calibration else "")
    # axfs is the tick/axis-label size. Panel F of the same figure measures
    # 7.5 pt Arial (45 px cap height at 600 dpi), so panel E matches it.
    if axfs is None: axfs = base
    fam = next((f for f in ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
                if f in {x.name for x in font_manager.fontManager.ttflist}), "sans-serif")
    plt.rcParams.update({
        "font.family": fam, "font.size": base,
        "axes.labelsize": axfs, "xtick.labelsize": axfs, "ytick.labelsize": axfs,
        "axes.linewidth": 0.8, "axes.edgecolor": INK, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.minor.width": 0.7, "ytick.minor.width": 0.7,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "mathtext.fontset": "custom", "mathtext.rm": fam,
        "mathtext.it": f"{fam}:italic", "mathtext.bf": f"{fam}:bold",
    })
    fig, ax = plt.subplots(figsize=(figw, figh))

    ymax = 0.0
    for name, cell, _, fill, curve in CONDS:
        sp = K.spec(cell)
        spu = K.spu_for(name) if correct_calibration else SPU
        S, v, e = K.mm_points(sp, spu)
        r = K.TAB.loc[name]
        Sf = np.linspace(0, 10.5, 400)
        # The fit curve carries the series color rather than black. With both
        # curves black, a black error bar on a black-edged marker sitting on a
        # black line had nothing to read against. Now black is used for one thing
        # only -- the error bar -- so it stays visible even at this size.
        ax.plot(Sf, 1e3*r["kcat (s-1)"]*Sf/(r["Km (uM)"]/1000.0 + Sf),
                color=curve, lw=1.1, zorder=2, solid_capstyle="round")
        # White ring under each marker, so a point on the curve separates from it.
        ax.plot(S/1000.0, v*1e3, "o", ms=5.0, mfc="white", mec="white", mew=0,
                ls="none", zorder=3)
        ax.plot(S/1000.0, v*1e3, "o", ms=3.4, mfc=fill, mec=curve, mew=0.6,
                ls="none", zorder=4)
        # Error bars drawn LAST, so nothing occludes them. Thin and uncapped: a
        # cap is what made this look like a strikethrough before, because the cap
        # was far wider than the bar was tall.
        #
        # Be aware of the geometry. On the printed panel the y-axis spans 0-10
        # over 94.9 pt, so the SEM of n = 3 gives a FULL bar of 0.03 to 1.59 pt
        # against a 3.4 pt marker -- at most 47% of the symbol, and for 8 of the
        # 12 points thinner than the 0.5 pt line used to draw it. Those eight
        # therefore render as a horizontal dash rather than a vertical bar, and no
        # choice of linewidth or marker size fixes that: the quantity really is
        # that small. The caption still needs to say so:
        #   "Error bars are s.e.m. of n = 3 technical replicates; where not
        #    visible they are smaller than the symbols."
        ax.errorbar(S/1000.0, v*1e3, yerr=e*1e3, fmt="none", ecolor="black",
                    elinewidth=0.6, capsize=1.6, capthick=0.6, zorder=6)
        ymax = max(ymax, float((v*1e3 + e*1e3).max()))

    ax.set_xlim(0, 10.4)
    # Top out at a round 10 rather than hugging the data. The extra headroom
    # drops the upper curve clear of the '+ Bicarbonate' block's superscripts,
    # which the curve used to run straight through.
    ax.set_ylim(0, 10.0)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="both", top=False, right=False, labelsize=axfs)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_xlabel("[Paraoxon] (mM)", fontsize=axfs, labelpad=2)
    ax.set_ylabel("$v$/[E] $\\times$10$^{-3}$ (s$^{-1}\\!$)", fontsize=axfs, labelpad=2)

    # Annotation blocks, placed as in the printed panel: plus-condition upper
    # left, minus-condition mid right. Each row is its own text object, so a
    # future correction is a click and a retype rather than a redraw.
    # Annotation blocks. Each row is placed individually at va="baseline" on a
    # fixed pitch, NOT stacked at va="top" and NOT joined into one multi-line
    # string. Both of those give uneven gaps here, because matplotlib measures a
    # mathtext row by its own extent and a row carrying a superscript
    # (s$^{-1}$, $\times$10$^{5}$) is taller than one without. That is what
    # opened the visible gap between the k_cat and K_M rows. Baselines on a
    # constant pitch are uniform whatever each row contains.
    LEAD = 0.076
    for (name, _, hdr, _, hcol), (x, y) in zip(CONDS, [(0.025, 1.00), (0.470, 0.495)]):
        ax.text(x, y, hdr, transform=ax.transAxes, ha="left", va="top",
                fontsize=base, fontweight="bold", color=hcol)
        for i_, line in enumerate(param_lines(name)):
            ax.text(x, y - 0.105 - i_*LEAD, line, transform=ax.transAxes,
                    ha="left", va="baseline", fontsize=base, color=INK)

    os.makedirs(OUT, exist_ok=True)
    if dropin:
        px_w, px_h, dpi = dropin
        # margins copied from the embedded raster so the axes land in register
        # Axes pinned to the band panels E, F and G now share: top 50.0 mm,
        # bottom 83.0 mm on the page, i.e. 33.0 mm of plot in all three. The
        # raster sits at y = 48.90073 mm and is 43.235897 mm tall, so those two
        # page positions are these fractions of it.
        fig.subplots_adjust(left=0.215, bottom=0.211319, right=0.985, top=0.974576)
        p = os.path.join(OUT, f"{STEM}__dropin.png")
        fig.savefig(p, dpi=dpi, facecolor="white")      # no tight bbox: fixed canvas
        from PIL import Image as _I
        im = _I.open(p)
        if im.size != (px_w, px_h):
            im.resize((px_w, px_h), _I.LANCZOS).save(p)
        print(f"wrote {os.path.relpath(p, ROOT)}  ({_I.open(p).size[0]}x{_I.open(p).size[1]} px)")
        plt.close(fig); return p
    for fmt in ("pdf", "png"):
        p = os.path.join(OUT, f"{stem}.{fmt}")
        fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
        print("wrote " + os.path.relpath(p, ROOT))
    plt.close(fig)

    print()
    for name, _, hdr, _, _ in CONDS:
        print(f"  {hdr}")
        for line in param_lines(name):
            print("    " + line.replace("$", "").replace("\\times", "x")
                              .replace("^{-1}", "^-1").replace("_", ""))


def build_dropin():
    """Pixel-exact replacement for the raster embedded in enzyme_experiments.svg.

    That panel is <image id="image1-36">, a 1396x1077 PNG placed at
    x=37.204056 y=48.90073 w=56.042076 h=43.235897 (mm). Rendering to the same
    1396x1077 canvas with the same axes fractions -- left spine at 0.145 of the
    width, bottom spine at 0.798 of the height, both measured off the embedded
    image -- means the swap needs no rescaling and nothing else in the figure
    moves. preserveAspectRatio="none" on that element would stretch a mismatched
    canvas, which is why the aspect is matched exactly (1.2962).
    """
    global _DROPIN
    _DROPIN = True
    px_w, px_h = 1396, 1077
    in_w = 56.042076/25.4                      # placed size on the page
    dpi  = px_w/in_w                           # 633 dpi, so text lands at print size
    fig = build(figw=in_w, figh=px_h/dpi, base=4.9, axfs=7.5,
                dropin=(px_w, px_h, dpi))
    return fig


if __name__ == "__main__":
    import sys
    build_dropin() if "--dropin" in sys.argv else build()
