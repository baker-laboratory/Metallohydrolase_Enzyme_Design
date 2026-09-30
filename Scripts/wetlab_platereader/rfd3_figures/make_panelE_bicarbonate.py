"""Render Figure 4E with the reference layout and optional condition-matched points.

The corrected variant uses the no-bicarbonate calibration for that series;
the default preserves the original figure for comparison."""
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
    """Return formatted parameter annotations from the reference workbook."""
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
        ax.plot(Sf, 1e3*r["kcat (s-1)"]*Sf/(r["Km (uM)"]/1000.0 + Sf),
                color=curve, lw=1.1, zorder=2, solid_capstyle="round")
        # White ring under each marker, so a point on the curve separates from it.
        ax.plot(S/1000.0, v*1e3, "o", ms=5.0, mfc="white", mec="white", mew=0,
                ls="none", zorder=3)
        ax.plot(S/1000.0, v*1e3, "o", ms=3.4, mfc=fill, mec=curve, mew=0.6,
                ls="none", zorder=4)
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
