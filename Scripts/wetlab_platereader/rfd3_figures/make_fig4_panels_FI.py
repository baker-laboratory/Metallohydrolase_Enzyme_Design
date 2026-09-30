# Adapted from FOR_RFdiffusion3_paper/scripts/make_fig4_panels_FI.py (2026-09-29).
# Paper geometry, colors and labels are retained; paths resolve inside this repository.
"""Figure 4 panels F and I, rebuilt with the standard symbol convention.

WHY THIS EXISTS
Both panels set the whole symbol in italic -- "k_cat/K_M" comes out as italic k,
italic cat, italic K, italic M -- because they were drawn with matplotlib
mathtext as $k_{cat}/K_M$, which italicizes everything inside math mode. The
IUPAC/IUBMB convention, and what the SI tables and captions already use, is an
italic quantity symbol with a roman descriptive subscript: italic k with roman
"cat", italic K with roman "M". This rebuild changes that and nothing else.

    was   $k_{cat}/K_M$      k, cat, K, M all italic
    now   $k_\\mathrm{cat}/K_\\mathrm{M}$   only k and K italic

Panel E was already correct (see make_panelE_bicarbonate.py, param_lines).

WHAT IS REPRODUCED, AND HOW FAITHFULLY
Both panels are embedded rasters in enzyme_experiments.svg -- panel F is
image1-35 (1002x852 px placed at 52.166553 x 44.357189 mm) and panel I is
image1-270 (1152x1452 px at 48.598057 x 61.253792 mm) -- so, as with panel E,
the text is pixels and has to be replaced rather than retyped. Geometry, type
sizes, colors and line weights below were measured off those two rasters at
their placed scale, so a --dropin render lands in register and nothing else in
the figure moves:

    panel F   axes 0.2266-0.9721 x 0.2124-0.8967 of the canvas, 7.5 pt type,
              bar height 0.8, bar edge 0.4 pt, spine 0.75 pt, ticks 3/1.8 pt
    panel I   axes 0.2127-0.9757, top 0.6811-0.9807, bottom 0.1832-0.4828,
              5.7 pt type, bar width 0.75, same edge and spine weights

Values come from paraoxon_kinetics_ALL_DATA.xlsx, the same source as Table S1
and Table S2, so the panels cannot drift from the tables. Both panels already
carried the total error (fit SE combined with 6% on [E]_o and the calibration),
and every value reproduced here matches what is printed. H170A is the one
special case: its Michaelis-Menten fit is unconstrained (K_M 27x the highest
[S]), so the table and the printed panel both use the first-order chord
estimate, 0.002 +/- 0.001, and so does this rebuild.

    python scripts/make_fig4_panels_FI.py            # standalone pdf/svg/png
    python scripts/make_fig4_panels_FI.py --dropin   # + swap-in PNGs at raster size
    python scripts/make_fig4_panels_FI.py --check    # measure the dropins vs source
"""
import importlib.util
import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[3] / "Manuscript_Data" / "Phosphotriesterase_RFdiffusion3_Science_2026")
OUT = os.path.join(ROOT, "wetlab_data_plots", "paper_figures", "main_fig4")

from . import make_kinetics_SI_figures as K

INK = "#0b0b0b"
TEAL, NAVY, MBLUE, PINK, PEACH = "#4fb9b0", "#4b5faa", "#6686c5", "#f0a4b3", "#ffc6b2"

# panel F: the parent plus the six knockouts, in the printed top-to-bottom order
KO = [("WT", "ZAPP-1"), ("H93A", "ZAPP-1 MUT1"), ("H89A", "ZAPP-1 MUT2"),
      ("K16A", "ZAPP-1 MUT3"), ("H170A", "ZAPP-1 MUT4"),
      ("H133A", "ZAPP-1 MUT5"), ("E92A", "ZAPP-1 MUT6")]

# panel I: four round 1 designs, then seven round 2, colored by scaffold family
DES = [("ZAPP-1", TEAL), ("R1 p1D7", TEAL), ("R1 p1D8", TEAL), ("R1 p1E10", TEAL),
       ("ZAPP-2", NAVY), ("R2 p2B12", NAVY), ("ZAPP-3", MBLUE), ("ZAPP-4", PINK),
       ("ZAPP-5", PEACH), ("R2 p1D9", PEACH), ("R2 p1H4", PEACH)]
SPLIT = 3.5                      # the round 1 / round 2 divider sits here

# placed geometry of the two rasters, in px and mm, straight from the SVG
RASTER = {"F": dict(px=(1002, 852), mm=(52.166553, 44.357189), image="image1-35"),
          "I": dict(px=(1152, 1452), mm=(48.598057, 61.253792), image="image1-270")}

# Mathtext leaves a gap after a superscript group, which is what separated the
# units ("M^-1 s^-1 )") in the first rebuild. A negative thin space closes it,
# giving the tight "M^-1s^-1)" of the printed panels. Only k and K are italic;
# the descriptive subscripts are roman, which is the whole point of the rebuild.
KCAT_S = "$k_\\mathrm{cat}$ (s$^{-1}\\!$)"
KCAT_KM_M = "$k_\\mathrm{cat}/K_\\mathrm{M}$ (M$^{-1}\\!$s$^{-1}\\!$)"


def rc(fs, labfs=None, spine=0.75, tick=1.9):
    """House rcParams at a given type size.

    Axis labels are set SMALLER than tick labels because that is what the
    printed panels do: measured on the rasters, panel F is 7.5 pt ticks with a
    6.9 pt axis label (parenthesis 44 px against 48 px at 7.5 pt) and panel I is
    5.7 pt ticks with a 5.6 pt label (parenthesis 44 px at 602 dpi). Matching tick size alone would leave the
    labels visibly oversized next to the untouched panels.
    """
    fam = next((f for f in ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
                if f in {x.name for x in font_manager.fontManager.ttflist}), "sans-serif")
    plt.rcParams.update({
        "font.family": fam, "font.size": fs,
        "axes.labelsize": labfs or fs,
        "xtick.labelsize": fs, "ytick.labelsize": fs,
        "axes.linewidth": spine, "axes.edgecolor": "black", "axes.labelcolor": "black",
        "text.color": "black", "xtick.color": "black", "ytick.color": "black",
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.width": spine, "ytick.major.width": spine,
        "xtick.major.size": tick, "ytick.major.size": tick,
        "xtick.top": False, "ytick.right": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "mathtext.fontset": "custom", "mathtext.rm": fam,
        "mathtext.it": f"{fam}:italic", "mathtext.bf": f"{fam}:bold",
    })


def ke(name):
    """k_cat/K_M and its total uncertainty, in M^-1 s^-1.

    H170A never approaches saturation, so its Michaelis-Menten fit is
    unconstrained and its k_cat/K_M standard deviation (0.157) is meaningless.
    Table S2 reports the first-order chord estimate instead, and so does the
    printed panel; this follows both.
    """
    r = K.TAB.loc[name]
    if str(r["kcat (reported)"]).strip() == "n.d.":
        return float(r["kcat/Km chord (M-1 s-1)"]), float(r["kcat/Km chord sd"])
    return float(r["kcat/Km (M-1 s-1)"]), float(r["kcat/Km sd (total)"])


def save(fig, stem, dropin=None):
    os.makedirs(OUT, exist_ok=True)
    if dropin:
        px_w, px_h, dpi = dropin
        p = os.path.join(OUT, f"{stem}__dropin.png")
        fig.savefig(p, dpi=dpi, facecolor="white")     # fixed canvas, no tight bbox
        from PIL import Image
        im = Image.open(p)
        if im.size != (px_w, px_h):
            im.resize((px_w, px_h), Image.LANCZOS).save(p)
        print("wrote %s  (%dx%d px)" % (os.path.relpath(p, ROOT), *Image.open(p).size))
    else:
        for fmt in ("pdf", "png"):
            p = os.path.join(OUT, f"{stem}.{fmt}")
            fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
            print("wrote " + os.path.relpath(p, ROOT))
    plt.close(fig)


def build_F(dropin=False):
    stem = "PTE__v2MAIN__fig4F__knockouts_kcatKM"
    px, mm = RASTER["F"]["px"], RASTER["F"]["mm"]
    in_w, in_h = mm[0] / 25.4, mm[1] / 25.4
    dpi = px[0] / in_w
    # tick marks at panel E's length (3 pt major, 1.8 pt minor) rather than the
    # 1.9 pt the printed panel F used, so the two panels' axes read alike
    # 7.5 pt type, 0.8 pt spines and 3/1.8 pt ticks: the same as panels E and G
    rc(7.5, labfs=7.5, tick=3.0, spine=0.8)
    fig = plt.figure(figsize=(in_w, in_h))
    # Axes pinned to the band E, F and G share -- top 50.0 mm, bottom 83.0 mm
    # on the page, 33.0 mm of plot. The raster sits at y = 48.651474 mm and is
    # 44.357189 mm tall; the printed panel had only 30.35 mm of plot here.
    ax = fig.add_axes([0.2266, 0.225634, 0.9721 - 0.2266, 0.743963])

    labels = [n for n, _ in KO]
    vals = [ke(k) for _, k in KO]
    y = np.arange(len(KO))[::-1]                       # first row at the top
    ax.barh(y, [v for v, _ in vals], height=0.8, color=TEAL,
            edgecolor="black", linewidth=0.4, zorder=2)
    ax.errorbar([v for v, _ in vals], y, xerr=[e for _, e in vals], fmt="none",
                ecolor="black", elinewidth=0.7, capsize=1.8, capthick=0.7, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    # solved from the source raster, where bar 6 centres on y=147 px and bar 0
    # on y=613 within an axes running 88-671: 77.7 px per category
    ax.set_ylim(-0.745, 6.762)
    ax.set_xlim(-0.18, 2.654)
    # denser than the printed panel, which labelled only 0, 1 and 2: majors
    # every 0.5 with a single minor between, so a bar can be read off the axis
    # without the ruler effect four minors gave
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="minor", length=1.8, width=0.7)
    ax.set_xlabel(KCAT_KM_M)
    # placed where the raster puts it: centred on x=599.5 px, y=791 px of the
    # 1002x852 canvas, which is 0.499 across and 0.206 of the axes height below it
    ax.xaxis.label.set_va("center")
    # 6.7 mm below the bottom spine: a little lower than the raster's 6.25 mm,
    # a little higher than the 7.13 mm that read as too far down
    ax.xaxis.set_label_coords(0.499, -0.203)
    ax.tick_params(which="both", top=False, right=False)

    save(fig, stem, dropin=(px[0], px[1], dpi) if dropin else None)
    print("  panel F values (M^-1 s^-1):")
    for (n, k), (v, e) in zip(KO, vals):
        print("    %-6s %8.4f +/- %.4f%s" % (n, v, e,
              "   (first-order chord estimate)" if n in ("H89A", "H170A") else ""))


def build_I(dropin=False):
    stem = "PTE__v2MAIN__fig4I__kcat_kcatKM"
    px, mm = RASTER["I"]["px"], RASTER["I"]["mm"]
    in_w, in_h = mm[0] / 25.4, mm[1] / 25.4
    dpi = px[0] / in_w
    rc(5.7, labfs=5.6, spine=0.6, tick=1.9)
    fig = plt.figure(figsize=(in_w, in_h))
    axT = fig.add_axes([0.2127, 0.6811, 0.9757 - 0.2127, 0.9807 - 0.6811])
    axB = fig.add_axes([0.2127, 0.1832, 0.9757 - 0.2127, 0.4828 - 0.1832])

    x = np.arange(len(DES))
    names = [n for n, _ in DES]
    cols = [c for _, c in DES]
    kcat = [(float(K.TAB.loc[n, "kcat (s-1)"]),
             float(K.TAB.loc[n, "kcat sd (total)"])) for n in names]
    keff = [(float(K.TAB.loc[n, "kcat/Km (M-1 s-1)"]),
             float(K.TAB.loc[n, "kcat/Km sd (total)"])) for n in names]

    for ax, series, ylab, ylim, loc, ref in (
            (axT, kcat, KCAT_S, (0, 0.0295), 0.01, kcat[0][0]),
            (axB, keff, KCAT_KM_M, (0, 6.42), 2, keff[0][0])):
        ax.bar(x, [v for v, _ in series], width=0.75, color=cols,
               edgecolor="black", linewidth=0.4, zorder=2)
        ax.errorbar(x, [v for v, _ in series], yerr=[e for _, e in series],
                    fmt="none", ecolor="black", elinewidth=0.7, capsize=1.8,
                    capthick=0.7, zorder=3)
        # ZAPP-1, the round 1 benchmark every later design is read against
        ax.axhline(ref, color="#808080", lw=0.6, ls=(0, (4, 3)), zorder=1)
        ax.axvline(SPLIT, color="#808080", lw=0.6, ls=(0, (1, 2)), zorder=1)
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_locator(MultipleLocator(loc))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis="y", which="minor", length=1.8, width=0.7)
        ax.set_ylabel(ylab, labelpad=2)
        ax.set_xlim(-0.94, 10.97)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right",
                           rotation_mode="anchor")
        ax.tick_params(which="both", top=False, right=False)

    save(fig, stem, dropin=(px[0], px[1], dpi) if dropin else None)
    print("  panel I values:")
    for n, (a, ae), (b, be) in zip(names, kcat, keff):
        print("    %-9s kcat %.5f +/- %.5f   kcat/KM %.3f +/- %.3f" % (n, a, ae, b, be))


def check():
    """Measure the drop-in renders against the rasters they replace."""
    import numpy as np
    from PIL import Image

    def geom(path):
        A = np.asarray(Image.open(path).convert("RGB")).astype(int)
        H, W, _ = A.shape
        dark = A.sum(2) < 260
        vs = [x for x in range(W) if dark[:, x].sum() > 0.15 * H]
        hs = [y for y in range(H) if dark[y].sum() > 0.30 * W]
        def runs(idx, gap=3):
            out, cur = [], [idx[0], idx[0]]
            for i in idx[1:]:
                if i - cur[1] <= gap: cur[1] = i
                else: out.append(tuple(cur)); cur = [i, i]
            out.append(tuple(cur)); return out
        return (W, H), runs(vs)[:2], runs(hs)[:2]

    for tag, src, new in (
            ("F", "/tmp/src_panelF.png",
             os.path.join(OUT, "PTE__v2MAIN__fig4F__knockouts_kcatKM__dropin.png")),
            ("I", "/tmp/src_panelI.png",
             os.path.join(OUT, "PTE__v2MAIN__fig4I__kcat_kcatKM__dropin.png"))):
        if not (os.path.exists(src) and os.path.exists(new)):
            print("panel %s: missing %s" % (tag, src if not os.path.exists(src) else new))
            continue
        print("panel %s  source %s\n          rebuilt %s" % (tag, geom(src), geom(new)))


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        drop = "--dropin" in sys.argv
        build_F(dropin=drop)
        build_I(dropin=drop)
