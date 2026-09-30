"""Plot the additional tagless ZAPP-1 comparison and its detector-linearity check."""
import os, sys, importlib.util
import numpy as np, matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[3] / "Manuscript_Data" / "Phosphotriesterase_RFdiffusion3_Science_2026")
from . import make_kinetics_SI_figures as m
from kinetics import parse_kinetics

SIG_E, CAL = 0.06, 0.00135                     # same error budget as the master table
KU, KUSD   = 5.0e-8, 1.3e-8                    # kuncat, +25 mM NaHCO3
TAGLESS_CELL = 7

def fit(sp):
    _, _, mm = parse_kinetics(data_path=sp["path"],
        blocks=[dict(rows=list("ABCDEF"), concentrations_uM=list(m.CONC),
                     enzyme_cols=sp["cols"], bg_cols=sp["bg"])],
        enzyme_uM=sp["E"], signal_per_uM=m.SPU_BIC,
        time_range_seconds=sp["tr"], baseline_subtract=True)
    kcat, Km = mm["kcat_per_s"], mm["Km_uM"]
    rk = mm["kcat_err_per_s"]/kcat
    ke = mm["kcat_over_Km_per_uM_per_s"]*1e6
    rke = (mm["kcat_over_Km_err_per_uM_per_s"]*1e6)/ke
    return dict(kcat=kcat, Km=Km, ke=ke,
                kcat_sd=kcat*np.hypot(np.hypot(rk, SIG_E), CAL),
                Km_sd=mm["Km_err_uM"], ke_sd=ke*np.hypot(np.hypot(rke, SIG_E), CAL),
                rat=kcat/KU, rat_sd=(kcat/KU)*np.hypot(np.hypot(rk, SIG_E), KUSD/KU))

def build():
    TL, TG = m.spec(TAGLESS_CELL), m.spec(9)       # tagless, tagged
    ftl, ftg = fit(TL), fit(TG)
    TEAL, GRAY_ = m.FAM["ZAPP-1"][1], "#8a8a84"

    fig = plt.figure(figsize=(6.6, 2.55))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.34,
                          left=0.085, right=0.775, top=0.80, bottom=0.20)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    # ---- A: progress curves, whole acquisition -------------------------------------
    m.draw_progress(axA, TL, m.SPU_BIC)
    axA.text(0.058, 0.930, "[E]$_0$ = %s µM" % m.plain(TL["E"], TL["E"]*SIG_E),
             transform=axA.transAxes, va="top", ha="left", fontsize=5.6, color=m.INK, zorder=7,
             bbox=dict(boxstyle="square,pad=0.30", fc="white", ec=m.INK2, lw=0.4))

    # ---- B: Michaelis-Menten, tagless vs tagged ------------------------------------
    sc = 1e-3
    for sp, f, col, lab, mk in ((TL, ftl, TEAL, "tagless", "o"), (TG, ftg, GRAY_, "tagged", "s")):
        S, v, e = m.mm_points(sp, m.SPU_BIC)
        xf = np.linspace(0, 10500, 300)
        axB.plot(xf/1000, (f["kcat"]*xf/(f["Km"]+xf))/sc, color=col, lw=1.1, zorder=2,
                 ls="-" if lab == "tagless" else (0, (4, 2)))
        axB.plot(S/1000, v/sc, mk, ms=3.6, mfc=col, mec="black", mew=0.5, ls="none", zorder=3)
        axB.errorbar(S/1000, v/sc, yerr=e/sc, fmt="none", ecolor="black", elinewidth=0.9,
                     capsize=2.0, capthick=0.9, zorder=5)
    axB.set_xlabel("[Paraoxon] (mM)", fontsize=m.LABFS, labelpad=2)
    axB.set_ylabel("$v_0$/[E]$_0$ (10$^{-3}$ s$^{-1}$)", fontsize=m.LABFS, labelpad=2)
    axB.set_xlim(0, 10.5); axB.set_ylim(0, None)
    axB.set_xticks([0, 4.8, 9.6]); m.style(axB)
    axB.xaxis.set_minor_locator(AutoMinorLocator(4))
    axB.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=4))
    axB.text(0.055, 0.945, "$n$ = 3", transform=axB.transAxes, ha="left", va="top",
             fontsize=6.0, color=m.INK2)
    axB.legend(handles=[Line2D([], [], color=TEAL, marker="o", ms=3.6, mec="black", mew=.5, lw=1.1),
                        Line2D([], [], color=GRAY_, marker="s", ms=3.6, mec="black", mew=.5,
                               lw=1.1, ls=(0, (4, 2)))],
               labels=["ZAPP-1 tagless", "ZAPP-1 (Strep-tag II)"], loc="lower right",
               frameon=False, fontsize=5.8, handlelength=2.0, labelspacing=0.25,
               borderpad=0.1, handletextpad=0.5)

    fig.canvas.draw()
    p1, p2 = axA.get_position(), axB.get_position()
    fig.text((p1.x0+p2.x1)/2, p1.y1 + 0.155, "ZAPP-1 without the C-terminal Strep-tag II",
             ha="center", va="bottom", fontsize=8.0, color=TEAL, fontweight="bold")
    line = (f"tagless   $k_{{cat}}$ = {m.plain(ftl['kcat'], ftl['kcat_sd'])} s$^{{-1}}$   "
            f"$K_{{M}}$ = {ftl['Km']/1000:.0f} ± {ftl['Km_sd']/1000:.0f} mM   "
            f"$k_{{cat}}$/$K_{{M}}$ = {m.plain(ftl['ke'], ftl['ke_sd'])} M$^{{-1}}$s$^{{-1}}$\n"
            f"tagged    $k_{{cat}}$ = {m.plain(ftg['kcat'], ftg['kcat_sd'])} s$^{{-1}}$   "
            f"$K_{{M}}$ = {ftg['Km']/1000:.0f} ± {ftg['Km_sd']/1000:.0f} mM   "
            f"$k_{{cat}}$/$K_{{M}}$ = {m.plain(ftg['ke'], ftg['ke_sd'])} M$^{{-1}}$s$^{{-1}}$")
    fig.text((p1.x0+p2.x1)/2, p1.y1 + 0.025, line, ha="center", va="bottom",
             fontsize=6.1, color=m.INK, linespacing=1.30)
    lg = fig.legend([Line2D([], [], color=m.RAMP[c], lw=1.8) for c in m.CONC],
                    [f"{c:,.0f}" for c in m.CONC], title="[Paraoxon] (µM)", loc="upper left",
                    bbox_to_anchor=(0.795, 1.00), frameon=False, fontsize=6,
                    title_fontsize=6.5, handlelength=1.3, labelspacing=0.30)
    lg._legend_box.align = "left"; fig.add_artist(lg)
    fig.legend([Patch(facecolor=m.GRAY, alpha=0.22, lw=0)], ["outside the\ninitial-velocity\nfit window"],
               loc="upper left", bbox_to_anchor=(0.795, 0.36), frameon=False, fontsize=6,
               handlelength=1.3, handleheight=1.1)._legend_box.align = "left"

    out = os.path.join(ROOT, "wetlab_data_plots", "paper_figures", "side"); os.makedirs(out, exist_ok=True)
    for ext, dpi in (("png", 150), ("pdf", None)):
        fig.savefig(os.path.join(out, f"SIDE__ZAPP1_tagless.{ext}"), dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print("ZAPP-1 tagless  vs  ZAPP-1 (Strep-tag II)   [E]0 = %.1f vs %.1f uM\n" % (TL["E"], TG["E"]))
    for k, unit, sd in (("kcat", "s-1", "kcat_sd"), ("Km", "uM", "Km_sd"),
                        ("ke", "M-1 s-1", "ke_sd"), ("rat", "", "rat_sd")):
        nm = {"kcat": "kcat", "Km": "Km", "ke": "kcat/Km", "rat": "kcat/kuncat"}[k]
        print(f"  {nm:12} {ftl[k]:12.4g} +/- {ftl[sd]:<10.3g}  {ftg[k]:12.4g} +/- {ftg[sd]:<10.3g}"
              f"   ratio {ftl[k]/ftg[k]:.2f}x")
    A_LIN = 2.5                                     # reader linear range
    print("\ndetector-saturation check (refit using only the window where every well is <2.5 AU):")
    for tag, sp, base in (("tagless", TL, ftl), ("tagged", TG, ftg)):
        piv = m.traces(sp); t = piv.index.to_numpy(float)
        Y = np.column_stack([piv[f"{r}{k}"].to_numpy(float)
                             for r in "ABCDEF" for k in sp["cols"]])
        bad = (Y > A_LIN) | ~np.isfinite(Y)
        idx = np.where(bad.any(axis=1))[0]
        t_lin = float(t[idx[0]]) if len(idx) else float(t[-1])
        r = fit(dict(sp, tr=(sp["tr"][0], t_lin)))
        print(f"  {tag:8} linear window 300-{t_lin:.0f} s ({t_lin/60:.1f} min): "
              f"kcat {r['kcat']:.4f} ({r['kcat']/base['kcat']-1:+.0%}), "
              f"Km {r['Km']/1000:.1f} mM, kcat/Km {r['ke']:.2f} ({r['ke']/base['ke']-1:+.0%})")
    print("\nsaved -> figures/side/SIDE__ZAPP1_tagless.{png,pdf}")
