# Adapted from FOR_RFdiffusion3_paper/scripts/make_SI_figure.py (2026-09-29).
# Paper geometry, colors and labels are retained; paths resolve inside this repository.
# Retain italic descriptive subscripts from the supplied PDF (the later source
# script changed these to roman without regenerating this particular PDF).
"""SI figure: kuncat determination (A,B) + rate enhancement and proficiency (C,D)."""
import sys
import os
import numpy as np, pandas as pd, matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from neo2_util import parse_neo2_kinetics

from . import make_kinetics_SI_figures as K
ROOT = K.ROOT
P = next(str(p) for p in __import__("pathlib").Path(K.RAW).glob("260807_kuncat*.xlsx"))

ROWS = list("ABCDEF"); CONC = np.array([9600,4800,2400,1200,600,300], float)
C_NO, C_BIC = "#1f5fa8", "#c2521f"
CFG = {"no":  dict(cols=[1,2,3], spu=0.00633035, color=C_NO,  mk="o",
                   label="40 mM HEPES, 50 mM NaCl, 200 µM ZnSO$_4$, pH 8.0\nno NaHCO$_3$"),
       "bic": dict(cols=[4,5,6], spu=0.00667808, color=C_BIC, mk="s",
                   label="40 mM HEPES, 50 mM NaCl, 200 µM ZnSO$_4$, pH 8.0\n+ 25 mM NaHCO$_3$")}

fam = next((f for f in ["Arial","Helvetica","Liberation Sans","DejaVu Sans"]
            if f in {x.name for x in font_manager.fontManager.ttflist}), "sans-serif")
BACKGROUND_STYLE = {
    "font.family": fam, "font.size": 11,
    "axes.labelsize": 11, "axes.titlesize": 9.5,
    "axes.linewidth": 1.0, "axes.edgecolor": "black", "axes.labelcolor": "black",
    "text.color": "black", "xtick.color": "black", "ytick.color": "black",
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.top": False, "ytick.right": False,          # never inherit all-sides ticks
    "xtick.bottom": True, "ytick.left": True,
    "axes.spines.top": True, "axes.spines.right": True,
    "axes.spines.bottom": True, "axes.spines.left": True,
    "xtick.minor.width": 0.8, "ytick.minor.width": 0.8,
    "xtick.minor.size": 2.2, "ytick.minor.size": 2.2,
    "xtick.major.width": 1.0, "ytick.major.width": 1.0,
    "xtick.major.size": 4, "ytick.major.size": 4,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "custom",
    "mathtext.rm": fam, "mathtext.it": f"{fam}:italic", "mathtext.bf": f"{fam}:bold",
}

def build():
    plt.rcParams.update(BACKGROUND_STYLE)
    raw = parse_neo2_kinetics(P)
    d = raw.groupby(["Well","time"], as_index=False).agg(signal=("value","mean"))
    d["time"] = d["time"].astype(float) - d.groupby("Well")["time"].transform("min")
    piv = d.pivot(index="time", columns="Well", values="signal").sort_index()
    t = piv.index.to_numpy(float); TEND = t.max()

    def block(cond, lo, hi):
        cf = CFG[cond]; m = (t >= lo) & (t <= hi); tt = t[m]
        reps = [np.polyfit(tt, np.column_stack([piv[f"{r}{k}"].to_numpy(float)[m]
                for k in cf["cols"]]), 1)[0]/cf["spu"] for r in ROWS]
        V = np.array([x.mean() for x in reps])
        E = np.array([x.std(ddof=1)/np.sqrt(len(x)) for x in reps])
        k, c = np.polyfit(CONC, V, 1); pred = k*CONC + c
        ss = np.sum((V-V.mean())**2)
        return k, c, (1-np.sum((V-pred)**2)/ss if ss > 0 else np.nan), V, E

    REP = {}
    for cond in CFG:
        A = []
        for lo in np.arange(900, TEND-2700+1, 300):
            for hi in np.arange(lo+2700, TEND+1, 300):
                k, c, r2, V, E = block(cond, lo, hi)
                if r2 >= 0.90:
                    jk = np.array([np.polyfit(np.delete(CONC,i), np.delete(V,i),1)[0] for i in range(6)])
                    A.append((lo, hi, k, np.sqrt(5/6*np.sum((jk-jk.mean())**2))))
        A = np.array(A); k = A[:,2]; med = np.median(k)
        p16, p84 = np.percentile(k, [16,84]); stat = np.median(A[:,3])
        err = max(np.hypot(med-p16, stat), np.hypot(p84-med, stat))
        b = A[np.argmin(np.abs(k-med))]
        REP[cond] = dict(k=med, err=err, lo=b[0], hi=b[1])

    def sig(x, n=2):
        from math import log10, floor
        return 0.0 if x == 0 else round(x, -(int(floor(log10(abs(x)))) - (n-1)))
    def thalf(k, err):
        L = np.log(2)/86400
        return f"{sig(L/(k+err)):.0f}–{sig(L/(k-err)):.0f} d"

    # ---------------- figure : Science full-page SI width = 18.3 cm = 7.2 in ----------------
    fig = plt.figure(figsize=(7.2, 7.1))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.30], hspace=0.40, wspace=0.44,
                          left=0.120, right=0.985, top=0.905, bottom=0.072)
    axA, axB = fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1])
    axC, axD = fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1])

    lims = []
    for ax, cond in zip((axA, axB), CFG):
        cf, rp = CFG[cond], REP[cond]
        k, c, r2, V, E = block(cond, rp["lo"], rp["hi"])
        Vc, Ec = (V-c)*1e3, E*1e3
        xf = np.linspace(0, 10500, 60)
        ax.fill_between(xf, (rp["k"]-rp["err"])*xf*1e3, (rp["k"]+rp["err"])*xf*1e3,
                        color=cf["color"], alpha=0.16, lw=0, zorder=1)
        ax.plot(xf, k*xf*1e3, color=cf["color"], lw=1.4, zorder=2)
        ax.errorbar(CONC, Vc, yerr=Ec, fmt="none", ecolor="black", elinewidth=1.0,
                    capsize=3, capthick=1.0, zorder=3)
        ax.plot(CONC, Vc, cf["mk"], ms=6, mfc=cf["color"], mec="black", mew=0.8, ls="none", zorder=4)
        ax.axhline(0, color="black", lw=0.7, zorder=1)
        lims.append(max((Vc+Ec).max(), k*10500*1e3))
        ax.set_xlabel("[Paraoxon] (mM)", labelpad=6)
        ax.set_ylabel("$v_0$ (10$^{-3}$ µM s$^{-1}$)", labelpad=6)
        ax.set_xlim(0, 10500); ax.set_xticks([0, 2400, 4800, 7200, 9600])
        ax.set_xticklabels(["0","2.4","4.8","7.2","9.6"])
        for sp in ax.spines.values(): sp.set_visible(True)      # box around the panel
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax.tick_params(which="both", top=False, right=False)
        ax.set_title(cf["label"], fontsize=9.5, pad=8, linespacing=1.4)
        ax.text(0.05, 0.94,
                f"$k$$_{{uncat}}$ = ({rp['k']*1e8:.1f} ± {rp['err']*1e8:.1f}) × 10$^{{-8}}$ s$^{{-1}}$\n"
                f"$t$$_{{1/2}}$ = {thalf(rp['k'], rp['err'])}",
                transform=ax.transAxes, fontsize=9.5, va="top", linespacing=1.7)
    ymax = max(lims)
    for ax in (axA, axB):
        ax.set_ylim(-0.035*ymax, 1.10*ymax)
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    # ---------------- C, D : rate enhancement and catalytic proficiency ----------------
    XL = K.XLSX
    df = pd.read_excel(XL, sheet_name="kinetics")
    df = df[df["class"] != "mutant"].rename(columns={"design":"design name"}).set_index("design name")
    df = df.rename(columns={"kcat/kuncat sd":"kcat/kuncat std",
                            "(kcat/Km)/kuncat sd (M-1)":"(kcat/Km)/kuncat std (M-1)"})
    # grouped by scaffold: Round 1 (PET_i1) then Round 2 (PTE_i2 s3, s7, s6, s1)
    ORDER = ["ZAPP-1", "ZAPP-1 (no NaHCO3)", "R1 p1D7", "R1 p1D8", "R1 p1E10",
             "ZAPP-2", "R2 p2B12", "ZAPP-3", "ZAPP-4", "ZAPP-5", "R2 p1D9", "R2 p1H4"]
    NR1 = 5                                     # first five are Round 1
    df = df.loc[ORDER]
    names = [n.replace(" (no NaHCO3)", "\n(no NaHCO$_3$)") for n in ORDER]
    cols = [C_NO if c == "no NaHCO3" else C_BIC for c in df["condition"]]
    y = np.arange(len(df))[::-1]                # first entry at the top

    for ax, val, err, xlab, scale, major in [
            (axC, "kcat/kuncat", "kcat/kuncat std",
             "$k$$_{cat}$/$k$$_{uncat}$ (× 10$^5$)", 1e5, 1.0),
            (axD, "(kcat/Km)/kuncat (M-1)", "(kcat/Km)/kuncat std (M-1)",
             "($k$$_{cat}$/$K$$_{M}$)/$k$$_{uncat}$ (× 10$^7$ M$^{-1}$)", 1e7, 2.0)]:
        v = df[val].to_numpy()/scale; e = df[err].to_numpy()/scale
        ax.barh(y, v, height=0.68, color=cols, edgecolor="black", linewidth=0.7, zorder=2)
        ax.errorbar(v, y, xerr=e, fmt="none", ecolor="black", elinewidth=0.9,
                    capsize=2.5, capthick=0.9, zorder=3)
        ax.axhline(y[NR1-1]-0.5, color="black", lw=0.7, ls=(0,(3,2.5)), zorder=1)
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9.5)
        ax.set_ylim(-0.75, len(df)-0.25)
        ax.set_xlabel(xlab, labelpad=6)
        ax.set_xlim(0, None)
        ax.xaxis.set_major_locator(MultipleLocator(major))
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        for sp in ax.spines.values(): sp.set_visible(True)      # box around the panel
        ax.tick_params(axis="x", which="both", top=False, bottom=True, direction="out")
        ax.tick_params(axis="y", which="both", left=True, right=False)
    axC.text(0.985, 0.985, "Round 1", transform=axC.transAxes, fontsize=8.5,
             ha="right", va="top", color="#52514e")
    axC.text(0.985, 0.40, "Round 2", transform=axC.transAxes, fontsize=8.5,
             ha="right", va="top", color="#52514e")

    for ax, L in ((axA,"A"), (axB,"B"), (axC,"C"), (axD,"D")):
        ax.text(-0.235 if ax in (axC, axD) else -0.20, 1.035, L, transform=ax.transAxes,
                fontsize=12, fontweight="normal", va="bottom", ha="left")

    FIGROOT = K.FIGDIR
    for ext, dpi in [("pdf", None), ("png", 150)]:
        outdir = os.path.join(FIGROOT, ext); os.makedirs(outdir, exist_ok=True)
        fig.savefig(os.path.join(outdir, f"PTE__v2SI__kuncat.{ext}"), dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    print("saved -> paper_figures/{pdf,png}/PTE__v2SI__kuncat.*")
    plt.close(fig)
