"""RFdiffusion3 paraoxon kinetics and manuscript figure reproduction.

The generic RFdiffusion2 fitter is reused without changing its numerical method.
Paper presentation lives separately in ``rfd3_figures``; the reference workbook
retains the originally reported precision and uncertainty budget.
"""
from pathlib import Path
from functools import lru_cache
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kinetics import parse_kinetics

SPU_BIC, SPU_NOBIC = 0.00667808, 0.00633035


@lru_cache(maxsize=1)
def paper_style():
    # Importing source drawing helpers must not alter a user's global style.
    with plt.rc_context():
        from rfd3_figures import make_kinetics_SI_figures as paper
    return paper


# Scaffold IDs are the global identifiers in the deposited all_designs sheet.
# The five characterized families retain the manuscript's label/curve colors.
SCREEN_SCAFFOLD_COLORS = {
    1: "#8a6db1", 2: "#bc8b45", 3: "#2e8b82", 4: "#8d5871",
    5: "#667f4b", 6: "#957357", 7: "#cf7c52", 8: "#699bb0",
    9: "#3c4d8c", 10: "#aa724b", 11: "#8b77aa", 12: "#c9647a",
    13: "#4a68a8", 14: "#738b38", 15: "#ad5f96", 16: "#477f91",
    17: "#9f8d48",
}
SCREEN_KINETICS_LABELS = {
    1: {"D1": "ZAPP-1", "D7": "R1 p1D7", "D8": "R1 p1D8", "E10": "R1 p1E10"},
    2: {"F7": "ZAPP-2", "H1": "ZAPP-3", "G5": "ZAPP-4", "C12": "ZAPP-5",
        "E24": "R2 p2B12", "B18": "R2 p1D9", "D8": "R2 p1H4"},
}


def load_screening_PTE_RFd3(round_number):
    """Load every raw screening trace and its deposited design/scaffold mapping.

    Mapping is checked against all 288 ordered designs. Round 2 uses interleaved
    row pairs: reader A–D contain source plate 1; E–H contain source plate 2.
    Global scaffold IDs are 1–6 (round 1) and 7–17 (round 2).
    """
    from neo2_util import parse_neo2_kinetics
    import re

    if round_number not in (1, 2):
        raise ValueError("round_number must be 1 or 2")
    paper = paper_style()
    designs = pd.read_excel(
        Path(paper.ROOT) / "supplemental_data" /
        "supp_data__denovo_PTE_DNA_and_protein_sequences.xlsx", sheet_name="all_designs")
    mapping = designs.loc[designs.design_campaign == round_number].copy()
    if round_number == 1:
        traces = (parse_neo2_kinetics(paper.SCREEN_R1)
                  .groupby(["Well", "time"], as_index=False).agg(value=("value", "mean"))
                  .pivot(index="time", columns="Well", values="value").sort_index())
        time_hours = (traces.index.to_numpy(float) - traces.index.min()) / 3600
        mapping["reader_well"] = mapping["well"]
    else:
        raw = pd.read_csv(paper.SCREEN_R2)
        wells = [c for c in raw if re.fullmatch(r"[A-H](?:[1-9]|1[0-9]|2[0-4])", str(c))]
        traces = raw[wells].apply(pd.to_numeric, errors="coerce")
        time_hours = raw["Time"].to_numpy(float) * 24
        time_hours -= time_hours[0]
        def reader_well(row):
            source_row, source_column = row.well[0], int(row.well[1:])
            row_index = "ABCDEFGH".index(source_row)
            reader_row = "ABCDEFGH"[(int(row.order_plate)-1)*4 + row_index//2]
            return f"{reader_row}{2*source_column-1+row_index%2}"
        mapping["reader_well"] = [reader_well(row) for row in mapping.itertuples()]
    expected = {1: 96, 2: 192}[round_number]
    if len(mapping) != expected or len(traces.columns) != expected:
        raise ValueError(f"Round {round_number}: expected {expected} designs and traces")
    if mapping.reader_well.duplicated().any() or set(mapping.reader_well) != set(traces):
        raise ValueError("Design mapping does not cover each reader well exactly once")
    mapping["screen_scaffold"] = mapping.scaffold - (6 if round_number == 2 else 0)
    mapping["color"] = mapping.scaffold.map(SCREEN_SCAFFOLD_COLORS)
    mapping["kinetics_tested"] = mapping.reader_well.isin(SCREEN_KINETICS_LABELS[round_number])
    mapping["plot_label"] = mapping.reader_well.map(SCREEN_KINETICS_LABELS[round_number])
    if mapping.color.isna().any():
        raise ValueError("A scaffold has no assigned color")
    return time_hours, traces, mapping


def plot_screening_PTE_RFd3(round_number, *, plot_path=None, show=True, dpi=600,
                             scaffold=None):
    """Plot every screening design individually, colored by its scaffold.

    SI typography, boxed axes, time in hours and first-read-referenced ΔA405
    are retained. Every design is a line: there is no envelope, percentile
    reduction, hit filter, or fitted replacement. Kinetics-tested designs are
    drawn last, bold, and directly labeled. ``scaffold`` optionally selects a
    global scaffold ID for inspection; the default always includes all designs.
    """
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    paper = paper_style()
    time_hours, traces, mapping = load_screening_PTE_RFd3(round_number)
    if scaffold is not None:
        mapping = mapping.loc[mapping.scaffold == scaffold]
        if mapping.empty:
            raise ValueError(f"Scaffold {scaffold} is absent from round {round_number}")
    conditions = {
        1: "300 µM paraoxon · 1% MeOH · 200 µM ZnSO$_4$ · 25 °C, 1 h",
        2: "600 µM paraoxon · 1% MeOH · 25 mM NaHCO$_3$ (final) · 25 °C, 1 h",
    }
    with plt.rc_context({**paper.STYLE, "mathtext.cal": paper.fam}):
        fig = plt.figure(figsize=(4.8, 3.0))
        ax = fig.add_axes([.12, .29, .85, .48])
        ends = []
        for row in mapping.sort_values("kinetics_tested", kind="stable").itertuples():
            y = traces[row.reader_well].to_numpy(float)
            y = y - y[0]
            ax.plot(time_hours, y, color=row.color,
                    lw=1.5 if row.kinetics_tested else .48,
                    alpha=1.0 if row.kinetics_tested else .7,
                    zorder=4 if row.kinetics_tested else 2,
                    label=row.design_id, gid=f"design:{row.design_id}")
            if row.kinetics_tested and np.isfinite(y[-1]):
                ends.append((y[-1], row.plot_label, row.color))
        ax.set_xlabel("Time (h)", fontsize=paper.LABFS, labelpad=2)
        ax.set_ylabel("$\\Delta$A$_{405}$", fontsize=paper.LABFS, labelpad=2)
        ax.set_xlim(0, time_hours.max() * 1.30)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + (hi-lo)*.10)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=4))
        paper.style(ax)
        paper.place_labels(ax, ends, time_hours[-1], fs=5.6)
        bounds = ax.get_position()
        center = (bounds.x0 + bounds.x1) / 2
        suffix = "" if scaffold is None else f", scaffold {scaffold}"
        fig.text(center, bounds.y1 + .325 / fig.get_figheight(),
                 f"Round {round_number} eluate screen{suffix}  ($\\mathbfit{{n}}$ = {len(mapping)} designs)",
                 ha="center", va="bottom", fontsize=7.8, color=paper.INK, fontweight="bold")
        fig.text(center, bounds.y1 + .05 / fig.get_figheight(),
                 conditions[round_number] + "\nall designs colored by scaffold; bold = kinetics tested",
                 ha="center", va="bottom", fontsize=6.1, color=paper.INK, linespacing=.94)
        handles = [Line2D([], [], color=SCREEN_SCAFFOLD_COLORS[s], lw=1.0,
                          label=f"{s} (n={sum(mapping.scaffold == s)})")
                   for s in sorted(mapping.scaffold.unique())]
        fig.legend(handles=handles, title="Scaffold (deposited model IDs)",
                   loc="lower center", bbox_to_anchor=(.55, .005), ncol=6,
                   frameon=False, fontsize=5.6, title_fontsize=5.8,
                   handlelength=1.4, columnspacing=1.0, labelspacing=.5)
        if plot_path:
            fig.savefig(plot_path, dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
    return fig, ax


def run_kinetics_PTE_RFd3(name, *, data_path, enzyme_uM, enzyme_cols, bg_cols,
                        condition="+25 mM NaHCO3", time_range_seconds=(300, 20000),
                        plot_path=None, show=True, save_eps=False):
    """Fit six paraoxon concentrations and draw RFdiffusion3 progress/MM panels.

    Progress curves show the complete acquisition with the paper's robust
    pre-window baseline, purple substrate palette, SEM bands and shaded excluded
    intervals. The MM panel uses scaffold colors and paper-reported annotations.
    Raw fits remain available even when the paper reports kcat and KM as n.d.
    """
    paper = paper_style()
    spu = SPU_NOBIC if condition == "no NaHCO3" else SPU_BIC
    df, reps, mm = parse_kinetics(
        data_path=str(data_path),
        blocks=[dict(rows=list("ABCDEF"), concentrations_uM=list(paper.CONC),
                     enzyme_cols=list(enzyme_cols), bg_cols=list(bg_cols))],
        enzyme_uM=enzyme_uM, signal_per_uM=spu,
        time_range_seconds=time_range_seconds, baseline_subtract=True)
    if show or plot_path:
        sp = dict(path=str(data_path), E=enzyme_uM, cols=list(enzyme_cols),
                  bg=list(bg_cols), tr=time_range_seconds)
        with plt.rc_context({**paper.STYLE, "mathtext.cal": paper.fam}):
            fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.65))
            fig.subplots_adjust(left=.10, right=.77, bottom=.23, top=.70, wspace=.55)
            paper.draw_progress(axs[0], sp, spu)
            if name == "ZAPP-1 tagless" or name not in paper.TAB.index:
                row = result_row(name, mm, condition)
                paper.TAB.loc[name] = row
                paper.FAM[name] = paper.FAM["ZAPP-1"]
            lines = paper.draw_mm(axs[1], name, sp, spu)
            fig.text(.43, .93, name, ha="center", fontsize=9,
                     color=paper.famcol(name)[1], fontweight="bold")
            fig.text(.43, .76, "\n".join(lines), ha="center", fontsize=6.5)
            _standalone_legend(fig, paper)
            if plot_path:
                fig.savefig(plot_path, dpi=150, bbox_inches="tight")
                if save_eps:
                    fig.savefig(Path(plot_path).with_suffix(".eps"), bbox_inches="tight")
            if show:
                plt.show()
            plt.close(fig)
    return mm


def _standalone_legend(fig, paper):
    """Stack legends by their rendered height on a short two-panel figure."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    substrate = fig.legend(
        [Line2D([], [], color=paper.RAMP[c], lw=1.8) for c in paper.CONC],
        [f"{c:,.0f}" for c in paper.CONC], title="[Paraoxon] (µM)",
        loc="upper left", bbox_to_anchor=(.79, .99), frameon=False,
        fontsize=6, title_fontsize=6.5, handlelength=1.3, labelspacing=.30)
    substrate._legend_box.align = "left"
    fig.canvas.draw()
    box = substrate.get_window_extent(fig.canvas.get_renderer()).transformed(
        fig.transFigure.inverted())
    window = fig.legend(
        [Patch(facecolor=paper.GRAY, alpha=.22, lw=0)],
        ["outside the\ninitial-velocity\nfit window"],
        loc="upper left", bbox_to_anchor=(.79, box.y0-.035), frameon=False,
        fontsize=6, handlelength=1.3, handleheight=1.1)
    window._legend_box.align = "left"
    return substrate, window


def result_row(name, mm, condition, kuncat=None):
    """Full-precision result with the manuscript's systematic error budget."""
    paper = paper_style()
    cal = .00192 if condition == "no NaHCO3" else .00135
    ku = kuncat or dict(k=5.0e-8, err=1.3e-8)
    kc, kce = mm["kcat_per_s"], mm["kcat_err_per_s"]
    ke, kee = mm["kcat_over_Km_per_uM_per_s"]*1e6, mm["kcat_over_Km_err_per_uM_per_s"]*1e6
    kc_sd = np.sqrt(kce**2 + (kc*.06)**2 + (kc*cal)**2)
    ke_sd = np.sqrt(kee**2 + (ke*.06)**2 + (ke*cal)**2)
    r = {"design": name, "class": "tagless comparison", "condition": condition,
         "kcat (s-1)": kc, "kcat fit sd": kce, "kcat sd (total)": kc_sd,
         "Km (uM)": mm["Km_uM"], "Km sd": mm["Km_err_uM"],
         "kcat/Km (M-1 s-1)": ke, "kcat/Km fit sd": kee,
         "kcat/Km sd (total)": ke_sd, "kuncat (s-1)": ku["k"], "kuncat sd": ku["err"],
         "kcat/kuncat": kc/ku["k"],
         # The shared standard-curve calibration cancels in these ratios.
         "kcat/kuncat sd": kc/ku["k"]*np.sqrt((kce/kc)**2+.06**2+(ku["err"]/ku["k"])**2),
         "(kcat/Km)/kuncat (M-1)": ke/ku["k"],
         "(kcat/Km)/kuncat sd (M-1)": ke/ku["k"]*np.sqrt((kee/ke)**2+.06**2+(ku["err"]/ku["k"])**2)}
    for col, v, e in [("kcat (reported)",kc,kc_sd),
                       ("Km (reported, mM)",mm["Km_uM"]/1000,mm["Km_err_uM"]/1000),
                       ("kcat/Km (reported)",ke,ke_sd),
                       ("kcat/kuncat (reported)",r["kcat/kuncat"],r["kcat/kuncat sd"]),
                       ("(kcat/Km)/kuncat (reported)",r["(kcat/Km)/kuncat (M-1)"],r["(kcat/Km)/kuncat sd (M-1)"])]:
        r[col] = paper.plain(v,e)
    return r


def validate_reference(fits, kuncat, output_dir=None):
    """Fail on differences exceeding saved precision / numerical solver drift.

    The paper workbook contains rounded fit numbers; the no-bicarbonate entry
    rescales already rounded numbers. Comparisons use those explicit precision
    limits rather than claiming bitwise equality with freshly fitted values.
    """
    paper = paper_style()
    ref = pd.read_excel(paper.XLSX, sheet_name="kinetics").set_index("design")
    missing = set(ref.index)-set(fits)
    if missing:
        raise ValueError(f"Run all paper kinetics first; missing {sorted(missing)}")
    columns = [("kcat_per_s","kcat (s-1)",1, .5e-6),
               ("kcat_err_per_s","kcat fit sd",1, .5e-6),
               ("Km_uM","Km (uM)",1,.005),
               ("Km_err_uM","Km sd",1,.005),
               ("kcat_over_Km_per_uM_per_s","kcat/Km (M-1 s-1)",1e6,.0005),
               ("kcat_over_Km_err_per_uM_per_s","kcat/Km fit sd",1e6,.0005)]
    checks=[]
    for name,row in ref.iterrows():
        factor = SPU_BIC/SPU_NOBIC if row["condition"]=="no NaHCO3" else 1.
        mm=fits[name]["mm"]
        for key,col,scale,precision in columns:
            actual, expected = mm[key]*scale, float(row[col])
            tolerance = precision*(factor if key not in ("Km_uM","Km_err_uM") else 1.)
            # Covariance SE varies slightly with scipy's finite-difference solver.
            tolerance += 3e-5*abs(expected) if "err" in key else 1e-7*abs(expected)
            checks.append(dict(design=name,quantity=col,refitted=actual,reference=expected,
                               tolerance=tolerance,passed=abs(actual-expected)<=tolerance))
    # Reproduce the workbook's propagated errors/ratios from its rounded
    # stored fit inputs. This is deliberately distinct from the fresh refits.
    for name,r in ref.iterrows():
        cal=.00192 if r['condition']=='no NaHCO3' else .00135
        kc,ke=r['kcat (s-1)'],r['kcat/Km (M-1 s-1)']
        ku,kusd=r['kuncat (s-1)'],r['kuncat sd']
        derived={
            'kcat sd (total)':np.sqrt(r['kcat fit sd']**2+(kc*.06)**2+(kc*cal)**2),
            'kcat/Km sd (total)':np.sqrt(r['kcat/Km fit sd']**2+(ke*.06)**2+(ke*cal)**2),
            'kcat/kuncat':kc/ku, '(kcat/Km)/kuncat (M-1)':ke/ku,
            'kcat/kuncat sd':kc/ku*np.sqrt((r['kcat fit sd']/kc)**2+.06**2+(kusd/ku)**2),
            '(kcat/Km)/kuncat sd (M-1)':ke/ku*np.sqrt((r['kcat/Km fit sd']/ke)**2+.06**2+(kusd/ku)**2)}
        for col,actual in derived.items():
            expected=float(r[col]);tolerance=max(1e-14,abs(expected)*1e-10)
            checks.append(dict(design=name,quantity=col,refitted=actual,reference=expected,
                               tolerance=tolerance,passed=abs(actual-expected)<=tolerance,
                               basis='propagation from saved rounded fit inputs'))
    # The two unresolved mutants use these through-origin estimates in the
    # figures. Check all six stored chord values and their total uncertainties.
    for name,row in ref[ref['class']=='mutant'].iterrows():
        mm=fits[name]['mm']; agg=mm['_agg']
        substrate=agg['concentration_uM'].to_numpy(float)
        rate=agg['v_mean_uM_per_s'].to_numpy(float)/fits[name]['enzyme_uM']
        slope=float(substrate@rate/(substrate@substrate))
        se=np.sqrt(np.sum((rate-slope*substrate)**2)/(len(substrate)-1)/(substrate@substrate))
        for col,actual in [('kcat/Km chord (M-1 s-1)',slope*1e6),
                           ('kcat/Km chord sd',np.hypot(se,.06*slope)*1e6)]:
            expected=float(row[col]); tolerance=.00005+abs(expected)*1e-7
            checks.append(dict(design=name,quantity=col,refitted=actual,reference=expected,
                               tolerance=tolerance,passed=abs(actual-expected)<=tolerance))
    for cond,r in kuncat.items():
        refrow=ref[ref.condition==cond].iloc[0]
        for key,col,tol in [("k","kuncat (s-1)",.5e-11),("err","kuncat sd",.5e-10 if cond!="no NaHCO3" else .5e-11)]:
            actual,expected=r[key],float(refrow[col])
            checks.append(dict(design=cond,quantity=col,refitted=actual,reference=expected,
                               tolerance=tol,passed=abs(actual-expected)<=tol))
    windows=pd.read_excel(paper.XLSX,sheet_name='kuncat').set_index('condition')
    for cond,r in kuncat.items():
        expected=int(windows.loc[cond,'n fit windows'])
        checks.append(dict(design=cond,quantity='n fit windows',refitted=r['n'],reference=expected,
                           tolerance=0,passed=r['n']==expected))
    result=pd.DataFrame(checks)
    result['basis']=result['basis'].fillna('raw-plate refit')
    if output_dir is not None:
        out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
        result.to_csv(out/"paper_reproduction_validation.csv",index=False)
    if not result.passed.all():
        raise AssertionError("Paper numerical validation failed:\n"+result[~result.passed].to_string(index=False))
    print(f"Paper reference: {len(result)} numerical checks passed at stored precision.")
    return result


def reproduce_paper_figures(notebook_namespace):
    """Rebuild the four kinetics SI pages, Fig. 4 E/F/G/I and tagless comparison.

    E is retained as a historical reproduction. A second E output fixes its
    inherited no-bicarbonate point-calibration inconsistency without replacing
    the reference. All source data come from the deposited dataset.
    """
    paper=paper_style()
    validate_reference(notebook_namespace["FITS"],notebook_namespace["KUNCAT"])
    with plt.rc_context(paper.STYLE):
        paper.build_all()
        from rfd3_figures import side_ZAPP1_tagless
        side_ZAPP1_tagless.build()
    with plt.rc_context():
        background = importlib.import_module("rfd3_figures.make_SI_figure")
        background.build()
    with plt.rc_context():
        from rfd3_figures import make_panelE_bicarbonate as panel_e
        panel_e.build()
        panel_e.build(correct_calibration=True)
        from rfd3_figures import make_fig4_panels_FI as panels_fi
        panels_fi.build_F(); panels_fi.build_I()
    with plt.rc_context():
        from rfd3_figures import make_fig4_panelG_typeset as panel_g
        ns=dict(notebook_namespace)
        ns["baseline_subtracted_trace"]=lambda w: ns["abs_df"][w].to_numpy(float)-ns["abs_df"][w].iloc[0]
        ns["SCAFFOLD_TO_ZAPP"]={1:5,3:2,6:4,7:3}
        from matplotlib.colors import to_rgb
        ns["BACKGROUND_TRACE_FLAT"]=tuple(.55*np.array(to_rgb(ns["BACKGROUND_TRACE_COLOR"]))+.45)
        panel_g.build(ns)
    return Path(paper.FIGDIR)
