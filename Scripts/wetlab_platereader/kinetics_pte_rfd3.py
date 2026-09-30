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
        with plt.rc_context(paper.STYLE):
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
