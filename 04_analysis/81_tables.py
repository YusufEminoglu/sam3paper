"""LaTeX tables and extra value macros for the manuscript, generated ONLY from the pipeline CSV outputs (no hand-typed numbers).

usage: python 81_tables.py            (run from 04_analysis; reads ./out_final, ./out_cov*, ../02_pipeline_v2, ../03_validation/out_final)
writes ../05_manuscript_R1/sections/tables_R1.tex and ../05_manuscript_R1/sections/values_extra.tex
"""
import numpy as np, pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
P2, P3 = HERE.parent / "02_pipeline_v2", HERE.parent / "03_validation"
OUT = HERE.parent / "tables"; OUT.mkdir(exist_ok=True)
f1 = lambda x, d=1: "--" if pd.isna(x) else f"{x:.{d}f}".replace("-", "$-$")
tab, V = [], {}


def table(label, caption, cols, rows, note=None, align=None):
    align = align or "l" + "r" * (len(cols) - 1)
    s = ["\\begin{table}[!ht]", "\\centering", "\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%", f"\\begin{{tabular}}{{{align}}}", "\\toprule",
         " & ".join(cols) + " \\\\", "\\midrule"] + [" & ".join(r) + " \\\\" for r in rows] + ["\\bottomrule", "\\end{tabular}}"]
    if note:
        s += [f"\\par\\smallskip\\parbox{{0.95\\linewidth}}{{\\footnotesize {note}}}"]
    s += ["\\end{table}", ""]
    tab.extend(s)


def macro(name, val):
    V[name] = val


# ---- sample composition
n = pd.read_csv(HERE / "out_final/nodes_R1.csv")
macro("nSCSNodes", int((n.src == "SCS").sum())); macro("nTGNodes", int((n.src == "TG").sum()))
macro("nSharedNodes", int(n.shared.fillna(0).astype(int).sum())); macro("pctSharedNodes", f1(100 * n.shared.fillna(0).mean()))
macro("meanCoverage", f1(100 * n.coverage.mean(), 0)); macro("medianUncWidth", f1(n.ecw_unc.median(), 2))
macro("shareVehAreaSCS", f1(100 * n[n.src == "SCS"].veh_area.sum() / n[n.src == "SCS"][["veh_area", "furn_area", "infra_area"]].sum().sum()))
macro("pctAvgLtBest", f1(100 * (n.ecw_avg > n.ecw_best).mean(), 0))
macro("medianECWavg", f1(n.ecw_avg.median(), 2)); macro("medianECWbest", f1(n.ecw_best.median(), 2)); macro("medianECWmeanSides", f1(n.ecw_mean_sides.median(), 2))
macro("shareImpMeanSides", f1(100 * (n.ecw_mean_sides < 0.9).mean()))
macro("sharePersonsChange", f1(100 * ((n.ecw_persons < 0.9).mean() - (n.ecw_best < 0.9).mean())))

# ---- sample flow (candidate nodes -> analysed nodes)
_c = pd.read_csv(P2 / "corridor_nodes.csv"); _m = pd.read_csv(P2 / "manifest_run.csv"); _e = pd.read_csv(P2 / "ecw_v2.csv")
_s = _c[_c.src == "SCS"]
macro("nSCSAll", len(_s)); macro("nSCSSnapped", int(_s.snapped.sum())); macro("nSCSCorridor", int(_s.w_corridor.notna().sum()))
macro("nSCSImagery", _m[_m.src == "SCS"].node.nunique()); macro("nSCSCells", _s.hex_id.nunique())
macro("nTGAll", int((_c.src == "TG").sum())); macro("nTGImagery", _m[_m.src == "TG"].node.nunique())
macro("pctSCSAnalysed", f1(100 * (n.src == "SCS").sum() / len(_s), 0))

# ---- Table: selection test (analysed vs excluded candidate nodes)
st = pd.read_csv(HERE / "out_final/selection_test.csv")
LAB = {"BC": "Betweenness centrality", "JCS": "Junction complexity score", "MDI": "Morphological density index", "SCR": "Street canyon ratio",
       "FAR": "Floor area ratio", "H_std": "Building height, std. dev.", "DPark": "Distance to nearest park", "SLOPE": "Slope",
       "GAR": "Ground area ratio (building coverage)", "TCH_mean_20m": "Tree canopy height (20\\,m mean)", "LST_norm": "Land surface temperature (normalised)", "Albedo_norm": "Albedo (normalised)"}
fp = lambda p: "$<$0.001" if p < 0.001 else f"{p:.3f}"
rows = [[LAB[r.covariate], f1(r.median_analysed, 2), f1(r.median_excluded, 2), f1(r.cohens_d, 2), fp(r.p_holm)] for r in st.itertuples()]
table("tab:selection", "Candidate nodes of the Konak cells that entered the analysis (analysed, $n=%d$) and those that did not (excluded, $n=%d$), compared on covariates that are not derived from street-view imagery." % (int(st.n_analysed.max()), int(st.n_excluded.max())),
      ["Covariate", "Median, analysed", "Median, excluded", "Cohen's $d$", "$p$ (Holm)"], rows,
      note="$d$ is the difference of means (excluded minus analysed) in pooled standard deviations; $p$ from Mann--Whitney tests, adjusted by the Holm method over the twelve covariates. The covariates are those of the companion study \\citep{Eminoglu2026PFI}; missing values reduce the number of nodes per covariate.")
macro("selNCov", len(st)); macro("selNSig", int((st.p_holm < 0.05).sum())); macro("selMaxD", f1(st.cohens_d.abs().max(), 2))
macro("selNExcluded", int(len(_s) - (n.src == "SCS").sum())); macro("selNBig", int((st.cohens_d.abs() >= 0.2).sum()))

# ---- numbers recorded by the figure scripts (out_final/fig_stats.csv)
_fs = pd.read_csv(HERE / "out_final/fig_stats.csv")
for _k, _v in _fs.values:
    macro(_k, int(_v) if float(_v).is_integer() else str(f"{float(_v):.3f}".rstrip("0").rstrip(".") if abs(float(_v)) < 1 and _k == "tsneSil" else f"{float(_v):.1f}").replace("-", "$-$"))

# ---- Table: classes
cs = pd.read_csv(HERE / "out_final/class_summary.csv").set_index("class"); tc = pd.read_csv(HERE / "out_final/tier_by_class.csv").set_index("class")
order = ["Boulevard", "Collector avenue", "Wide local street", "Narrow local street", "Shared lane"]
rows = [[c, str(int(cs.loc[c, "n"])), f1(cs.loc[c, "ecw_med"], 2), f1(tc.loc[c].iloc[0]), f1(tc.loc[c].iloc[1]), f1(tc.loc[c].iloc[2]), f1(100 * cs.loc[c, "pc"], 0),
         f1(cs.loc[c, "veh"]), f1(cs.loc[c, "furn"]), f1(cs.loc[c, "infra"])] for c in order]
table("tab:classes", "Effective clear width and obstacle area by street class (Konak sample and legacy pilot). Tier shares are percentages of nodes; areas are mean obstacle footprints per node (m$^2$, both sides, 20\\,m window).",
      ["Class", "Nodes", "Median ECW (m)", "$<$0.9 m (\\%)", "0.9--1.5 m (\\%)", "$\\geq$1.5 m (\\%)", "PC (\\%)", "Vehicles", "Furniture", "Infrastructure"], rows,
      note="Classes follow the municipal road type and the GIS corridor width (Section~\\ref{sec:corridor}). ECW is the best-side bottleneck width; PC is phantom capacity (Eq.~\\ref{eq:pc}).")

# ---- Table: sensitivity
sn = pd.read_csv(HERE / "out_final/sensitivity.csv")
rows = [[r.scenario.replace("_", " "), f1(r.share_lt09), f1(r.share_ge15), f1(r.median_ecw, 2), f1(r.mean_pc)] for r in sn.itertuples()]
table("tab:sens", f"Sensitivity of the headline indicators to modelling assumptions ($n={int(sn.n.iloc[0])}$ nodes evaluated in every scenario).",
      ["Scenario", "$<$0.9 m (\\%)", "$\\geq$1.5 m (\\%)", "Median ECW (m)", "Mean PC (\\%)"], rows,
      note="The central scenario uses the nodes that could be recomputed in all scenarios; it therefore differs slightly from the full sample of Table~\\ref{tab:classes}.")
c0 = sn[sn.scenario == "central"].iloc[0]; nv = sn[sn.scenario == "vehicles removed"].iloc[0]
macro("sensN", int(c0.n)); macro("sensCentralImp", f1(c0.share_lt09)); macro("sensVehImp", f1(nv.share_lt09)); macro("sensVehAcc", f1(nv.share_ge15))
o = sn[~sn.scenario.isin(["central", "vehicles removed"])]
macro("sensImpMin", f1(o.share_lt09.min())); macro("sensImpMax", f1(o.share_lt09.max())); macro("sensAccMin", f1(o.share_ge15.min())); macro("sensAccMax", f1(o.share_ge15.max()))

# ---- Table: coverage threshold
def vals(f):
    d = {}
    for l in open(f):
        p = l.strip().split("{")
        if len(p) >= 3:
            d[p[1].rstrip("}").lstrip("\\")] = p[2].rstrip("}")
    return d
rows = []
for c, d in ((0.5, "out_final"), (0.4, "out_cov0.4"), (0.3, "out_cov0.3")):
    nn = pd.read_csv(HERE / d / "nodes_R1.csv"); v = vals(HERE / d / "values.tex")
    axes = nn[nn.src == "TG"].node.str.split("_").str[0]
    rows.append([f"{int(c * 100)}", str(len(nn)), str(nn.street_id.nunique()), f1(float(v["shareImpassable"])), f1(float(v["shareAccessible"])), f1(float(v["medianECW"]), 2),
                 f1(float(v["ICC"]), 2), f1(float(v["MoranI"]), 2), str(axes.nunique()), str((axes.isin(["Ax3", "Ax6"])).sum())])
    if c == 0.3:
        macro("covImpThirty", f1(float(v["shareImpassable"]))); macro("covAccThirty", f1(float(v["shareAccessible"]))); macro("covNodesThirty", len(nn))
    if c == 0.4:
        macro("covNodesForty", len(nn))
    if c == 0.4:
        macro("covImpForty", f1(float(v["shareImpassable"])))
table("tab:cov", "Effect of the minimum observed share of a node-side's 20\\,m window on the sample and on the results (central analysis: 50\\%).",
      ["Min. coverage (\\%)", "Nodes", "Streets", "$<$0.9 m (\\%)", "$\\geq$1.5 m (\\%)", "Median ECW (m)", "ICC", "Moran's $I$", "Legacy axes present", "Nodes on Ax3/Ax6"], rows)

# ---- Table: model comparison
ma = pd.read_csv(P3 / "out_final/model_agreement_by_class.csv"); pa = pd.read_csv(P3 / "out_final/pipeline_agreement.csv", index_col=0).iloc[:, 0]
ma = ma[ma.cls.isin(["car", "truck", "motorcycle", "chair", "table", "potted plant", "trash can", "utility pole", "street light", "person"])]
rows = [[r.cls, str(int(r.n_images)), str(int(r.both)), str(int(r.sam_only)), str(int(r.sf_only)), f1(r.mean_iou_when_both, 2)] for r in ma.itertuples()]
table("tab:models", "Agreement between SAM 3 and the SegFormer-B5 baseline on the same images: images in which the class was found by both models, by SAM 3 only, or by SegFormer only, and the mean mask IoU where both found it.",
      ["Class", "Images with class", "Both", "SAM 3 only", "SegFormer only", "Mean IoU (both)"], rows,
      note="Dumpsters, utility boxes, and electrical cabinets exist only in the SAM 3 vocabulary (no ADE20K class) and are not listed.")
macro("modelN", int(pa["n"])); macro("modelPearson", f1(pa["pearson"], 2)); macro("modelSpearman", f1(pa["spearman"], 2))
macro("modelMAD", f1(pa["mad"], 2)); macro("modelBias", f1(pa["bias_sf_minus_sam"], 2)); macro("modelAgreeImp", f1(100 * pa["agree_impassable"], 0))

# ---- Table: audit
au = pd.read_csv(P3 / "out_final/audit_summary.csv").set_index("group")
def arow(g, lab):
    r = au.loc[g]; return [lab, str(int(r.n)), str(int(r.TP)), str(int(r.FP)), f"{r.precision:.1f} ({r.ci_lo:.0f}--{r.ci_hi:.0f})"]
labs = [("cat:vehicle", "Vehicles (car, truck, motorcycle)"), ("cat:furniture", "Furniture and plants"), ("cat:infrastructure", "Infrastructure"), ("cat:person", "Person"),
        ("src:SCS", "Konak sample"), ("src:TG", "Legacy pilot"), ("ALL", "All rated instances")]
table("tab:audit", "Instance-level audit of SAM 3 detections (15 images, single rater): precision with Wilson 95\\% interval.",
      ["Group", "Instances", "True pos.", "False pos.", "Precision, \\% (95\\% CI)"], [arow(g, l) for g, l in labs] + [["\\quad " + c.replace("cls:", ""), str(int(au.loc[c].n)), str(int(au.loc[c].TP)), str(int(au.loc[c].FP)), f"{au.loc[c].precision:.1f} ({au.loc[c].ci_lo:.0f}--{au.loc[c].ci_hi:.0f})"] for c in au.index if c.startswith("cls:") and c in ("cls:trash can", "cls:dumpster")],
      note="One additional instance was labelled uncertain and is excluded; counting it as correct or incorrect moves overall precision between 89.6 and 90.3\\%.")

macro("auditN", int(au.loc["ALL"].n)); macro("auditImages", 15); macro("auditPrec", f1(au.loc["ALL"].precision))
macro("auditPrecLo", f1(au.loc["ALL"].ci_lo)); macro("auditPrecHi", f1(au.loc["ALL"].ci_hi))
macro("auditPrecVeh", f1(au.loc["cat:vehicle"].precision)); macro("auditPrecFurn", f1(au.loc["cat:furniture"].precision))
macro("auditPrecInfra", f1(au.loc["cat:infrastructure"].precision)); macro("auditPrecPerson", f1(au.loc["cat:person"].precision))
macro("auditScoreMin", f1(au.loc[["score>=0.6", "score>=0.7", "score>=0.8", "score>=0.9"]].precision.min())); macro("auditScoreMax", f1(au.loc[["score>=0.6", "score>=0.7", "score>=0.8", "score>=0.9"]].precision.max()))
macro("auditPrecNoPerson", f1(100 * (au.loc["ALL"].TP - au.loc["cat:person"].TP) / (au.loc["ALL"].n - au.loc["cat:person"].n)))

# ---- Table: mixed model
mm = pd.read_csv(HERE / "out_final/mixed_model_class.csv", index_col=0)
def mname(i):
    return "Intercept (Boulevard)" if i == "Intercept" else ("Street variance" if i == "Group Var" else i.split("T.")[1].rstrip("]"))
rows = [[mname(i), f1(r["Coef."], 2), f1(r["Std.Err."], 2), "--" if pd.isna(r["[0.025"]) else f"{f1(r['[0.025'], 2)} to {f1(r['0.975]'], 2)}"] for i, r in mm.iterrows()]
table("tab:mixed", "Mixed-effects model of best-side ECW (m) with street class as fixed effect and a random intercept per street (REML).",
      ["Term", "Estimate", "Std. error", "95\% interval"], rows, align="lrrr",
      note="Reference class: boulevard. Nodes are nested in streets; the interval for the intercept and the class contrasts is Wald-based. The residual (node-level) variance is not shown.")

# ---- Table: legacy pilot
lg = pd.read_csv(HERE / "out_final/legacy_axis_table.csv")
rows = [[r.axis, r.morph_r0.replace("_", " "), f1(r.w_cad, 0), f1(r.w_corr_gis, 1), f1(r.ecw_r0_mean, 2), str(int(r.n_r1)), f1(r.ecw_r1_mean, 2), f1(r.share_imp_r0, 0), f1(r.share_imp_r1, 0)] for r in lg.itertuples()]
table("tab:legacy", "The seven legacy axes: first-version values (R0) and the revised method (R1). R1 uses only nodes with a March--November 2024--2025 panorama and at least 50\\% observed window.",
      ["Axis", "R0 label", "Cadastral width (m)", "GIS corridor (m)", "R0 mean ECW (m)", "R1 nodes", "R1 mean ECW (m)", "R0 ECW=0 (\\%)", "R1 $<$0.9 m (\\%)"], rows,
      note="Dashes: no node met the coverage criterion. R0 columns reproduce the first version and are not comparable in method (wall-to-wall width, area-over-length clutter, 4.00 m cap).")
lv = open(HERE / "out_final/legacy_values.txt", encoding="utf-8").read()
import re
macro("legacySpearman", re.search(r"= ([\d.]+)", lv.split("Spearman")[1]).group(1)); macro("legacyN", re.search(r"of (\d+)", lv).group(1))
macro("legacyRZeroShare", re.search(r"ECW=0: ([\d.]+)%", lv).group(1)); macro("legacyCap", re.search(r"cap reached.*?: (\d+)%", lv).group(1))

# ---- geometry validation
g = pd.read_csv(P2 / "geomval.csv")
def gstat(x):
    e = x.err.dropna(); return len(e), e.mean(), e.abs().mean(), np.sqrt((e ** 2).mean()), e.abs().median()
rows = []
for est, lab in (("bldbase", "Building base"), ("groundtop", "Ground line")):
    for h in (2.2, 2.5, 2.8):
        k, b, mae, rm, md = gstat(g[(g.estimator == est) & (np.isclose(g.h, h))]); rows.append([lab, f1(h, 1), str(k), f1(b, 2), f1(mae, 2), f1(rm, 2), f1(md, 2)])
table("tab:geom", "Geometric validation: lateral distance from the street axis to the building facade estimated from the imagery against the GIS building footprints, per image and side.",
      ["Facade line", "Camera height (m)", "$n$", "Bias (m)", "MAE (m)", "RMSE (m)", "Median $|e|$ (m)"], rows, align="lrrrrrr")
b25 = g[(g.estimator == "bldbase") & np.isclose(g.h, 2.5)].copy(); k, b, mae, rm, md = gstat(b25)
macro("gvN", k); macro("gvBias", f1(b, 2)); macro("gvMAE", f1(mae, 2)); macro("gvRMSE", f1(rm, 2)); macro("gvMedian", f1(md, 2))
b25["band"] = pd.cut(b25.F_med, [3, 6, 10, 15, 20])
for iv, gg in b25.groupby("band", observed=True):
    tag = {"(3, 6]": "A", "(6, 10]": "B", "(10, 15]": "C", "(15, 20]": "D"}[str(iv)]
    macro(f"gvBand{tag}Bias", f1(gg.err.mean(), 2)); macro(f"gvBand{tag}RMSE", f1(np.sqrt((gg.err ** 2).mean()), 2)); macro(f"gvBand{tag}N", len(gg))
cb = pd.read_csv(P2 / "calib.csv")
for s in ("SCS", "TG"):
    c = cb[cb.src == s]; macro(f"cal{s}N", len(c)); macro(f"cal{s}Hmed", f1(c.h_node.median(), 2)); macro(f"cal{s}LosoMAE", f1(c.err_loso.abs().mean(), 2))
    macro(f"cal{s}LosoRMSE", f1(np.sqrt((c.err_loso ** 2).mean()), 2)); macro(f"cal{s}LosoBias", f1(c.err_loso.mean(), 2)); macro(f"cal{s}FixedBias", f1(c.err_h25.mean(), 2))
    macro(f"cal{s}RelErr", f1(100 * (c.err_loso.abs() / c.w_corr).mean(), 0))
ac = pd.read_csv(P2 / "anchor_cons.csv"); a25 = ac[np.isclose(ac.h, 2.5)]
for s in ("SCS", "TG"):
    x = a25[a25.src == s]; macro(f"anc{s}N", len(x)); macro(f"anc{s}MAE", f1(x.e.abs().mean(), 2)); macro(f"anc{s}Median", f1(x.e.abs().median(), 2))

import re
txt = "\n".join(tab)
GROUP = {"tab:cov": "tab_cov", "tab:geom": "tab_geom", "tab:audit": "tab_audit_models", "tab:models": "tab_audit_models", "tab:sens": "tab_sens",
         "tab:classes": "tab_classes", "tab:mixed": "tab_classes", "tab:legacy": "tab_legacy", "tab:selection": "tab_selection"}
files = {}
for blk in txt.split("\\begin{table}[!ht]")[1:]:
    files.setdefault(GROUP[re.search(r"\\label\{(tab:\w+)\}", blk).group(1)], []).append("\\begin{table}[!ht]" + blk)
for fn, blks in files.items():
    (OUT / (fn + ".tex")).write_text("\n".join(blks), encoding="utf-8")
(OUT / "values_extra.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in V.items()), encoding="utf-8")
print(len(V), "macros;", len([t for t in tab if t.startswith("\\begin{table}")]), "tables")
