"""Values of the quality-control pass (2026-10-26), computed from the pipeline outputs only.

usage: python 83_qc_values.py      (run from anywhere)
reads  out_final/nodes_R1.csv, out_final/sensitivity*.csv, ../02_pipeline_v2/{ecw_v2,corridor_nodes,manifest_run}.csv
writes ../05_manuscript_R1/sections/values_qc.tex   (macros; read by build_main_R1.py and the response letter)
       ../05_manuscript_R1/sections/tab_sens.tex    (Table: sensitivity)

Why a separate script: the sensitivity run now starts from the same per-node calibrated camera height as the headline run
(60_sensitivity.py), so its central scenario, the vehicle comparison and the sensitivity table are recomputed here; the
remaining definitions follow the text of the manuscript (side-level coverage, the Monte Carlo interval used for the tier
robustness, Eq. 5 without the 3.0 m reference width, structural impassability on the wider analysed side).
"""
import numpy as np, pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
P2 = HERE.parent / "02_pipeline_v2"
OUT = HERE.parent / "05_manuscript_R1" / "sections"
OUT.mkdir(parents=True, exist_ok=True)
V = {}
f1 = lambda x, d=1: f"{x:.{d}f}"
half_up = lambda x, d=1: f"{np.floor(x * 10 ** d + 0.5) / 10 ** d:.{d}f}"      # conventional rounding, not round-half-even
macro = lambda k, v: V.__setitem__(k, v)

n = pd.read_csv(HERE / "out_final/nodes_R1.csv")
e = pd.read_csv(P2 / "ecw_v2.csv")
ea = e[(e.coverage >= 0.5) & e.ecw_min.notna()].copy()
ea = ea[ea.node.isin(n.node)]
assert ea.node.nunique() == len(n) == 557, (ea.node.nunique(), len(n))

# ---- coverage: mean observed share of the analysed node-sides (slices with enough observed ground)
macro("meanCoverage", half_up(100 * ea.coverage.mean(), 0))
macro("nSidesCheck", str(len(ea)))

# ---- sample flow: Konak candidates with a corridor width AND imagery
c = pd.read_csv(P2 / "corridor_nodes.csv"); m = pd.read_csv(P2 / "manifest_run.csv")
c["node"] = np.where(c.src == "SCS", "scs_" + c.id.astype(str).str.replace(r"\.0$", "", regex=True), c.id)
s = c[c.src == "SCS"]
img = set(m[m.src == "SCS"].node)
macro("nSCSImagery", str(int((s.w_corridor.notna() & s.node.isin(img)).sum())))

# ---- Monte Carlo interval used for tier robustness (Figure 5): larger of the two side bounds, widened to the central value
g = ea.groupby("node").agg(lo=("ecw_min_mc_p05", "max"), hi=("ecw_min_mc_p95", "max")).reindex(n.node).reset_index()
nn = n.merge(g, on="node")
lo = np.minimum(nn.lo.fillna(nn.ecw_best), nn.ecw_best); hi = np.maximum(nn.hi.fillna(nn.ecw_best), nn.ecw_best)
macro("medianUncWidth", f1((hi - lo).median(), 2))

# ---- Eq. 5 without the 3.0 m reference width (W_nom = W_p), node value weighted by W_nom as in the central PC
ea["pc_nocap"] = 1 - np.minimum(ea.ecw_min, ea.zone_w) / ea.zone_w
pcn = ea.groupby("node").apply(lambda x: np.average(x.pc_nocap, weights=x.zone_w), include_groups=False)
macro("meanPCnocap", f1(100 * pcn.mean()))

# ---- structural impassability: the wider analysed side has a pedestrian-intended zone below 0.9 m
zmax = ea.groupby("node").zone_w.max().reindex(n.node).values
imp = n.ecw_best.values < 0.9
st = imp & (zmax < 0.9)
macro("robImpAllCheck", str(int(imp.sum())))
macro("structImpN", str(int(st.sum()))); macro("structImpPct", f1(100 * st.sum() / imp.sum())); macro("obstrImpN", str(int(imp.sum() - st.sum())))
import sys; sys.path.insert(0, str(HERE)); from sigfig_common import load      # dominant group exactly as in the alluvial figure
L = load().set_index("node").reindex(n.node)
nz = (L.dominant.values == "None") & imp
macro("sankNoObsImpN", str(int(nz.sum()))); macro("sankNoObsImpNarrow", str(int((nz & (zmax < 0.9)).sum())))
for k, lab in (("Vehicles", "Veh"), ("Furniture", "Furn"), ("Plants", "Plant"), ("Infrastructure", "Infra"), ("None", "NoObs")):
    macro(f"dom{lab}PctCheck", f1(100 * (L.dominant.values == k).mean()))
macro("sankPlantDomPct", f1(100 * (L.dominant.values == "Plants").mean()))

# ---- sensitivity (central = headline model on the nodes evaluated in every scenario)
sn = pd.read_csv(HERE / "out_final/sensitivity.csv")
sd = pd.read_csv(HERE / "out_final/sensitivity_nodes.csv")
cen = sd[sd.scenario == "central"].set_index("node").ecw_best
both = cen.index.intersection(n.set_index("node").index)
same = np.isclose(cen.loc[both].values, n.set_index("node").ecw_best.loc[both].values)
macro("sensCentralMatchPct", f1(100 * same.mean()))
c0 = sn[sn.scenario == "central"].iloc[0]; vr = sn[sn.scenario == "vehicles removed"].iloc[0]; pa = sn[sn.scenario == "persons added"].iloc[0]
macro("sensN", str(int(c0.n))); macro("sensCentralImp", half_up(c0.share_lt09)); macro("sensVehImp", half_up(vr.share_lt09)); macro("sensVehAcc", half_up(vr.share_ge15))
macro("sensPersonsImp", half_up(pa.share_lt09))
o = sn[~sn.scenario.isin(["central", "vehicles removed"])]
macro("sensImpMin", half_up(o.share_lt09.min())); macro("sensImpMax", half_up(o.share_lt09.max()))
macro("sensAccMin", half_up(o.share_ge15.min())); macro("sensAccMax", half_up(o.share_ge15.max()))

LAB = {"central": "Central", "lane_w -0.5 m": "Lane width $-$0.5\\,m", "lane_w +0.5 m": "Lane width $+$0.5\\,m",
       "camera h x0.88": "Camera height $\\times$0.88", "camera h x1.12": "Camera height $\\times$1.12",
       "score thr 0.5": "Score threshold 0.5", "score thr 0.7": "Score threshold 0.7",
       "depth prior x0.7": "Depth priors $\\times$0.7", "depth prior x1.3": "Depth priors $\\times$1.3",
       "vehicles removed": "Vehicles removed", "persons added": "Persons added as obstacles",
       "shared threshold 1.0 m": "Shared-lane threshold 1.0\\,m", "shared threshold 2.0 m": "Shared-lane threshold 2.0\\,m"}
rows = [f"{LAB[r.scenario]} & {half_up(r.share_lt09)} & {half_up(r.share_ge15)} & {r.median_ecw:.2f} & {half_up(r.mean_pc)} \\\\" for r in sn.itertuples()]
tab = ["\\begin{table}[!ht]", "\\centering", "\\small",
       f"\\caption{{Sensitivity of the headline indicators to modelling assumptions ($n={int(c0.n)}$ nodes evaluated in every scenario).}}",
       "\\label{tab:sens}", "\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%", "\\begin{tabular}{lrrrr}", "\\toprule",
       "Scenario & $<$0.9 m (\\%) & $\\geq$1.5 m (\\%) & Median ECW (m) & Mean PC (\\%) \\\\", "\\midrule"] + rows + ["\\bottomrule", "\\end{tabular}}",
       "\\par\\smallskip\\parbox{0.95\\linewidth}{\\footnotesize Every scenario starts from the per-node calibrated camera height of the central analysis; the camera-height scenarios scale it by 0.88 and 1.12. The scenarios use the nodes that could be recomputed in all of them, so the central row differs slightly from the full sample of Table~\\ref{tab:classes}.}",
       "\\end{table}", ""]
(OUT / "tab_sens.tex").write_text("\n".join(tab), encoding="utf-8")

# ---- vehicle comparison: central against vehicles removed, node by node
cl = n.set_index("node")["class"]
p = sd.pivot_table(index="node", columns="scenario", values="ecw_best")
d = p[["central", "vehicles removed"]].dropna()
d = d[d.index.isin(n.node)]
tier = lambda x: np.where(x < 0.9, "I", np.where(x < 1.5, "R", "A"))
to, tf = tier(d["central"].values), tier(d["vehicles removed"].values)
k = d.assign(to=to, tf=tf, cls=cl.reindex(d.index).values)
ni = int((k.to == "I").sum()); dep = k[(k.to == "I") & (k.tf != "I")]; stay = int(((k.to == "I") & (k.tf == "I")).sum())
macro("cfN", str(len(k))); macro("cfImpN", str(ni)); macro("cfDepN", str(len(dep))); macro("cfDepPct", f1(100 * len(dep) / ni))
macro("cfStructN", str(stay)); macro("cfStructPct", f1(100 * stay / ni)); macro("cfRestrN", str(int(((k.to == "R") & (k.tf == "A")).sum())))
macro("cfImpAfterN", str(int((k.tf == "I").sum()))); macro("cfImpAfterPct", f1(100 * (k.tf == "I").mean())); macro("cfImpBeforePct", f1(100 * (k.to == "I").mean()))
for key, c_ in (("cfVehNarrow", "Narrow local street"), ("cfVehWide", "Wide local street"), ("cfVehShared", "Shared lane"), ("cfVehColl", "Collector avenue"), ("cfVehBlvd", "Boulevard")):
    macro(key, str(int((dep.cls == c_).sum())))
for key, c_ in (("cfImpShared", "Shared lane"), ("cfImpWide", "Wide local street"), ("cfImpNarrow", "Narrow local street"), ("cfImpColl", "Collector avenue")):
    macro(key, str(int(((k.to == "I") & (k.cls == c_)).sum())))

# ---- worked example (Figure 4): the pipeline value of the illustrated node-side (legacy node Ax2_40, left side)
_x = e[(e.node == "Ax2_40") & (e.side == "L")].iloc[0]
macro("exSideZone", f1(_x.zone_w, 2)); macro("exSideEcw", f1(_x.ecw_min, 2))

lines = ["% Quality-control values (83_qc_values.py); every number is computed from the pipeline outputs."]
lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in V.items()]
(OUT / "values_qc.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
for k, v in V.items():
    print(f"{k:22s} {v}")
