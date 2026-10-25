"""Selection test: do the Konak candidate nodes that entered the analysis differ from those that did not, on covariates that do not come from street-view imagery?

Covariates: node-level urbanity, terrain, and thermal metrics of the companion study (pfi_merged_full_imputed.csv; the covariates used are not image-derived and are available for every generated node).
Groups: analysed (in nodes_R1.csv) vs excluded; excluded nodes are also split by the first filter that removed them.
Output: out_final/selection_test.csv (per covariate: medians, Cohen's d, Mann-Whitney p, Holm-adjusted p), out_final/selection_groups.csv
usage: python 82_selection_test.py
"""
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
import os

HERE = Path(__file__).parent
P2 = HERE.parent / "02_pipeline_v2"
COV = Path(os.environ.get("PC_COVARIATES", HERE.parent / "data/external/companion_covariates.csv"))
VARS = ["BC", "JCS", "MDI", "SCR", "FAR", "H_std", "DPark", "SLOPE", "GAR", "TCH_mean_20m", "LST_norm", "Albedo_norm"]

c = pd.read_csv(P2 / "corridor_nodes.csv"); c = c[c.src == "SCS"].copy()
c["node_id"] = c.id.astype(int)
man = pd.read_csv(P2 / "manifest_run.csv"); man = man[man.src == "SCS"]
imaged = set(man.node.str.replace("scs_", "", regex=False).astype(int))
nodes = pd.read_csv(HERE / "out_final/nodes_R1.csv"); analysed = set(nodes[nodes.src == "SCS"].node.str.replace("scs_", "", regex=False).astype(int))
def status(r):
    if r.node_id in analysed: return "analysed"
    if not r.snapped: return "excluded: not snapped"
    if pd.isna(r.w_corridor): return "excluded: no corridor width"
    if r.node_id not in imaged: return "excluded: no qualifying panorama"
    return "excluded: coverage"
c["status"] = c.apply(status, axis=1)
cov = pd.read_csv(COV)[["node_id"] + VARS]
d = c[["node_id", "status"]].merge(cov, on="node_id", how="left")
print(d.status.value_counts().to_string())
d.groupby("status").size().rename("n").to_csv(HERE / "out_final/selection_groups.csv")

a = d[d.status == "analysed"]; x = d[d.status != "analysed"]
rows = []
for v in VARS:
    u, w = a[v].dropna(), x[v].dropna()
    sp = np.sqrt(((len(u) - 1) * u.var() + (len(w) - 1) * w.var()) / (len(u) + len(w) - 2))
    rows.append(dict(covariate=v, n_analysed=len(u), n_excluded=len(w), median_analysed=u.median(), median_excluded=w.median(),
                     cohens_d=(w.mean() - u.mean()) / sp, mw_p=stats.mannwhitneyu(u, w).pvalue))
r = pd.DataFrame(rows)
o = r.mw_p.argsort().values; m = len(r); adj = np.empty(m); run = 0
for k, i in enumerate(o):                       # Holm step-down
    run = max(run, (m - k) * r.mw_p.iloc[i]); adj[i] = min(1, run)
r["p_holm"] = adj
r.round(4).to_csv(HERE / "out_final/selection_test.csv", index=False)
print(r.round(3).to_string(index=False))
print("max |d| =", r.cohens_d.abs().max().round(3), "| covariates with p_holm<0.05:", int((r.p_holm < 0.05).sum()), "| |d|>=0.2:", int((r.cohens_d.abs() >= 0.2).sum()))
