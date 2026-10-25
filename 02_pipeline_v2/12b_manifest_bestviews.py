"""Manifest v2: for each SCS Konak node keep the 2 cardinal views that best cover the pedestrian zones (geometric coverage)."""
import pandas as pd, numpy as np, os, itertools
cov = pd.read_csv("view_coverage.csv")
SCS_IMG = os.environ.get("PC_SCS_IMAGES", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "external", "gsv"))
HEAD = {"N": 0, "E": 90, "S": 180, "W": 270}
rows = []
for node, g in cov.groupby("node"):
    piv = g.pivot_table(index="side", columns="view", values="vis", aggfunc="max").reindex(columns=list("NESW")).fillna(0)
    avail = [d for d in "NESW" if os.path.exists(os.path.join(SCS_IMG, f"node{int(node)}_{d}.jpg"))]
    if not avail: continue
    best = max(itertools.combinations(avail, min(2, len(avail))),
               key=lambda pr: sum(min(1.0, piv.loc[s, list(pr)].sum()) for s in piv.index))
    for d in best:
        rows.append(dict(key=f"scs_{int(node)}_{d}", path=os.path.join(SCS_IMG, f"node{int(node)}_{d}.jpg"), node=f"scs_{int(node)}",
                         side="", heading=HEAD[d], pitch=-10.0, src="SCS"))
m = pd.DataFrame(rows)
tg = pd.read_csv("manifest_TG.csv")
out = pd.concat([tg, m], ignore_index=True)
out.to_csv("manifest_run.csv", index=False)
print("TG", len(tg), "SCS images", len(m), "SCS nodes", m.node.nunique())
