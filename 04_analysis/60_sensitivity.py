"""Scenario sensitivity of the headline indicators (share of nodes with best-side ECW < 0.9 m, mean phantom capacity).

usage: python 60_sensitivity.py <seg_dir> <manifest_run.csv> <out_csv> [max_nodes]
Scenarios: central; lane width -0.5 / +0.5 m; camera height 2.2 / 2.8 m; score threshold 0.5 / 0.7; footprint depth x0.7 / x1.3;
           vehicles removed (upper bound of clearance); persons added as obstacles (lower bound); shared-mode threshold 1.0 / 2.0 m.
"""
import sys, json, importlib.util, numpy as np, pandas as pd
from pathlib import Path

P2 = Path(__file__).resolve().parents[1] / "02_pipeline_v2"
sys.path.insert(0, str(P2))
spec = importlib.util.spec_from_file_location("ecw2", P2 / "30_ecw_v2.py"); ecw2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(ecw2)

seg_dir, manifest, out_csv = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
max_nodes = int(sys.argv[4]) if len(sys.argv) > 4 else 10 ** 9
corr = pd.read_csv(P2 / "corridor_nodes.csv")
corr["node"] = np.where(corr.src == "SCS", "scs_" + corr.id.astype(str).str.replace(r"\.0$", "", regex=True), corr.id)
corr = corr.set_index("node")
man = pd.read_csv(manifest)

SC = {
    "central": {},
    "lane_w -0.5 m": dict(dlane=-0.5), "lane_w +0.5 m": dict(dlane=+0.5),
    "camera h 2.2 m": dict(prm=dict(h=2.2)), "camera h 2.8 m": dict(prm=dict(h=2.8)),
    "score thr 0.5": dict(prm=dict(thr=0.5)), "score thr 0.7": dict(prm=dict(thr=0.7)),
    "depth prior x0.7": dict(prm=dict(depth_scale=0.7)), "depth prior x1.3": dict(prm=dict(depth_scale=1.3)),
    "vehicles removed": dict(drop={"car", "truck", "motorcycle"}),
    "persons added": dict(persons=True),
    "shared threshold 1.0 m": dict(shared=1.0), "shared threshold 2.0 m": dict(shared=2.0),
}
rows = []; n = 0
for node_id, g in man.groupby("node", sort=False):
    if node_id not in corr.index:
        continue
    node = corr.loc[node_id]
    if pd.isna(node.w_corridor) or pd.isna(node.axis_azimuth):
        continue
    imgs = []
    for r in g.itertuples():
        jf = seg_dir / (r.key + ".json")
        if jf.exists():
            dd = json.load(open(jf))["dets"]; nz = np.load(seg_dir / (r.key + ".npz"))
            imgs.append(dict(heading=r.heading, pitch=r.pitch, dets=dd, npz=nz, uv=ecw2.anchor.facade_uv(dd, nz)))
    if not imgs:
        continue
    n += 1
    if n > max_nodes:
        break
    for name, sc in SC.items():
        nd = node.copy()
        if "dlane" in sc:
            nd["w_carriage"] = max(0.0, node.lanes * (node.lane_w + sc["dlane"]))
        ecw2.SHARED_MAX = sc.get("shared", 1.5)
        prm = dict(ecw2.CENTRAL); prm.update(sc.get("prm", {}))
        use = imgs
        if "drop" in sc:
            use = [dict(i, dets=[d for d in i["dets"] if d["prompt"] not in sc["drop"]]) for i in imgs]
        best = []; caps = []
        for side in "LR":
            e = ecw2.evaluate(nd, side, use, prm, with_persons=sc.get("persons", False))
            if e and e["coverage"] >= 0.5 and not np.isnan(e["ecw_min"]):
                best.append(e["ecw_min"]); wn = min(e["zone_w"], 3.0); caps.append(1 - min(e["ecw_min"], wn) / wn)
        if best:
            rows.append(dict(node=node_id, scenario=name, ecw_best=max(best), pc=np.mean(caps)))
ecw2.SHARED_MAX = 1.5
df = pd.DataFrame(rows); df.to_csv(out_csv.replace(".csv", "_nodes.csv"), index=False)
common = df.groupby("node").scenario.nunique()
df = df[df.node.isin(common[common == len(SC)].index)]
summ = df.groupby("scenario").agg(n=("node", "size"), share_lt09=("ecw_best", lambda x: (x < 0.9).mean() * 100),
                                    share_ge15=("ecw_best", lambda x: (x >= 1.5).mean() * 100), median_ecw=("ecw_best", "median"),
                                    mean_pc=("pc", lambda x: x.mean() * 100)).reindex(list(SC)).round(2)
summ.to_csv(out_csv); print(summ.to_string())
