"""Independent consistency check of the ground-plane geometry.

Two views of the same node come from the same camera position, so the lateral camera offset estimated from a facade on the left
must agree with the one estimated from a facade on the right (they use different images, different facades and different
GIS distances).  The disagreement  e = delta_left_image - delta_right_image  is a direct measure of the cross-street error of the
pixel-to-metre conversion (scale, pitch, height) over distances of ~5-15 m.  No manual annotation involved.
usage: python 41_anchor_consistency.py seg_dir manifest out_csv
"""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
import anchor, geom

seg, man, out = Path(sys.argv[1]), pd.read_csv(sys.argv[2]), sys.argv[3]
corr = pd.read_csv("corridor_nodes.csv")
corr["node"] = np.where(corr.src == "SCS", "scs_" + corr.id.astype(str).str.replace(r"\.0$", "", regex=True), corr.id)
corr = corr.set_index("node")
rows = []
for node_id, g in man.groupby("node", sort=False):
    if node_id not in corr.index or len(g) < 2:
        continue
    n = corr.loc[node_id]
    if pd.isna(n.d_left) or pd.isna(n.d_right) or pd.isna(n.axis_azimuth):
        continue
    ds = []
    for r in g.itertuples():
        jf = seg / (r.key + ".json")
        if not jf.exists():
            continue
        dets = json.load(open(jf))["dets"]; nz = np.load(seg / (r.key + ".npz"))
        uv = anchor.facade_uv(dets, nz)
        for h in (2.5, 2.2, 2.8):
            d = anchor.estimate_delta(uv, n, r.heading, r.pitch, h)
            if d is None:
                continue
            R, F, ok = geom.pixels_to_ground(uv[:, 0], uv[:, 1], h, r.pitch); ok &= (F > 3) & (F < 25)
            E, N = geom.cam_to_world(R[ok], F[ok], r.heading, n.x_node, n.y_node)
            s, l = geom.world_to_axis(E, N, n.x_snap, n.y_snap, n.axis_azimuth)
            ds.append(dict(h=h, delta=d, side=np.sign(np.median(l)), F=float(np.median(F[ok])), key=r.key))
    df = pd.DataFrame(ds)
    if df.empty:
        continue
    for h, gg in df.groupby("h"):
        L = gg[gg.side > 0]; R_ = gg[gg.side < 0]
        if len(L) and len(R_):
            rows.append(dict(node=node_id, src=n.src, h=h, delta_L=L.delta.iloc[0], delta_R=R_.delta.iloc[0], e=L.delta.iloc[0] - R_.delta.iloc[0],
                             F_L=L.F.iloc[0], F_R=R_.F.iloc[0], w_corridor=n.w_corridor))
d = pd.DataFrame(rows); d.to_csv(out, index=False)
for (src, h), g in d.groupby(["src", "h"]):
    print(f"{src} h={h}: n={len(g)}  bias={g.e.mean():+.2f}  MAE={g.e.abs().mean():.2f}  RMSE={np.sqrt((g.e**2).mean()):.2f}  median|e|={g.e.abs().median():.2f}  corr(e,w)={g.e.corr(g.w_corridor):.2f}")
