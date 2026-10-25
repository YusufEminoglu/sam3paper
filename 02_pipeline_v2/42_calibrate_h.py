"""Per-node camera-height (scale) calibration from the two opposite facades + cross-validated width reconstruction.

All ground coordinates are proportional to the camera height h.  A node imaged toward both facades gives
W_meas(h0) = r_left + r_right (m, at h0 = 2.5);  the GIS building-to-building width W_corr then implies h_node = 2.5 * W_corr / W_meas.
Validation (no manual data): for every node the width is re-predicted with the MEDIAN h of all OTHER streets (leave-one-street-out)
and compared with the GIS width -> out-of-sample error of the pixel-to-metre conversion over the full street cross-section.
usage: python 42_calibrate_h.py seg_dir manifest out_csv
"""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
import anchor

seg, man, out = Path(sys.argv[1]), pd.read_csv(sys.argv[2]), sys.argv[3]
corr = pd.read_csv("corridor_nodes.csv")
corr["node"] = np.where(corr.src == "SCS", "scs_" + corr.id.astype(str).str.replace(r"\.0$", "", regex=True), corr.id)
corr = corr.set_index("node")
rows = []
for node_id, g in man.groupby("node", sort=False):
    if node_id not in corr.index:
        continue
    n = corr.loc[node_id]
    if pd.isna(n.w_corridor) or pd.isna(n.axis_azimuth):
        continue
    rs = []
    for r in g.itertuples():
        jf = seg / (r.key + ".json")
        if not jf.exists():
            continue
        uv = anchor.facade_uv(json.load(open(jf))["dets"], np.load(seg / (r.key + ".npz")))
        v = anchor.facade_r(uv, n, r.heading, r.pitch, 2.5)
        if v is not None:
            rs.append(v)
    L = [x for x in rs if x > 0]; R = [-x for x in rs if x < 0]
    if L and R:
        wm = np.median(L) + np.median(R)
        rows.append(dict(node=node_id, src=n.src, street_id=n.street_id, w_corr=n.w_corridor, w_meas25=wm, h_node=2.5 * n.w_corridor / wm))
d = pd.DataFrame(rows)
d = d[(d.h_node > 1.0) & (d.h_node < 4.5)]
# leave-one-street-out
pred = []
for i, r in d.iterrows():
    oth = d[(d.src == r.src) & (d.street_id != r.street_id)]
    hm = oth.h_node.median() if len(oth) >= 5 else d[d.street_id != r.street_id].h_node.median()
    pred.append(r.w_meas25 * hm / 2.5)
d["w_pred_loso"] = pred; d["err_loso"] = d.w_pred_loso - d.w_corr
d["w_pred_h25"] = d.w_meas25; d["err_h25"] = d.w_pred_h25 - d.w_corr
d.to_csv(out, index=False)
for src, g in d.groupby("src"):
    print(f"{src}: n={len(g)} nodes, {g.street_id.nunique()} streets | h_node median {g.h_node.median():.2f} m (IQR {g.h_node.quantile(.25):.2f}-{g.h_node.quantile(.75):.2f})"
          f" | fixed h=2.5: bias {g.err_h25.mean():+.2f} MAE {g.err_h25.abs().mean():.2f} m | LOSO: bias {g.err_loso.mean():+.2f} MAE {g.err_loso.abs().mean():.2f} RMSE {np.sqrt((g.err_loso**2).mean()):.2f} m,"
          f" mean rel. err {(g.err_loso.abs()/g.w_corr).mean()*100:.0f}%")
