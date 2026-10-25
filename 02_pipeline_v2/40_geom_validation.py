"""Independent geometric validation of the pixel -> ground-plane conversion.

Reference: the lateral distance from the street axis to the building facade, taken from the municipal GIS building footprints
(YAPI), i.e. the SAME quantity d_left / d_right that bounds the pedestrian zone but obtained from a completely different source.
Estimate : SAM 3 'building' mask -> lowest pixel of every column where the building meets the ground (pixel just below the base
           belongs to the sidewalk/road masks, not to an obstacle/person) -> ground projection (h = 2.5 m, pitch from the image
           metadata) -> lateral offset in the axis frame.
Outputs  : per-image error table and summary (bias, MAE, RMSE) by forward-distance band, plus the sensitivity to camera height.
Nothing here uses manual annotation; the reference is GIS data.
"""
import json, sys, numpy as np, pandas as pd
from pathlib import Path
import geom

HERE = Path(__file__).parent
ALLOWED_BELOW = {"sidewalk", "road"}
BLOCK = {"car", "truck", "motorcycle", "person", "chair", "table", "potted plant", "trash can", "dumpster", "utility pole",
         "street light", "utility box", "electrical cabinet"}


def load_masks(seg_dir, key):
    npz = np.load(Path(seg_dir) / (key + ".npz"))
    j = json.load(open(Path(seg_dir) / (key + ".json")))
    shape = tuple(npz["shape"])
    masks = [np.unpackbits(npz["masks"][i])[: shape[0] * shape[1]].reshape(shape).astype(bool) for i in range(len(j["dets"]))] if len(j["dets"]) else []
    return j["dets"], masks, shape


def facade_estimates(dets, masks, shape, thr=0.6):
    bld = np.zeros(shape, bool); ground = np.zeros(shape, bool); block = np.zeros(shape, bool)
    for d, m in zip(dets, masks):
        if d["score"] < thr:
            continue
        if d["prompt"] == "building":
            bld |= m
        elif d["prompt"] in ALLOWED_BELOW:
            ground |= m
        elif d["prompt"] in BLOCK:
            block |= m
    cols = np.where(bld.any(0))[0][::3]
    pts = []
    for c in cols:
        v = np.where(bld[:, c])[0].max()
        if v > shape[0] - 8:
            continue
        below = slice(v + 2, v + 7)
        if ground[below, c].mean() > 0.25 and not block[max(v - 2, 0):v + 4, c].any():
            pts.append((c + 0.5, v + 1.0))
    return np.array(pts)


def facade_estimates_ground(dets, masks, shape, thr=0.6):
    """Facade line from the ground (sidewalk/road) mask: first ground pixel at/below the building base (a true ground point)."""
    bld = np.zeros(shape, bool); ground = np.zeros(shape, bool); block = np.zeros(shape, bool)
    for d, m in zip(dets, masks):
        if d["score"] < thr:
            continue
        if d["prompt"] == "building":
            bld |= m
        elif d["prompt"] in ALLOWED_BELOW:
            ground |= m
        elif d["prompt"] in BLOCK:
            block |= m
    pts = []
    for c in np.where(bld.any(0))[0][::3]:
        vb = np.where(bld[:, c])[0].max()
        rng_ = np.where(ground[max(vb - 5, 0):min(vb + 60, shape[0] - 12), c])[0]
        if len(rng_) == 0:
            continue
        vg = max(vb - 5, 0) + rng_.min()
        if ground[vg:vg + 10, c].mean() > 0.7 and not block[max(vb - 2, 0):vg + 6, c].any():
            pts.append((c + 0.5, vg + 0.0))
    return np.array(pts)


def main(seg_dir, manifest, out_csv, h=2.5):
    corr = pd.read_csv(HERE / "corridor_nodes.csv")
    corr["node"] = np.where(corr.src == "SCS", "scs_" + corr.id.astype(str).str.replace(r"\.0$", "", regex=True), corr.id)
    corr = corr.set_index("node")
    man = pd.read_csv(manifest)
    rows = []
    for r in man.itertuples():
        if not (Path(seg_dir) / (r.key + ".json")).exists() or r.node not in corr.index:
            continue
        n = corr.loc[r.node]
        if pd.isna(n.d_left) or pd.isna(n.axis_azimuth):
            continue
        dets, masks, shape = load_masks(seg_dir, r.key)
        for est_name, uv in (('bldbase', facade_estimates(dets, masks, shape)), ('groundtop', facade_estimates_ground(dets, masks, shape))):
          if len(uv) < 8:
            continue
          for hh in (h, 2.2, 2.8):
              R, F, ok = geom.pixels_to_ground(uv[:, 0], uv[:, 1], hh, r.pitch)
              ok &= (F > 3) & (F < 20)
              if ok.sum() < 8:
                  continue
              E, N = geom.cam_to_world(R[ok], F[ok], r.heading, n.x_node, n.y_node)
              s, l = geom.world_to_axis(E, N, n.x_snap, n.y_snap, n.axis_azimuth)
              side_left = np.median(l) > 0
              est = float(np.median(np.abs(l)))
              ref = float(n.d_left if side_left else n.d_right)
              rows.append(dict(estimator=est_name, key=r.key, node=r.node, src=r.src, h=hh, side="L" if side_left else "R", n_pts=int(ok.sum()),
                               F_med=float(np.median(F[ok])), est=est, ref=ref, err=est - ref))
    df = pd.DataFrame(rows).dropna(subset=["ref"])
    df.to_csv(out_csv, index=False)
    for (en, hh), g in df.groupby(["estimator", "h"]):
        e = g.err.values
        print(f"{en} h={hh}: n={len(g)}  bias={e.mean():+.2f} m  MAE={np.abs(e).mean():.2f} m  RMSE={np.sqrt((e**2).mean()):.2f} m  median|err|={np.median(np.abs(e)):.2f}")
    g = df[(df.h == h) & (df.estimator == 'groundtop')]
    if len(g):
        g = g.assign(band=pd.cut(g.F_med, [3, 6, 10, 15, 20]))
        print(g.groupby("band", observed=True).err.agg(["count", "mean", lambda x: np.abs(x).mean()]).round(2).to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
