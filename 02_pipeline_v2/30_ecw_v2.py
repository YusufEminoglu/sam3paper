"""Phantom capacity v2: SAM 3 masks -> ground-plane footprints -> continuous clearance in the pedestrian-intended zone.

Node-level multi-view fusion.  For every node and every street side (left/right of the axis direction):
  zone     : pedestrian-intended lateral interval from the GIS corridor
             left  sidewalk  l in [ +Wc/2 , +dL ]     right sidewalk  l in [ -dR , -Wc/2 ]     (Wc = lanes x lane width)
             'shared' mode (W_corridor - Wc < SHARED_MAX): whole half-corridor [0,dL] / [-dR,0]
  window   : along-axis window s in [-10, +10] m around the node
  blocked  : union over all views of the obstacle footprints (visible ground-contact line of each mask, extended away
             from the camera by a class depth prior).  Vehicles in the carriageway are outside the zone and are ignored.
  visible  : union over views of the ground cells that project inside the image (F in [F_MIN, F_MAX])
  slices   : only along-axis slices in which >= VIS_MIN of the zone width is visible are evaluated
  outputs  : ECW_min (bottleneck = min over slices of the longest contiguous free lateral run),
             ECW_avg (mean total free width), zone width, coverage, per-class blocked area, person footprint (separate demand variable)
Uncertainty: Monte-Carlo over camera height, pitch, FOV, depth priors, camera lateral offset and score threshold.
"""
import json, sys, numpy as np, pandas as pd, shapely
from pathlib import Path
from shapely.geometry import MultiPoint
import geom, anchor

HERE = Path(__file__).parent
OBST = ["car", "truck", "motorcycle", "chair", "table", "potted plant", "trash can", "dumpster", "utility pole",
        "street light", "utility box", "electrical cabinet"]
SHARED_MAX = 1.5
DS, DL = 0.10, 0.05
F_MIN, F_MAX = 1.5, 25.0
HALF_WIN = 10.0
VIS_MIN, MIN_SLICES = 0.5, 10
CENTRAL = dict(h=2.5, dpitch=0.0, dfov=0.0, depth_scale=1.0, cam_lat=0.0, thr=0.6)


def unpack(npz, i, shape):
    return np.unpackbits(npz["masks"][i])[: shape[0] * shape[1]].reshape(shape).astype(bool)


def axis_vecs(node):
    az = np.radians(node.axis_azimuth)
    return (np.sin(az), np.cos(az)), (-np.cos(az), np.sin(az))          # (along E,N), (left E,N)


def cam_from_off(node, off):
    _, (le, ln) = axis_vecs(node)
    return node.x_node + off * le, node.y_node + off * ln


def image_offset(img, node, prm):
    """Camera lateral offset (m, axis-left): facade-anchored estimate + residual uncertainty (prm['cam_lat'] is a z-score)."""
    d = anchor.estimate_delta(img.get("uv"), node, img["heading"], img["pitch"], prm["h"], prm["dpitch"], prm["dfov"])
    if d is None:
        return prm["cam_lat"] * anchor.SIGMA_UN, False
    return d + prm["cam_lat"] * anchor.SIGMA_AN, True


def axis_to_world(S, L, node):
    (te, tn), (le, ln) = axis_vecs(node)
    return node.x_snap + S * te + L * le, node.y_snap + S * tn + L * ln


def polys_for_image(img, node, prm, classes):
    dets, npz, heading, pitch = img["dets"], img["npz"], img["heading"], img["pitch"]
    shape = tuple(npz["shape"])
    off, _ = image_offset(img, node, prm)
    ce, cn = cam_from_off(node, off)
    th = np.radians(heading)
    fs, fl = geom.world_to_axis(np.array([node.x_snap + np.sin(th)]), np.array([node.y_snap + np.cos(th)]), node.x_snap, node.y_snap, node.axis_azimuth)
    fs, fl = float(fs[0]), float(fl[0])
    out = []
    for i, d in enumerate(dets):
        if d["prompt"] not in classes or d["score"] < prm["thr"]:
            continue
        uv = geom.base_contact_points(unpack(npz, i, shape))
        if len(uv) < 2:
            continue
        R, F, ok = geom.pixels_to_ground(uv[:, 0], uv[:, 1], prm["h"], pitch + prm["dpitch"], fov=90.0 + prm["dfov"])
        ok &= (F > F_MIN) & (F < F_MAX)
        if ok.sum() < 2:
            continue
        E, N = geom.cam_to_world(R[ok], F[ok], heading, ce, cn)
        s, l = geom.world_to_axis(E, N, node.x_snap, node.y_snap, node.axis_azimuth)
        dep = (geom.DEPTH_PRIOR.get(d["prompt"], 0.6) * prm["depth_scale"]) if d["prompt"] != "person" else 0.5
        pts = np.vstack([np.stack([s, l], 1), np.stack([s + fs * dep, l + fl * dep], 1)])
        out.append((d["prompt"], MultiPoint(pts).convex_hull))
    return out


def zone_bounds(node, side):
    Wc = node.w_carriage
    shared = (node.w_corridor - Wc) < SHARED_MAX
    if side == "L":
        return (0.0 if shared else Wc / 2.0), node.d_left, shared
    return -node.d_right, (0.0 if shared else -Wc / 2.0), shared


def _grid(lo, hi):
    ss = np.arange(-HALF_WIN, HALF_WIN + 1e-9, DS)
    ll = np.arange(lo + DL / 2, hi, DL)
    return np.meshgrid(ss, ll, indexing="ij")


def evaluate(node, side, images, prm=CENTRAL, with_persons=False):
    lo, hi, shared = zone_bounds(node, side)
    if hi - lo < 0.3:
        return None
    S, L = _grid(lo, hi)
    E, N = axis_to_world(S, L, node)
    vis = np.zeros(S.shape, bool)
    n_anch = 0
    blocked = np.zeros(S.shape, bool)
    classes = set(OBST) | ({"person"} if with_persons else set())
    for img in images:
        th = np.radians(img["heading"])
        off, anch = image_offset(img, node, prm)
        n_anch += int(anch)
        ce, cn = cam_from_off(node, off)
        dE, dN = E - ce, N - cn
        F = dE * np.sin(th) + dN * np.cos(th)
        R = dE * np.cos(th) - dN * np.sin(th)
        u, v = geom.ground_to_pixels(R, F, prm["h"], img["pitch"] + prm["dpitch"], fov=90.0 + prm["dfov"])
        vis |= (F > F_MIN) & (F < F_MAX) & (u >= 0) & (u < 640) & (v >= 0) & (v < 640)
        for c, poly in polys_for_image(img, node, prm, classes):
            blocked |= shapely.contains_xy(poly, S, L)
    ok = vis.mean(1) >= VIS_MIN
    base = dict(zone_w=hi - lo, shared=int(shared), coverage=float(ok.mean()), n_anchored=n_anch)
    if ok.sum() < MIN_SLICES:
        return base | dict(ecw_min=np.nan, ecw_avg=np.nan)
    fr = ~blocked[ok]
    run = np.zeros(fr.shape[0])
    for k in range(fr.shape[0]):
        best = cur = 0
        for x in fr[k]:
            cur = cur + 1 if x else 0
            if cur > best:
                best = cur
        run[k] = best * DL
    return base | dict(ecw_min=float(run.min()), ecw_avg=float(fr.sum(1).mean() * DL), ecw_p10=float(np.percentile(run, 10)),
                       blocked_area=float((blocked & vis).sum() * DS * DL))


def class_areas(node, side, images, prm=CENTRAL):
    lo, hi, _ = zone_bounds(node, side)
    S, L = _grid(lo, hi)
    acc = {c: np.zeros(S.shape, bool) for c in OBST + ["person"]}
    for img in images:
        for c, poly in polys_for_image(img, node, prm, set(OBST) | {"person"}):
            acc[c] |= shapely.contains_xy(poly, S, L)
    return {("a_" + c.replace(" ", "_")): float(m.sum() * DS * DL) for c, m in acc.items()}


def mc_draws(rng, n, base_h=2.5):
    for _ in range(n):
        yield dict(h=base_h * rng.normal(1.0, 0.08), dpitch=rng.normal(0, 1.5), dfov=rng.normal(0, 1.5),
                   depth_scale=rng.uniform(0.7, 1.3), cam_lat=rng.normal(0, 1.0), thr=float(rng.choice([0.5, 0.6, 0.7])))


def main(seg_dir, manifest, out_csv, n_mc=30, limit=0, calib_csv=None):
    corr = pd.read_csv(HERE / "corridor_nodes.csv")
    corr["node"] = np.where(corr.src == "SCS", "scs_" + corr.id.astype(str).str.replace(r"\.0$", "", regex=True), corr.id)
    corr = corr.set_index("node")
    man = pd.read_csv(manifest)
    rng = np.random.default_rng(42)
    hmap, hmed = {}, {}
    if calib_csv and Path(calib_csv).exists():
        cal = pd.read_csv(calib_csv)
        hmap = dict(zip(cal.node, cal.h_node))
        cnt = cal.groupby("src").h_node.agg(["size", "median"])
        hmed = {k: float(v["median"]) for k, v in cnt.iterrows() if v["size"] >= 15}      # source-level default only if >= 15 calibrated nodes
    rows, nn = [], 0
    for node_id, g in man.groupby("node", sort=False):
        if node_id not in corr.index:
            continue
        node = corr.loc[node_id]
        if pd.isna(node.w_corridor) or pd.isna(node.axis_azimuth):
            continue
        imgs = []
        for r in g.itertuples():
            jf = Path(seg_dir) / (r.key + ".json")
            if jf.exists():
                dd = json.load(open(jf))["dets"]; nz = np.load(Path(seg_dir) / (r.key + ".npz"))
                imgs.append(dict(heading=r.heading, pitch=r.pitch, dets=dd, npz=nz, uv=anchor.facade_uv(dd, nz)))
        if not imgs:
            continue
        nn += 1
        if limit and nn > limit:
            break
        base_h = hmap.get(node_id, hmed.get(node.src, 2.5))
        CEN = dict(CENTRAL, h=base_h)
        for side in "LR":
            c = evaluate(node, side, imgs, CEN)
            if c is None:
                continue
            rec = dict(node=node_id, src=node.src, side=side, n_views=len(imgs), h_used=base_h, h_calibrated=int(node_id in hmap)); rec.update(c)
            if not np.isnan(c["ecw_min"]):
                rec["ecw_min_with_persons"] = evaluate(node, side, imgs, CEN, with_persons=True)["ecw_min"]
                rec.update(class_areas(node, side, imgs, CEN))
                mins = [e["ecw_min"] for e in (evaluate(node, side, imgs, p) for p in mc_draws(rng, n_mc, base_h)) if e and not np.isnan(e["ecw_min"])]
                if mins:
                    rec.update(ecw_min_mc_p05=np.percentile(mins, 5), ecw_min_mc_p50=np.percentile(mins, 50), ecw_min_mc_p95=np.percentile(mins, 95))
            rows.append(rec)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("wrote", out_csv, len(rows), "node-sides from", nn, "nodes")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 30, int(sys.argv[5]) if len(sys.argv) > 5 else 0, sys.argv[6] if len(sys.argv) > 6 else None)
