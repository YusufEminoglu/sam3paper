"""Facade anchoring of the camera lateral position.

GSV panoramas are captured from a car driving in a lane, not on the street centre-line, and the pano can be up to 15 m from the
requested node, so the camera's lateral offset from the GIS axis is unknown (up to several metres).  Every image that sees a facade
gives an estimate of that offset: the ground-contact line of the building (first sidewalk/road pixel at/below the building base)
is projected to the ground plane and compared with the GIS facade distance (d_left / d_right).  The offset delta (metres along
the axis-left direction) that reconciles the two is applied to the camera position of that image.  Images without a reliable
facade line stay unanchored and carry a larger lateral uncertainty (SIGMA_UN).
"""
import numpy as np
import geom

SIGMA_AN, SIGMA_UN = 0.30, 2.00          # metres, lateral camera uncertainty after / without anchoring
DELTA_MAX = 6.0
BLOCK = {"car", "truck", "motorcycle", "person", "chair", "table", "potted plant", "trash can", "dumpster", "utility pole",
         "street light", "utility box", "electrical cabinet"}


def facade_uv(dets, npz, thr=0.6):
    """Pixel points (u, v) of the building/ground junction, one per sampled image column."""
    shape = tuple(npz["shape"])
    bld = np.zeros(shape, bool); ground = np.zeros(shape, bool); block = np.zeros(shape, bool)
    for i, d in enumerate(dets):
        if d["score"] < thr or d["prompt"] not in ("building", "sidewalk", "road") and d["prompt"] not in BLOCK:
            continue
        m = np.unpackbits(npz["masks"][i])[: shape[0] * shape[1]].reshape(shape).astype(bool)
        if d["prompt"] == "building":
            bld |= m
        elif d["prompt"] in ("sidewalk", "road"):
            ground |= m
        else:
            block |= m
    pts = []
    for c in np.where(bld.any(0))[0][::3]:
        vb = np.where(bld[:, c])[0].max()
        rr = np.where(ground[max(vb - 5, 0):min(vb + 60, shape[0] - 12), c])[0]
        if len(rr) == 0:
            continue
        vg = max(vb - 5, 0) + rr.min()
        if ground[vg:vg + 10, c].mean() > 0.7 and not block[max(vb - 2, 0):vg + 6, c].any():
            pts.append((c + 0.5, float(vg)))
    return np.array(pts)


def estimate_delta(uv, node, heading, pitch, h=2.5, dpitch=0.0, dfov=0.0):
    """Return delta (m, along axis-left) or None.  Camera is assumed at the node point before anchoring."""
    if uv is None or len(uv) < 8:
        return None
    R, F, ok = geom.pixels_to_ground(uv[:, 0], uv[:, 1], h, pitch + dpitch, fov=90.0 + dfov)
    ok &= (F > 3) & (F < 25)
    if ok.sum() < 8:
        return None
    E, N = geom.cam_to_world(R[ok], F[ok], heading, node.x_node, node.y_node)
    s, l = geom.world_to_axis(E, N, node.x_snap, node.y_snap, node.axis_azimuth)
    if np.subtract(*np.percentile(l, [75, 25])) > 2.0:      # a facade is a line parallel to the axis
        return None
    ml = float(np.median(l))
    ref = node.d_left if ml > 0 else -node.d_right
    if np.isnan(ref):
        return None
    delta = ref - ml
    return delta if abs(delta) <= DELTA_MAX else None


def facade_r(uv, node, heading, pitch, h=2.5):
    """Signed lateral distance (m, +left) from the camera to the facade line, and the side.  All ground coordinates scale linearly with h."""
    if uv is None or len(uv) < 8:
        return None
    R, F, ok = geom.pixels_to_ground(uv[:, 0], uv[:, 1], h, pitch)
    ok &= (F > 3) & (F < 25)
    if ok.sum() < 8:
        return None
    E, N = geom.cam_to_world(R[ok], F[ok], heading, node.x_node, node.y_node)
    s, l = geom.world_to_axis(E, N, node.x_snap, node.y_snap, node.axis_azimuth)
    if np.subtract(*np.percentile(l, [75, 25])) > 2.0:
        return None
    _, l_node = geom.world_to_axis(np.array([node.x_node]), np.array([node.y_node]), node.x_snap, node.y_snap, node.axis_azimuth)
    return float(np.median(l) - l_node[0])
