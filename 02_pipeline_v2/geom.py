"""Ground-plane geometry for phantom-capacity v2 (replaces the fixed pixel/10240 and bbox*5/600 shortcuts).

Camera model (Google Street View Static API, rectilinear, square image):
    f_px = (W/2) / tan(FOV/2);  principal point at image centre;  pitch alpha_down = -pitch (GSV: negative = looking down)
    camera height h above a flat ground plane (nominal 2.5 m; varied in the uncertainty analysis)
Pixel (u, v) -> ground point in the camera-aligned frame (R = right, F = forward, both on the ground plane):
    ray = x*R + y*down + 1*axis,  x=(u-cx)/f, y=(v-cy)/f,  axis = (F cos a, U -sin a),  down = (F -sin a, U -cos a)
    t = h / (y cos a + sin a)     (valid below the horizon only)
    R_g = t x,   F_g = t (cos a - y sin a)
Then rotated by the compass heading into (East, North) and into the street-axis frame (s along the axis, l lateral, left +).
"""
import numpy as np

W_DEFAULT, FOV_DEFAULT = 640, 90.0


def focal_px(w=W_DEFAULT, fov=FOV_DEFAULT):
    return (w / 2.0) / np.tan(np.radians(fov) / 2.0)


def pixels_to_ground(u, v, h=2.5, pitch=-10.0, w=W_DEFAULT, fov=FOV_DEFAULT):
    """Return (R, F, valid) ground coordinates in metres relative to the camera foot point."""
    u = np.asarray(u, float); v = np.asarray(v, float)
    f = focal_px(w, fov); cx = cy = w / 2.0
    a = np.radians(-pitch)
    x = (u - cx) / f; y = (v - cy) / f
    den = y * np.cos(a) + np.sin(a)
    valid = den > 1e-3
    t = np.where(valid, h / np.where(valid, den, 1.0), np.nan)
    return t * x, t * (np.cos(a) - y * np.sin(a)), valid


def ground_to_pixels(R, F, h=2.5, pitch=-10.0, w=W_DEFAULT, fov=FOV_DEFAULT):
    """Inverse of pixels_to_ground (used by the tests and the projected-footprint overlays)."""
    f = focal_px(w, fov); cx = cy = w / 2.0
    a = np.radians(-pitch)
    # ground point vector from camera P = (R, F, -h);  zc = P . axis = F cos a + h sin a;  yc = P . down = h cos a - F sin a
    zc = F * np.cos(a) + h * np.sin(a)
    yc = h * np.cos(a) - F * np.sin(a)
    return cx + f * R / zc, cy + f * yc / zc


def cam_to_world(R, F, heading_deg, cam_e, cam_n):
    th = np.radians(heading_deg)
    fe, fn = np.sin(th), np.cos(th)          # forward unit (E, N)
    re, rn = np.cos(th), -np.sin(th)         # right unit (E, N)
    return cam_e + F * fe + R * re, cam_n + F * fn + R * rn


def world_to_axis(E, N, q_e, q_n, axis_az_deg):
    """s along the axis (clockwise-from-north azimuth), l lateral to the LEFT of the axis direction."""
    az = np.radians(axis_az_deg)
    te, tn = np.sin(az), np.cos(az)
    le, ln = -tn, te
    dE, dN = np.asarray(E) - q_e, np.asarray(N) - q_n
    return dE * te + dN * tn, dE * le + dN * ln


def base_contact_points(mask, step=4):
    """Lowest mask pixel of each image column: the visible ground-contact line of the object."""
    cols = np.where(mask.any(axis=0))[0][::step]
    if len(cols) == 0:
        return np.empty((0, 2))
    vs = np.array([np.where(mask[:, c])[0].max() for c in cols])
    return np.stack([cols.astype(float) + 0.5, vs.astype(float) + 1.0], axis=1)


# typical ground-contact depth (extent away from the camera) per class, metres.  These are PRIORS, not measurements;
# the uncertainty analysis perturbs them by +-30 %.
DEPTH_PRIOR = {"car": 1.8, "truck": 2.4, "motorcycle": 0.8, "chair": 0.5, "table": 0.7, "potted plant": 0.6,
               "trash can": 0.6, "dumpster": 1.2, "utility pole": 0.3, "street light": 0.3, "utility box": 0.5,
               "electrical cabinet": 0.5}
