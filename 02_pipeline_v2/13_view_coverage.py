"""Geometric coverage of each pedestrian zone by each of the 4 cardinal views (no segmentation needed)."""
import numpy as np, pandas as pd, geom, importlib.util, sys
spec = importlib.util.spec_from_file_location("e", "30_ecw_v2.py"); e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
c = pd.read_csv("corridor_nodes.csv"); c = c[(c.src == "SCS") & c.snapped & c.w_corridor.notna()]
rows = []
for _, n in c.iterrows():
    Wc = n.w_carriage; shared = (n.w_corridor - Wc) < e.SHARED_MAX
    for side in "LR":
        lo, hi = ((0.0 if shared else Wc / 2), n.d_left) if side == "L" else (-n.d_right, (0.0 if shared else -Wc / 2))
        if hi - lo < 0.3: continue
        ss = np.arange(-10, 10.01, 0.25); ll = np.arange(lo + 0.05, hi, 0.1)
        S, L = np.meshgrid(ss, ll, indexing="ij")
        # camera at the node point in axis frame
        s_c, l_c = geom.world_to_axis(np.array([n.x_node]), np.array([n.y_node]), n.x_snap, n.y_snap, n.axis_azimuth)
        E, N = e._axis_to_world(S + float(s_c[0]) * 0, L, n)
        E, N = e._axis_to_world(S, L, n)
        for d, hd in (("N", 0), ("E", 90), ("S", 180), ("W", 270)):
            th = np.radians(hd); dE, dN = E - n.x_node, N - n.y_node
            F = dE * np.sin(th) + dN * np.cos(th); R = dE * np.cos(th) - dN * np.sin(th)
            u, v = geom.ground_to_pixels(R, F, 2.5, -10.0)
            vis = (F > 1.5) & (F < 30) & (u >= 0) & (u < 640) & (v >= 0) & (v < 640)
            rows.append(dict(node=n.id, side=side, view=d, vis=vis.mean(), zone_w=hi - lo, shared=shared))
df = pd.DataFrame(rows); df.to_csv("view_coverage.csv", index=False)
best = df.groupby(["node", "side"]).vis.max().reset_index()
print("zone-sides:", len(best), "  best single view coverage: median", best.vis.median().round(2), " share>=0.5:", (best.vis >= .5).mean().round(2), " share>=0.3:", (best.vis >= .3).mean().round(2))
# union of the best two views
piv = df.pivot_table(index=["node", "side"], columns="view", values="vis")
print("mean coverage by view:\n", piv.mean().round(2).to_string())
top2 = piv.apply(lambda r: r.nlargest(2).sum(), axis=1).clip(upper=1)
print("best-2-view coverage (upper bound sum): median", top2.median().round(2), " share>=0.5:", (top2 >= .5).mean().round(2))
