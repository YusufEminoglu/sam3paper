"""Derive per-node street corridor geometry from the municipal smart-city GeoPackage (Konak).

For each node (SCS Konak nodes + the 63 original TiGIS nodes):
  1. snap to the nearest road centre-line (YOLORTAHATTI) within SNAP_MAX
  2. local tangent / normal from the line (+-6 m window)
  3. five transects (offsets -10,-5,0,5,10 m along the axis), each cast SIDE_MAX to both sides;
     first building facade hit on each side  ->  d_left, d_right  (median over transects)
  4. W_corridor = d_left + d_right   (building-to-building, NOT the cadastral right-of-way)
  5. carriageway width = lanes x lane_width(YOLTIP) ; pedestrian-intended width W_ped = W_corridor - W_carriageway
Outputs: corridor_nodes.csv
"""
import os
import numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.strtree import STRtree
from pathlib import Path

HERE = Path(__file__).parent
SCS = Path(os.environ.get("PC_SCS_NODES", HERE.parent / "00_inputs/konak_gsv_candidates.csv"))
TG = Path(os.environ.get("PC_LEGACY_NODES", HERE.parent / "00_inputs/legacy_gsv_nodes.csv"))
SNAP_MAX, SIDE_MAX = 20.0, 30.0
OFFS = (-10, -5, 0, 5, 10)
# assumed lane widths by road class (user-specified rule: highest class 4.0 m, lowest 2.5 m)
LANE_W = {"OTOYOL": 4.0, "BULVAR": 4.0, "MEYDAN": 3.5, "CADDE": 3.5, "SOKAK": 3.0}
LANE_W_DEFAULT = 2.5
LANES_DEFAULT = {"BULVAR": 2, "CADDE": 2, "SOKAK": 1, "MEYDAN": 2}   # when SERITSAYISI is null/0

roads = gpd.read_file(HERE / "konak_layers.gpkg", layer="roads")
bld = gpd.read_file(HERE / "konak_layers.gpkg", layer="buildings")
roads = roads[roads.geometry.notna()].reset_index(drop=True)
bld = bld[bld.geometry.notna() & bld.geometry.is_valid].reset_index(drop=True)
bgeoms = bld.geometry.values
btree = STRtree(bgeoms)
rgeoms = roads.geometry.values
rtree = STRtree(rgeoms)
floors = bld["ZEMINUSTUKATSAYISI"].fillna(0).values

EXCLUDED_HEX = {97}   # H3 893f66a6e73ffff: 25 candidates, none snapped-with-corridor, none imaged, none analysed (industrial/TARIS plot); dropped from the study
scs = pd.read_csv(SCS)
scs = scs[scs.district == "Konak"][["node_id", "lat", "lon", "hex_id"]].rename(columns={"node_id": "id"})
scs = scs[~scs.hex_id.isin(EXCLUDED_HEX)]
scs["src"] = "SCS"
tg = pd.read_csv(TG)
tg = tg.assign(id=("Ax" + tg.Axis_ID.astype(str) + "_" + tg.distance.astype(str)), src="TG",
               lat=tg.Lat, lon=tg.Lon)[["id", "lat", "lon", "src"]]
allp = pd.concat([scs, tg], ignore_index=True)
pts = gpd.GeoDataFrame(allp, geometry=gpd.points_from_xy(allp.lon, allp.lat), crs=4326).to_crs(roads.crs)


def tangent(line, s, w=6.0):
    a = line.interpolate(max(s - w, 0)); b = line.interpolate(min(s + w, line.length))
    v = np.array([b.x - a.x, b.y - a.y]); n = np.hypot(*v)
    return v / n if n > 0 else np.array([1.0, 0.0])


def first_hit(p, direction, cand_idx_cache=None):
    ray = LineString([p, (p[0] + direction[0] * SIDE_MAX, p[1] + direction[1] * SIDE_MAX)])
    idx = btree.query(ray, predicate="intersects")
    if len(idx) == 0:
        return np.nan, np.nan
    best, bf = np.inf, np.nan
    P = Point(p)
    for i in idx:
        d = P.distance(bgeoms[i].intersection(ray))
        if d < best:
            best, bf = d, floors[i]
    return best, bf


rows = []
for _, r in pts.iterrows():
    P = r.geometry
    cand = rtree.query(P.buffer(SNAP_MAX))
    rec = dict(id=r.id, src=r.src, lat=r.lat, lon=r.lon, hex_id=r.get("hex_id", np.nan))
    if len(cand) == 0:
        rows.append(rec | dict(snapped=False)); continue
    dd = [(P.distance(rgeoms[i]), i) for i in cand]
    d0, i0 = min(dd)
    # rtree.query(buffer) is a bounding-box test: a MultiLineString road feature that bundles several
    # disconnected segments under one OBJECTID can have a bounding box far larger than SNAP_MAX, so a
    # candidate can pass the bbox test even though its true nearest point is well beyond SNAP_MAX. The
    # true distance d0 must therefore be checked explicitly (fixed after the R1 submission: 36/881 nodes
    # marked snapped=True had snap_dist up to 100.7 m; 2 of them were in the analysed sample).
    if d0 > SNAP_MAX:
        rows.append(rec | dict(snapped=False, snap_dist=d0)); continue
    line = rgeoms[i0]
    if line.geom_type == "MultiLineString":
        line = min(line.geoms, key=lambda g: P.distance(g))
    s = line.project(P)
    q = line.interpolate(s)
    t = tangent(line, s)
    n = np.array([-t[1], t[0]])
    dl, dr, fl, fr = [], [], [], []
    for o in OFFS:
        c = np.array([q.x, q.y]) + t * o
        a, fa = first_hit(tuple(c), n)     # left
        b, fb = first_hit(tuple(c), -n)    # right
        dl.append(a); dr.append(b); fl.append(fa); fr.append(fb)
    L, R = np.nanmedian(dl) if not np.all(np.isnan(dl)) else np.nan, np.nanmedian(dr) if not np.all(np.isnan(dr)) else np.nan
    rr = roads.iloc[i0]
    yt = rr.YOLTIP if isinstance(rr.YOLTIP, str) else None
    lanes = rr.SERITSAYISI if pd.notna(rr.SERITSAYISI) and rr.SERITSAYISI > 0 else LANES_DEFAULT.get(yt, 1)
    lw = LANE_W.get(yt, LANE_W_DEFAULT)
    az = (np.degrees(np.arctan2(t[0], t[1])) + 360) % 360      # azimuth of the axis, clockwise from north
    rec |= dict(snapped=True, road_oid=rr.OBJECTID, yolun_islevi=rr.YOLUNISLEVI, kaplama=rr.KAPLAMA, snap_dist=d0, x_node=P.x, y_node=P.y, x_snap=q.x, y_snap=q.y, road_name=rr.ADINUMARASI, yoltip=yt, lanes_raw=rr.SERITSAYISI, lanes=lanes,
                lane_w=lw, w_carriage=lanes * lw, axis_azimuth=az, d_left=L, d_right=R,
                w_corridor=(L + R) if not (np.isnan(L) or np.isnan(R)) else np.nan,
                floors_left=np.nanmedian(fl) if not np.all(np.isnan(fl)) else np.nan,
                floors_right=np.nanmedian(fr) if not np.all(np.isnan(fr)) else np.nan,
                n_hit=int(np.sum(~np.isnan(dl)) + np.sum(~np.isnan(dr))))
    rows.append(rec)

out = pd.DataFrame(rows)
out["w_ped"] = out.w_corridor - out.w_carriage
out.to_csv(HERE / "corridor_nodes.csv", index=False)
print(out.groupby("src").agg(n=("id", "size"), snapped=("snapped", "sum"), has_w=("w_corridor", lambda s: s.notna().sum())))
print(out[out.src == "SCS"].w_corridor.describe())
print(out[out.src == "SCS"].w_ped.describe())
