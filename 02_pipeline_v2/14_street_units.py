"""Group road features into street axes: same name (ADINUMARASI) + same class, merged where lines touch."""
import geopandas as gpd, pandas as pd, numpy as np, shapely
from shapely.ops import linemerge, unary_union
roads = gpd.read_file("konak_layers.gpkg", layer="roads")
roads = roads[roads.geometry.notna()].copy()
roads["nm"] = roads.ADINUMARASI.fillna("").str.strip().str.upper()
roads["yt"] = roads.YOLTIP.fillna("NA")
sid = {}
for (nm, yt), g in roads.groupby(["nm", "yt"]):
    if nm == "":
        for i in g.index: sid[roads.loc[i, "OBJECTID"]] = f"anon_{roads.loc[i,'OBJECTID']}"
        continue
    merged = shapely.line_merge(unary_union(list(g.geometry.values)))
    parts = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
    for _, r in g.iterrows():
        mid = r.geometry.interpolate(0.5, normalized=True)
        k = int(np.argmin([mid.distance(p) for p in parts]))
        sid[r.OBJECTID] = f"{yt}_{nm}_{k}"
roads["street_id"] = roads.OBJECTID.map(sid)
c = pd.read_csv("corridor_nodes.csv")
c["street_id"] = c.road_oid.map(sid)
c.to_csv("corridor_nodes.csv", index=False)
s = c[(c.src == "SCS") & c.w_corridor.notna()]
g = s.groupby("street_id").size()
print("streets:", len(g), " nodes/street median", g.median(), " streets>=3 nodes:", (g >= 3).sum(), " >=5:", (g >= 5).sum(), " nodes in streets>=3:", g[g >= 3].sum())
