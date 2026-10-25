import geopandas as gpd, pandas as pd, numpy as np, pyogrio, os, sys
G = os.environ.get("PC_MUNICIPAL_GPKG")
if not G:
    sys.exit("PC_MUNICIPAL_GPKG is not set (path to the municipal smart-city GeoPackage)")
ilce = gpd.read_file(G, layer="ILCEALANI", where="ADINUMARASI='KONAK'")
print("Konak area km2", ilce.geometry.area.sum()/1e6, ilce.total_bounds)
minx, miny, maxx, maxy = ilce.total_bounds
roads = gpd.read_file(G, layer="YOLORTAHATTI", bbox=(minx, miny, maxx, maxy))
roads = roads[roads.intersects(ilce.union_all())]
print("roads in Konak", len(roads)); print(roads.YOLTIP.value_counts(dropna=False).to_string())
print(pd.crosstab(roads.YOLTIP.fillna('NA'), roads.SERITSAYISI.fillna(-1)).to_string())
print(roads.KAPLAMA.value_counts(dropna=False).head(8).to_string()); print(roads.YOLUNISLEVI.value_counts(dropna=False).head(8).to_string())
b = gpd.read_file(G, layer="YAPI", bbox=(minx, miny, maxx, maxy))
print("buildings in bbox", len(b)); print(b.ZEMINUSTUKATSAYISI.describe().to_string())
for n,g in (("roads",roads),("buildings",b),("boundary",ilce)): g.to_file("konak_layers.gpkg", layer=n, driver="GPKG")
