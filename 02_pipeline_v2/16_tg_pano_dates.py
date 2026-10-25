"""Capture month/year of the Street View panorama behind each legacy TG node (reviewer 1.2: transient objects, single-time GSV).

usage: set GSV_API_KEY, then  python 16_tg_pano_dates.py  ->  tg_pano_dates.csv
Uses only the Street View *metadata* endpoint (no images are requested; Google does not bill metadata calls).
"""
import os, sys, time, json, urllib.parse, urllib.request, pandas as pd
from pathlib import Path

KEY = os.environ.get("GSV_API_KEY")
if not KEY:
    sys.exit("GSV_API_KEY is not set")
HERE = Path(__file__).parent
nodes = pd.read_csv(os.environ.get("PC_LEGACY_NODES", HERE.parent / "00_inputs/legacy_gsv_nodes.csv"))
rows = []
for r in nodes.itertuples():
    q = urllib.parse.urlencode(dict(location=f"{r.Lat},{r.Lon}", radius=25, source="outdoor", key=KEY))
    j = json.load(urllib.request.urlopen("https://maps.googleapis.com/maps/api/streetview/metadata?" + q, timeout=30))
    rows.append(dict(node=f"Ax{int(r.Axis_ID)}_{int(r.distance)}", lat=r.Lat, lon=r.Lon, status=j.get("status"),
                     pano_id=j.get("pano_id"), date=j.get("date"), used_year=r.gsv_year))
    time.sleep(0.05)
df = pd.DataFrame(rows)
df.to_csv("tg_pano_dates.csv", index=False)
print(df.status.value_counts().to_dict())
print(df.date.str[:7].value_counts().sort_index().to_string())
print("year mismatch vs stored gsv_year:", (df.date.str[:4] != df.used_year.astype(str)).sum())
