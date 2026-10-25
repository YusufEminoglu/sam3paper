"""Select the two most street-perpendicular cardinal views per SCS Konak node (+ the 126 original TiGIS images)."""
import pandas as pd, numpy as np, os
c = pd.read_csv("corridor_nodes.csv")
_HERE = os.path.dirname(os.path.abspath(__file__))
SCS_IMG = os.environ.get("PC_SCS_IMAGES", os.path.join(_HERE, "..", "..", "data", "external", "gsv"))
TG_IMG = os.environ.get("PC_LEGACY_IMAGES", os.path.join(_HERE, "..", "..", "data", "external", "gsv_legacy"))
tg_nodes = pd.read_csv(os.environ.get("PC_LEGACY_NODES", os.path.join(_HERE, "..", "00_inputs", "legacy_gsv_nodes.csv")))
CARD = {"N": 0, "E": 90, "S": 180, "W": 270}
rows = []
s = c[(c.src == "SCS") & c.snapped & c.w_corridor.notna()]
for _, r in s.iterrows():
    for side, off in (("L", -90), ("R", 90)):
        want = (r.axis_azimuth + off) % 360
        d = min(CARD, key=lambda k: min(abs(CARD[k] - want), 360 - abs(CARD[k] - want)))
        p = os.path.join(SCS_IMG, f"node{int(r.id)}_{d}.jpg")
        if os.path.exists(p):
            rows.append(dict(key=f"scs_{int(r.id)}_{d}", path=p, node=f"scs_{int(r.id)}", side=side, heading=CARD[d], pitch=-10.0, src="SCS"))
m = pd.DataFrame(rows).drop_duplicates("key")     # a node whose two wanted views coincide keeps one row per side label
for _, t in tg_nodes.iterrows():
    for side, sign in (("L", -90), ("R", 90)):
        p = os.path.join(TG_IMG, f"Axis{int(t.Axis_ID)}_Dist{int(t.distance)}_{'Left' if side=='L' else 'Right'}.jpg")
        if os.path.exists(p):
            m.loc[len(m)] = dict(key=f"tg_Ax{int(t.Axis_ID)}_{int(t.distance)}_{side}", path=p, node=f"Ax{int(t.Axis_ID)}_{int(t.distance)}",
                                 side=side, heading=(t.angle + sign) % 360, pitch=0.0, src="TG")
m.to_csv("manifest_all.csv", index=False)
m[m.src == "TG"].to_csv("manifest_TG.csv", index=False)
print(m.groupby("src").size(), m.node.nunique())
