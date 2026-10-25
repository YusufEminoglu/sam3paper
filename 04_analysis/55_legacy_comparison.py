"""Legacy pilot: the seven original axes under the first-version method (R0) and the revised method (R1).

usage: python 55_legacy_comparison.py <nodes_R1.csv> <corridor_nodes.csv> <out_dir>
Reads the R0 result table (sam3_results.csv, 63 rows) and writes: legacy_axis_table.csv, legacy_node_table.csv, legacy_values.txt
"""
import sys, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

nodes = pd.read_csv(sys.argv[1]); cor = pd.read_csv(sys.argv[2]); out = Path(sys.argv[3]); out.mkdir(exist_ok=True, parents=True)
r0 = pd.read_csv(Path(__file__).resolve().parents[1] / "legacy_v1/sam3_results.csv")
r0["node"] = "Ax" + r0.Axis_ID.astype(str) + "_" + r0.Distance_m.astype(int).astype(str)
cor["node"] = np.where(cor.src == "SCS", "scs_" + cor.id.astype(str).str.replace(r"\.0$", "", regex=True), cor.id)
tg = cor[cor.src == "TG"][["node", "road_name", "yoltip", "lanes", "w_corridor", "w_carriage"]]
m = r0.merge(tg, on="node", how="left").merge(nodes[["node", "ecw_best", "pc", "coverage", "tier", "ecw_p05", "ecw_p95"]], on="node", how="left")
ecw_col = "ECW_m" if "ECW_m" in m else [c for c in m.columns if c.lower().startswith("ecw")][0]
m["ecw_r0"] = m[ecw_col]; m["axis"] = "Ax" + m.Axis_ID.astype(str)
m.to_csv(out / "legacy_node_table.csv", index=False)
ax = m.groupby("axis").agg(morph_r0=("Morphology", "first"), w_cad=("Width_Cad_m", "first"), road=("road_name", lambda s: s.mode().iat[0] if s.notna().any() else ""),
                           w_corr_gis=("w_corridor", "median"), ecw_r0_mean=("ecw_r0", "mean"), n_r1=("ecw_best", "count"),
                           ecw_r1_mean=("ecw_best", "mean"), share_imp_r0=("ecw_r0", lambda x: (x == 0).mean() * 100),
                           share_imp_r1=("ecw_best", lambda x: (x.dropna() < 0.9).mean() * 100 if x.notna().any() else np.nan)).round(2)
ax.to_csv(out / "legacy_axis_table.csv")
v = m.dropna(subset=["ecw_best", "ecw_r0"])
lines = [f"nodes with R1 estimate: {len(v)} of {len(m)}",
         f"Spearman(R0 ECW, R1 ECW_best) = {stats.spearmanr(v.ecw_r0, v.ecw_best)[0]:.2f}" if len(v) > 5 else "n too small",
         f"R0 share ECW=0: {(m.ecw_r0 == 0).mean() * 100:.1f}% (n={len(m)}); R1 share impassable among evaluated: {(v.ecw_best < 0.9).mean() * 100:.1f}% (n={len(v)})",
         f"R0 clutter cap reached (either side 4.00 m): {((m['Clutter_Left_m'] >= 4.0) | (m['Clutter_Right_m'] >= 4.0)).mean() * 100:.0f}%"]
(out / "legacy_values.txt").write_text("\n".join(lines), encoding="utf8"); print("\n".join(lines)); print(ax.to_string())
