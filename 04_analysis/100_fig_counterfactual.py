"""Counterfactual figure: which impassable nodes are impassable because of vehicles present in the frame?
(a) alluvial from the observed ECW tier to the tier without vehicles, (b) map of the densest sampled neighbourhood coloured by dependence on vehicles,
(c) share of impassable nodes that are structural or vehicle-dependent, by street class.
Both tiers come from the sensitivity run (scenarios 'central' and 'vehicles removed'), i.e. the same computation with vehicles deleted from the masks;
nodes are those of the analysed sample. Addresses the concern that a single frame captures transient vehicles."""
import geopandas as gpd
from matplotlib.patches import FancyBboxPatch
from sigfig_common import *

n = load()
s = pd.read_csv(HERE / "out_final/sensitivity_nodes.csv")
c = s[s.scenario == "central"].set_index("node").ecw_best.rename("c"); v = s[s.scenario == "vehicles removed"].set_index("node").ecw_best.rename("v")
d = n.set_index("node").join(pd.concat([c, v], axis=1), how="inner").dropna(subset=["c", "v"]).reset_index()
tier = lambda x: np.where(x < 0.9, "Impassable", np.where(x < 1.5, "Restricted", "Accessible"))
d["t_obs"], d["t_free"] = tier(d.c), tier(d.v)
def cat(r):
    if r.t_obs == "Impassable": return "Structural" if r.t_free == "Impassable" else "Vehicle-dependent"
    if r.t_obs == "Restricted" and r.t_free == "Accessible": return "Restricted by vehicles"
    return "Unaffected"
d["cat"] = d.apply(cat, axis=1)
CAT_COL = {"Structural": "#8e3b3b", "Vehicle-dependent": "#e08a3c", "Restricted by vehicles": "#e8d28a", "Unaffected": "#c9d3e6"}
N = len(d)

fig = plt.figure(figsize=(19, 7.6), facecolor=BG)
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.15, 0.85], wspace=0.10)
ax1, ax2, ax3 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

# ---- (a) alluvial ---------------------------------------------------------------------------------------------------------
ax1.axis("off"); ax1.set_xlim(0, 10); ax1.set_ylim(0, 10.4)
H, PAD, Y0, BW = 8.6, 0.16, 0.5, 1.05
cnt = lambda col, t: int((d[col] == t).sum())
def lay(col, x):
    out, y = {}, Y0
    for t in TIERS[::-1]:
        h = cnt(col, t) / N * H; out[t] = (x, y, h); y += h + PAD
    return out
L, R = lay("t_obs", 1.6), lay("t_free", 8.4)
offl = {t: 0.0 for t in TIERS}; offr = {t: 0.0 for t in TIERS}
for a in TIERS[::-1]:
    for b in TIERS[::-1]:
        k = int(((d.t_obs == a) & (d.t_free == b)).sum())
        if k == 0: continue
        h = k / N * H; t = np.linspace(0, 1, 120); sm = 3 * t ** 2 - 2 * t ** 3
        y1 = L[a][1] + offl[a] + h; y2 = R[b][1] + offr[b] + h; xs = 1.6 + BW + (8.4 - BW - 1.6 - BW) * t
        top = y1 + (y2 - y1) * sm; bot = (y1 - h) + ((y2 - h) - (y1 - h)) * sm
        col = "#e08a3c" if (a == "Impassable" and b != "Impassable") else TIER_COL[a]
        ax1.fill(np.r_[xs, xs[::-1]], np.r_[top, bot[::-1]], color=col, alpha=0.55 if col == "#e08a3c" else 0.32, lw=0)
        if k >= 12: ax1.text(xs[18], (top[18] + bot[18]) / 2, str(k), ha="center", va="center", fontsize=8.5, fontweight="bold", color="#444")
        offl[a] += h; offr[b] += h
for side, P, col in ((1.6, L, "t_obs"), (8.4, R, "t_free")):
    for t, (x, y, h) in P.items():
        ax1.add_patch(FancyBboxPatch((x - BW, y), 2 * BW, h, boxstyle="round,pad=0.01", fc=TIER_COL[t], ec="white", lw=1, alpha=0.95))
        ax1.text(x, y + h / 2, f"{t}\n{cnt(col, t)} ({100 * cnt(col, t) / N:.0f}%)", ha="center", va="center", fontsize=8, fontweight="bold", color="white", multialignment="center")
ax1.text(1.6, 9.85, "As observed", ha="center", fontsize=11, fontweight="bold"); ax1.text(8.4, 9.85, "Vehicles removed", ha="center", fontsize=11, fontweight="bold")
ax1.set_title("(a) ECW tier before and after removing vehicles", fontsize=11.5, fontweight="bold")

# ---- (b) map ------------------------------------------------------------------------------------------------------------
G = REV / "02_pipeline_v2/konak_layers.gpkg"; roads = gpd.read_file(G, layer="roads"); bld = gpd.read_file(G, layer="buildings")
sc = d[d.src == "SCS"]; from scipy.spatial import cKDTree
tr = cKDTree(np.c_[sc.x_node, sc.y_node]); cn = [len(q) for q in tr.query_ball_point(np.c_[sc.x_node, sc.y_node], 350)]
i0 = int(np.argmax(cn)); cx, cy = sc.x_node.iloc[i0], sc.y_node.iloc[i0]; HW, HH = 520, 400; w = (cx - HW, cy - HH, cx + HW, cy + HH)
ax2.set_aspect("equal"); ax2.set_facecolor("white"); ax2.set_xticks([]); ax2.set_yticks([])
bld.cx[w[0]:w[2], w[1]:w[3]].plot(ax=ax2, color="#e4e4e4", ec="#d0d0d0", lw=0.25); roads.cx[w[0]:w[2], w[1]:w[3]].plot(ax=ax2, color="#a9a9a9", lw=0.9)
sub = d[d.x_node.between(w[0], w[2]) & d.y_node.between(w[1], w[3])]
for k in ("Unaffected", "Restricted by vehicles", "Structural", "Vehicle-dependent"):
    q = sub[sub.cat == k]; ax2.scatter(q.x_node, q.y_node, s=44 if k == "Unaffected" else 84, c=CAT_COL[k], edgecolors="white", lw=0.8, zorder=4, label=f"{k} ({(d.cat == k).sum()} in sample)")
ax2.set_xlim(w[0], w[2]); ax2.set_ylim(w[1], w[3]); ax2.plot([w[0] + 40, w[0] + 240], [w[1] + 30, w[1] + 30], color="#222", lw=3); ax2.text(w[0] + 140, w[1] + 55, "200 m", ha="center", fontsize=8.5)
ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=8.5, frameon=False, title="Dependence on vehicles present in the frame", title_fontsize=8.8)
ax2.set_title("(b) The densest sampled neighbourhood", fontsize=11.5, fontweight="bold")

# ---- (c) by street class ---------------------------------------------------------------------------------------------
ax3.set_facecolor(BG)
imp = d[d.t_obs == "Impassable"]
for i, cl in enumerate(CLASSES):
    q = imp[imp["class"] == cl]; nn = len(q)
    if nn == 0: ax3.text(3, i, "no impassable nodes", va="center", fontsize=8.5, color="#888"); continue
    sh = 100 * (q.cat == "Structural").sum() / nn; dep = 100 * (q.cat == "Vehicle-dependent").sum() / nn
    ax3.barh(i, sh, color=CAT_COL["Structural"], height=0.62); ax3.barh(i, dep, left=sh, color=CAT_COL["Vehicle-dependent"], height=0.62)
    ax3.text(sh / 2, i, f"{sh:.0f}", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold") if sh > 8 else None
    ax3.text(sh + dep / 2, i, f"{dep:.0f}", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold") if dep > 8 else None
    ax3.text(101.5, i, f"n={nn}", va="center", fontsize=8, color="#666")
ax3.set_yticks(range(len(CLASSES))); ax3.set_yticklabels([c_.replace(" ", "\n", 1) for c_ in CLASSES], fontsize=9); ax3.invert_yaxis(); ax3.set_xlim(0, 112)
for t, c_ in zip(ax3.get_yticklabels(), CLASSES): t.set_color(CLS_COL[c_])
ax3.set_xlabel("Share of impassable nodes (%)", fontsize=9.5)
ax3.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=CAT_COL[k]) for k in ("Structural", "Vehicle-dependent")], labels=["still impassable without vehicles", "no longer impassable without vehicles"], loc="lower center", bbox_to_anchor=(0.5, -0.24), frameon=False, fontsize=8.5)
ax3.set_title("(c) Impassable nodes by street class", fontsize=11.5, fontweight="bold")
save(fig, "fig_sig_counterfactual")

n_imp = int((d.t_obs == "Impassable").sum()); dep = int((d.cat == "Vehicle-dependent").sum()); st_ = int((d.cat == "Structural").sum()); rv = int((d.cat == "Restricted by vehicles").sum())
stat("cfN", N); stat("cfImpN", n_imp); stat("cfDepN", dep); stat("cfDepPct", round(100 * dep / n_imp, 1)); stat("cfStructN", st_); stat("cfStructPct", round(100 * st_ / n_imp, 1))
stat("cfRestrN", rv); stat("cfImpAfterN", int((d.t_free == "Impassable").sum())); stat("cfImpAfterPct", round(100 * (d.t_free == "Impassable").mean(), 1)); stat("cfImpBeforePct", round(100 * (d.t_obs == "Impassable").mean(), 1))
print(d.cat.value_counts().to_dict(), n_imp)
