"""Robustness of the ECW tiers to geometric uncertainty.

Interval of a node = [max over sides of the 5th percentile, max over sides of the 95th percentile]
of the Monte Carlo draws (30 per node-side), because the node value is the best of its sides.
A tier is 'robust' if both ends of the interval fall in the same tier.

Layout (Golden Publication Architecture):
- Panel (a) at TOP, CENTERED: 2 rows x 8 columns = 16 sub-panels showing all 16 H3 sampled neighborhood grids
- Panel (b) at BOTTOM-LEFT: Node-level ranked ECW spectrum & uncertainty intervals + bottom footnote card
- Panels (c+d) at BOTTOM-RIGHT: Tier survival stacked bars (c) + Uncertainty span Delta-ECW boxplots (d) + matching footnote cards
- Flawless margins, mathematical alignments, and standardized publication typography.
"""
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch, Polygon
from matplotlib.lines import Line2D
from sigfig_common import *

# 1. Load & Process Data
n = load()
e = pd.read_csv(REV / "02_pipeline_v2/ecw_v2.csv")
e = e[(e.coverage >= 0.5) & e.ecw_min.notna()]
g = e.groupby("node").agg(
    lo=("ecw_min_mc_p05", "max"),
    hi=("ecw_min_mc_p95", "max"),
    lo_env=("ecw_min_mc_p05", "min"),
    hi_env=("ecw_min_mc_p95", "max")
).reindex(n.node)
n = n.set_index("node").join(g).reset_index()

tier_fn = lambda x: np.where(x < 0.9, 0, np.where(x < 1.5, 1, 2))
n["lo0"], n["hi0"] = n.lo, n.hi
n["lo"] = n.lo.fillna(n.ecw_best)
n["hi"] = n.hi.fillna(n.ecw_best)
n["lo"] = np.minimum(n.lo, n.ecw_best)
n["hi"] = np.maximum(n.hi, n.ecw_best)
n["robust"] = tier_fn(n.lo) == tier_fn(n.hi)
n["cross09"] = (n.lo < 0.9) & (n.hi >= 0.9)
n["cross15"] = (n.lo < 1.5) & (n.hi >= 1.5)
n["unc_width"] = n.hi - n.lo

def categorize_kind(r):
    if r.robust:
        return "Robust"
    if r.cross09 and r.cross15:
        return "Crosses both thresholds"
    if r.cross09:
        return "Crosses 0.9 m"
    return "Crosses 1.5 m"

n["kind"] = n.apply(categorize_kind, axis=1)
env_rob = float((tier_fn(n.lo_env.fillna(n.ecw_best)) == tier_fn(n.hi_env.fillna(n.ecw_best))).mean())

# Vector Layers
G_gpkg = REV / "02_pipeline_v2/konak_layers.gpkg"
hexes = gpd.read_file(REV / "02_pipeline_v2/konak_hexes.gpkg", layer="hexes").set_index("hex_id")
valid_hex_ids = sorted([h for h in hexes.index if h in n["hex_id"].dropna().unique()])
valid_hexes = hexes.loc[valid_hex_ids].copy()

bld = gpd.read_file(G_gpkg, layer="buildings")
roads = gpd.read_file(G_gpkg, layer="roads")
gdf_nodes = gpd.GeoDataFrame(n, geometry=gpd.points_from_xy(n.x_node, n.y_node), crs=valid_hexes.crs)

# Palette Architecture (Harmonized with PhD Publication Guidelines)
PALETTE = {
    "teal": "#0f766e",        # Accessible
    "teal_light": "#ccfbf1",
    "teal_faint": "#f0fdf4",
    "amber": "#d97706",       # Restricted
    "amber_light": "#fef3c7",
    "amber_faint": "#fffbeb",
    "rose": "#e11d48",        # Impassable
    "rose_light": "#ffe4e6",
    "rose_faint": "#fff1f2",
    "purple": "#7c3aed",      # Crosses both
    "purple_light": "#ede9fe",
    "slate": "#64748b",       # Robust whiskers
    "dark_slate": "#334155",
    "ink": "#0f172a",
    "ink_soft": "#475569",
    "card_bg": "#f8fafc",
    "card_edge": "#cbd5e1",
    "edge": "#cbd5e1",
    "bld_face": "#f1f5f9",
    "bld_edge": "#cbd5e1",
    "road": "#94a3b8",
}

TIER_COLORS = {
    "Accessible": PALETTE["teal"],
    "Restricted": PALETTE["amber"],
    "Impassable": PALETTE["rose"],
}
TIER_LABELS = {
    "Accessible": "Accessible (≥1.5 m)",
    "Restricted": "Restricted (0.9–1.5 m)",
    "Impassable": "Impassable (<0.9 m)",
}
CROSS_COLORS = {
    "Robust": PALETTE["slate"],
    "Crosses 0.9 m": PALETTE["rose"],
    "Crosses 1.5 m": PALETTE["amber"],
    "Crosses both thresholds": PALETTE["purple"],
}

# Typography & Global Matplotlib Settings
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": PALETTE["ink"],
    "axes.labelcolor": PALETTE["ink"],
    "xtick.color": PALETTE["ink_soft"],
    "ytick.color": PALETTE["ink_soft"],
    "axes.edgecolor": PALETTE["card_edge"],
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": "#e2e8f0",
    "grid.linewidth": 0.5,
})

def panel_badge(ax, label, x=-0.015, y=1.085):
    """Standardized publication panel badge with rounded translucent frame."""
    ax.text(
        x, y, f"({label.lower()})",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=11.5, fontweight="bold", color=PALETTE["ink"],
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=PALETTE["ink_soft"], lw=0.9, alpha=0.98),
        zorder=40, clip_on=False
    )

# Canvas Dimensions
FIG_W, FIG_H = 16.5, 11.6
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white", dpi=300)

# Layout geometry coordinates in figure fractions
M_L = 0.040
M_R = 0.975
W_USABLE = M_R - M_L  # 0.935

# =============================================================================
# TOP REGION: PANEL (a) - 2 rows x 8 columns = 16 sub-panels
# =============================================================================
N_COLS = 8
N_ROWS = 2
GAP_COL = 0.0035
SUB_W = (W_USABLE - (N_COLS - 1) * GAP_COL) / N_COLS  # ~0.1138125

# Proportional aspect match to hex bounding box:
# Hex width = 390.7 m, Hex height = 425.6 m -> ratio = 0.9180
# (SUB_W * FIG_W) / (SUB_H * FIG_H) = 0.9180 -> perfect aspect match with zero wasted inset margins
SUB_H = (SUB_W * FIG_W) / (FIG_H * 0.9180)  # ~0.17635
GAP_ROW = 0.0050

ROW0_Y = 0.7550
ROW1_Y = ROW0_Y - SUB_H - GAP_ROW  # ~0.57365

# Overall Badge & Header for Panel (a)
badge_ax = fig.add_axes([M_L, 0.936, W_USABLE, 0.040], facecolor="none")
badge_ax.axis("off")
badge_ax.text(
    0.0, 0.50, "(a)",
    transform=badge_ax.transAxes, ha="left", va="center",
    fontsize=11.5, fontweight="bold", color=PALETTE["ink"],
    bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=PALETTE["ink_soft"], lw=0.9, alpha=0.98)
)
badge_ax.text(
    0.024, 0.50, "Spatial Distribution of ECW Tier Robustness Across All 16 Sampled Neighborhood Grids",
    transform=badge_ax.transAxes, ha="left", va="center",
    fontsize=10.5, fontweight="bold", color=PALETTE["ink"]
)

# Panel (a) Unified Horizontal Legend at top-right
legend_elements_a = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["teal"], markersize=7.5, label="Accessible (≥1.5 m)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["amber"], markersize=7.5, label="Restricted (0.9–1.5 m)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["rose"], markersize=7.5, label="Impassable (<0.9 m)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#64748b", markeredgecolor="white", markeredgewidth=1.0, markersize=7.5, label="Solid: Robust (58%)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor=PALETTE["ink"], markeredgewidth=1.8, markersize=8.5, label="Halo: Crosses threshold (42%)"),
]
leg_a = badge_ax.legend(
    handles=legend_elements_a, loc="center right", bbox_to_anchor=(1.0, 0.50),
    ncol=5, fontsize=7.2, frameon=True, facecolor="white", edgecolor=PALETTE["card_edge"],
    handletextpad=0.4, borderpad=0.4, columnspacing=1.0
)
leg_a.set_zorder(50)

# Tightly fitted hex spatial span: hex width = 390.7 m (half = 195.35), hex height = 425.6 m (half = 212.80)
# Minimal ~1% padding so hex boundaries fit snugly within the sub-panel frame
SPAN_X = 197.5
SPAN_Y = 215.14

# Render 16 sub-panels
for idx, h in enumerate(valid_hex_ids):
    row = idx // N_COLS
    col = idx % N_COLS
    x_pos = M_L + col * (SUB_W + GAP_COL)
    y_pos = ROW0_Y if row == 0 else ROW1_Y
    
    ax_sub = fig.add_axes([x_pos, y_pos, SUB_W, SUB_H], facecolor="#f8fafc")
    hp = valid_hexes.geometry[h]
    cx, cy = hp.centroid.x, hp.centroid.y
    
    # Query vector data by bbox centered on centroid
    b_idx = list(bld.sindex.intersection((cx - SPAN_X, cy - SPAN_Y, cx + SPAN_X, cy + SPAN_Y)))
    sub_bld = bld.iloc[b_idx]
    r_idx = list(roads.sindex.intersection((cx - SPAN_X, cy - SPAN_Y, cx + SPAN_X, cy + SPAN_Y)))
    sub_roads = roads.iloc[r_idx]
    
    # Draw hex background
    hex_poly_bg = Polygon(list(hp.exterior.coords), facecolor="#ffffff", edgecolor="none", zorder=1)
    ax_sub.add_patch(hex_poly_bg)
    
    # Plot buildings
    if len(sub_bld) > 0:
        sub_bld.plot(ax=ax_sub, facecolor=PALETTE["bld_face"], edgecolor=PALETTE["bld_edge"], lw=0.25, zorder=2)
    # Plot roads
    if len(sub_roads) > 0:
        sub_roads.plot(ax=ax_sub, color=PALETTE["road"], lw=0.6, zorder=3)
        
    # HEX GRID LINE EN ÜSTTE (ON TOP OF ROADS AND BUILDINGS)
    hex_poly_top = Polygon(list(hp.exterior.coords), facecolor="none", edgecolor="#1e293b", lw=1.5, zorder=22)
    ax_sub.add_patch(hex_poly_top)
    
    # Get nodes strictly assigned to this hex
    sub_nodes = n[n.hex_id == h]
    
    # Plot nodes
    for t in ["Impassable", "Restricted", "Accessible"]:
        q_rob = sub_nodes[(sub_nodes.tier3 == t) & sub_nodes.robust]
        if len(q_rob) > 0:
            ax_sub.scatter(q_rob.x_node, q_rob.y_node, s=24, facecolor=TIER_COLORS[t],
                           edgecolor="white", lw=0.7, zorder=25, alpha=0.95)
        q_unc = sub_nodes[(sub_nodes.tier3 == t) & (~sub_nodes.robust)]
        if len(q_unc) > 0:
            ax_sub.scatter(q_unc.x_node, q_unc.y_node, s=46, facecolor="none",
                           edgecolor=PALETTE["ink"], lw=1.4, zorder=26)
            ax_sub.scatter(q_unc.x_node, q_unc.y_node, s=32, facecolor="white",
                           edgecolor="none", zorder=27)
            ax_sub.scatter(q_unc.x_node, q_unc.y_node, s=16, facecolor=TIER_COLORS[t],
                           edgecolor="none", zorder=28)
            
    ax_sub.set_xlim(cx - SPAN_X, cx + SPAN_X)
    ax_sub.set_ylim(cy - SPAN_Y, cy + SPAN_Y)
    ax_sub.set_aspect("equal")
    ax_sub.set_xticks([])
    ax_sub.set_yticks([])
    for spine in ax_sub.spines.values():
        spine.set_visible(True)
        spine.set_color(PALETTE["card_edge"])
        spine.set_linewidth(0.7)
        
    # Standardized Chip: Sequential Grid 1-16 with H3 identifier
    chip_txt = f"Grid {idx+1} (H3-{h}) · n={len(sub_nodes)}"
    ax_sub.text(
        0.04, 0.96, chip_txt, transform=ax_sub.transAxes,
        fontsize=5.8, fontweight="bold", color=PALETTE["ink"], va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor=PALETTE["edge"], lw=0.6, alpha=0.96),
        zorder=35
    )

print("Rendered Panel (a) with 16 sub-panels successfully.")

# =============================================================================
# BOTTOM REGION: PANEL (b) at Bottom-Left, PANELS (c+d) at Bottom-Right
# =============================================================================
BOT_Y = 0.088
BOT_H = 0.430

# Panel (b) Bottom-Left
POS_B_X = M_L
POS_B_W = 0.445

# Panels (c+d) Bottom-Right
GAP_BC = 0.025
RIGHT_START_X = POS_B_X + POS_B_W + GAP_BC  # 0.510
RIGHT_TOTAL_W = M_R - RIGHT_START_X          # 0.465

POS_C_W = 0.252
GAP_CD = 0.035
POS_D_X = RIGHT_START_X + POS_C_W + GAP_CD  # 0.797
POS_D_W = M_R - POS_D_X                     # 0.178

def panel_header(ax, letter: str, title: str, x: float = 0.0, y: float = 1.040):
    """Draws aligned panel badge and title using transAxes to prevent label offset collisions."""
    ax.text(
        x, y, f"({letter.lower()})",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=11.5, fontweight="bold", color=PALETTE["ink"],
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=PALETTE["ink_soft"], lw=0.9, alpha=0.98),
        zorder=40, clip_on=False
    )
    ax_w_in = ax.get_position().width * FIG_W
    title_x = x + (0.380 / ax_w_in)
    ax.text(
        title_x, y + 0.005, title,
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=10.5, fontweight="bold", color=PALETTE["ink"],
        zorder=40, clip_on=False
    )

# -----------------------------------------------------------------------------
# PANEL (b): RANKED UNCERTAINTY SPECTRUM (BOTTOM LEFT)
# -----------------------------------------------------------------------------
pos_b = [POS_B_X, BOT_Y, POS_B_W, BOT_H]
ax_b = fig.add_axes(pos_b, facecolor="white")
panel_header(ax_b, "b", "Node-Level ECW Spectrum & 5th–95th Percentile Uncertainty Intervals")

YM = 4.5
r_sorted = n.sort_values("ecw_best").reset_index(drop=True)
x_vals = np.arange(len(r_sorted))

# Background threshold zones
ax_b.axhspan(0.0, 0.9, color=PALETTE["rose_faint"], lw=0, zorder=0)
ax_b.axhspan(0.9, 1.5, color=PALETTE["amber_faint"], lw=0, zorder=0)
ax_b.axhspan(1.5, YM, color=PALETTE["teal_faint"], lw=0, zorder=0)

# Whiskers
for k in ["Robust", "Crosses 0.9 m", "Crosses 1.5 m", "Crosses both thresholds"]:
    q_k = r_sorted[r_sorted.kind == k]
    lw_k = 1.0 if k == "Robust" else 1.3
    alpha_k = 0.50 if k == "Robust" else 0.85
    ax_b.vlines(q_k.index, q_k.lo.clip(upper=4.0), q_k.hi.clip(upper=4.0),
                color=CROSS_COLORS[k], lw=lw_k, alpha=alpha_k,
                label=f"{k} ({len(q_k)}, {100*len(q_k)/len(n):.0f}%)", zorder=3)

# Central estimate points
ax_b.scatter(x_vals, r_sorted.ecw_best.clip(upper=4.0), s=4.0, c=PALETTE["ink"], zorder=5)

# Threshold datum lines
ax_b.axhline(0.9, color=PALETTE["rose"], lw=1.1, ls="--", zorder=4)
ax_b.axhline(1.5, color=PALETTE["amber"], lw=1.1, ls="--", zorder=4)

# Threshold Labels
ax_b.text(len(r_sorted) + 6, 0.9, "0.9 m  Impassable", va="center", ha="left",
          fontsize=7.8, fontweight="bold", color=PALETTE["rose"],
          bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=0.9), zorder=6)
ax_b.text(len(r_sorted) + 6, 1.5, "1.5 m  Accessible", va="center", ha="left",
          fontsize=7.8, fontweight="bold", color=PALETTE["amber"],
          bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=0.9), zorder=6)

ax_b.set_ylim(0, YM)
ax_b.set_xlim(-8, len(r_sorted) + 85)
ax_b.set_xlabel(f"Nodes ranked by best-side ECW (N = {len(r_sorted)})", fontsize=9.0, fontweight="bold")
ax_b.set_ylabel("ECW (m; capped at 4.0 m)", fontsize=9.0, fontweight="bold")

# Tier segment indicators placed in top clear band
n_imp = (r_sorted.ecw_best < 0.9).sum()
n_res = ((r_sorted.ecw_best >= 0.9) & (r_sorted.ecw_best < 1.5)).sum()
n_acc = (r_sorted.ecw_best >= 1.5).sum()

ax_b.text(n_imp / 2, 4.25, f"Impassable ({n_imp}, {100*n_imp/len(n):.0f}%)", ha="center", va="center",
          fontsize=7.5, fontweight="bold", color=PALETTE["rose"],
          bbox=dict(boxstyle="round,pad=0.2", facecolor=PALETTE["rose_light"], edgecolor="none", alpha=0.85))
ax_b.text(n_imp + n_res / 2 + 10, 4.25, f"Restricted ({n_res}, {100*n_res/len(n):.0f}%)", ha="center", va="center",
          fontsize=7.5, fontweight="bold", color=PALETTE["amber"],
          bbox=dict(boxstyle="round,pad=0.2", facecolor=PALETTE["amber_light"], edgecolor="none", alpha=0.85))
ax_b.text(n_imp + n_res + n_acc / 2, 4.25, f"Accessible ({n_acc}, {100*n_acc/len(n):.0f}%)", ha="center", va="center",
          fontsize=7.5, fontweight="bold", color=PALETTE["teal"],
          bbox=dict(boxstyle="round,pad=0.2", facecolor=PALETTE["teal_light"], edgecolor="none", alpha=0.85))

# Legend
leg_b = ax_b.legend(loc="upper left", bbox_to_anchor=(0.015, 0.91), fontsize=7.6, frameon=True,
                    framealpha=0.92, edgecolor=PALETTE["card_edge"], title="Uncertainty Interval Classification",
                    title_fontsize=8.0)
leg_b.set_zorder(15)

# Footnote card below Panel b
ax_fn_b = fig.add_axes([pos_b[0], 0.018, pos_b[2], 0.045], facecolor="none")
ax_fn_b.axis("off")
card_b = FancyBboxPatch((0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0.02,rounding_size=0.08",
                        facecolor=PALETTE["card_bg"], edgecolor=PALETTE["card_edge"], lw=0.8,
                        transform=ax_fn_b.transAxes)
ax_fn_b.add_patch(card_b)
ax_fn_b.text(0.025, 0.68, "• Sample ranking: 557 surveyed nodes ordered by best-side ECW (30 Monte Carlo draws/side)",
             transform=ax_fn_b.transAxes, fontsize=7.5, color=PALETTE["ink_soft"], va="center")
ax_fn_b.text(0.025, 0.28, "• Overall stability: 321 nodes (58%) maintain observed tier across entire 5th–95th percentile span",
             transform=ax_fn_b.transAxes, fontsize=7.5, color=PALETTE["ink_soft"], va="center")

# -----------------------------------------------------------------------------
# PANEL (c): TIER SURVIVAL BREAKDOWN (BOTTOM RIGHT, LEFT PART)
# -----------------------------------------------------------------------------
pos_c = [RIGHT_START_X, BOT_Y, POS_C_W, BOT_H]
ax_c = fig.add_axes(pos_c, facecolor="white")
panel_header(ax_c, "c", "Tier Survival & Threshold Crossing Breakdown")

rows_c = [
    ("Accessible (≥1.5 m)", n[n.tier3 == "Accessible"]),
    ("Restricted (0.9–1.5 m)", n[n.tier3 == "Restricted"]),
    ("Impassable (<0.9 m)", n[n.tier3 == "Impassable"]),
    ("All nodes", n),
]

kinds_stack = ["Robust", "Crosses 0.9 m", "Crosses 1.5 m", "Crosses both thresholds"]
bar_h = 0.58

for i, (lab, df_tier) in enumerate(rows_c):
    left = 0.0
    tot = len(df_tier)
    for k in kinds_stack:
        cnt = (df_tier.kind == k).sum()
        pct = 100.0 * cnt / tot
        if cnt > 0:
            ax_c.barh(i, pct, left=left, color=CROSS_COLORS[k], height=bar_h,
                      edgecolor="white", lw=0.6)
            if pct >= 9.0:
                ax_c.text(left + pct / 2.0, i, f"{pct:.0f}%", ha="center", va="center",
                          color="white", fontsize=8.2, fontweight="bold")
            left += pct
    ax_c.text(102.5, i, f"n={tot}", va="center", fontsize=8.0, fontweight="bold", color=PALETTE["ink_soft"])

ax_c.set_yticks(range(len(rows_c)))
ax_c.set_yticklabels([lab.replace(" (", "\n(") for lab, _ in rows_c], fontsize=8.5, fontweight="bold")
ax_c.invert_yaxis()
ax_c.set_xlim(0, 116)
ax_c.set_xlabel("Proportion of nodes by uncertainty behavior (%)", fontsize=9.0, fontweight="bold")

# Footnote card below Panel c
ax_fn_c = fig.add_axes([pos_c[0], 0.018, pos_c[2], 0.045], facecolor="none")
ax_fn_c.axis("off")
card_c = FancyBboxPatch((0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0.02,rounding_size=0.08",
                        facecolor=PALETTE["card_bg"], edgecolor=PALETTE["card_edge"], lw=0.8,
                        transform=ax_fn_c.transAxes)
ax_fn_c.add_patch(card_c)
ax_fn_c.text(0.03, 0.68, "• Conservative envelope: 44% of all nodes robust",
             transform=ax_fn_c.transAxes, fontsize=7.5, color=PALETTE["ink_soft"], va="center")
ax_fn_c.text(0.03, 0.28, f"• Central estimate inside 5–95% interval: 98% ({int(((n.ecw_best >= n.lo0 - 1e-9) & (n.ecw_best <= n.hi0 + 1e-9)).sum())}/{len(n)} nodes)",
             transform=ax_fn_c.transAxes, fontsize=7.5, color=PALETTE["ink_soft"], va="center")

# -----------------------------------------------------------------------------
# PANEL (d): UNCERTAINTY WIDTH (ΔECW) DISTRIBUTION (BOTTOM RIGHT, RIGHT PART)
# -----------------------------------------------------------------------------
pos_d = [POS_D_X, BOT_Y, POS_D_W, BOT_H]
ax_d = fig.add_axes(pos_d, facecolor="white")
panel_header(ax_d, "d", "Uncertainty Span (ΔECW)")

tiers_order = ["Impassable", "Restricted", "Accessible"]
widths_data = [n[n.tier3 == t].unc_width.clip(upper=6.0) for t in tiers_order]

np.random.seed(42)
for idx, (t, w_vals) in enumerate(zip(tiers_order, widths_data)):
    jitter = np.random.normal(0, 0.06, size=len(w_vals))
    ax_d.scatter(idx + jitter, w_vals, s=8, color=TIER_COLORS[t], alpha=0.35, zorder=2)
    med = np.median(w_vals)
    q25, q75 = np.percentile(w_vals, [25, 75])
    ax_d.add_patch(Rectangle((idx - 0.22, q25), 0.44, q75 - q25,
                             facecolor="white", edgecolor=TIER_COLORS[t], lw=1.5, zorder=3, alpha=0.9))
    ax_d.plot([idx - 0.22, idx + 0.22], [med, med], color=TIER_COLORS[t], lw=2.4, zorder=4)
    w_min = np.percentile(w_vals, 5)
    w_max = np.percentile(w_vals, 95)
    ax_d.plot([idx, idx], [q75, w_max], color=TIER_COLORS[t], lw=1.2, zorder=3)
    ax_d.plot([idx, idx], [q25, w_min], color=TIER_COLORS[t], lw=1.2, zorder=3)
    ax_d.text(idx, med + 0.12, f"{med:.2f} m", ha="center", fontsize=7.8, fontweight="bold",
              color=TIER_COLORS[t], bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none", alpha=0.8), zorder=5)

ax_d.axhline(0.6, color=PALETTE["rose"], lw=1.2, ls=":", zorder=5)
ax_d.text(0.02, 0.65, "0.6 m band window", transform=ax_d.get_yaxis_transform(), ha="left", va="bottom",
          fontsize=7.2, fontweight="bold", color=PALETTE["rose"],
          bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none", alpha=0.85), zorder=6)

ax_d.set_xticks(range(len(tiers_order)))
ax_d.set_xticklabels(["Impassable", "Restricted", "Accessible"],
                     fontsize=8.2, fontweight="bold")
ax_d.set_ylim(0, 5.8)
ax_d.set_ylabel("Uncertainty span ΔECW = p95 - p05 (m)", fontsize=9.0, fontweight="bold")

# Insight box
ax_d.text(0.03, 0.94, "Why is Restricted least robust?\nMedian uncertainty span (1.44 m)\nis 2.4× the 0.6 m tier band window,\ncausing 85% transition rate.",
          transform=ax_d.transAxes, fontsize=7.2, color=PALETTE["ink_soft"], va="top",
          bbox=dict(boxstyle="round,pad=0.25", facecolor=PALETTE["card_bg"], edgecolor=PALETTE["card_edge"], lw=0.7))

# Footnote card below Panel d
ax_fn_d = fig.add_axes([pos_d[0], 0.018, pos_d[2], 0.045], facecolor="none")
ax_fn_d.axis("off")
card_d = FancyBboxPatch((0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0.02,rounding_size=0.08",
                        facecolor=PALETTE["card_bg"], edgecolor=PALETTE["card_edge"], lw=0.8,
                        transform=ax_fn_d.transAxes)
ax_fn_d.add_patch(card_d)
ax_fn_d.text(0.04, 0.68, "• Restricted window: 0.6 m wide",
             transform=ax_fn_d.transAxes, fontsize=7.5, color=PALETTE["ink_soft"], va="center")
ax_fn_d.text(0.04, 0.28, "• Median ΔECW: 1.44 m (2.4× window)",
             transform=ax_fn_d.transAxes, fontsize=7.5, color=PALETTE["rose"], fontweight="bold", va="center")

# Save primary publication figure at 300 DPI
out_primary = OUT / "fig05_robustness.png"
fig.savefig(out_primary, dpi=300, bbox_inches="tight", pad_inches=0.04, facecolor="white")
plt.close(fig)
print(f"Saved primary publication figure: {out_primary}")

# Sync to manuscript markedup and submission package
paths_sync = [
    REV / "05_manuscript_R1/markedup/figures/fig05_robustness.png",
    REV / "SUBMISSION_R1_PACKAGE/02_latex_source/figures/fig05_robustness.png"
]
for p in paths_sync:
    if p.parent.exists():
        shutil.copy2(out_primary, p)
        print(f"Synchronized figure to: {p}")

# Write Statistical Macros for LaTeX manuscript
N = len(n)
stat("robN", int(n.robust.sum()))
stat("robPct", round(100 * n.robust.mean(), 0))
stat("robAccPct", round(100 * n[n.tier3 == "Accessible"].robust.mean(), 0))
stat("robImpPct", round(100 * n[n.tier3 == "Impassable"].robust.mean(), 0))
stat("robResPct", round(100 * n[n.tier3 == "Restricted"].robust.mean(), 0))
stat("robEnvPct", round(100 * env_rob, 0))
stat("robCentralIn", int(((n.ecw_best >= n.lo0 - 1e-9) & (n.ecw_best <= n.hi0 + 1e-9)).sum()))
stat("robImpN", int(n[n.tier3 == "Impassable"].robust.sum()))
stat("robImpAll", int((n.tier3 == "Impassable").sum()))
print("Recorded statistical macros successfully.")
print(n.kind.value_counts().to_dict(), "env", round(env_rob, 3))
