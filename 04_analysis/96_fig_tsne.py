"""Figure 10: t-SNE Feature-Space Embedding of Pedestrian Network Nodes.

Panel (a): 2D t-SNE projection of 557 nodes across 7 multidimensional features
(corridor width, H/W, ECW, phantom capacity, and log footprints of vehicles,
furniture+plants, and infrastructure), colored by functional street classes with
1.3σ covariance ellipses and overlaid feature correlation biplot vectors.
Panel (b): Accessibility tier partition (Accessible >= 1.5 m, Restricted 0.9-1.5 m,
Impassable < 0.9 m) with overlaid 2D Gaussian KDE density contours for impassable nodes.
Deterministic seed (random_state=42, perplexity=30).
Redesigned to a standardized 3:2 landscape aspect ratio matching Figure 11.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
from sigfig_common import *

def build_fig10():
    n = load()
    
    # 7 Multi-dimensional features
    F = {
        "Corridor width": n.w_corridor,
        "H/W": n.hw.fillna(n.hw.median()),
        "ECW": n.ecw_best.clip(upper=12),
        "Phantom capacity": n.pc,
        "Vehicle area": np.log1p(n.g_Vehicles),
        "Furniture area": np.log1p(n.g_Furniture + n.g_Plants),
        "Infrastructure area": np.log1p(n.g_Infrastructure)
    }
    
    X = StandardScaler().fit_transform(pd.DataFrame(F).values)
    emb = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=2000,
               learning_rate="auto", init="pca").fit_transform(X)
    n["ex"], n["ey"] = emb[:, 0], emb[:, 1]
    sil = silhouette_score(emb, n["class"].astype("category").cat.codes.values)
    
    # Feature correlations with embedding axes
    cx = pd.DataFrame(F).apply(lambda s: np.corrcoef(s, n.ex)[0, 1])
    cy = pd.DataFrame(F).apply(lambda s: np.corrcoef(s, n.ey)[0, 1])
    
    # Canvas: Standardized 3:2 landscape publication aspect ratio (matching Figure 11)
    FIG_W = 13.30
    FIG_H = 9.00
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="#FFFFFF")
    
    # -------------------------------------------------------------------------
    # MAIN PANELS (a) & (b) COORDINATES
    # -------------------------------------------------------------------------
    panel_w = 0.428
    panel_h = 0.815
    panel_y = 0.085
    ax1_x = 0.060
    ax2_x = 0.535
    
    # -------------------------------------------------------------------------
    # PANEL (a): FUNCTIONAL STREET CLASS MANIFOLD & FEATURE BIPLOT
    # -------------------------------------------------------------------------
    ax1 = fig.add_axes([ax1_x, panel_y, panel_w, panel_h])
    ax1.set_facecolor("#FFFFFF")
    
    # Draw Covariance Ellipses and Scatter
    for c in CLASSES:
        d = n[n["class"] == c]
        conf_ellipse(d.ex.values, d.ey.values, ax1, n_std=1.3,
                     facecolor=CLS_COL[c], alpha=0.10, edgecolor=CLS_COL[c],
                     lw=1.3, ls="--", zorder=1)
        ax1.scatter(d.ex, d.ey, c=CLS_COL[c], s=34, marker=CLS_MARK[c],
                    alpha=0.85, edgecolors="white", lw=0.6, zorder=3)
                    
    # Overlay Feature Correlation Biplot Vectors directly inside the cluster cloud
    ox, oy = 0.0, 0.0
    vec_scale = 18.0
    vector_colors = {
        "ECW": "#0284c7",
        "Corridor width": "#059669",
        "Phantom capacity": "#b91c1c",
        "Vehicle area": "#dc2626",
        "Furniture area": "#d97706",
        "Infrastructure area": "#7c3aed",
        "H/W": "#64748b"
    }
    
    label_offsets = {
        "ECW": (1.2, 0.0, "left", "center"),
        "Corridor width": (1.0, 0.8, "left", "bottom"),
        "Phantom capacity": (-1.0, 0.8, "right", "bottom"),
        "Vehicle area": (-1.2, 0.5, "right", "center"),
        "Furniture area": (0.0, 1.2, "center", "bottom"),
        "Infrastructure area": (1.0, 0.8, "left", "bottom"),
        "H/W": (-1.0, -0.8, "right", "top")
    }
    
    ax1.scatter([ox], [oy], color="#475569", s=30, marker="+", zorder=7, lw=1.4)
    
    for k in F:
        vx, vy = cx[k] * vec_scale, cy[k] * vec_scale
        ax1.annotate("", xy=(ox + vx, oy + vy), xytext=(ox, oy),
                     arrowprops=dict(arrowstyle="-|>", color=vector_colors[k], lw=1.5, mutation_scale=8.0),
                     zorder=7)
        dx, dy, ha_align, va_align = label_offsets[k]
        txt = ax1.text(ox + vx + dx, oy + vy + dy, k,
                       fontsize=6.8, fontweight="bold", color=vector_colors[k],
                       ha=ha_align, va=va_align, zorder=9)
        txt.set_path_effects([pe.withStroke(linewidth=2.6, foreground="white")])
    
    # Clean Unboxed Legend (no filled container card)
    legend_handles_a = [
        Line2D([0], [0], marker=CLS_MARK[c], color="w", label=f"{c}  (n = {len(n[n['class'] == c])})",
               markerfacecolor=CLS_COL[c], markersize=7.0, markeredgewidth=0.6, markeredgecolor="white")
        for c in CLASSES
    ]
    leg_a = ax1.legend(handles=legend_handles_a, title="FUNCTIONAL STREET CLASS",
                       loc="upper left", frameon=False, fontsize=7.2, title_fontsize=7.6, labelspacing=0.40)
    leg_a._legend_box.align = "left"
    leg_a.get_title().set_fontweight("bold")
    leg_a.get_title().set_color("#475569")
    for text in leg_a.get_texts():
        text.set_color("#334155")
                    
    ax1.set_xlabel("t-SNE dimension 1 (ECW & Corridor Capacity Axis)", fontsize=8.2, fontweight="bold", color="#475569")
    ax1.set_ylabel("t-SNE dimension 2 (Obstacle Clutter Axis)", fontsize=8.2, fontweight="bold", color="#475569")
    ax1.tick_params(labelsize=7.4, colors="#64748b")
    ax1.grid(True, linestyle="--", alpha=0.35, color="#cbd5e1")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_color("#cbd5e1")
    ax1.spines["bottom"].set_color("#cbd5e1")
    
    # Panel (a) Badge & Title (Standardized style matching fig05-fig11)
    HEADER_Y = 0.958
    SUB_Y = 0.933
    bbox_badge = dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#64748b", lw=0.9, alpha=0.98)
    
    fig.text(ax1_x, HEADER_Y, "(a)", fontsize=11.5, fontweight="bold", color="#0f172a",
             bbox=bbox_badge, va="center", ha="left", transform=fig.transFigure, zorder=10)
    fig.text(ax1_x + 0.030, HEADER_Y, "Functional street class manifold & morphological feature biplot",
             fontsize=9.2, fontweight="bold", color="#1e293b", va="center", ha="left", transform=fig.transFigure, zorder=10)
    fig.text(ax1_x + 0.030, SUB_Y, f"2D t-SNE projection (n = 557) with 1.3σ covariance ellipses | Silhouette s = {sil:.3f}",
             fontsize=7.0, color="#64748b", va="center", ha="left", transform=fig.transFigure, zorder=10)

    # -------------------------------------------------------------------------
    # PANEL (b): ACCESSIBILITY TIER PARTITION & IMPASSABLE DENSITY
    # -------------------------------------------------------------------------
    ax2 = fig.add_axes([ax2_x, panel_y, panel_w, panel_h])
    ax2.set_facecolor("#FFFFFF")
    
    TM = {"Impassable": "X", "Restricted": "s", "Accessible": "o"}
    tier_colors = {"Impassable": "#dc2626", "Restricted": "#d97706", "Accessible": "#0284c7"}
    
    # Impassable 2D KDE density contours (clean isolines, no warm background fill)
    imp = n[n.tier3 == "Impassable"]
    gx, gy = np.mgrid[n.ex.min() - 5:n.ex.max() + 5:200j, n.ey.min() - 5:n.ey.max() + 5:200j]
    dens = gaussian_kde(np.c_[imp.ex, imp.ey].T, bw_method=0.22)(np.c_[gx.ravel(), gy.ravel()].T).reshape(gx.shape)
    
    ax2.contour(gx, gy, dens, levels=4, colors=["#f87171"], linewidths=0.75, linestyles="--", alpha=0.75, zorder=1)
    
    # Scatter nodes by tier (Accessible first, Impassable on top)
    for t in ["Accessible", "Restricted", "Impassable"]:
        d = n[n.tier3 == t]
        ax2.scatter(d.ex, d.ey, c=tier_colors[t], s=34, marker=TM[t],
                    alpha=0.82, edgecolors="white", lw=0.6, zorder=3)
                    
    # Clean Unboxed Legend (no filled container card)
    tier_order = [
        ("Accessible", "Accessible (≥ 1.5 m)", 345, 61.9),
        ("Restricted", "Restricted (0.9–1.5 m)", 84, 15.1),
        ("Impassable", "Impassable (< 0.9 m)", 128, 23.0)
    ]
    legend_handles_b = [
        Line2D([0], [0], marker=TM[t_key], color="w", label=f"{t_lab}  ·  n = {t_n} ({t_pct:.1f}%)",
               markerfacecolor=tier_colors[t_key], markersize=7.0, markeredgewidth=0.6, markeredgecolor="white")
        for (t_key, t_lab, t_n, t_pct) in tier_order
    ] + [
        Line2D([0], [0], color="#f87171", lw=1.2, ls="--", label="Impassable density (2D KDE isolines)")
    ]
    leg_b = ax2.legend(handles=legend_handles_b, title="EFFECTIVE CLEAR WIDTH (ECW) TIER",
                       loc="upper left", frameon=False, fontsize=7.2, title_fontsize=7.6, labelspacing=0.40)
    leg_b._legend_box.align = "left"
    leg_b.get_title().set_fontweight("bold")
    leg_b.get_title().set_color("#475569")
    for text in leg_b.get_texts():
        text.set_color("#334155")

    ax2.set_xlabel("t-SNE dimension 1 (ECW & Corridor Capacity Axis)", fontsize=8.2, fontweight="bold", color="#475569")
    ax2.set_ylabel("t-SNE dimension 2 (Obstacle Clutter Axis)", fontsize=8.2, fontweight="bold", color="#475569")
    ax2.tick_params(labelsize=7.4, colors="#64748b")
    ax2.grid(True, linestyle="--", alpha=0.35, color="#cbd5e1")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_color("#cbd5e1")
    ax2.spines["bottom"].set_color("#cbd5e1")
    
    # Match limits with ax1
    x_min = min(ax1.get_xlim()[0], ax2.get_xlim()[0])
    x_max = max(ax1.get_xlim()[1], ax2.get_xlim()[1])
    y_min = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
    y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    
    # Panel (b) Header
    fig.text(ax2_x, HEADER_Y, "(b)", fontsize=11.5, fontweight="bold", color="#0f172a",
             bbox=bbox_badge, va="center", ha="left", transform=fig.transFigure, zorder=10)
    fig.text(ax2_x + 0.030, HEADER_Y, "Accessibility tier partition & impassable node density",
             fontsize=9.2, fontweight="bold", color="#1e293b", va="center", ha="left", transform=fig.transFigure, zorder=10)
    fig.text(ax2_x + 0.030, SUB_Y, f"Clearance tier segmentation with 2D Gaussian KDE density contours (n = {len(imp)}, 23.0%)",
             fontsize=7.0, color="#64748b", va="center", ha="left", transform=fig.transFigure, zorder=10)

    # -------------------------------------------------------------------------
    # SAVE & STATS
    # -------------------------------------------------------------------------
    out_primary = OUT / "fig10_tsne.png"
    fig.savefig(out_primary, dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)
    print(f"Saved primary publication figure: {out_primary}")

    # Synchronize across manuscript and submission package directories
    paths_sync = [
        REV / "05_manuscript_R1/figures/fig10_tsne.png",
        REV / "05_manuscript_R1/markedup/figures/fig10_tsne.png",
        REV / "SUBMISSION_R1_PACKAGE/02_latex_source/figures/fig10_tsne.png"
    ]
    for p in paths_sync:
        if p.resolve() != out_primary.resolve() and p.parent.exists():
            shutil.copy2(out_primary, p)
            print(f"Synchronized figure to: {p}")

    stat("tsneSil", round(float(sil), 3))
    stat("tsneFeatures", len(F))
    print(f"fig10_tsne generated successfully. Silhouette: {sil:.3f}, Features: {len(F)}")

if __name__ == "__main__":
    build_fig10()
