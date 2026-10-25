"""
Figure 6: Overview of obstacle footprint and composition.
Generates a 10/10 pixel-perfect publication figure with a 5:3 width ratio
between Panel a (5 parts, 1:1 polar circle) and Panel b (3 parts, structured matrix)
sharing the exact same height.
"""
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import scipy.ndimage

from sigfig_common import *

def get_luminance(hex_code):
    """Calculate relative perceptual luminance of a hex color."""
    hex_code = hex_code.lstrip("#")
    r, g, b = [int(hex_code[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
    return 0.299 * r + 0.587 * g + 0.114 * b

def build_fig06():
    n = load()
    
    # =========================================================================
    # MINIMIZED MARGINS & COMPACT ACADEMIC GEOMETRY
    # =========================================================================
    PANEL_H_IN = 7.65
    PANEL_A_W_IN = 7.65
    PANEL_B_W_IN = 5.10
    GAP_IN = 0.85
    LEFT_MARGIN_IN = 0.50
    
    FIG_W = 14.60
    FIG_H = 9.20
    
    FOOT_Y = 0.015
    FOOT_H = 0.088
    
    MAIN_Y = 0.115
    MAIN_H = PANEL_H_IN / FIG_H  # 7.65 / 9.20 = 0.8315
    
    POS_A_X = LEFT_MARGIN_IN / FIG_W
    POS_A_W = PANEL_A_W_IN / FIG_W
    
    POS_B_X = POS_A_X + POS_A_W + (GAP_IN / FIG_W)
    POS_B_W = PANEL_B_W_IN / FIG_W
    
    HDR_Y = MAIN_Y + MAIN_H + 0.008
    
    INK = "#0f172a"
    INK_MUTED = "#64748b"
    
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="#FFFFFF", dpi=300)
    
    tot = float(n.obs_total.sum())
    n_nodes = len(n)
    n_imp = int((n.tier3 == 'Impassable').sum())
    pct_imp = 100.0 * n_imp / n_nodes
    mean_pc = 100.0 * float(n.pc.mean())
    dom = n[[f"g_{g}" for g in GROUPS]].sum().idxmax()[2:]
    dom_pct = 100.0 * float(n[f"g_{dom}"].sum()) / tot
    
    # Panel (a) Badge (shifted slightly right and down)
    fig.text(POS_A_X + 0.012, HDR_Y - 0.006, "(a)", fontsize=11.5, fontweight="bold", color=INK,
             bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=INK_MUTED, lw=0.9, alpha=0.98),
             va="center", ha="left", transform=fig.transFigure, zorder=10)

    # Panel (b) Badge (shifted slightly right and down)
    fig.text(POS_B_X + 0.012, HDR_Y - 0.006, "(b)", fontsize=11.5, fontweight="bold", color=INK,
             bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=INK_MUTED, lw=0.9, alpha=0.98),
             va="center", ha="left", transform=fig.transFigure, zorder=10)

    # =========================================================================
    # PANEL A: RADIAL STACKED BAR RING (POLAR AXES)
    # =========================================================================
    ax1 = fig.add_axes([POS_A_X, MAIN_Y, POS_A_W, MAIN_H], projection="polar")
    
    rows, sectors = [], []
    for c in CLASSES:
        d = n[n["class"] == c].sort_values("ecw_best")
        start = len(rows)
        rows += [r for r in d[list(OBJ)].values]
        sectors.append((c, start, len(rows)))
        rows.append(np.zeros(len(OBJ))) # gap between classes
        
    M = np.array(rows)
    N = len(M)
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    w = 2 * np.pi / N * 0.92
    bottom = np.zeros(N)
    
    for j, k in enumerate(OBJ):
        v = np.log1p(M[:, j])
        ax1.bar(ang, v, width=w, bottom=bottom, color=OBJ[k], edgecolor="white", lw=0.15, zorder=3)
        bottom += v
        
    mx = bottom.max()
    ax1.set_theta_offset(np.pi / 2)
    ax1.set_theta_direction(-1)
    
    # Calibrated radial limits to perfectly accommodate Collector avenue trend (peaks at 6.74)
    r_hole = -2.25
    r_outer = 7.80
    ax1.set_ylim(r_hole, r_outer)
    ax1.set_frame_on(False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.grid(False)
    
    # Inner category circle, filled trendlines, and sector radial dividers
    for c, a, b in sectors:
        cnt = b - a
        th_sec = ang[a:b]
        sec_totals = bottom[a:b]
        sigma = max(1.5, cnt / 18.0)
        smooth_trend = scipy.ndimage.gaussian_filter1d(sec_totals, sigma=sigma, mode="nearest")
        
        # 1. Innermost category circle: bold colored foundation arc at r = 0
        th_arc = np.linspace(ang[a], ang[b - 1] + w, 100)
        ax1.plot(th_arc, np.zeros_like(th_arc), color=CLS_COL[c], lw=8.0, solid_capstyle="butt", zorder=6)
        
        # 2. Rich filled trendline expanding outward
        ax1.fill_between(th_sec, 0, smooth_trend, color=CLS_COL[c], alpha=0.32, zorder=2)
        # 3. Crisp trendline stroke
        ax1.plot(th_sec, smooth_trend, color=CLS_COL[c], lw=2.4, zorder=4, ls="-")
        
        # 4. Subtle radial divider line at start of sector
        div_th = ang[a] - w * 0.5
        sec_max = sec_totals.max()
        ax1.plot([div_th, div_th], [0, max(sec_max * 1.02, 3.8)], color="#cbd5e1", lw=0.75, ls="--", zorder=1)
        
        # 5. Snug adaptive class label right outside this sector's data
        mid = (ang[a] + ang[b - 1]) / 2
        c_sub = n[n["class"] == c]
        mean_obs = float(c_sub.obs_total.mean())
        
        lbl_r = max(np.percentile(sec_totals, 95) + 0.55, smooth_trend.max() + 0.65)
        lbl_r = min(lbl_r, r_outer * 0.96)
        
        ax1.text(mid, lbl_r, f"{c}\n(n={cnt} · {mean_obs:.1f} m²)",
                 ha="center", va="center", fontsize=7.8, fontweight="bold", color=CLS_COL[c],
                 bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=CLS_COL[c], lw=0.85, alpha=0.96),
                 zorder=10)

    # Concentric Reference Footprint Rings
    th_full = np.linspace(0, 2 * np.pi, 200)
    for ref_v, ref_lbl in [(np.log1p(2), "2 m²"), (np.log1p(10), "10 m²"), (np.log1p(30), "30 m²"), (np.log1p(100), "100 m²")]:
        ax1.plot(th_full, np.full_like(th_full, ref_v), color="#94a3b8", ls=":", lw=0.65, alpha=0.7, zorder=2)
        ax1.text(0.04, ref_v + 0.05, ref_lbl, color="#64748b", fontsize=6.8, ha="left", va="bottom", zorder=5)

    # =========================================================================
    # PANEL A: CENTRAL FOCAL MEDALLION (COMPACT INNER HUB MATCHING r_hole)
    # =========================================================================
    ax_circ = fig.add_axes([POS_A_X, MAIN_Y, POS_A_W, MAIN_H], facecolor="none")
    ax_circ.axis("off")
    ax_circ.set_xlim(0, 1)
    ax_circ.set_ylim(0, 1)
    
    med_frac = abs(r_hole) / (r_outer - r_hole) * 0.5
    medallion = patches.Circle((0.5, 0.5), med_frac, facecolor="white", edgecolor="#cbd5e1",
                                linewidth=1.2, zorder=7, transform=ax_circ.transAxes)
    ax_circ.add_patch(medallion)
    medallion_inner = patches.Circle((0.5, 0.5), med_frac - 0.008, facecolor="#f8fafc", edgecolor="#e2e8f0",
                                     linewidth=0.6, linestyle="--", zorder=8, transform=ax_circ.transAxes)
    ax_circ.add_patch(medallion_inner)
    
    ax_circ.text(0.5, 0.585, "TOTAL CLUTTER", ha="center", va="center",
                 fontsize=7.6, fontweight="bold", color="#475569", transform=ax_circ.transAxes, zorder=9)
    ax_circ.text(0.5, 0.535, f"{tot:,.0f} m²", ha="center", va="center",
                 fontsize=15.5, fontweight="bold", color="#b91c1c", transform=ax_circ.transAxes, zorder=9)
    ax_circ.text(0.5, 0.485, f"Impassable: {pct_imp:.0f}% ({n_imp} nodes)", ha="center", va="center",
                 fontsize=7.4, fontweight="bold", color="#b91c1c",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="#fee2e2", edgecolor="#fca5a5", lw=0.5),
                 transform=ax_circ.transAxes, zorder=9)
    ax_circ.text(0.5, 0.440, f"Dominant: Vehicles ({dom_pct:.0f}%)", ha="center", va="center",
                 fontsize=7.2, color="#475569", transform=ax_circ.transAxes, zorder=9)

    # -------------------------------------------------------------------------
    # PANEL A: TREND & RADIAL GUIDE LEGEND (Upper Left Open Quadrant)
    # -------------------------------------------------------------------------
    leg_elements_a = [
        Line2D([0], [0], color="#475569", lw=2.4, label="Class trendline (Gaussian smooth)"),
        patches.Patch(facecolor="#94a3b8", alpha=0.32, edgecolor="none", label="Cumulative clutter envelope"),
        Line2D([0], [0], color="#94a3b8", ls=":", lw=1.0, label="Concentric log(1+m²) reference rings"),
        Line2D([0], [0], color="#cbd5e1", ls="--", lw=0.8, label="Street class sector divider"),
    ]
    leg_a = ax_circ.legend(handles=leg_elements_a, loc="upper left", bbox_to_anchor=(0.01, 0.99),
                           frameon=True, facecolor="white", edgecolor="#e2e8f0", framealpha=0.92,
                           fontsize=7.0, title="RADIAL PLOT GUIDE",
                           title_fontsize=7.4, labelspacing=0.38)
    leg_a._legend_box.align = "left"
    leg_a.get_title().set_fontweight("bold")
    leg_a.get_title().set_color("#475569")
    for text in leg_a.get_texts():
        text.set_color("#334155")

    # =========================================================================
    # PANEL A: FOOTER (STANDARDIZED COLOR LEGEND CARD)
    # =========================================================================
    ax_a_foot = fig.add_axes([POS_A_X, FOOT_Y, POS_A_W, FOOT_H], facecolor="none")
    ax_a_foot.axis("off")
    
    a_foot_card = FancyBboxPatch((0.0, 0.0), 1.0, 1.0,
                                 boxstyle="round,pad=0.0,rounding_size=0.04",
                                 facecolor="none", edgecolor="#cbd5e1", linewidth=0.85,
                                 transform=ax_a_foot.transAxes, zorder=1)
    ax_a_foot.add_patch(a_foot_card)
    
    col_x_offsets = [0.03, 0.22, 0.39, 0.56]
    for idx, (grp, o_list) in enumerate(GROUPS.items()):
        gx = col_x_offsets[idx]
        ax_a_foot.text(gx, 0.82, grp.upper(), fontsize=7.8, fontweight="bold", color=GROUP_COL[grp],
                       va="center", transform=ax_a_foot.transAxes, zorder=2)
        n_items = len(o_list)
        for i_idx, o_name in enumerate(o_list):
            if grp == "Infrastructure":
                sub_col = i_idx // 3
                row_in_sub = i_idx % 3
                ix = gx + sub_col * 0.20
                iy = 0.56 - row_in_sub * 0.22
            else:
                ix = gx
                iy = 0.56 - i_idx * 0.22
            swatch = patches.Rectangle((ix, iy - 0.060), 0.022, 0.12, facecolor=OBJ[o_name],
                                       edgecolor="none", transform=ax_a_foot.transAxes, zorder=2)
            ax_a_foot.add_patch(swatch)
            ax_a_foot.text(ix + 0.030, iy, OBJ_LABEL[o_name], fontsize=7.2, color="#334155",
                           va="center", transform=ax_a_foot.transAxes, zorder=2)

    # =========================================================================
    # PANEL B: STRUCTURED MATRIX (SQUARE TILES & ISOLATED FUNCTIONAL BRACKETS)
    # =========================================================================
    ax2 = fig.add_axes([POS_B_X, MAIN_Y, POS_B_W, MAIN_H], facecolor="white")
    
    mean = n.groupby("class")[list(OBJ)].mean().reindex(CLASSES)
    all_mean = n[list(OBJ)].mean()
    class_totals = mean.sum(axis=1) # total m2 per node for each class
    overall_total = float(all_mean.sum()) # overall total m2 per node
    
    order = [
        "electrical_cabinet", "street_light", "trash_can", "utility_pole", "dumpster", "utility_box", # Infra (1..6)
        "potted_plant", # Plant (7)
        "table", "chair", # Furniture (8, 9)
        "motorcycle", "truck", "car" # Vehicles (10, 11, 12)
    ]
    
    num_cols = len(CLASSES) + 1 # 5 classes + 1 All Classes column (0..5)
    num_rows = len(order) + 1 # 12 obstacles + 1 Marginal Total row (0..12)
    
    # Draw Background Cells / Tiles
    for i in range(num_cols):
        for j in range(num_rows):
            is_marginal_col = (i == len(CLASSES))
            is_marginal_row = (j == 0)
            
            if is_marginal_col and is_marginal_row:
                cell_bg = "#e2e8f0" # corner intersection
                edge_col = "#cbd5e1"
                lw_cell = 0.95
            elif is_marginal_col or is_marginal_row:
                cell_bg = "#f1f5f9" # marginal highlight
                edge_col = "#e2e8f0"
                lw_cell = 0.80
            else:
                cell_bg = "#ffffff" if (i + j) % 2 == 0 else "#f8fafc"
                edge_col = "#e2e8f0"
                lw_cell = 0.55
                
            cell = FancyBboxPatch((i - 0.48, j - 0.45), 0.96, 0.90,
                                  boxstyle="round,pad=0.0,rounding_size=0.06",
                                  facecolor=cell_bg, edgecolor=edge_col, linewidth=lw_cell,
                                  zorder=1)
            ax2.add_patch(cell)

    # Bottom Marginal Row: Total Footprint per Node (at j = 0)
    for i, c in enumerate(CLASSES):
        ctot = class_totals[c]
        ax2.text(i, 0, f"{ctot:.1f} m²", ha="center", va="center",
                 fontsize=8.5, fontweight="bold", color=CLS_COL[c], zorder=4)
    # Overall citywide total
    ax2.text(len(CLASSES), 0, f"{overall_total:.1f} m²", ha="center", va="center",
             fontsize=8.8, fontweight="bold", color="#0f172a", zorder=4)

    # Plot Matrix Bubbles for rows j = 1..12
    for j_idx, k in enumerate(order):
        y_pos = j_idx + 1 # rows 1 to 12
        lum = get_luminance(OBJ[k])
        
        # 5 Street Classes
        for i, c in enumerate(CLASSES):
            v = float(mean.loc[c, k])
            if v < 0.02:
                # Negligible: draw a subtle dash
                ax2.text(i, y_pos, "—", ha="center", va="center",
                         fontsize=7.5, color="#94a3b8", zorder=2)
                continue
                
            s_val = np.log1p(v * 6.5) * 380
            
            if v >= 0.10:
                ax2.scatter(i, y_pos, s=s_val, color=OBJ[k], alpha=0.90, edgecolors="white", lw=1.1, zorder=3)
                txt_col = "white" if (lum < 0.62 and s_val > 150) else "#1e293b"
                lbl_str = f"{v:.1f}"
                ax2.text(i, y_pos, lbl_str, ha="center", va="center",
                         fontsize=7.4, fontweight="bold", color=txt_col, zorder=4)
            else:
                # Small (0.02 <= v < 0.10)
                ax2.text(i, y_pos, f"{v:.2f}", ha="center", va="center",
                         fontsize=6.8, fontweight="bold", color="#475569", zorder=4)
                         
        # Marginal Column (All Classes / Citywide Average) at i = len(CLASSES)
        v_all = float(all_mean[k])
        s_all = np.log1p(v_all * 6.5) * 380
        if v_all >= 0.10:
            ax2.scatter(len(CLASSES), y_pos, s=s_all, color=OBJ[k], alpha=0.92, edgecolors="#334155", lw=0.8, zorder=3)
            txt_col = "white" if (lum < 0.62 and s_all > 150) else "#1e293b"
            lbl_str = f"{v_all:.1f}"
            ax2.text(len(CLASSES), y_pos, lbl_str, ha="center", va="center",
                     fontsize=7.4, fontweight="bold", color=txt_col, zorder=4)
        else:
            ax2.text(len(CLASSES), y_pos, f"{v_all:.2f}", ha="center", va="center",
                     fontsize=6.8, fontweight="bold", color="#475569", zorder=4)

    # Set Y-axis Labels
    y_tick_positions = [0] + list(range(1, 13))
    y_tick_labels = ["Total Footprint\n(m²/node)"] + [OBJ_LABEL[k] for k in order]
    ax2.set_yticks(y_tick_positions)
    ax2.set_yticklabels(y_tick_labels, fontsize=8.4, fontweight="normal")
    ax2.get_yticklabels()[0].set_fontweight("bold")
    ax2.get_yticklabels()[0].set_color("#0f172a")

    # Column Headers at Top of Matrix (y = 12.95)
    for idx, c in enumerate(CLASSES):
        n_c = int((n['class'] == c).sum())
        c_lines = c.replace(" ", "\n", 1) + f"\n(n={n_c})"
        ax2.text(idx, 12.95, c_lines, ha="center", va="bottom",
                 fontsize=8.0, fontweight="bold", color=CLS_COL[c], zorder=5)
    # All Classes Header
    ax2.text(len(CLASSES), 12.95, "All Classes\n(Citywide)\n(n=557)", ha="center", va="bottom",
             fontsize=8.0, fontweight="bold", color="#0f172a", zorder=5)

    ax2.set_xticks([])
    ax2.set_xlim(-0.95, num_cols - 0.35)
    ax2.set_ylim(-0.75, 13.9)
    
    for s in ["top", "right", "left", "bottom"]:
        ax2.spines[s].set_visible(False)
    ax2.tick_params(axis="both", which="both", length=0)
    ax2.grid(False)

    # Functional Group Accent Brackets & Labels on Left of Y-axis
    group_ranges = [
        ("INFRASTRUCTURE", 1, 6, GROUP_COL["Infrastructure"]),
        ("PLANTS", 7, 7, GROUP_COL["Plants"]),
        ("FURNITURE", 8, 9, GROUP_COL["Furniture"]),
        ("VEHICLES", 10, 12, GROUP_COL["Vehicles"]),
    ]
    for g_lbl, y_s, y_e, g_col in group_ranges:
        bracket_x = -0.72
        ax2.plot([bracket_x, bracket_x], [y_s - 0.38, y_e + 0.38], color=g_col, lw=3.0,
                 solid_capstyle="round", clip_on=False, zorder=5)
        mid_y = (y_s + y_e) / 2.0
        ax2.text(bracket_x - 0.16, mid_y, g_lbl, ha="center", va="center",
                 fontsize=6.4, fontweight="bold", color=g_col, rotation=90, clip_on=False, zorder=5)

    # =========================================================================
    # PANEL B: FOOTER (BUBBLE SCALE & MATRIX ANNOTATIONS CARD)
    # =========================================================================
    ax_b_foot = fig.add_axes([POS_B_X, FOOT_Y, POS_B_W, FOOT_H], facecolor="none")
    ax_b_foot.axis("off")
    
    b_foot_card = FancyBboxPatch((0.0, 0.0), 1.0, 1.0,
                                 boxstyle="round,pad=0.0,rounding_size=0.04",
                                 facecolor="none", edgecolor="#cbd5e1", linewidth=0.85,
                                 transform=ax_b_foot.transAxes, zorder=1)
    ax_b_foot.add_patch(b_foot_card)
    
    # Title & description
    ax_b_foot.text(0.04, 0.78, "BUBBLE SCALE & ANNOTATIONS",
                   fontsize=7.4, fontweight="bold", color="#475569", va="center", transform=ax_b_foot.transAxes, zorder=2)
    ax_b_foot.text(0.04, 0.38, "Bubble area reflects obstacle footprint (m²).\nEm-dash (—) indicates negligible clutter\n(< 0.02 m²/node).",
                   fontsize=6.6, color="#64748b", va="center", transform=ax_b_foot.transAxes, linespacing=1.25, zorder=2)
                   
    # Horizontal Scale Circles shifted to the right with ample clearance from text
    scale_vals = [0.5, 2.0, 5.0, 10.0, 25.0]
    ax_b_foot.text(0.52, 0.78, "SCALE (m²/node):", fontsize=6.8, fontweight="bold", color="#64748b",
                   ha="left", va="center", transform=ax_b_foot.transAxes, zorder=2)
                   
    for s_idx, sv in enumerate(scale_vals):
        sx_pos = 0.54 + s_idx * 0.072
        sy_pos = 0.38
        s_marker = np.log1p(sv * 6.5) * 380
        ax_b_foot.scatter(sx_pos, sy_pos, s=s_marker, color="#94a3b8", alpha=0.75,
                          edgecolors="white", lw=0.6, transform=ax_b_foot.transAxes, zorder=3)
        ax_b_foot.text(sx_pos, 0.12, f"{sv:g}", ha="center", va="center",
                       fontsize=6.2, fontweight="bold", color="#475569", transform=ax_b_foot.transAxes, zorder=3)

    # -------------------------------------------------------------------------
    # SAVE & SYNCHRONIZE 300 DPI PUBLICATION FIGURE
    # -------------------------------------------------------------------------
    out_primary = OUT / "fig06_overview.png"
    fig.savefig(out_primary, dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)
    print(f"Saved primary publication figure: {out_primary}")
    
    # Sync to manuscript markedup and submission package
    paths_sync = [
        REV / "05_manuscript_R1/markedup/figures/fig06_overview.png",
        REV / "SUBMISSION_R1_PACKAGE/02_latex_source/figures/fig06_overview.png"
    ]
    for p in paths_sync:
        if p.parent.exists():
            shutil.copy2(out_primary, p)
            print(f"Synchronized figure to: {p}")
            
    # Save statistics for LaTeX macros
    stat("ovTotalSqm", round(float(tot), 0))
    stat("ovDomPct", round(float(dom_pct), 1))
    stat("ovCarSqmBlvd", round(float(mean.loc["Boulevard", "car"]), 1))
    stat("ovCarSqmNarrow", round(float(mean.loc["Narrow local street", "car"]), 1))
    stat("ovFurnSqmCollector", round(float(mean.loc["Collector avenue", ["chair", "table"]].sum()), 1))

if __name__ == "__main__":
    build_fig06()
