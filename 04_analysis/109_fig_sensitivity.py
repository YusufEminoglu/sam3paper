"""Figure 11: Sensitivity Analysis Suite.

Comprehensive 3-panel plate layout (Target 3:2 landscape aspect ratio):
- Panel (a): Parametric Perturbation Tornado: Differential shift in impassable share (<0.9 m)
  across 12 modelling variants relative to the central scenario (22.2%).
- Panel (b): Empirical Cumulative Distribution Functions (CDFs) of node-level ECW across 536 common nodes,
  showing the tight parametric tolerance envelope vs. the radical rightward shift under vehicle removal.
- Panel (c): Multi-metric phase space: co-variation of mean phantom capacity vs. impassable share (r = +0.953),
  featuring 2D parametric cluster ellipse, central and counterfactual anchors, and linear coupling trajectory.
"""
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Ellipse
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from pathlib import Path
from sigfig_common import *
from scipy.stats import pearsonr

def conf_ellipse(x, y, ax, n_std=1.8, **kwargs):
    """Draw a 2D confidence / tolerance ellipse covering n_std deviations."""
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * n_std * np.sqrt(np.maximum(vals, 1e-6))
    ell = Ellipse(xy=(np.mean(x), np.mean(y)), width=w, height=h, angle=theta, **kwargs)
    ax.add_patch(ell)
    return ell

def build_fig11():
    # Load sensitivity summary and node-level data
    A = HERE / "out_final"
    s = pd.read_csv(A / "sensitivity.csv").set_index("scenario")
    sn = pd.read_csv(A / "sensitivity_nodes.csv")
    
    # 536 common nodes evaluated across all 13 scenarios
    common = sn.groupby("node").scenario.nunique()
    common_nodes = common[common == len(s)].index
    sn_common = sn[sn.node.isin(common_nodes)]
    
    base_imp = s.loc["central", "share_lt09"]
    base_acc = s.loc["central", "share_ge15"]
    base_ecw = s.loc["central", "median_ecw"]
    base_pc  = s.loc["central", "mean_pc"]
    N_COMMON = int(s.loc["central", "n"])
    
    # Canvas dimensions tuned for exact 3:2 (~1.500:1) landscape publication aspect ratio
    FIG_W = 13.17
    FIG_H = 9.333
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="#FFFFFF")
    
    # -------------------------------------------------------------------------
    # LAYOUT COORDINATES
    # -------------------------------------------------------------------------
    ax1_x = 0.220
    panel_a_w = 0.355
    panel_a_y = 0.080
    panel_a_h = 0.835
    
    ax2_x = 0.630
    panel_bc_w = 0.340
    
    panel_b_y = 0.535
    panel_b_h = 0.380
    
    panel_c_y = 0.080
    panel_c_h = 0.335
    
    # -------------------------------------------------------------------------
    # PANEL (a): TORNADO DIAGRAM OF SENSITIVITY DEVIATIONS
    # -------------------------------------------------------------------------
    ax1 = fig.add_axes([ax1_x, panel_a_y, panel_a_w, panel_a_h])
    ax1.set_facecolor("#FFFFFF")
    
    # 12 scenarios ordered logically from largest negative shift to largest positive shift
    s_diff = s.drop("central").copy()
    s_diff["d_imp"] = s_diff["share_lt09"] - base_imp
    s_diff = s_diff.sort_values("d_imp", ascending=True)
    
    labels_clean = {
        "vehicles removed": "Parked vehicles removed (upper clearance bound)",
        "shared threshold 2.0 m": "Shared-street threshold expanded (2.0 m)",
        "depth prior x0.7": "Footprint depth priors compressed (×0.7)",
        "lane_w +0.5 m": "Carriageway lane width expanded (+0.5 m)",
        "score thr 0.7": "Detector confidence threshold raised (0.7)",
        "camera h x0.88": "Calibrated camera height lowered (×0.88)",
        "lane_w -0.5 m": "Carriageway lane width narrowed (-0.5 m)",
        "camera h x1.12": "Calibrated camera height raised (×1.12)",
        "score thr 0.5": "Detector confidence threshold lowered (0.5)",
        "persons added": "Pedestrians included as fixed obstacles",
        "depth prior x1.3": "Footprint depth priors dilated (×1.3)",
        "shared threshold 1.0 m": "Shared-street threshold tightened (1.0 m)"
    }
    
    category_map = {
        "vehicles removed": "Counterfactual",
        "shared threshold 2.0 m": "Geometry",
        "depth prior x0.7": "Optical",
        "lane_w +0.5 m": "Geometry",
        "score thr 0.7": "Detection",
        "camera h x0.88": "Optical",
        "lane_w -0.5 m": "Geometry",
        "camera h x1.12": "Optical",
        "score thr 0.5": "Detection",
        "persons added": "Detection",
        "depth prior x1.3": "Optical",
        "shared threshold 1.0 m": "Geometry"
    }
    
    cat_colors = {
        "Counterfactual": "#0d9488",
        "Geometry": "#0284c7",
        "Optical": "#7c3aed",
        "Detection": "#d97706"
    }
    
    y_positions = np.arange(len(s_diff))
    for y_i in y_positions:
        if y_i % 2 == 1:
            ax1.axhspan(y_i - 0.42, y_i + 0.42, color="#f8fafc", zorder=0)
            
    ax1.axvline(0, color="#64748b", lw=1.2, ls="-", zorder=2)
    ax1.axvspan(-3.5, 3.5, color="#f1f5f9", alpha=0.5, zorder=0)
    ax1.text(-3.4, len(s_diff) - 0.35, "Parametric tolerance (±3.5 pp)",
             fontsize=6.5, fontstyle="italic", color="#94a3b8", zorder=1)
    
    bar_height = 0.52
    for idx, (sc_name, row) in enumerate(s_diff.iterrows()):
        d_val = row["d_imp"]
        cat = category_map[sc_name]
        bar_col = cat_colors[cat]
            
        rect = FancyBboxPatch((min(0, d_val), idx - bar_height / 2), abs(d_val), bar_height,
                              boxstyle="round,pad=0.0,rounding_size=0.10",
                              facecolor=bar_col, edgecolor="none", alpha=0.90, zorder=3)
        ax1.add_patch(rect)
        
        sign_str = "+" if d_val > 0 else ""
        val_str = f"{sign_str}{d_val:.2f} pp  ({row['share_lt09']:.1f}%)"
        
        if sc_name == "vehicles removed":
            # Position label directly above the bar with white text buffer halo
            txt_veh = ax1.text(d_val + 0.30, idx + 0.34, val_str, ha="left", va="bottom",
                               fontsize=6.8, fontweight="bold", color=bar_col, zorder=6)
            txt_veh.set_path_effects([pe.withStroke(linewidth=2.6, foreground="white")])
        elif d_val < 0:
            ax1.text(d_val - 0.25, idx, val_str, ha="right", va="center",
                     fontsize=6.8, fontweight="bold", color=bar_col, zorder=5)
        else:
            ax1.text(d_val + 0.25, idx, val_str, ha="left", va="center",
                     fontsize=6.8, fontweight="bold", color=bar_col, zorder=5)
                     
    ax1.set_yticks(y_positions)
    ax1.set_yticklabels([labels_clean[k] for k in s_diff.index], fontsize=7.6, fontweight="bold", color="#334155")
    ax1.set_xlabel(f"Change in impassable share (pp relative to central = {base_imp:.1f}%)",
                   fontsize=8.0, fontweight="bold", color="#475569")
    ax1.set_xlim(-12.8, 4.8)
    ax1.set_ylim(-0.7, len(s_diff) + 0.1)
    ax1.tick_params(labelsize=7.5, colors="#64748b")
    ax1.grid(True, axis="x", linestyle="--", alpha=0.45, color="#cbd5e1")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_color("#cbd5e1")
    ax1.spines["bottom"].set_color("#cbd5e1")
    
    leg_patches = [
        mpatches.Patch(color=cat_colors["Geometry"], label="Road geometry"),
        mpatches.Patch(color=cat_colors["Optical"], label="Optical calibration"),
        mpatches.Patch(color=cat_colors["Detection"], label="Semantic detection"),
        mpatches.Patch(color=cat_colors["Counterfactual"], label="Parking exclusion")
    ]
    leg = ax1.legend(handles=leg_patches, loc="upper left", bbox_to_anchor=(0.02, 0.98),
                     fontsize=7.2, frameon=False, title="MODELLING DOMAIN",
                     title_fontsize=7.5, labelspacing=0.40)
    leg._legend_box.align = "left"
    leg.get_title().set_fontweight("bold")
    leg.get_title().set_color("#475569")
    for text in leg.get_texts():
        text.set_color("#334155")
               
    bbox_badge = dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#64748b", lw=0.9, alpha=0.98)

    HEADER_Y_TOP = 0.955
    SUB_Y_TOP = 0.932
    HEADER_Y_BOT = 0.455
    SUB_Y_BOT = 0.432

    # Panel (a) Header
    fig.text(0.030, HEADER_Y_TOP, "(a)", fontsize=11.5, fontweight="bold", color="#0f172a",
             bbox=bbox_badge, va="center", ha="left", transform=fig.transFigure, zorder=10)
    fig.text(0.060, HEADER_Y_TOP, f"Parametric perturbation tornado: differential shift in impassable share (n = {N_COMMON})",
             fontsize=9.8, fontweight="bold", color="#0f172a", va="center", ha="left", transform=fig.transFigure, zorder=10)
    fig.text(0.060, SUB_Y_TOP, f"Deviation in impassable node share (<0.9 m) across {len(s) - 1} modelling variants relative to central baseline ({base_imp:.1f}%)",
             fontsize=7.4, fontweight="normal", color="#64748b", va="center", ha="left", transform=fig.transFigure, zorder=10)

    # -------------------------------------------------------------------------
    # PANEL (b): EMPIRICAL CUMULATIVE DISTRIBUTION FUNCTIONS (ECDF)
    # -------------------------------------------------------------------------
    ax2 = fig.add_axes([ax2_x, panel_b_y, panel_bc_w, panel_b_h])
    ax2.set_facecolor("#FFFFFF")
    
    ax2.axvspan(0, 0.9, color="#fee2e2", alpha=0.45, zorder=0)
    ax2.axvspan(0.9, 1.5, color="#fef3c7", alpha=0.45, zorder=0)
    ax2.axvspan(1.5, 6.0, color="#f0f9ff", alpha=0.45, zorder=0)
    
    ax2.text(0.40, 94, "IMPASSABLE\n(<0.9 m)", color="#b91c1c", fontsize=6.2, fontweight="bold", ha="center", zorder=1)
    ax2.text(1.22, 94, "RESTRICTED\n(0.9–1.5 m)", color="#b45309", fontsize=6.2, fontweight="bold", ha="center", zorder=1)
    ax2.text(3.75, 94, "ACCESSIBLE (≥1.5 m)", color="#0369a1", fontsize=6.2, fontweight="bold", ha="center", zorder=1)
    
    ax2.axvline(0.9, color="#dc2626", lw=1.1, ls="--", zorder=2)
    ax2.axvline(1.5, color="#d97706", lw=1.1, ls="--", zorder=2)
    
    x_eval = np.linspace(0, 6, 300)
    parametric_scenarios = [sc for sc in s.index if sc not in ["central", "vehicles removed"]]
    ecdf_matrix = []
    for sc in parametric_scenarios:
        ecw_vals = sn_common[sn_common.scenario == sc]["ecw_best"].values
        y_vals = np.array([(ecw_vals <= x).mean() * 100 for x in x_eval])
        ecdf_matrix.append(y_vals)
        ax2.plot(x_eval, y_vals, color="#94a3b8", lw=0.7, alpha=0.55, zorder=3)
        
    ecdf_min = np.min(ecdf_matrix, axis=0)
    ecdf_max = np.max(ecdf_matrix, axis=0)
    ax2.fill_between(x_eval, ecdf_min, ecdf_max, color="#cbd5e1", alpha=0.40, zorder=2,
                     label=f"{len(parametric_scenarios)} Parametric Variants Envelope")
    
    sh1_ecw = sn_common[sn_common.scenario == "shared threshold 1.0 m"]["ecw_best"].values
    y_sh1 = np.array([(sh1_ecw <= x).mean() * 100 for x in x_eval])
    ax2.plot(x_eval, y_sh1, color="#ea580c", lw=1.4, ls=":", zorder=4, label="Shared threshold 1.0 m")
    
    cen_ecw = sn_common[sn_common.scenario == "central"]["ecw_best"].values
    y_cen = np.array([(cen_ecw <= x).mean() * 100 for x in x_eval])
    ax2.plot(x_eval, y_cen, color="#0f172a", lw=2.4, zorder=5, label=f"Central Baseline (n = {N_COMMON})")
    
    veh_ecw = sn_common[sn_common.scenario == "vehicles removed"]["ecw_best"].values
    y_veh = np.array([(veh_ecw <= x).mean() * 100 for x in x_eval])
    ax2.plot(x_eval, y_veh, color="#0d9488", lw=2.2, ls="--", zorder=6, label="Vehicles Removed (Counterfactual)")
    
    ax2.plot(0.9, base_imp, marker="o", markersize=6.0, color="#0f172a", zorder=7)
    ax2.text(0.98, base_imp - 1, f"Central: {base_imp:.1f}%", fontsize=6.8, fontweight="bold", color="#0f172a", zorder=7)
    
    veh_imp = s.loc["vehicles removed", "share_lt09"]
    ax2.plot(0.9, veh_imp, marker="o", markersize=6.0, color="#0d9488", zorder=7)
    ax2.text(0.98, veh_imp - 1, f"Vehicles Removed: {veh_imp:.1f}%", fontsize=6.8, fontweight="bold", color="#0d9488", zorder=7)
    
    ax2.set_xlim(0, 5.5)
    ax2.set_ylim(0, 102)
    ax2.set_xlabel("Effective Clear Width (ECW, meters)", fontsize=8.0, fontweight="bold", color="#475569")
    ax2.set_ylabel("Cumulative share of nodes (%)", fontsize=8.0, fontweight="bold", color="#475569")
    ax2.tick_params(labelsize=7.5, colors="#64748b")
    ax2.grid(True, linestyle="--", alpha=0.35, color="#cbd5e1")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_color("#cbd5e1")
    ax2.spines["bottom"].set_color("#cbd5e1")
    ax2.legend(loc="lower right", fontsize=7.0, frameon=False)
    
    # Panel (b) Header
    fig.text(ax2_x, HEADER_Y_TOP, "(b)", fontsize=11.5, fontweight="bold", color="#0f172a",
             bbox=bbox_badge, va="center", ha="left", transform=fig.transFigure, zorder=10)
    fig.text(ax2_x + 0.030, HEADER_Y_TOP, "CDFs of node-level Effective Clear Width (ECW)",
             fontsize=8.8, fontweight="bold", color="#0f172a", va="center", ha="left", transform=fig.transFigure, zorder=10)
    fig.text(ax2_x + 0.030, SUB_Y_TOP, "Tight parametric envelope; vehicle exclusion shifts curve rightward",
             fontsize=7.0, fontweight="normal", color="#64748b", va="center", ha="left", transform=fig.transFigure, zorder=10)

    # -------------------------------------------------------------------------
    # PANEL (c): MULTI-METRIC PHASE SPACE
    # -------------------------------------------------------------------------
    ax3 = fig.add_axes([ax2_x, panel_c_y, panel_bc_w, panel_c_h])
    ax3.set_facecolor("#FFFFFF")
    
    ax3.axhline(base_imp, color="#cbd5e1", ls=":", lw=1.2, zorder=1)
    ax3.axvline(base_pc, color="#cbd5e1", ls=":", lw=1.2, zorder=1)
    
    param_rows = s.loc[parametric_scenarios]
    px, py = param_rows["mean_pc"].values, param_rows["share_lt09"].values
    conf_ellipse(px, py, ax3, n_std=1.8, facecolor="#f1f5f9", edgecolor="#94a3b8",
                 linestyle="--", linewidth=1.2, alpha=0.65, zorder=2)
                 
    ax3.text(35.8, 26.3, f"Parametric Invariance Cluster ({len(parametric_scenarios)} variants)\nPC: {px.min():.1f}–{px.max():.1f}%  •  Imp: {py.min():.1f}–{py.max():.1f}%",
             fontsize=6.8, fontweight="bold", color="#475569", ha="right", va="bottom", zorder=6)
    ax3.annotate("", xy=(32.8, 24.8), xytext=(32.8, 26.1),
                 arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=1.0), zorder=6)
                 
    for sc, row in s.iterrows():
        cat = "Central" if sc == "central" else category_map[sc]
        c_col = "#0f172a" if sc == "central" else cat_colors[cat]
        
        if sc == "central":
            ax3.scatter(row["mean_pc"], row["share_lt09"], s=115, color="#0f172a",
                        marker="D", edgecolor="white", lw=1.4, zorder=7)
            ax3.text(row["mean_pc"] - 0.45, row["share_lt09"] - 0.4, f"Central Baseline ({row['mean_pc']:.1f}%, {row['share_lt09']:.1f}%)",
                     fontsize=7.2, fontweight="bold", color="#0f172a", ha="right", va="top", zorder=8)
        elif sc == "vehicles removed":
            ax3.scatter(row["mean_pc"], row["share_lt09"], s=125, color="#0d9488",
                        marker="s", edgecolor="white", lw=1.4, zorder=7)
            ax3.annotate(f"Vehicles Removed (Counterfactual)\nPC: {row['mean_pc']:.1f}% ({100 * (row['mean_pc'] / base_pc - 1):+.0f}%)  •  Imp: {row['share_lt09']:.1f}% ({100 * (row['share_lt09'] / base_imp - 1):+.0f}%)",
                         xy=(row["mean_pc"], row["share_lt09"]), xytext=(17.2, 14.2),
                         fontsize=7.0, fontweight="bold", color="#0d9488", va="bottom",
                         arrowprops=dict(arrowstyle="->", color="#0d9488", lw=1.1), zorder=8)
        else:
            ax3.scatter(row["mean_pc"], row["share_lt09"], s=55, color=c_col,
                        edgecolor="white", lw=0.8, zorder=5)

    m_slope, b_intercept = np.polyfit(s["mean_pc"], s["share_lt09"], 1)
    x_line = np.linspace(16, 36, 100)
    ax3.plot(x_line, m_slope * x_line + b_intercept, color="#94a3b8", lw=1.3, ls="-.", zorder=3)
    
    _r, _p = pearsonr(s["mean_pc"], s["share_lt09"])
    ax3.text(35.5, 9.6, f"Linear phase-space coupling:  r = {_r:+.3f}  (R² = {_r ** 2:.3f}, {'p < 0.001' if _p < 0.001 else f'p = {_p:.3f}'})\nPhantom capacity scales with the impassable node share",
             fontsize=6.8, fontstyle="italic", color="#64748b", ha="right", va="center", zorder=4)

    leg_handles_c = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#0f172a", markersize=7.0, label="Central baseline"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#0d9488", markersize=7.5, label="Vehicles removed"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=cat_colors["Geometry"], markersize=6.5, label="Road geometry"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=cat_colors["Optical"], markersize=6.5, label="Optical calibration"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=cat_colors["Detection"], markersize=6.5, label="Semantic detection"),
    ]
    leg_c = ax3.legend(handles=leg_handles_c, loc="upper left", bbox_to_anchor=(0.02, 0.98),
                       fontsize=6.8, frameon=False, title="MODELLING DOMAIN",
                       title_fontsize=7.2, labelspacing=0.35)
    leg_c._legend_box.align = "left"
    leg_c.get_title().set_fontweight("bold")
    leg_c.get_title().set_color("#475569")
    for text in leg_c.get_texts():
        text.set_color("#334155")
             
    ax3.set_xlim(16, 36)
    ax3.set_ylim(8, 29)
    ax3.set_xlabel("Mean Phantom Capacity (%)", fontsize=8.0, fontweight="bold", color="#475569")
    ax3.set_ylabel("Share of Impassable Nodes (% < 0.9 m)", fontsize=8.0, fontweight="bold", color="#475569")
    ax3.tick_params(labelsize=7.5, colors="#64748b")
    ax3.grid(True, linestyle="--", alpha=0.35, color="#cbd5e1")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.spines["left"].set_color("#cbd5e1")
    ax3.spines["bottom"].set_color("#cbd5e1")
    
    # Panel (c) Header
    fig.text(ax2_x, HEADER_Y_BOT, "(c)", fontsize=11.5, fontweight="bold", color="#0f172a",
             bbox=bbox_badge, va="center", ha="left", transform=fig.transFigure, zorder=10)
    fig.text(ax2_x + 0.030, HEADER_Y_BOT, "Phase space: phantom capacity vs. impassable share",
             fontsize=8.8, fontweight="bold", color="#0f172a", va="center", ha="left", transform=fig.transFigure, zorder=10)
    fig.text(ax2_x + 0.030, SUB_Y_BOT, "Parametric variants group around central; vehicle exclusion slashes both",
             fontsize=7.0, fontweight="normal", color="#64748b", va="center", ha="left", transform=fig.transFigure, zorder=10)

    # -------------------------------------------------------------------------
    # SAVE & RETURN
    # -------------------------------------------------------------------------
    out_primary = OUT / "fig11_sensitivity.png"
    fig.savefig(out_primary, dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)
    print(f"Saved primary publication figure: {out_primary}")

    # Synchronize across manuscript and submission package directories
    paths_sync = [
        REV / "05_manuscript_R1/figures/fig11_sensitivity.png",
        REV / "05_manuscript_R1/markedup/figures/fig11_sensitivity.png",
        REV / "SUBMISSION_R1_PACKAGE/02_latex_source/figures/fig11_sensitivity.png"
    ]
    for p in paths_sync:
        if p.resolve() != out_primary.resolve() and p.parent.exists():
            shutil.copy2(out_primary, p)
            print(f"Synchronized figure to: {p}")

    print(f"fig11_sensitivity generated successfully. Scenarios evaluated: {len(s)}")

if __name__ == "__main__":
    build_fig11()
