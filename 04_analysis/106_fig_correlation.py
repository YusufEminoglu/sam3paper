import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde, pearsonr
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sigfig_common import *

def build_figS1():
    n = pd.read_csv(HERE / "out_final/nodes_R1.csv")
    n["floors"] = n[["floors_left", "floors_right"]].mean(axis=1)
    n["blocked"] = n.veh_area + n.furn_area + n.infra_area
    n["pc100"] = n.pc * 100
    n["cov100"] = n.coverage * 100

    VARS = [
        ("w_corridor", "Corridor width $W_c$ (m)", "Corridor $W_c$", "m"),
        ("w_carriage", "Carriageway width $W_v$ (m)", "Carriageway $W_v$", "m"),
        ("zone_w", "Pedestrian zone width $W_p$ (m)", "Ped. zone $W_p$", "m"),
        ("floors", "Building floors (mean)", "Floors", "floors"),
        ("blocked", "Blocked ground area ($m^2$)", "Blocked area", "$m^2$"),
        ("cov100", "Observation coverage (%)", "Coverage", "%"),
        ("ecw_best", "Effective Clear Width (m)", "ECW", "m"),
        ("pc100", "Phantom capacity (%)", "Phantom cap.", "%")
    ]
    cols = [v for v, _, _, _ in VARS]
    labs = [l for _, l, _, _ in VARS]
    short_labs = [s for _, _, s, _ in VARS]
    units = [u for _, _, _, u in VARS]
    
    d = n[cols].dropna()
    k = len(cols)
    
    R = np.zeros((k, k))
    P_vals = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            r_val, p_val = pearsonr(d[cols[i]], d[cols[j]])
            R[i, j] = r_val
            P_vals[i, j] = p_val
            
    # Compact canvas without bottom text boxes: 18.8 x 9.0 in
    FIG_W, FIG_H = 18.8, 9.2
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="#FFFFFF")
    
    panel_y = 0.055
    mat_h_in = 7.20
    mat_w_in = 7.20
    mat_h = mat_h_in / FIG_H
    mat_w = mat_w_in / FIG_W
    cbar_w = 0.012
    cbar_gap = 0.015
    ax1_x = 0.040
    panel2_x = 0.518
    panel2_w = 0.442
    
    # -------------------------------------------------------------------------
    # PANEL (a): BIVARIATE CORRELATION MATRIX (HARMONIZED COLORS FROM PANEL B)
    # -------------------------------------------------------------------------
    ax1 = fig.add_axes([ax1_x, panel_y, mat_w, mat_h])
    ax1.set_facecolor("#FFFFFF")
    
    # Colormap derived directly from Panel B palette:
    # Blue (#1d4ed8, IQR) -> White (#FFFFFF) -> Slate-Grey (#1e293b, Panel B bars and text)
    cmap_div = LinearSegmentedColormap.from_list(
        "panel_b_harmonized_grey",
        ["#1d4ed8", "#3b82f6", "#93c5fd", "#f8fafc", "#cbd5e1", "#64748b", "#334155", "#1e293b"],
        N=256
    )
    
    im = ax1.imshow(R, cmap=cmap_div, vmin=-1.0, vmax=1.0, aspect="equal")
    
    ax1.set_xticks(np.arange(k) - 0.5, minor=True)
    ax1.set_yticks(np.arange(k) - 0.5, minor=True)
    ax1.grid(which="minor", color="#FFFFFF", linestyle="-", linewidth=2.0)
    ax1.tick_params(which="minor", size=0)
    
    for i in range(k):
        for j in range(k):
            val = R[i, j]
            p_val = P_vals[i, j]
            sig = "***" if p_val < 0.001 and i != j else ("**" if p_val < 0.01 and i != j else ("*" if p_val < 0.05 and i != j else ""))
            txt_col = "white" if abs(val) > 0.45 else "#1e293b"
            
            if i == j:
                ax1.text(j, i, "1.00", ha="center", va="center", fontsize=8.2, fontweight="bold", color=txt_col)
            else:
                ax1.text(j, i - 0.10, f"{val:+.2f}", ha="center", va="center", fontsize=8.0, fontweight="bold", color=txt_col)
                if sig:
                    ax1.text(j, i + 0.22, sig, ha="center", va="center", fontsize=7.0, fontweight="bold", color=txt_col)

    # Highlight key cells using Panel B's teal accent (#0d9488 from mean line)
    key_pairs = [(6, 2), (6, 0), (7, 6), (6, 5)]
    for row_idx, col_idx in key_pairs:
        rect = Rectangle((col_idx - 0.48, row_idx - 0.48), 0.96, 0.96,
                         fill=False, edgecolor="#0d9488", linewidth=2.2, zorder=10)
        ax1.add_patch(rect)
        rect_sym = Rectangle((row_idx - 0.48, col_idx - 0.48), 0.96, 0.96,
                             fill=False, edgecolor="#0d9488", linewidth=2.2, zorder=10)
        ax1.add_patch(rect_sym)
        
    ax1.set_xticks(range(k))
    ax1.set_xticklabels(short_labs, rotation=35, ha="right", fontsize=8.2, fontweight="bold", color="#334155")
    ax1.set_yticks(range(k))
    ax1.set_yticklabels(labs, fontsize=8.2, fontweight="bold", color="#334155")
    for s in ["top", "right", "left", "bottom"]:
        ax1.spines[s].set_visible(False)
    
    # Colorbar
    ax_cb = fig.add_axes([ax1_x + mat_w + cbar_gap, panel_y + 0.04, cbar_w, mat_h - 0.08])
    cb = fig.colorbar(im, cax=ax_cb)
    cb.set_label("Pearson correlation coefficient (r)", fontsize=8.5, fontweight="bold", color="#475569")
    cb.ax.tick_params(labelsize=7.5, colors="#64748b")
    cb.outline.set_edgecolor("#cbd5e1")
    cb.outline.set_linewidth(0.8)
    
    # Panel (a) Badge & Title
    ax1.text(0.00, 1.055, "(a)", transform=ax1.transAxes,
             fontsize=11.5, fontweight="bold", color="#1e293b",
             bbox=dict(facecolor="white", edgecolor="#64748b", boxstyle="round,pad=0.2", lw=0.8, alpha=0.95),
             zorder=10)
    ax1.text(0.085, 1.060, "Bivariate Pearson correlation matrix across morphological and survey metrics",
             transform=ax1.transAxes, fontsize=11.2, fontweight="bold", color="#1e293b", va="center", zorder=10)
    ax1.text(0.085, 1.025, f"Evaluated across n = {len(d)} nodes. Highlighted boxes indicate key walkability & validity couplings (* p<0.05, ** p<0.01, *** p<0.001)",
             transform=ax1.transAxes, fontsize=8.0, fontweight="normal", color="#64748b", va="center", zorder=10)

    # -------------------------------------------------------------------------
    # PANEL (b): UNIVARIATE DISTRIBUTIONS (4x2 GRID)
    # -------------------------------------------------------------------------
    n_rows, n_cols = 4, 2
    sub_w = 0.203
    sub_h = 0.142
    gap_x = 0.036
    gap_y = 0.050
    
    ax_b_header = fig.add_axes([panel2_x, panel_y, panel2_w, mat_h])
    ax_b_header.axis("off")
    ax_b_header.text(0.00, 1.055, "(b)", transform=ax_b_header.transAxes,
                     fontsize=11.5, fontweight="bold", color="#1e293b",
                     bbox=dict(facecolor="white", edgecolor="#64748b", boxstyle="round,pad=0.2", lw=0.8, alpha=0.95),
                     zorder=10)
    ax_b_header.text(0.075, 1.060, "Empirical distributions, kernel densities, and dispersion metrics",
                     transform=ax_b_header.transAxes, fontsize=11.2, fontweight="bold", color="#1e293b", va="center", zorder=10)
    ax_b_header.text(0.075, 1.025, "Histogram bins with Gaussian KDE density curve, median (dashed red), and mean (dotted teal) for each metric",
                     transform=ax_b_header.transAxes, fontsize=8.0, fontweight="normal", color="#64748b", va="center", zorder=10)

    for idx, (c, l, s_, u_) in enumerate(VARS):
        r_i = idx // n_cols
        c_i = idx % n_cols
        row_plot = (n_rows - 1) - r_i
        
        px = panel2_x + c_i * (sub_w + gap_x)
        py = panel_y + row_plot * (sub_h + gap_y)
        
        ax_sub = fig.add_axes([px, py, sub_w, sub_h])
        ax_sub.set_facecolor("#FFFFFF")
        
        vals = d[c].values
        v_min, v_max = vals.min(), vals.max()
        v_med = np.median(vals)
        v_mean = np.mean(vals)
        v_q25 = np.percentile(vals, 25)
        v_q75 = np.percentile(vals, 75)
        
        n_bins = 24
        counts, bin_edges, _ = ax_sub.hist(vals, bins=n_bins, color="#cbd5e1", edgecolor="#FFFFFF", lw=0.6, density=True, zorder=2)
        
        kde_x = np.linspace(v_min, v_max, 200)
        max_y = counts.max()
        try:
            kde = gaussian_kde(vals, bw_method="scott")
            kde_y = kde(kde_x)
            ax_sub.plot(kde_x, kde_y, color="#1e293b", lw=1.5, zorder=4)
            max_y = max(max_y, kde_y.max())
        except Exception:
            pass
            
        ax_sub.set_ylim(0, max_y * 1.30)
        ax_sub.axvspan(v_q25, v_q75, color="#3b82f6", alpha=0.10, zorder=1)
        ax_sub.axvline(v_med, color="#dc2626", lw=1.3, ls="--", zorder=5)
        ax_sub.axvline(v_mean, color="#0d9488", lw=1.2, ls=":", zorder=5)
        
        title_str = f"{s_}   ·   Med: {v_med:.1f}   Mean: {v_mean:.1f} {u_}"
        ax_sub.set_title(title_str, fontsize=7.2, fontweight="bold", color="#1e293b", pad=3.5)
                    
        ax_sub.tick_params(labelsize=6.8, colors="#64748b", pad=1.5)
        ax_sub.grid(True, linestyle="--", alpha=0.35, color="#e2e8f0")
        ax_sub.spines["top"].set_visible(False)
        ax_sub.spines["right"].set_visible(False)
        ax_sub.spines["left"].set_color("#cbd5e1")
        ax_sub.spines["bottom"].set_color("#cbd5e1")
        ax_sub.set_yticks([])

    # (BOTTOM ANALYTICAL CARDS REMOVED AS REQUESTED)
    
    save(fig, "figS1_correlation")
    stat("corrN", len(d))
    print(f"figS1_correlation generated successfully. Analysed nodes: {len(d)}")

if __name__ == "__main__":
    build_figS1()
