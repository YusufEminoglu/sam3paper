"""Figure 09: Polar Corridor Spirals and Network-Level Accessibility Compliance Dashboard.

(a-k) Polar plots of best-side effective clear width (ECW) along the eleven best-sampled corridors
      (>= 6 nodes), ordered by node position along the corridor's principal axis (clockwise from North, 1 -> n).
      Concentric zones denote impassable (< 0.9 m, pale rose) and restricted (0.9-1.5 m, pale amber) thresholds.
      Header cards summarize corridor width (Wc), mean ECW, sample size, and dual-tier fragmentation status.
(l)   Network-level compliance dashboard showing horizontal stacked bar distributions of surveyed nodes (n = 557)
      across the 5 functional street classes and 3 accessibility tiers, alongside the continuous ECW spline color ramp.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline
from pathlib import Path
import shutil
from sigfig_common import *

def clean_street_name(raw):
    s = str(raw).strip()
    mapping = {
        "KIBRIS ŞEHİTLERİ": "Kıbrıs Şehitleri",
        "DR. MUSTAFA ENVER BEY": "Dr. Mustafa Enver Bey",
        "CUMHURİYET": "Cumhuriyet",
        "PİRİ REİS": "Piri Reis",
        "MELTEM": "Meltem",
        "ADEM YAVUZ": "Adem Yavuz",
        "ŞEHİT FETHİBEY": "Şehit Fethibey",
        "HALİL RIFATPAŞA": "Halil Rıfatpaşa",
    }
    return mapping.get(s, s.title())

def build_fig09():
    n = load()
    ECW_MAX = 12.0
    top = n.groupby("street_id").size().sort_values(ascending=False)
    top = top[top >= 6].index[:11]
    
    # Load values from values.tex if present
    V = {}
    v_path = HERE / "out_final/values.tex"
    if v_path.exists():
        for l in open(v_path, encoding="utf-8"):
            p = l.strip().split("{")
            if len(p) >= 3:
                V[p[1].rstrip("}").lstrip("\\")] = p[2].rstrip("}")
                
    # Unified accessible colormap ramp:
    # Impassable (#dc2626) -> Restricted (#d97706) -> Accessible (#0284c7 -> #1e40af)
    ramp = LinearSegmentedColormap.from_list(
        "ecw_pub",
        ["#dc2626", "#ea580c", "#d97706", "#0284c7", "#1e40af"],
        N=256
    )
    
    # Canvas: 18.0 x 13.5 inches (balanced 4:3 grid with tight, clean margins)
    fig = plt.figure(figsize=(18.0, 13.5), facecolor="#FFFFFF")
    
    # -------------------------------------------------------------------------
    # MAIN GRID COORDINATES: 3 ROWS x 4 COLS (Fully symmetric, zero wasted space)
    # -------------------------------------------------------------------------
    col_x = [0.019, 0.263, 0.507, 0.751]
    row_y = [0.670, 0.348, 0.026]
    cell_w = 0.230
    cell_h = 0.304
    
    panel_letters = [chr(ord('a') + i) for i in range(12)]
    corridor_bench = []
    
    # -------------------------------------------------------------------------
    # 1. PLOT 11 POLAR STREET PROFILES (Panels a to k)
    # -------------------------------------------------------------------------
    for idx, sid in enumerate(top):
        r_idx = idx // 4
        c_idx = idx % 4
        x0 = col_x[c_idx]
        y0 = row_y[r_idx]
        
        d = n[n.street_id == sid].copy()
        
        # Sort nodes along principal axis
        xy = d[["x_node", "y_node"]].values
        xy = xy - xy.mean(0)
        u, s, vt = np.linalg.svd(xy, full_matrices=False)
        d["t"] = xy @ vt[0]
        d = d.sort_values("t")
        
        e = d.ecw_plot.values
        k = len(e)
        cls = d["class"].mode().iat[0]
        col = CLS_COL[cls]
        wc = d.w_corridor.mean()
        mean_ecw = d.ecw_best.mean()
        imp_pct = (d.tier3 == "Impassable").mean() * 100
        res_pct = (d.tier3 == "Restricted").mean() * 100
        acc_pct = (d.tier3 == "Accessible").mean() * 100
        has_both = {"Impassable", "Accessible"} <= set(d.tier3)
        zero_imp = (imp_pct == 0.0)
        
        corridor_bench.append({
            "letter": panel_letters[idx],
            "name": clean_street_name(d.road_name.iat[0])[:22],
            "k": k,
            "wc": wc,
            "imp_pct": imp_pct,
            "res_pct": res_pct,
            "acc_pct": acc_pct,
            "has_both": has_both,
            "zero_imp": zero_imp,
            "col": col
        })
        
        # Header axes for clean card
        ax_h = fig.add_axes([x0, y0 + cell_h - 0.038, cell_w, 0.038])
        ax_h.axis("off")
        
        # Card background (clean white, matching panel l)
        h_card = FancyBboxPatch((0.0, 0.0), 1.0, 1.0,
                                boxstyle="round,pad=0.0,rounding_size=0.08",
                                facecolor="#ffffff", edgecolor="#e2e8f0", linewidth=0.8,
                                transform=ax_h.transAxes, zorder=1)
        ax_h.add_patch(h_card)
        
        # Standardized panel badge style matching fig05, 06, 07, 08, S5
        badge_bbox = dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#64748b", lw=0.9, alpha=0.98)
        ax_h.text(0.025, 0.68, f"({panel_letters[idx]})",
                  fontsize=12.2, fontweight="bold", color="#0f172a",
                  va="center", ha="left", transform=ax_h.transAxes,
                  bbox=badge_bbox, zorder=10)
                  
        # Clean Street Name (enlarged font)
        nm = clean_street_name(d.road_name.iat[0])[:22]
        ax_h.text(0.145, 0.68, nm,
                  fontsize=10.5, fontweight="bold", color=col,
                  va="center", transform=ax_h.transAxes, zorder=2)
                  
        # Status Badge (Dual-Tier or 0% Impassable)
        if has_both:
            status_box = FancyBboxPatch((0.66, 0.44), 0.31, 0.48,
                                        boxstyle="round,pad=0.01,rounding_size=0.05",
                                        facecolor="#fee2e2", edgecolor="#fca5a5", lw=0.7,
                                        transform=ax_h.transAxes, zorder=3)
            ax_h.add_patch(status_box)
            ax_h.text(0.815, 0.68, "Dual-Tier", fontsize=7.2, fontweight="bold",
                      color="#991b1b", ha="center", va="center", transform=ax_h.transAxes, zorder=4)
        elif imp_pct == 0.0:
            status_box = FancyBboxPatch((0.60, 0.44), 0.37, 0.48,
                                        boxstyle="round,pad=0.01,rounding_size=0.05",
                                        facecolor="#ecfdf5", edgecolor="#a7f3d0", lw=0.7,
                                        transform=ax_h.transAxes, zorder=3)
            ax_h.add_patch(status_box)
            ax_h.text(0.785, 0.68, "0% Impassable", fontsize=7.0, fontweight="bold",
                      color="#047857", ha="center", va="center", transform=ax_h.transAxes, zorder=4)
                      
        # Concise Subtitle metrics (enlarged font)
        metric_str = f"W_c = {wc:.0f}m  •  mean ECW = {mean_ecw:.1f}m  •  n = {k}"
        ax_h.text(0.035, 0.24, metric_str,
                  fontsize=8.0, color="#64748b", va="center", transform=ax_h.transAxes, zorder=2)

        # Polar plot axes: perfectly centered square with comfortable clearance
        ax_p = fig.add_axes([x0 + 0.029, y0 + 0.020, 0.172, 0.226], projection="polar")
        ax_p.set_facecolor("#FFFFFF")
        
        th = np.linspace(0, 2 * np.pi, k, endpoint=False)
        thi = np.linspace(0, 2 * np.pi, 300, endpoint=False)
        
        # Periodic spline
        spl = make_interp_spline(np.r_[th, th[0] + 2 * np.pi], np.r_[e, e[0]],
                                 k=min(3, k), bc_type="periodic" if k >= 4 else None)
        sm = np.clip(spl(thi), 0, ECW_MAX)
        
        # Subtle radial hairline spokes from center to each node direction
        for ang in th:
            ax_p.plot([ang, ang], [0, ECW_MAX], color="#e2e8f0", lw=0.55, ls=":", zorder=1)
        
        # 1. Background Zone Shading
        # Impassable zone: 0 to 0.9 m (pale rose)
        ax_p.fill_between(thi, 0, 0.9, color="#fee2e2", alpha=0.50, zorder=0)
        # Restricted zone: 0.9 to 1.5 m (pale amber)
        ax_p.fill_between(thi, 0.9, 1.5, color="#fef3c7", alpha=0.45, zorder=0)
        
        # 2. Reference Threshold Rings
        ax_p.plot(thi, np.full_like(thi, 0.9), color="#dc2626", lw=1.2, ls="--", alpha=0.92, zorder=1)
        ax_p.plot(thi, np.full_like(thi, 1.5), color="#d97706", lw=1.2, ls=":", alpha=0.95, zorder=1)
        
        # Outer grid rings
        for rv in (4, 8, 12):
            ax_p.plot(thi, np.full_like(thi, rv), color="#cbd5e1", lw=0.65, ls="-", zorder=1)
            
        # 3. Soft fill and colored spline ribbon
        ax_p.fill(thi, sm, color=col, alpha=0.22, lw=0, zorder=2)
        for i in range(len(thi) - 1):
            ax_p.plot(thi[i:i+2], sm[i:i+2], color=ramp(sm[i] / ECW_MAX),
                      lw=2.7, alpha=0.95, solid_capstyle="round", zorder=3)
                      
        # 4. Individual Node Points
        node_cols = [ramp(v / ECW_MAX) for v in e]
        ax_p.scatter(th, e, c=node_cols, s=56, zorder=6, edgecolors="#ffffff", linewidths=1.2)
        
        # 5. Polar ticks and orientation
        ax_p.set_theta_zero_location("N")
        ax_p.set_theta_direction(-1)
        ax_p.set_xticks(th)
        ax_p.set_xticklabels([str(i + 1) for i in range(k)], fontsize=9.2, fontweight="bold", color="#334155")
        ax_p.tick_params(axis="x", pad=3)
        
        ax_p.set_ylim(0, ECW_MAX + 0.8)
        ax_p.set_yticks([4, 8, 12])
        ax_p.set_yticklabels(["4m", "8m", "12m"], fontsize=7.2, fontweight="bold", color="#64748b",
                             bbox=dict(facecolor="#ffffff", edgecolor="#cbd5e1", boxstyle="round,pad=0.2", lw=0.5, alpha=0.95))
        ax_p.set_rlabel_position(45)
        
        ax_p.spines["polar"].set_linewidth(0.65)
        ax_p.spines["polar"].set_color("#cbd5e1")
        ax_p.grid(color="#f1f5f9", lw=0.5)

    # -------------------------------------------------------------------------
    # 2. PANEL (l): CORRIDOR COMPLIANCE BENCHMARK (Cell 3, 4)
    # -------------------------------------------------------------------------
    lx0, ly0 = col_x[3], row_y[2]
    
    # Outer Card for Panel (l) - almost pure white (#ffffff), crisp subtle border
    ax_l_bg = fig.add_axes([lx0, ly0, cell_w, cell_h])
    ax_l_bg.axis("off")
    l_card = FancyBboxPatch((0.0, 0.0), 1.0, 1.0,
                            boxstyle="round,pad=0.0,rounding_size=0.04",
                            facecolor="#ffffff", edgecolor="#e2e8f0", linewidth=0.9,
                            transform=ax_l_bg.transAxes, zorder=1)
    ax_l_bg.add_patch(l_card)
    
    # Standardized Title & Badge for Panel (l) with enlarged fonts
    badge_bbox = dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#64748b", lw=0.9, alpha=0.98)
    ax_l_bg.text(0.035, 0.94, "(l)", fontsize=12.2, fontweight="bold", color="#0f172a",
                 va="center", ha="left", transform=ax_l_bg.transAxes,
                 bbox=badge_bbox, zorder=10)
    ax_l_bg.text(0.135, 0.94, "Corridor Compliance Benchmark", fontsize=11.2, fontweight="bold",
                 color="#0f172a", va="center", transform=ax_l_bg.transAxes, zorder=2)
    ax_l_bg.text(0.038, 0.885, "Accessibility tier distributions for corridors a–k (n = 88 nodes)",
                 fontsize=8.0, color="#64748b", va="center", transform=ax_l_bg.transAxes, zorder=2)
                 
    # Legend placed at the TOP of panel (l), right under subtitle
    tier_colors = {
        "Impassable": "#dc2626",
        "Restricted": "#d97706",
        "Accessible": "#0284c7"
    }
    
    leg_x = [0.038, 0.355, 0.67]
    leg_labels = ["Impassable (<0.9m)", "Restricted (0.9-1.5m)", "Accessible (≥1.5m)"]
    for t_idx, t_name in enumerate(["Impassable", "Restricted", "Accessible"]):
        lx = leg_x[t_idx]
        swatch = FancyBboxPatch((lx, 0.822), 0.030, 0.032,
                                boxstyle="round,pad=0.005",
                                facecolor=tier_colors[t_name], edgecolor="none",
                                transform=ax_l_bg.transAxes, zorder=3)
        ax_l_bg.add_patch(swatch)
        ax_l_bg.text(lx + 0.038, 0.838, leg_labels[t_idx],
                     fontsize=7.2, fontweight="bold", color="#475569", va="center",
                     transform=ax_l_bg.transAxes, zorder=3)
                     
    # Inner axes for horizontal stacked bars - pure white background
    df_cb = pd.DataFrame(corridor_bench)
    ax_bars = fig.add_axes([lx0 + 0.068, ly0 + 0.045, cell_w - 0.078, cell_h - 0.118])
    ax_bars.set_facecolor("#ffffff")
    
    y_pos = np.arange(len(df_cb))[::-1]  # (a) at top, (k) at bottom
    bar_h = 0.64
    left = np.zeros(len(df_cb))
    
    for t_name, t_col in [("Impassable", "#dc2626"), ("Restricted", "#d97706"), ("Accessible", "#0284c7")]:
        pct_col = f"{t_name.lower()[:3]}_pct"
        vals = df_cb[pct_col].values
        ax_bars.barh(y_pos, vals, left=left, color=t_col, edgecolor="#ffffff",
                     linewidth=0.7, height=bar_h, zorder=3)
        for i, (l_val, w_val) in enumerate(zip(left, vals)):
            if w_val >= 21.0:
                ax_bars.text(l_val + w_val / 2.0, y_pos[i], f"{w_val:.0f}%",
                             ha="center", va="center", fontsize=7.0, color="#ffffff",
                             fontweight="bold", zorder=4)
        left += vals
        
    y_labels = [f"({r['letter']}) {r['name'][:14]}" for _, r in df_cb.iterrows()]
    ax_bars.set_yticks(y_pos)
    ax_bars.set_yticklabels(y_labels, fontsize=7.6, color="#1e293b", fontweight="500")
    
    for i, (_, r) in enumerate(df_cb.iterrows()):
        if r["has_both"]:
            ax_bars.text(102.0, y_pos[i], "Dual", fontsize=6.8, fontweight="bold",
                         color="#991b1b", va="center", zorder=4,
                         bbox=dict(boxstyle="round,pad=0.12", fc="#fee2e2", ec="#fca5a5", lw=0.5))
        elif r["zero_imp"]:
            ax_bars.text(102.0, y_pos[i], "0% Imp", fontsize=6.5, fontweight="bold",
                         color="#047857", va="center", zorder=4,
                         bbox=dict(boxstyle="round,pad=0.12", fc="#ecfdf5", ec="#a7f3d0", lw=0.5))
            
    ax_bars.set_xlim(0, 115)
    ax_bars.set_xlabel("Share of surveyed nodes in corridor (%)", fontsize=7.6, fontweight="bold", color="#64748b", labelpad=3)
    ax_bars.tick_params(axis="x", labelsize=7.0, colors="#64748b", pad=2)
    ax_bars.grid(axis="x", linestyle="--", alpha=0.35, color="#cbd5e1", zorder=0)
    ax_bars.spines["top"].set_visible(False)
    ax_bars.spines["right"].set_visible(False)
    ax_bars.spines["left"].set_visible(False)
    ax_bars.spines["bottom"].set_color("#cbd5e1")
    
    # Footnote with enlarged font
    ax_l_bg.text(0.038, 0.016, "Dual-Tier fragmentation: 6 of 11 corridors contain both impassable (<0.9m) & accessible nodes",
                 fontsize=7.0, fontweight="bold", color="#991b1b", transform=ax_l_bg.transAxes, zorder=3)

    # -------------------------------------------------------------------------
    # SAVE & SYNCHRONIZE 300 DPI PUBLICATION FIGURE (FIGURE 09)
    # -------------------------------------------------------------------------
    out_primary = OUT / "fig09_spiral.png"
    fig.savefig(out_primary, dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    print(f"Saved primary publication figure: {out_primary}")

    # Synchronize across manuscript and submission package directories
    paths_sync = [
        REV / "05_manuscript_R1/markedup/figures/fig09_spiral.png",
        REV / "SUBMISSION_R1_PACKAGE/02_latex_source/figures/fig09_spiral.png"
    ]
    for p in paths_sync:
        if p.parent.exists():
            shutil.copy2(out_primary, p)
            print(f"Synchronized figure to: {p}")

    stat("spiralStreets", len(top))
    stat("spiralBoth", sum(1 for s_ in top if {"Impassable", "Accessible"} <= set(n[n.street_id == s_].tier3)))
    print(f"fig09_spiral generated successfully. Streets: {len(top)}, Dual-tier: {sum(1 for s_ in top if {'Impassable', 'Accessible'} <= set(n[n.street_id == s_].tier3))}")

if __name__ == "__main__":
    build_fig09()
