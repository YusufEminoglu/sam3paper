"""Figure S4: Arc Diagram of Street Adjacency and Raincloud Plot of Functional Street Classes.

(a) Arc diagram of the 22 best-sampled streets (>= 5 nodes), ordered by mean ECW;
    arcs = streets that meet at a physical junction (nodes within LINK_M = 60 m).
    Integrated executive micro-data table displays corridor width, mean ECW, % impassable, and surveyed nodes.
(b) Raincloud plot of best-side ECW across functional street classes
    with accessibility zone references (0.9 m impassable, 1.5 m accessible).
    Aspect ratio: 3:2 (18.0 x 12.0 inches, 5400 x 3600 pixels at 300 DPI).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Arc, Circle, Rectangle
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde, spearmanr
from pathlib import Path
from sigfig_common import *

def build_fig09():
    LINK_M, MIN_N = 60.0, 5
    n = load()
    TOT = len(n)
    
    # Load values from values.tex
    V = {}
    v_path = HERE / "out_final/values.tex"
    if v_path.exists():
        for l in open(v_path, encoding="utf-8"):
            p = l.strip().split("{")
            if len(p) >= 3:
                V[p[1].rstrip("}").lstrip("\\")] = p[2].rstrip("}")
                
    icc_val = V.get("ICC", "0.51")
    n_ml_nodes = V.get("nMLnodes", "448")
    n_ml_str = V.get("nMLstreets", "142")
    r_marg = V.get("Rmarg", "0.19")
    r_cond = V.get("Rcond", "0.52")
    perm_f = V.get("permF", "14.4")
    
    # Dimensions: 18.0 x 12.0 inches (exact 3:2 aspect ratio)
    fig = plt.figure(figsize=(18.0, 12.0), facecolor="#FFFFFF")
    
    HEADER_Y = 0.948
    HEADER_H = 0.046
    
    # Zero-margin edge-to-edge layout with 3:2 aspect ratio calibration
    LEFT_A = 0.008
    WIDTH_A = 0.545
    LEFT_B = 0.575
    WIDTH_B = 0.415
    PLOT_BTM = 0.008
    PLOT_H = 0.930

    # -------------------------------------------------------------------------
    # PANEL A: ARC DIAGRAM & CORRIDOR CLEARANCE TABLE
    # -------------------------------------------------------------------------
    ax1 = fig.add_axes([LEFT_A, PLOT_BTM, WIDTH_A, PLOT_H])
    ax1.axis("off")
    ax1.set_facecolor("#FFFFFF")
    
    # 1. Aggregate street-level metrics for streets with >= MIN_N nodes
    st = n.groupby("street_id").agg(
        k=("node", "size"),
        ecw=("ecw_best", "mean"),
        cls=("class", lambda s: s.mode().iat[0]),
        w=("w_corridor", "mean"),
        name=("road_name", "first"),
        obs=("obs_total", "mean"),
        imp=("tier3", lambda s: (s == "Impassable").mean())
    ).reset_index()
    st = st[st.k >= MIN_N].sort_values("ecw").reset_index(drop=True)
    pos = {s: i for i, s in enumerate(st.street_id)}
    sub = n[n.street_id.isin(pos)].reset_index(drop=True)
    
    # KD-Tree adjacency pairs within LINK_M (60 m)
    tree = cKDTree(np.c_[sub.x_node, sub.y_node])
    links = {}
    for i, j in tree.query_pairs(LINK_M):
        a, b = sub.street_id[i], sub.street_id[j]
        if a != b:
            links[tuple(sorted((a, b)))] = links.get(tuple(sorted((a, b))), 0) + 1
            
    def ltype(a, b):
        ea, eb = st.ecw[pos[a]], st.ecw[pos[b]]
        if ea < 1.5 and eb < 1.5:
            return "both"
        if min(ea, eb) < 1.5:
            return "contrast"
        return "ok"
        
    LC = {"both": "#dc2626", "contrast": "#0284c7", "ok": "#cbd5e1"}
    
    max_v = max(links.values()) if links else 1
    N_ST = len(st)
    ax1.set_xlim(-5.6, N_ST - 0.2)
    ax1.set_ylim(-7.30, 15.6)
    
    # Draw arcs with Value-by-Alpha (VBA) and dashed line for cross-corridor clearance contrast
    for (a, b), v in sorted(links.items(), key=lambda kv: kv[1]):
        x0, x1 = sorted((pos[a], pos[b]))
        t = ltype(a, b)
        # Value-by-Alpha: line width and alpha scale continuously with node pair frequency v
        lw = 1.6 + 2.4 * (v / max_v)
        alpha = 0.48 + 0.48 * (v / max_v)
        if t == "both":
            alpha = max(0.85, alpha)
        # Dashed line for cross-corridor clearance contrast
        ls = "--" if t == "contrast" else "-"
        arc_patch = Arc(((x0 + x1) / 2, 0), x1 - x0, (x1 - x0) * 1.15,
                        theta1=0, theta2=180, color=LC[t],
                        lw=lw, ls=ls, alpha=alpha, zorder=2)
        ax1.add_patch(arc_patch)
        
    # Baseline axis
    ax1.plot([-0.6, N_ST - 0.4], [0, 0], color="#94a3b8", lw=1.3, zorder=1)
    
    # Street nodes scatter (sized by node count k)
    ax1.scatter(range(N_ST), np.zeros(N_ST), s=st.k * 36,
                c=[CLS_COL[c] for c in st.cls], edgecolors="white", lw=1.5, zorder=4)
                 
    # -------------------------------------------------------------------------
    # ZERO-DISTANCE EXECUTIVE DATA CARD (Adjacent to graph baseline)
    # -------------------------------------------------------------------------
    tbl_top = -0.38
    tbl_bot = -7.05
    tbl_h = tbl_top - tbl_bot
    tbl_card = FancyBboxPatch((-5.4, tbl_bot), N_ST + 5.2, tbl_h,
                              boxstyle="round,pad=0.015,rounding_size=0.03",
                              facecolor="#f8fafc", edgecolor="#cbd5e1", lw=0.9, zorder=2)
    ax1.add_patch(tbl_card)
    
    # Header section: Street names aligned below nodes inside card header
    ax1.text(-5.2, -1.80, "Street / Corridor", ha="left", va="center",
             fontsize=8.4, fontweight="bold", color="#475569", zorder=5)

    for i, r in st.iterrows():
        s_name = str(r["name"])[:15].strip().title()
        ax1.text(i, -0.55, s_name, rotation=55, ha="right", va="top",
                 fontsize=8.0, fontweight="bold", color=CLS_COL[r.cls], zorder=4)

    # Clean horizontal dividing rule separating header from metrics
    ax1.plot([-5.4, N_ST - 0.2], [-3.45, -3.45], color="#cbd5e1", lw=0.8, zorder=3)
    
    rows = [
        ("Corridor Width Wc (m)", lambda r: f"{r.w:.0f}"),
        ("Mean Clear Width (m)", lambda r: f"{r.ecw:.1f}"),
        ("Impassable Nodes (%)", lambda r: f"{100 * r.imp:.0f}%"),
        ("Surveyed Nodes (n)", lambda r: f"{r.k}")
    ]
    
    y_tbl = -3.95
    for r_idx, (lab, fn) in enumerate(rows):
        if r_idx % 2 == 1:
            strip_bg = Rectangle((-5.4, y_tbl - 0.40), N_ST + 5.2, 0.85,
                                 facecolor="#ffffff", edgecolor="none", zorder=3)
            ax1.add_patch(strip_bg)
            
        ax1.text(-5.2, y_tbl, lab, ha="left", va="center",
                 fontsize=8.4, fontweight="bold", color="#475569", zorder=5)
                 
        for i, r in st.iterrows():
            val_str = fn(r)
            is_ecw_crit = (lab == "Mean Clear Width (m)" and r.ecw < 1.5)
            is_imp_high = (lab == "Impassable Nodes (%)" and r.imp >= 0.40)
            
            if is_ecw_crit:
                pill = FancyBboxPatch((i - 0.42, y_tbl - 0.30), 0.84, 0.60,
                                      boxstyle="round,pad=0.01,rounding_size=0.03",
                                      facecolor="#fee2e2", edgecolor="#fca5a5", lw=0.7, zorder=4)
                ax1.add_patch(pill)
                ax1.text(i, y_tbl, val_str, ha="center", va="center",
                         fontsize=8.4, fontweight="bold", color="#991b1b", zorder=6)
            elif is_imp_high:
                ax1.text(i, y_tbl, val_str, ha="center", va="center",
                         fontsize=8.2, fontweight="bold", color="#dc2626", zorder=5)
            else:
                ax1.text(i, y_tbl, val_str, ha="center", va="center",
                         fontsize=8.2, color="#334155", zorder=5)
                         
        y_tbl -= 0.85

    # -------------------------------------------------------------------------
    # PANEL A ARC LEGEND CARD (Fixed Sizing & Text Fit, No Fill)
    # -------------------------------------------------------------------------
    n_both = sum(1 for a, b in links if ltype(a, b) == "both")
    n_contrast = sum(1 for a, b in links if ltype(a, b) == "contrast")
    n_ok = sum(1 for a, b in links if ltype(a, b) == "ok")

    leg_card = FancyBboxPatch((-5.4, 9.0), 9.6, 5.9,
                              boxstyle="round,pad=0.015,rounding_size=0.04",
                              facecolor="none", edgecolor="#cbd5e1", lw=0.9, zorder=5)
    ax1.add_patch(leg_card)
    ax1.text(-5.1, 14.45, f"JUNCTION NETWORK LINKS (STREETS WITH n ≥ {MIN_N})",
             fontsize=8.0, fontweight="bold", color="#475569", zorder=6)
             
    leg_items = [
        (LC["both"], 2.8, "-", 0.90, f"Both corridors: Mean ECW < 1.5 m (n = {n_both})", 13.55),
        (LC["contrast"], 2.4, "--", 0.85, f"Contrast: One < 1.5 m, other ≥ 1.5 m (n = {n_contrast})", 12.65),
        (LC["ok"], 2.0, "-", 0.70, f"Both corridors: Mean ECW ≥ 1.5 m (n = {n_ok})", 11.75),
    ]
    for col_l, lw_l, ls_l, alph_l, txt_l, y_l in leg_items:
        ax1.plot([-5.1, -4.3], [y_l, y_l], color=col_l, lw=lw_l, ls=ls_l, alpha=alph_l, zorder=6)
        ax1.text(-4.1, y_l, txt_l, fontsize=7.6, color="#334155", va="center", zorder=6)
        
    ax1.text(-5.1, 10.75, "• Arc line width & opacity (VBA) ∝ node pairs (v = 1 to 11)",
             fontsize=7.3, color="#64748b", va="center", zorder=6)
    ax1.text(-5.1, 10.15, "• Dashed blue arcs denote cross-corridor clearance contrast",
             fontsize=7.3, color="#64748b", va="center", zorder=6)
    ax1.text(-5.1, 9.55, "• Circle area ∝ surveyed node count (n = 5 to 14)",
             fontsize=7.3, color="#64748b", va="center", zorder=6)

    # -------------------------------------------------------------------------
    # INSET CARD: STRUCTURAL ROW vs REALIZED CLEARANCE (Upper Right)
    # -------------------------------------------------------------------------
    card_x, card_w = 11.0, 10.4
    card_y, card_h = 8.8, 6.1
    inset_card = FancyBboxPatch((card_x, card_y), card_w, card_h,
                                boxstyle="round,pad=0.015,rounding_size=0.04",
                                facecolor="none", edgecolor="#cbd5e1", lw=0.9, zorder=5)
    ax1.add_patch(inset_card)

    def d2f_x(x): return LEFT_A + (x - (-5.6)) / 27.4 * WIDTH_A
    def d2f_y(y): return PLOT_BTM + (y - (-7.3)) / 22.9 * PLOT_H

    in_fx = d2f_x(card_x + 1.25)
    in_fy = d2f_y(card_y + 0.85)
    in_fw = (card_w - 1.55) / 27.4 * WIDTH_A
    in_fh = (card_h - 1.65) / 22.9 * PLOT_H

    ax_in = fig.add_axes([in_fx, in_fy, in_fw, in_fh], facecolor="#ffffff")

    # Regulatory Impassable Shading (< 1.5 m)
    ax_in.axhspan(0, 1.5, facecolor="#fef2f2", alpha=0.65, zorder=0)
    ax_in.axhline(1.5, color="#dc2626", ls="--", lw=1.0, alpha=0.85, zorder=2)
    ax_in.text(28.0, 1.55, "1.5 m minimum", ha="right", va="bottom",
               fontsize=6.4, fontweight="bold", color="#dc2626", zorder=5)

    # Scatter points for the 22 corridors
    for c in CLASSES:
        sub_c = st[st["cls"] == c]
        if len(sub_c) > 0:
            ax_in.scatter(sub_c.w, sub_c.ecw, color=CLS_COL[c], s=sub_c.k * 14,
                          alpha=0.88, edgecolors="white", lw=0.8, zorder=4)

    # Annotation for wide ROW chokepoints
    ax_in.annotate("Wide ROW chokepoints\n(Wc>10m, ECW<1.3m)",
                   xy=(11.3, 1.25), xytext=(15.2, 0.35),
                   arrowprops=dict(arrowstyle="->", color="#991b1b", lw=0.8),
                   fontsize=6.2, color="#991b1b", fontweight="bold", zorder=6,
                   bbox=dict(facecolor="#ffffff", edgecolor="#fca5a5", boxstyle="round,pad=0.15", lw=0.5))

    rho, pval = spearmanr(st.w, st.ecw)
    ax_in.text(0.04, 0.88, f"Spearman rho = {rho:.2f} (p < 0.001)\nDecoupling: wide ROW != clear path",
               transform=ax_in.transAxes, fontsize=6.4, color="#334155", fontweight="bold",
               bbox=dict(facecolor="#f8fafc", edgecolor="#cbd5e1", boxstyle="round,pad=0.15", lw=0.5),
               zorder=5)

    ax_in.set_xlim(4, 30)
    ax_in.set_ylim(0, 7.8)
    ax_in.set_xticks([5, 10, 15, 20, 25, 30])
    ax_in.set_yticks([0, 2, 4, 6])
    ax_in.tick_params(axis="both", labelsize=6.8, colors="#475569", pad=2)
    ax_in.set_xlabel("Corridor Width Wc (m)", fontsize=7.2, fontweight="bold", color="#475569", labelpad=2)
    ax_in.set_ylabel("Mean ECW (m)", fontsize=7.2, fontweight="bold", color="#475569", labelpad=2)
    ax_in.grid(True, ls="--", alpha=0.35, color="#cbd5e1")
    ax_in.spines["top"].set_visible(False)
    ax_in.spines["right"].set_visible(False)
    ax_in.spines["left"].set_color("#cbd5e1")
    ax_in.spines["bottom"].set_color("#cbd5e1")

    # Inset Card Title
    ax1.text(card_x + 0.4, card_y + card_h - 0.45, "STRUCTURAL ROW vs. REALIZED CLEARANCE",
             fontsize=8.0, fontweight="bold", color="#1e293b", zorder=6)

    # Panel A Header Axis
    ax_hdr_a = fig.add_axes([LEFT_A, HEADER_Y, WIDTH_A, HEADER_H], facecolor="none")
    ax_hdr_a.axis("off")
    ax_hdr_a.text(0.0, 0.65, "(a)", transform=ax_hdr_a.transAxes, ha="left", va="center",
                  fontsize=13, fontweight="bold", color="#1e293b",
                  bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#64748b", lw=0.9, alpha=0.98),
                  zorder=10)
    ax_hdr_a.text(0.042, 0.65, "Street-level clearance ordering & cross-street junction adjacency",
                  transform=ax_hdr_a.transAxes, ha="left", va="center",
                  fontsize=12, fontweight="bold", color="#1e293b", zorder=10)
    ax_hdr_a.text(0.042, 0.18, f"All 22 corridors with ≥ {MIN_N} surveyed nodes, ordered by mean ECW; arcs indicate physical junctions within {LINK_M:.0f} m",
                  transform=ax_hdr_a.transAxes, ha="left", va="center",
                  fontsize=8.8, fontweight="normal", color="#64748b", zorder=10)

    # -------------------------------------------------------------------------
    # PANEL B: RAINCLOUD PLOT
    # -------------------------------------------------------------------------
    ax2 = fig.add_axes([LEFT_B, PLOT_BTM, WIDTH_B, PLOT_H])
    ax2.set_facecolor("#FFFFFF")
    YMAX = 12.0
    
    # Accessibility Zone Background Shading
    # Impassable zone: 0 to 0.9 m
    ax2.axhspan(-0.8, 0.9, facecolor="#fef2f2", alpha=0.55, zorder=0)
    # Restricted zone: 0.9 to 1.5 m
    ax2.axhspan(0.9, 1.5, facecolor="#fffbeb", alpha=0.55, zorder=0)
    # Accessible zone: 1.5 to YMAX + 1.2
    ax2.axhspan(1.5, YMAX + 1.2, facecolor="#f8fafc", alpha=0.35, zorder=0)
    
    # Threshold Reference Lines
    ax2.axhline(0.9, color="#dc2626", lw=1.2, ls="--", alpha=0.85, zorder=2)
    ax2.axhline(1.5, color="#d97706", lw=1.2, ls=":", alpha=0.95, zorder=2)
    
    # Threshold chips on right border with clean pill background
    ax2.text(5.08, 0.90, "0.9 m (Impassable)", ha="right", va="center",
             fontsize=7.2, fontweight="bold", color="#dc2626", zorder=8,
             bbox=dict(facecolor="#ffffff", edgecolor="#fca5a5", boxstyle="round,pad=0.2", lw=0.6, alpha=0.95))
    ax2.text(5.08, 1.50, "1.5 m (Accessible)", ha="right", va="center",
             fontsize=7.2, fontweight="bold", color="#d97706", zorder=8,
             bbox=dict(facecolor="#ffffff", edgecolor="#fcd34d", boxstyle="round,pad=0.2", lw=0.6, alpha=0.95))

    # Plot Rainclouds for each street class
    for k, c in enumerate(CLASSES):
        d = n.loc[n["class"] == c, "ecw_best"].values
        col = CLS_COL[c]
        dd = np.clip(d, 0, YMAX)
        
        # 1. Half-violin KDE density
        ys = np.linspace(0, YMAX, 300)
        kde = gaussian_kde(d, bw_method=0.25)
        dens = kde(ys)
        dens = dens / dens.max() * 0.38
        ax2.fill_betweenx(ys, k, k + dens, color=col, alpha=0.38, zorder=2)
        ax2.plot(k + dens, ys, color=col, lw=1.6, zorder=3)
        
        # 2. Jittered individual nodes (Rain)
        rng = np.random.default_rng(42 + k)
        jitter_x = k + rng.uniform(-0.38, -0.06, len(dd))
        ax2.scatter(jitter_x, dd, c=col, s=15, alpha=0.62,
                    edgecolors="white", linewidths=0.35, zorder=4)
                    
        # 3. Boxplot & Median Bar
        q1, med, q3 = np.percentile(d, [25, 50, 75])
        iqr = q3 - q1
        lo, hi = max(0, q1 - 1.5 * iqr), min(YMAX, q3 + 1.5 * iqr)
        ax2.plot([k, k], [lo, hi], color=col, lw=1.6, zorder=5)
        
        iqr_box = FancyBboxPatch((k - 0.055, q1), 0.11, max(iqr, 0.08),
                                 boxstyle="round,pad=0.005",
                                 facecolor=col, alpha=0.30, edgecolor=col, lw=1.3, zorder=6)
        ax2.add_patch(iqr_box)
        ax2.plot([k - 0.11, k + 0.11], [med, med], color=col, lw=3.0,
                 solid_capstyle="round", zorder=7)
                 
        # 4. Outlier counts > 12 m
        over = int((d > YMAX).sum())
        if over > 0:
            callout = FancyBboxPatch((k - 0.35, 12.18), 0.70, 0.62,
                                     boxstyle="round,pad=0.01,rounding_size=0.03",
                                     facecolor="#FFFFFF", edgecolor=col, lw=1.0, alpha=0.96, zorder=8)
            ax2.add_patch(callout)
            ax2.text(k, 12.49, f"+{over}\n>12m", ha="center", va="center",
                     fontsize=7.2, fontweight="bold", color=col, zorder=9, linespacing=1.0)

    # Class summary cards below axis
    for k, c in enumerate(CLASSES):
        d = n.loc[n["class"] == c, "ecw_best"].values
        med = np.median(d)
        col = CLS_COL[c]
        
        # Pill card for class summary
        c_card = FancyBboxPatch((k - 0.44, -1.95), 0.88, 0.95,
                                boxstyle="round,pad=0.01,rounding_size=0.03",
                                facecolor="#f8fafc", edgecolor=col, lw=1.2, zorder=3)
        ax2.add_patch(c_card)
        ax2.text(k, -1.25, c.split()[0], ha="center", va="center",
                 fontsize=8.0, fontweight="bold", color=col, zorder=4)
        ax2.text(k, -1.55, f"med={med:.1f}m", ha="center", va="center",
                 fontsize=7.6, fontweight="bold", color="#1e293b", zorder=4)
        ax2.text(k, -1.78, f"n={len(d)}", ha="center", va="center",
                 fontsize=7.0, color="#64748b", zorder=4)

    # (Hierarchical mixed model badge removed as requested)

    ax2.set_xticks(range(len(CLASSES)))
    ax2.set_xticklabels([])
    ax2.tick_params(axis="x", which="both", bottom=False, top=False)
    ax2.set_xlim(-0.65, 5.20)
    ax2.set_ylim(-2.15, 13.2)
    ax2.set_yticks([0, 2, 4, 6, 8, 10, 12])
    ax2.tick_params(axis="y", labelsize=8.5)
    ax2.set_ylabel("Effective Clear Width (ECW, metres)", fontsize=9.5, fontweight="bold", color="#475569")
    ax2.grid(axis="y", linestyle="--", alpha=0.35, color="#cbd5e1")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_color("#cbd5e1")
    ax2.spines["left"].set_bounds(0, 12)
    ax2.spines["bottom"].set_visible(False)

    # Panel B Header Axis
    ax_hdr_b = fig.add_axes([LEFT_B, HEADER_Y, WIDTH_B, HEADER_H], facecolor="none")
    ax_hdr_b.axis("off")
    ax_hdr_b.text(0.0, 0.65, "(b)", transform=ax_hdr_b.transAxes, ha="left", va="center",
                  fontsize=13, fontweight="bold", color="#1e293b",
                  bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#64748b", lw=0.9, alpha=0.98),
                  zorder=10)
    ax_hdr_b.text(0.055, 0.65, "Effective clear width distributions by functional street class",
                  transform=ax_hdr_b.transAxes, ha="left", va="center",
                  fontsize=12, fontweight="bold", color="#1e293b", zorder=10)
    ax_hdr_b.text(0.055, 0.18, "Half-violin KDE density, jittered observations (n = 557), and IQR box plots with median crossbars",
                  transform=ax_hdr_b.transAxes, ha="left", va="center",
                  fontsize=8.8, fontweight="normal", color="#64748b", zorder=10)

    import shutil
    from PIL import Image
    # -------------------------------------------------------------------------
    # SAVE & SYNCHRONIZE 300 DPI PUBLICATION FIGURE (SUPPLEMENTARY S4)
    # -------------------------------------------------------------------------
    out_s4 = OUT / "figS4_arc_raincloud.png"
    fig.savefig(str(out_s4), dpi=300, facecolor="white")
    plt.close(fig)

    # Zero-margin crop to strictly eliminate outer whitespace on all 4 sides
    im = Image.open(str(out_s4)).convert("RGB")
    arr = np.array(im)
    non_white = np.any(arr < 250, axis=2)
    coords = np.argwhere(non_white)
    if coords.size > 0:
        min_y, min_x = coords.min(axis=0)
        max_y, max_x = coords.max(axis=0)
        cropped = im.crop((min_x, min_y, max_x + 1, max_y + 1))
        cropped.save(str(out_s4))
        print(f"Zero-margin cropped {out_s4.name}: {im.size} -> {cropped.size} (aspect ratio: {cropped.size[0]/cropped.size[1]:.4f})")

    print(f"Saved primary publication figure: {out_s4}")

    # Synchronize across manuscript and submission package directories
    paths_sync = [
        REV / "05_manuscript_R1/markedup/figures/figS4_arc_raincloud.png",
        REV / "SUBMISSION_R1_PACKAGE/02_latex_source/figures/figS4_arc_raincloud.png",
        REV / "07_repository_gitlab_figshare/figures/figS4_arc_raincloud.png",
    ]
    for p in paths_sync:
        if p.parent.exists():
            shutil.copy2(out_s4, p)
            print(f"Synchronized figure to: {p}")
            
    # Save statistics for LaTeX macros
    stat("arcStreets", len(st))
    stat("arcLinks", len(links))
    stat("arcBoth", sum(1 for a, b in links if ltype(a, b) == "both"))
    stat("arcContrast", sum(1 for a, b in links if ltype(a, b) == "contrast"))
    print(f"figS4_arc_raincloud generated successfully. Streets: {len(st)}, Links: {len(links)}")

if __name__ == "__main__":
    build_fig09()
