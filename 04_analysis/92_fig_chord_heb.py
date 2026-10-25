"""Relational structure figure: (a) chord diagram of obstacle footprint flows between street classes
and obstacle groups, (b) hierarchical edge bundling of cross-street junction links within 60 m.
Premier publication-standard overhaul adhering to publication_figure_design_guide.md.
Revised version of Figure 5 from the original submission; all values computed directly from R1 node table.
"""
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from pathlib import Path
from sigfig_common import *

def build_fig07():
    LINK_M = 60.0
    n = load()
    tot = float(n.obs_total.sum())
    
    # -------------------------------------------------------------------------
    # COMPACT 2:1 GEOMETRY: MINIMIZED WHITE SPACE & INTEGRATED PANEL BADGES
    # -------------------------------------------------------------------------
    # Data envelope: X [-1.48, 1.48] (dx = 2.96), Y [-1.28, 1.28] (dy = 2.56)
    # Ratio dx / dy = 1.15625
    FIG_H = 6.80
    FIG_W = FIG_H * 2.0  # 13.60 in (exact 2:1 canvas)
    
    MAIN_Y = 0.082
    MAIN_H = 0.840
    PANEL_H_IN = FIG_H * MAIN_H  # 5.712 in
    PANEL_W_IN = PANEL_H_IN * (2.96 / 2.56)  # 6.604 in
    
    GAP_IN = 0.20  # Tightened inter-panel gap
    LEFT_IN = (FIG_W - (2 * PANEL_W_IN + GAP_IN)) / 2.0
    
    POS_A_X = LEFT_IN / FIG_W
    POS_A_W = PANEL_W_IN / FIG_W
    
    POS_B_X = POS_A_X + POS_A_W + (GAP_IN / FIG_W)
    POS_B_W = PANEL_W_IN / FIG_W
    
    FOOT_Y = 0.012
    FOOT_H = 0.068
    
    INK = "#0f172a"
    INK_MUTED = "#64748b"
    
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="#FFFFFF", dpi=300)

    # -------------------------------------------------------------------------
    # PANEL A: CHORD DIAGRAM (HARMONIZED RADIUS & SNUG LABELS)
    # -------------------------------------------------------------------------
    ax1 = fig.add_axes([POS_A_X, MAIN_Y, POS_A_W, MAIN_H])
    ax1.set_aspect("equal")
    ax1.axis("off")
    ax1.set_facecolor("#FFFFFF")
    
    # Integrated Panel (a) Badge sitting right at the shoulder of the diagram
    ax1.text(0.04, 0.94, "(a)", fontsize=11.5, fontweight="bold", color=INK,
             bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor=INK_MUTED, lw=0.9, alpha=0.98),
             va="center", ha="left", transform=ax1.transAxes, zorder=20)
    
    labels = CLASSES + list(GROUPS)
    cols = [CLS_COL[c] for c in CLASSES] + [GROUP_COL[g] for g in GROUPS]
    flows = [(i, len(CLASSES) + j, n.loc[n["class"] == c, f"g_{g}"].sum())
             for i, c in enumerate(CLASSES) for j, g in enumerate(GROUPS)]
             
    N = len(labels)
    GAP = 0.038
    total = np.zeros(N)
    for i, j, v in flows:
        total[i] += v
        total[j] += v
        
    grand = total.sum()
    tot_actual = tot
    span = total / grand * (2 * np.pi - N * GAP)
    
    start = np.zeros(N)
    cur = 0.0
    for k in range(N):
        start[k] = cur
        cur += span[k] + GAP
        
    veh = labels.index("Vehicles")
    start += np.pi / 2 - (start[veh] + span[veh] / 2) # Vehicles centered at 12 o'clock
    
    # Outer arc reaches R = 1.07, outer labels at 1.115
    R_RIB, R_IN, R_OUT, R_LAB = 0.88, 0.92, 1.07, 1.115
    off = np.zeros(N)
    
    # Draw Bézier chord ribbons sorted by flow size
    for i, j, v in sorted(flows, key=lambda f: -f[2]):
        fi, fj = v / total[i], v / total[j]
        a1 = start[i] + off[i]
        a2 = start[j] + off[j]
        chord_ribbon(ax1, a1, a1 + fi * span[i], a2, a2 + fj * span[j], r=R_RIB, color=cols[i], alpha=0.34)
        off[i] += fi * span[i]
        off[j] += fj * span[j]
        
    # Draw Outer Arcs & Labels
    for k in range(N):
        a1, a2 = start[k], start[k] + span[k]
        th = np.linspace(a1, a2, 240)
        ax1.fill(np.r_[R_IN * np.cos(th), R_OUT * np.cos(th[::-1])],
                 np.r_[R_IN * np.sin(th), R_OUT * np.sin(th[::-1])],
                 color=cols[k], alpha=0.95, lw=0, zorder=5)
                 
        mid = (a1 + a2) / 2
        deg = np.degrees(mid) % 360
        actual_pct = 100.0 * total[k] / tot_actual
        actual_sqm = total[k]
        
        # Position label outside arc
        cos_m, sin_m = np.cos(mid), np.sin(mid)
        ha = "left" if cos_m > 0.25 else ("right" if cos_m < -0.25 else "center")
        va = "bottom" if sin_m > 0.55 else ("top" if sin_m < -0.55 else "center")
        
        lbl_text = f"{labels[k]}\n{actual_pct:.1f}% ({actual_sqm:,.0f} m²)"
        ax1.text(R_LAB * cos_m, R_LAB * sin_m, lbl_text,
                 fontsize=8.4, fontweight="bold", color=cols[k],
                 ha=ha, va=va, zorder=7,
                 path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
                 
        # Inside arc text if span is large enough
        if np.degrees(span[k]) > 14:
            rot_deg = deg - 90 if np.sin(mid) >= 0 else deg + 90
            r_arc_txt = (R_IN + R_OUT) / 2
            ax1.text(r_arc_txt * cos_m, r_arc_txt * sin_m, f"{actual_pct:.0f}%",
                     fontsize=7.4, color="white", fontweight="bold",
                     ha="center", va="center", zorder=8, rotation=rot_deg)

    # Central Medallion Donut Pill
    cent_bg = Circle((0, 0), 0.25, facecolor="#FFFFFF", edgecolor="#cbd5e1", lw=1.0, zorder=6)
    ax1.add_patch(cent_bg)
    ax1.text(0, 0.08, "MACRO FLOW", ha="center", va="center",
             fontsize=7.5, fontweight="bold", color="#64748b", zorder=7)
    ax1.text(0, -0.01, f"{tot_actual:,.0f} m²", ha="center", va="center",
             fontsize=11.5, fontweight="bold", color=OBJ["car"], zorder=7)
    ax1.text(0, -0.10, "obstacle footprint", ha="center", va="center",
             fontsize=7.0, style="italic", color="#475569", zorder=7)
             
    ax1.set_xlim(-1.48, 1.48)
    ax1.set_ylim(-1.28, 1.28)

    # -------------------------------------------------------------------------
    # PANEL B: HIERARCHICAL EDGE BUNDLING
    # -------------------------------------------------------------------------
    ax2 = fig.add_axes([POS_B_X, MAIN_Y, POS_B_W, MAIN_H])
    ax2.set_aspect("equal")
    ax2.axis("off")
    ax2.set_facecolor("#FFFFFF")
    
    # Integrated Panel (b) Badge sitting right at the shoulder of the diagram
    ax2.text(0.04, 0.94, "(b)", fontsize=11.5, fontweight="bold", color=INK,
             bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor=INK_MUTED, lw=0.9, alpha=0.98),
             va="center", ha="left", transform=ax2.transAxes, zorder=20)
    
    d = n.sort_values(["class", "street_id", "node"],
                      key=lambda s: s.map({c: i for i, c in enumerate(CLASSES)}) if s.name == "class" else s).reset_index(drop=True)
    d["ci"] = d["class"].map({c: i for i, c in enumerate(CLASSES)})
    
    NC = len(CLASSES)
    GAPD = 4.0
    cnt = d.groupby("ci").size().reindex(range(NC)).values
    per_deg = (360.0 - NC * GAPD) / cnt.sum()
    
    ang = np.zeros(len(d))
    sect = []
    a0 = 0.0
    for ci in range(NC):
        idx = np.where(d.ci.values == ci)[0]
        ang[idx] = np.radians(a0 + np.arange(len(idx)) * per_deg + per_deg / 2)
        sect.append((a0, a0 + len(idx) * per_deg))
        a0 += len(idx) * per_deg + GAPD
        
    RN, RI = 1.0, 0.40
    lx, ly = RN * np.cos(ang), RN * np.sin(ang)
    hub = {ci: (RI * np.cos(np.radians(np.mean(sect[ci]))), RI * np.sin(np.radians(np.mean(sect[ci]))))
           for ci in range(NC)}
           
    tree = cKDTree(np.c_[d.x_node, d.y_node])
    pairs = [(i, j) for i, j in tree.query_pairs(LINK_M) if d.street_id[i] != d.street_id[j]]
    tier = d.tier3.values
    
    def etype(i, j):
        a, b = sorted([tier[i], tier[j]])
        if a == b == "Impassable":
            return "both"
        if "Impassable" in (a, b) and "Accessible" in (a, b):
            return "contrast"
        if "Impassable" in (a, b):
            return "half"
        return "ok"
        
    ECOL = {
        "both": "#dc2626",      # Bright acute rose/red
        "contrast": "#0284c7",  # Bright sky/slate blue
        "half": "#f59e0b",      # Warm amber
        "ok": "#cbd5e1"         # Subtle neutral grey
    }
    ELW = {
        "both": (1.45, 0.72),
        "contrast": (1.15, 0.50),
        "half": (0.85, 0.32),
        "ok": (0.50, 0.16)
    }
    
    beta = 0.82
    ecount = {k: 0 for k in ECOL}
    
    # Draw bundled edges with strict z-order
    for order_type in ("ok", "half", "contrast", "both"):
        for i, j in pairs:
            if etype(i, j) != order_type:
                continue
            ecount[order_type] += 1
            ca, cb = hub[d.ci[i]], hub[d.ci[j]]
            p1 = (lx[i] * (1 - beta) + ca[0] * beta, ly[i] * (1 - beta) + ca[1] * beta)
            p2 = (lx[j] * (1 - beta) + cb[0] * beta, ly[j] * (1 - beta) + cb[1] * beta)
            m = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            t = np.linspace(0, 1, 60)
            
            def bez(a, c1, c2, b_):
                return ((1 - t) ** 3 * a[0] + 3 * (1 - t) ** 2 * t * c1[0] + 3 * (1 - t) * t ** 2 * c2[0] + t ** 3 * b_[0],
                        (1 - t) ** 3 * a[1] + 3 * (1 - t) ** 2 * t * c1[1] + 3 * (1 - t) * t ** 2 * c2[1] + t ** 3 * b_[1])
                        
            x1, y1 = bez((lx[i], ly[i]), p1, m, m)
            x2, y2 = bez(m, m, p2, (lx[j], ly[j]))
            lw, al = ELW[order_type]
            ax2.plot(np.r_[x1, x2], np.r_[y1, y2], color=ECOL[order_type], lw=lw, alpha=al, zorder=2)
            
    # Node markers at R = 1.0 (colored by street class)
    ax2.scatter(lx, ly, c=[CLS_COL[c] for c in d["class"]], s=20, zorder=5, edgecolors="white", linewidths=0.5)
    
    # Outer tier track markers at R = 1.055 (colored by clearance tier)
    ax2.scatter(1.055 * np.cos(ang), 1.055 * np.sin(ang),
                c=[TIER_COL[t] for t in tier], s=14, zorder=6, marker="s", edgecolors="white", linewidths=0.3)
                
    # Class Arc Ribbons around perimeter
    for ci, c in enumerate(CLASSES):
        a_s, a_e = np.radians(sect[ci][0]), np.radians(sect[ci][1])
        arc_th = np.linspace(a_s, a_e, 100)
        ax2.plot(1.095 * np.cos(arc_th), 1.095 * np.sin(arc_th), color=CLS_COL[c], lw=4.5, solid_capstyle="butt", zorder=7)
        
        mid = np.radians(np.mean(sect[ci]))
        cnt_ci = int(cnt[ci])
        # Snug position for class labels right outside the arc
        ax2.text(1.20 * np.cos(mid), 1.20 * np.sin(mid), f"{c}\n(n={cnt_ci})",
                 ha="center", va="center", fontsize=8.8, fontweight="bold", color=CLS_COL[c], zorder=8,
                 path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
                 
        # Bundling Hub Badges
        kx, ky = hub[ci]
        hub_patch = FancyBboxPatch((kx - 0.12, ky - 0.05), 0.24, 0.10,
                                   boxstyle="round,pad=0.015,rounding_size=0.03",
                                   facecolor="#FFFFFF", edgecolor=CLS_COL[c], lw=1.4, alpha=0.95, zorder=10)
        ax2.add_patch(hub_patch)
        ax2.text(kx, ky, c.split()[0], ha="center", va="center",
                 fontsize=7.4, fontweight="bold", color=CLS_COL[c], zorder=11)
                 
    ax2.set_xlim(-1.48, 1.48)
    ax2.set_ylim(-1.28, 1.28)

    # -------------------------------------------------------------------------
    # CLEAN UNBOXED LEGENDS (ZERO FILLED CONTAINER CARDS)
    # -------------------------------------------------------------------------
    # Panel (a) Bottom Note: Obstacle allocation summary
    ax_foot_a = fig.add_axes([POS_A_X, FOOT_Y, POS_A_W, FOOT_H], facecolor="none")
    ax_foot_a.axis("off")
    ax_foot_a.text(0.00, 0.16, f"Macro clutter: {tot:,.0f} m² total footprint across {len(n)} surveyed nodes. Ribbon widths reflect directional square meter burden.",
                   fontsize=7.2, fontstyle="italic", color="#64748b", va="center", transform=ax_foot_a.transAxes)

    # Panel (b) Bottom Legend: Clearance tier track (ABOVE Junction Links) and link typology (BELOW Junction Links)
    ax_foot_b = fig.add_axes([POS_B_X, FOOT_Y, POS_B_W, FOOT_H], facecolor="none")
    ax_foot_b.axis("off")
    
    # 1. Outer track tier key (MOVED FROM BOTTOM-RIGHT TO ABOVE JUNCTION LINKS at y=0.85)
    ax_foot_b.text(0.00, 0.85, "CLEARANCE TIER:",
                   fontsize=6.8, fontweight="bold", color="#64748b", va="center", transform=ax_foot_b.transAxes)
    tier_info = [
        ("Impassable (<0.9m)", TIER_COL["Impassable"], 0.16),
        ("Restricted (0.9–1.5m)", TIER_COL["Restricted"], 0.38),
        ("Accessible (≥1.5m)", TIER_COL["Accessible"], 0.62)
    ]
    for t_name, t_col, tx_pos in tier_info:
        ax_foot_b.scatter(tx_pos, 0.85, color=t_col, s=16, marker="s", transform=ax_foot_b.transAxes)
        ax_foot_b.text(tx_pos + 0.018, 0.85, t_name, fontsize=6.5, color="#334155", va="center", transform=ax_foot_b.transAxes)

    # 2. Junction links header (FIXED at y=0.48)
    ax_foot_b.text(0.00, 0.48, f"JUNCTION LINKS (≤{LINK_M:.0f} m):",
                   fontsize=7.2, fontweight="bold", color="#475569", va="center", transform=ax_foot_b.transAxes)
                   
    # 3. 4 link categories with lines (FIXED at y=0.16, baseline-aligned with Panel a)
    link_info = [
        ("Both impassable", ecount["both"], ECOL["both"], 0.00),
        ("Impassable/Accessible", ecount["contrast"], ECOL["contrast"], 0.22),
        ("Impassable/Restricted", ecount["half"], ECOL["half"], 0.47),
        ("Clear walkway", ecount["ok"], "#94a3b8", 0.72)
    ]
    for lbl, cnt_val, col_val, lx_pos in link_info:
        pct_val = 100.0 * cnt_val / len(pairs)
        ax_foot_b.plot([lx_pos, lx_pos + 0.024], [0.16, 0.16], color=col_val, lw=2.4,
                       transform=ax_foot_b.transAxes)
        ax_foot_b.text(lx_pos + 0.030, 0.16, f"{lbl} ({pct_val:.0f}%)",
                       fontsize=6.8, color="#334155", va="center", transform=ax_foot_b.transAxes)

    # -------------------------------------------------------------------------
    # SAVE & SYNCHRONIZE 300 DPI PUBLICATION FIGURE
    # -------------------------------------------------------------------------
    out_primary = OUT / "fig07_chord_heb.png"
    fig.savefig(out_primary, dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)
    
    # Zero out excess white margin on the right
    from PIL import Image
    im = Image.open(out_primary)
    arr = np.array(im.convert("L"))
    cols = np.where((arr < 250).sum(axis=0) > 0)[0]
    left_pad = int(cols.min())
    right_max = int(cols.max())
    im_cropped = im.crop((0, 0, min(im.width, right_max + left_pad + 2), im.height))
    im_cropped.save(out_primary)
    print(f"Trimmed right margin: width reduced from {im.width} to {im_cropped.width} px")
    print(f"Saved primary publication figure: {out_primary}")
    
    # Sync to manuscript markedup and submission package
    paths_sync = [
        REV / "05_manuscript_R1/markedup/figures/fig07_chord_heb.png",
        REV / "SUBMISSION_R1_PACKAGE/02_latex_source/figures/fig07_chord_heb.png"
    ]
    for p in paths_sync:
        if p.parent.exists():
            shutil.copy2(out_primary, p)
            print(f"Synchronized figure to: {p}")
            
    # Save statistics for LaTeX macros
    stat("hebLinks", len(pairs))
    stat("hebBoth", ecount["both"])
    stat("hebContrast", ecount["contrast"])
    stat("hebHalf", ecount["half"])
    stat("hebOk", ecount["ok"])
    stat("hebLinkM", int(LINK_M))
    print(f"fig07_chord_heb generated successfully. Links: {len(pairs)}, Both: {ecount['both']}, Contrast: {ecount['contrast']}")

if __name__ == "__main__":
    build_fig07()
