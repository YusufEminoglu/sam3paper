"""Alluvial (Sankey) & Conditional Risk Decomposition figure:
(a) multi-stage alluvial flow: street class -> dominant obstacle group -> ECW tier (flow width = node count),
(b) conditional tier distributions and relative risk of impassability across obstacle groups.
Premier publication-standard overhaul adhering to publication_figure_design_guide.md.
Revised version of Figure 6 from the original submission; all values computed directly from R1 node table.
"""
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from pathlib import Path
from sigfig_common import *

def build_fig08():
    n = load()
    TOT = len(n)
    
    # -------------------------------------------------------------------------
    # GEOMETRY ARCHITECTURE: 18.0 x 9.8 in (Standardized width, tight padding)
    # -------------------------------------------------------------------------
    FIG_W = 18.0
    FIG_H = 9.8
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="#FFFFFF", dpi=300)
    
    LEFT_IN = 0.65
    PANEL_A_W_IN = 10.10
    GAP_IN = 0.75
    PANEL_B_W_IN = 5.85
    # Right margin = 18.0 - (0.65 + 10.10 + 0.75 + 5.85) = 0.65 in (symmetric!)
    
    POS_A_X = LEFT_IN / FIG_W
    POS_A_W = PANEL_A_W_IN / FIG_W
    
    POS_B_X = POS_A_X + POS_A_W + (GAP_IN / FIG_W)
    POS_B_W = PANEL_B_W_IN / FIG_W
    
    MAIN_Y = 0.055
    MAIN_H = 0.815  # ends at 0.870
    
    HDR_Y = 0.885
    HDR_H = 0.060
    
    INK = "#0f172a"
    INK_MUTED = "#64748b"
    
    # -------------------------------------------------------------------------
    # PANEL A: STANDARDIZED HEADER
    # -------------------------------------------------------------------------
    ax_hdr_a = fig.add_axes([POS_A_X, HDR_Y, POS_A_W, HDR_H], facecolor="none")
    ax_hdr_a.axis("off")
    ax_hdr_a.text(0.0, 0.65, "(a)", transform=ax_hdr_a.transAxes, ha="left", va="center",
                  fontsize=12, fontweight="bold", color=INK,
                  bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=INK_MUTED, lw=0.9, alpha=0.98),
                  zorder=10)
    ax_hdr_a.text(0.042, 0.65, "Multi-stage alluvial flow: Street class → Dominant obstacle group → ECW tier",
                  transform=ax_hdr_a.transAxes, ha="left", va="center",
                  fontsize=11.5, fontweight="bold", color=INK, zorder=10)
    ax_hdr_a.text(0.042, 0.18, "Flow width is proportional to surveyed node count (n = 557); dominant group represents the largest footprint at the node",
                  transform=ax_hdr_a.transAxes, ha="left", va="center",
                  fontsize=8.3, fontweight="normal", color=INK_MUTED, zorder=10)

    # -------------------------------------------------------------------------
    # PANEL B: STANDARDIZED HEADER
    # -------------------------------------------------------------------------
    ax_hdr_b = fig.add_axes([POS_B_X, HDR_Y, POS_B_W, HDR_H], facecolor="none")
    ax_hdr_b.axis("off")
    ax_hdr_b.text(0.0, 0.65, "(b)", transform=ax_hdr_b.transAxes, ha="left", va="center",
                  fontsize=12, fontweight="bold", color=INK,
                  bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=INK_MUTED, lw=0.9, alpha=0.98),
                  zorder=10)
    ax_hdr_b.text(0.075, 0.65, "Conditional tier outcomes & relative risk by obstacle group",
                  transform=ax_hdr_b.transAxes, ha="left", va="center",
                  fontsize=11.5, fontweight="bold", color=INK, zorder=10)
    ax_hdr_b.text(0.075, 0.18, "Distribution of ECW clearance tiers and risk ratio relative to clear walkways",
                  transform=ax_hdr_b.transAxes, ha="left", va="center",
                  fontsize=8.3, fontweight="normal", color=INK_MUTED, zorder=10)

    # -------------------------------------------------------------------------
    # PANEL A: ALLUVIAL MULTI-STAGE FLOW DIAGRAM
    # -------------------------------------------------------------------------
    ax1 = fig.add_axes([POS_A_X, MAIN_Y, POS_A_W, MAIN_H])
    ax1.axis("off")
    ax1.set_xlim(0, 9.10)
    ax1.set_ylim(0.15, 10.45)
    
    DOM = ["Vehicles", "Furniture", "Plants", "Infrastructure", "None"]
    DOM_LAB = {
        "Vehicles": "Vehicles",
        "Furniture": "Furniture",
        "Plants": "Plants",
        "Infrastructure": "Infrastructure",
        "None": "No obstacle"
    }
    
    Y0, H, PAD, BW, YH = 0.35, 8.65, 0.14, 0.58, 9.75
    
    def layout(items, xc):
        out, cur = {}, Y0
        for key, v in items:
            h = v / TOT * H
            out[key] = (xc, cur, h, v)
            cur += h + PAD
        return out
        
    cls_items = [(c, int((n["class"] == c).sum())) for c in CLASSES]
    dom_items = [(g, int((n.dominant == g).sum())) for g in DOM if (n.dominant == g).sum() > 0]
    tier_items = [(t, int((n.tier3 == t).sum())) for t in TIERS[::-1]]
    
    XC1, XC2, XC3 = 1.15, 4.35, 7.50
    P1 = layout(cls_items, XC1)
    P2 = layout(dom_items, XC2)
    P3 = layout(tier_items, XC3)
    
    def band(x1, y1, x2, y2, h, color, alpha=0.34):
        t = np.linspace(0, 1, 120)
        s = 3 * t ** 2 - 2 * t ** 3
        xt = x1 + (x2 - x1) * t
        top = y1 + (y2 - y1) * s
        bot = (y1 - h) + ((y2 - h) - (y1 - h)) * s
        ax1.fill(np.r_[xt, xt[::-1]], np.r_[top, bot[::-1]], color=color, alpha=alpha, lw=0.3, edgecolor=color)
        
    def flow(P_a, P_b, col_a, col_b, colour_of):
        off_a = {k: 0.0 for k in P_a}
        off_b = {k: 0.0 for k in P_b}
        for ka in P_a:
            for kb in P_b:
                v = int(((n[col_a] == ka) & (n[col_b] == kb)).sum())
                if v == 0:
                    continue
                h = v / TOT * H
                band(P_a[ka][0] + BW, P_a[ka][1] + off_a[ka] + h,
                     P_b[kb][0] - BW, P_b[kb][1] + off_b[kb] + h,
                     h, colour_of(ka, kb))
                off_a[ka] += h
                off_b[kb] += h
                
    flow(P1, P2, "class", "dominant", lambda a, b: CLS_COL[a])
    flow(P2, P3, "dominant", "tier3", lambda a, b: GROUP_COL[a])
    
    # Draw Stage Bars
    def draw_bars(P, colour_fn, label_fn, is_dom=False):
        for k, (x, y, h, v) in P.items():
            c_fill = colour_fn(k)
            rect = FancyBboxPatch((x - BW, y), 2 * BW, max(h, 0.05),
                                  boxstyle="round,pad=0.012,rounding_size=0.05",
                                  facecolor=c_fill, edgecolor="white", lw=1.2, alpha=0.95, zorder=5)
            ax1.add_patch(rect)
            
            pct = 100.0 * v / TOT
            if h >= 0.45:
                tc = "#1e293b" if k == "None" or k == "Furniture" or k == "Restricted" else "white"
                ax1.text(x, y + h / 2, label_fn(k, v, pct),
                         ha="center", va="center", fontsize=8.0, fontweight="bold",
                         color=tc, multialignment="center", zorder=6)
            elif is_dom and k == "Plants":
                callout = FancyBboxPatch((x + BW + 0.12, y + h / 2 - 0.20), 1.62, 0.40,
                                         boxstyle="round,pad=0.01,rounding_size=0.04",
                                         facecolor="#FFFFFF", edgecolor=GROUP_COL["Plants"], lw=1.2,
                                         alpha=0.98, zorder=7)
                ax1.add_patch(callout)
                ax1.plot([x + BW, x + BW + 0.12], [y + h / 2, y + h / 2],
                         color=GROUP_COL["Plants"], lw=1.2, zorder=7)
                ax1.text(x + BW + 0.93, y + h / 2, f"Plants: {pct:.1f}% (n={v})",
                         ha="center", va="center", fontsize=7.4, fontweight="bold",
                         color=GROUP_COL["Plants"], zorder=8)
            else:
                tc = "#1e293b" if k == "None" or k == "Furniture" else "white"
                ax1.text(x, y + h / 2, f"{pct:.0f}%",
                         ha="center", va="center", fontsize=7.0, fontweight="bold",
                         color=tc, zorder=6)
                         
    draw_bars(P1, lambda k: CLS_COL[k], lambda k, v, p: f"{k}\n{p:.1f}% (n={v})")
    draw_bars(P2, lambda k: GROUP_COL[k], lambda k, v, p: f"{DOM_LAB.get(k, k)}\n{p:.1f}% (n={v})", is_dom=True)
    draw_bars(P3, lambda k: TIER_COL[k], lambda k, v, p: f"{k}\n{p:.1f}% (n={v})")
    
    # Stage Column Header Labels (Clean unboxed text, no filled container cards)
    stage_headers = [
        (XC1, "STAGE 1: STREET ENVIRONMENT", "Surveyed Street Class (n = 557)"),
        (XC2, "STAGE 2: CLUTTER TYPOLOGY", "Dominant Obstacle Group at Node"),
        (XC3, "STAGE 3: CLEARANCE OUTCOME", "Effective Clear Width (ECW) Tier")
    ]
    for x_c, title_s, sub_s in stage_headers:
        ax1.text(x_c, YH + 0.14, title_s, ha="center", va="center",
                 fontsize=8.0, fontweight="bold", color="#1e293b", zorder=5)
        ax1.text(x_c, YH - 0.10, sub_s, ha="center", va="center",
                 fontsize=6.8, fontweight="normal", color="#64748b", zorder=5)
                 
    # Right Side Accolade Grouping Cards
    acc = P3["Accessible"]
    imp = P3["Impassable"]
    res = P3["Restricted"]
    
    # 1. Accessible Accolade Bracket & Vertical Labels (No filled card)
    acc_y_bot = acc[1]
    acc_y_top = acc[1] + acc[2]
    acc_mid = (acc_y_bot + acc_y_top) / 2
    bracket_x1 = XC3 + 0.65
    bracket_x2 = bracket_x1 + 0.13
    ax1.plot([bracket_x1, bracket_x2, bracket_x2, bracket_x1], [acc_y_bot, acc_y_bot, acc_y_top, acc_y_top],
             color=TIER_COL["Accessible"], lw=1.8, zorder=5)
             
    lbl_x1 = bracket_x2 + 0.26
    lbl_x2 = lbl_x1 + 0.28
    ax1.text(lbl_x1, acc_mid, f"ACCESSIBLE  •  61.9% ({acc[3]} nodes)", ha="center", va="center",
             rotation=90, fontsize=8.6, fontweight="bold", color=TIER_COL["Accessible"], zorder=6)
    ax1.text(lbl_x2, acc_mid, "ECW ≥ 1.50 m (Compliant)  •  Unobstructed", ha="center", va="center",
             rotation=90, fontsize=7.0, color="#0284c7", zorder=6)

    # 2. Sub-Standard Accolade Bracket & Vertical Labels (No filled card)
    sub_y_bot = res[1]
    sub_y_top = imp[1] + imp[2]
    sub_mid = (sub_y_bot + sub_y_top) / 2
    ax1.plot([bracket_x1, bracket_x2, bracket_x2, bracket_x1], [sub_y_bot, sub_y_bot, sub_y_top, sub_y_top],
             color=TIER_COL["Impassable"], lw=1.8, zorder=5)
             
    ax1.text(lbl_x1, sub_mid, f"SUB-STANDARD  •  38.1% ({imp[3] + res[3]} nodes)", ha="center", va="center",
             rotation=90, fontsize=8.6, fontweight="bold", color="#991b1b", zorder=6)
    ax1.text(lbl_x2, sub_mid, f"ECW < 1.50 m (Non-compliant)  •  {imp[3]} Impassable, {res[3]} Restricted",
             ha="center", va="center", rotation=90, fontsize=6.8, color="#b91c1c", zorder=6)

    # -------------------------------------------------------------------------
    # PANEL B: CONDITIONAL TIER OUTCOMES & RISK DECOMPOSITION
    # -------------------------------------------------------------------------
    B2_Y = 0.055
    B2_H = 0.355
    B1_Y = 0.485
    B1_H = 0.355
    
    ax_b1 = fig.add_axes([POS_B_X, B1_Y, POS_B_W, B1_H])
    ax_b1.set_facecolor("#FFFFFF")
    
    dom_order = ["None", "Infrastructure", "Furniture", "Vehicles", "Plants"]
    dom_display = ["No obstacle", "Infrastructure", "Furniture", "Vehicles", "Plants"]
    
    ct_counts = pd.crosstab(n.dominant, n.tier3).reindex(dom_order)
    ct_pct = (ct_counts.div(ct_counts.sum(axis=1), axis=0) * 100.0)
    
    y_pos = np.arange(len(dom_order))
    bar_h = 0.58
    
    p_acc = ct_pct["Accessible"].values
    p_res = ct_pct["Restricted"].values
    p_imp = ct_pct["Impassable"].values
    
    b_acc = ax_b1.barh(y_pos, p_acc, height=bar_h, color=TIER_COL["Accessible"], alpha=0.92,
                       edgecolor="white", linewidth=1.0, label="Accessible (≥1.5m)")
    b_res = ax_b1.barh(y_pos, p_res, left=p_acc, height=bar_h, color=TIER_COL["Restricted"], alpha=0.92,
                       edgecolor="white", linewidth=1.0, label="Restricted (0.9-1.5m)")
    b_imp = ax_b1.barh(y_pos, p_imp, left=p_acc + p_res, height=bar_h, color=TIER_COL["Impassable"], alpha=0.92,
                       edgecolor="white", linewidth=1.0, label="Impassable (<0.9m)")
                       
    # Percentage labels inside segments
    for idx in range(len(dom_order)):
        if p_acc[idx] > 12:
            ax_b1.text(p_acc[idx] / 2, y_pos[idx], f"{p_acc[idx]:.1f}%",
                       ha="center", va="center", fontsize=7.2, fontweight="bold", color="white")
        if p_res[idx] > 8:
            ax_b1.text(p_acc[idx] + p_res[idx] / 2, y_pos[idx], f"{p_res[idx]:.1f}%",
                       ha="center", va="center", fontsize=7.2, fontweight="bold", color="#1e293b")
        if p_imp[idx] > 8:
            ax_b1.text(p_acc[idx] + p_res[idx] + p_imp[idx] / 2, y_pos[idx], f"{p_imp[idx]:.1f}%",
                       ha="center", va="center", fontsize=7.2, fontweight="bold", color="white")
                       
    ax_b1.set_yticks(y_pos)
    ax_b1.set_yticklabels([f"{d}\n(n={ct_counts.sum(axis=1).values[i]})" for i, d in enumerate(dom_display)],
                          fontsize=7.8, fontweight="bold", color="#1e293b")
    ax_b1.set_xlim(0, 100)
    ax_b1.set_ylim(-0.6, 4.65)
    ax_b1.set_xlabel("Conditional Distribution within Group (%)", fontsize=7.8, fontweight="bold", color="#475569")
    ax_b1.xaxis.set_major_formatter(ticker.PercentFormatter())
    ax_b1.spines["top"].set_visible(False)
    ax_b1.spines["right"].set_visible(False)
    ax_b1.spines["left"].set_color("#cbd5e1")
    ax_b1.spines["bottom"].set_color("#cbd5e1")
    ax_b1.grid(axis="x", linestyle="--", alpha=0.35, color="#cbd5e1")
    
    # Legend cleanly placed above plot
    ax_b1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False,
                 fontsize=7.4, handlelength=1.1, handleheight=0.85)

    # B2: Impassability Relative Risk Ratio (RR vs. No Obstacle)
    ax_b2 = fig.add_axes([POS_B_X, B2_Y, POS_B_W, B2_H])
    ax_b2.set_facecolor("#FFFFFF")
    
    base_rate = ct_counts.loc["None", "Impassable"] / ct_counts.loc["None"].sum()
    group_rates = ct_counts["Impassable"] / ct_counts.sum(axis=1)
    rr_vals = (group_rates / base_rate).values
    
    rr_colors = [
        "#94a3b8" if d == "None" else
        GROUP_COL[d] for d in dom_order
    ]
    
    bars_rr = ax_b2.barh(y_pos, rr_vals, height=bar_h, color=rr_colors, alpha=0.92,
                         edgecolor="white", linewidth=1.0)
                         
    # Reference line at RR = 1.0 (Baseline)
    ax_b2.axvline(1.0, color="#dc2626", linestyle=":", lw=1.2, zorder=3)
    ax_b2.text(1.05, 4.32, "1.0× Baseline (No obstacle: 8.3%)", color="#dc2626",
               fontsize=7.0, fontweight="bold", va="center")
               
    # Annotate values at bar ends
    for idx, rect in enumerate(bars_rr):
        w = rect.get_width()
        imp_n = ct_counts.loc[dom_order[idx], 'Impassable']
        tot_n = ct_counts.sum(axis=1).values[idx]
        ax_b2.text(w + 0.10, y_pos[idx], f"{w:.2f}× ({imp_n}/{tot_n})",
                   ha="left", va="center", fontsize=7.2, fontweight="bold", color="#334155")
                   
    ax_b2.set_yticks(y_pos)
    ax_b2.set_yticklabels(dom_display, fontsize=7.8, fontweight="bold", color="#1e293b")
    ax_b2.set_xlim(0, 4.8)
    ax_b2.set_ylim(-0.6, 4.65)
    ax_b2.set_xlabel("Relative Risk of Impassability (RR vs. No Obstacle Baseline)",
                     fontsize=7.8, fontweight="bold", color="#475569")
    ax_b2.spines["top"].set_visible(False)
    ax_b2.spines["right"].set_visible(False)
    ax_b2.spines["left"].set_color("#cbd5e1")
    ax_b2.spines["bottom"].set_color("#cbd5e1")
    ax_b2.grid(axis="x", linestyle="--", alpha=0.35, color="#cbd5e1")

    # -------------------------------------------------------------------------
    # SAVE & SYNCHRONIZE 300 DPI PUBLICATION FIGURE
    # -------------------------------------------------------------------------
    out_primary = OUT / "fig08_sankey.png"
    fig.savefig(out_primary, dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)
    print(f"Saved primary publication figure: {out_primary}")
    
    # Sync to manuscript markedup and submission package
    paths_sync = [
        REV / "05_manuscript_R1/markedup/figures/fig08_sankey.png",
        REV / "SUBMISSION_R1_PACKAGE/02_latex_source/figures/fig08_sankey.png"
    ]
    for p in paths_sync:
        if p.parent.exists():
            shutil.copy2(out_primary, p)
            print(f"Synchronized figure to: {p}")
            
    # Save statistics for LaTeX macros
    stat("sankVehDomPct", round(100 * (n.dominant == "Vehicles").mean(), 1))
    stat("sankNoObsPct", round(100 * (n.dominant == "None").mean(), 1))
    stat("sankVehImpPct", round(100 * ((n.dominant == "Vehicles") & (n.tier3 == "Impassable")).sum() / max(1, (n.dominant == "Vehicles").sum()), 1))
    stat("sankNoObsImpPct", round(100 * ((n.dominant == "None") & (n.tier3 == "Impassable")).sum() / max(1, (n.dominant == "None").sum()), 1))
    stat("sankInfraDomPct", round(100 * (n.dominant == "Infrastructure").mean(), 1))
    stat("sankFurnDomPct", round(100 * (n.dominant == "Furniture").mean(), 1))
    
    _z = n[(n.dominant == "None") & (n.tier3 == "Impassable")]
    stat("sankNoObsImpN", len(_z))
    stat("sankNoObsImpNarrow", int((_z.zone_w < 0.9).sum()))
    print("fig08_sankey generated successfully.")

if __name__ == "__main__":
    build_fig08()
