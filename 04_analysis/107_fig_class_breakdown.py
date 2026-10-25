"""Figure S2: ECW Tiers and Obstacle Composition by Street Class.

Comprehensive supplementary plate layout:
- Panel (a): ECW accessibility tier distribution across the 5 street classes with
  exact segment percentage labels and sample sizes.
- Panel (b): Best-side ECW empirical dispersion across street classes featuring
  shaded accessibility zones (<0.9 m impassable, 0.9-1.5 m restricted, >=1.5 m accessible),
  horizontal IQR boxes, prominent median bars with numerical readouts, and jittered strip points.
- Panel (c): Obstacle composition and ground area footprint by street class (Vehicles,
  Commercial furniture & plants, Fixed infrastructure) with total area badges.
- Panel (d): Phase-space analysis of Phantom Capacity against Blocked Ground Area
  (symmetric logarithmic axis) colored by street class, with binned median trend ribbon
  and explanatory structural vs. buffer callouts.
- Symmetrical bottom analytical summary cards synthesizing street hierarchy polarization
  and obstacle morphology dynamics.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
from pathlib import Path
from sigfig_common import *

def build_figS2():
    n = pd.read_csv(HERE / "out_final/nodes_R1.csv")
    CLS_ORDER = ["Boulevard", "Collector avenue", "Wide local street", "Narrow local street", "Shared lane"]
    n["class"] = pd.Categorical(n["class"], CLS_ORDER, ordered=True)

    TIER_COLORS = {
        "Impassable (<0.9 m)": "#c77d7d",
        "Restricted (0.9-1.5 m)": "#cfb57e",
        "Accessible (>=1.5 m)": "#4575b4"
    }

    OBS_COLORS = {
        "Vehicles": "#c77d7d",
        "Commercial furniture and plants": "#cfb57e",
        "Fixed infrastructure": "#7d94b5"
    }

    # Setup canvas (3:2 aspect ratio: 18.0 x 12.0 inches, 5400 x 3600 px @ 300 DPI)
    FIG_W, FIG_H = 18.0, 12.0
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="#FFFFFF")

    # -------------------------------------------------------------------------
    # MAIN COORDINATES (Zero-margin edge-to-edge layout calibrated for 3:2 aspect ratio)
    # -------------------------------------------------------------------------
    panel1_x = 0.090
    panel_w = 0.388
    panel2_x = 0.522

    row2_y = 0.045
    row2_h = 0.410

    row1_y = 0.535
    row1_h = 0.410

    # -------------------------------------------------------------------------
    # PANEL (a): ECW TIER SHARE BY STREET CLASS
    # -------------------------------------------------------------------------
    ax_a = fig.add_axes([panel1_x, row1_y, panel_w, row1_h])
    ax_a.set_facecolor("#FFFFFF")

    # Header & Badges
    ax_a.text(0.00, 1.095, "(a)", transform=ax_a.transAxes,
              fontsize=12, fontweight="bold", color="#1e293b",
              bbox=dict(facecolor="white", edgecolor="#64748b", boxstyle="round,pad=0.2", lw=0.8, alpha=0.95),
              zorder=10)
    ax_a.text(0.065, 1.095, "ECW accessibility tier distribution by street class (n = 557)",
              transform=ax_a.transAxes, fontsize=11.5, fontweight="bold", color="#1e293b", va="center", zorder=10)

    # Crosstabulation of tiers by class
    t = pd.crosstab(n["class"], n.tier, normalize="index").mul(100).reindex(CLS_ORDER).dropna(how="all")
    nn = n.groupby("class", observed=True).size().reindex(t.index)

    y_pos = np.arange(len(CLS_ORDER))
    bar_h = 0.60
    left_acc = np.zeros(len(t))

    tier_keys = [
        ("Impassable (<0.9 m)", TIER_COLORS["Impassable (<0.9 m)"]),
        ("Restricted (0.9-1.5 m)", TIER_COLORS["Restricted (0.9-1.5 m)"]),
        ("Accessible (>=1.5 m)", TIER_COLORS["Accessible (>=1.5 m)"])
    ]

    for t_name, t_col in tier_keys:
        if t_name in t:
            vals = t[t_name].values
            bars = ax_a.barh(y_pos, vals, left=left_acc, height=bar_h, color=t_col, edgecolor="#FFFFFF", lw=1.5, label=t_name)

            # Annotate segments
            for idx, val in enumerate(vals):
                if val >= 7.5:
                    mid_x = left_acc[idx] + val / 2.0
                    ax_a.text(mid_x, y_pos[idx], f"{val:.1f}%", ha="center", va="center",
                              fontsize=8.5, fontweight="bold", color="#FFFFFF", zorder=5)
            left_acc += vals

    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels([f"{c}\n(n = {nn[c]})" for c in CLS_ORDER], fontsize=8.8, fontweight="bold", color="#334155")
    ax_a.invert_yaxis()
    ax_a.set_xlim(0, 100)
    ax_a.set_xlabel("Share of nodes within street class (%)", fontsize=8.8, fontweight="bold", color="#334155", labelpad=6)
    ax_a.tick_params(axis="x", labelsize=8.0, colors="#64748b")
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.spines["left"].set_color("#cbd5e1")
    ax_a.spines["bottom"].set_color("#cbd5e1")
    ax_a.grid(axis="x", color="#f1f5f9", linestyle="--", linewidth=0.8, zorder=0)

    # Inset Legend for Panel (a) positioned cleanly above top-right spine
    leg_a = ax_a.legend(loc="lower right", bbox_to_anchor=(1.0, 1.01), ncol=3,
                        frameon=True, facecolor="#FFFFFF", edgecolor="#cbd5e1",
                        fontsize=7.6, handlelength=1.3, handleheight=0.8)
    leg_a.get_frame().set_boxstyle("round,pad=0.25,rounding_size=0.03")

    # -------------------------------------------------------------------------
    # PANEL (b): BEST-SIDE ECW EMPIRICAL DISPERSION
    # -------------------------------------------------------------------------
    ax_b = fig.add_axes([panel2_x, row1_y, panel_w, row1_h])
    ax_b.set_facecolor("#FFFFFF")

    # Header & Badges
    ax_b.text(0.00, 1.105, "(b)", transform=ax_b.transAxes,
              fontsize=12, fontweight="bold", color="#1e293b",
              bbox=dict(facecolor="white", edgecolor="#64748b", boxstyle="round,pad=0.2", lw=0.8, alpha=0.95),
              zorder=10)
    ax_b.text(0.065, 1.110, "Effective Clear Width (ECW) empirical dispersion across street classes",
              transform=ax_b.transAxes, fontsize=11.5, fontweight="bold", color="#1e293b", va="center", zorder=10)
    ax_b.text(0.065, 1.065, "Best-side bottleneck clearance; values > 8.0 m clipped to 8.0 m for cross-class comparison",
              transform=ax_b.transAxes, fontsize=8.2, fontweight="normal", color="#64748b", va="center", zorder=10)

    # Background Tier Zones
    ax_b.axvspan(0.0, 0.9, color="#fef2f2", alpha=0.75, zorder=0)
    ax_b.axvspan(0.9, 1.5, color="#fffbeb", alpha=0.75, zorder=0)
    ax_b.axvspan(1.5, 8.5, color="#f0f9ff", alpha=0.65, zorder=0)

    # Threshold dashed lines
    ax_b.axvline(0.9, color="#dc2626", ls="--", lw=1.2, zorder=2)
    ax_b.axvline(1.5, color="#d97706", ls="--", lw=1.2, zorder=2)

    # Zone Labels (compact title-case to fit precisely within narrow regulatory bands)
    ax_b.text(0.45, -0.60, "Impassable\n<0.9 m",
              color="#b91c1c", fontsize=6.2, fontweight="bold", ha="center", va="center", zorder=10)
    ax_b.text(1.20, -0.60, "Restricted\n0.9–1.5 m",
              color="#b45309", fontsize=5.8, fontweight="bold", ha="center", va="center", zorder=10)
    ax_b.text(5.00, -0.60, "Accessible (≥1.5 m)",
              color="#1d4ed8", fontsize=7.0, fontweight="bold", ha="center", va="center", zorder=10)

    rng = np.random.default_rng(42)

    for idx, c in enumerate(CLS_ORDER):
        g = n[n["class"] == c]
        ecw_clipped = g.ecw_best.clip(upper=8.0).values
        med_val = np.median(g.ecw_best)
        q25_val = np.percentile(g.ecw_best, 25)
        q75_val = np.percentile(g.ecw_best, 75)

        # 1. Jittered strip points
        jitter_y = idx + 0.12 + rng.uniform(-0.10, 0.10, len(g))
        ax_b.scatter(ecw_clipped, jitter_y, s=20, alpha=0.55,
                     color=CLS_COL[c], edgecolors="#FFFFFF", linewidths=0.35, zorder=4)

        # Highlight clipped values (>8.0 m) with distinct slash
        n_clipped = (g.ecw_best > 8.0).sum()
        if n_clipped > 0:
            ax_b.scatter([8.0] * n_clipped, jitter_y[g.ecw_best > 8.0], s=24,
                         marker="|", color="#1e293b", linewidths=1.2, zorder=5)

        # 2. Horizontal Boxplot / IQR bar above points
        box_y = idx - 0.14
        box_h = 0.20
        iqr_rect = Rectangle((q25_val, box_y - box_h / 2), max(0.05, min(8.0, q75_val) - q25_val), box_h,
                             facecolor=CLS_COL[c], alpha=0.25, edgecolor=CLS_COL[c], lw=1.2, zorder=3)
        ax_b.add_patch(iqr_rect)

        # Draw median vertical line
        ax_b.plot([min(8.0, med_val), min(8.0, med_val)], [box_y - box_h / 2 - 0.04, box_y + box_h / 2 + 0.04],
                  color="#0f172a", lw=2.4, zorder=6)

        # Median text readout
        med_label = f"Med: {med_val:.1f} m"
        ax_b.text(min(7.4, med_val + 0.15), box_y, med_label,
                  va="center", ha="left", fontsize=7.8, fontweight="bold", color="#0f172a", zorder=7)

    ax_b.set_ylim(-0.95, 4.65)
    ax_b.yaxis.tick_right()
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(CLS_ORDER, fontsize=8.8, fontweight="bold", color="#334155")
    ax_b.tick_params(axis="y", right=False, pad=8)
    ax_b.invert_yaxis()
    ax_b.set_xlim(-0.1, 8.5)
    ax_b.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8])
    ax_b.set_xticklabels(["0", "1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "≥8.0 m"], fontsize=8.0, color="#64748b")
    ax_b.set_xlabel("Best-side Effective Clear Width (m)", fontsize=8.8, fontweight="bold", color="#334155", labelpad=6)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.spines["left"].set_visible(False)
    ax_b.spines["bottom"].set_color("#cbd5e1")
    ax_b.grid(axis="x", color="#f1f5f9", linestyle="--", linewidth=0.8, zorder=0)

    # -------------------------------------------------------------------------
    # PANEL (c): OBSTACLE COMPOSITION BY STREET CLASS
    # -------------------------------------------------------------------------
    ax_c = fig.add_axes([panel1_x, row2_y, panel_w, row2_h])
    ax_c.set_facecolor("#FFFFFF")

    # Header & Badges
    ax_c.text(0.00, 1.070, "(c)", transform=ax_c.transAxes,
              fontsize=11.5, fontweight="bold", color="#1e293b",
              bbox=dict(facecolor="white", edgecolor="#64748b", boxstyle="round,pad=0.2", lw=0.8, alpha=0.95),
              zorder=10)
    ax_c.text(0.080, 1.075, "Obstacle composition and ground area footprint by street class",
              transform=ax_c.transAxes, fontsize=11.2, fontweight="bold", color="#1e293b", va="center", zorder=10)
    ax_c.text(0.080, 1.035, "Mean ground area obstructed per 20 m node window (m²) partitioned by object typology",
              transform=ax_c.transAxes, fontsize=8.0, fontweight="normal", color="#64748b", va="center", zorder=10)

    cs = n.groupby("class", observed=True).agg(
        Vehicles=("veh_area", "mean"),
        F=("furn_area", "mean"),
        I=("infra_area", "mean")
    ).reindex(CLS_ORDER).dropna(how="all")
    cs = cs.rename(columns={"F": "Commercial furniture and plants", "I": "Fixed infrastructure"})

    x_c = np.arange(len(cs))
    bar_w = 0.54
    bot_acc = np.zeros(len(cs))

    for o_name, o_col in OBS_COLORS.items():
        vals = cs[o_name].values
        ax_c.bar(x_c, vals, bottom=bot_acc, width=bar_w, color=o_col, edgecolor="#FFFFFF", lw=1.5, label=o_name)

        # Annotate prominent segments
        for idx, v in enumerate(vals):
            if v >= 2.5:
                mid_y = bot_acc[idx] + v / 2.0
                pct = (v / cs.sum(axis=1).iloc[idx]) * 100
                ax_c.text(x_c[idx], mid_y, f"{v:.1f} m²\n({pct:.0f}%)", ha="center", va="center",
                          fontsize=7.4, fontweight="bold", color="#FFFFFF", zorder=5)
        bot_acc += vals

    # Add total blocked area pill on top of each bar
    for idx, total_v in enumerate(bot_acc):
        ax_c.text(x_c[idx], total_v + 0.8, f"{total_v:.1f} m²", ha="center", va="bottom",
                  fontsize=8.2, fontweight="bold", color="#1e293b", zorder=5)

    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels([c.replace(" ", "\n", 1) for c in cs.index], fontsize=8.2, fontweight="bold", color="#334155")
    ax_c.set_ylim(0, 34)
    ax_c.set_ylabel("Mean blocked ground area per node (m²)", fontsize=8.8, fontweight="bold", color="#334155", labelpad=6)
    ax_c.tick_params(axis="y", labelsize=8.0, colors="#64748b")
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.spines["left"].set_color("#cbd5e1")
    ax_c.spines["bottom"].set_color("#cbd5e1")
    ax_c.grid(axis="y", color="#f1f5f9", linestyle="--", linewidth=0.8, zorder=0)

    # Inset Legend for Panel (c) moved to upper-right corner where y is empty!
    leg_c = ax_c.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98),
                        frameon=True, facecolor="#FFFFFF", edgecolor="#cbd5e1",
                        fontsize=7.8, handlelength=1.4, handleheight=0.9)
    leg_c.get_frame().set_boxstyle("round,pad=0.3,rounding_size=0.03")

    # -------------------------------------------------------------------------
    # PANEL (d): PHANTOM CAPACITY VS BLOCKED AREA PHASE SPACE
    # -------------------------------------------------------------------------
    ax_d = fig.add_axes([panel2_x, row2_y, panel_w, row2_h])
    ax_d.set_facecolor("#FFFFFF")

    # Header & Badges
    ax_d.text(0.00, 1.070, "(d)", transform=ax_d.transAxes,
              fontsize=11.5, fontweight="bold", color="#1e293b",
              bbox=dict(facecolor="white", edgecolor="#64748b", boxstyle="round,pad=0.2", lw=0.8, alpha=0.95),
              zorder=10)
    ax_d.text(0.080, 1.075, "Phase space: Phantom capacity against blocked ground area",
              transform=ax_d.transAxes, fontsize=11.2, fontweight="bold", color="#1e293b", va="center", zorder=10)
    ax_d.text(0.080, 1.035, "Symmetric logarithmic x-axis highlighting structural vs. clutter-induced pedestrian chokepoints",
              transform=ax_d.transAxes, fontsize=8.0, fontweight="normal", color="#64748b", va="center", zorder=10)

    # Compute blocked area & PC
    blocked = n.veh_area + n.furn_area + n.infra_area
    pc100 = n.pc * 100

    # Scatter points colored by class
    for c in CLS_ORDER:
        mask = n["class"] == c
        ax_d.scatter(blocked[mask], pc100[mask],
                     s=24, alpha=0.65, color=CLS_COL[c],
                     edgecolors="#FFFFFF", linewidths=0.35, label=c, zorder=3)

    # Symlog x-axis
    ax_d.set_xscale("symlog", linthresh=1.0)
    ax_d.set_xlim(-0.1, 110)
    ax_d.set_ylim(-3, 105)

    ax_d.set_xticks([0, 1, 3, 10, 30, 100])
    ax_d.set_xticklabels(["0", "1", "3", "10", "30", "100 m²"], fontsize=8.0, color="#64748b")
    ax_d.set_xlabel("Blocked ground area in pedestrian zone (m², symlog)", fontsize=8.8, fontweight="bold", color="#334155", labelpad=6)
    ax_d.set_ylabel("Phantom capacity (%)", fontsize=8.8, fontweight="bold", color="#334155", labelpad=6)
    ax_d.tick_params(axis="both", labelsize=8.0, colors="#64748b")

    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)
    ax_d.spines["left"].set_color("#cbd5e1")
    ax_d.spines["bottom"].set_color("#cbd5e1")
    ax_d.grid(True, color="#f1f5f9", linestyle="--", linewidth=0.8, zorder=0)

    # Binned trend ribbon
    bins = [0, 0.5, 2.0, 6.0, 15.0, 40.0, 120.0]
    bin_centers = []
    med_pcs = []
    q25_pcs = []
    q75_pcs = []
    for bi in range(len(bins) - 1):
        m_b = (blocked >= bins[bi]) & (blocked < bins[bi+1])
        if m_b.sum() > 5:
            bin_centers.append(np.median(blocked[m_b]))
            med_pcs.append(np.median(pc100[m_b]))
            q25_pcs.append(np.percentile(pc100[m_b], 25))
            q75_pcs.append(np.percentile(pc100[m_b], 75))

    ax_d.fill_between(bin_centers, q25_pcs, q75_pcs, color="#475569", alpha=0.12, zorder=4)
    ax_d.plot(bin_centers, med_pcs, color="#0f172a", lw=2.2, ls="-", marker="o", markersize=4.5,
              markerfacecolor="#0f172a", markeredgecolor="#FFFFFF", markeredgewidth=1.0,
              label="Binned median", zorder=5)

    # Explanatory Inset Callout Annotations with Arrows (Zero Occlusion, unboxed)
    ax_d.annotate(
        "STRUCTURAL CHOKEPOINTS\n• 100% capacity loss with <1 m² clutter\n• Severe narrow-street vulnerability",
        xy=(0.5, 96), xytext=(0.15, 68),
        arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.2),
        fontsize=6.8, fontweight="bold", color="#991b1b",
        zorder=10
    )

    ax_d.annotate(
        "BUFFER RESILIENCE\n• Low PC (<25%) despite >20 m² clutter\n• Boulevards absorb massive curb parking",
        xy=(35, 3), xytext=(12, 22),
        arrowprops=dict(arrowstyle="->", color="#2563eb", lw=1.2),
        fontsize=6.8, fontweight="bold", color="#1e40af",
        zorder=10
    )

    # Legend for Panel (d)
    leg_d = ax_d.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98),
                        frameon=True, facecolor="#FFFFFF", edgecolor="#cbd5e1",
                        fontsize=7.4, handlelength=1.0, handleheight=0.8)
    leg_d.get_frame().set_boxstyle("round,pad=0.2,rounding_size=0.03")

    # (Bottom analytical cards removed as requested)

    # -------------------------------------------------------------------------
    # SAVE & SYNCHRONIZE 300 DPI PUBLICATION FIGURE (SUPPLEMENTARY S2)
    # -------------------------------------------------------------------------
    out_s2 = OUT / "figS2_class_breakdown.png"
    fig.savefig(out_s2, dpi=300, facecolor="white")
    plt.close(fig)

    # Zero-margin crop to strictly eliminate outer whitespace on all 4 sides
    from PIL import Image
    im = Image.open(out_s2).convert("RGB")
    arr = np.array(im)
    non_white = np.any(arr < 250, axis=2)
    coords = np.argwhere(non_white)
    if coords.size > 0:
        min_y, min_x = coords.min(axis=0)
        max_y, max_x = coords.max(axis=0)
        cropped = im.crop((min_x, min_y, max_x + 1, max_y + 1))
        cropped.save(out_s2)
        print(f"Zero-margin cropped {out_s2.name}: {im.size} -> {cropped.size} (aspect ratio: {cropped.size[0]/cropped.size[1]:.4f})")

    print(f"Saved primary publication figure: {out_s2}")

    # Synchronize across manuscript and submission package directories
    import shutil
    paths_sync = [
        REV / "05_manuscript_R1/markedup/figures/figS2_class_breakdown.png",
        REV / "SUBMISSION_R1_PACKAGE/02_latex_source/figures/figS2_class_breakdown.png",
        REV / "07_repository_gitlab_figshare/figures/figS2_class_breakdown.png",
    ]
    for p in paths_sync:
        if p.parent.exists():
            shutil.copy2(out_s2, p)
            print(f"Synchronized figure to: {p}")

    print("figS2_class_breakdown generated successfully.")

if __name__ == "__main__":
    build_figS2()
