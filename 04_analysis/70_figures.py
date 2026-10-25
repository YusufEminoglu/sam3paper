"""Figures for the revised manuscript (same visual language as the first version: warm ground, muted palette, lettered panels).

usage: python 70_figures.py <analysis_dir> <pipeline_dir> <validation_dir> <fig_out_dir>
Reads: nodes_R1.csv, streets_R1.csv, class_summary.csv, tier_by_class.csv (analysis); calib.csv, anchor_cons.csv, corridor_nodes.csv,
       konak_layers.gpkg (pipeline); sensitivity.csv (analysis); model_agreement_by_class.csv, pipeline_agreement.csv, audit_summary.csv
       (validation).  Every figure is skipped with a message if its input is missing.
"""
import sys, warnings, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

warnings.filterwarnings("ignore")
A, P, V, O = (Path(x) for x in sys.argv[1:5]); O.mkdir(parents=True, exist_ok=True)
BG, INK = "white", "#3a3a3a"
TIER = {"Impassable (<0.9 m)": "#b5525c", "Restricted (0.9-1.5 m)": "#e0a458", "Accessible (>=1.5 m)": "#6a9bb5"}
OBS = {"Vehicles": "#c9767a", "Commercial furniture and plants": "#d4b27a", "Fixed infrastructure": "#7a8fb5"}
CLS_ORDER = ["Shared lane", "Narrow local street", "Wide local street", "Collector avenue", "Boulevard"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#999", "axes.facecolor": BG, "figure.facecolor": BG,
                     "axes.spines.top": False, "axes.spines.right": False, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK})


def letter(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=17, style="italic", color="#888", fontweight="bold", va="bottom", ha="right")


def save(fig, name):
    fig.savefig(O / name, dpi=300, facecolor=BG, bbox_inches="tight", pad_inches=0.02); plt.close(fig); print("saved", name)


def have(*ps):
    return all(Path(p).exists() for p in ps)


nodes = pd.read_csv(A / "nodes_R1.csv") if have(A / "nodes_R1.csv") else None
if nodes is not None:
    nodes["class"] = pd.Categorical(nodes["class"], CLS_ORDER, ordered=True)

# ---- Fig: tiers by street class + ECW distribution ------------------------------------------------------------------
if nodes is not None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1, 1.15]))
    t = pd.crosstab(nodes["class"], nodes.tier, normalize="index").mul(100).reindex(CLS_ORDER).dropna(how="all")
    n = nodes.groupby("class").size().reindex(t.index)
    left = np.zeros(len(t))
    for k, c in TIER.items():
        if k in t:
            ax[0].barh(range(len(t)), t[k], left=left, color=c, edgecolor=BG, label=k); left += t[k].values
    ax[0].set_yticks(range(len(t))); ax[0].set_yticklabels([f"{i} (n={n[i]})" for i in t.index]); ax[0].invert_yaxis()
    ax[0].set_xlabel("Share of nodes (%)"); ax[0].legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1)
    letter(ax[0], "(a)")
    rng = np.random.default_rng(3)
    for i, c in enumerate(t.index):
        g = nodes[nodes["class"] == c]
        ax[1].scatter(g.ecw_best.clip(upper=8), i + rng.uniform(-.25, .25, len(g)), s=10, alpha=.5, color="#556", lw=0)
        ax[1].plot([g.ecw_best.median()] * 2, [i - .35, i + .35], color="#b5525c", lw=2)
    ax[1].axvline(0.9, color="#999", ls="--", lw=.8); ax[1].axvline(1.5, color="#999", ls="--", lw=.8)
    ax[1].set_yticks(range(len(t))); ax[1].set_yticklabels(t.index); ax[1].invert_yaxis(); ax[1].set_xlabel("Best-side ECW (m; values above 8 m plotted at 8 m)")
    letter(ax[1], "(b)")
    save(fig, "figS2_tiers_by_class.png")

# ---- Fig: obstacle composition ----------------------------------------------------------------------------------------
if nodes is not None:
    cs = nodes.groupby("class").agg(Vehicles=("veh_area", "mean"), F=("furn_area", "mean"), I=("infra_area", "mean")).reindex(CLS_ORDER).dropna(how="all")
    cs = cs.rename(columns={"F": "Commercial furniture and plants", "I": "Fixed infrastructure"})
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw=dict(width_ratios=[1.2, 1]))
    b = np.zeros(len(cs))
    for k, c in OBS.items():
        ax[0].bar(range(len(cs)), cs[k], bottom=b, color=c, edgecolor=BG, label=k); b += cs[k].values
    ax[0].set_xticks(range(len(cs))); ax[0].set_xticklabels([c.replace(" ", "\n", 1) for c in cs.index], fontsize=8)
    ax[0].set_ylabel("Mean blocked ground area per node (m$^2$)"); ax[0].legend(frameon=False, fontsize=7.5); letter(ax[0], "(a)")
    ax[1].scatter(nodes.veh_area + nodes.furn_area + nodes.infra_area + 0.01, nodes.pc * 100, s=10, alpha=.45, color="#556", lw=0)
    ax[1].set_xscale("symlog", linthresh=1); ax[1].set_xlabel("Blocked ground area in the pedestrian zone (m$^2$, symlog)"); ax[1].set_ylabel("Phantom capacity (%)"); letter(ax[1], "(b)")
    save(fig, "figS3_obstacle_composition.png")

# ---- Fig: geometry validation ----------------------------------------------------------------------------------------
if have(P / "calib.csv", P / "anchor_cons.csv"):
    c = pd.read_csv(P / "calib.csv"); a = pd.read_csv(P / "anchor_cons.csv"); a = a[a.h == 2.5]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    for src, col in (("TG", "#c9767a"), ("SCS", "#6a9bb5")):
        g = c[c.src == src]
        ax[0].scatter(g.w_corr, g.w_pred_loso, s=14, alpha=.7, color=col, lw=0, label=f"{'Legacy axes' if src=='TG' else 'Konak sample'} (n={len(g)})")
    mx = max(c.w_corr.max(), c.w_pred_loso.max()); ax[0].plot([0, mx], [0, mx], color="#999", lw=.8)
    ax[0].set_xlabel("GIS building-to-building width (m)"); ax[0].set_ylabel("Width reconstructed from imagery (m)"); ax[0].legend(frameon=False, fontsize=7.5); letter(ax[0], "(a)")
    ax[1].hist(c.h_node.clip(1, 4), bins=20, color="#8fac8b", edgecolor=BG); ax[1].set_xlabel("Node-level effective camera height (m)"); ax[1].set_ylabel("Nodes"); letter(ax[1], "(b)")
    ax[2].hist(a.e.clip(-8, 8), bins=24, color="#d4b27a", edgecolor=BG); ax[2].axvline(0, color="#999", lw=.8)
    ax[2].set_xlabel("Left minus right camera-offset estimate (m)"); ax[2].set_ylabel("Nodes"); letter(ax[2], "(c)")
    save(fig, "figS6_geometry_validation.png")

# ---- Fig: uncertainty ------------------------------------------------------------------------------------------------
if nodes is not None:
    g = nodes.dropna(subset=["ecw_p05", "ecw_p95"]).sort_values("ecw_best")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(g))
    ax.fill_between(x, g.ecw_p05.clip(upper=6), g.ecw_p95.clip(upper=6), color="#b8c8d8", alpha=.7, lw=0, label="Monte Carlo 5th-95th percentile")
    ax.plot(x, g.ecw_best.clip(upper=6), color="#3a3a3a", lw=1, label="Central estimate")
    ax.axhline(0.9, color="#b5525c", ls="--", lw=.8); ax.axhline(1.5, color="#e0a458", ls="--", lw=.8)
    ax.set_ylim(0, 6); ax.set_xlabel("Nodes ranked by best-side ECW"); ax.set_ylabel("ECW (m)"); ax.legend(frameon=False, fontsize=8)
    save(fig, "figS1_uncertainty.png")

# ---- Fig: sensitivity ------------------------------------------------------------------------------------------------
if have(A / "sensitivity.csv"):
    s = pd.read_csv(A / "sensitivity.csv", index_col=0)
    base = s.loc["central", "share_lt09"]
    d = (s.share_lt09 - base).drop("central").sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.barh(range(len(d)), d.values, color=["#6a9bb5" if v < 0 else "#b5525c" for v in d.values]); ax.axvline(0, color="#666", lw=.8)
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d.index, fontsize=8)
    ax.set_xlabel(f"Change in share of impassable nodes (percentage points; central = {base:.1f}%)")
    save(fig, "fig12_sensitivity.png")

# ---- Fig: model comparison -------------------------------------------------------------------------------------------
if have(V / "model_agreement_by_class.csv"):
    m = pd.read_csv(V / "model_agreement_by_class.csv").sort_values("mean_iou_when_both")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw=dict(width_ratios=[1, 1]))
    ax[0].barh(m.cls, m.mean_iou_when_both, color="#7a8fb5"); ax[0].set_xlabel("Mean IoU where both models detect the class"); letter(ax[0], "(a)")
    y = np.arange(len(m)); ax[1].barh(y, m.sam_only, color="#c9767a", label="SAM 3 only"); ax[1].barh(y, -m.sf_only, color="#8fac8b", label="SegFormer only")
    ax[1].set_yticks(y); ax[1].set_yticklabels(m.cls); ax[1].axvline(0, color="#666", lw=.8); ax[1].set_xlabel("Images with detection by one model only"); ax[1].legend(frameon=False, fontsize=8); letter(ax[1], "(b)")
    save(fig, "figS4_model_comparison.png")
print("done")
