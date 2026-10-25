"""Shared data, palette and helpers for the signature figures of the revision (R1).

Design language follows the first version's figures (soft/muted palette, radial, chord, Sankey, spiral, raincloud, t-SNE), but every element
is computed from the R1 output files: nodes_R1.csv (559 nodes, 251 streets) and ecw_v2.csv (per node-side obstacle areas). Nothing is typed in.
"""
import warnings
import os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.patches import Ellipse

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
REV = HERE.parent
OUT = Path(os.environ.get("PC_FIG_DIR", REV / "figures")); OUT.mkdir(exist_ok=True, parents=True)
DPI = 200
FIGSCALE = 1.0       # designed as full-width plates, like the first version; lower it only after re-checking every layout

# ---- palette (from the first version) ---------------------------------------------------------------------------------
OBJ = {"car": "#c77d7d", "truck": "#a16363", "motorcycle": "#d49b7b", "chair": "#cfb57e", "table": "#e0d1b1", "potted_plant": "#91ad91",
       "trash_can": "#b5b5b5", "dumpster": "#8c8c8c", "utility_pole": "#7d94b5", "street_light": "#617491", "utility_box": "#89b1c2",
       "electrical_cabinet": "#a8c6d4"}
OBJ_LABEL = {k: k.replace("_", " ").title() for k in OBJ}
GROUPS = {"Vehicles": ["car", "truck", "motorcycle"], "Furniture": ["chair", "table"], "Plants": ["potted_plant"],
          "Infrastructure": ["trash_can", "dumpster", "utility_pole", "street_light", "utility_box", "electrical_cabinet"]}
GROUP_COL = {"Vehicles": "#c77d7d", "Furniture": "#cfb57e", "Plants": "#91ad91", "Infrastructure": "#7d94b5", "None": "#d8d5cf"}
CLASSES = ["Boulevard", "Collector avenue", "Wide local street", "Narrow local street", "Shared lane"]
CLS_COL = {"Boulevard": "#6b5b95", "Collector avenue": "#4f9d9d", "Wide local street": "#8fae5a", "Narrow local street": "#d08c60", "Shared lane": "#a0729a"}
CLS_MARK = {"Boulevard": "D", "Collector avenue": "o", "Wide local street": "s", "Narrow local street": "^", "Shared lane": "v"}
TIERS = ["Impassable", "Restricted", "Accessible"]
TIER_LABEL = {"Impassable": "Impassable (<0.9 m)", "Restricted": "Restricted (0.9–1.5 m)", "Accessible": "Accessible (≥1.5 m)"}
TIER_COL = {"Impassable": "#c77d7d", "Restricted": "#cfb57e", "Accessible": "#4575b4"}
TXT, SUB, BG, BG_AX = "#333333", "#888888", "white", "white"

plt.rcParams.update({"text.color": TXT, "axes.labelcolor": TXT, "xtick.color": TXT, "ytick.color": TXT, "axes.edgecolor": "#aaa", "axes.linewidth": 0.5,
                     "font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False, "grid.color": "#dcdad5", "grid.linewidth": 0.4})


def save(fig, name, dpi=DPI, ext="png"):
    import shutil
    kw = dict(pil_kwargs={"quality": 88}) if ext == "jpg" else {}
    target_p = OUT / (name + "." + ext)
    fig.savefig(str(target_p), dpi=dpi, bbox_inches="tight", pad_inches=0.02, facecolor=fig.get_facecolor(), **kw)
    plt.close(fig)
    print("saved", name)
    # optional sync into a manuscript tree, if one is present (the research working copy)
    sync_dirs = [
        REV / "05_manuscript_R1" / "figures",
        REV / "05_manuscript_R1" / "markedup" / "figures",
        REV / "SUBMISSION_R1_PACKAGE" / "02_latex_source" / "figures",
    ]
    for sd in sync_dirs:
        if sd == OUT or not sd.parent.exists():
            continue
        sd.mkdir(parents=True, exist_ok=True)
        (sd / (name + "." + ext)).write_bytes(target_p.read_bytes())


def load():
    """Analysed nodes with per-class obstacle areas (m2, both sides, 20 m window), dominant obstacle group and tier labels."""
    n = pd.read_csv(HERE / "out_final/nodes_R1.csv")
    e = pd.read_csv(REV / "02_pipeline_v2/ecw_v2.csv")
    e = e[(e.coverage >= 0.5) & e.ecw_min.notna()]
    acols = [c for c in e.columns if c.startswith("a_") and c != "a_person"]
    a = e.groupby("node")[acols].sum().rename(columns=lambda c: c[2:])
    n = n.merge(a, left_on="node", right_index=True, how="left")
    for k in OBJ:
        if k not in n: n[k] = 0.0
        n[k] = n[k].fillna(0.0)
    for g, ks in GROUPS.items():
        n["g_" + g] = n[ks].sum(axis=1)
    n["obs_total"] = n[[f"g_{g}" for g in GROUPS]].sum(axis=1)
    gcols = [f"g_{g}" for g in GROUPS]
    n["dominant"] = np.where(n.obs_total > 0.05, n[gcols].idxmax(axis=1).str[2:], "None")
    n["tier3"] = pd.cut(n.ecw_best, [-1e-9, 0.9, 1.5, 1e9], right=False, labels=TIERS).astype(str)
    n["ecw_plot"] = n.ecw_best.clip(upper=12.0)
    return n


def chord_ribbon(ax, a1s, a1e, a2s, a2e, r=0.76, color="blue", alpha=0.28):
    """Bezier chord ribbon between two arc segments (first version's construction)."""
    t = np.linspace(0, 1, 80); arc1 = np.linspace(a1s, a1e, 60); arc2 = np.linspace(a2s, a2e, 60)
    def bez(pa, pb):
        return (1 - t) ** 2 * pa[0] + t ** 2 * pb[0], (1 - t) ** 2 * pa[1] + t ** 2 * pb[1]
    p = lambda a: r * np.array([np.cos(a), np.sin(a)])
    b1x, b1y = bez(p(a1e), p(a2s)); b2x, b2y = bez(p(a2e), p(a1s))
    xs = np.r_[r * np.cos(arc1), b1x, r * np.cos(arc2[::-1]), b2x]; ys = np.r_[r * np.sin(arc1), b1y, r * np.sin(arc2[::-1]), b2y]
    ax.fill(xs, ys, color=color, alpha=alpha, linewidth=0, zorder=1)


def conf_ellipse(x, y, ax, n_std=2.0, **kw):
    """Covariance ellipse (n_std standard deviations) for a 2-D scatter."""
    if len(x) < 3: return
    cov = np.cov(x, y); pear = np.clip(cov[0, 1] / np.sqrt(max(cov[0, 0] * cov[1, 1], 1e-30)), -1, 1)
    ell = Ellipse((0, 0), width=np.sqrt(1 + pear) * 2, height=np.sqrt(1 - pear) * 2, **kw)
    tr = mtransforms.Affine2D().rotate_deg(45).scale(np.sqrt(cov[0, 0]) * n_std, np.sqrt(cov[1, 1]) * n_std).translate(np.mean(x), np.mean(y))
    ell.set_transform(tr + ax.transData)
    return ax.add_patch(ell)


def stat(key, value):
    """Record a number computed for a figure (out_final/fig_stats.csv); 81_tables.py turns these into manuscript macros."""
    p = HERE / "out_final/fig_stats.csv"
    d = dict(pd.read_csv(p).values) if p.exists() else {}
    d[key] = value
    pd.DataFrame(sorted(d.items()), columns=["key", "value"]).to_csv(p, index=False)


# shrink every canvas by FIGSCALE while keeping font sizes in points
_sub, _fig = plt.subplots, plt.figure
def _scale(kw):
    if kw.get("figsize"): kw["figsize"] = tuple(v * FIGSCALE for v in kw["figsize"])
    return kw
plt.subplots = lambda *a, **kw: _sub(*a, **_scale(kw))
plt.figure = lambda *a, **kw: _fig(*a, **_scale(kw))
