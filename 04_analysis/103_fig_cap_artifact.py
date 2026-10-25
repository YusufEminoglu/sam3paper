"""Evidence for the reviewer's remark that a fixed clutter cap can mechanically produce a relation between street width and ECW.
(a) first version: ECW against cadastral width; the 4.00 m cap per side puts 25 nodes exactly on ECW = W - 8 m, (b) distribution of the per-side clutter depth of the first version (spike at the cap),
(c) revised method: ECW against the width of the pedestrian-intended zone of the better side: ECW can equal the zone width only where nothing obstructs it, and is otherwise below."""
from matplotlib.lines import Line2D
from sigfig_common import *

r0 = pd.read_csv(REV / "legacy_v1/sam3_results.csv")
w, e, cl, cr = r0.Width_Cad_m, r0.ECW_SAM3_m, r0.Clutter_Left_m, r0.Clutter_Right_m
ncap = (cl >= 3.999).astype(int) + (cr >= 3.999).astype(int)
CAPC = {0: "#4575b4", 1: "#d49b7b", 2: "#c0392b"}

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 6.4), facecolor=BG, gridspec_kw=dict(width_ratios=[1.0, 0.8, 1.0], wspace=0.24))
ax1.set_facecolor(BG_AX)
xx = np.linspace(4, 22, 50); ax1.plot(xx, np.maximum(xx - 8, 0), color="#c0392b", lw=1.4, ls="--", label="ECW = W − 8 m (both sides at the 4.00 m cap)", zorder=1)
rng = np.random.default_rng(3)
for k in (0, 1, 2):
    m = ncap == k; ax1.scatter(w[m] + rng.uniform(-0.15, 0.15, m.sum()), e[m], s=46, c=CAPC[k], edgecolors="white", lw=0.6, alpha=0.9, zorder=3, label=f"{k} side(s) at the cap ({m.sum()} nodes)")
ax1.set_xlabel("Cadastral width W (m)", fontsize=10); ax1.set_ylabel("ECW of the first version (m)", fontsize=10); ax1.grid(True, lw=0.3)
ax1.legend(fontsize=8.5, loc="upper left", framealpha=0.92, edgecolor="#ccc")
ax1.text(0.98, 0.04, f"Pearson r = {np.corrcoef(w, e)[0, 1]:.2f}\n{int((ncap >= 1).sum())} of {len(r0)} nodes ({100 * (ncap >= 1).mean():.0f}%) have a side at the cap;\n{int((ncap == 2).sum())} nodes lie exactly on the line", transform=ax1.transAxes, ha="right", va="bottom", fontsize=8.8, style="italic",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.92))
ax1.set_title("(a) First version: ECW against cadastral width", fontsize=11.5, fontweight="bold")

ax2.set_facecolor(BG_AX); vals = np.r_[cl.values, cr.values]
ax2.hist(vals, bins=np.arange(0, 4.21, 0.2), color="#cfb57e", edgecolor="white"); ax2.axvline(4.0, color="#c0392b", lw=1.2, ls="--")
ax2.set_xlabel("Clutter depth per side (m)", fontsize=10); ax2.set_ylabel("Number of node sides", fontsize=10)
ax2.text(0.97, 0.95, f"{100 * (vals >= 3.999).mean():.0f}% of the {len(vals)} side values\nequal the 4.00 m cap", transform=ax2.transAxes, ha="right", va="top", fontsize=9.5, fontweight="bold", color="#c0392b")
ax2.set_title("(b) First version: clutter depth", fontsize=11.5, fontweight="bold")

n = load(); ev = pd.read_csv(REV / "02_pipeline_v2/ecw_v2.csv"); ev = ev[(ev.coverage >= 0.5) & ev.ecw_min.notna()]
best = ev.sort_values("ecw_min").groupby("node").tail(1).set_index("node")
d = n.set_index("node").join(best[["zone_w"]].rename(columns={"zone_w": "zbest"}), how="inner").dropna(subset=["zbest"])
ax3.set_facecolor(BG_AX); mx = 12.0
sc = ax3.scatter(d.zbest.clip(upper=mx), d.ecw_best.clip(upper=mx), s=16, c=np.log1p(d.obs_total), cmap="YlOrRd", edgecolors="none", alpha=0.85, zorder=3)
ax3.plot([0, mx], [0, mx], color="#333", lw=1.2, ls="--", zorder=2)
ax3.set_xlim(0, mx); ax3.set_ylim(0, mx); ax3.set_aspect("equal"); ax3.set_xlabel("Width of the pedestrian-intended zone, better side (m; capped at 12 m)", fontsize=10); ax3.set_ylabel("ECW of the revised method (m)", fontsize=10)
cb = fig.colorbar(sc, ax=ax3, fraction=0.045, pad=0.02); cb.set_label("log(1 + obstacle footprint, m²)", fontsize=8.5); cb.ax.tick_params(labelsize=7.5)
free = float(((d.zbest - d.ecw_best) < 0.10).mean()); below = float(((d.zbest - d.ecw_best) >= 0.5).mean())
ax3.text(0.03, 0.96, f"On the line (ECW within 0.1 m of the zone): {100 * free:.0f}% of nodes,\nthe zone is unobstructed there.\n≥ 0.5 m below the line: {100 * below:.0f}% of nodes", transform=ax3.transAxes, ha="left", va="top", fontsize=8.6, style="italic",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.92))
ax3.set_title("(c) Revised method: ECW against zone width", fontsize=11.5, fontweight="bold")
save(fig, "fig_sig_cap_artifact")
stat("capEitherPct", round(100 * (ncap >= 1).mean(), 0)); stat("capBothN", int((ncap == 2).sum())); stat("capSidePct", round(100 * (vals >= 3.999).mean(), 0)); stat("capOldCorr", round(float(np.corrcoef(w, e)[0, 1]), 2))
stat("capNewFreePct", round(100 * free, 0)); stat("capNewBelowPct", round(100 * below, 0)); stat("capNewCorr", round(float(n.ecw_best.corr(n.w_corridor)), 2))
print("free", round(free, 3), "below", round(below, 3), "r1 corr", round(float(n.ecw_best.corr(n.w_corridor)), 2))
