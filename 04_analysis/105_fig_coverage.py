"""Observation coverage and street width.
(a) observed share of the 20 m window of every node-side against corridor width, by imagery source, (b) share of node-sides that meet the 50% criterion by corridor-width class and source,
(c) effect of the criterion (50, 40, 30%) on the sample and on the impassable share.
Finding: coverage is not lower on narrow streets in the Konak sample (views along the street); the legacy pilot (views across the street, pitch 0 deg) has low coverage at all widths."""
from sigfig_common import *

ev = pd.read_csv(REV / "02_pipeline_v2/ecw_v2.csv")
c = pd.read_csv(REV / "02_pipeline_v2/corridor_nodes.csv"); c["node"] = np.where(c.src == "SCS", "scs_" + c.id.astype(str).str.replace(r"\.0$", "", regex=True), c.id)
d = ev.merge(c[["node", "w_corridor"]], on="node", how="left").dropna(subset=["w_corridor", "coverage"])
d["ax"] = np.where(d.src == "TG", d.node.str.split("_").str[0], ""); d["met"] = d.coverage >= 0.5
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 6.2), facecolor=BG, gridspec_kw=dict(width_ratios=[1.25, 0.95, 0.85], wspace=0.24))
ax1.set_facecolor(BG_AX); rng = np.random.default_rng(1)
for met, col, lab in ((False, "#c77d7d", "below 50%"), (True, "#4575b4", "at least 50%")):
    q = d[(d.met == met) & (d.src == "SCS")]; ax1.scatter(q.w_corridor, q.coverage + rng.uniform(-0.005, 0.005, len(q)), s=10, c=col, alpha=0.5, edgecolors="none", label=f"Konak sample, {lab} ({len(q)})")
tg = d[d.src == "TG"]; ax1.scatter(tg.w_corridor, tg.coverage, s=42, marker="D", c=[("#c0392b" if a in ("Ax3", "Ax6") else "#777") for a in tg.ax], edgecolors="white", lw=0.6, label=f"legacy pilot ({len(tg)}); red = Ax3 and Ax6", zorder=4)
ax1.axhline(0.5, color="#333", lw=1, ls="--"); ax1.set_xscale("log"); ax1.set_xticks([4, 6, 8, 10, 15, 20, 30, 45]); ax1.set_xticklabels(["4", "6", "8", "10", "15", "20", "30", "45"])
ax1.set_xlabel("Corridor width W$_c$ (m, log scale)", fontsize=10); ax1.set_ylabel("Observed share of the 20 m window", fontsize=10); ax1.legend(fontsize=8.2, loc="lower right", framealpha=0.92, edgecolor="#ccc")
ax1.set_title("(a) Observation coverage of every node-side", fontsize=11.5, fontweight="bold")

bins = [0, 6, 8, 10, 15, 25, 100]; lab = ["< 6", "6–8", "8–10", "10–15", "15–25", "≥ 25"]; d["wb"] = pd.cut(d.w_corridor, bins, labels=lab, right=False)
ax2.set_facecolor(BG); wd = 0.38
for k, (src, col, nm) in enumerate((("SCS", "#4575b4", "Konak sample (views along the street)"), ("TG", "#777", "legacy pilot (views across the street)"))):
    sh = d[d.src == src].groupby("wb", observed=True).met.agg(["mean", "size"]).reindex(lab).fillna(0)
    xs = np.arange(len(lab)) + (k - 0.5) * wd; ax2.bar(xs, 100 * sh["mean"], width=wd, color=col, label=nm)
    for x, m_, s_ in zip(xs, sh["mean"], sh["size"]):
        if s_ > 0: ax2.text(x, 100 * m_ + 1, f"{100 * m_:.0f}", ha="center", fontsize=7.8); ax2.text(x, 1.5, f"n={int(s_)}", ha="center", va="bottom", fontsize=6.6, color="white", rotation=90)
ax2.set_xticks(range(len(lab))); ax2.set_xticklabels(lab); ax2.set_ylim(0, 115); ax2.set_xlabel("Corridor width W$_c$ (m)", fontsize=10); ax2.set_ylabel("Node-sides meeting the 50% criterion (%)", fontsize=10)
ax2.legend(fontsize=8.2, loc="upper center", ncol=1, framealpha=0.92, edgecolor="#ccc"); ax2.set_title("(b) Share of node-sides analysed", fontsize=11.5, fontweight="bold")

def vals(f):
    o = {}
    for l in open(f, encoding="utf-8"):
        p = l.strip().split("{")
        if len(p) >= 3: o[p[1].rstrip("}").lstrip("\\")] = p[2].rstrip("}")
    return o
rows = []
for cth, dd in ((50, "out_final"), (40, "out_cov0.4"), (30, "out_cov0.3")):
    nn = pd.read_csv(HERE / dd / "nodes_R1.csv"); v = vals(HERE / dd / "values.tex"); rows.append((cth, len(nn), float(v["shareImpassable"]), int(nn[nn.src == "TG"].node.str.split("_").str[0].isin(["Ax3", "Ax6"]).sum())))
ax3.set_facecolor(BG); x = np.arange(3); ax3.bar(x, [r[2] for r in rows], color=TIER_COL["Impassable"], width=0.55)
for i, r in enumerate(rows): ax3.text(i, r[2] + 0.5, f"{r[2]:.1f}%\nn={r[1]}\nAx3+Ax6: {r[3]}", ha="center", fontsize=8.6)
ax3.set_xticks(x); ax3.set_xticklabels([f"≥ {r[0]}%" for r in rows]); ax3.set_ylim(0, 31); ax3.set_xlabel("Minimum observed share of the window", fontsize=10); ax3.set_ylabel("Impassable nodes (%)", fontsize=10)
ax3.set_title("(c) Effect of the criterion", fontsize=11.5, fontweight="bold")
save(fig, "figS5_coverage")
s = d[d.src == "SCS"]; t = d[d.src == "TG"]
stat("covSCSPct", round(100 * s.met.mean(), 0)); stat("covSCSNarrowPct", round(100 * s[s.w_corridor < 6].met.mean(), 0)); stat("covSCSLowestPct", round(100 * s.groupby("wb", observed=True).met.mean().min(), 0))
stat("covTGPct", round(100 * t.met.mean(), 0)); stat("covMedSCSnarrow", round(float(s[s.w_corridor < 8].coverage.median()), 2)); stat("covMedTGnarrow", round(float(t[t.w_corridor < 8].coverage.median()), 2))
print("done")
