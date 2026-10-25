"""Pipeline validation, merged into one figure to keep the supplement at 4 figures: segmentation agreement, observation coverage, and
geometric accuracy (previously separate Figures S4, S5 and S6).
(a)-(b) SAM 3 vs SegFormer-B5 on the same images. (c)-(e) Observation coverage and the effect of its 50% criterion. (f)-(h) Corridor-width
reconstruction accuracy, camera height and left/right offset consistency."""
from sigfig_common import *

P = REV / "02_pipeline_v2"; V = REV / "03_validation/out_final"

fig = plt.figure(figsize=(17.0, 13.4), facecolor=BG)
gs = fig.add_gridspec(3, 6, hspace=0.62, wspace=1.05)

# ---- (a)-(b) segmentation agreement, SAM 3 vs SegFormer-B5 -----------------------------------------------------------------
ax_a = fig.add_subplot(gs[0, 0:3]); ax_b = fig.add_subplot(gs[0, 3:6])
m = pd.read_csv(V / "model_agreement_by_class.csv").sort_values("mean_iou_when_both")
ax_a.barh(m.cls, m.mean_iou_when_both, color="#7d94b5"); ax_a.set_xlabel("Mean IoU where both models detect the class")
ax_a.set_title("(a) Segmentation agreement by class", fontsize=11.5, fontweight="bold")
y = np.arange(len(m)); ax_b.barh(y, m.sam_only, color="#c77d7d", label="SAM 3 only"); ax_b.barh(y, -m.sf_only, color="#91ad91", label="SegFormer only")
ax_b.set_yticks(y); ax_b.set_yticklabels(m.cls); ax_b.axvline(0, color="#666", lw=.8); ax_b.set_xlabel("Images with detection by one model only")
ax_b.legend(frameon=False, fontsize=8); ax_b.set_title("(b) Detections unique to one model", fontsize=11.5, fontweight="bold")

# ---- (c)-(e) observation coverage -------------------------------------------------------------------------------------------
ax_c = fig.add_subplot(gs[1, 0:2]); ax_d = fig.add_subplot(gs[1, 2:4]); ax_e = fig.add_subplot(gs[1, 4:6])
ev = pd.read_csv(P / "ecw_v2.csv")
c = pd.read_csv(P / "corridor_nodes.csv"); c["node"] = np.where(c.src == "SCS", "scs_" + c.id.astype(str).str.replace(r"\.0$", "", regex=True), c.id)
d = ev.merge(c[["node", "w_corridor"]], on="node", how="left").dropna(subset=["w_corridor", "coverage"])
d["ax"] = np.where(d.src == "TG", d.node.str.split("_").str[0], ""); d["met"] = d.coverage >= 0.5
rng = np.random.default_rng(1)
for met, col, lab in ((False, "#c77d7d", "below 50%"), (True, "#4575b4", "at least 50%")):
    q = d[(d.met == met) & (d.src == "SCS")]
    ax_c.scatter(q.w_corridor, q.coverage + rng.uniform(-0.005, 0.005, len(q)), s=10, c=col, alpha=0.5, edgecolors="none", label=f"Konak sample, {lab} ({len(q)})")
tg = d[d.src == "TG"]
ax_c.scatter(tg.w_corridor, tg.coverage, s=42, marker="D", c=[("#c0392b" if a in ("Ax3", "Ax6") else "#777") for a in tg.ax], edgecolors="white", lw=0.6, label=f"legacy pilot ({len(tg)}); red = Ax3 and Ax6", zorder=4)
ax_c.axhline(0.5, color="#333", lw=1, ls="--"); ax_c.set_xscale("log"); ax_c.set_xticks([4, 6, 8, 10, 15, 20, 30, 45]); ax_c.set_xticklabels(["4", "6", "8", "10", "15", "20", "30", "45"])
ax_c.set_xlabel("Corridor width $W_c$ (m, log scale)", fontsize=9.5); ax_c.set_ylabel("Observed share of the 20 m window", fontsize=9.5)
ax_c.legend(fontsize=7.0, loc="lower right", framealpha=0.92, edgecolor="#ccc"); ax_c.set_title("(c) Observation coverage of every node-side", fontsize=11.5, fontweight="bold")

bins = [0, 6, 8, 10, 15, 25, 100]; lab = ["< 6", "6-8", "8-10", "10-15", "15-25", ">= 25"]; d["wb"] = pd.cut(d.w_corridor, bins, labels=lab, right=False)
wd = 0.38
for k_, (src, col, nm) in enumerate((("SCS", "#4575b4", "Konak sample (along street)"), ("TG", "#777", "legacy pilot (across street)"))):
    sh = d[d.src == src].groupby("wb", observed=True).met.agg(["mean", "size"]).reindex(lab).fillna(0)
    xs = np.arange(len(lab)) + (k_ - 0.5) * wd; ax_d.bar(xs, 100 * sh["mean"], width=wd, color=col, label=nm)
    for x, m_, s_ in zip(xs, sh["mean"], sh["size"]):
        if s_ > 0: ax_d.text(x, 100 * m_ + 1, f"{100 * m_:.0f}", ha="center", fontsize=7.0); ax_d.text(x, 1.5, f"n={int(s_)}", ha="center", va="bottom", fontsize=6.0, color="white", rotation=90)
ax_d.set_xticks(range(len(lab))); ax_d.set_xticklabels(lab); ax_d.set_ylim(0, 115)
ax_d.set_xlabel("Corridor width $W_c$ (m)", fontsize=9.5); ax_d.set_ylabel("Node-sides meeting the 50% criterion (%)", fontsize=9.5)
ax_d.legend(fontsize=7.0, loc="upper center", ncol=1, framealpha=0.92, edgecolor="#ccc"); ax_d.set_title("(d) Share of node-sides analysed", fontsize=11.5, fontweight="bold")

def vals(f):
    o = {}
    for l in open(f, encoding="utf-8"):
        p = l.strip().split("{")
        if len(p) >= 3: o[p[1].rstrip("}").lstrip("\\")] = p[2].rstrip("}")
    return o
rows = []
for cth, dd in ((50, "out_final"), (40, "out_cov0.4"), (30, "out_cov0.3")):
    nn = pd.read_csv(HERE / dd / "nodes_R1.csv"); v = vals(HERE / dd / "values.tex")
    rows.append((cth, len(nn), float(v["shareImpassable"]), int(nn[nn.src == "TG"].node.str.split("_").str[0].isin(["Ax3", "Ax6"]).sum())))
x = np.arange(3); ax_e.bar(x, [r[2] for r in rows], color=TIER_COL["Impassable"], width=0.55)
for i, r in enumerate(rows): ax_e.text(i, r[2] + 0.5, f"{r[2]:.1f}%\nn={r[1]}\nAx3+Ax6: {r[3]}", ha="center", fontsize=7.6)
ax_e.set_xticks(x); ax_e.set_xticklabels([f">= {r[0]}%" for r in rows]); ax_e.set_ylim(0, 31)
ax_e.set_xlabel("Minimum observed share of the window", fontsize=9.5); ax_e.set_ylabel("Impassable nodes (%)", fontsize=9.5)
ax_e.set_title("(e) Effect of the coverage criterion", fontsize=11.5, fontweight="bold")

# ---- (f)-(h) geometric validation ----------------------------------------------------------------------------------------------
ax_f = fig.add_subplot(gs[2, 0:2]); ax_g = fig.add_subplot(gs[2, 2:4]); ax_h = fig.add_subplot(gs[2, 4:6])
calib = pd.read_csv(P / "calib.csv"); anc = pd.read_csv(P / "anchor_cons.csv"); anc = anc[anc.h == 2.5]
for src, col in (("TG", "#c77d7d"), ("SCS", "#4575b4")):
    g = calib[calib.src == src]
    ax_f.scatter(g.w_corr, g.w_pred_loso, s=14, alpha=.7, color=col, lw=0, label=f"{'Legacy axes' if src == 'TG' else 'Konak sample'} (n={len(g)})")
mx = max(calib.w_corr.max(), calib.w_pred_loso.max()); ax_f.plot([0, mx], [0, mx], color="#999", lw=.8)
ax_f.set_xlabel("GIS building-to-building width (m)", fontsize=9.5); ax_f.set_ylabel("Width reconstructed from imagery (m)", fontsize=9.5)
ax_f.legend(frameon=False, fontsize=7.2); ax_f.set_title("(f) Corridor-width reconstruction accuracy", fontsize=11.5, fontweight="bold")
ax_g.hist(calib.h_node.clip(1, 4), bins=20, color="#91ad91", edgecolor="white")
ax_g.set_xlabel("Node-level effective camera height (m)", fontsize=9.5); ax_g.set_ylabel("Nodes", fontsize=9.5); ax_g.set_title("(g) Effective camera height", fontsize=11.5, fontweight="bold")
ax_h.hist(anc.e.clip(-8, 8), bins=24, color="#cfb57e", edgecolor="white"); ax_h.axvline(0, color="#999", lw=.8)
ax_h.set_xlabel("Left minus right camera-offset estimate (m)", fontsize=9.5); ax_h.set_ylabel("Nodes", fontsize=9.5); ax_h.set_title("(h) Left/right offset consistency", fontsize=11.5, fontweight="bold")

save(fig, "figS3_pipeline_validation")

s = d[d.src == "SCS"]; t = d[d.src == "TG"]
stat("covSCSPct", round(100 * s.met.mean(), 0)); stat("covSCSNarrowPct", round(100 * s[s.w_corridor < 6].met.mean(), 0)); stat("covSCSLowestPct", round(100 * s.groupby("wb", observed=True).met.mean().min(), 0))
stat("covTGPct", round(100 * t.met.mean(), 0)); stat("covMedSCSnarrow", round(float(s[s.w_corridor < 8].coverage.median()), 2)); stat("covMedTGnarrow", round(float(t[t.w_corridor < 8].coverage.median()), 2))
print("done")
