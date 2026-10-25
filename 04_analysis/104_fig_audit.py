"""Instance-level audit of SAM 3: (a) precision by obstacle class and group with Wilson 95% intervals, (b) a gallery of the false positives found by the rater
(mask outlined on the frame; class, score, and the frame's identifier). Instances are numbered as in the audit tool (score >= 0.6, listed classes, file order)."""
import json
from PIL import Image
from sigfig_common import *

AUD = REV / "03_validation/audit"; SEG = REV / "02_pipeline_v2/seg_v2"
au = pd.read_csv(REV / "03_validation/out_final/audit_summary.csv").set_index("group")
ins = pd.read_csv(AUD / "audit_instances_filled.csv", encoding="utf-8-sig"); man = pd.read_csv(REV / "02_pipeline_v2/manifest_run.csv").set_index("key")
SHOW = ["person", "car", "truck", "motorcycle", "chair", "table", "potted plant", "trash can", "dumpster", "utility pole", "street light", "utility box", "electrical cabinet"]

fig = plt.figure(figsize=(19, 8.2), facecolor=BG)
gs = fig.add_gridspec(1, 2, width_ratios=[0.78, 1.5], wspace=0.10)
ax = fig.add_subplot(gs[0]); ax.set_facecolor(BG)
order = [g for g in au.index if g.startswith("cls:")]; order = sorted(order, key=lambda g: au.loc[g].precision)
rows = order + ["cat:vehicle", "cat:furniture", "cat:infrastructure", "cat:person", "ALL"]
LAB = {"cat:vehicle": "Vehicles (group)", "cat:furniture": "Furniture and plants (group)", "cat:infrastructure": "Infrastructure (group)", "cat:person": "Person (group)", "ALL": "All rated instances"}
y = 0; ticks = []
for g in rows:
    r = au.loc[g]; grp = not g.startswith("cls:"); c = "#333" if g == "ALL" else ("#7d94b5" if grp else OBJ.get(g[4:].replace(" ", "_"), "#888"))
    ax.hlines(y, r.ci_lo, r.ci_hi, color=c, lw=2.2 if grp else 1.6, alpha=0.9); ax.scatter(r.precision, y, s=70 if grp else 55, color=c, zorder=3, edgecolors="white", lw=0.8)
    ax.text(101.5, y, f"{int(r.TP)}/{int(r.n)}", va="center", fontsize=8.5, color="#555"); ticks.append((y, LAB.get(g, g[4:].title())))
    y += 1 if g != order[-1] else 1.8
    if g == "cat:person": y += 0.6
ax.set_yticks([t for t, _ in ticks]); ax.set_yticklabels([l for _, l in ticks], fontsize=9.5); ax.invert_yaxis(); ax.set_xlim(20, 108)
ax.axvline(au.loc["ALL"].precision, color="#999", lw=0.9, ls="--"); ax.set_xlabel("Precision (%) with Wilson 95% interval; label = correct / rated", fontsize=9.5); ax.grid(axis="x", lw=0.3)
ax.set_title("(a) Precision by class (single rater, 15 images)", fontsize=11.5, fontweight="bold")

# ---- (b) false-positive gallery ---------------------------------------------------------------------------------------------------
fp = ins[ins.judgement == "FP"].copy(); fp["score"] = fp.score.astype(float)
pick = pd.concat([g.head(2) for _, g in fp.sort_values("score", ascending=False).groupby("cls")]).sort_values(["cls", "score"], ascending=[True, False]).head(8)
sub = gs[1].subgridspec(2, 4, wspace=0.05, hspace=0.20)
for k, r in enumerate(pick.itertuples()):
    a = fig.add_subplot(sub[k // 4, k % 4]); a.axis("off")
    j = json.load(open(SEG / f"{r.key}.json", encoding="utf-8")); z = np.load(SEG / f"{r.key}.npz"); sh = tuple(z["shape"]); idx = 0; mask = None
    for i, d in enumerate(j["dets"]):
        if d["score"] < 0.6 or d["prompt"] not in SHOW: continue
        idx += 1
        if idx == r.inst: mask = np.unpackbits(z["masks"][i])[:sh[0] * sh[1]].reshape(sh).astype(bool)
    im = np.array(Image.open(man.loc[r.key, "path"]).convert("RGB")).astype(float); out = im.copy()
    er = mask.copy(); er[1:] &= mask[:-1]; er[:-1] &= mask[1:]; er[:, 1:] &= mask[:, :-1]; er[:, :-1] &= mask[:, 1:]
    out[mask] = 0.55 * out[mask] + 0.45 * np.array([255, 60, 60]); out[mask & ~er] = (255, 235, 0)
    ys, xs = np.where(mask); m = 55; y0, y1, x0, x1 = max(0, ys.min() - m), min(sh[0], ys.max() + m), max(0, xs.min() - m), min(sh[1], xs.max() + m)
    a.imshow(out[y0:y1, x0:x1].astype(np.uint8)); a.set_title(f"{r.cls}  (score {r.score:.2f})\nframe {r.key}", fontsize=8.6, fontweight="bold", color="#8e3b3b", pad=3)
fig.text(0.56, 0.955, "(b) False positives found in the audit: highlighted mask judged not to be an object of the stated class", fontsize=11.5, fontweight="bold", ha="left")
save(fig, "fig_sig_audit")
stat("audFPshown", len(pick)); stat("audFPtotal", int((ins.judgement == "FP").sum()))
print(len(pick), "shown of", int((ins.judgement == "FP").sum()), "FP")
