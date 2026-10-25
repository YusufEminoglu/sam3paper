"""Representative street-view frames and SAM 3 masks for the seven legacy axes, in the layout of the first version's Figure 3 (columns = axes; rows = left frame, left masks, right frame, right masks).
Differences from the first version: the frames are the re-retrieved panoramas whose capture month is fixed and known (provenance table), the header gives the GIS corridor width and the capture month
instead of the four typologies, and the masks are the stored SAM 3 output (score >= 0.6). Node choice: the node used in the first version if it has a panorama from March-November 2024-2025, otherwise the
nearest such node on the same axis."""
import sys, json
from PIL import Image
from sigfig_common import *

PIPE = REV / "02_pipeline_v2"
sys.path.insert(0, str(PIPE))
from imgpath import image_path
man = pd.read_csv(PIPE / "manifest_run.csv").set_index("key"); prov = pd.read_csv(PIPE / "tg_images_R1_provenance.csv").set_index("node")
cor = pd.read_csv(PIPE / "corridor_nodes.csv"); cor = cor[cor.src == "TG"].set_index("id")
AXC = {"Ax1": "#c77d7d", "Ax2": "#a16363", "Ax3": "#7d94b5", "Ax4": "#617491", "Ax5": "#89b1c2", "Ax6": "#cfb57e", "Ax7": "#91ad91"}
FIRST = {"Ax1": 80, "Ax2": 160, "Ax3": 40, "Ax4": 40, "Ax5": 20, "Ax6": 120, "Ax7": 60}
COL = {"Vehicles": (230, 57, 70), "Furniture": (244, 163, 0), "Plants": (76, 175, 80), "Infrastructure": (140, 60, 200), "Person": (59, 111, 216)}
GRP = {}
for g, ks in GROUPS.items():
    for k in ks: GRP[k.replace("_", " ")] = g
GRP["person"] = "Person"

def pick(ax):
    d0 = FIRST[ax]; c = []
    for k in range(0, 200, 20):
        nid = f"{ax}_{k}"
        if nid in prov.index and prov.loc[nid, "eligible"] and f"tg_{ax}_{k}_L" in man.index and f"tg_{ax}_{k}_R" in man.index: c.append((abs(k - d0), k))
    return sorted(c)[0][1] if c else None

def overlay(key):
    j = json.load(open(PIPE / "seg_v2" / f"{key}.json", encoding="utf-8")); z = np.load(PIPE / "seg_v2" / f"{key}.npz"); sh = tuple(z["shape"])
    img = np.array(Image.open(image_path(man.loc[key])).convert("RGB")).astype(float); out = img.copy()
    for i, d in enumerate(j["dets"]):
        g = GRP.get(d["prompt"])
        if g is None or d["score"] < 0.6: continue
        m = np.unpackbits(z["masks"][i])[:sh[0] * sh[1]].reshape(sh).astype(bool); out[m] = 0.45 * out[m] + 0.55 * np.array(COL[g], float)
    return img.astype(np.uint8), out.astype(np.uint8)

axes_ids = list(AXC); ncol = len(axes_ids)
fig, ax = plt.subplots(4, ncol, figsize=(2.05 * ncol + 0.5, 2.05 * 4 + 1.3), facecolor="white", gridspec_kw=dict(wspace=0.025, hspace=0.025, left=0.03, right=0.999, top=0.885, bottom=0.05))
info = []
for c, a in enumerate(axes_ids):
    k = pick(a)
    if k is None:
        for r in range(4): ax[r, c].axis("off")
        ax[0, c].text(0.5, 0.5, f"{a}\nno node with a panorama\nin the March–November\nwindow of 2024–2025", ha="center", va="center", fontsize=8.5, transform=ax[0, c].transAxes, color="#777")
        continue
    nid = f"{a}_{k}"; wc = cor.loc[nid, "w_corridor"] if nid in cor.index else np.nan; date = str(prov.loc[nid, "pano_date"])
    for r, (side, key) in enumerate((("L", f"tg_{a}_{k}_L"), ("L", f"tg_{a}_{k}_L"), ("R", f"tg_{a}_{k}_R"), ("R", f"tg_{a}_{k}_R"))):
        o, ov = overlay(key); ax[r, c].imshow(ov if r % 2 else o); ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
        for s in ax[r, c].spines.values(): s.set_color(AXC[a]); s.set_linewidth(1.4)
    fig.text(ax[0, c].get_position().x0 + ax[0, c].get_position().width / 2, 0.965, a, ha="center", va="center", fontsize=11.5, fontweight="bold", color=AXC[a])
    fig.text(ax[0, c].get_position().x0 + ax[0, c].get_position().width / 2, 0.925, f"W$_c$ = {wc:.0f} m  ·  node at {k} m\ncaptured {date}", ha="center", va="center", fontsize=7.6, color="#555", linespacing=1.35)
    fig.add_artist(plt.Rectangle((ax[0, c].get_position().x0, 0.893), ax[0, c].get_position().width, 0.008, transform=fig.transFigure, color=AXC[a], lw=0))
    info.append((a, k))
for r, lab in enumerate(("Left: street-view frame", "Left: SAM 3 masks", "Right: street-view frame", "Right: SAM 3 masks")):
    fig.text(0.012, ax[r, 0].get_position().y0 + ax[r, 0].get_position().height / 2, lab, rotation=90, ha="center", va="center", fontsize=8.6, fontweight="bold", color="#444")
import matplotlib.patches as mpatches
fig.legend(handles=[mpatches.Patch(color=np.array(v) / 255, label=k) for k, v in COL.items()], loc="lower center", ncol=5, frameon=False, fontsize=8.6, bbox_to_anchor=(0.5, -0.005))
save(fig, "figS3_imagery", ext="jpg")
print(info)
