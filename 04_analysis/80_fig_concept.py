"""Conceptual cross-section: corridor width, carriageway width, pedestrian-intended width, obstacles, and Effective Clear Width.
usage: python 80_fig_concept.py  -> ../05_manuscript_R1/figures/fig03_concept.png (schematic, not to scale; no data)
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

OUT = Path(__file__).parent.parent / "figures"; OUT.mkdir(exist_ok=True)
C_BLD, C_ROAD, C_PED, C_OBS = "#8c8c8c", "#d9d9d9", "#cfe3f5", "#c0392b"
fig, ax = plt.subplots(figsize=(8.2, 3.9), dpi=200)
Wc, Wv, dL, dR = 14.0, 6.0, 7.0, 7.0            # schematic only
xL, xR = -dL, dR
ax.add_patch(Rectangle((xL - 3, 0), 3, 8, color=C_BLD)); ax.add_patch(Rectangle((xR, 0), 3, 8, color=C_BLD))
ax.add_patch(Rectangle((-Wv / 2, 0), Wv, 0.6, color=C_ROAD))
ax.add_patch(Rectangle((xL, 0), dL - Wv / 2, 0.6, color=C_PED)); ax.add_patch(Rectangle((Wv / 2, 0), dR - Wv / 2, 0.6, color=C_PED))
# obstacles inside the pedestrian zones (schematic footprints)
ax.add_patch(Rectangle((xL + 0.3, 0.6), 1.4, 1.2, color=C_OBS)); ax.add_patch(Rectangle((-Wv / 2 - 1.3, 0.6), 0.7, 0.9, color=C_OBS))
ax.add_patch(Rectangle((Wv / 2 + 0.3, 0.6), 1.2, 1.1, color=C_OBS))
ax.text(0, 2.6, "vehicle in carriageway:\nnot counted", ha="center", fontsize=7, style="italic")
ax.add_patch(Rectangle((-1.2, 0.6), 2.4, 1.1, fill=False, ec="#777", ls="--", lw=0.8))
def dim(x0, x1, y, txt, col="k", dy=0.35):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="<->", mutation_scale=8, lw=1, color=col)); ax.text((x0 + x1) / 2, y + dy, txt, ha="center", fontsize=8, color=col)
dim(xL, xR, 6.6, "corridor width $W_c$ (building to building, from GIS footprints)")
dim(-Wv / 2, Wv / 2, 4.9, "carriageway $W_v$ (assumed: lanes $\\times$ lane width)")
dim(xL, -Wv / 2, 3.6, "$W_p^L$", "#1f5f99"); dim(Wv / 2, xR, 3.6, "$W_p^R$", "#1f5f99")
dim(xL + 1.8, -Wv / 2 - 1.4, 2.4, "ECW: widest continuous\nfree run (bottleneck)", "#1e7a3a", dy=0.25)
ax.text(xL - 1.5, 5, "building", rotation=90, va="center", ha="center", color="w", fontsize=8); ax.text(xR + 1.5, 5, "building", rotation=90, va="center", ha="center", color="w", fontsize=8)
ax.text(xL + 1.0, 0.15, "pedestrian-intended zone", fontsize=7, color="#1f5f99")
ax.set_xlim(xL - 3.2, xR + 3.2); ax.set_ylim(0, 8); ax.axis("off")
fig.savefig(OUT / "fig03_concept.png", bbox_inches="tight", pad_inches=0.02); print("saved", OUT / "fig03_concept.png")
