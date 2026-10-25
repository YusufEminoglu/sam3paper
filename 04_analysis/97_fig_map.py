"""Study-area figure, hero-hub layout: the map of Konak with the analysed nodes coloured by ECW tier in the centre and, around it, one 3D axonometric
inset for each of the 16 sampled H3 cells with a candidate node (buildings extruded by their floor count, streets, analysed nodes as tier-coloured prisms).
Hex 97 (H3 893f66a6e73ffff, an industrial/TARIS plot) is dropped: none of its 25 candidates ever had a corridor width, imagery, or analysis.
Layout, palette, furniture (segmented scale bar, north arrow, stacked legend), card frames, morphology chips and 90-degree callout leaders follow
the house style of the companion typology figure (f11_typology.py).  Replaces the first version's two-panel map (panel b dropped).
All positions come from the node table, the corridor table, the municipal layers (EPSG:5253) and the sampled-cell polygons of the companion study."""
import itertools
import geopandas as gpd
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import unary_union
from sigfig_common import *

mpl.rcParams.update({"font.family": "DejaVu Sans"})
G = REV / "02_pipeline_v2/konak_layers.gpkg"

# ---- house-style palette (f11_typology.py) ------------------------------------------------------------------------
PAL = {"land": "#ffffff", "ink": "#1a1f2b", "ink_soft": "#475569", "ink_faint": "#9aa3b2", "edge": "#cbd5e1", "slate_border": "#475569", "dark_slate": "#1e293b"}
M_PER_FLOOR = 3.5
ROAD_W = {"BULVAR": 14.0, "CADDE": 9.0, "SOKAK": 5.0, "MEYDAN": 12.0}

# ---- data ----------------------------------------------------------------------------------------------------------
n = load()
c = pd.read_csv(REV / "02_pipeline_v2/corridor_nodes.csv"); c = c[c.src == "SCS"]
c["node"] = "scs_" + c.id.astype(int).astype(str); excl = c[~c.node.isin(n.node)]
roads = gpd.read_file(G, layer="roads"); bound = gpd.read_file(G, layer="boundary"); bld = gpd.read_file(G, layer="buildings")
bld = bld[bld.geometry.notna() & bld.geometry.is_valid]; roads = roads[roads.geometry.notna()]
hexes = gpd.read_file(REV / "02_pipeline_v2/konak_hexes.gpkg", layer="hexes").set_index("hex_id")
hexes = hexes.loc[hexes.index.isin(set(c[c.src == "SCS"].hex_id.dropna().unique()) | set(n[n.src == "SCS"].hex_id.dropna().unique()))]  # drop cells with no remaining SCS candidate (e.g. hex 97: industrial plot, 0 candidates ever qualified)
u = bound.union_all(); boundary = gpd.GeoDataFrame(geometry=[Polygon(u.exterior) if u.geom_type == "Polygon" else MultiPolygon([Polygon(p.exterior) for p in u.geoms])], crs=bound.crs)
xmin, ymin, xmax, ymax = boundary.total_bounds; map_ar = (ymax - ymin) / (xmax - xmin)

s = n[n.src == "SCS"]
hexes["hx"] = hexes.geometry.centroid.x; hexes["hy"] = hexes.geometry.centroid.y

# ---- canvas and card grid (inches) -----------------------------------------------------------------------------------
w_card, h_card, gap_x = 2.45, 1.95, 0.16   # card size follows the house style of f11_typology.py
N_TOP, N_BOT, N_LEFT, N_RIGHT = 5, 5, 3, 3   # 16 cards total, matching the 16 sampled cells with a candidate node
span_row = N_TOP * w_card + (N_TOP - 1) * gap_x
GUTTER_MARGIN = 0.50   # reserved band, clear of any side-column card, that houses up to 5 stacked top/bottom leader tracks (4*0.09 in + slack)
side_stack_h = max(N_LEFT, N_RIGHT) * h_card + (max(N_LEFT, N_RIGHT) - 1) * 0.18
fig_w = span_row + 0.60; fig_h = side_stack_h + 2 * GUTTER_MARGIN + 0.40 + 2 * h_card   # canvas sized to the 5-wide row, the 3-tall side columns, and the gutter bands
margin_l = (fig_w - span_row) / 2.0
y_bot = 0.20; bot_row_top = y_bot + h_card; y_top = fig_h - 0.20 - h_card
map_w = span_row - 2 * w_card - 0.80; map_h = map_w * map_ar; map_x = margin_l + w_card + 0.40
map_y = bot_row_top + (y_top - bot_row_top - map_h) / 2.0
mid_h = y_top - bot_row_top

def col_y(k, n):
    """k-th of n vertically stacked slots, centred on the band between the bottom and top rows, inset by GUTTER_MARGIN
    at both ends so the top/bottom leader gutters always have room to stack their tracks without entering a card row."""
    inner_bot, inner_top = bot_row_top + GUTTER_MARGIN, y_top - GUTTER_MARGIN
    return inner_bot + (inner_top - inner_bot) / 2 + ((n - 1) / 2 - k) * (h_card + 0.18) - h_card / 2

# card slots: (side, k) -> (lower-left x, lower-left y) and the anchor of the leader on the card border
slots = []
for k in range(N_TOP): slots.append(("top", k, margin_l + k * (w_card + gap_x), y_top))
for k in range(N_RIGHT): slots.append(("right", k, margin_l + (N_TOP - 1) * (w_card + gap_x), col_y(k, N_RIGHT)))
for k in range(N_BOT): slots.append(("bot", k, margin_l + (N_BOT - 1 - k) * (w_card + gap_x), y_bot))
for k in range(N_LEFT): slots.append(("left", k, margin_l, col_y(k, N_LEFT)))

def anchor(sd, x, y):
    return {"top": (x + w_card / 2, y), "bot": (x + w_card / 2, y + h_card), "left": (x + w_card, y + h_card / 2), "right": (x, y + h_card / 2)}[sd]

# hex position in figure inches (map axes limits are set below with the exact aspect)
pad = 0.0
ax_lim = (xmin - pad, xmax + pad, ymin - pad, ymax + pad)
def to_in(x, y):
    return map_x + (x - ax_lim[0]) / (ax_lim[1] - ax_lim[0]) * map_w, map_y + (y - ax_lim[2]) / (ax_lim[3] - ax_lim[2]) * map_h

# ---- assign hexes to slots: simulated annealing on crossings + leader length + clashes with the furniture box ---------------------------
import random
hid = list(hexes.index)
TR = 0.09
col_top = max(y + h_card for sd, k, x, y in slots if sd in ("left", "right")); col_bot = min(y for sd, k, x, y in slots if sd in ("left", "right"))
GT = max(0.10, col_top - (map_y + map_h) + 0.09); GB = max(0.10, map_y - col_bot + 0.09); GUT0 = 0.10   # top/bottom gutters clear the side columns
F_BOX = (map_x, map_y + 0.47 * map_h, map_x + 0.50 * map_w, map_y + map_h)     # furniture region (title, legend, scale) in inches

def path(h, side, x, y, t):
    px, py = to_in(hexes.hx[h], hexes.hy[h]); ax_, ay_ = anchor(side, x, y)
    if side == "top": g = map_y + map_h + GT + t * TR; return [(px, py), (px, g), (ax_, g), (ax_, ay_)]
    if side == "bot": g = map_y - GB - t * TR; return [(px, py), (px, g), (ax_, g), (ax_, ay_)]
    if side == "left": g = map_x - GUT0 - t * TR; return [(px, py), (g, py), (g, ay_), (ax_, ay_)]
    g = map_x + map_w + GUT0 + t * TR; return [(px, py), (g, py), (g, ay_), (ax_, ay_)]

def seg_cross(a, b, c_, d):
    def ccw(p, q, r): return (r[1] - p[1]) * (q[0] - p[0]) - (q[1] - p[1]) * (r[0] - p[0])
    d1, d2, d3, d4 = ccw(a, b, c_), ccw(a, b, d), ccw(c_, d, a), ccw(c_, d, b)
    return (d1 * d2 < 0) and (d3 * d4 < 0)

def n_cross(paths):
    tot = 0
    for p, q in itertools.combinations(paths, 2):
        for i in range(len(p) - 1):
            for j in range(len(q) - 1):
                tot += seg_cross(p[i], p[i + 1], q[j], q[j + 1])
    return tot

def hits_box(pts, box):
    x0, y0, x1, y1 = box; k = 0
    for (ax_, ay_), (bx_, by_) in zip(pts[:-1], pts[1:]):
        lx, hx_ = sorted((ax_, bx_)); ly, hy_ = sorted((ay_, by_))
        if lx < x1 and hx_ > x0 and ly < y1 and hy_ > y0: k += 1
    return k

def card_box(sd, x, y, pad=0.03):
    """card footprint in figure inches, shrunk a little so a leader's own anchor on the border doesn't count as a hit"""
    return (x + pad, y + pad, x + w_card - pad, y + h_card - pad)

def card_hits(state, ps):
    """each leader must avoid every OTHER card's body, not just the furniture legend"""
    boxes = {h: card_box(*state[h][:1], state[h][2], state[h][3]) for h in hid}
    return sum(hits_box(ps[h], boxes[h2]) for h in hid for h2 in hid if h2 != h)

SIDE_KEY = {"top": lambda h: hexes.hx[h], "bot": lambda h: -hexes.hx[h], "right": lambda h: -hexes.hy[h], "left": lambda h: -hexes.hy[h]}
def normalize(state):
    """within a side, cards follow the order of their cells along that side (west to east / north to south)"""
    out = {}
    for sd in ("top", "right", "bot", "left"):
        hs = sorted([h for h in hid if state[h][0] == sd], key=SIDE_KEY[sd]); sl = sorted([v for v in state.values() if v[0] == sd], key=lambda v: v[1])
        for h, v in zip(hs, sl): out[h] = v
    return out

def heur_tracks(state):
    """pairwise: the leader that crosses fewer others when placed on the lower track goes lower; Copeland score per side"""
    tr = {}
    for sd in ("top", "right", "bot", "left"):
        hs = [h for h in hid if state[h][0] == sd]; win = {h: 0 for h in hs}
        for i, j in itertools.combinations(hs, 2):
            pi0, pj1 = path(i, sd, state[i][2], state[i][3], 0), path(j, sd, state[j][2], state[j][3], 1)
            pi1, pj0 = path(i, sd, state[i][2], state[i][3], 1), path(j, sd, state[j][2], state[j][3], 0)
            ci, cj = n_cross([pi0, pj1]), n_cross([pi1, pj0])
            if ci < cj: win[i] += 1
            elif cj < ci: win[j] += 1
        for r, h in enumerate(sorted(hs, key=lambda h: -win[h])): tr[h] = r
    return tr

def energy(state):
    state = normalize(state); tr = heur_tracks(state); ps = {h: path(h, *state[h][:1], state[h][2], state[h][3], tr[h]) for h in hid}
    ln = sum(abs(p[i + 1][0] - p[i][0]) + abs(p[i + 1][1] - p[i][1]) for p in ps.values() for i in range(3))
    return 1000 * n_cross(list(ps.values())) + ln + 800 * sum(hits_box(p, F_BOX) for p in ps.values()) + 1000 * card_hits(state, ps)

cost = np.zeros((len(hid), len(slots)))
for i, h in enumerate(hid):
    px, py = to_in(hexes.hx[h], hexes.hy[h])
    for j, (sd, k, x, y) in enumerate(slots):
        ax_, ay_ = anchor(sd, x, y); cost[i, j] = abs(px - ax_) + abs(py - ay_)
ri, ci = linear_sum_assignment(cost)
state = {hid[i]: slots[j] for i, j in zip(ri, ci)}
bestS, bestE = dict(state), energy(state)
for seed in range(6):
    rng = random.Random(seed); cur = dict(bestS) if seed == 0 else {h: slots[j] for h, j in zip(hid, rng.sample(range(len(slots)), len(hid)))}
    E = energy(cur); T = 600.0
    for it in range(2500):
        a_, b_ = rng.sample(hid, 2); cur[a_], cur[b_] = cur[b_], cur[a_]
        E2 = energy(cur)
        if E2 < E or rng.random() < np.exp((E - E2) / T): E = E2
        else: cur[a_], cur[b_] = cur[b_], cur[a_]
        if E < bestE: bestS, bestE = dict(cur), E
        T = max(2.0, T * 0.998)
    print("seed", seed, "best energy", round(bestE, 1), flush=True)
    if bestE < 1000: break

improved = True                                     # exhaustive 2-opt polish: repeatedly apply the single best-improving pairwise swap
while improved:
    improved = False; bestc_, bestpair = bestE, None
    for a_, b_ in itertools.combinations(hid, 2):
        cur = dict(bestS); cur[a_], cur[b_] = cur[b_], cur[a_]
        e = energy(cur)
        if e < bestc_ - 1e-9: bestc_, bestpair = e, (a_, b_)
    if bestpair is not None:
        a_, b_ = bestpair; bestS[a_], bestS[b_] = bestS[b_], bestS[a_]; bestE = bestc_; improved = True
print("2-opt polished energy:", round(bestE, 1), flush=True)
assign = normalize(bestS); print("annealed energy:", round(bestE, 1))

by_side = {sd: [h for h in hid if assign[h][0] == sd] for sd in ("top", "right", "bot", "left")}
tracks = heur_tracks(assign)
def total(tr):
    ps = {h: path(h, assign[h][0], assign[h][2], assign[h][3], tr[h]) for h in hid}
    return 1000 * n_cross(list(ps.values())) + card_hits(assign, ps)
for sd in ("top", "bot", "right", "left"):          # exact track order, one side at a time
    hs = by_side[sd]; bestc, bestp = None, None
    for perm in itertools.permutations(range(len(hs))):
        for h, t_ in zip(hs, perm): tracks[h] = t_
        cc = total(tracks)
        if bestc is None or cc < bestc: bestc, bestp = cc, perm
        if cc == 0: break
    for h, t_ in zip(hs, bestp): tracks[h] = t_
ps_final = {h: path(h, assign[h][0], assign[h][2], assign[h][3], tracks[h]) for h in hid}
N_CROSS = n_cross(list(ps_final.values())); N_HIT = sum(hits_box(p, F_BOX) for p in ps_final.values()); N_CARDHIT = card_hits(assign, ps_final)
print("leader crossings:", N_CROSS, " furniture hits:", N_HIT, " card hits:", N_CARDHIT)
if N_CARDHIT:
    _boxes = {h: card_box(*assign[h][:1], assign[h][2], assign[h][3]) for h in hid}
    for h in hid:
        for h2 in hid:
            if h2 != h and hits_box(ps_final[h], _boxes[h2]): print("  card-hit: leader of", h, "crosses card of", h2)

# ---- per-hex statistics --------------------------------------------------------------------------------------------------------
def polys_of(g):
    if g is None or g.is_empty: return []
    if g.geom_type == "Polygon": return [g]
    if g.geom_type in ("MultiPolygon", "GeometryCollection"): return [q for gg in g.geoms for q in polys_of(gg)]
    return []

def hex_stats(h):
    hp = hexes.geometry[h]; sub = bld[bld.intersects(hp)]
    fp, fa, rows = 0.0, 0.0, []
    for g, fl in zip(sub.geometry, sub.ZEMINUSTUKATSAYISI.fillna(1)):
        fl = 1 if fl <= 0 else int(fl)
        for p in polys_of(g.intersection(hp)):
            fp += p.area; fa += p.area * fl; rows.append((p, fl))
    return rows, len(rows), (fa / fp if fp > 0 else 0.0), fp / hp.area, fa / hp.area

# ---- figure -------------------------------------------------------------------------------------------------------------------------
fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
ax_map = fig.add_axes([map_x / fig_w, map_y / fig_h, map_w / fig_w, map_h / fig_h]); ax_map.set_facecolor(PAL["land"])
roads.plot(ax=ax_map, color="#dfe5ec", linewidth=0.30, zorder=2)
boundary.boundary.plot(ax=ax_map, color=PAL["dark_slate"], linewidth=1.0, zorder=6)
ax_map.scatter(excl.x_node, excl.y_node, s=7, c="#9aa3b2", marker="x", lw=0.5, zorder=3)
for t in TIERS[::-1]:
    d = n[n.tier3 == t]; ax_map.scatter(d.x_node, d.y_node, s=15, c=TIER_COL[t], edgecolors="white", lw=0.3, zorder=4)
ax_map.set_xlim(ax_lim[0], ax_lim[1]); ax_map.set_ylim(ax_lim[2], ax_lim[3]); ax_map.set_aspect("equal"); ax_map.axis("off"); ax_map.apply_aspect()

# tags: clockwise from the upper left
order = sorted(hid, key=lambda h: ({"top": 0, "right": 1, "bot": 2, "left": 3}[assign[h][0]], assign[h][1]))
tag = {h: i + 1 for i, h in enumerate(order)}
tier_of = {}
for h in hid:
    d = s[s.hex_id == h]; tier_of[h] = d.tier3.value_counts().idxmax() if len(d) else None
HEXCOL = {h: (TIER_COL[tier_of[h]] if tier_of[h] else PAL["ink_soft"]) for h in hid}

# ---- in-gulf furniture (f11 stack: north arrow -> scale bar -> legend -> text box) -------------------------------------
dx, dy = ax_lim[1] - ax_lim[0], ax_lim[3] - ax_lim[2]
FX, FY = 0.005, 0.905                                     # anchor in axes fraction (set from the rendered gulf)
x0 = ax_lim[0] + 0.355 * dx; base_y = ax_lim[2] + 0.555 * dy; slen = 1000.0; sb_h = dy * 0.010; seg_w = slen / 2
ax_map.add_patch(Rectangle((x0, base_y), seg_w, sb_h, facecolor=PAL["ink"], edgecolor="black", linewidth=0.4, zorder=25))
ax_map.add_patch(Rectangle((x0 + seg_w, base_y), seg_w, sb_h, facecolor="white", edgecolor="black", linewidth=0.4, zorder=25))
for val, xp in [("0", x0), ("0.5", x0 + seg_w), ("1", x0 + slen)]:
    ax_map.text(xp, base_y + sb_h * 1.3, val, ha="center", va="bottom", fontsize=8, fontweight="bold", color=PAL["ink"], zorder=25)
ax_map.text(x0 + slen + dx * 0.008, base_y + sb_h * 0.1, "km", ha="left", va="bottom", fontsize=8, fontweight="bold", color=PAL["ink"], zorder=25)
cxn = x0 + slen / 2; na_base = base_y + sb_h * 4.6; na_w = dx * 0.013; na_h = dy * 0.032
ax_map.add_patch(mpatches.Polygon([[cxn, na_base + na_h], [cxn - na_w / 2, na_base], [cxn + na_w / 2, na_base]], closed=True, facecolor=PAL["ink"], edgecolor="black", linewidth=0.4, zorder=25))
ax_map.text(cxn, na_base + na_h + dy * 0.006, "N", ha="center", va="bottom", fontsize=9, fontweight="bold", color=PAL["ink"], zorder=25)

cnt_t = {t: int((n.tier3 == t).sum()) for t in TIERS}
handles = [Line2D([], [], marker="o", ls="", markerfacecolor=TIER_COL[t], markeredgecolor="white", markersize=6.5, label=f"{TIER_LABEL[t]} (n={cnt_t[t]})") for t in TIERS]
handles.append(Line2D([], [], marker="x", ls="", color="#9aa3b2", markersize=5, label=f"Candidate node not analysed (n={len(excl)})"))
handles.append(Patch(facecolor="white", edgecolor=PAL["dark_slate"], linewidth=1.0, label=f"Sampled H3 cell with 3D inset (n={len(hid)})"))
leg = ax_map.legend(handles=handles, loc="upper left", bbox_to_anchor=(FX, FY), title="Effective Clear Width (ECW) tier", fontsize=8.4, title_fontsize=9.2,
                    frameon=True, facecolor="white", edgecolor=PAL["edge"], framealpha=0.96, borderpad=0.35, labelspacing=0.30)
leg.get_title().set_fontweight("bold"); leg.set_zorder(20)
fig.canvas.draw(); bb = leg.get_window_extent(fig.canvas.get_renderer()).transformed(ax_map.transAxes.inverted())
summary = (f"Konak sample: {int((n.src == 'SCS').sum())} nodes in {s.hex_id.nunique()} of {len(hid)} cells\n"
           f"Legacy pilot: {int((n.src == 'TG').sum())} nodes on 7 axes\n"
           f"Inset: buildings at 3.5 m per floor, nodes as prisms\nMorphology chip: BCR (coverage) · FAR (density)")
ax_map.text(FX, bb.y0 - 0.014, summary, transform=ax_map.transAxes, ha="left", va="top", fontsize=6.8, fontweight="bold", color=PAL["ink_soft"],
            bbox={"facecolor": "white", "edgecolor": PAL["edge"], "boxstyle": "round,pad=0.25", "linewidth": 0.40, "alpha": 0.94}, zorder=28)
ax_map.text(0.005, 0.995, f"Konak: analysed nodes by ECW tier (N = {len(n)})", transform=ax_map.transAxes, ha="left", va="top", fontsize=9.6, fontweight="bold", color=PAL["dark_slate"],
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=PAL["dark_slate"], alpha=0.96, lw=0.5), zorder=30, clip_on=False)

# ---- cards -------------------------------------------------------------------------------------------------------------------------------
for h in hid:
    side, k, cx_card, cy_card = assign[h]; col = HEXCOL[h]; hp = hexes.geometry[h]
    for dx_s, dy_s, px_, py_, a_ in [(0.015, -0.018, 0.032, 0.032, 0.030), (0.010, -0.013, 0.020, 0.020, 0.045), (0.006, -0.008, 0.010, 0.010, 0.065), (0.003, -0.004, 0.004, 0.004, 0.085)]:
        fig.add_artist(FancyBboxPatch(((cx_card + dx_s - px_ / 2) / fig_w, (cy_card + dy_s - py_ / 2) / fig_h), (w_card + px_) / fig_w, (h_card + py_) / fig_h, transform=fig.transFigure,
                                      boxstyle="round,pad=0,rounding_size=0.008", facecolor="#0f172a", edgecolor="none", alpha=a_, zorder=2))
    fig.add_artist(FancyBboxPatch((cx_card / fig_w, cy_card / fig_h), w_card / fig_w, h_card / fig_h, transform=fig.transFigure, boxstyle="round,pad=0,rounding_size=0.006",
                                  facecolor="white", edgecolor=PAL["edge"], linewidth=0.85, zorder=4))
    ax3 = fig.add_axes([cx_card / fig_w, cy_card / fig_h, w_card / fig_w, h_card / fig_h], projection="3d", zorder=10); ax3.computed_zorder = False; ax3.set_facecolor("none")
    minx, miny, maxx, maxy = hp.bounds; wl, hl = maxx - minx, maxy - miny
    rows, n_b, avg_fl, bcr, far = hex_stats(h)
    loc = lambda p: ([x - minx for x in p.exterior.coords.xy[0]][:-1], [y - miny for y in p.exterior.coords.xy[1]][:-1])

    ground = Poly3DCollection([[(x - minx, y - miny, 0) for x, y in list(hp.exterior.coords)[:-1]]], facecolor="#f8fafc", edgecolor=PAL["edge"], linewidth=0.7); ground.set_zorder(1); ax3.add_collection3d(ground)
    grid = []
    hp_l = Polygon([(x - minx, y - miny) for x, y in hp.exterior.coords])
    for gx in np.arange(0, wl + 0.1, 30.0):
        seg = LineString([(gx, 0), (gx, hl)]).intersection(hp_l)
        for q in ([seg] if seg.geom_type == "LineString" else list(getattr(seg, "geoms", []))):
            if q.length > 0: grid.append([(q.coords[0][0], q.coords[0][1], 0.02), (q.coords[-1][0], q.coords[-1][1], 0.02)])
    for gy in np.arange(0, hl + 0.1, 30.0):
        seg = LineString([(0, gy), (wl, gy)]).intersection(hp_l)
        for q in ([seg] if seg.geom_type == "LineString" else list(getattr(seg, "geoms", []))):
            if q.length > 0: grid.append([(q.coords[0][0], q.coords[0][1], 0.02), (q.coords[-1][0], q.coords[-1][1], 0.02)])
    gc = Line3DCollection(grid, colors="#e2e8f0", linewidths=0.45); gc.set_zorder(1.5); ax3.add_collection3d(gc)

    rs = roads[roads.intersects(hp)]; st_faces = []
    for g, yt in zip(rs.geometry, rs.YOLTIP):
        for p in polys_of(g.buffer(ROAD_W.get(yt, 6.0) / 2.0).intersection(hp)):
            xs, ys = loc(p)
            if len(xs) >= 3: st_faces.append([(xs[i], ys[i], 0.08) for i in range(len(xs))])
    if st_faces:
        sc = Poly3DCollection(st_faces, facecolor="#e2e8f0", edgecolor="none"); sc.set_zorder(2); ax3.add_collection3d(sc)

    faces, fcol, flines = [], [], []
    roof = mpl.colors.to_rgba("#f1f5f9", 0.95); wl_c = mpl.colors.to_rgba("#94a3b8", 0.50); wd_c = mpl.colors.to_rgba("#64748b", 0.55); max_h = 10.0
    for p, fl in rows:
        height = fl * M_PER_FLOOR; max_h = max(max_h, height); xs, ys = loc(p); m = len(xs)
        if m < 3: continue
        faces.append([(xs[i], ys[i], height) for i in range(m)]); fcol.append(roof)
        for i in range(m):
            j = (i + 1) % m
            faces.append([(xs[i], ys[i], 0), (xs[j], ys[j], 0), (xs[j], ys[j], height), (xs[i], ys[i], height)])
            fcol.append(wl_c if (xs[j] - xs[i]) - (ys[j] - ys[i]) >= 0 else wd_c)
            for f in range(1, fl): flines.append([(xs[i], ys[i], f * M_PER_FLOOR), (xs[j], ys[j], f * M_PER_FLOOR)])
    if faces:
        bc = Poly3DCollection(faces, facecolor=fcol, edgecolor="#94a3b8", linewidth=0.10); bc.set_zorder(4); ax3.add_collection3d(bc)
    if flines:
        fc_ = Line3DCollection(flines, colors="#64748b", linewidths=0.08); fc_.set_zorder(5); ax3.add_collection3d(fc_)

    # nodes: analysed nodes as tier-coloured prisms (f11 target style), candidates not analysed as grey crosses
    tf, tc = [], []
    d_an = s[s.hex_id == h]
    for _, r in d_an.iterrows():
        col_t = TIER_COL[r.tier3]; ang = np.linspace(0, 2 * np.pi, 13)[:-1]; R, H = 6.0, 11.0
        px, py = r.x_node - minx, r.y_node - miny; ring = [(px + R * np.cos(a), py + R * np.sin(a)) for a in ang]
        tf.append([(x, y, H) for x, y in ring]); tc.append(mpl.colors.to_rgba(col_t, 0.95))
        for i in range(len(ring)):
            j = (i + 1) % len(ring); tf.append([(*ring[i], 0), (*ring[j], 0), (*ring[j], H), (*ring[i], H)]); tc.append(mpl.colors.to_rgba(col_t, 0.70))
    if tf:
        tcoll = Poly3DCollection(tf, facecolor=tc, edgecolor="#1e293b", linewidth=0.22); tcoll.set_zorder(6); ax3.add_collection3d(tcoll)
    ex = excl[excl.hex_id == h]; xl = []
    for _, r in ex.iterrows():
        px, py = r.x_node - minx, r.y_node - miny; xl += [[(px - 5, py - 5, 0.3), (px + 5, py + 5, 0.3)], [(px - 5, py + 5, 0.3), (px + 5, py - 5, 0.3)]]
    if xl:
        xc = Line3DCollection(xl, colors="#64748b", linewidths=0.6); xc.set_zorder(5.5); ax3.add_collection3d(xc)

    ax3.set_xlim(0, wl); ax3.set_ylim(0, hl); zmax = max(max_h * 1.10, 30.0); ax3.set_zlim(0, zmax)
    ax3.set_box_aspect((wl, hl, zmax * 1.6), zoom=1.36); ax3.view_init(elev=34, azim=-60); ax3.set_axis_off()

    fig.add_artist(FancyBboxPatch((cx_card / fig_w, cy_card / fig_h), w_card / fig_w, h_card / fig_h, transform=fig.transFigure, boxstyle="round,pad=0,rounding_size=0.006",
                                  facecolor="none", edgecolor=PAL["edge"], linewidth=0.85, zorder=30))
    bx, by = (cx_card + 0.05) / fig_w, (cy_card + h_card - 0.05) / fig_h
    fig.text(bx, by, f"({tag[h]})", ha="left", va="top", fontsize=8.0, fontweight="bold", color="white", bbox={"facecolor": col, "edgecolor": "none", "boxstyle": "round,pad=0.20"}, zorder=32)
    med = d_an.ecw_best.median() if len(d_an) else float("nan")
    l2 = f"{len(d_an)} analysed · {len(ex)} not analysed"
    fig.text(bx + 0.40 / fig_w, by, f"H3 {hexes.h3_index[h]}\n{l2}", ha="left", va="top", fontsize=6.1, fontweight="bold", color=PAL["ink"],
             bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.90, "pad": 0.15}, zorder=31)
    sh = {t: (100 * (d_an.tier3 == t).mean() if len(d_an) else 0) for t in TIERS}
    hud = f"⌂ {n_b} bld   ↕ {avg_fl:.1f} fl   {hp.area / 1e4:.1f} ha\nBCR: {bcr:.2f}   FAR: {far:.2f}" + (f"   med. ECW {med:.1f} m" if len(d_an) else "") + f"\nImp. {sh['Impassable']:.0f}% · Restr. {sh['Restricted']:.0f}% · Acc. {sh['Accessible']:.0f}%"
    fig.text((cx_card + 0.05) / fig_w, (cy_card + 0.05) / fig_h, hud, ha="left", va="bottom", fontsize=6.0, fontweight="bold", color=PAL["ink_soft"],
             bbox={"facecolor": "white", "edgecolor": PAL["edge"], "boxstyle": "round,pad=0.18", "linewidth": 0.40, "alpha": 0.92}, zorder=33)

    # hex outline on the map, origin marker and 90-degree leader
    ring = list(hp.exterior.coords); ax_map.plot([q[0] for q in ring], [q[1] for q in ring], color=col, lw=1.5, zorder=15, solid_joinstyle="round")
    ax_map.scatter([hexes.hx[h]], [hexes.hy[h]], marker="o", s=26, facecolor="white", edgecolor=col, linewidth=1.3, zorder=22)
    pts = path(h, side, cx_card, cy_card, tracks[h]); pts = [(a / fig_w, b / fig_h) for a, b in pts]
    fig.add_artist(Line2D([p[0] for p in pts], [p[1] for p in pts], transform=fig.transFigure, color=col, linewidth=1.0, linestyle="--", alpha=0.85, zorder=25))
    fig.add_artist(plt.Circle(pts[-1], radius=0.0034, transform=fig.transFigure, facecolor=col, edgecolor="white", linewidth=0.6, zorder=35))
    ax_map.annotate(str(tag[h]), (hexes.hx[h], hexes.hy[h]), xytext=(0, 9), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="white",
                    bbox={"facecolor": col, "edgecolor": "none", "boxstyle": "round,pad=0.15"}, zorder=30)

save(fig, "fig04_map")
