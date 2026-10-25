#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 1 (\\label{fig:workflow}) -- "Methodological workflow and computational pipeline".

TiGIS R1 revision.  1:1 square composition on pure white, three interconnected tiers
linked by thin conduits, flat-vector / axonometric hybrid artwork, fully controlled
typography (Lato), geometrically correct photogrammetry and the complete
ADA/EAA-style accessibility threshold set.

Output: 4096 x 4096 px PNG (600 dpi metadata) -> fig01_workflow.png

Run:  python3 src/fig01_workflow.py [out.png ...]
"""
from __future__ import annotations

import glob
import math
import os
import sys

from PIL import Image
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import (
    Arc,
    Circle,
    Ellipse,
    FancyBboxPatch,
    Polygon,
    Rectangle,
    Wedge,
)
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.patheffects as pe

# Ensure Lato fonts are registered even when executed from Windows host Python
for _font_dir in (
    "/usr/share/fonts/truetype/lato",
    r"\\wsl.localhost\Ubuntu\usr\share\fonts\truetype\lato",
):
    if os.path.isdir(_font_dir):
        for _ttf in glob.glob(os.path.join(_font_dir, "*.ttf")):
            try:
                fm.fontManager.addfont(_ttf)
            except Exception:
                pass

# --------------------------------------------------------------------------------------
# Editorial palette (master guide, section 4)
# --------------------------------------------------------------------------------------
NAVY = "#2C4A6F"    # navy slate
PINE = "#2E523C"    # forest pine
OCHRE = "#B87333"   # warm umber / ochre
BRICK = "#9E4740"   # muted brick red
TEAL = "#244D47"    # mineral teal
INK = "#1E293B"     # deep charcoal text
GREY = "#64748B"    # muted slate grey
HAIR = "#D7DEE6"    # hairline
SOFT = "#EEF2F6"    # neutral tint
PAPER = "#FFFFFF"
SAGE = "#8FA98F"    # vegetation tint
SLATE_L = "#A9BCCD"
ASPHALT = "#C9CFD6"
SAND = "#E4DED0"

PX = 4096           # canvas edge (px)
DPI = 600

plt.rcParams.update({
    "font.family": "Lato",
    "font.size": 7.5,
    "figure.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "axes.linewidth": 0.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Lato",
    "mathtext.it": "Lato:italic",
    "mathtext.bf": "Lato:bold",
    "mathtext.cal": "Lato",
    "mathtext.sf": "Lato",
    "mathtext.default": "regular",
})

_FIT = []          # (text artist, max width in px, min size in pt)


# --------------------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------------------
def fr(x0, y0, w, h):
    """Local (u, v in 0..1, v up) -> figure-pixel mapper."""
    def T(u, v):
        return x0 + u * w, y0 + v * h
    return T


def poly(ax, pts, fc="none", ec="none", lw=1.0, z=3, alpha=1.0, ls="-"):
    p = Polygon(pts, closed=True, facecolor=fc, edgecolor=ec, linewidth=lw,
                zorder=z, alpha=alpha, linestyle=ls, joinstyle="round")
    ax.add_patch(p)
    return p


def rr(ax, x, y, w, h, r=18, fc=PAPER, ec="none", lw=1.0, z=2, alpha=1.0,
       shadow=False, ls="-"):
    r = min(r, w / 2 - 1, h / 2 - 1)
    box = FancyBboxPatch((x + r, y + r), w - 2 * r, h - 2 * r,
                         boxstyle=f"round,pad={r},rounding_size={r}",
                         facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z,
                         alpha=alpha, linestyle=ls, mutation_aspect=1)
    if shadow:
        box.set_path_effects([
            pe.SimplePatchShadow(offset=(0, -14), shadow_rgbFace="#98A5B3",
                                 alpha=0.22, rho=0.986),
            pe.Normal(),
        ])
    ax.add_patch(box)
    return box


def txt(ax, x, y, s, size=7.5, weight=400, color=INK, ha="left", va="center",
        z=6, style="normal", alpha=1.0, lsp=1.25, maxw=None, minsize=4.2, **kw):
    t = ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha,
                va=va, zorder=z, fontstyle=style, alpha=alpha, linespacing=lsp, **kw)
    if maxw:
        _FIT.append((t, maxw, minsize, size))
    return t


def line(ax, p0, p1, color=INK, lw=1.0, z=4, ls="-", alpha=1.0, cap="round"):
    ln, = ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, linewidth=lw,
                   zorder=z, linestyle=ls, alpha=alpha, solid_capstyle=cap,
                   dash_capstyle="butt")
    return ln


def circle(ax, c, r, fc="none", ec="none", lw=1.0, z=4, alpha=1.0, ls="-"):
    p = Circle(c, r, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z,
               alpha=alpha, linestyle=ls)
    ax.add_patch(p)
    return p


def arc(ax, c, r, th1, th2, ec=INK, lw=1.0, z=4, alpha=1.0, ls="-"):
    p = Arc(c, 2 * r, 2 * r, angle=0, theta1=th1, theta2=th2, edgecolor=ec,
            linewidth=lw, zorder=z, alpha=alpha, linestyle=ls)
    ax.add_patch(p)
    return p


def arrowhead(ax, tip, ang, length=26, half=15, color=NAVY, z=6):
    bx = tip[0] - length * math.cos(ang)
    by = tip[1] - length * math.sin(ang)
    nx, ny = -math.sin(ang) * half, math.cos(ang) * half
    poly(ax, [(tip[0], tip[1]), (bx + nx, by + ny), (bx - nx, by - ny)],
         fc=color, ec=color, lw=0.5, z=z)


def rounded_path(points, radius=34, seg=12):
    """Round the corners of an orthogonal-ish polyline."""
    pts = [np.asarray(p, float) for p in points]
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        p0, p1, p2 = pts[i - 1], pts[i], pts[i + 1]
        d0, d1 = p0 - p1, p2 - p1
        l0, l1 = np.linalg.norm(d0), np.linalg.norm(d1)
        if l0 < 1e-6 or l1 < 1e-6:
            continue
        r = min(radius, l0 * 0.45, l1 * 0.45)
        a = p1 + d0 / l0 * r
        b = p1 + d1 / l1 * r
        out.append(a)
        for k in range(seg + 1):
            t = k / seg
            mt = 1 - t
            out.append(mt * mt * a + 2 * mt * t * p1 + t * t * b)
    out.append(pts[-1])
    return np.asarray(out)


def conduit(ax, points, color=NAVY, lw=3.0, head=30, z=4, radius=36):
    """Thin flow conduit: rounded open polyline + solid arrow head."""
    dense = rounded_path(points, radius=radius)
    a, b = dense[-2].copy(), dense[-1].copy()
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    # Pull shaft end back inside the arrowhead so round cap does not protrude past tip
    shaft = dense.copy()
    shaft[-1] = b - np.array([math.cos(ang), math.sin(ang)]) * (head * 0.65)
    codes = [Path.MOVETO] + [Path.LINETO] * (len(shaft) - 1)
    ax.add_patch(PathPatch(Path(shaft, codes), fill=False, edgecolor=color,
                           linewidth=lw, capstyle="round", joinstyle="round",
                           zorder=z))
    arrowhead(ax, b, ang, head, head * 0.56, color, z + 1)


def conduit_label(ax, x, y, s, color, rot=0, z=9):
    """Small data label riding on or above a conduit (white halo breaks the line)."""
    t = ax.text(x, y, s, fontsize=5.8, fontweight=700, color=color,
                ha="center", va="center", zorder=z, rotation=rot,
                bbox=dict(boxstyle="round,pad=0.26,rounding_size=0.4",
                          fc=PAPER, ec=HAIR, lw=1.0, alpha=0.98))
    return t


def dim_arrow(ax, p0, p1, color=INK, lw=1.8, z=6):
    line(ax, p0, p1, color=color, lw=lw, z=z)
    for p, base in ((p0, p1), (p1, p0)):
        ang = math.atan2(base[1] - p[1], base[0] - p[0])
        arrowhead(ax, p, ang, 15, 8, color, z)


def hexpoly(cx, cy, r):
    return [(cx + r * math.cos(math.radians(60 * k)),
             cy + r * math.sin(math.radians(60 * k))) for k in range(6)]


def swatch(ax, x, y, color, r=16, alpha=0.90, ec="none", lw=1.0, hatch=False):
    if hatch:
        box = rr(ax, x - r, y - r, 2 * r, 2 * r, r=4, fc=PAPER, ec=color, lw=1.3, z=6)
        for k in range(-2, 3):
            ln = line(ax, (x - r + k * 10 - 6, y - r), (x - r + k * 10 + 2 * r - 6, y + r),
                      color="#7F8EA0", lw=1.0, z=7)
            ln.set_clip_path(box)
    else:
        rr(ax, x - r, y - r, 2 * r, 2 * r, r=4, fc=color, ec=ec, lw=lw, z=6, alpha=alpha)


def apply_fits(fig):
    """Shrink any over-wide registered text so nothing can overflow its frame."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    for t, maxw, minsize, base in _FIT:
        bb = t.get_window_extent(renderer=rend)
        if bb.width > maxw and bb.width > 0:
            new = max(minsize, base * maxw / bb.width)
            t.set_fontsize(new)


# --------------------------------------------------------------------------------------
# Frame furniture
# --------------------------------------------------------------------------------------
def tier_frame(ax, x0, x1, y_bot, y_top, label, accent, tint):
    """Colored dashed tier enclosure with the tier label riding on its top border."""
    pad_x, pad_bot, pad_top = 26, 20, 32
    rr(ax, x0 - pad_x, y_bot - pad_bot,
       (x1 - x0) + 2 * pad_x, (y_top - y_bot) + pad_bot + pad_top,
       r=30, fc=tint, ec=accent, lw=1.35, z=1, ls=(0, (4.0, 3.0)), alpha=0.92)
    t = ax.text(x0 + 26, y_top + pad_top, label, fontsize=6.3, fontweight=700,
                color=accent, ha="left", va="center", zorder=8,
                bbox=dict(boxstyle="round,pad=0.34,rounding_size=0.6",
                          fc=PAPER, ec=accent, lw=1.25))
    return t


def draw_header_icons(ax, x_right, y_center, icon_names, radius=37.5, gap=20):
    """Draws pure software/algorithm logos floating cleanly on the right of the header, without boxes or text."""
    x_cur = x_right
    icons_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons")
    if not os.path.exists(icons_dir):
        cur = os.path.dirname(os.path.abspath(__file__))
        while cur and os.path.dirname(cur) != cur:
            cand = os.path.join(cur, "src", "icons")
            if os.path.exists(cand):
                icons_dir = cand
                break
            cur = os.path.dirname(cur)

    for name in reversed(icon_names):
        p = os.path.join(icons_dir, f"{name}.png")
        if os.path.exists(p):
            im = Image.open(p)
            ext = [x_cur - 2 * radius, x_cur, y_center - radius, y_center + radius]
            ax.imshow(im, extent=ext, zorder=7, aspect="auto")
            x_cur -= 2 * radius + gap


def stage_badge(ax, cx, cy, n, color, r=35):
    circle(ax, (cx, cy), r, fc=color, ec="none", z=6)
    txt(ax, cx, cy - 1, str(n), size=7.5, weight=700, color=PAPER, ha="center",
        va="center", z=7)


def card_header(ax, x0, y1, x1, n, accent, title, subtitle=None, icons=None):
    stage_badge(ax, x0 + 62, y1 - 60, n, accent, r=35)
    txt(ax, x0 + 116, y1 - 42, title, size=8.0, weight=700, color=INK, ha="left",
        va="center", z=7, maxw=x1 - x0 - 150)
    if subtitle:
        txt(ax, x0 + 116, y1 - 114, subtitle, size=5.8, weight=400, color=GREY,
            ha="left", va="center", z=7, maxw=x1 - x0 - 150)
    if icons:
        draw_header_icons(ax, x1 - 50, y1 - 60, icons, radius=37.5, gap=20)
    line(ax, (x0 + 28, y1 - 154), (x1 - 28, y1 - 154), color=HAIR, lw=1.4, z=5)


# --------------------------------------------------------------------------------------
# Panel -- street canyon (one-point perspective corridor with transect rays)
# --------------------------------------------------------------------------------------
def draw_street_canyon(ax, x0, y0, w, h):
    T = fr(x0, y0, w, h)
    SC_Y, SC_H = 0.265, 0.720
    VP = (0.5, 0.55)

    def S(qx, qy):
        return T(qx, SC_Y + SC_H * qy)

    def PT(x, v, d):
        s = 1.0 - 0.855 * d
        return S(VP[0] + (x - VP[0]) * s, VP[1] + (v - VP[1]) * s)

    X_L, X_R = 0.150, 0.850
    X_KL, X_KR = 0.295, 0.705
    D = 0.92

    # ground
    poly(ax, [PT(X_L, 0, 0), PT(X_KL, 0, 0), PT(X_KL, 0, D), PT(X_L, 0, D)],
         fc=SAND, ec="none", z=3)
    poly(ax, [PT(X_KR, 0, 0), PT(X_R, 0, 0), PT(X_R, 0, D), PT(X_KR, 0, D)],
         fc=SAND, ec="none", z=3)
    poly(ax, [PT(X_KL, 0, 0), PT(X_KR, 0, 0), PT(X_KR, 0, D), PT(X_KL, 0, D)],
         fc=ASPHALT, ec="none", z=3)
    for xk in (X_KL, X_KR):
        line(ax, PT(xk, 0, 0), PT(xk, 0, D), color="#B7BFC8", lw=2.0, z=4)
    d = 0.02
    while d < D - 0.02:
        dd = min(d + 0.048, D)
        line(ax, PT(0.5, 0, d), PT(0.5, 0, dd), color=PAPER, lw=3.4, z=4)
        d = dd + 0.038

    # buildings
    for xf in (X_L, X_R):
        poly(ax, [PT(xf, 0, 0), PT(xf, 0, D), PT(xf, 0.99, D), PT(xf, 0.99, 0)],
             fc="#DDE6EF", ec="#22364C", lw=1.2, z=4)
        for v0 in (0.17, 0.38, 0.59, 0.80):
            for col in range(5):
                d0 = 0.05 + col * 0.180
                d1 = d0 + 0.110
                if d1 > D - 0.02:
                    break
                poly(ax, [PT(xf, v0, d0), PT(xf, v0, d1), PT(xf, v0 + 0.125, d1),
                          PT(xf, v0 + 0.125, d0)],
                     fc="#C4D2E1", ec="#3C5474", lw=0.9, z=5)
        poly(ax, [PT(xf, 0.99, 0), PT(xf, 0.99, D), PT(xf, 1.02, D), PT(xf, 1.02, 0)],
             fc="#22364C", ec="none", z=5)
    poly(ax, [PT(0.0, 0, 0), PT(X_L, 0, 0), PT(X_L, 0.99, 0), PT(0.0, 0.99, 0)],
         fc="#B6C4D3", ec="#22364C", lw=1.2, z=4)
    poly(ax, [PT(X_R, 0, 0), PT(1.0, 0, 0), PT(1.0, 0.99, 0), PT(X_R, 0.99, 0)],
         fc="#B6C4D3", ec="#22364C", lw=1.2, z=4)
    for xa, xb in ((0.0, X_L), (X_R, 1.0)):
        for v0 in (0.17, 0.38, 0.59, 0.80):
            for col in range(2):
                u0 = xa + 0.030 + col * 0.078
                u1 = u0 + 0.050
                if u1 > xb - 0.015:
                    continue
                poly(ax, [PT(u0, v0, 0), PT(u1, v0, 0), PT(u1, v0 + 0.125, 0),
                          PT(u0, v0 + 0.125, 0)],
                     fc="#C8D5E3", ec="#3C5474", lw=0.8, z=5)

    # transect rays
    cam = PT(0.5, 0.0, 0.10)
    for dd in (0.26, 0.46, 0.66):
        for xf in (X_L, X_R):
            line(ax, cam, PT(xf, 0.26, dd), color=OCHRE, lw=1.1, z=6, alpha=0.9)
    circle(ax, cam, 9, fc=OCHRE, ec=PAPER, lw=1.3, z=7)

    # objects (far -> near)
    def tree(xc, dc, r=0.034, col=SAGE):
        s = 1.0 - 0.855 * dc
        p = PT(xc, 0, dc)
        line(ax, p, (p[0], p[1] + 0.24 * h * s), color="#6B5334", lw=max(1.2, 2.4 * s), z=5)
        circle(ax, (p[0], p[1] + 0.27 * h * s), r * w * s, fc=col, ec="#5E7A62",
               lw=1.0, z=6)

    def car(xc, dc, wid, col=BRICK):
        s = 1.0 - 0.855 * dc
        x0c, x1c = xc - wid / 2, xc + wid / 2
        p = PT(xc, 0, dc)
        bw, bh = wid * w * s, 0.105 * h * s
        body = [(p[0] - bw / 2, p[1]), (p[0] + bw / 2, p[1]),
                (p[0] + bw / 2, p[1] + bh * 0.62), (p[0] - bw / 2, p[1] + bh * 0.62)]
        cab = [(p[0] - bw * 0.30, p[1] + bh * 0.62), (p[0] + bw * 0.30, p[1] + bh * 0.62),
               (p[0] + bw * 0.22, p[1] + bh), (p[0] - bw * 0.22, p[1] + bh)]
        poly(ax, body, fc=col, ec="#5E2A26", lw=1.0, z=6)
        poly(ax, cab, fc=col, ec="#5E2A26", lw=1.0, z=6)
        for sgn in (-1, 1):
            circle(ax, (p[0] + sgn * bw * 0.30, p[1] + bh * 0.05),
                   max(2.0, 5.0 * s), fc="#2B2F36", ec="none", z=6)

    def person(xc, dc, col="#43536B"):
        s = 1.0 - 0.855 * dc
        p = PT(xc, 0, dc)
        hh = 0.125 * h * s
        circle(ax, (p[0], p[1] + hh * 0.90), max(2.0, 5.0 * s), fc=col, ec="none", z=6)
        poly(ax, [(p[0] - hh * 0.14, p[1]), (p[0] + hh * 0.14, p[1]),
                  (p[0] + hh * 0.11, p[1] + hh * 0.70), (p[0] - hh * 0.11, p[1] + hh * 0.70)],
             fc=col, ec="none", z=6)

    tree(0.235, 0.60, 0.030)
    tree(0.765, 0.64, 0.028)
    car(0.360, 0.52, 0.100, BRICK)
    tree(0.215, 0.30, 0.036)
    car(0.350, 0.20, 0.125, BRICK)
    tree(0.785, 0.30, 0.032)
    car(0.650, 0.24, 0.125, "#8E5A4C")
    person(0.215, 0.12)
    person(0.765, 0.13)
    person(0.425, 0.42)

    # dimension strip
    dy = y0 + 0.168 * h
    x_of = lambda u: x0 + u * w
    for a, b, lab in ((X_L, X_KL, r"$W_p$"), (X_KL, X_KR, r"$W_v$"),
                      (X_KR, X_R, r"$W_p$")):
        p0, p1 = (x_of(a), dy), (x_of(b), dy)
        line(ax, p0, p1, color=INK, lw=1.8, z=6)
        for p in (p0, p1):
            line(ax, (p[0], p[1] - 10), (p[0], p[1] + 10), color=INK, lw=1.5, z=6)
        txt(ax, (p0[0] + p1[0]) / 2, dy + 34, lab, size=7.8, weight=700, color=INK,
            ha="center", va="center", z=7)
    dy2 = y0 + 0.048 * h
    p0, p1 = (x_of(X_L), dy2), (x_of(X_R), dy2)
    line(ax, p0, p1, color=BRICK, lw=2.0, z=6)
    for p in (p0, p1):
        line(ax, (p[0], p[1] - 12), (p[0], p[1] + 12), color=BRICK, lw=1.7, z=6)
    txt(ax, (p0[0] + p1[0]) / 2, dy2, r"$W_c$  corridor width", size=7.6,
        weight=700, color=BRICK, ha="center", va="center", z=8,
        bbox=dict(boxstyle="round,pad=0.22", fc=PAPER, ec="none", alpha=0.98))


# --------------------------------------------------------------------------------------
# Panel -- 360 camera rig
# --------------------------------------------------------------------------------------
def draw_camera_rig(ax, x0, y0, w, h):
    T = fr(x0, y0, w, h)
    R = 0.28 * w
    cx, cy = T(0.42, 0.58)

    circle(ax, (cx, cy), R, fc="#E1EBF5", ec="#8FA9C2", lw=1.3, z=3, alpha=0.8)
    for frac in (0.40, 0.74):
        ax.add_patch(Ellipse((cx, cy), 2 * R * frac, 2 * R, facecolor="none",
                             edgecolor="#A9C0D4", linewidth=0.8,
                             linestyle=(0, (4, 4)), zorder=4))
    ax.add_patch(Ellipse((cx, cy), 2 * R, 2 * R * 0.28, facecolor="none",
                         edgecolor="#A9C0D4", linewidth=0.8, linestyle=(0, (4, 4)),
                         zorder=4))

    # tripod
    hub = T(0.42, 0.30)
    for fx, fy in ((0.08, 0.09), (0.42, 0.07), (0.76, 0.09)):
        line(ax, hub, T(fx, fy), color="#39465A", lw=3.6, z=5)
    line(ax, hub, (cx, cy), color="#39465A", lw=3.8, z=5)

    # camera body
    bw, bh = 0.30 * w, 0.145 * h
    rr(ax, cx - bw / 2, cy - bh / 2, bw, bh, r=10, fc="#39465A", ec="#1E2833",
       lw=1.2, z=7)
    rr(ax, cx - bw / 2 + 9, cy - bh / 2 + 9, bw - 18, bh * 0.34, r=5, fc="#4E5E74",
       ec="none", z=8)
    for ox in (-0.082, 0.0, 0.082):
        circle(ax, (cx + ox * w, cy), 0.028 * w, fc="#16202B", ec="#8FA9C2", lw=1.2, z=8)
        circle(ax, (cx + ox * w, cy), 0.011 * w, fc="#9FC4DC", ec="none", z=9)

    # cardinal viewing vectors
    for ang in (0, 90, 180, 270):
        a = math.radians(ang)
        p0 = (cx + 0.100 * w * math.cos(a), cy + 0.100 * w * math.sin(a))
        p1 = (cx + 0.240 * w * math.cos(a), cy + 0.240 * w * math.sin(a))
        line(ax, p0, p1, color=TEAL, lw=1.5, z=9)
        arrowhead(ax, p1, a, 17, 9, TEAL, 10)

    # downward pitch
    py = cy - 0.210 * h
    line(ax, (cx + 0.02 * w, py), (cx + 0.29 * w, py), color=GREY, lw=0.9, z=8,
         ls=(0, (4, 3)))
    a = math.radians(-10)
    p0 = (cx + 0.13 * w, py)
    p1 = (cx + 0.29 * w * math.cos(a), py + 0.29 * w * math.sin(a))
    line(ax, p0, p1, color=OCHRE, lw=1.6, z=9)
    arrowhead(ax, p1, a, 17, 9, OCHRE, 10)
    arc(ax, p0, 0.095 * w, -10, 0, ec=OCHRE, lw=1.0, z=9)
    txt(ax, p1[0] + 22, p1[1] - 4, "−10°", size=7.2, weight=700, color=OCHRE,
        ha="left", va="center", z=10)

    # labels
    line(ax, (cx + R * 0.22, cy + R * 0.94), (cx + R * 0.36, cy + R * 1.10),
         color="#9AA7B4", lw=0.9, z=6)
    txt(ax, cx + R * 0.40, cy + R * 1.12, "coverage dome", size=6.2, weight=400,
        color=GREY, ha="center", va="center", z=7)


# --------------------------------------------------------------------------------------
# Panel -- geospatial node grid
# --------------------------------------------------------------------------------------
def draw_node_map(ax, x0, y0, w, h):
    T = fr(x0, y0, w, h)
    horiz = [
        [(0.03, 0.18), (0.28, 0.15), (0.54, 0.21), (0.78, 0.17), (0.97, 0.24)],
        [(0.02, 0.44), (0.30, 0.41), (0.56, 0.46), (0.80, 0.42), (0.98, 0.47)],
        [(0.02, 0.72), (0.26, 0.69), (0.54, 0.74), (0.78, 0.70), (0.98, 0.75)],
        [(0.04, 0.94), (0.32, 0.91), (0.60, 0.95), (0.96, 0.92)],
    ]
    vert = [
        [(0.16, 0.02), (0.19, 0.30), (0.15, 0.60), (0.21, 0.98)],
        [(0.42, 0.01), (0.40, 0.32), (0.45, 0.62), (0.41, 0.99)],
        [(0.68, 0.02), (0.72, 0.34), (0.66, 0.64), (0.71, 0.98)],
        [(0.90, 0.02), (0.87, 0.30), (0.92, 0.60), (0.88, 0.98)],
    ]
    roads = horiz + vert
    for s in roads:
        pts = [T(*p) for p in s]
        for i in range(len(pts) - 1):
            line(ax, pts[i], pts[i + 1], color="#DCE3EB", lw=2.0, z=3, cap="butt")
    for s in roads:
        pts = [T(*p) for p in s]
        for i in range(len(pts) - 1):
            line(ax, pts[i], pts[i + 1], color="#FFFFFF", lw=1.0, z=4, cap="butt")

    centres = [(0.19, 0.28), (0.36, 0.50), (0.56, 0.31), (0.67, 0.73), (0.30, 0.80),
               (0.86, 0.50), (0.12, 0.56), (0.47, 0.86), (0.88, 0.84), (0.61, 0.55),
               (0.80, 0.33), (0.24, 0.12)]
    for (u, v) in centres:
        c = T(u, v)
        poly(ax, hexpoly(c[0], c[1], 0.066 * w), fc=TEAL, ec=TEAL, lw=1.2, z=5, alpha=0.13)
        poly(ax, hexpoly(c[0], c[1], 0.066 * w), fc="none", ec=TEAL, lw=1.3, z=6, alpha=0.72)
    for (u, v) in centres:
        c = T(u, v)
        circle(ax, c, 6.5, fc=TEAL, ec=PAPER, lw=1.3, z=7)

    nx, ny = T(0.955, 0.12)
    line(ax, (nx, ny), (nx, ny + 44), color=INK, lw=1.8, z=8)
    arrowhead(ax, (nx, ny + 50), math.pi / 2, 17, 9, INK, 9)
    txt(ax, nx, ny - 32, "N", size=6.6, weight=700, color=INK, ha="center",
        va="center", z=9)




# --------------------------------------------------------------------------------------
# Panel -- zero-shot panoptic segmentation scene
# --------------------------------------------------------------------------------------
def draw_segment_scene(ax, x0, y0, w, h):
    T = fr(x0, y0, w, h)
    VP = (0.52, 0.50)

    def PT(x, v, d):
        s = 1.0 - 0.86 * d
        return T(VP[0] + (x - VP[0]) * s, VP[1] + (v - VP[1]) * s)

    X_L, X_R = 0.115, 0.885
    X_KL, X_KR = 0.315, 0.725
    D = 0.94

    poly(ax, [PT(X_L, 0, 0), PT(X_KL, 0, 0), PT(X_KL, 0, D), PT(X_L, 0, D)],
         fc=SAND, ec="none", z=3)
    poly(ax, [PT(X_KR, 0, 0), PT(X_R, 0, 0), PT(X_R, 0, D), PT(X_KR, 0, D)],
         fc=SAND, ec="none", z=3)
    poly(ax, [PT(X_KL, 0, 0), PT(X_KR, 0, 0), PT(X_KR, 0, D), PT(X_KL, 0, D)],
         fc="#C6CCD3", ec="none", z=3)
    d = 0.02
    while d < D - 0.02:
        dd = min(d + 0.042, D)
        line(ax, PT(0.52, 0, d), PT(0.52, 0, dd), color=PAPER, lw=1.8, z=4)
        d = dd + 0.062

    for xf in (X_L, X_R):
        poly(ax, [PT(xf, 0, 0), PT(xf, 0, D), PT(xf, 0.86, D), PT(xf, 0.86, 0)],
             fc="#D6DBE1", ec="#B3BBC4", lw=1.1, z=4)
        for v0 in (0.14, 0.34, 0.54, 0.74):
            for col in range(4):
                d0 = 0.08 + col * 0.225
                d1 = d0 + 0.125
                if d1 > D - 0.02:
                    break
                poly(ax, [PT(xf, v0, d0), PT(xf, v0, d1), PT(xf, v0 + 0.12, d1),
                          PT(xf, v0 + 0.12, d0)],
                     fc="#C1C9D2", ec="#AAB3BD", lw=0.8, z=5)

    def mask_poly(pts, col, alpha=0.80, z=6):
        poly(ax, pts, fc=col, ec="none", z=z, alpha=alpha)

    def baseline(pts, col=INK):
        ax.add_patch(Polygon(pts, closed=True, facecolor="none", edgecolor=col,
                             linewidth=1.0, zorder=9, linestyle=(0, (3, 2))))

    def vehicle(xc, dc, wid, col=BRICK, hgt=0.105):
        s = 1.0 - 0.86 * dc
        p = PT(xc, 0, dc)
        bw, bh = wid * w * s, hgt * h * s
        body = [(p[0] - bw / 2, p[1]), (p[0] + bw / 2, p[1]),
                (p[0] + bw / 2, p[1] + bh * 0.60), (p[0] - bw / 2, p[1] + bh * 0.60)]
        cab = [(p[0] - bw * 0.30, p[1] + bh * 0.60), (p[0] + bw * 0.30, p[1] + bh * 0.60),
               (p[0] + bw * 0.22, p[1] + bh), (p[0] - bw * 0.22, p[1] + bh)]
        mask_poly(body, col, 0.82)
        mask_poly(cab, col, 0.82)
        poly(ax, body, fc="none", ec="#6E2F2B", lw=0.8, z=7)
        poly(ax, cab, fc="none", ec="#6E2F2B", lw=0.8, z=7)
        for sgn in (-1, 1):
            circle(ax, (p[0] + sgn * bw * 0.30, p[1] + bh * 0.05),
                   max(1.8, 4.6 * s), fc="#2B2F36", ec="none", z=7)
        baseline([(p[0] - bw * 0.56, p[1] - 3), (p[0] + bw * 0.56, p[1] - 3),
                  (p[0] + bw * 0.50, p[1] + 4), (p[0] - bw * 0.50, p[1] + 4)])

    def pole(xc, dc, col=SLATE_L, hp=0.38, arm=True):
        p0, p1 = PT(xc, 0, dc), PT(xc, hp, dc)
        line(ax, p0, p1, color=col, lw=2.0, z=6)
        if arm:
            a0, a1 = PT(xc - 0.05, hp, dc), PT(xc - 0.05, hp - 0.02, dc)
            line(ax, p1, a0, color=col, lw=1.6, z=6)
            circle(ax, a1, 4.5, fc=col, ec="none", z=7)
        baseline([(p0[0] - 9, p0[1] - 2), (p0[0] + 9, p0[1] - 2),
                  (p0[0] + 9, p0[1] + 4), (p0[0] - 9, p0[1] + 4)], col=col)

    def block(xc, dc, wid, col=SLATE_L, hgt=0.070):
        s = 1.0 - 0.86 * dc
        p = PT(xc, 0, dc)
        bw, bh = wid * w * s, hgt * h * s
        body = [(p[0] - bw / 2, p[1]), (p[0] + bw / 2, p[1]),
                (p[0] + bw / 2, p[1] + bh), (p[0] - bw / 2, p[1] + bh)]
        mask_poly(body, col, 0.82)
        poly(ax, body, fc="none", ec="#41586E", lw=0.8, z=7)
        baseline([(p[0] - bw * 0.62, p[1] - 2), (p[0] + bw * 0.62, p[1] - 2),
                  (p[0] + bw * 0.62, p[1] + 4), (p[0] - bw * 0.62, p[1] + 4)])

    def cafe_set(xc, dc, col=OCHRE):
        s = 1.0 - 0.86 * dc
        p = PT(xc, 0, dc)
        tw, th = 0.032 * w * s, 0.026 * h * s
        mask_poly([(p[0] - tw, p[1] + th), (p[0] + tw, p[1] + th),
                   (p[0] + tw * 0.7, p[1] + th * 2.2), (p[0] - tw * 0.7, p[1] + th * 2.2)],
                  col, 0.82)
        line(ax, (p[0], p[1] + th), (p[0], p[1] + th * 2.2), color=col,
             lw=3.0 * s + 1, z=7)
        for sgn in (-1, 1):
            cx2 = p[0] + sgn * tw * 2.0
            mask_poly([(cx2 - tw * 0.42, p[1]), (cx2 + tw * 0.42, p[1]),
                       (cx2 + tw * 0.42, p[1] + th * 1.4),
                       (cx2 - tw * 0.42, p[1] + th * 1.4)], col, 0.82)
        baseline([(p[0] - tw * 2.8, p[1] - 2), (p[0] + tw * 2.8, p[1] - 2),
                  (p[0] + tw * 2.8, p[1] + 4), (p[0] - tw * 2.8, p[1] + 4)])

    def tree(xc, dc, col=SAGE, r=0.042):
        s = 1.0 - 0.86 * dc
        p = PT(xc, 0, dc)
        line(ax, p, (p[0], p[1] + 0.26 * h * s), color="#6B5334", lw=max(1.8, 3.6 * s), z=5)
        circle(ax, (p[0], p[1] + 0.30 * h * s), r * w * s, fc=col, ec="#5E7A62",
               lw=1.0, z=6)
        baseline([(p[0] - 0.018 * w * s, p[1] - 2), (p[0] + 0.018 * w * s, p[1] - 2),
                  (p[0] + 0.018 * w * s, p[1] + 4), (p[0] - 0.018 * w * s, p[1] + 4)])

    vehicle(0.44, 0.80, 0.075, BRICK)
    tree(0.82, 0.74, SAGE, 0.032)
    vehicle(0.60, 0.58, 0.090, BRICK)
    pole(0.170, 0.56, SLATE_L, 0.36, arm=True)
    tree(0.87, 0.50, SAGE, 0.036)
    vehicle(0.370, 0.36, 0.110, BRICK)
    cafe_set(0.80, 0.36, OCHRE)
    pole(0.855, 0.24, SLATE_L, 0.32, arm=False)
    block(0.200, 0.22, 0.038, SLATE_L, 0.062)
    vehicle(0.630, 0.15, 0.140, "#7E6A9B")
    tree(0.865, 0.10, SAGE, 0.044)
    vehicle(0.335, 0.10, 0.145, BRICK)


# --------------------------------------------------------------------------------------
# Panel -- inverse pinhole projection and ECW caliper
# --------------------------------------------------------------------------------------
def draw_pinhole(ax, x0, y0, w, h):
    T = fr(x0, y0, w, h)

    O = np.array([0.145, 0.115])
    EX = np.array([0.520, 0.175])
    EY = np.array([0.135, 0.285])

    def G(a, b):
        q = O + a * EX + b * EY
        return T(q[0], q[1])

    def UP(p, dz):
        return (p[0], p[1] + dz * h)

    corners = [G(0, 0), G(1, 0), G(1, 1), G(0, 1)]
    poly(ax, corners, fc="#F3F6F9", ec="#B6C2CE", lw=1.3, z=3)
    for k in range(1, 6):
        line(ax, G(k / 6, 0), G(k / 6, 1), color="#DDE4EB", lw=0.9, z=4)
        line(ax, G(0, k / 6), G(1, k / 6), color="#DDE4EB", lw=0.9, z=4)

    line(ax, G(0, 0), G(1.13, 0), color=INK, lw=1.8, z=5)
    arrowhead(ax, G(1.13, 0), math.atan2(EX[1], EX[0]), 18, 10, INK, 6)
    line(ax, G(0, 0), G(0, 1.13), color=INK, lw=1.8, z=5)
    arrowhead(ax, G(0, 1.13), math.atan2(EY[1], EY[0]), 18, 10, INK, 6)
    txt(ax, G(1.13, 0)[0] + 16, G(1.13, 0)[1] - 4, "$X$", size=8.0, weight=700,
        color=INK, ha="left", va="center", z=7)
    txt(ax, G(0, 1.25)[0] + 6, G(0, 1.25)[1] + 6, "$Y$", size=8.0, weight=700,
        color=INK, ha="center", va="center", z=7)
    txt(ax, (G(0, 0)[0] + G(1, 0)[0]) / 2 + 20, G(0, 0)[1] - 32,
        "metric ground plane (2 m grid)",
        size=6.0, weight=400, color=GREY, ha="center", va="center", z=6)

    def obstacle(a0, b0, a1, b1, dz, col="#5A6B7E", edge="#3E4C5C"):
        base = [G(a0, b0), G(a1, b0), G(a1, b1), G(a0, b1)]
        top = [UP(p, dz) for p in base]
        poly(ax, [base[0], base[1], top[1], top[0]], fc=col, ec=edge, lw=1.0, z=6)
        poly(ax, [base[1], base[2], top[2], top[1]], fc="#4A5869", ec=edge, lw=1.0, z=6)
        poly(ax, top, fc="#7E8DA0", ec=edge, lw=1.0, z=7)

    obstacle(0.46, 0.28, 0.60, 0.40, 0.080, BRICK, "#6E2F2B")
    obstacle(0.69, 0.28, 0.83, 0.40, 0.080, SLATE_L, "#41586E")
    obstacle(0.28, 0.58, 0.38, 0.68, 0.065, SLATE_L, "#41586E")

    p0, p1 = G(0.60, 0.34), G(0.69, 0.34)
    dim_arrow(ax, p0, p1, color=BRICK, lw=1.5, z=10)
    mid = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
    ang_ecw = math.degrees(math.atan2(EX[1], EX[0]))
    nx_, ny_ = -math.sin(math.radians(ang_ecw)) * 28, math.cos(math.radians(ang_ecw)) * 28
    txt(ax, mid[0] + nx_, mid[1] + ny_, "ECW", size=7.6, weight=700, color=BRICK,
        ha="center", va="center", z=12, rotation=ang_ecw,
        bbox=dict(boxstyle="round,pad=0.18", fc=PAPER, ec="none", alpha=0.95))

    C = np.array([0.32, 0.89])
    Cp = np.array(T(C[0], C[1]))
    t = 0.45
    image = [Cp + t * (np.array(p) - Cp) for p in corners]
    for p in corners:
        b = np.array(p)
        line(ax, tuple(Cp), tuple(Cp + t * (b - Cp)), color=OCHRE, lw=1.1, z=7, alpha=0.9)
        line(ax, tuple(Cp + t * (b - Cp)), tuple(b), color=OCHRE, lw=0.9, z=6,
             alpha=0.5, ls=(0, (4, 3)))
    poly(ax, [tuple(p) for p in image], fc="#E9F0F7", ec=NAVY, lw=1.3, z=8)
    for k in range(1, 4):
        line(ax, tuple(image[0] + (image[3] - image[0]) * (k / 4)),
             tuple(image[1] + (image[2] - image[1]) * (k / 4)),
             color="#A9C0D6", lw=0.7, z=9)
        line(ax, tuple(image[0] + (image[1] - image[0]) * (k / 4)),
             tuple(image[3] + (image[2] - image[3]) * (k / 4)),
             color="#A9C0D6", lw=0.7, z=9)
    circle(ax, Cp, 10, fc=INK, ec=PAPER, lw=1.3, z=10)
    txt(ax, Cp[0] + 22, Cp[1] + 18, "camera centre", size=6.4, weight=700, color=INK,
        ha="left", va="center", z=11)

    pix = image[1] + (image[2] - image[1]) * 0.40
    line(ax, tuple(Cp), tuple(pix), color=BRICK, lw=1.4, z=10)
    circle(ax, tuple(pix), 7.0, fc=PAPER, ec=BRICK, lw=1.5, z=11)
    txt(ax, pix[0] + 24, pix[1] + 4, "$(u, v)$", size=7.2, weight=700, color=BRICK,
        ha="left", va="center", z=12,
        bbox=dict(boxstyle="round,pad=0.14", fc=PAPER, ec="none", alpha=0.92))


# --------------------------------------------------------------------------------------
# Panel -- three-tier regulatory accessibility gauge
# --------------------------------------------------------------------------------------
def draw_gauge(ax, x0, y0, w, h):
    T = fr(x0, y0, w, h)
    R = 0.34 * h
    cx, cy = T(0.50, 0.44)
    VMAX = 3.0

    def ang(v):
        return 180.0 - (v / VMAX) * 180.0

    for v0, v1, col in ((0.0, 0.9, BRICK), (0.9, 1.5, OCHRE), (1.5, VMAX, PINE)):
        ax.add_patch(Wedge((cx, cy), R, ang(v1), ang(v0), width=R * 0.245,
                           facecolor=col, edgecolor=PAPER, linewidth=2.2, zorder=6))
    for v in (0.0, 0.9, 1.5, 3.0):
        a = math.radians(ang(v))
        ri, ro = R - R * 0.245, R + 10
        line(ax, (cx + ri * math.cos(a), cy + ri * math.sin(a)),
             (cx + ro * math.cos(a), cy + ro * math.sin(a)), color=INK, lw=1.3, z=8)
        dist = R + (38 if v in (0.9, 1.5) else 28)
        lp = (cx + dist * math.cos(a), cy + dist * math.sin(a))
        lab = {0.0: "0", 0.9: "0.9 m", 1.5: "1.5 m", 3.0: "3.0 m"}[v]
        ca = math.cos(a)
        ha = "left" if ca > 0.15 else ("right" if ca < -0.15 else "center")
        va = "bottom" if math.sin(a) > 0.6 else "center"
        txt(ax, lp[0], lp[1], lab, size=6.8, weight=700, color=INK, ha=ha,
            va=va, z=9)

    a = math.radians(ang(1.2))
    p1 = (cx + (R - R * 0.245 - 6) * math.cos(a), cy + (R - R * 0.245 - 6) * math.sin(a))
    line(ax, (cx, cy), p1, color=INK, lw=2.4, z=9)
    circle(ax, (cx, cy), 12, fc=INK, ec=PAPER, lw=1.6, z=10)
    txt(ax, cx, cy - 54, "ECW", size=7.8, weight=700, color=INK, ha="center",
        va="center", z=10)
    txt(ax, cx, cy - 106, "illustrative reading 1.2 m", size=6.0, weight=400,
        color=GREY, ha="center", va="center", z=10)


# --------------------------------------------------------------------------------------
# Tier-3 mini panels
# --------------------------------------------------------------------------------------
def draw_variance_tree(ax, x0, y0, w, h):
    ch = h - 112                      # chart band on top, captions underneath
    T = fr(x0, y0 + 112, w, ch)

    def box(u, v, wf, hf, label, fc, fs=5.4):
        a, b = T(u - wf / 2, v - hf / 2), T(u + wf / 2, v + hf / 2)
        rr(ax, a[0], a[1], b[0] - a[0], b[1] - a[1], r=6, fc=fc, ec="none", z=6)
        txt(ax, (a[0] + b[0]) / 2, (a[1] + b[1]) / 2, label, size=fs, weight=700,
            color=PAPER, ha="center", va="center", z=7, maxw=(b[0] - a[0]) - 8)

    def conn(u0, v0, u1, v1):
        line(ax, T(u0, v0), T(u1, v1), color="#9AA7B4", lw=1.0, z=5)

    box(0.50, 0.90, 0.62, 0.16, "street class (fixed)", NAVY, fs=5.4)
    conn(0.50, 0.82, 0.50, 0.76)
    line(ax, T(0.26, 0.76), T(0.76, 0.76), color="#9AA7B4", lw=1.0, z=5)
    conn(0.26, 0.76, 0.26, 0.68)
    conn(0.76, 0.76, 0.76, 0.68)
    box(0.26, 0.60, 0.46, 0.17, "street (random)", TEAL, fs=5.4)
    box(0.76, 0.60, 0.34, 0.17, "node", PINE, fs=5.4)
    conn(0.26, 0.515, 0.26, 0.44)
    line(ax, T(0.13, 0.44), T(0.39, 0.44), color="#9AA7B4", lw=1.0, z=5)
    conn(0.13, 0.44, 0.13, 0.36)
    conn(0.39, 0.44, 0.39, 0.36)
    box(0.13, 0.26, 0.22, 0.18, "ECW", PINE, fs=5.4)
    box(0.39, 0.26, 0.22, 0.18, "PC %", PINE, fs=5.4)
    conn(0.76, 0.515, 0.76, 0.45)
    box(0.76, 0.36, 0.32, 0.16, "residual", "#8894A2", fs=5.2)
    txt(ax, x0 + w / 2, y0 + 76, "hierarchical variance decomposition",
        size=6.0, weight=400, color=GREY, ha="center", va="center", z=7, maxw=w - 10)
    txt(ax, x0 + w / 2, y0 + 24, "ICC = 0.51", size=7.0, weight=700,
        color=NAVY, ha="center", va="center", z=7)


def draw_chord(ax, x0, y0, w, h):
    ch = h - 112
    T = fr(x0, y0 + 112, w, ch)
    c = T(0.5, 0.54)
    R = 0.43 * min(w, ch)
    groups = [(10, NAVY), (100, TEAL), (190, OCHRE), (280, BRICK)]
    span = 78.0
    for gi, (a0, col) in enumerate(groups):
        arc(ax, c, R, a0, a0 + span, ec=col, lw=2.8, z=6)
        for k in range(3):
            t0 = math.radians(a0 + 8 + k * 24)
            base = groups[(gi + 2) % 4][0]
            t3 = math.radians(base + 8 + ((gi + k) % 3) * 24 + 15)
            p0 = np.array([c[0] + R * math.cos(t0), c[1] + R * math.sin(t0)])
            p3 = np.array([c[0] + R * math.cos(t3), c[1] + R * math.sin(t3)])
            c1 = np.array(c) + (p0 - np.array(c)) * 0.45
            c2 = np.array(c) + (p3 - np.array(c)) * 0.45
            ax.add_patch(PathPatch(Path([tuple(p0), tuple(c1), tuple(c2), tuple(p3)],
                                        [Path.MOVETO, Path.CURVE4, Path.CURVE4,
                                         Path.CURVE4]),
                                   facecolor="none", edgecolor=col, linewidth=1.0,
                                   alpha=0.55, zorder=5, capstyle="round"))
    circle(ax, c, R * 0.50, fc=PAPER, ec="none", z=6)
    txt(ax, x0 + w / 2, y0 + 76, "chord diagram: obstacle footprint",
        size=6.0, weight=400, color=GREY, ha="center", va="center", z=7, maxw=w - 10)
    txt(ax, x0 + w / 2, y0 + 24, "5 street classes", size=7.0, weight=700,
        color=NAVY, ha="center", va="center", z=7)


def draw_moran(ax, x0, y0, w, h):
    ch = h - 112
    ax0, ax1 = x0 + 0.14 * w, x0 + 0.94 * w
    ay0, ay1 = y0 + 112 + 0.24 * ch, y0 + 112 + 0.80 * ch
    line(ax, (ax0, ay0), (ax1, ay0), color="#9AA7B4", lw=0.9, z=5)
    line(ax, (ax0, ay0), (ax0, ay1), color="#9AA7B4", lw=0.9, z=5)
    zx = ax0 + 0.50 * (ax1 - ax0)
    zy = ay0 + 0.50 * (ay1 - ay0)
    line(ax, (zx, ay0), (zx, ay1), color="#C3CCD6", lw=0.8, z=5, ls=(0, (4, 3)))
    line(ax, (ax0, zy), (ax1, zy), color="#C3CCD6", lw=0.8, z=5, ls=(0, (4, 3)))
    rng = np.random.default_rng(11)
    xs = rng.normal(0, 1, 320)
    ys = 0.38 * xs + rng.normal(0, 0.82, 320)
    for xv, yv in zip(xs, ys):
        px = ax0 + (xv + 2.4) / 4.8 * (ax1 - ax0)
        py = ay0 + (yv + 2.4) / 4.8 * (ay1 - ay0)
        if ax0 <= px <= ax1 and ay0 <= py <= ay1:
            circle(ax, (px, py), 3.2, fc=NAVY, ec="none", z=6, alpha=0.42)
    p0 = (ax0, ay0 + (0.38 * -2.4 + 2.4) / 4.8 * (ay1 - ay0))
    p1 = (ax1, ay0 + (0.38 * 2.4 + 2.4) / 4.8 * (ay1 - ay0))
    line(ax, p0, p1, color=BRICK, lw=1.5, z=7)
    txt(ax, ax1 - 4, p1[1] + 28, "OLS fit", size=5.6, weight=700, color=BRICK,
        ha="right", va="center", z=8)
    txt(ax, ax0 + 6, ay1 + 24, "standardised ECW", size=5.6, weight=400, color=GREY,
        ha="left", va="center", z=7)
    txt(ax, (ax0 + ax1) / 2, ay0 - 30, "spatial lag", size=5.6, weight=400,
        color=GREY, ha="center", va="center", z=7)
    txt(ax, x0 + w / 2, y0 + 76, "Moran's I scatter", size=6.0,
        weight=400, color=GREY, ha="center", va="center", z=7, maxw=w - 10)
    txt(ax, x0 + w / 2, y0 + 24, "I = 0.28  (p = 0.001)", size=7.0,
        weight=700, color=NAVY, ha="center", va="center", z=7)


def draw_tornado(ax, x0, y0, w, h):
    """Sensitivity of the impassable share: bars grow from a zero axis, labels left."""
    ch = h - 112
    ybase = y0 + 112
    axis = x0 + 0.62 * w
    scale = (0.24 * w) / 7.5
    items = [("vehicles", -7.2, BRICK), ("coverage", -1.2, BRICK),
             ("pitch", 0.7, TEAL), ("mask score", 0.4, TEAL)]
    n = len(items)
    top = ybase + 0.84 * ch
    step = (0.68 * ch) / (n - 1) if n > 1 else 0
    for i, (lab, val, col) in enumerate(items):
        cy = top - i * step
        txt(ax, x0 + 0.33 * w, cy, lab, size=5.6, weight=400, color=GREY,
            ha="right", va="center", z=7)
        line(ax, (x0 + 0.35 * w, cy), (x0 + 0.76 * w, cy),
             color="#E2E8F0", lw=0.8, z=4, ls=(0, (2, 2)))
        x_end = axis + val * scale
        bx0, bx1 = min(axis, x_end), max(axis, x_end)
        ax.add_patch(Rectangle((bx0, cy - 8), max(6.0, bx1 - bx0), 16,
                               facecolor=col, edgecolor="none", alpha=0.88, zorder=6))
        circle(ax, (x_end, cy), 6.5, fc=col, ec=PAPER, lw=1.1, z=7)
        txt(ax, x0 + 0.98 * w, cy, f"{val:+.1f} pp", size=5.6, weight=700,
            color=col, ha="right", va="center", z=7)
    line(ax, (axis, top + step * 0.35), (axis, top - (n - 1) * step - step * 0.35),
         color=INK, lw=1.2, z=7)
    txt(ax, x0 + w / 2, y0 + 76,
        "sensitivity of the impassable share (pp)", size=6.0, weight=400, color=GREY,
        ha="center", va="center", z=7, maxw=w - 10)
    txt(ax, x0 + w / 2, y0 + 24, "19.6–25.4 %  (n = 536)", size=7.0,
        weight=700, color=NAVY, ha="center", va="center", z=7)


def draw_tablet(ax, x0, y0, w, h):
    ch = h - 112
    bw, bh = 0.56 * w, 0.92 * ch
    bx, by = x0 + (w - bw) / 2, y0 + 112 + 0.04 * ch
    rr(ax, bx - 8, by - 8, bw + 16, bh + 16, r=15, fc="#C6D0DA", ec="#A3AEBB",
       lw=1.1, z=5)
    rr(ax, bx, by, bw, bh, r=12, fc="#2E3A4A", ec="#1E2833", lw=1.2, z=6)
    sw, sh = bw - 16, bh - 28
    sx, sy = bx + 8, by + 16
    rr(ax, sx, sy, sw, sh, r=6, fc=PAPER, ec="none", z=7)
    hh = 0.22 * sh
    rr(ax, sx, sy + sh - hh, sw, hh, r=6, fc=NAVY, ec="none", z=8)
    ax.add_patch(Rectangle((sx, sy + sh - hh), sw, hh * 0.45, facecolor=NAVY,
                           edgecolor="none", zorder=8))
    txt(ax, sx + 12, sy + sh - hh * 0.50, "ROW audit", size=4.6, weight=700, color=PAPER,
        ha="left", va="center", z=9, minsize=3.6)
    circle(ax, (sx + sw - 14, sy + sh - hh * 0.50), 3.8, fc="none", ec=PAPER, lw=1.0, z=9)
    row_h = 0.19 * sh
    for i, (lab, col) in enumerate((("Accessible", PINE), ("Restricted", OCHRE),
                                    ("Impassable", BRICK))):
        ry = sy + sh * (0.62 - i * 0.235)
        rr(ax, sx + 8, ry - row_h / 2, sw - 16, row_h, r=5, fc="#F5F7FA", ec="#E2E7ED",
           lw=0.8, z=9)
        circle(ax, (sx + 18, ry), 4.8, fc=col, ec="none", z=10)
        txt(ax, sx + 30, ry, lab, size=4.3, weight=700, color=INK, ha="left",
            va="center", z=10, minsize=3.6)
        circle(ax, (sx + sw - 18, ry), 3.0, fc="#C6CFD8", ec="none", z=10)
    circle(ax, (bx + bw / 2, by + 8), 3.8, fc="#3E4B5C", ec="none", z=9)
    txt(ax, x0 + w / 2, y0 + 76, "Digital ROW screening", size=6.0,
        weight=700, color=INK, ha="center", va="center", z=8, maxw=w - 10)
    txt(ax, x0 + w / 2, y0 + 24, "prioritises field checks", size=5.8,
        weight=400, color=GREY, ha="center", va="center", z=8, maxw=w - 10)


def draw_curb_ramp(ax, x0, y0, w, h):
    ch = h - 112
    S = min(w * 0.68, ch * 1.42)
    cx = x0 + w * 0.50
    cy0 = y0 + 112 + ch * 0.10

    def iso(u, v, dz=0.0):
        return (cx + (u - v) * 0.52 * S,
                cy0 + (u + v) * 0.32 * S + dz * S)

    poly(ax, [iso(0, 0), iso(1, 0), iso(1, 1), iso(0, 1)], fc="#E7E2D6",
         ec="#C6BFAE", lw=1.0, z=4)
    poly(ax, [iso(0, 0), iso(1, 0), iso(1, 0.34), iso(0, 0.34)], fc="#D4CEBE",
         ec="#BEB7A6", lw=1.0, z=5)
    poly(ax, [iso(0.26, 0.36), iso(0.74, 0.36), iso(0.74, 0.86), iso(0.26, 0.86)],
         fc="#C7CCD3", ec="#98A2AD", lw=1.0, z=6)
    for i in range(4):
        for j in range(6):
            circle(ax, iso(0.32 + i * 0.12, 0.42 + j * 0.068), 2.8, fc=OCHRE,
                   ec="none", z=7, alpha=0.88)
    for u in (0.38, 0.58):
        a, b = iso(u, 0.88), iso(u, 1.00)
        line(ax, a, b, color=PINE, lw=1.6, z=8)
        arrowhead(ax, b, math.atan2(b[1] - a[1], b[0] - a[0]), 14, 7, PINE, 9)
    txt(ax, x0 + w / 2, y0 + 76, "Barrier-free curb ramp", size=6.0,
        weight=700, color=INK, ha="center", va="center", z=8, maxw=w - 10)
    txt(ax, x0 + w / 2, y0 + 24, "ADA / EAA ≥ 1.5 m clear path", size=5.8, weight=700,
        color=PINE, ha="center", va="center", z=8, maxw=w - 10)


def draw_bollards(ax, x0, y0, w, h):
    ch = h - 112
    S = min(w * 0.68, ch * 1.42)
    cx = x0 + w * 0.50
    cy0 = y0 + 112 + ch * 0.08

    def iso(u, v, dz=0.0):
        return (cx + (u - v) * 0.52 * S,
                cy0 + (u + v) * 0.32 * S + dz * S)

    poly(ax, [iso(0, 0.05), iso(1, 0.05), iso(1, 0.62), iso(0, 0.62)],
         fc="#E7E2D6", ec="#C6BFAE", lw=1.0, z=4)
    poly(ax, [iso(0, 0.62), iso(1, 0.62), iso(1, 1.0), iso(0, 1.0)],
         fc="#C6CCD3", ec="#AEB6BF", lw=1.0, z=4)
    poly(ax, [iso(0.22, 0.74), iso(0.74, 0.74), iso(0.74, 0.96), iso(0.22, 0.96)],
         fc="#8E9AA6", ec="#6E7A86", lw=1.0, z=5)
    poly(ax, [iso(0.30, 0.78), iso(0.66, 0.78), iso(0.66, 0.92), iso(0.30, 0.92)],
         fc="#7C8894", ec="none", z=6)
    for i, u in enumerate((0.30, 0.52, 0.74)):
        b = iso(u, 0.56)
        hgt = (0.30 - i * 0.015) * S
        rr(ax, b[0] - 10, b[1] - 4, 20, hgt, r=8, fc="#8C5E44", ec="#6A4833",
           lw=1.0, z=7)
        rr(ax, b[0] - 10, b[1] + hgt - 16, 20, 16, r=6, fc="#F2F4F6",
           ec="#C8D0D8", lw=0.9, z=8)
    txt(ax, x0 + w / 2, y0 + 76, "Curb-parking regulation", size=6.0,
        weight=700, color=INK, ha="center", va="center", z=8, maxw=w - 10)
    txt(ax, x0 + w / 2, y0 + 24, "−48.8 % impassable (23.3 → 11.9 %)", size=5.8,
        weight=700, color=BRICK, ha="center", va="center", z=8, maxw=w - 10)


def draw_utility_reloc(ax, x0, y0, w, h):
    """Sidewalk furnishing-strip consolidation: relocates utility boxes & cafe tables."""
    ch = h - 112
    S = min(w * 0.68, ch * 1.42)
    cx = x0 + w * 0.50
    cy0 = y0 + 112 + ch * 0.08

    def iso(u, v, dz=0.0):
        return (cx + (u - v) * 0.52 * S,
                cy0 + (u + v) * 0.32 * S + dz * S)

    # Sidewalk base + dedicated curb furnishing strip
    poly(ax, [iso(0, 0.05), iso(1, 0.05), iso(1, 0.68), iso(0, 0.68)],
         fc="#E7E2D6", ec="#C6BFAE", lw=1.0, z=4)
    poly(ax, [iso(0, 0.68), iso(1, 0.68), iso(1, 0.96), iso(0, 0.96)],
         fc="#D5CFC0", ec="#B8B09E", lw=1.0, z=4)
    # Green clear-corridor band along the main pedestrian path
    poly(ax, [iso(0.06, 0.14), iso(0.94, 0.14), iso(0.94, 0.56), iso(0.06, 0.56)],
         fc=PINE, ec="none", lw=0.0, z=5, alpha=0.16)
    line(ax, iso(0.06, 0.62), iso(0.94, 0.62), color=OCHRE, lw=1.4, z=6, ls=(0, (4, 3)))

    # Consolidated utility cabinet + street furniture in outer strip
    cab_b = [iso(0.22, 0.72), iso(0.44, 0.72), iso(0.44, 0.90), iso(0.22, 0.90)]
    cab_t = [iso(0.22, 0.72, 0.18), iso(0.44, 0.72, 0.18),
             iso(0.44, 0.90, 0.18), iso(0.22, 0.90, 0.18)]
    poly(ax, [cab_b[0], cab_b[1], cab_t[1], cab_t[0]], fc=SLATE_L, ec="#41586E", lw=1.0, z=7)
    poly(ax, [cab_b[1], cab_b[2], cab_t[2], cab_t[1]], fc="#7E95AB", ec="#41586E", lw=1.0, z=7)
    poly(ax, cab_t, fc="#C4D4E2", ec="#41586E", lw=1.0, z=8)

    # Street tree / planter in outer furnishing strip
    tb = iso(0.72, 0.78)
    line(ax, tb, (tb[0], tb[1] + 0.13 * S), color="#6B5334", lw=2.6, z=7)
    circle(ax, (tb[0], tb[1] + 0.17 * S), 0.068 * S, fc=SAGE, ec="#5E7A62", lw=1.0, z=8)

    # Clear flow arrow along unobstructed pedestrian zone
    a0, a1 = iso(0.18, 0.35), iso(0.84, 0.35)
    line(ax, a0, a1, color=PINE, lw=1.8, z=7)
    arrowhead(ax, a1, math.atan2(a1[1] - a0[1], a1[0] - a0[0]), 16, 8, PINE, 8)

    txt(ax, x0 + w / 2, y0 + 76, "Utility & furniture setback", size=6.0,
        weight=700, color=INK, ha="center", va="center", z=8, maxw=w - 10)
    txt(ax, x0 + w / 2, y0 + 24, "consolidates 24.8 % non-vehicle blocks", size=5.8,
        weight=700, color=TEAL, ha="center", va="center", z=8, maxw=w - 10)


# --------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------
def build_figure():
    fig = plt.figure(figsize=(PX / DPI, PX / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, PX)
    ax.set_ylim(0, PX)
    ax.axis("off")
    ax.set_facecolor(PAPER)
    fig.patch.set_facecolor(PAPER)

    M = 104
    CW = PX - 2 * M
    GUT = 220
    COLW = (CW - GUT) / 2
    XL, XR = M, M + COLW + GUT
    XR_END = M + CW

    T1_TOP, T1_BOT = 3986, 2786
    T2_TOP, T2_BOT = 2566, 1376
    T3_TOP, T3_BOT = 1136, 96

    # Colored dashed tier enclosures with tier labels sitting directly on the top dashed border
    tier_frame(ax, XL, XR_END, T1_BOT, T1_TOP, "TOP TIER  ·  STAGE 1", NAVY, "#F6F9FC")
    tier_frame(ax, XL, XR_END, T2_BOT, T2_TOP, "MIDDLE TIER  ·  STAGES 2 & 3", TEAL, "#F5FAF8")
    tier_frame(ax, XL, XR_END, T3_BOT, T3_TOP, "BOTTOM TIER  ·  STAGES 4 & 5", PINE, "#F6FAF6")

    # Tier 1 is a single unified full-width card (Stage 1); Tiers 2 & 3 have two columns
    rr(ax, XL, T1_BOT, CW, T1_TOP - T1_BOT, r=26, fc=PAPER, ec=HAIR, lw=1.6,
       z=2, shadow=True)
    for x in (XL, XR):
        rr(ax, x, T2_BOT, COLW, T2_TOP - T2_BOT, r=26, fc=PAPER, ec=HAIR, lw=1.6,
           z=2, shadow=True)
        rr(ax, x, T3_BOT, COLW, T3_TOP - T3_BOT, r=26, fc=PAPER, ec=HAIR, lw=1.6,
           z=2, shadow=True)

    # ===================== TIER 1 (Unified Stage 1) ==============================
    card_header(
        ax, XL, T1_TOP, XR_END, 1, NAVY,
        "Corridor geometry from municipal GIS & panoramic SVI acquisition",
        "building footprints, street axes and 30 m transects · 640 × 640 px panoramas at 50 m spacing · 16 H3 cells · 531 Konak + 26 audit nodes (N = 557)",
        icons=["qgis", "python", "gsv", "h3"],
    )
    draw_street_canyon(ax, XL + 40, T1_BOT + 225, COLW - 40, (T1_TOP - 170) - (T1_BOT + 225))
    circle(ax, (XL + 56, T1_BOT + 184), 8, fc=OCHRE, ec="none", z=8)
    txt(ax, XL + 78, T1_BOT + 184, "SVI camera node; rays cast 30 m to each side",
        size=6.4, weight=400, color=GREY, ha="left", va="center", z=8, maxw=COLW - 90)
    txt(ax, XL + 44, T1_BOT + 126,
        "The first footprint hit gives the facade distance",
        size=6.4, weight=400, color=GREY, ha="left", va="center", z=8, maxw=COLW - 50)
    txt(ax, XL + 44, T1_BOT + 66, r"$W_c = d_L + d_R$", size=8.0, weight=700,
        color=NAVY, ha="left", va="center", z=8)
    txt(ax, XL + COLW - 10, T1_BOT + 66, "nominal width is not walkable width",
        size=6.4, weight=400, color=GREY, ha="right", va="center", z=8,
        maxw=0.58 * COLW)

    # Vertical divider between GIS corridor (left) and SVI / H3 acquisition (right)
    mid_t1 = (XL + COLW + XR) / 2
    line(ax, (mid_t1, T1_BOT + 46), (mid_t1, T1_TOP - 175), color=HAIR, lw=1.3, z=5)

    draw_camera_rig(ax, XR - 10, T1_BOT + 200, 0.46 * (COLW - 20), (T1_TOP - 175) - (T1_BOT + 200))
    draw_node_map(ax, XR - 10 + 0.48 * (COLW - 20), T1_BOT + 200, 0.52 * (COLW - 20),
                  (T1_TOP - 175) - (T1_BOT + 200))
    txt(ax, XR - 10 + 0.23 * (COLW - 20), T1_BOT + 144,
        "cardinal along-street views (±90°)", size=6.4, weight=400, color=GREY,
        ha="center", va="center", z=8, maxw=0.45 * (COLW - 20))
    txt(ax, XR - 10 + 0.23 * (COLW - 20), T1_BOT + 86,
        "−10° downward tilt pitch", size=6.4, weight=400, color=GREY,
        ha="center", va="center", z=8, maxw=0.45 * (COLW - 20))
    txt(ax, XR - 10 + 0.74 * (COLW - 20), T1_BOT + 144, "H3 node grid", size=6.4,
        weight=400, color=GREY, ha="center", va="center", z=8, maxw=0.50 * (COLW - 20))
    txt(ax, XR - 10 + 0.74 * (COLW - 20), T1_BOT + 86,
        "50 m node spacing", size=6.4, weight=400, color=GREY, ha="center",
        va="center", z=8, maxw=0.50 * (COLW - 20))
    line(ax, (XR - 10 + 0.465 * (COLW - 20), T1_BOT + 60),
         (XR - 10 + 0.465 * (COLW - 20), T1_TOP - 175),
         color=HAIR, lw=1.2, z=5)

    # ===================== TIER 2 =================================================
    card_header(ax, XL, T2_TOP, XL + COLW, 2, BRICK,
                "Zero-shot panoptic perception (SAM 3)",
                "open-vocabulary instance masks, no fine-tuning · 12 obstacle classes",
                icons=["meta", "pytorch"])
    draw_segment_scene(ax, XL + 40, T2_BOT + 510, COLW - 80, (T2_TOP - 170) - (T2_BOT + 510))
    leg = [("Vehicles", BRICK), ("Utility infrastructure", SLATE_L),
           ("Street furniture", OCHRE), ("Vegetation", SAGE)]
    for i, (lab, col) in enumerate(leg):
        yy = T2_BOT + 458 - i * 68
        swatch(ax, XL + 64, yy, col, r=16, alpha=0.90)
        txt(ax, XL + 102, yy, lab, size=7.2, weight=400, color=INK, ha="left",
            va="center", z=8)
    swatch(ax, XL + 64, T2_BOT + 458 - 4 * 68, INK, r=16, hatch=True)
    txt(ax, XL + 102, T2_BOT + 458 - 4 * 68, "Ground-contact baseline footprint",
        size=7.2, weight=400, color=INK, ha="left", va="center", z=8)
    txt(ax, XL + 44, T2_BOT + 104,
        "Masks are projected onto the ground plane; the contact footprint of an",
        size=6.4, weight=400, color=GREY, ha="left", va="center", z=8, maxw=COLW - 90)
    txt(ax, XL + 44, T2_BOT + 54, "object is what subtracts from the pedestrian zone",
        size=6.4, weight=400, color=GREY, ha="left", va="center", z=8, maxw=COLW - 90)

    card_header(ax, XR, T2_TOP, XR_END, 3, TEAL,
                "Inverse photogrammetry & ECW derivation",
                "facade-anchored camera calibration, then ray projection onto the ground plane",
                icons=["numpy", "scipy", "opencv"])
    draw_pinhole(ax, XR + 30, T2_BOT + 240, 0.55 * (COLW - 60), (T2_TOP - 170) - (T2_BOT + 240))
    draw_gauge(ax, XR + 30 + 0.58 * (COLW - 60), T2_BOT + 565, 0.42 * (COLW - 60),
               (T2_TOP - 170) - (T2_BOT + 565))
    fx = XR + 30 + 0.58 * (COLW - 60)
    fw = 0.42 * (COLW - 60)
    rr(ax, fx, T2_BOT + 280, fw, 245, r=14, fc="#F7F9FB", ec=HAIR, lw=1.2, z=6)
    txt(ax, fx + 22, T2_BOT + 476, r"$\mathrm{ECW} = \mathrm{min}_{s}\, W_{\mathrm{free}}(s)$",
        size=8.0, weight=700, color=TEAL, ha="left", va="center", z=8, maxw=fw - 44)
    txt(ax, fx + 22, T2_BOT + 402, r"$\mathrm{PC} = (W_{\mathrm{nom}} - \mathrm{ECW})\,/\,W_{\mathrm{nom}}$",
        size=7.2, weight=700, color=BRICK, ha="left", va="center", z=8, maxw=fw - 44)
    txt(ax, fx + 22, T2_BOT + 330, "bottleneck of the continuous free path",
        size=5.3, weight=400, color=GREY, ha="left", va="center", z=8, maxw=fw - 44)
    tiers = [("Impassable", r"$\mathrm{ECW} < 0.9$ m", BRICK, "23.0 %"),
             ("Restricted", r"$0.9 - 1.5$ m", OCHRE, "15.1 %"),
             ("Accessible", r"$\mathrm{ECW} \geq 1.5$ m", PINE, "61.9 %")]
    colw = (COLW - 60 - 36) / 3
    for i, (name, rng, col, share) in enumerate(tiers):
        cx0 = XR + 30 + i * (colw + 18)
        rr(ax, cx0, T2_BOT + 96, colw, 126, r=10, fc="#F8FAFC", ec=HAIR, lw=1.0, z=5)
        swatch(ax, cx0 + 26, T2_BOT + 182, col, r=13)
        txt(ax, cx0 + 50, T2_BOT + 182, name, size=6.4, weight=700, color=INK,
            ha="left", va="center", z=8)
        txt(ax, cx0 + colw - 18, T2_BOT + 182, share, size=6.6, weight=700,
            color=col, ha="right", va="center", z=8)
        txt(ax, cx0 + 50, T2_BOT + 128, rng, size=5.8, weight=400, color=GREY,
            ha="left", va="center", z=8)
    txt(ax, XR + 30, T2_BOT + 50, "ADA / EAA-style thresholds · shares of n = 557 nodes",
        size=6.0, weight=400, color=GREY, ha="left", va="center", z=8, maxw=COLW - 60)

    # ===================== TIER 3 =================================================
    card_header(ax, XL, T3_TOP, XL + COLW, 4, PINE,
                "Spatial modelling & validation",
                "variance decomposition, network topology and spatial autocorrelation",
                icons=["pysal", "networkx"])
    pw = (COLW - 80 - 60) / 2
    ph = ((T3_TOP - 170) - (T3_BOT + 36) - 30) / 2
    draw_variance_tree(ax, XL + 40, T3_TOP - 170 - ph, pw, ph)
    draw_chord(ax, XL + 40 + pw + 60, T3_TOP - 170 - ph, pw, ph)
    draw_moran(ax, XL + 40, T3_BOT + 36, pw, ph)
    draw_tornado(ax, XL + 40 + pw + 60, T3_BOT + 36, pw, ph)
    line(ax, (XL + 40 + pw + 30, T3_BOT + 30), (XL + 40 + pw + 30, T3_TOP - 160),
         color=HAIR, lw=1.2, z=5)
    line(ax, (XL + 36, T3_BOT + 36 + ph + 15), (XL + COLW - 36, T3_BOT + 36 + ph + 15),
         color=HAIR, lw=1.2, z=5)

    card_header(ax, XR, T3_TOP, XR_END, 5, OCHRE,
                "Municipal planning & interventions",
                "phantom capacity translated into four targeted right-of-way interventions",
                icons=["cityhall", "wheelchair"])
    draw_tablet(ax, XR + 40, T3_TOP - 170 - ph, pw, ph)
    draw_bollards(ax, XR + 40 + pw + 60, T3_TOP - 170 - ph, pw, ph)
    draw_utility_reloc(ax, XR + 40, T3_BOT + 36, pw, ph)
    draw_curb_ramp(ax, XR + 40 + pw + 60, T3_BOT + 36, pw, ph)
    line(ax, (XR + 40 + pw + 30, T3_BOT + 30), (XR + 40 + pw + 30, T3_TOP - 160),
         color=HAIR, lw=1.2, z=5)
    line(ax, (XR + 36, T3_BOT + 36 + ph + 15), (XR_END - 36, T3_BOT + 36 + ph + 15),
         color=HAIR, lw=1.2, z=5)

    # ===================== CONDUITS ==============================================
    g1 = ((T1_BOT - 20) + (T2_TOP + 32)) / 2   # channel midpoint between tier 1 and tier 2 frames
    g2 = ((T2_BOT - 20) + (T3_TOP + 32)) / 2   # channel midpoint between tier 2 and tier 3 frames
    cxl = XL + 0.50 * COLW
    cxr = XR + 0.50 * COLW

    # 1 -> 2 and 1 -> 3 (ingestion -> perception / homography)
    circle(ax, (cxl, T1_BOT - 4), 6, fc=NAVY, ec=PAPER, lw=1.2, z=9)
    conduit(ax, [(cxl, T1_BOT - 4), (cxl, g1), (cxl, T2_TOP + 6)], color=NAVY, lw=2.8,
            head=26, z=8, radius=40)
    circle(ax, (cxr, T1_BOT - 4), 6, fc=NAVY, ec=PAPER, lw=1.2, z=9)
    conduit(ax, [(cxr, T1_BOT - 4), (cxr, g1), (cxr, T2_TOP + 6)], color=NAVY, lw=2.8,
            head=26, z=8, radius=40)

    # 2 -> 3 (masks -> homography within tier 2)
    ymid2 = T2_BOT + 0.54 * (T2_TOP - T2_BOT)
    circle(ax, (XL + COLW + 8, ymid2), 6.5, fc=BRICK, ec=PAPER, lw=1.2, z=10)
    conduit(ax, [(XL + COLW + 8, ymid2), (XR - 8, ymid2)], color=BRICK, lw=2.8,
            head=26, z=9)

    # 3 -> 4 (ECW -> spatial modelling: tier 2 right -> tier 3 left)
    circle(ax, (XR + 0.30 * COLW, T2_BOT - 4), 6, fc=TEAL, ec=PAPER, lw=1.2, z=9)
    conduit(ax, [(XR + 0.30 * COLW, T2_BOT - 4), (XR + 0.30 * COLW, g2),
                 (cxl, g2), (cxl, T3_TOP + 6)], color=TEAL, lw=2.8, head=26, z=8, radius=44)

    # 4 -> 5 (modelling -> interventions within tier 3)
    ymid3 = T3_BOT + 0.50 * (T3_TOP - T3_BOT)
    circle(ax, (XL + COLW + 8, ymid3), 6.5, fc=PINE, ec=PAPER, lw=1.2, z=10)
    conduit(ax, [(XL + COLW + 8, ymid3), (XR - 8, ymid3)], color=PINE, lw=2.8,
            head=26, z=9)

    # conduit data labels
    conduit_label(ax, cxl, g1 + 6, "footprints", NAVY, 0, z=10)
    conduit_label(ax, cxr, g1 + 6, "panoramas", NAVY, 0, z=10)
    conduit_label(ax, (XL + COLW + XR) / 2, ymid2 + 84, "instance\nmasks", BRICK, 0, z=10)
    conduit_label(ax, (cxl + XR + 0.30 * COLW) / 2, g2, "ECW · PC %", TEAL, 0, z=10)
    conduit_label(ax, (XL + COLW + XR) / 2, ymid3 + 84, "clearance\ntiers", PINE, 0, z=10)

    apply_fits(fig)
    return fig


def main(out_paths):
    fig = build_figure()
    for p in out_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fig.savefig(p, dpi=DPI, facecolor=PAPER, edgecolor="none")
        print(f"wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    p = os.path.abspath(__file__)
    cur = os.path.dirname(p)
    root = cur
    while cur and os.path.dirname(cur) != cur:
        if os.path.exists(os.path.join(cur, "submission")):
            root = cur
            break
        cur = os.path.dirname(cur)
    rel = "submission/TiGIS_R1_revision"
    outs = [
        os.path.join(root, rel, "05_manuscript_R1/figures/fig01_workflow.png"),
        os.path.join(root, rel, "05_manuscript_R1/markedup/figures/fig01_workflow.png"),
        os.path.join(root, rel,
                     "SUBMISSION_R1_PACKAGE/02_latex_source/figures/fig01_workflow.png"),
        os.path.join(root, rel,
                     "07_repository_gitlab_figshare/figures/fig01_workflow.png"),
    ]
    if len(sys.argv) > 1:
        outs = sys.argv[1:]
    main(outs)
