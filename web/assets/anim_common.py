"""Shared pieces for the *_v2 Manim scenes: a left-hand computation pane whose equations are
written top to bottom, one substitution per line, with every '=' (or continuation '+' / '×')
in one aligned column — and a thin step bar for the top of the frame.

No LaTeX is installed, so everything is manim Text. Each equation line is ONE Text object (one
shared baseline); it is shifted sideways so its '=' glyph lands on the pane's column, and
vertically so that '=' glyph sits on the row's centre line.
"""
from manim import *

MATH = "Cambria Math"  # Consolas has no ℝ ∈ ⌊⌋ ∏ ₖ glyphs; every formula is set in this
MONO = "Consolas"
INK = "#E7EBF3"
INK2 = "#AEB7CA"
MUTED = "#7C869C"
DONE = "#3B4252"
RULE = "#232B3B"
WARM = "#F4C26B"
VIOLET = "#C2B4FF"
PANE_FILL = "#0B0F17"


class EqPane:
    """Row specs:
        ("h", text[, color])           header
        ("c", text[, color])           comment (smaller, muted)
        ("e", lhs, rhs[, color])       lhs = rhs   ('' lhs = continuation line starting with '=')
        (op, rhs[, color])             continuation starting with op, op in {"+", "×", "−"}
        ("gap",)                       half a row of space
    """

    def __init__(self, left, right, top, size=19, line_h=0.44):
        self.left, self.right, self.top = left, right, top
        self.size, self.line_h = size, line_h
        self.items = []  # [(mobject or None, height used)]
        self.y = top
        self.eq_x = None
        probe = Text("x = 1", font=MATH, font_size=size)
        self.gap = probe[1].get_left()[0] - probe[0].get_right()[0]

    # ---------- layout ----------
    def _lhs_w(self, s):
        return Text(s, font=MATH, font_size=self.size).width if s else 0.0

    def _eq_x_for(self, rows):
        ws = [self._lhs_w(r[1]) for r in rows if r[0] == "e"]
        return self.left + 0.18 + (max(ws) if ws else 0.0) + self.gap

    def _fit(self, t, anchor_x):
        if t.get_right()[0] > self.right:
            f = (self.right - anchor_x) / max(1e-6, t.get_right()[0] - anchor_x)
            t.scale(f, about_point=np.array([anchor_x, t.get_y(), 0]))

    def _make(self, row, y):
        kind = row[0]
        lh = self.line_h
        if kind == "gap":
            return None, lh * 0.5
        if kind in ("h", "c"):
            color = row[2] if len(row) > 2 else (WARM if kind == "h" else MUTED)
            size = self.size + 1 if kind == "h" else self.size - 4
            t = Text(row[1], font=MATH, font_size=size, color=color)
            x0 = self.left + (0.1 if kind == "h" else 0.18)
            t.move_to([x0, y - lh / 2, 0], aligned_edge=LEFT)
            self._fit(t, x0)
            return t, lh * (1.2 if kind == "h" else 0.85)
        if kind == "e":
            lhs, rhs = row[1], row[2]
            color = row[3] if len(row) > 3 else INK
            s = f"{lhs} = {rhs}" if lhs else f"= {rhs}"
            lw = self._lhs_w(lhs)
        else:  # continuation with an operator
            rhs = row[1]
            color = row[2] if len(row) > 2 else INK
            s = f"{kind} {rhs}"
            lhs, lw = "", 0.0
        t = Text(s, font=MATH, font_size=self.size, color=color)
        left = self.eq_x - lw - self.gap if lhs else self.eq_x
        t.shift(RIGHT * (left - t.get_left()[0]))
        op = min(t.submobjects, key=lambda m: abs(m.get_left()[0] - self.eq_x))
        t.shift(UP * ((y - lh / 2) - op.get_y()))
        self._fit(t, op.get_left()[0])
        return t, lh

    # ---------- public: each returns a list of animations ----------
    def set(self, rows, eq_x=None):
        old = VGroup(*[m for m, _ in self.items if m is not None])
        self.items, self.y = [], self.top
        self.eq_x = eq_x if eq_x is not None else self._eq_x_for(rows)
        new = self._add(rows)
        anims = [FadeOut(old)] if len(old) else []
        return anims + ([FadeIn(new)] if len(new) else [])

    def append(self, rows):
        if self.eq_x is None:
            self.eq_x = self._eq_x_for(rows)
        new = self._add(rows)
        return [FadeIn(new)] if len(new) else []

    def pop(self, n):
        out = VGroup()
        for _ in range(min(n, len(self.items))):
            m, h = self.items.pop()
            self.y += h
            if m is not None:
                out.add(m)
        return [FadeOut(out)] if len(out) else []

    def clear(self):
        return self.pop(len(self.items))

    def _add(self, rows):
        g = VGroup()
        for row in rows:
            m, h = self._make(row, self.y)
            self.items.append((m, h))
            self.y -= h
            if m is not None:
                g.add(m)
        return g


def pane_frame(left, right, top, bottom, title="computation"):
    box = RoundedRectangle(corner_radius=0.08, width=right - left, height=top - bottom,
                           fill_color=PANE_FILL, fill_opacity=1, stroke_color=RULE, stroke_width=1.5)
    box.move_to([(left + right) / 2, (top + bottom) / 2, 0])
    lab = Text(title, font=MONO, font_size=13, color=DONE)
    lab.move_to(box.get_corner(UR) + np.array([-0.12, -0.14, 0]), aligned_edge=UR)
    return VGroup(box, lab)


import json as _json
import os as _os

TOY_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "toy-data.json")


def load_toy():
    """luna_v3.html's toy dataset + pipeline output, exported by export_toy.mjs"""
    with open(TOY_PATH, encoding="utf-8") as f:
        return _json.load(f)


class Proj3D:
    """Orthographic view of the box lo..hi (data coords), z3 up — the same projection as the
    page's 3D grid: yaw turns about the vertical axis, pitch tilts it toward the viewer."""

    def __init__(self, lo, hi, center, scale, yaw=-0.62, pitch=0.42):
        self.lo, self.hi = list(lo), list(hi)
        self.center, self.scale = np.array(center, dtype=float), scale
        self.yaw, self.pitch = yaw, pitch

    def unit(self, X, Y, Z, yaw=None):
        yaw = self.yaw if yaw is None else yaw
        cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(self.pitch), np.sin(self.pitch)
        u, v = X * cy - Y * sy, X * sy + Y * cy
        return self.center + self.scale * np.array([u, Z * cp + v * sp, 0.0]), v * cp - Z * sp

    def nrm(self, v, d):
        return 2 * (v - self.lo[d]) / (self.hi[d] - self.lo[d]) - 1

    def data(self, z, yaw=None):
        return self.unit(self.nrm(z[0], 0), self.nrm(z[1], 1), self.nrm(z[2], 2), yaw)

    def back(self, ax, sg, yaw=None):
        n = [0, 0, 0]
        n[ax] = sg
        return self.unit(*n, yaw=yaw)[1] - self.unit(0, 0, 0, yaw=yaw)[1] > 0

    def face_lines(self, m, yaw=None, color="#2F3A50", width=1.0):
        """grid lines (m cells per side) on the three faces at the back"""
        g = VGroup()
        t = lambda k: -1 + 2 * k / m
        for ax in range(3):
            for sg in (-1, 1):
                if not self.back(ax, sg, yaw):
                    continue
                o = [d for d in range(3) if d != ax]
                for k in range(m + 1):
                    for d1, d2 in ((o[0], o[1]), (o[1], o[0])):
                        A, B = [0, 0, 0], [0, 0, 0]
                        A[ax] = B[ax] = sg
                        A[d1] = B[d1] = t(k)
                        A[d2], B[d2] = -1, 1
                        g.add(Line(self.unit(*A, yaw=yaw)[0], self.unit(*B, yaw=yaw)[0], color=color,
                                   stroke_width=width * (1.6 if k % m == 0 else 1.0)))
        return g

    def wire_box(self, lo, hi, yaw=None, color=WARM, width=3):
        """12 edges of the box lo..hi given in [-1, 1] coords"""
        c = lambda i: self.unit(hi[0] if i & 1 else lo[0], hi[1] if i & 2 else lo[1], hi[2] if i & 4 else lo[2], yaw=yaw)[0]
        return VGroup(*[Line(c(i), c(j), color=color, stroke_width=width) for i, j in
                        [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]])


class StepBar(VGroup):
    """Tokenize › Embed › … with the active step in WARM, past steps INK2, future steps DONE."""

    def __init__(self, names, y, size=17):
        super().__init__()
        # one Text for the whole bar (one shared baseline); each name is sliced out of it by
        # its count of visible glyphs, since spaces never become submobjects
        full = Text("   ›   ".join(names), font=MATH, font_size=size + 2, color=DONE)
        assert len(full.submobjects) == sum(1 for ch in "›".join(names) if not ch.isspace())
        self.names, vis = [], 0
        for i, n in enumerate(names):
            w = sum(1 for ch in n if not ch.isspace())
            self.names.append(full[vis:vis + w])
            vis += w + 1  # + the "›"
        self.add(full)
        self.move_to([0, y, 0])

    def go(self, i):
        anims = []
        for k, t in enumerate(self.names):
            c = WARM if k == i else (INK2 if (i is not None and k < i) else DONE)
            anims.append(t.animate.set_color(c))
        return anims
