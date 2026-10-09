"""
LUNA Explorer — intro animation: "text -> tokens -> last hidden layer -> 2-D point"

Baked (not live/interactive) because the layer-by-layer walkthrough, the individual-cell
compress animation, and the falling symbols are cheap to script by hand in Manim and look
far better pre-rendered than driven by a JS progress value. Rendered once offline with
`manim`, encoded with ffmpeg (both already installed locally), then embedded as an mp4 via
a plain <video controls> tag — which also gives free, reliable click/drag seeking, fixing
the earlier custom-scrubber problems outright.

Run:
    manim -qh --fps 30 scene.py Intro
Output:
    media/videos/scene/1080p30/Intro.mp4
"""
import math
import random

from manim import *

config.background_color = "#05070c"

TOKENS = ["Water", "boils", "at", "100", "degrees", "Celsius", "at", "sea", "level", "."]
N_LAYERS = 5
N_CELLS = 12
MONO = "Consolas"

INK = "#E7EBF3"
INK2 = "#AEB7CA"
MUTED = "#7C869C"
RULE = "#232B3B"
ACCENT = "#8CB8FF"
WARM = "#F4C26B"
WARM_SOFT = "#2A2314"
SURFACE = "#0F141E"
SUNK = "#0B0F17"


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def cell_val(ti, layer, cell):
    r = random.Random(("val", ti, layer, cell))
    return max(-1.0, min(1.0, r.gauss(0, 0.5)))


def show_symbol(ti, layer, cell):
    r = random.Random(("sym", ti, layer, cell))
    return r.random() < 0.45


def token_point(ti, word):
    r = random.Random(("pt", word, ti))
    ang = r.random() * 2 * math.pi
    rad = 0.35 + abs(r.gauss(0, 0.4)) * 1.15
    return math.cos(ang) * rad, math.sin(ang) * rad


def gray(val):
    """val in [-1,1] -> pure black-to-white grayscale (no hue), floored so cells stay legible."""
    t = (max(-1.0, min(1.0, val)) + 1) / 2
    lo, hi = 0.12, 0.92
    g = lo + (hi - lo) * t
    return rgb_to_color((g, g, g))


class Intro(MovingCameraScene):
    def construct(self):
        self.camera.frame.set(width=14.4)

        # ---------- caption row (subtitle-style token strip) ----------
        cap_words = [Text(w, font=MONO, font_size=26, color=MUTED) for w in TOKENS]
        caption = VGroup(*cap_words).arrange(RIGHT, buff=0.22).to_edge(UP, buff=0.5)
        cap_words[0].set_color(WARM)   # token 0 starts active — it's about to be walked through
        self.play(FadeIn(caption, shift=UP * 0.2), run_time=0.6)

        # ---------- flowchart row ----------
        flow_y = 2.15
        boxes = VGroup()
        for i in range(N_LAYERS):
            label = "last" if i == N_LAYERS - 1 else str(i + 1)
            box = RoundedRectangle(corner_radius=0.08, width=2.0, height=0.9,
                                    fill_color=SURFACE, fill_opacity=1, stroke_color=RULE, stroke_width=2)
            txt = Text(label, font=MONO, font_size=24, color=INK2)
            txt.move_to(box.get_center())
            boxes.add(VGroup(box, txt))
        boxes.arrange(RIGHT, buff=0.35).move_to([0, flow_y, 0])
        arrows = VGroup(*[
            Arrow(boxes[i].get_right(), boxes[i + 1].get_left(), buff=0.04, color=MUTED, stroke_width=3,
                  max_tip_length_to_length_ratio=0.35)
            for i in range(N_LAYERS - 1)
        ])
        flow_caption = Text("last hidden layer (simulated, 4,096 dims)", font=MONO, font_size=20, color="#C2B4FF")
        flow_caption.next_to(boxes, UP, buff=0.3)
        self.play(FadeIn(boxes), FadeIn(arrows), FadeIn(flow_caption), run_time=0.6)
        caption_box = [flow_caption]   # mutable cell so the nested helper can swap the mobject

        def change_caption(text, color, run_time):
            """Crossfade to new caption text instead of Transform — Transform between two Text
            mobjects with a different glyph count morphs letters into each other, which reads
            as garbled mid-transition; a plain fade out/in stays legible throughout."""
            old = caption_box[0]
            new = Text(text, font=MONO, font_size=20, color=color).move_to(old)
            self.play(FadeOut(old, shift=UP * 0.08), FadeIn(new, shift=UP * 0.08), run_time=run_time)
            caption_box[0] = new

        def set_active_box(i, prev_i=None):
            anims = []
            if prev_i is not None:
                pbox, ptxt = boxes[prev_i]
                anims += [pbox.animate.set_stroke(ACCENT).set_fill(color="#16233D"), ptxt.animate.set_color(ACCENT)]
            box, txt = boxes[i]
            anims += [box.animate.set_stroke(WARM, width=3).set_fill(color=WARM_SOFT), txt.animate.set_color(WARM)]
            return anims

        # ---------- 1xN probe grid ----------
        grid_y = 0.65
        cells = VGroup()
        for j in range(N_CELLS):
            c = Square(side_length=0.78, fill_color=gray(0), fill_opacity=1, stroke_color=RULE, stroke_width=1.5)
            cells.add(c)
        cells.arrange(RIGHT, buff=0.08).move_to([0, grid_y, 0])
        cell_width = cells[0].width
        cell_x = [c.get_center()[0] for c in cells]
        self.play(FadeIn(cells), run_time=0.4)

        map_rect = Rectangle(width=9.6, height=3.6, fill_color=SUNK, fill_opacity=1, stroke_color=RULE, stroke_width=2)
        map_rect.move_to([0, -2.35, 0])
        map_label = Text("2-D point cloud (PCA, simulated)", font=MONO, font_size=20, color=MUTED)
        map_label.next_to(map_rect, UP, buff=0.18, aligned_edge=LEFT).align_to(map_rect, LEFT)
        self.play(FadeIn(map_rect), FadeIn(map_label), run_time=0.4)

        def map_point(x, y):
            hw, hh = map_rect.width / 2 - 0.4, map_rect.height / 2 - 0.4
            vr = 2.3
            px = map_rect.get_center()[0] + clamp(x / vr, -1, 1) * hw
            py = map_rect.get_center()[1] + clamp(y / vr, -1, 1) * hh
            return np.array([px, py, 0])

        dots = VGroup()

        def highlight_token(i, prev_i=None):
            # color only, never scale — scaling in place without an exact inverse on the way back
            # down accumulates drift across ten tokens and the caption row creeps out of alignment
            anims = []
            if prev_i is not None:
                anims.append(cap_words[prev_i].animate.set_color(INK2))
            anims.append(cap_words[i].animate.set_color(WARM))
            return anims

        def layer_pass(ti, layer, run_time):
            """A subset of cells each get a +/- symbol that appears just above the cell and falls
            straight down into it, fading out over that same fall — not a separate fade-in then
            fade-out — so it visibly sinks into the cell as it disappears. Each symbol-cell's
            shade updates to its new value at the same time."""
            syms, color_anims, cols = VGroup(), [], []
            for j in range(N_CELLS):
                if not show_symbol(ti, layer, j):
                    continue
                v = cell_val(ti, layer, j)
                sign = "+" if v >= 0 else "−"
                sym = Text(sign, font=MONO, font_size=30, weight=BOLD, color=WHITE)
                target = cells[j].get_center()
                start = np.array([target[0], target[1] + cells[j].height * 0.9, 0])   # just above the cell, not off-frame
                sym.move_to(start)
                syms.add(sym)
                cols.append(j)
                color_anims.append(cells[j].animate.set_fill(gray(v)))
            if len(syms) == 0:
                self.wait(run_time)
                return
            self.play(
                LaggedStart(*[
                    s.animate.move_to(cells[j].get_center()).set_opacity(0)
                    for s, j in zip(syms, cols)
                ], lag_ratio=0.5 / max(1, len(syms))),
                *color_anims,
                run_time=run_time,
            )
            self.remove(syms)

        def compress_and_travel(ti, word, run_time, show_caption):
            """Each of the 12 cells narrows and slides toward the centre (not a group scale);
            the two result cells fade in at the exact width/position the row converges to."""
            x, y = token_point(ti, word)
            res_w, res_gap = 1.5, 0.12
            res_centers = [np.array([-res_w / 2 - res_gap / 2, grid_y, 0]), np.array([res_w / 2 + res_gap / 2, grid_y, 0])]
            # every sub-step is a fraction of run_time (with small floors) so the whole thing
            # actually shrinks as run_time shrinks for later, faster tokens
            cap_rt = min(0.35, run_time * 0.25) if show_caption else 0
            shrink_rt, grow_rt, travel_rt = run_time * 0.35, run_time * 0.2, run_time * 0.25
            land_rt, ring_rt = max(0.05, run_time * 0.08), max(0.08, run_time * 0.15)
            if show_caption:
                change_caption("compress 4,096 dims → 2 numbers", "#C2B4FF", cap_rt)
            shrink_anims = []
            for j, c in enumerate(cells):
                target_x = res_centers[0][0] if j < N_CELLS / 2 else res_centers[1][0]
                shrink_anims.append(c.animate.stretch_to_fit_width(0.02).move_to([target_x, grid_y, 0]).set_opacity(0))
            result = VGroup()
            for cx, val in zip(res_centers, (x, y)):
                box = RoundedRectangle(corner_radius=0.06, width=res_w, height=0.95, fill_color=WARM_SOFT,
                                        fill_opacity=0, stroke_color=WARM, stroke_width=2, stroke_opacity=0)
                box.move_to(cx)
                num = Text(f"{val:.2f}", font=MONO, font_size=24, weight=BOLD, color=WARM).move_to(cx)
                num.set_opacity(0)
                result.add(box, num)
            self.play(*shrink_anims, run_time=shrink_rt)
            self.play(result.animate.set_opacity(1), run_time=grow_rt)
            if show_caption:
                change_caption("→ a 2-D point", "#C2B4FF", cap_rt)
            target = map_point(x, y)
            self.play(result.animate.move_to(target).scale(0.08), run_time=travel_rt)
            dot = Dot(target, radius=0.07, color=ACCENT)
            ring = Circle(radius=0.07, color=WARM, stroke_width=2).move_to(target)
            self.play(FadeIn(dot), FadeOut(result), run_time=land_rt)
            self.play(ring.animate.scale(4).set_stroke(opacity=0), run_time=ring_rt)
            self.remove(ring)
            dots.add(dot)
            # reset the 12 cells for the next token
            for j, c in enumerate(cells):
                c.stretch_to_fit_width(cell_width).move_to([cell_x[j], grid_y, 0]).set_opacity(1).set_fill(gray(0))

        # ===================== token 0: the full walkthrough =====================
        prev_box = None
        for layer in range(N_LAYERS):
            self.play(*set_active_box(layer, prev_box), run_time=0.22)
            prev_box = layer
            layer_pass(0, layer, run_time=0.75)
        change_caption("extract the last hidden layer →", WARM, 0.35)
        self.play(boxes[-1][0].animate.set_stroke(width=4).set_fill(color="#3A2E10"), run_time=0.3)
        self.play(boxes[-1][0].animate.set_stroke(width=3).set_fill(color=WARM_SOFT), run_time=0.2)
        compress_and_travel(0, TOKENS[0], run_time=1.3, show_caption=True)
        self.play(*highlight_token(1, 0), run_time=0.2)

        # ===================== tokens 1..N-1: faster every time =====================
        rt = 0.75
        for i in range(1, len(TOKENS)):
            self.play(boxes[-1][0].animate.set_stroke(WARM, width=3).set_fill(color=WARM_SOFT), run_time=max(0.03, min(0.1, rt * 0.2)))
            layer_pass(i, N_LAYERS - 1, run_time=max(0.13, rt * 0.4))
            compress_and_travel(i, TOKENS[i], run_time=max(0.26, rt * 0.6), show_caption=False)
            if i < len(TOKENS) - 1:
                self.play(*highlight_token(i + 1, i), run_time=max(0.06, min(0.15, rt * 0.3)))
            else:
                self.play(cap_words[i].animate.set_color(INK2), run_time=0.12)
            rt *= 0.76

        self.play(FadeOut(boxes), FadeOut(arrows), run_time=0.4)
        change_caption(f"all {len(TOKENS)} tokens mapped to a point", MUTED, 0.4)
        self.wait(1.2)
