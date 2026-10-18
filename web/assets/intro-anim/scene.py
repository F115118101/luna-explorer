"""LUNA Explorer — intro animation: "text -> tokens -> last hidden layer -> PCA -> DTMC state"

Baked (not live/interactive) because the layer-by-layer walkthrough, the individual-cell
compress animation, and the falling symbols are cheap to script by hand in Manim and look
far better pre-rendered than driven by a JS progress value. Rendered once offline with
`manim`, encoded with ffmpeg (both already installed locally), then embedded as an mp4 via
a plain <video controls> tag — which also gives free, reliable click/drag seeking, fixing
the earlier custom-scrubber problems outright.

Run:
    manim -qh --fps 30 scene.py Intro
then re-encode with a keyframe every 10 frames, or dragging the page's seek bar lags (the frame only catches up
after decoding from the previous keyframe; ffmpeg's default is one keyframe per 250 frames = 8.3 s):
    ffmpeg -i <rendered>.mp4 -c:v libx264 -crf 20 -preset slow -pix_fmt yuv420p -g 10 -keyint_min 10 -sc_threshold 0 -movflags +faststart -an intro.mp4
Output:
    media/videos/scene/1080p30/Intro.mp4
"""
import math
import random

from manim import *

config.background_color = "#05070c"

TOKENS = [
    "Water", "boils", "at", "100", "degrees", "Celsius", "at", "sea", "level",
    "."
]
N_BLOCK_PULSES = 4  # sampled snapshots inside the transformer-block stage (a decoder has many more than this)
N_CELLS = 12  # how many hidden-state values we draw as a schematic probe (not a real dimension count)
MONO = "Consolas"

INK = "#E7EBF3"
INK2 = "#AEB7CA"
MUTED = "#7C869C"
DONE = "#3B4252"
RULE = "#232B3B"
ACCENT = "#8CB8FF"
WARM = "#F4C26B"
WARM_SOFT = "#2A2314"
GREY_SOFT = "#161B26"
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
    t = (max(-1.0, min(1.0, val)) + 1) / 2
    lo, hi = 0.0, 1.0
    g = lo + (hi - lo) * t
    return rgb_to_color((g, g, g))


class Intro(MovingCameraScene):

    def construct(self):
        self.camera.frame.set(width=16.0)

        # ======================================================================
        # 🔧 修改：將流程圖向右移避免被切掉，並精簡流程圖內的文字
        # ======================================================================
        LEFT_X = -4.5
        stage_w, stage_h = 2.4, 0.8

        def make_stage(lines,
                       y,
                       x=LEFT_X,
                       w=stage_w,
                       font_size=17,
                       height=stage_h,
                       fill=SURFACE):
            box = RoundedRectangle(corner_radius=0.09,
                                   width=w,
                                   height=height,
                                   fill_color=fill,
                                   fill_opacity=1,
                                   stroke_color=RULE,
                                   stroke_width=2)
            txt = Text(lines,
                       font=MONO,
                       font_size=font_size,
                       color=INK2,
                       line_spacing=0.9)
            if txt.width > w - 0.3:
                txt.scale_to_fit_width(w - 0.3)
            txt.move_to(box.get_center())
            return VGroup(box, txt).move_to([x, y, 0])

        # evenly spaced (1.3 apart — enough that the connecting arrows have a real shaft,
        # not just a tip) and symmetric around y=0 so the whole chart sits centered
        stage_token = make_stage("Tokenization", 2.6)
        stage_embed = make_stage("Embedding", 1.3)
        stage_block = make_stage("Transformer Blocks", 0.0)
        stage_last = make_stage("Last Hidden State", -1.3)

        # 將分支分開對稱放置，並縮小寬度使其完美適應畫面
        sample_stage = make_stage("Next-Token Sampling",
                                  -2.6,
                                  x=LEFT_X - 1.5,
                                  w=2.6,
                                  height=0.8,
                                  fill=GREY_SOFT)
        pca_stage = make_stage("PCA", -2.6, x=LEFT_X + 1.5, w=2.6, height=0.8)

        stage_boxes = [
            stage_token, stage_embed, stage_block, stage_last, pca_stage
        ]

        # all arrows share the same stroke_width + tip_length so none looks thicker/bigger than another
        ARROW_SW, ARROW_TIP = 3, 0.15
        pipe_arrows = VGroup(*[
            Arrow(stage_boxes[i].get_bottom(),
                  stage_boxes[i + 1].get_top(),
                  buff=0.06,
                  color=MUTED,
                  stroke_width=ARROW_SW,
                  tip_length=ARROW_TIP) for i in range(3)
        ])

        branch_sample = DashedVMobject(
            Arrow(stage_last.get_bottom(),
                  sample_stage.get_top(),
                  buff=0.06,
                  color=MUTED,
                  stroke_width=ARROW_SW,
                  tip_length=ARROW_TIP),
            num_dashes=10,
        )
        branch_pca = Arrow(stage_last.get_bottom(),
                           pca_stage.get_top(),
                           buff=0.06,
                           color=MUTED,
                           stroke_width=ARROW_SW,
                           tip_length=ARROW_TIP)

        loop_arrow = CurvedArrow(
            stage_block.get_right() + RIGHT * 0.05 + UP * 0.25,
            stage_block.get_right() + RIGHT * 0.05 + DOWN * 0.25,
            angle=-TAU / 4,
            color=MUTED,
            stroke_width=ARROW_SW,
            tip_length=ARROW_TIP)
        loop_label = Text("xN", font=MONO, font_size=16,
                          color=MUTED).next_to(loop_arrow, RIGHT, buff=0.08)

        # the whole flowchart (boxes + every arrow) fades in together, in one synchronized beat
        self.play(FadeIn(VGroup(*stage_boxes)),
                  FadeIn(sample_stage),
                  FadeIn(pipe_arrows),
                  FadeIn(branch_sample),
                  FadeIn(branch_pca),
                  FadeIn(loop_arrow),
                  FadeIn(loop_label),
                  run_time=0.6)

        # the box's own gold border is set the instant a stage activates — complete, not
        # animated. Progress is shown by the box's interior filling left -> right: the fill
        # is the box's own rounded-rect shape intersected with a plain rectangle mask that
        # grows with the tracker, via manim's own boolean-op + always_redraw primitives — so
        # the corners stay exactly the box's corners and the fill butts right up against the
        # border with no gap, instead of a hand-rolled inset rectangle.
        active_borders = []
        active_trackers = []

        def activate_stage(i, prev_i=None):
            if prev_i is not None:
                pbox, ptxt = stage_boxes[prev_i]
                pbox.set_fill(color="#16233D").set_stroke(ACCENT, width=2)
                ptxt.set_color(ACCENT)
                if active_borders:
                    b = active_borders.pop()
                    b.clear_updaters()
                    self.remove(b)
                if active_trackers:
                    active_trackers.pop()

            box, txt = stage_boxes[i]
            # box itself is an OPAQUE rectangle (fill + stroke as one shape) — giving it a
            # higher z-index than the fill would also lift its own opaque interior above the
            # fill and hide it completely. So the border has to be its own separate, stroke-only
            # mobject layered above the fill, instead of relying on box's built-in stroke.
            box.set_stroke(WARM_SOFT, width=0)  # box's own stroke hidden; the dedicated border below replaces it
            txt.set_color(WARM)
            txt.set_z_index(5)  # text: topmost

            fill_shape = RoundedRectangle(corner_radius=0.09, width=box.width, height=box.height,
                                          fill_color=WARM_SOFT, fill_opacity=1,
                                          stroke_width=0).move_to(box)
            left_x, cy = box.get_left()[0], box.get_y()
            tracker = ValueTracker(0)

            def make_fill():
                alpha = clamp(tracker.get_value(), 0, 1)
                w = max(0.001, box.width * alpha)
                mask = Rectangle(width=w, height=box.height + 0.5, fill_opacity=1, stroke_width=0)
                mask.move_to([left_x + w / 2, cy, 0])
                clipped = Intersection(fill_shape, mask, fill_color=WARM_SOFT,
                                       fill_opacity=1, stroke_width=0)
                clipped.set_z_index(1)  # fill: above box's own base interior, below the border
                return clipped

            fill_mobj = always_redraw(make_fill)

            border = RoundedRectangle(corner_radius=0.09, width=box.width, height=box.height,
                                      stroke_color=WARM, stroke_width=4, fill_opacity=0).move_to(box)
            border.set_z_index(2)  # border: always drawn on top of the fill

            overlay = VGroup(fill_mobj, border)
            self.add(overlay)

            active_borders.append(overlay)
            active_trackers.append(tracker)

            return tracker

        # ---------- right column: 1xN probe grid + PCA/DTMC map ----------
        # right-side content is scaled down a notch (SCALE_R) so it reads as a clearly
        # separate, slightly smaller panel next to the left pipeline rather than matching
        # it 1:1 in visual weight
        SCALE_R = 0.88
        RIGHT_X = 2.8
        grid_y = 1.15  # chosen so flow_caption/TOKENS/grid/map stack centers on y=0, like the left column

        cells = VGroup()
        for j in range(N_CELLS):
            v = cell_val(0, -1, j)
            c = Square(side_length=0.7 * SCALE_R,
                       fill_color=gray(v),
                       fill_opacity=1,
                       stroke_color=RULE,
                       stroke_width=1.5)
            cells.add(c)
        cells.arrange(RIGHT, buff=0.07 * SCALE_R).move_to([RIGHT_X, grid_y, 0])
        cell_width = cells[0].width
        cell_x = [c.get_center()[0] for c in cells]

        # TOKENS sits right above the grid, centered on it, raised closer to the top of frame
        # one single Text for the whole line (same fix as semantics-anim make_row): separate
        # Text objects each get their own tight box, so arrange() centering them made "." and
        # descender-less words bob up and down. One string = one shared baseline. Per-word
        # slices are taken by visible-glyph count, since spaces never become submobjects.
        caption = Text(" ".join(TOKENS), font=MONO, font_size=23, color=MUTED)
        cap_words = []
        vis_pos = 0
        for w in TOKENS:
            cap_words.append(caption[vis_pos:vis_pos + len(w)])
            vis_pos += len(w)
        caption.next_to(cells, UP, buff=0.5)
        cap_words[0].set_color(WARM)
        self.play(FadeIn(caption, shift=UP * 0.2), run_time=0.6)

        def connect_lines(ti):
            return VGroup(*[
                # start from the row's shared bottom, not the word's own box bottom ("." would sit high)
                Line([cap_words[ti].get_x(), caption.get_bottom()[1], 0],
                     c.get_top(),
                     color=MUTED,
                     stroke_opacity=0.5) for c in cells
            ])

        # centered on the right column, same as TOKENS — stacked directly above it
        flow_caption = Text("hidden-layer values",
                            font=MONO,
                            font_size=18,
                            color="#C2B4FF")
        flow_caption.next_to(caption, UP, buff=0.3)
        caption_box = [flow_caption]

        def change_caption(text, color, run_time):
            old = caption_box[0]
            new = Text(text, font=MONO, font_size=18, color=color).move_to(old)
            self.play(FadeOut(old, shift=UP * 0.08),
                      FadeIn(new, shift=UP * 0.08),
                      run_time=run_time)
            caption_box[0] = new

        map_rect = Rectangle(width=9.2 * SCALE_R,
                             height=3.4 * SCALE_R,
                             fill_color=SUNK,
                             fill_opacity=1,
                             stroke_color=RULE,
                             stroke_width=2)
        map_rect.move_to([RIGHT_X, -1.4, 0])
        map_label = Text("Dimensionality Reduction (simulated)",
                         font=MONO,
                         font_size=18,
                         color=MUTED)
        map_label.next_to(map_rect, UP, buff=0.18,
                          aligned_edge=LEFT).align_to(map_rect, LEFT)

        def map_point(x, y):
            hw, hh = map_rect.width / 2 - 0.4, map_rect.height / 2 - 0.4
            vr = 2.3
            px = map_rect.get_center()[0] + clamp(x / vr, -1, 1) * hw
            py = map_rect.get_center()[1] + clamp(y / vr, -1, 1) * hh
            return np.array([px, py, 0])

        dots = VGroup()

        def highlight_token(i, prev_i=None):
            anims = []
            if prev_i is not None:
                anims.append(cap_words[prev_i].animate.set_color(DONE))
            if i is not None:
                anims.append(cap_words[i].animate.set_color(WARM))
            return anims

        def layer_pass(ti,
                       layer,
                       run_time,
                       tracker=None,
                       start_frac=0.0,
                       end_frac=1.0):
            syms, color_anims, cols = VGroup(), [], []
            for j in range(N_CELLS):
                if not show_symbol(ti, layer, j):
                    continue
                v = cell_val(ti, layer, j)
                sign = "+" if v >= 0 else "-"
                sym = Text(sign,
                           font=MONO,
                           font_size=25,
                           weight=BOLD,
                           color=WHITE)
                target = cells[j].get_center()
                start = np.array(
                    [target[0], target[1] + cells[j].height * 0.9, 0])
                sym.move_to(start)
                syms.add(sym)
                cols.append(j)
                color_anims.append(cells[j].animate.set_fill(gray(v)))

            anims = []
            if len(syms) > 0:
                anims.append(
                    LaggedStart(*[
                        s.animate.move_to(cells[j].get_center()).set_opacity(0)
                        for s, j in zip(syms, cols)
                    ],
                                lag_ratio=0.5 / max(1, len(syms))))
                anims.extend(color_anims)

            if tracker is not None:
                anims.append(tracker.animate(rate_func=linear).set_value(end_frac))

            if anims:
                self.play(*anims, run_time=run_time)
            else:
                self.wait(run_time)

            if len(syms) > 0:
                self.remove(syms)

        def compress_and_travel(ti,
                                word,
                                run_time,
                                show_caption,
                                tracker=None):
            x, y = token_point(ti, word)
            res_w, res_gap = 1.5 * SCALE_R, 0.12 * SCALE_R
            res_centers = [
                np.array([RIGHT_X - res_w / 2 - res_gap / 2, grid_y, 0]),
                np.array([RIGHT_X + res_w / 2 + res_gap / 2, grid_y, 0])
            ]
            cap_rt = min(0.35, run_time * 0.25) if show_caption else 0
            shrink_rt, travel_rt = run_time * 0.45, run_time * 0.25
            land_rt, ring_rt = max(0.05, run_time * 0.08), max(
                0.08, run_time * 0.15)
            total = shrink_rt + travel_rt + land_rt + ring_rt

            current_time = 0

            def get_t_anim(rt):
                nonlocal current_time
                current_time += rt
                return [tracker.animate(rate_func=linear).set_value(current_time /
                                                  total)] if tracker is not None else []

            if show_caption:
                # 🔧 修改：改為 Dimensionality Reduction (降低維度)
                change_caption("PCA: Dimensionality Reduction", "#C2B4FF",
                               cap_rt)

            shrink_anims = []
            for j, c in enumerate(cells):
                target_x = res_centers[0][
                    0] if j < N_CELLS / 2 else res_centers[1][0]
                shrink_anims.append(
                    c.animate.stretch_to_fit_width(0.02).move_to(
                        [target_x, grid_y, 0]).set_opacity(0))
            result = VGroup()
            for cx, val in zip(res_centers, (x, y)):
                box = RoundedRectangle(corner_radius=0.06,
                                       width=res_w,
                                       height=0.95 * SCALE_R,
                                       fill_color=WARM_SOFT,
                                       fill_opacity=0,
                                       stroke_color=WARM,
                                       stroke_width=2,
                                       stroke_opacity=0)
                box.move_to(cx)
                num = Text(f"{val:.2f}",
                           font=MONO,
                           font_size=21,
                           weight=BOLD,
                           color=WARM).move_to(cx)
                num.set_opacity(0)
                result.add(box, num)

            self.play(*shrink_anims,
                      result.animate.set_opacity(1),
                      *get_t_anim(shrink_rt),
                      run_time=shrink_rt)
            target = map_point(x, y)
            self.play(result.animate.move_to(target).scale(0.08),
                      *get_t_anim(travel_rt),
                      run_time=travel_rt)
            dot = Dot(target, radius=0.07 * SCALE_R, color=ACCENT)
            ring = Circle(radius=0.07 * SCALE_R, color=WARM,
                          stroke_width=2).move_to(target)
            self.play(FadeIn(dot),
                      FadeOut(result),
                      *get_t_anim(land_rt),
                      run_time=land_rt)
            self.play(ring.animate.scale(4).set_stroke(opacity=0),
                      *get_t_anim(ring_rt),
                      run_time=ring_rt)
            self.remove(ring)
            if show_caption:
                # 🔧 修改：修改文案，移除怪異的 map to 2d space
                change_caption("Repeat for all tokens", "#C2B4FF", cap_rt)
            dots.add(dot)

            for j, c in enumerate(cells):
                next_val = cell_val(ti +
                                    1, -1, j) if ti + 1 < len(TOKENS) else 0
                c.stretch_to_fit_width(cell_width).move_to(
                    [cell_x[j], grid_y,
                     0]).set_opacity(1).set_fill(gray(next_val))

        # ===================== token 0: the full walkthrough =====================
        tracker = activate_stage(0)
        self.play(tracker.animate(rate_func=linear).set_value(1.0), run_time=0.4)

        tracker = activate_stage(1, 0)
        lines = connect_lines(0)
        self.play(FadeIn(cells, shift=UP * 0.1),
                  Create(lines),
                  tracker.animate(rate_func=linear).set_value(0.5),
                  run_time=0.5)
        self.play(FadeOut(lines),
                  FadeIn(map_rect),
                  FadeIn(map_label),
                  FadeIn(flow_caption),
                  tracker.animate(rate_func=linear).set_value(1.0),
                  run_time=0.5)

        tracker = activate_stage(2, 1)
        # each of the N_BLOCK_PULSES "layers" gets its own full empty -> gold sweep, not one
        # continuous sweep shared across all of them — so the fill resets before every pulse
        for pulse in range(N_BLOCK_PULSES):
            tracker.set_value(0)
            layer_pass(0, pulse, run_time=0.65, tracker=tracker, end_frac=1.0)

        tracker = activate_stage(3, 2)
        layer_pass(0,
                   N_BLOCK_PULSES,
                   run_time=0.7,
                   tracker=tracker,
                   start_frac=0,
                   end_frac=1.0)

        tracker = activate_stage(4, 3)
        compress_and_travel(0,
                            TOKENS[0],
                            run_time=1.5,
                            show_caption=True,
                            tracker=tracker)

        self.play(*highlight_token(1, 0), run_time=0.2)

        # ===================== tokens 1..N-1: faster every time =====================
        rt = 0.75
        for i in range(1, len(TOKENS)):
            tracker = activate_stage(0, 4)
            self.play(tracker.animate(rate_func=linear).set_value(1.0),
                      run_time=max(0.03, rt * 0.15))

            tracker = activate_stage(1, 0)
            lines = connect_lines(i)
            self.play(Create(lines),
                      tracker.animate(rate_func=linear).set_value(0.5),
                      run_time=max(0.03, rt * 0.2))
            self.play(FadeOut(lines),
                      tracker.animate(rate_func=linear).set_value(1.0),
                      run_time=max(0.03, rt * 0.2))

            tracker = activate_stage(2, 1)
            block_total = max(0.1, rt * 0.4)
            block_rt = max(0.02, block_total / N_BLOCK_PULSES)
            # same number of pulses as token 0's walkthrough — just each one faster, so the
            # "transformer block runs N times" beat stays consistent instead of collapsing to one
            for pulse in range(N_BLOCK_PULSES):
                tracker.set_value(0)
                layer_pass(i, pulse, run_time=block_rt, tracker=tracker, end_frac=1.0)

            tracker = activate_stage(3, 2)
            layer_pass(i,
                       N_BLOCK_PULSES,
                       run_time=max(0.1, rt * 0.3),
                       tracker=tracker,
                       start_frac=0,
                       end_frac=1)

            tracker = activate_stage(4, 3)
            compress_and_travel(i,
                                TOKENS[i],
                                run_time=max(0.26, rt * 0.6),
                                show_caption=False,
                                tracker=tracker)

            if i < len(TOKENS) - 1:
                self.play(*highlight_token(i + 1, i),
                          run_time=max(0.06, min(0.15, rt * 0.3)))
            else:
                self.play(*highlight_token(None, i), run_time=0.12)
            rt *= 0.76

        if active_borders:
            b = active_borders.pop()
            b.clear_updaters()
            self.remove(b)

        change_caption(f"all {len(TOKENS)} tokens processed", MUTED, 0.4)
        self.wait(1.2)
