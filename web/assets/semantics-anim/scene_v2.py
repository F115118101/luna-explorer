"""LUNA Explorer — section 05 intro v2: labels of the training answers -> every token carries its
answer's label into its abstract state -> theta per state -> the example answer's score.
Uses luna_v3.html's toy data and the authors' algorithm exactly as the page does
(toy-data.json, exported by ../export_toy.mjs from the page's own code):

  1. Collect: each of the 160 training answers is labelled y = 1 (true) or y = 0 (false); every
     one of its token windows adds 1 to pos or neg of the state it lands in.
  2. theta(s) = pos(s) / (pos(s) + neg(s)), worked out in full for one state, then every state at
     once. A state never seen in training gets theta = 0 (authors' code).
  3. Read: the example answer gets no label; each window reads theta of its state, and
     score = (1/n) * sum of those theta, every term written; score < tau = 0.45 -> flagged.

The 125 states (PCA 3, 5 cells per dimension) are drawn as five 5 x 5 slices, one per c3.
A state's place is decoded from its number s = c1 + 5 c2 + 25 c3, like the code (two training
windows have c = 5 because the code does not clip, and so share a number with another box).
Every computation is written in the pane on the left. Kept next to the original scene.py.

Run:
    node ../export_toy.mjs
    manim -ql --fps 15 scene_v2.py TraceV2          # preview
    manim -qh --fps 30 scene_v2.py TraceV2
    ffmpeg -i media/videos/scene_v2/1080p30/TraceV2.mp4 -c:v libx264 -crf 20 -preset slow -pix_fmt yuv420p -g 10 -keyint_min 10 -sc_threshold 0 -movflags +faststart -an trace-v2.mp4
"""
import os
import sys

from manim import *

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from anim_common import EqPane, pane_frame, load_toy, MATH, MONO  # noqa: E402

config.background_color = "#05070c"

TOY = load_toy()
TOKENS = [t.replace("▁", "_") for t in TOY["tokens"]]  # ▁ is drawn as several paths; the page's formulas write _
PROMPT = TOY["prompt"].strip()
M = TOY["config"]["m"]
TRAIN = TOY["train"]
RUN = next(t for t in TOY["test"] if t["run"])
RA = RUN["a"]
TAU = 0.45

INK = "#E7EBF3"
INK2 = "#AEB7CA"
MUTED = "#7C869C"
DONE = "#3B4252"
RULE = "#232B3B"
AXIS = "#2F3A50"
WARM = "#F4C26B"
GOOD = "#5FD3A0"
BAD = "#FF7B8B"
SUNK = "#0B0F17"
EMPTY = "#0E131C"

_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def sub(s):
    return str(s).translate(_SUB)


def sem_col(th):
    """the page's semCol(): red (255,123,139) -> green (95,211,160) by theta"""
    a, b = (255, 123, 139), (95, 211, 160)
    return rgb_to_color(tuple((x + (y - x) * th) / 255 for x, y in zip(a, b)))


def decode(s):
    return s % M, (s // M) % M, s // (M * M)


def chunks(xs, n):
    return [xs[k:k + n] for k in range(0, len(xs), n)]


class TraceV2(MovingCameraScene):
    def construct(self):
        self.camera.frame.set(width=16.0)

        def play(anims, rt):
            if anims:
                self.play(*anims, run_time=rt)
            else:
                self.wait(rt)

        title = [Text("training labels → into every token's state → θ̄ per state → the example's score",
                      font=MATH, font_size=21, color=MUTED).move_to([0, 4.05, 0])]

        def set_title(s, color=MUTED):
            new = Text(s, font=MATH, font_size=21, color=color).move_to([0, 4.05, 0])  # θ̄ needs the math font
            anims = [FadeOut(title[0]), FadeIn(new)]
            title[0] = new
            return anims

        # ---------- left: computation pane ----------
        PL, PR, PT, PB = -7.75, -2.05, 2.95, -4.3
        frame = pane_frame(PL, PR, PT, PB)
        pane = EqPane(PL + 0.05, PR - 0.15, PT - 0.25, size=18, line_h=0.4)

        # ---------- right: the 5 x 5 x 5 grid as a stack of five plates (one per c3), seen in 3D ----------
        # a plate is the c1-c2 plane; plates are pulled apart vertically so each one can be read.
        # Upper plates are drawn over lower ones, so a lower plate's back row may be hidden.
        YAW, PITCH, S3, LIFT = -0.25, 0.22, 3.2, 1.27
        CX3, CY3 = 3.05, -3.2                     # centre of the bottom plate (c3 = 0)
        cyw, syw, spt = np.cos(YAW), np.sin(YAW), np.sin(PITCH)

        def p3(X, Y, c3):
            """X, Y in [-1, 1] on plate c3 -> screen point, and depth (larger = further back)"""
            u, v = X * cyw - Y * syw, X * syw + Y * cyw
            return np.array([CX3 + S3 * u, CY3 + S3 * spt * v + LIFT * c3, 0]), v

        edge = lambda k: -1 + 2 * k / M

        def quad(c1, c2, c3, inset=0.035):
            x0, x1 = edge(c1) + inset, edge(c1 + 1) - inset
            y0, y1 = edge(c2) + inset, edge(c2 + 1) - inset
            return [p3(x, y, c3)[0] for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]

        sq, txt = {}, {}
        grid = VGroup()
        zorder = {}
        for c3 in range(M):
            corners = [p3(x, y, c3)[0] for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            plate = Polygon(*corners, fill_color="#0A0E15", fill_opacity=0.96, stroke_color=AXIS, stroke_width=1.5)
            plate.set_z_index(c3 * 100)
            grid.add(plate)
            cells = sorted(((c1, c2) for c1 in range(M) for c2 in range(M)),
                           key=lambda c: -p3(edge(c[0]) + 0.2, edge(c[1]) + 0.2, c3)[1])
            for rank, (c1, c2) in enumerate(cells):
                s = c1 + M * c2 + M * M * c3
                sq[s] = Polygon(*quad(c1, c2, c3), fill_color=EMPTY, fill_opacity=1, stroke_color=RULE, stroke_width=1)
                zorder[s] = c3 * 100 + 2 + rank * 2
                sq[s].set_z_index(zorder[s])
                grid.add(sq[s])
            lab = Text(f"c₃ = {c3}", font=MATH, font_size=15, color=INK2)
            lab.next_to(p3(-1, -1, c3)[0] if p3(-1, -1, c3)[0][0] < p3(-1, 1, c3)[0][0] else p3(-1, 1, c3)[0], LEFT, buff=0.18)
            grid.add(lab.set_z_index(c3 * 100 + 90))
        # the four vertical corner lines tie the plates into one cube
        for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            grid.add(DashedLine(p3(x, y, 0)[0], p3(x, y, M - 1)[0], color=AXIS, stroke_width=1, dash_length=0.08).set_z_index(-1))
        # axis names on the bottom plate's front edges
        grid.add(Text("c₁ →", font=MATH, font_size=14, color=MUTED).move_to(p3(0, -1, 0)[0] + DOWN * 0.22),
                 Text("c₂ →", font=MATH, font_size=14, color=MUTED).next_to(p3(1, 0, 0)[0], RIGHT, buff=0.12))

        pos, neg = {}, {}

        def count_text(s):
            p, n = pos.get(s, 0), neg.get(s, 0)
            c = sq[s].get_center()
            g = VGroup(Text(str(p), font=MONO, font_size=11, color=GOOD).move_to(c + LEFT * 0.16),
                       Text(str(n), font=MONO, font_size=11, color=BAD).move_to(c + RIGHT * 0.16))
            return g.set_z_index(zorder[s] + 1)

        def outline(s):
            return sq[s].copy().set_fill(opacity=0).set_stroke(WARM, width=4).set_z_index(1000)

        # status corner (bottom right, under the stack)
        SX, SY = 6.0, -4.05
        legend = VGroup(Text("in each box:", font=MATH, font_size=15, color=MUTED),
                        VGroup(Text("pos", font=MONO, font_size=15, color=GOOD),
                               Text("neg", font=MONO, font_size=15, color=BAD)).arrange(RIGHT, buff=0.25)
                        ).arrange(RIGHT, buff=0.2).move_to([SX, SY, 0])

        # ---------- token row ----------
        ROW_Y = 3.38
        row = [None]

        def set_row(s, color=INK2):
            new = Text(s, font=MONO, font_size=20, color=color)
            if new.width > 15.2:
                new.scale_to_fit_width(15.2)
            new.move_to([0, ROW_Y, 0])
            anims = ([FadeOut(row[0])] if row[0] is not None else []) + [FadeIn(new)]
            row[0] = new
            return anims

        self.play(FadeIn(title[0]), FadeIn(frame), FadeIn(grid), FadeIn(legend), run_time=0.7)

        # ===================== 1. collect labels =====================
        play(pane.set([
            ("h", "1 · Collect labels"),
            ("e", "y", "1 if the answer is true, 0 if false"),
            ("c", "every token of the answer carries y into its state s̄"),
            ("e", "pos(s̄)", "# tokens with y = 1 that landed in s̄"),
            ("e", "neg(s̄)", "# tokens with y = 0 that landed in s̄"),
            ("gap",),
        ], eq_x=pane._eq_x_for([("e", "neg(s̄)", "")])), 0.5)
        self.wait(2.0)

        def add_answer(k, rt, flying=True, status=True):
            tr = TRAIN[k]
            y, states = tr["y"], [w["s"] for w in tr["a"]]
            anims = []
            if status:
                anims += set_row(f"training answer {k + 1} of {len(TRAIN)}  ·  y = {y} ({'true' if y else 'false'}):  "
                                 + " → ".join(f"s̄{s}" for s in states), GOOD if y else BAD)
                anims += (pane.pop(3) if k else []) + pane.append([
                    ("e", "answer", f"{k + 1} of {len(TRAIN)}"),
                    ("e", "y", f"{y}", GOOD if y else BAD),
                    ("e", "states", ", ".join(f"s̄{s}" for s in states)),
                ])
                play(anims, rt * 0.35)
                anims = []
            chips = VGroup()
            for s in states:
                (pos if y else neg)[s] = (pos if y else neg).get(s, 0) + 1
                if flying:
                    chip = Text(str(y), font=MONO, font_size=16, weight=BOLD, color=GOOD if y else BAD).set_z_index(2000)
                    chip.move_to(row[0].get_bottom() + DOWN * 0.25)
                    chips.add(chip)
                    anims.append(chip.animate.move_to(sq[s]).scale(0.5).set_opacity(0))
            for s in set(states):
                new = count_text(s)
                anims.append(Transform(txt[s], new) if s in txt else FadeIn(new))
                txt.setdefault(s, new)
            self.add(chips)
            play(anims, rt * 0.65)
            self.remove(chips)

        for k in range(3):                       # slowly
            add_answer(k, 2.4)
            self.wait(1.2)
        for k in range(3, 12):                   # one by one, faster
            add_answer(k, 0.55)
        # the rest in batches: every box's counts updated together
        k = 12
        for size in (8, 12, 20, 30, 40, 38):
            ks = list(range(k, min(len(TRAIN), k + size)))
            for kk in ks:
                tr = TRAIN[kk]
                for w in tr["a"]:
                    (pos if tr["y"] else neg)[w["s"]] = (pos if tr["y"] else neg).get(w["s"], 0) + 1
            touched = {w["s"] for kk in ks for w in TRAIN[kk]["a"]}
            anims = set_row(f"training answers {ks[0] + 1} to {ks[-1] + 1} of {len(TRAIN)}", INK2)
            anims += pane.pop(3) + pane.append([("e", "answers", f"{ks[0] + 1} to {ks[-1] + 1} of {len(TRAIN)}"),
                                                 ("gap",), ("gap",)])
            for s in touched:
                new = count_text(s)
                if s in txt:
                    anims.append(Transform(txt[s], new))
                else:
                    anims.append(FadeIn(new))
                    txt[s] = new
                anims.append(Indicate(sq[s], color=WARM, scale_factor=1.0))
            play(anims, 0.7)
            k = ks[-1] + 1
        assert k == len(TRAIN)
        total = sum(pos.values()) + sum(neg.values())
        play(set_row(f"all {len(TRAIN)} training answers collected: {total} tokens in {len(set(pos) | set(neg))} states", INK2)
             + pane.pop(3) + pane.append([("e", "Σ pos + Σ neg", f"{sum(pos.values())} + {sum(neg.values())}"),
                                          ("e", "", f"{total} tokens", WARM)]), 0.5)
        self.wait(1.5)

        # ===================== 2. theta per state, once =====================
        ex = 37   # the state of "_dens": the page's lowest step
        p, n = pos.get(ex, 0), neg.get(ex, 0)
        theta = {s: pos.get(s, 0) / (pos.get(s, 0) + neg.get(s, 0)) for s in set(pos) | set(neg)}
        ring = outline(ex)
        play(set_title("one θ̄ per state: the share of true answers among its tokens")
             + [FadeOut(row[0])] + pane.set([
                 ("h", "2 · θ̄ per state"),
                 ("e", "θ̄(s̄)", "pos(s̄) / (pos(s̄) + neg(s̄))"),
                 ("c", "a state never seen in training: θ̄ = 0 (authors' code)"),
                 ("gap",),
                 ("c", f'for example s̄{ex}, the state of "_dens":'),
                 ("e", f"θ̄(s̄{ex})", f"{p} / ({p} + {n})"),
                 ("e", "", f"{p} / {p + n}"),
                 ("e", "", f"{theta[ex]:.3f}", WARM),
             ]) + [Create(ring)], 0.7)
        row[0] = None
        self.wait(4.0)
        anims = [FadeOut(ring), FadeOut(legend)]
        for s, m in txt.items():
            th = theta[s]
            anims += [sq[s].animate.set_fill(sem_col(th), opacity=0.55),
                      Transform(m, Text(f"{th:.2f}", font=MONO, font_size=12, color=INK).move_to(sq[s]).set_z_index(zorder[s] + 1))]
        anims += pane.append([("c", "every other state: the same formula, all at once")])
        key = VGroup(Text("θ̄ = 0", font=MATH, font_size=15, color=MUTED), *[
            Square(side_length=0.22, fill_color=sem_col(v), fill_opacity=0.8, stroke_width=0) for v in (0, .25, .5, .75, 1)],
            Text("1", font=MONO, font_size=13, color=MUTED)).arrange(RIGHT, buff=0.06).move_to([SX, SY, 0])
        anims.append(FadeIn(key))
        play(anims, 1.2)
        self.wait(1.8)

        # ===================== 3. the example reads θ̄ =====================
        words_txt = " ".join(TOKENS)
        q_row = Text(PROMPT + "  " + words_txt, font=MONO, font_size=20, color=INK2)
        vis = sum(1 for ch in PROMPT if not ch.isspace())
        assert len(q_row.submobjects) == vis + sum(len(w) for w in TOKENS)
        q_row[:vis].set_color(MUTED)
        words = []
        for w in TOKENS:
            words.append(q_row[vis:vis + len(w)])
            vis += len(w)
        q_row.move_to([0, ROW_Y, 0])
        play(set_title("the example answer: no label, each token reads θ̄ of its state", WARM)
             + [FadeIn(q_row)] + pane.set([("h", "3 · Read θ̄ for the example"),
                                           ("c", "no label this time; the last token “.” is not used")],
                                          eq_x=pane._eq_x_for([("e", "θ̄(s̄64)", "")])), 0.5)
        base_y = q_row.get_bottom()[1] - 0.2
        th_run = []
        for i, w in enumerate(RA):
            s = w["s"]
            th = theta.get(s, 0.0)
            th_run.append(th)
            ring = outline(s)
            num = Text(f"{th:.2f}", font=MONO, font_size=14, weight=BOLD, color=sem_col(th)).move_to(sq[s]).set_z_index(2000)
            play([FadeIn(ring), words[i].animate.set_color(WARM), num.animate.move_to([words[i].get_x(), base_y, 0])]
                 + pane.append([("e", f"θ̄(s̄{s})", f"{pos.get(s, 0)} / {pos.get(s, 0) + neg.get(s, 0)} = {th:.3f}    {TOKENS[i]}")]), 0.55)
            play([FadeOut(ring), words[i].animate.set_color(DONE)], 0.2)
        play([words[-1].animate.set_color(DONE)], 0.2)
        self.wait(1.0)

        # slices -> bars of the 8 θ̄ values (the page's semantics figure)
        n_s = len(th_run)
        CX0, CX1, CY0, CY1 = -0.9, 6.3, -3.2, 2.2
        Yv = lambda v: CY0 + (CY1 - CY0) * v
        bw = (CX1 - CX0) / n_s
        axes = VGroup(Line([CX0, CY0, 0], [CX1, CY0, 0], color=AXIS, stroke_width=1.5))
        for v in (0, 0.5, 1):
            axes.add(Text(f"{v:g}", font=MONO, font_size=13, color=MUTED).next_to([CX0, Yv(v), 0], LEFT, buff=0.12),
                     DashedLine([CX0, Yv(v), 0], [CX1, Yv(v), 0], color=AXIS, stroke_width=1, dash_length=0.06))
        axes.add(Text("θ̄", font=MATH, font_size=18, color=MUTED).move_to([CX0 - 0.35, CY1 + 0.3, 0]))
        bars, blabs = VGroup(), VGroup()
        for i, (th, w) in enumerate(zip(th_run, RA)):
            x = CX0 + (i + 0.5) * bw
            h = max(0.02, (CY1 - CY0) * th)
            bars.add(Rectangle(width=bw * 0.62, height=h, fill_color=sem_col(th), fill_opacity=0.9, stroke_width=0)
                     .move_to([x, CY0 + h / 2, 0]))
            blabs.add(Text(f"{th:.2f}", font=MONO, font_size=13, color=INK).move_to([x, CY0 + h + 0.18, 0]),
                      Text(TOKENS[i], font=MONO, font_size=13, color=INK2).move_to([x, CY0 - 0.25, 0]),
                      Text(f"s̄{w['s']}", font=MATH, font_size=12, color=MUTED).move_to([x, CY0 - 0.48, 0]))
        play([FadeOut(grid), FadeOut(key)] + [FadeOut(m) for m in txt.values()] + [FadeIn(axes)], 0.6)
        play([LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.1), FadeIn(blabs)], 1.0)

        # ===================== 4. the score and the decision =====================
        score = sum(th_run) / n_s
        terms = [f"{v:.3f}" for v in th_run]
        parts = chunks(terms, 4)
        rows = [("h", "4 · Score and decision"),
                ("e", "score", "(1/n) · Σₜ θ̄(s̄ₜ)"),
                ("e", "", "(" + " + ".join(parts[0]) + " +"),
                ("+", " + ".join(parts[1]) + f") / {n_s}"),
                ("e", "", f"{sum(th_run):.3f} / {n_s}"),
                ("e", "", f"{score:.3f}", WARM),
                ("gap",),
                ("c", "hallucination ⇔ score < τ"),
                ("e", "τ", f"{TAU:.2f}   (threshold, chosen by hand)"),
                ("e", "verdict", f"{score:.3f} < {TAU:.2f}  →  flagged" if score < TAU else f"{score:.3f} ≥ {TAU:.2f}  →  normal",
                 BAD if score < TAU else GOOD),
                ("c", "true label: false answer, so the flag is correct" if score < TAU else "true label: false answer, so this is missed")]
        play(pane.set(rows), 0.6)
        self.wait(1.0)
        score_line = DashedLine([CX0, Yv(score), 0], [CX1, Yv(score), 0], color=INK, stroke_width=2)
        score_lab = Text(f"score {score:.3f}", font=MATH, font_size=16, color=INK).next_to([CX1, Yv(score), 0], RIGHT, buff=0.1).shift(DOWN * 0.13)
        tau_line = Line([CX0, Yv(TAU), 0], [CX1, Yv(TAU), 0], color=WARM, stroke_width=2.5)
        tau_lab = Text(f"τ = {TAU:.2f}", font=MATH, font_size=16, color=WARM).next_to([CX1, Yv(TAU), 0], RIGHT, buff=0.1).shift(UP * 0.13)
        play([Create(score_line), FadeIn(score_lab)], 0.6)
        play([Create(tau_line), FadeIn(tau_lab)], 0.6)
        self.wait(3.0)
