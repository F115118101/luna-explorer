"""LUNA Explorer — section-04 animation v2: the 3D grid (PCA 3, 5 cells per dimension = 125
abstract states) -> the example answer's states -> DTMC nodes and transition probabilities ->
the example's trace probability. Uses luna_v3.html's toy data and the authors' algorithm exactly
as the page does (toy-data.json, exported by ../export_toy.mjs from the page's own code):

  * u_i = (max_i - min_i) / 5 over all training tokens (no padding)
  * c_i = trunc((z_i - min_i) / u_i)   (numpy astype(int), no clipping)
  * state number s = c1 + 5 c2 + 25 c3
  * P(s -> s') = #(s -> s') / #(s -> .)  counted over the 160 training traces
  * the DTMC drawn = the page's 12 most visited states + the example's own states

Every computation is written in the pane on the left (top to bottom, broken at '=', '=' signs in
one column). Kept next to the original scene.py so the two renders can be compared.

Run:
    node ../export_toy.mjs
    manim -ql --fps 15 scene_v2.py DtmcV2          # preview
    manim -qh --fps 30 scene_v2.py DtmcV2
    ffmpeg -i media/videos/scene_v2/1080p30/DtmcV2.mp4 -c:v libx264 -crf 20 -preset slow -pix_fmt yuv420p -g 10 -keyint_min 10 -sc_threshold 0 -movflags +faststart -an dtmc-v2.mp4
"""
import math
import os
import sys

from manim import *

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from anim_common import EqPane, Proj3D, pane_frame, load_toy, MATH, MONO  # noqa: E402

config.background_color = "#05070c"

TOY = load_toy()
TOKENS = [t.replace("▁", "_") for t in TOY["tokens"]]  # ▁ is drawn as several paths; the page's formulas write _
M = TOY["config"]["m"]
LO, HI, U = TOY["grid"]["lo"], TOY["grid"]["hi"], TOY["grid"]["unit"]
RUN = next(t for t in TOY["test"] if t["run"])
RA = RUN["a"]                                   # 8 windows: the last token "." is never used
COUNTS = {int(k): v for k, v in TOY["counts"].items()}
TRANS = {tuple(map(int, k.split(">"))): v for k, v in TOY["trans"].items()}
OUT = {int(k): v for k, v in TOY["out"].items()}
TOP12 = TOY["top12"]
RUN_TRANS = TOY["runTrans"]

INK = "#E7EBF3"
INK2 = "#AEB7CA"
MUTED = "#7C869C"
DONE = "#3B4252"
RULE = "#2F3A50"
ACCENT = "#C2B4FF"   # the page draws P̄ in violet
WARM = "#F4C26B"
GOOD = "#5FD3A0"
BAD = "#FF7B8B"
NODE_FILL = "#161D2B"

_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def sub(s):
    return str(s).translate(_SUB)


def f2(v):
    return f"{v:.2f}".replace("-", "−")


def p2(v):
    return f"({f2(v)})" if v < 0 else f2(v)


def cell_of_state(s):
    return (s % M, (s // M) % M, s // (M * M))


def P(a, b):
    return TRANS.get((a, b), 0) / OUT[a] if OUT.get(a) else 0.0


def _tip(arc, color, opacity, tip):
    arc.add_tip(tip_length=tip, tip_width=tip * 0.9)
    arc.get_tip().set_fill(color, opacity=opacity).set_stroke(width=0)
    return arc


def curved_edge(ca, cb, ra, rb, angle, color, width, opacity, tip, dashed=False):
    full = ArcBetweenPoints(ca, cb, angle=angle)

    def find(target, r, from_start):
        lo, hi = (0.0, 0.5) if from_start else (0.5, 1.0)
        for _ in range(30):
            mid = (lo + hi) / 2
            inside = np.linalg.norm(full.point_from_proportion(mid) - target) < r
            if from_start:
                lo, hi = (mid, hi) if inside else (lo, mid)
            else:
                lo, hi = (lo, mid) if inside else (mid, hi)
        return (lo + hi) / 2

    t1, t2 = find(ca, ra + 0.03, True), find(cb, rb + 0.05, False)
    arc = ArcBetweenPoints(full.point_from_proportion(t1), full.point_from_proportion(t2),
                           angle=angle * (t2 - t1), stroke_color=color, stroke_width=width,
                           stroke_opacity=opacity)
    arc = _tip(arc, color, opacity, tip)
    if dashed:
        tip_m = arc.get_tip()
        arc.remove(tip_m)
        return VGroup(DashedVMobject(arc, num_dashes=14), tip_m)
    return arc


def self_loop(c, r, color, width, opacity, tip):
    a1 = c + r * np.array([math.cos(1.95), math.sin(1.95), 0])
    a2 = c + r * np.array([math.cos(1.19), math.sin(1.19), 0])
    loop = ArcBetweenPoints(a1, a2, angle=-3.6, stroke_color=color, stroke_width=width,
                            stroke_opacity=opacity)
    return _tip(loop, color, opacity, tip)


class DtmcV2(MovingCameraScene):
    def construct(self):
        self.camera.frame.set(width=16.0)

        def play(anims, rt):
            if anims:
                self.play(*anims, run_time=rt)
            else:
                self.wait(rt)

        # ---------- left: computation pane (+ small print for the long table) ----------
        PL, PR, PT, PB = -7.75, -2.05, 3.0, -4.3
        frame = pane_frame(PL, PR, PT, PB)
        pane = EqPane(PL + 0.05, PR - 0.15, PT - 0.25, size=18, line_h=0.4)
        tpane = EqPane(PL + 0.05, PR - 0.15, PT - 0.25, size=15, line_h=0.31)

        # ---------- right: the 5 × 5 × 5 grid in 3D ----------
        proj = Proj3D(LO, HI, center=[3.05, -0.85, 0], scale=2.05, yaw=-0.62, pitch=0.42)
        yaw = ValueTracker(proj.yaw)
        cloud = [(w["zm"], tr["y"]) for tr in TOY["train"] for w in tr["a"]][::2]
        state = {"cloud": True, "cubes": []}   # cubes: [(cell, color, width)]
        t = lambda k: -1 + 2 * k / M

        def scene3d(yv):
            g = VGroup(proj.face_lines(M, yv, color=RULE, width=1.0))
            for ax in (0, 1):
                o = 1 - ax
                gsg = min((-1, 1), key=lambda s: proj.unit(*([s, 0, -1] if o == 0 else [0, s, -1]), yaw=yv)[1])
                for k in range(M):
                    q = [0, 0, -1]
                    q[o] = gsg * 1.14
                    q[ax] = t(k) + 1 / M
                    g.add(Text(str(k), font=MONO, font_size=12, color=MUTED).move_to(proj.unit(*q, yaw=yv)[0]))
                q = [0, 0, -1]
                q[o] = gsg * 1.42
                g.add(Text(f"z{sub(ax + 1)} → c{sub(ax + 1)}", font=MATH, font_size=16, color=INK2).move_to(proj.unit(*q, yaw=yv)[0]))
            X, Y = min(((x, y) for x in (-1, 1) for y in (-1, 1)), key=lambda c: proj.unit(c[0], c[1], 0, yaw=yv)[0][0])
            for k in range(M):
                g.add(Text(str(k), font=MONO, font_size=12, color=MUTED)
                      .next_to(proj.unit(X, Y, t(k) + 1 / M, yaw=yv)[0], LEFT, buff=0.1))
            g.add(Text("z₃ → c₃", font=MATH, font_size=16, color=INK2).move_to(proj.unit(X, Y, 1.22, yaw=yv)[0]))
            if state["cloud"]:
                pts = sorted(((proj.data(z, yv)[1], proj.data(z, yv)[0], y) for z, y in cloud), key=lambda q: -q[0])
                g.add(VGroup(*[Dot(p, radius=0.028, color=GOOD if y else BAD, fill_opacity=0.5) for _, p, y in pts]))
            for c, col, wd in state["cubes"]:
                g.add(proj.wire_box([t(v) for v in c], [t(v + 1) for v in c], yv, color=col, width=wd))
            return g

        space = [scene3d(yaw.get_value()).set_z_index(-1)]

        # The 3D picture is rebuilt by refresh() (remove the old one, add the new one) and turned frame
        # by frame by turn(). An updater that swaps the picture's children leaves the old picture in the
        # list of mobjects the scene redraws during an animation, which shows up as a ghost.
        def refresh():
            new = scene3d(yaw.get_value()).set_z_index(-1)
            self.remove(space[0])
            self.add(new)
            space[0] = new

        def turn(target, run_time, rate=smooth):
            y0, n = yaw.get_value(), max(1, int(run_time * config.frame_rate))
            for k in range(1, n + 1):
                yaw.set_value(y0 + (target - y0) * rate(k / n))
                refresh()
                self.wait(1 / config.frame_rate)
        legend = VGroup(Dot(radius=0.05, color=GOOD), Text("true answers", font=MATH, font_size=15, color=MUTED),
                        Dot(radius=0.05, color=BAD), Text("false answers (training tokens)", font=MATH, font_size=15, color=MUTED))
        legend.arrange(RIGHT, buff=0.1).move_to([3.05, -4.15, 0])

        # ---------- the answer ----------
        sentence = Text(" ".join(TOKENS), font=MONO, font_size=28, color=INK2).move_to([0, 3.82, 0])
        words, vis = [], 0
        for w in TOKENS:
            words.append(sentence[vis:vis + len(w)])
            vis += len(w)

        # 1. the grid and its cell widths
        self.play(FadeIn(frame), FadeIn(space[0]), FadeIn(legend), run_time=0.9)
        rows = [("h", "Grid: 5 cells per PCA dimension"),
                ("c", "min, max over every training token's (z₁, z₂, z₃)"),
                ("e", "uᵢ", "(maxᵢ − minᵢ) / 5")]
        for d in range(3):
            rows += [("e", f"u{sub(d + 1)}", f"({f2(HI[d])} − {p2(LO[d])}) / 5"),
                     ("e", "", f2(U[d]), WARM)]
        play(pane.set(rows), 0.6)
        self.wait(2.4)
        turn(-0.15, 3.0, there_and_back_with_pause)

        # 2. cells -> state numbers
        play(pane.set([
            ("h", "Abstract states"),
            ("e", "cᵢ", "trunc((zᵢ − minᵢ) / uᵢ)"),
            ("c", "cell number along dimension i, counted from 0"),
            ("e", "s̄", "c₁ + 5·c₂ + 25·c₃"),
            ("c", "one number per box, like a base-5 number"),
            ("e", "#boxes", "5³ = 125"),
            ("e", "#seen", f"{len(COUNTS)}   (boxes with at least one training token)"),
        ]), 0.6)
        self.wait(2.6)

        # 3. the example's tokens fall into their boxes
        self.play(FadeIn(sentence, shift=DOWN * 0.15), run_time=0.5)

        def make_ball():
            b = VGroup(Circle(radius=0.17, stroke_width=0, fill_color=WARM, fill_opacity=0.3),
                       Dot(radius=0.1, color=WARM))
            return b.set_z_index(5)

        spawn_y = sentence.get_bottom()[1] - 0.28
        balls = []
        for i, w in enumerate(RA):
            ball = make_ball().move_to([words[i].get_x(), spawn_y, 0])
            balls.append(ball)
            z, c, s = w["zm"], w["c"], w["s"]
            rows = [("h", f'"{TOKENS[i]}" → which state?'),
                    ("e", "z", f"({f2(z[0])}, {f2(z[1])}, {f2(z[2])})")]
            for d in range(3):
                q = (z[d] - LO[d]) / U[d]
                rows += [("e", f"c{sub(d + 1)}", f"trunc(({f2(z[d])} − {p2(LO[d])}) / {f2(U[d])})"),
                         ("e", "", f"trunc({q:.2f})"),
                         ("e", "", f"{c[d]}", WARM)]
            rows += [("e", "s̄", "c₁ + 5·c₂ + 25·c₃"),
                     ("e", "", f"{c[0]} + 5·{c[1]} + 25·{c[2]}"),
                     ("e", "", f"{s}", WARM)]
            slow = i < 2
            play([words[i].animate.set_color(WARM), GrowFromCenter(ball)] + pane.set(rows), 0.5 if slow else 0.3)
            if slow:
                self.wait(3.4 if i == 0 else 1.8)
            state["cubes"] = [(tuple(x["c"]), DONE, 1.6) for x in RA[:i]] + [(tuple(c), WARM, 3)]
            refresh()
            play([ball.animate.move_to(proj.data(z, yaw.get_value())[0]),
                  words[i].animate.set_color(DONE)], 0.5 if slow else 0.4)
            if not slow:
                self.wait(0.5)
        # the last token is never used (the authors' loop range)
        play([words[-1].animate.set_color(DONE)] + pane.set([
            ("h", f'"{TOKENS[-1]}": not used'),
            ("c", "the authors' loop stops one token before the end,"),
            ("c", f"so {len(TOKENS)} tokens give {len(RA)} states"),
            ("e", "trace", " → ".join(f"s̄{w['s']}" for w in RA[:4])),
            ("e", "", " → ".join(f"s̄{w['s']}" for w in RA[4:])),
        ]), 0.5)
        self.wait(2.0)

        # 4. boxes -> DTMC nodes (the page's 12 most visited states + the example's states)
        node_ids = list(dict.fromkeys(TOP12 + [w["s"] for w in RA]))
        visits = {s: COUNTS[s]["pos"] + COUNTS[s]["neg"] for s in node_ids}
        # view chosen by a search for the largest gap between node circles (fits an 8 x 6 area)
        nproj = Proj3D(LO, HI, center=[0, 0, 0], scale=6.49, yaw=-0.55, pitch=0.35)
        cent = lambda s, pr, yv=None: pr.unit(*[t(v) + 1 / M for v in cell_of_state(s)], yaw=yv)[0]
        pos = {s: cent(s, nproj) for s in node_ids}
        bx = [min(q[0] for q in pos.values()), max(q[0] for q in pos.values())]
        by = [min(q[1] for q in pos.values()), max(q[1] for q in pos.values())]
        pos = {s: q - np.array([sum(bx) / 2 - 3.1, sum(by) / 2 + 0.45, 0]) for s, q in pos.items()}
        radius = {s: 0.12 + 0.035 * math.sqrt(visits[s]) for s in node_ids}
        run_ids = {w["s"] for w in RA}
        nodes, nlabels = {}, VGroup()
        for s in node_ids:
            nodes[s] = Circle(radius=radius[s], fill_color=NODE_FILL, fill_opacity=1,
                              stroke_color=WARM if s in run_ids else ACCENT,
                              stroke_width=2.4 if s in run_ids else 1.6).move_to(pos[s]).set_z_index(2)
            lab = Text(f"s̄{s}", font=MATH, font_size=14, color=WARM if s in run_ids else INK2)
            lab.next_to(nodes[s], DOWN, buff=0.05).set_z_index(6)
            lab.add_background_rectangle(color=config.background_color, opacity=0.8, buff=0.02)
            nlabels.add(lab)
        starts = {s_: proj.unit(*[t(v) + 1 / M for v in cell_of_state(s_)], yaw=yaw.get_value())[0] for s_ in node_ids}
        ghosts = [Circle(radius=0.05, fill_color=NODE_FILL, fill_opacity=1, stroke_color=ACCENT, stroke_width=1).move_to(starts[s])
                  for s in node_ids]
        state["cloud"] = False
        state["cubes"] = []
        refresh()
        play([space[0].animate.set_opacity(0.25), FadeOut(legend)] + [FadeIn(g) for g in ghosts] + pane.set([
            ("h", "DTMC nodes"),
            ("c", "each abstract state is one node;"),
            ("c", "drawn: the page's 12 most visited states"),
            ("c", f"+ the example's states ({', '.join(f's̄{s}' for s in dict.fromkeys(w['s'] for w in RA))})"),
            ("c", "node size ∝ √(visits)"),
        ]), 0.7)
        play([ReplacementTransform(g, nodes[s]) for g, s in zip(ghosts, node_ids)]
             + [b.animate.move_to(pos[w["s"]]).scale(0.7) for b, w in zip(balls, RA)]
             + [FadeIn(nlabels), FadeOut(space[0])], 1.3)

        # 5. edges: P̄ from the training counts
        BEND = -0.5
        edges = VGroup()
        for a in node_ids:
            for b in node_ids:
                p, n = P(a, b), TRANS.get((a, b), 0)
                on_path = any(r["a"] == a and r["b"] == b for r in RUN_TRANS)
                if n == 0 or (p < 0.06 and not on_path):
                    continue
                op = 0.2 + 0.75 * min(1.0, p * 2.5)
                if a == b:
                    if p < 0.12:
                        continue
                    edges.add(self_loop(pos[a], radius[a], ACCENT, 0.6 + 2.5 * p, op * 0.8, 0.07))
                else:
                    edges.add(curved_edge(pos[a], pos[b], radius[a], radius[b], BEND, ACCENT,
                                          0.6 + 8 * p, op, 0.07 + 0.2 * p))
        edges.set_z_index(-1)
        play(pane.set([
            ("h", "Transitions → weighted edges"),
            ("e", "P̄(s̄→s̄′)", "#(s̄→s̄′) / #(s̄→·)"),
            ("c", "#(s̄→s̄′): steps from s̄ to s̄′ in the 160 training traces"),
            ("c", "#(s̄→·): every step that leaves s̄;   edge width ∝ P̄"),
        ]), 0.5)
        self.play(LaggedStart(*[Create(e) for e in edges], lag_ratio=0.02), run_time=2.0)
        self.wait(1.2)

        a0 = RA[0]["s"]
        row = sorted(((b, n) for (a, b), n in TRANS.items() if a == a0), key=lambda q: -q[1])
        rows = [("h", f'One full row: leaving s̄{a0} ("{TOKENS[0]}")'),
                ("e", f"#(s̄{a0}→·)", " + ".join(str(n) for _, n in row)),
                ("e", "", f"{OUT[a0]}"),
                ("gap",)]
        rows += [("e", f"P̄(s̄{a0}→s̄{b})", f"{n} / {OUT[a0]} = {n / OUT[a0]:.3f}") for b, n in row]
        rows += [("e", "Σ", f"{OUT[a0]} / {OUT[a0]} = 1", WARM)]
        ring = Circle(radius=radius[a0] + 0.09, color=WARM, stroke_width=3).move_to(pos[a0])
        play(pane.clear() + tpane.set(rows) + [Create(ring)], 0.6)
        self.wait(4.5)

        # 6. the example's trace: one P̄ per step, multiplied
        play(tpane.clear() + [FadeOut(ring)] + pane.set([
            ("h", "Probability of the example's trace"),
            ("e", "P(trace)", "∏ₜ P̄(s̄ₜ → s̄ₜ₊₁)"),
        ]), 0.5)
        for i, r in enumerate(RUN_TRANS):
            a, b, n, o, p = r["a"], r["b"], r["n"], r["out"], r["p"]
            unseen = n == 0
            col = BAD if unseen else WARM
            if a == b:
                hl = self_loop(pos[a], radius[a] + 0.04, col, 4, 1, 0.14)
            else:
                hl = curved_edge(pos[a], pos[b], radius[a], radius[b], BEND, col, 4.5, 1, 0.18, dashed=unseen)
            hl.set_z_index(3)
            txt = (f"{n}/{o} = {p:.3f}    s̄{a}→s̄{b},  {TOKENS[i]} → {TOKENS[i + 1]}" if not unseen
                   else f"0/{o} = 0    s̄{a}→s̄{b}: never seen in training")
            row = ("e", "", txt, col) if i == 0 else ("×", txt, col)
            play([Create(hl), words[i].animate.set_color(WARM), words[i + 1].animate.set_color(WARM)]
                 + pane.append([row]), 0.7)
            play([words[i].animate.set_color(DONE)], 0.15)
        prod = 1.0
        for r in RUN_TRANS:
            prod *= r["p"]
        m_, e_ = f"{prod:.2e}".split("e")
        sci = f"{m_} × 10" + str(int(e_)).translate(str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻"))
        play(pane.append([("e", "", sci if prod else "0", WARM)]
                         + ([("c", "one step never seen in training makes the whole product 0")] if not prod else
                            [("c", "every step was seen in training, so no factor is 0")])), 0.5)
        self.wait(3.0)
