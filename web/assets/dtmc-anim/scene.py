"""LUNA Explorer — section-04 animation: "grid cells -> DTMC nodes, one sentence's trace through it"

Baked with Manim like intro-anim / semantics-anim, embedded as a plain <video>.

What it shows:
  1. The concrete states are the REAL `states2d` train points from web/luna-data.json (our own
     GPT-2, PCA-2) — the same points §03's Partition explorer draws — on the same 5x5 grid
     §03 uses by default (train bounding box + 0.4 padding, n = 5). Grid and grey dots appear
     together, then every dot takes its cell's color.
  2. "Water boils at 100 degrees Celsius at sea level ." — each token becomes a bright ball and
     drops, in order, into a cell. Which cell is HAND-PICKED (see PATH), not a real projection.
  3. Every non-empty cell turns into a DTMC node (empty cells have no state and vanish), the
     layout relaxes into a graph, and directed curved edges appear, styled like §04's
     Transition-model builder: node size ∝ sqrt(stationary π), edge width/opacity ∝ P-hat.
     P-hat is estimated by counting SIMULATED traces (see transition_model); the sentence is one
     of them, so its every step is an edge, and each ball stays in its node.
  4. The sentence's trace is highlighted in yellow, edge by edge.

Run:
    manim -qh --fps 30 scene.py Dtmc
Output:
    media/videos/scene/1080p30/Dtmc.mp4
then re-encode with a keyframe every 10 frames, or dragging the page's seek bar lags (the frame only catches up
after decoding from the previous keyframe; ffmpeg's default is one keyframe per 250 frames = 8.3 s):
    ffmpeg -i <rendered>.mp4 -c:v libx264 -crf 20 -preset slow -pix_fmt yuv420p -g 10 -keyint_min 10 -sc_threshold 0 -movflags +faststart -an dtmc.mp4
"""
import json
import math
import os
import random

from manim import *

config.background_color = "#05070c"

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "..", "..", "luna-data.json")

MONO = "Consolas"
INK = "#E7EBF3"
INK2 = "#AEB7CA"
MUTED = "#7C869C"
DONE = "#3B4252"
RULE = "#2F3A50"
ACCENT = "#8CB8FF"
WARM = "#F4C26B"
NODE_FILL = "#161D2B"

# luna.html's PAL with its yellows (#F4C26B, #E8D27A) and lime (#B5D66B) taken out, so only the
# sentence's balls and trace read as yellow
PAL = ["#8CB8FF", "#5FD3A0", "#C2B4FF", "#FF9E7B", "#6FD8E8", "#F28BC0", "#9AA8FF"]

GRID_N = 5
TOKENS = [
    "Water", "boils", "at", "100", "degrees", "Celsius", "at", "sea", "level", "."
]
# (col, row-from-bottom) cell each token is dropped into — same (i, j) indexing as §03's grid
PATH = [(2, 3), (2, 4), (3, 4), (3, 3), (4, 3), (4, 2), (3, 2), (2, 2), (1, 0), (2, 0)]

with open(DATA_PATH, encoding="utf-8") as f:
    TRAIN = [(p["x"], p["y"]) for p in json.load(f)["states2d"]["points"] if p["split"] == "train"]


def cell_of(x, y, box):
    x0, x1, y0, y1 = box
    i = min(GRID_N - 1, int((x - x0) / (x1 - x0) * GRID_N))
    j = min(GRID_N - 1, int((y - y0) / (y1 - y0) * GRID_N))
    return i, j


def color_cells(cells):
    """Greedy coloring so no two touching cells (8-neighbourhood) share a color."""
    out = {}
    for c in sorted(cells):
        used = {out.get((c[0] + di, c[1] + dj)) for di in (-1, 0, 1) for dj in (-1, 0, 1)}
        out[c] = next(p for p in PAL if p not in used)
    return out


def graph_layout(nodes, center):
    """Node positions: each cell keeps its column/row, but rows with no states are dropped,
    alternate rows are offset by half a step (so edges don't stack up on straight lines) and a
    small seeded jitter keeps it from looking like a spreadsheet."""
    rows = sorted({c[1] for c in nodes}, reverse=True)
    jr = random.Random("layout")
    out = {}
    for c in nodes:
        r = rows.index(c[1])
        out[c] = center + np.array([(c[0] - 2) * 2.15 + (0.5 if r % 2 else -0.2) + jr.uniform(-.12, .12),
                                    ((len(rows) - 1) / 2 - r) * 1.6 + jr.uniform(-.1, .1), 0])
    return out


def transition_model(nodes, counts, layout, n_traces=400, length=12):
    """Simulated, but estimated the way LUNA does it: a hidden "true" P (nearby, dense cells
    likelier, with log-normal noise so some routes are much stronger than others) generates
    many traces; P-hat = #(a->b) / #(out of a). The sentence itself is one more trace, so each
    of its steps has count >= 1 and is always an edge. Returns P-hat and its stationary π."""
    r = random.Random("traces")
    n = len(nodes)
    true = []
    for a in nodes:
        w = [math.exp(-np.linalg.norm(layout[a] - layout[b]) / 1.6) * math.sqrt(counts[b])
             * math.exp(r.gauss(0, 1.0)) * (1.4 if a == b else 1) for b in nodes]
        t = sum(w)
        true.append([v / t for v in w])
    start_w = [counts[c] for c in nodes]
    cnt = [[0] * n for _ in range(n)]
    for _ in range(n_traces):
        s0 = r.choices(range(n), weights=start_w)[0]
        for _ in range(length):
            s1 = r.choices(range(n), weights=true[s0])[0]
            cnt[s0][s1] += 1
            s0 = s1
    idx = {c: k for k, c in enumerate(nodes)}
    for a, b in zip(PATH, PATH[1:]):
        cnt[idx[a]][idx[b]] += 1
    P = [[v / sum(row) for v in row] if sum(row) else [1 / n] * n for row in cnt]
    pi = [1 / n] * n
    for _ in range(500):
        pi = [sum(pi[i] * P[i][j] for i in range(n)) for j in range(n)]
    return P, pi, cnt


def _tip(arc, color, opacity, tip):
    arc.add_tip(tip_length=tip, tip_width=tip * 0.9)
    arc.get_tip().set_fill(color, opacity=opacity).set_stroke(width=0)
    return arc


def curved_edge(ca, cb, ra, rb, angle, color, width, opacity, tip):
    """Arc from node a to node b (always bowing to the same side of travel, so a->b and b->a
    never overlap), trimmed to the two circles' rims, with an arrow tip at b."""
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
    return _tip(arc, color, opacity, tip)


def self_loop(c, r, color, width, opacity, tip):
    a1 = c + r * np.array([math.cos(1.95), math.sin(1.95), 0])
    a2 = c + r * np.array([math.cos(1.19), math.sin(1.19), 0])
    loop = ArcBetweenPoints(a1, a2, angle=-3.6, stroke_color=color, stroke_width=width,
                            stroke_opacity=opacity)
    return _tip(loop, color, opacity, tip)


class Dtmc(MovingCameraScene):
    def construct(self):
        self.camera.frame.set(width=16.0)

        # ---------- §03's grid over the real train points ----------
        xs, ys = [p[0] for p in TRAIN], [p[1] for p in TRAIN]
        box = (min(xs) - .4, max(xs) + .4, min(ys) - .4, max(ys) + .4)
        x0, x1, y0, y1 = box
        GW, GH, GC = 10.0, 5.4, np.array([0, -0.8, 0])

        def to_scene(x, y):
            return GC + np.array([((x - x0) / (x1 - x0) - .5) * GW,
                                  ((y - y0) / (y1 - y0) - .5) * GH, 0])

        cw, ch = GW / GRID_N, GH / GRID_N

        def cell_center(c):
            return GC + np.array([(c[0] + .5) * cw - GW / 2, (c[1] + .5) * ch - GH / 2, 0])

        members = {}
        for x, y in TRAIN:
            members.setdefault(cell_of(x, y, box), []).append(to_scene(x, y))
        nodes = sorted(members)
        counts = {c: len(v) for c, v in members.items()}
        colors = color_cells(nodes)
        assert all(c in members for c in PATH)

        rects = {}
        for i in range(GRID_N):
            for j in range(GRID_N):
                rects[(i, j)] = Rectangle(width=cw, height=ch, stroke_color=RULE, stroke_width=1.2,
                                          fill_color=NODE_FILL, fill_opacity=0).move_to(
                                              cell_center((i, j)))
        dots = {c: VGroup(*[Dot(p, radius=0.036, color=MUTED, fill_opacity=0.6)
                            for p in members[c]]).set_z_index(1) for c in nodes}

        # ---------- sentence + caption ----------
        sentence = Text(" ".join(TOKENS), font=MONO, font_size=30, color=INK2)
        sentence.move_to([0, 3.75, 0])
        words, vis = [], 0
        for w in TOKENS:
            words.append(sentence[vis:vis + len(w)])
            vis += len(w)

        caption = [Text("concrete states (PCA-2, real GPT-2) on a 5x5 grid",
                        font=MONO, font_size=18, color=MUTED).move_to([0, 3.05, 0])]

        def change_caption(text, color=MUTED, run_time=0.4):
            old = caption[0]
            new = Text(text, font=MONO, font_size=18, color=color).move_to(old)
            self.play(FadeOut(old, shift=UP * 0.1), FadeIn(new, shift=UP * 0.1),
                      run_time=run_time)
            caption[0] = new

        # 1. grid and grey dots together
        self.play(FadeIn(VGroup(*rects.values())), FadeIn(VGroup(*dots.values())),
                  FadeIn(caption[0]), run_time=0.9)
        self.wait(0.4)

        # 2. every cell is one abstract state -> its dots take its color
        change_caption("each cell = one abstract state", INK2)
        self.play(*[dots[c].animate.set_color(colors[c]).set_opacity(0.75) for c in nodes],
                  *[rects[c].animate.set_fill(colors[c], opacity=0.07).set_stroke(colors[c], opacity=0.6)
                    for c in nodes],
                  run_time=0.9)
        self.wait(0.4)

        # 3. tokens -> balls dropped into their cells
        self.play(FadeIn(sentence, shift=DOWN * 0.15), run_time=0.5)
        change_caption("each token's concrete state falls into one cell", INK2)

        def make_ball():
            ball = VGroup(Circle(radius=0.2, stroke_width=0, fill_color=WARM, fill_opacity=0.3),
                          Dot(radius=0.12, color=WARM))
            ball.set_z_index(5)
            return ball

        spawn_y = sentence.get_bottom()[1] - 0.25
        lr = random.Random("land")
        balls = []
        for i, c in enumerate(PATH):
            ball = make_ball().move_to([words[i].get_x(), spawn_y, 0])
            balls.append(ball)
            land = lr.choice(members[c])
            self.play(words[i].animate.set_color(WARM), GrowFromCenter(ball), run_time=0.22)
            self.play(ball.animate.move_to(land), words[i].animate.set_color(DONE),
                      rects[c].animate.set_stroke(WARM, width=2.4, opacity=1), run_time=0.42)
        self.wait(0.4)

        # ---------- transition model + graph layout ----------
        layout = graph_layout(nodes, GC)
        P, pi, cnt = transition_model(nodes, counts, layout)
        k_of = {c: k for k, c in enumerate(nodes)}
        radii = [0.24 + 0.4 * math.sqrt(p) for p in pi]
        node_pos = [layout[c] for c in nodes]

        # 4. cells -> circles (DTMC nodes); dots pulled inside, empty cells vanish
        change_caption("abstract states become DTMC nodes")
        anims = []
        for c in nodes:
            k = k_of[c]
            ctr = cell_center(c)
            circ = Circle(radius=radii[k], fill_color=NODE_FILL, fill_opacity=1,
                          stroke_color=WARM if c in PATH else colors[c],
                          stroke_width=2.4 if c in PATH else 1.6).move_to(ctr)
            anims.append(Transform(rects[c], circ))
            g = dots[c]
            half_diag = max(0.05, max(np.linalg.norm(d.get_center() - ctr) for d in g))
            anims.append(g.animate.scale(radii[k] * 0.78 / half_diag, about_point=ctr))
            if c in PATH:
                anims.append(balls[PATH.index(c)].animate.move_to(ctr))
        anims += [FadeOut(rects[c]) for c in rects if c not in members]
        self.play(*anims, run_time=1.1)

        # 5. relax the grid into the graph layout
        move = []
        for c in nodes:
            grp = VGroup(rects[c], dots[c])
            if c in PATH:
                grp.add(balls[PATH.index(c)])
            move.append(grp.animate.shift(node_pos[k_of[c]] - cell_center(c)))
        self.play(*move, run_time=1.0)

        # 6. directed, curved, weighted edges — like §04's Transition-model builder
        change_caption("transitions between states become weighted, directed edges")
        BEND = -0.55
        edge_mobs = VGroup()
        path_pairs = {(k_of[a], k_of[b]) for a, b in zip(PATH, PATH[1:])}
        for a in range(len(nodes)):
            for b in range(len(nodes)):
                p = P[a][b]
                # drop one-off transitions (noise), but never the sentence's own steps
                if (p < 0.03 or cnt[a][b] < 2) and (a, b) not in path_pairs:
                    continue
                op = 0.16 + 0.8 * min(1.0, p * 2.2)
                if a == b:
                    if p < 0.12:
                        continue
                    # kept thin and small: a self-loop is one more edge, not the node's outline
                    e = self_loop(node_pos[a], radii[a], ACCENT, 0.6 + 2.2 * p, op * 0.8, 0.08)
                else:
                    e = curved_edge(node_pos[a], node_pos[b], radii[a], radii[b], BEND, ACCENT,
                                    0.5 + 10 * p, op, 0.07 + 0.22 * p)
                edge_mobs.add(e)
        edge_mobs.set_z_index(-1)
        self.play(LaggedStart(*[Create(e) for e in edge_mobs], lag_ratio=0.02), run_time=2.0)
        self.wait(0.5)

        # 7. this sentence's trace in yellow
        change_caption("this sentence's trace through the DTMC", WARM)
        labels = VGroup()
        for i, c in enumerate(PATH):
            k = k_of[c]
            # label goes on whichever side points furthest away from its own trace edges
            nb = [node_pos[k_of[n]] - node_pos[k] for n in PATH[max(0, i - 1):i + 2] if n != c]
            nb = [d / np.linalg.norm(d) for d in nb]
            side = max([DOWN, UP, LEFT, RIGHT], key=lambda v: min(-np.dot(v, d) for d in nb))
            lab = Text(TOKENS[i], font=MONO, font_size=15, color=WARM)
            lab.next_to(rects[c], side, buff=0.07)
            lab.add_background_rectangle(color=config.background_color, opacity=0.85, buff=0.04)
            lab.set_z_index(6)
            labels.add(lab)

        self.play(words[0].animate.set_color(WARM), FadeIn(labels[0]),
                  Indicate(balls[0], color=WARM, scale_factor=1.3), run_time=0.4)
        for i in range(1, len(PATH)):
            a, b = k_of[PATH[i - 1]], k_of[PATH[i]]
            hl = curved_edge(node_pos[a], node_pos[b], radii[a], radii[b], BEND, WARM, 5, 1, 0.2)
            hl.set_z_index(3)
            self.play(Create(hl), words[i].animate.set_color(WARM), FadeIn(labels[i]),
                      Indicate(balls[i], color=WARM, scale_factor=1.3), run_time=0.45)
        self.wait(2.0)
