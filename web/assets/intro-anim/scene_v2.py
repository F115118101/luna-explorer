"""LUNA Explorer — intro animation v2: token -> h^(0) -> 32 blocks -> last hidden state -> PCA to 3
numbers -> a point in 3D, using luna_v3.html's running example and configuration
(Llama2-7b, paper Table IX: PCA 3 · Grid 5 per dim).

Layout: thin step bar on top, the prompt + answer under it, a computation pane on the left
(equations top to bottom, broken at every '=', '=' signs in one column) and the figures on the
right: the hidden vector (entries 1-6, 4095, 4096, as in the page's hidden-state table) and the
3D PCA space, which turns now and then so its depth can be read.

Numbers that come from the page (via toy-data.json, exported by ../export_toy.mjs):
  * the tokens, the last-layer entries shown (the page's hidden-state table at layer 32),
  * every token's PCA coordinates z = (z1, z2, z3) and the grey training cloud.
Simulated (seeded) here, consistent with those: token ids, the layer 1..31 vectors (a random walk
that ends at the page's layer-32 values), the attention / MLP split, W and mu entries; z1's
remaining-dimensions term is whatever makes the written sum equal the page's z1.

Kept next to the original scene.py so the two renders can be compared.

Run:
    node ../export_toy.mjs                         # after changing the page's toy data
    manim -ql --fps 15 scene_v2.py IntroV2          # preview
    manim -qh --fps 30 scene_v2.py IntroV2
    ffmpeg -i media/videos/scene_v2/1080p30/IntroV2.mp4 -c:v libx264 -crf 20 -preset slow -pix_fmt yuv420p -g 10 -keyint_min 10 -sc_threshold 0 -movflags +faststart -an intro-v2.mp4
"""
import os
import random
import sys

from manim import *

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from anim_common import EqPane, StepBar, Proj3D, pane_frame, load_toy, MATH, MONO  # noqa: E402

config.background_color = "#05070c"

TOY = load_toy()
TOKENS = [t.replace("▁", "_") for t in TOY["tokens"]]  # ▁ is drawn as several paths; the page's formulas write _ too
PROMPT = TOY["prompt"]
RUN = next(t for t in TOY["test"] if t["run"])
Z = RUN["z"]                                   # 9 tokens x 3
H32 = TOY["hidden32"]["rows"]                  # 9 tokens x 8 shown entries
DIMS = TOY["hidden32"]["cols"]                 # [1..6, 4095, 4096]
N_LAYERS = 32
SHOWN_LAYERS = [1, 8, 16, 31]
D_MODEL = 4096
VOCAB = 32000
N_SHOW = len(DIMS)

INK = "#E7EBF3"
INK2 = "#AEB7CA"
MUTED = "#7C869C"
DONE = "#3B4252"
RULE = "#232B3B"
AXIS = "#2F3A50"
ACCENT = "#8CB8FF"
WARM = "#F4C26B"

_SUP = str.maketrans("0123456789()-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁽⁾⁻")
_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def sup(s):
    return str(s).translate(_SUP)


def sub(s):
    return str(s).translate(_SUB)


def f2(v):
    return f"{v:.2f}".replace("-", "−")


def p2(v):
    return f"({f2(v)})" if v < 0 else f2(v)


def hl(l):
    """h with a layer superscript; 'l' -> modifier letter ˡ (there is no superscript ℓ)"""
    return "h" + sup("(" + str(l) + ")").replace("l", "ˡ")


# ---------- simulated pieces, consistent with the page's numbers (string seeds) ----------
def token_id(word):
    return random.Random(f"id|{word}").randrange(VOCAB)


def trajectory(ti):
    """h[l] for l = 0..32 (8 shown entries). h[32] = the page's table; h[31] .. h[0] walk
    backwards with h[l] = h[l-1] + attn[l] + mlp[l] exact at 2 decimals; h[0] = E[id]."""
    r = random.Random(f"traj|{ti}")
    h = [None] * (N_LAYERS + 1)
    attn, mlp = [None] * (N_LAYERS + 1), [None] * (N_LAYERS + 1)
    h[N_LAYERS] = [round(v, 2) for v in H32[ti]]
    for l in range(N_LAYERS, 0, -1):
        attn[l] = [round(r.gauss(0, 0.08), 2) for _ in range(N_SHOW)]
        mlp[l] = [round(r.gauss(0, 0.08), 2) for _ in range(N_SHOW)]
        h[l - 1] = [round(h[l][j] - attn[l][j] - mlp[l][j], 2) for j in range(N_SHOW)]
    return h, attn, mlp


def pca_terms(ti, h_last, z1):
    r = random.Random(f"pca|{ti}")
    W = [round(r.gauss(0, 0.15), 2) for _ in range(N_SHOW)]
    mu = [round(r.gauss(0, 0.05), 2) for _ in range(N_SHOW)]
    prods = [round(W[i] * (h_last[i] - mu[i]), 2) for i in range(N_SHOW)]
    rest = round(z1 - sum(prods), 2)
    return W, mu, prods, rest


class IntroV2(MovingCameraScene):

    def construct(self):
        self.camera.frame.set(width=16.0)

        # ---------- top: step bar + prompt and answer ----------
        steps = StepBar(["Tokenize", "Embed", f"Blocks ×{N_LAYERS}", "Last layer", "PCA → ℝ³"], 4.05)
        prompt = PROMPT.strip()
        caption = Text(prompt + "  " + " ".join(TOKENS), font=MONO, font_size=20, color=MUTED)
        vis = sum(1 for ch in prompt if not ch.isspace())
        assert len(caption.submobjects) == vis + sum(len(w) for w in TOKENS)
        caption[:vis].set_color(DONE)
        cap_words = []
        for w in TOKENS:
            cap_words.append(caption[vis:vis + len(w)])
            vis += len(w)
        caption.move_to([0, 3.45, 0])

        # ---------- left: computation pane ----------
        PL, PR, PT, PB = -7.75, -2.05, 2.95, -4.3
        frame = pane_frame(PL, PR, PT, PB)
        pane = EqPane(PL + 0.05, PR - 0.15, PT - 0.25, size=19, line_h=0.41)

        # ---------- right: vector row (entries 1-6, …, 4095, 4096) ----------
        RX = 3.0
        VEC_LABEL_Y, VEC_Y, IDX_Y = 2.75, 2.22, 1.88
        SLOT_W = 0.86
        cols = [0, 1, 2, 3, 4, 5, 7, 8]       # column 6 holds the "…"
        col_cx = [RX - 0.1 + (j - 4) * SLOT_W for j in range(9)]
        lbr = Text("[", font=MONO, font_size=30, color=MUTED).move_to([col_cx[0] - 0.62, VEC_Y, 0])
        rbr = Text("]", font=MONO, font_size=30, color=MUTED).move_to([col_cx[-1] + 0.5, VEC_Y, 0])
        ell = Text("…", font=MONO, font_size=24, color=MUTED).move_to([col_cx[6], VEC_Y, 0])
        idx_row = VGroup(*[Text(str(d), font=MONO, font_size=12, color=DONE).move_to([col_cx[c], IDX_Y, 0])
                           for d, c in zip(DIMS, cols)])
        vec_center = np.array([(lbr.get_x() + rbr.get_x()) / 2, VEC_Y, 0])
        brackets = VGroup(lbr, rbr, ell, idx_row)

        def make_vec(vals, color=INK):
            g = VGroup()
            for v, c in zip(vals, cols):
                t = Text(f2(v), font=MONO, font_size=20, color=color)
                t.move_to([col_cx[c] + 0.35, VEC_Y - 0.1, 0], aligned_edge=DR)
                g.add(t)
            return g

        def vec_label(s, color=INK2):
            t = Text(s, font=MATH, font_size=20, color=color)
            return t.move_to([lbr.get_left()[0], VEC_LABEL_Y, 0], aligned_edge=LEFT)

        # ---------- right: 3D PCA space ----------
        lo, hi = TOY["grid"]["lo"], TOY["grid"]["hi"]
        proj = Proj3D(lo, hi, center=[3.1, -1.25, 0], scale=1.7, yaw=-0.78, pitch=0.42)
        yaw = ValueTracker(proj.yaw)
        cloud = [w["zm"] for k, tr in enumerate(TOY["train"]) for w in tr["a"]][::4]
        TICKS = [(-30, -15, 0, 15, 30), (-15, 0, 15), (-15, 0, 15)]
        shown = []   # (z, word) of the tokens already projected

        def plot(yv):
            g = VGroup(proj.face_lines(4, yv, color=AXIS, width=1.0))
            # tick values: z1 and z2 along the front bottom edges, z3 up the left vertical edge
            for ax in (0, 1):
                o = 1 - ax
                gsg = min((-1, 1), key=lambda s: proj.unit(*([s, 0, -1] if o == 0 else [0, s, -1]), yaw=yv)[1])
                for v in TICKS[ax]:
                    q = [0, 0, -1]
                    q[o] = gsg * 1.14
                    q[ax] = proj.nrm(v, ax)
                    g.add(Text(f"{v}".replace("-", "−"), font=MONO, font_size=11, color=MUTED)
                          .move_to(proj.unit(*q, yaw=yv)[0]))
                q = [0, 0, -1]
                q[o] = gsg * 1.42
                g.add(Text(f"z{sub(ax + 1)}", font=MATH, font_size=17, color=INK2).move_to(proj.unit(*q, yaw=yv)[0]))
            X, Y = min(((x, y) for x in (-1, 1) for y in (-1, 1)), key=lambda c: proj.unit(c[0], c[1], 0, yaw=yv)[0][0])
            for v in TICKS[2]:
                g.add(Text(f"{v}".replace("-", "−"), font=MONO, font_size=11, color=MUTED)
                      .next_to(proj.unit(X, Y, proj.nrm(v, 2), yaw=yv)[0], LEFT, buff=0.08))
            g.add(Text("z₃", font=MATH, font_size=17, color=INK2).move_to(proj.unit(X, Y, 1.25, yaw=yv)[0]))
            pts = sorted(((proj.data(z, yv)[1], proj.data(z, yv)[0]) for z in cloud), key=lambda t: -t[0])
            g.add(VGroup(*[Dot(p, radius=0.025, color=MUTED, fill_opacity=0.45) for _, p in pts]))
            for z, word in shown:
                p = proj.data(z, yv)[0]
                g.add(Dot(p, radius=0.07, color=ACCENT).set_z_index(3),
                      Text(word, font=MONO, font_size=13, color=INK2).next_to(p, UR, buff=0.04))
            return g

        space = [plot(yaw.get_value()).set_z_index(-1)]

        # The 3D picture is rebuilt by refresh() (remove the old one, add the new one) and turned frame
        # by frame by turn(). An updater that swaps the picture's children leaves the old picture in the
        # list of mobjects the scene redraws during an animation, which shows up as a ghost.
        def refresh():
            new = plot(yaw.get_value()).set_z_index(-1)
            self.remove(space[0])
            self.add(new)
            space[0] = new

        def turn(target, run_time, rate=smooth):
            y0, n = yaw.get_value(), max(1, int(run_time * config.frame_rate))
            for k in range(1, n + 1):
                yaw.set_value(y0 + (target - y0) * rate(k / n))
                refresh()
                self.wait(1 / config.frame_rate)
        space_title = Text("PCA space ℝ³ (grey: training tokens)", font=MATH, font_size=15, color=MUTED)
        space_title.move_to([-1.55, 1.45, 0], aligned_edge=LEFT)

        slots = {}

        def swap(key, new):
            anims = []
            old = slots.get(key)
            if old is not None:
                anims.append(FadeOut(old))
            if new is not None:
                anims.append(FadeIn(new))
            slots[key] = new
            return anims

        def play(anims, rt):
            if anims:
                self.play(*anims, run_time=rt)
            else:
                self.wait(rt)

        cap_words[0].set_color(WARM)
        self.play(FadeIn(steps), FadeIn(caption, shift=UP * 0.2), FadeIn(frame), FadeIn(space[0]),
                  FadeIn(space_title), FadeIn(brackets), run_time=0.8)

        results = []

        def to_space(ti, word, z, rt):
            z_txt = Text(f"z = [{f2(z[0])}, {f2(z[1])}, {f2(z[2])}]", font=MATH, font_size=24,
                         color=WARM).move_to(vec_center)
            old_vec = slots.pop('vec')
            play([old_vec.animate.arrange(RIGHT, buff=0).scale(0.05).move_to(vec_center).set_opacity(0),
                  FadeIn(z_txt, scale=1.4), *swap('vlabel', None)], rt * 0.4)
            self.remove(old_vec)
            target = proj.data(z, yaw.get_value())[0]
            play([z_txt.animate.scale(0.35).move_to(target)], rt * 0.35)
            ring = Circle(radius=0.07, color=WARM, stroke_width=2).move_to(target)
            shown.append((z, word))
            refresh()
            play([FadeOut(z_txt), ring.animate.scale(4).set_stroke(opacity=0)], rt * 0.25)
            self.remove(ring)
            results.append((word, z))

        # ===================== token 1: every substitution written out =====================
        ti, word, z = 0, TOKENS[0], Z[0]
        tid = token_id(word)
        h, attn, mlp = trajectory(ti)

        play(steps.go(0) + pane.set([
            ("h", "1 · Tokenize"),
            ("c", "SentencePiece: _ (▁ on the page) marks a space before the token"),
            ("e", f'id("{word}")', f"{tid}"),
            ("c", f"vocabulary size = {VOCAB};  ids simulated"),
        ]), 0.6)
        self.wait(1.6)

        play(steps.go(1) + pane.set([
            ("h", "2 · Embed"),
            ("e", hl(0), f"E[{tid}]"),
            ("c", f"E ∈ ℝ{sup(VOCAB)}ˣ{sup(D_MODEL)}: token table, one row per id"),
            ("c", "Llama-2 adds no position vector here (RoPE inside attention)"),
            ("gap",),
            ("e", f"{hl(0)}₁", f"E[{tid}, 1]"),
            ("e", "", f2(h[0][0]), WARM),
        ]) + swap('vlabel', vec_label(f"{hl(0)} ∈ ℝ{sup(D_MODEL)}"))
          + swap('vec', make_vec(h[0])), 0.7)
        self.wait(2.4)

        play(steps.go(2) + pane.set([
            ("h", f"3 · Transformer blocks,  ℓ = 1 to {N_LAYERS}"),
            ("e", "h'", f"{hl('l-1')} + Attn(RMSNorm({hl('l-1')}))"),
            ("e", hl("l"), "h' + MLP(RMSNorm(h'))"),
            ("c", "Attn₁, MLP₁ = entry 1 of each sub-layer's output"),
            ("gap",),
        ], eq_x=pane._eq_x_for([("e", f"{hl(SHOWN_LAYERS[-1])}₁", "")])), 0.6)
        self.wait(1.4)
        for k, L in enumerate(SHOWN_LAYERS):
            rows = [
                ("e", f"{hl(L)}₁", f"{hl(L - 1)}₁ + Attn₁ + MLP₁"),
                ("e", "", f"{f2(h[L - 1][0])} + {p2(attn[L][0])} + {p2(mlp[L][0])}"),
                ("e", "", f2(h[L][0]), WARM),
            ]
            anims = (pane.pop(3) if k else []) + pane.append(rows)
            anims += swap('vlabel', vec_label(f"{hl(L)} ∈ ℝ{sup(D_MODEL)}   (block ℓ = {L} of {N_LAYERS})"))
            anims += swap('vec', make_vec(h[L]))
            play(anims, 0.5)
            self.wait(1.3 if k == 0 else 0.6)

        L = N_LAYERS
        play(steps.go(3) + pane.set([
            ("h", "4 · Last layer = concrete state"),
            ("e", "h", f"{hl(L)} ∈ ℝ{sup(D_MODEL)}"),
            ("c", "Hugging Face hidden_states[32], after the final RMSNorm"),
            ("c", "(the authors' code: block_idx = 31)"),
            ("c", "these entries = the page's hidden-state table"),
        ]) + swap('vlabel', vec_label(f"h = {hl(L)} ∈ ℝ{sup(D_MODEL)}   (last layer)", WARM))
          + swap('vec', make_vec(h[L], WARM)), 0.6)
        self.wait(2.2)

        W, mu, prods, rest = pca_terms(ti, h[L], z[0])
        terms = [f"{p2(W[i])}·({f2(h[L][i])} − {p2(mu[i])})" for i in range(N_SHOW)]
        sum_terms = [f2(prods[0])] + [p2(v) for v in prods[1:]] + [p2(rest)]
        play(steps.go(4) + pane.set([
            ("h", "5 · PCA to 3 numbers"),
            ("e", "z", f"Wᵀ(h − μ) ∈ ℝ³"),
            ("c", f"W ∈ ℝ{sup(D_MODEL)}ˣ³: top-3 principal axes,  μ ∈ ℝ{sup(D_MODEL)}: mean"),
            ("e", "z₁", f"Σᵢ₌₁{sup(D_MODEL)} W[i,1]·(hᵢ − μᵢ)"),
            ("c", "written out: i = 1 to 6, then i = 4095, 4096, then the rest"),
            ("e", "", f"{terms[0]} + {terms[1]}"),
            ("+", f"{terms[2]} + {terms[3]}"),
            ("+", f"{terms[4]} + {terms[5]}"),
            ("+", f"{terms[6]} + {terms[7]}"),
            ("+", f"Σᵢ₌₇{sup(4094)} W[i,1]·(hᵢ − μᵢ)"),
            ("e", "", " + ".join(sum_terms[:4])),
            ("+", " + ".join(sum_terms[4:])),
            ("e", "", f2(z[0]), WARM),
            ("c", "z₂, z₃: the same sum with columns 2 and 3 of W"),
            ("e", "z₂", f2(z[1]), WARM),
            ("e", "z₃", f2(z[2]), WARM),
        ]), 0.7)
        self.wait(5.0)
        to_space(ti, word, z, 1.8)
        # turn the space so its depth can be read
        turn(-0.15, 2.4, there_and_back_with_pause)
        self.wait(0.3)

        # ===================== tokens 2..9: compact pane =====================
        rt = 2.6
        for ti in range(1, len(TOKENS)):
            word, z = TOKENS[ti], Z[ti]
            tid = token_id(word)
            h, attn, mlp = trajectory(ti)
            st = rt / 6
            anims = [cap_words[ti - 1].animate.set_color(DONE), cap_words[ti].animate.set_color(WARM)]
            play(anims + steps.go(0) + pane.set([
                ("h", f'token {ti + 1} of {len(TOKENS)}: "{word}"'),
                ("e", f'id("{word}")', f"{tid}"),
            ]), st)
            play(steps.go(1) + pane.append([("e", hl(0), f"E[{tid}]")])
                 + swap('vlabel', vec_label(f"{hl(0)} ∈ ℝ{sup(D_MODEL)}"))
                 + swap('vec', make_vec(h[0])), st)
            play(steps.go(2) + pane.append([("e", hl("l"), f"h' + MLP(RMSNorm(h')),  ℓ = 1 to {N_LAYERS}")]), st * 0.4)
            for L in SHOWN_LAYERS:
                play(swap('vlabel', vec_label(f"{hl(L)} ∈ ℝ{sup(D_MODEL)}")) + swap('vec', make_vec(h[L])), st * 0.15)
            play(steps.go(3) + pane.append([("e", "h", hl(N_LAYERS))])
                 + swap('vlabel', vec_label(f"h = {hl(N_LAYERS)} ∈ ℝ{sup(D_MODEL)}", WARM))
                 + swap('vec', make_vec(h[N_LAYERS], WARM)), st)
            play(steps.go(4) + pane.append([("e", "z", "Wᵀ(h − μ)"),
                                            ("e", "", f"[{f2(z[0])}, {f2(z[1])}, {f2(z[2])}]", WARM)]), st)
            to_space(ti, word, z, st * 1.4)
            rt = max(1.2, rt * 0.85)

        # ===================== summary + one more turn =====================
        rows = [("h", f"{len(TOKENS)} tokens → {len(TOKENS)} points in ℝ³")]
        rows += [("e", f'z("{w}")', f"({f2(z[0])}, {f2(z[1])}, {f2(z[2])})") for w, z in results]
        play([cap_words[-1].animate.set_color(DONE)] + steps.go(None) + pane.set(rows)
             + [FadeOut(brackets)] + swap('vlabel', None), 0.6)
        turn(-0.05, 4.0, there_and_back_with_pause)
        self.wait(1.5)
