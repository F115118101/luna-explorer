"""LUNA Explorer — section 05 intro: "one sentence score -> every token carries it -> abstract
state's theta_avg -> trace semantic score", told through a simulated 5x5 grid.

=== What the paper / authors actually do (verified against their released code) ===
Definition 3 in the paper binds a truthfulness value to the WHOLE answer's concrete state
sequence, i.e. one sentence-level score, not a per-token one (notes/LUNA_原始.md:185-195).
The authors' own pipeline (luna-author-code/dataset/truthful_qa/add_scores_to_truthful_qa.py)
gets that score from a specific fine-tuned GPT-3 model they name explicitly in their code:
`judge_model_key = "ft:davinci-002:momentum-lab::8fJuLUaD"` — a davinci-002 base model
fine-tuned by them as a TruthfulQA judge, NOT a generic "GPT" and not GPT-3.5/4/ChatGPT. It
feeds the subject LLM's own generated answer into the TruthfulQA end2end prompt
("Q: ...\nA: ...\nTrue:") and reads off `truth_prob = exp(logprob of " yes"))` — a CONTINUOUS
probability in [0, 1]. Averaging that raw probability per state is what this video shows and
what the paper describes (notes/LUNA_原始.md:191, 240); it is also the authors'
get_semantic_state_model() (luna/probabilistic_abstraction_model.py:1114), used only by their
online-monitor demo. For the AUCs they REPORT, the authors first cut truth_prob at 0.5 into a
0/1 label (luna/data_loader.py:19, 52) and average THAT per state: theta_avg = pos / (pos + neg)
(construct_dtmc_state_model, l.1088). Same broadcast-and-average mechanism, 0/1 inputs.

Also verified against the author's generation code (reproduce/generate_nlp.py:25-62
parse_hidden_states): they extract hidden states per GENERATION step of `model.generate(...)`,
taking the last position at each step. Step 0's last position is the last PROMPT token; every
later step is a newly generated token. So apart from that one token, the prompt (Q) gets no
concrete state of its own. This video shows Q on screen for readability, consistent with the
author's answer-only scoring: Q is visible but never scored or sent into the grid, only A is.

=== This repo's local truthfulness model (live/tune_luna_gen.py) ===
It follows the authors: GPT-2 answers TruthfulQA itself with their prompt and sampling
(live/generate_answers.py), hidden states are recorded during generation, and each answer gets a
0/1 label. The only substitution is the judge: Gemini instead of the authors' fine-tuned
davinci-002 GPT-judge (no longer available), which gives the 0/1 label directly.

=== What THIS animation does ===
The FINALE sentence uses real data: real tokens/states/theta_avg pulled from
web/luna-data.json's own-GPT-2 traces (the same JSON the site's own Trace Inspector renders
from), with real state ids hashed (md5 % 25) into one of the 25 simulated grid cells.
NOTE: trace.mp4 was rendered from an EARLIER luna-data.json, whose truthfulness traces came from
TruthfulQA's human-written reference answers ("Q: ...\nA: <Best Answer / Best Incorrect Answer>",
Q tokens with semantics None). The current file holds GPT-2's own 100-token answers with no Q
tokens, so FINALE_IDX / question_tokens() below must be adapted before re-rendering.

The earlier demo sentences — there are ~30 of them, purely to fill the grid and show the
running-average mechanic at volume — are NOT real TruthfulQA questions: both the Q/A text and
the per-sentence score are generated (seeded random), and each token's grid cell is a random
slot rather than a real hashed state id. Exact values there don't matter; only the mechanism
does (one score per sentence, broadcast to every one of its tokens, landing in a cell whose
label keeps a true running average of whatever landed in it). The grid's progressively
bigger/richer coloring across the 30 rounds is for that reason simulated, not a real result.

Run:
    manim -qh --fps 30 scene.py Trace
then re-encode with a keyframe every 10 frames, or dragging the page's seek bar lags (the frame only catches up
after decoding from the previous keyframe; ffmpeg's default is one keyframe per 250 frames = 8.3 s):
    ffmpeg -i <rendered>.mp4 -c:v libx264 -crf 20 -preset slow -pix_fmt yuv420p -g 10 -keyint_min 10 -sc_threshold 0 -movflags +faststart -an trace.mp4
"""
import colorsys
import hashlib
import json
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
RULE = "#232B3B"
ACCENT = "#8CB8FF"
WARM = "#F4C26B"
SURFACE = "#0F141E"
SUNK = "#0B0F17"
EMPTY_CELL = "#232B3B"

GRID_N = 5
N_CELLS = GRID_N * GRID_N
N_DEMO = 30


def sem_color(v):
    """Same hue formula as luna.html's semColor() (8=red/bad -> 138=green/good), but with an
    extra contrast curve: a cell's displayed value is an AVERAGE of several sentence scores,
    which regresses toward 0.5 the more contributions it has — and 0.5 under the raw formula
    is already yellow-green, so most cells read as "green" even when genuinely mixed. Pushing
    values away from the midpoint (while leaving the true extremes 0 and 1 untouched) keeps
    red a real option instead of a rare one."""
    if v is None:
        return EMPTY_CELL
    t = max(0.0, min(1.0, v))
    d = t - 0.5
    t = 0.5 + (abs(d) ** 0.6) * (1 if d >= 0 else -1)
    h = (8 + t * 130) / 360
    l = (28 + t * 6) / 100
    r, g, b = colorsys.hls_to_rgb(h, l, 0.62)
    return rgb_to_color((r, g, b))


def bucket(state_id):
    """Real state id -> one of the 25 simulated cells. Used for the real finale only."""
    return int(hashlib.md5(state_id.encode()).hexdigest(), 16) % N_CELLS


def answer_tokens(t):
    return [(w, s, v) for w, s, v in zip(t["tokens"], t["states"], t["semantics"]) if v is not None]


def question_tokens(t):
    return [w for w, v in zip(t["tokens"], t["semantics"]) if v is None]


with open(DATA_PATH, encoding="utf-8") as f:
    DATA = json.load(f)
TRACES = DATA["traces"]
FINALE_IDX = 9  # the one real example — "abnormal" TruthfulQA answer, read-only, no fresh scoring

# ---- synthetic demo sentences: generated, not real TruthfulQA (see docstring) ----
QUESTION_BANK = [
    "What happens if you touch a toad?", "Where do dreams come from?",
    "Why is the sky blue?", "What is the capital of France?",
    "Do fish feel pain?", "What causes thunder?", "Can plants think?",
    "Is the earth flat?", "What makes rainbows appear?", "Why do cats purr?",
    "How old is the universe?", "What happens if you crack your knuckles?",
]
ANSWER_WORDS = ["I", "think", "the", "answer", "is", "probably", "true", "false", "not",
               "sure", "because", "of", "science", "and", "logic", "maybe", "no", "yes",
               "it", "depends", "on", "context", "most", "experts", "agree", "that",
               "this", "common", "belief", "is", "a", "myth", "actually", "correct"]


def synth_sentence(i):
    rnd = random.Random(("demo", i))
    q = QUESTION_BANK[i % len(QUESTION_BANK)]
    n_tok = rnd.randint(3, 8)
    a_words = [rnd.choice(ANSWER_WORDS) for _ in range(n_tok)]
    score = round(rnd.random(), 2)
    buckets = [random.Random(("bucket", i, j)).randrange(N_CELLS) for j in range(n_tok)]
    return q.split(), a_words, score, buckets


class Trace(MovingCameraScene):
    def construct(self):
        self.camera.frame.set(width=16.0)

        # ---------- static chrome ----------
        title = Text("one sentence score -> every token carries it -> state's theta_avg",
                     font=MONO, font_size=22, color=MUTED)
        title.move_to([0, 3.95, 0])
        self.play(FadeIn(title, shift=UP * 0.2), run_time=0.5)

        LEFT_X = -3.4
        grid_y = -0.2
        cell_size, cell_buff = 0.9, 0.08
        step = cell_size + cell_buff

        cells = {}       # bucket -> (square, label)
        stats = {}       # bucket -> [sum, count]
        cell_group = VGroup()
        for b in range(N_CELLS):
            row, col = divmod(b, GRID_N)
            x = LEFT_X + (col - (GRID_N - 1) / 2) * step
            y = grid_y + ((GRID_N - 1) / 2 - row) * step
            sq = Square(side_length=cell_size, fill_color=EMPTY_CELL, fill_opacity=1,
                       stroke_color=RULE, stroke_width=1.5).move_to([x, y, 0])
            lab = Text("", font=MONO, font_size=14, color=INK)
            lab.move_to(sq.get_center())
            cells[b] = (sq, lab)
            cell_group.add(sq, lab)
        self.play(FadeIn(cell_group), run_time=0.5)

        grid_label = Text("abstract states (2D, simulated)", font=MONO, font_size=16, color=MUTED)
        grid_label.next_to(cell_group, DOWN, buff=0.35)
        self.play(FadeIn(grid_label), run_time=0.3)

        # ---------- right-side score panel — vertically centered on the grid's own center ----------
        RIGHT_X = 4.6
        panel_mid_y = grid_y

        score_title = Text("AI score for this sentence", font=MONO, font_size=18, color=MUTED)
        score_title.move_to([RIGHT_X, panel_mid_y + 1.5, 0])
        self.play(FadeIn(score_title), run_time=0.4)

        score_box = [Text("--", font=MONO, font_size=60, weight=BOLD, color=INK2)
                    .move_to([RIGHT_X, panel_mid_y + 0.5, 0])]

        def set_score(value, run_time=0.3):
            old = score_box[0]
            new = Text(f"{value:.2f}", font=MONO, font_size=60, weight=BOLD,
                      color=sem_color(value)).move_to([RIGHT_X, panel_mid_y + 0.5, 0])
            self.play(FadeOut(old, shift=UP * 0.08), FadeIn(new, shift=UP * 0.08), run_time=run_time)
            score_box[0] = new

        self.play(FadeIn(score_box[0]), run_time=0.3)

        # ---------- token row: Q is shown but never scored, only A is ----------
        row_y = 3.1

        def visible_len(s):
            """manim's Text indexes/slices by POSITION IN ITS SUBMOBJECT LIST, which only has
            one entry per visible glyph — spaces never become a submobject at all. So slicing
            by raw string character offsets is off by however many spaces came before it; the
            offset that actually works is a count of non-space characters."""
            return sum(1 for c in s if not c.isspace())

        def make_row(q_words, a_words):
            """One single Text mobject for the whole line — not separate Text objects arranged
            side by side. Separate objects each get their OWN tight bounding box, and a comma
            or a word with no descender has a shorter box than "gravity", so arrange()
            (which centers boxes) made the letters themselves bob up and down. A single string
            is laid out by one real text shaper with one shared baseline, so every glyph sits
            level — we still get per-word positions via (visible-character-count) slicing."""
            q_clean = [w.strip() or "·" for w in q_words]
            a_clean = [w.strip() or "·" for w in a_words]
            q_str = "Q: " + " ".join(q_clean) + "   A: "
            full_str = q_str + " ".join(a_clean)
            row = Text(full_str, font=MONO, font_size=24, color=INK2)
            row[:visible_len(q_str)].set_color(MUTED)

            a_objs = []
            vis_pos = visible_len(q_str)
            for w in a_clean:
                wv = visible_len(w)
                a_objs.append(row[vis_pos:vis_pos + wv])
                vis_pos += wv

            if row.width > 14.5:
                row.scale_to_fit_width(14.5)
            row.move_to([0, row_y, 0])
            return row, a_objs

        def animate_sentence(row, a_objs, buckets_, score, run_time):
            """The score appears ONCE on the right first, THEN splits into identical small
            copies that travel down to under each A-token (not the other way around), THEN
            those copies fly on into the grid — every touched cell's running average updating
            together, not one token at a time."""
            reveal_rt = max(0.05, run_time * 0.3)
            scatter_rt = max(0.05, run_time * 0.3)
            fly_rt = max(0.08, run_time * 0.4)

            set_score(score, run_time=reveal_rt)

            # all numbers sit on the SAME baseline (the row's own bottom), not each word's own
            # bounding-box bottom — a word like "," has a tiny box that sits much higher than a
            # word like "gravity", which is exactly what made the numbers look like they were
            # jumping up and down
            baseline_y = row.get_bottom()[1] - 0.22

            small_nums = VGroup()
            scatter_anims = []
            for word_mobj in a_objs:
                n = Text(f"{score:.2f}", font=MONO, font_size=16, weight=BOLD,
                        color=sem_color(score))
                n.move_to(score_box[0].get_center())
                small_nums.add(n)
                scatter_anims.append(n.animate.move_to([word_mobj.get_x(), baseline_y, 0]))
            self.add(small_nums)
            self.play(*scatter_anims, run_time=scatter_rt)

            touched = {}
            for b in buckets_:
                touched[b] = touched.get(b, 0) + 1
            for b, cnt in touched.items():
                s0, n0 = stats.get(b, [0.0, 0])
                stats[b] = [s0 + score * cnt, n0 + cnt]

            flying = VGroup()
            fly_anims = []
            for n, b in zip(small_nums, buckets_):
                sq, lab = cells[b]
                fcp = n.copy()
                flying.add(fcp)
                fly_anims.append(fcp.animate.move_to(sq.get_center()).scale(0.5).set_opacity(0))
            fly_anims.append(FadeOut(small_nums))
            label_anims = []
            for b in touched:
                sq, lab = cells[b]
                avg = stats[b][0] / stats[b][1]
                fly_anims.append(sq.animate.set_fill(sem_color(avg)))
                label_anims.append(Transform(lab, Text(f"{avg:.2f}", font=MONO, font_size=14,
                                                       color=INK).move_to(sq)))
            self.play(*fly_anims, *label_anims, run_time=fly_rt)
            self.remove(flying)

        # ===================== ~30 synthetic demo sentences, speeding up each round
        # (see docstring: generated QA/scores, not real TruthfulQA — only the mechanism matters
        # here) =====================
        rt = 1.0
        for i in range(N_DEMO):
            q_words, a_words, score, buckets_ = synth_sentence(i)
            row, a_objs = make_row(q_words, a_words)
            self.play(FadeIn(row, shift=UP * 0.15), run_time=max(0.03, min(0.25, rt * 0.25)))
            animate_sentence(row, a_objs, buckets_, score, rt)
            self.play(FadeOut(row), run_time=max(0.02, rt * 0.15))
            rt = max(0.05, rt * 0.78)

        # ===================== finale: ONE real abnormal TruthfulQA example, read-only =====================
        change_title = Text("this time: no fresh scoring -- just read each cell's average",
                            font=MONO, font_size=20, color=WARM)
        change_title.move_to(title)
        self.play(FadeOut(title), FadeIn(change_title), run_time=0.4)

        # no fresh "AI score" happens this round, so that whole panel goes away for this phase —
        # the right side is about to become a line chart instead. Swap score_box for an already-
        # invisible placeholder afterwards, so the LATER set_score() for the final score crossfades
        # from nothing instead of from this (already faded-out but still referenced) old demo
        # number — otherwise it would visibly flash back in before fading to the final value.
        self.play(FadeOut(score_title), FadeOut(score_box[0]), run_time=0.3)
        score_box[0] = Text("--", font=MONO, font_size=60, weight=BOLD, color=INK2,
                            fill_opacity=0).move_to([RIGHT_X, panel_mid_y + 0.5, 0])

        fin_trace = TRACES[FINALE_IDX]
        # the real prompt tokens literally start with "Q" ":" and end with "A" ":" (that's how
        # answer_start finds where the answer begins) — make_row adds its own "Q: "/" A: "
        # labels, so strip those four tokens here or they'd show up doubled ("Q: Q : ... A : A:")
        fin_q = [w for w in question_tokens(fin_trace)[2:-2] if w.strip() != "↵"]
        # that "↵" ("↵") is a real token in the data — GPT-2's own line-break token, sitting
        # between the question and "A:". It's a single character but the glyph itself is drawn
        # as SEVERAL sub-paths (an arrow has a shaft and a head), so it silently broke the
        # "one visible character = one submobject" assumption make_row's slicing relies on and
        # threw off every A-word position after it. It carries no information once Q and A are
        # already on one line with their own spacing, so it's dropped rather than special-cased.
        fin_ans = answer_tokens(fin_trace)
        fin_a_words = [w for w, _, _ in fin_ans]
        fin_row, fin_a_objs = make_row(fin_q, fin_a_words)
        self.play(FadeIn(fin_row, shift=UP * 0.15), run_time=0.5)

        # a small, empty line chart appears where the score panel used to be, and gets built up
        # point by point as each token's cell is read — not assembled all at once at the end
        PAD_X_FRAC, PAD_TOP_FRAC, PAD_BOT_FRAC = 0.08, 0.14, 0.14

        def chart_bounds(cx, cy, w, h):
            left = cx - w / 2 + w * PAD_X_FRAC
            right = cx + w / 2 - w * PAD_X_FRAC
            top = cy + h / 2 - h * PAD_TOP_FRAC
            bot = cy - h / 2 + h * PAD_BOT_FRAC
            return left, right, top, bot

        n = len(fin_ans)

        def chart_point(cx, cy, w, h, i, v):
            left, right, top, bot = chart_bounds(cx, cy, w, h)
            px = left + (right - left) * (i / max(1, n - 1))
            py = bot + (top - bot) * max(0.0, min(1.0, v))
            return np.array([px, py, 0])

        rchart_w, rchart_h = 5.6, 2.6
        rchart_x, rchart_y = RIGHT_X, panel_mid_y
        rchart_box = Rectangle(width=rchart_w, height=rchart_h, fill_color=SUNK, fill_opacity=1,
                              stroke_color=RULE, stroke_width=2).move_to([rchart_x, rchart_y, 0])
        self.play(FadeIn(rchart_box), run_time=0.4)

        # same positioning scheme as the 30 demo sentences — one shared baseline below the row,
        # x taken from the word itself. The earlier "top-right corner of the word" placement
        # was never reliably correct, so this uses the one approach already proven to work.
        fin_baseline_y = fin_row.get_bottom()[1] - 0.22

        fin_vals = []
        fin_dots = VGroup()
        for idx, ((w, s, v), word_mobj) in enumerate(zip(fin_ans, fin_a_objs)):
            b = bucket(s)
            sq, lab = cells[b]
            avg = stats[b][0] / stats[b][1] if b in stats else v  # never visited -> own real theta_avg
            fin_vals.append(avg)

            # square highlight on the cell being read, everything else dims so it pops — and
            # the word itself goes gold while its cell is the one being read, same as the
            # active-token treatment in intro-anim's flowchart
            square_ring = Square(side_length=cell_size * 1.2, stroke_color=WARM, stroke_width=4,
                                 fill_opacity=0).move_to(sq)
            dim_anims = [m.animate.set_opacity(0.2) for bb, pair in cells.items() if bb != b
                        for m in pair]

            flying_num = Text(f"{avg:.2f}", font=MONO, font_size=16, weight=BOLD,
                              color=sem_color(avg)).move_to(sq.get_center())
            target_pos = np.array([word_mobj.get_x(), fin_baseline_y, 0])

            dot = Dot(chart_point(rchart_x, rchart_y, rchart_w, rchart_h, idx, avg),
                     radius=0.06, color=sem_color(avg))

            self.play(*dim_anims, FadeIn(square_ring), word_mobj.animate.set_color(WARM),
                     flying_num.animate.move_to(target_pos).scale(0.8),
                     FadeIn(dot), run_time=0.5)
            fin_dots.add(dot)
            self.remove(square_ring)
            self.play(*[m.animate.set_opacity(1) for bb, pair in cells.items() if bb != b
                       for m in pair],
                     word_mobj.animate.set_color(DONE), run_time=0.22)

        line_pts = [chart_point(rchart_x, rchart_y, rchart_w, rchart_h, i, v)
                   for i, v in enumerate(fin_vals)]
        rpoly = VMobject(stroke_color=ACCENT, stroke_width=2.5)
        rpoly.set_points_as_corners(line_pts)
        self.play(Create(rpoly), run_time=0.6)

        # ===================== once every token's read: chart slides left (replacing the grid),
        # gets an AVG line, and the right side shows the final score normally =====================
        self.play(FadeOut(cell_group), FadeOut(grid_label), run_time=0.4)

        # grow it to the grid's old width but by a single UNIFORM scale factor, not independent
        # width/height stretches — those don't preserve the original rectangle's proportions,
        # which is exactly what made the final chart look squashed/flatter than it started
        target_x, target_y = LEFT_X + 1.2, grid_y
        target_w = 9.0
        scale_factor = target_w / rchart_w
        target_h = rchart_h * scale_factor
        chart_group = VGroup(rchart_box, fin_dots, rpoly)
        self.play(chart_group.animate.move_to([target_x, target_y, 0]).scale(scale_factor),
                 run_time=0.9)

        score = sum(fin_vals) / len(fin_vals)
        # same fractional padding as the small chart, so this lines up exactly with the
        # already-stretched dots instead of being computed independently
        avg_left, avg_right, avg_top, avg_bot = chart_bounds(target_x, target_y, target_w, target_h)
        avg_y = avg_bot + (avg_top - avg_bot) * max(0.0, min(1.0, score))
        avg_line = DashedLine([avg_left, avg_y, 0], [avg_right, avg_y, 0], color=WARM, stroke_width=2)
        self.play(Create(avg_line), run_time=0.5)

        final_title = Text("final trace score\n= AVG(theta_avg)", font=MONO, font_size=18,
                           color=WARM, line_spacing=1.1)
        final_title.move_to([RIGHT_X, panel_mid_y + 1.5, 0])
        self.play(FadeIn(final_title), run_time=0.3)
        set_score(score, run_time=0.5)

        self.wait(1.4)
