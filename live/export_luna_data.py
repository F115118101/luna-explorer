"""Export real GPT-2 data for the web page (Lab 1, 3, 4 and the section-09 versions table) into web/luna-data.json.

Usage:  python live/export_luna_data.py
Two separate models:
  OOD (plausibility)  luna_model.pkl      from tune_luna.py      normal = "Q: ...\nA: <Best Answer>", abnormal = the same text garbled
  HAL (truthfulness)  luna_model_gen.pkl  from tune_luna_gen.py  GPT-2's own answers (generate_answers.py), labelled 0/1 by Gemini
Each part re-creates its script's split, so TEST items were never used to fit PCA / clustering / DTMC, and checks that the
re-computed test AUC equals the stored one before anything is written.
"""
import csv, json, os, pickle, random, datetime
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer
from luna_grid import GridPartition  # noqa: F401  (needed to unpickle)
from luna_prompt import question_prompt

SEED = 0
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "web", "luna-data.json")
M = pickle.load(open(os.path.join(HERE, "luna_model.pkl"), "rb"))          # OOD
G = pickle.load(open(os.path.join(HERE, "luna_model_gen.pkl"), "rb"))      # HAL
LAYERS = M["layers"]
_vo = json.load(open(os.path.join(HERE, "luna_versions.json"), encoding="utf-8"))
_vh = json.load(open(os.path.join(HERE, "luna_gen_versions.json"), encoding="utf-8"))
VERSIONS = [dict(task=task, version=v["version"], dimsPerToken=v["dims_per_token"], partition=v["partition"], pca=v["pca"], states=v["states"],
                 val=v["val"], test=v["test"], selected=v["selected"]) for task, vs in (("ood", _vo), ("hal", _vh)) for v in vs["versions"]]

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
tok = AutoTokenizer.from_pretrained(M["model"])
lm = AutoModel.from_pretrained(M["model"], output_hidden_states=True).eval()
auc = lambda p, q: roc_auc_score([1] * len(p) + [0] * len(q), list(p) + list(q))


def clean(t): return t.replace("Ġ", " ").replace("Ċ", "↵")


def youden(pos, neg):                  # threshold t (flag abnormal if score <= t) maximising TPR - FPR on the test scores
    ts = np.unique(np.concatenate([pos, neg])); best = max(ts, key=lambda t: (np.mean(neg <= t) - np.mean(pos <= t)))
    return float(best)


# =============================== OOD part (luna_model.pkl) ===============================
def garble(text):                      # identical to tune_luna.py (same random stream)
    w = text.split(); random.shuffle(w)
    junk = ["zxq", "@@#", "qwrtp", "0xFF", "ñññ", "~~~"]
    for _ in range(max(1, len(w) // 3)):
        w.insert(random.randrange(len(w) + 1), random.choice(junk))
    return " ".join(w)


def embed(text):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        hs = lm(**ids).hidden_states
    h = np.stack([hs[l][0].numpy() for l in LAYERS], 1).astype(np.float16)
    return tok.convert_ids_to_tokens(ids["input_ids"][0]), h.reshape(len(h), -1).astype(np.float32)


def answer_start(tokens):
    for i in range(len(tokens) - 2, -1, -1):
        if tokens[i] == "A" and tokens[i + 1] == ":": return i + 2
    return 0


with open(os.path.join(HERE, "TruthfulQA.csv"), encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
random.shuffle(rows)
n = len(rows); a, b = int(n * 0.6), int(n * 0.8)
train, val, test = rows[:a], rows[a:b], rows[b:]
qa = lambda q, ans: f"Q: {q}\nA: {ans}"
for r in val:                          # consume the random stream exactly as tune_luna.py did
    garble(qa(r["Question"], r["Best Answer"]))
test_items = []
for r in test:
    t = qa(r["Question"], r["Best Answer"]); test_items.append(dict(q=r["Question"], true=t, garb=garble(t)))
print("OOD:", len(train), len(val), len(test), "questions (train/val/test)")

P = M["P"]
recs = []                              # OOD test traces
for it in test_items:
    for kind in ("true", "garb"):
        tokens, h = embed(it[kind]); s = M["km"].predict(M["pca"].transform(h))
        recs.append(dict(q=it["q"], kind=kind, text=it[kind], tokens=tokens, h=h, s=s, start=answer_start(tokens)))


def plaus(r):
    s, st = r["s"], r["start"]
    tr = P[s[:-1], s[1:]][max(st - 1, 0):]
    return float(tr.mean()) if len(tr) else 0.0


sc_ood = {k: np.array([plaus(r) for r in recs if r["kind"] == k]) for k in ("true", "garb")}
ap = auc(sc_ood["true"], sc_ood["garb"])
print(f"OOD recomputed test AUC {ap:.4f}   stored {M['auc_plaus']:.4f}")
assert abs(ap - M["auc_plaus"]) < 0.01, "OOD split not reproduced - refusing to write"

# =============================== HAL part (luna_model_gen.pkl) ===============================
with open(os.path.join(HERE, "gpt2_answers.jsonl"), encoding="utf-8") as f:
    gen = {d["idx"]: d for d in map(json.loads, f)}
with open(os.path.join(HERE, "gpt2_judged.jsonl"), encoding="utf-8") as f:
    label = {d["idx"]: d["label"] for d in map(json.loads, f)}
ids = sorted(i for i in gen if i in label)
_, test_h = train_test_split(ids, test_size=0.2, random_state=42)          # same split as tune_luna_gen.py
assert sorted(test_h) == sorted(G["test_ids"]), "HAL split differs from tune_luna_gen.py"
theta = G["theta"]


def hal_trace(i):
    h = np.load(os.path.join(HERE, "gen_hidden", f"{i}.npy"))[:, G["block_idx"], :]
    s = G["km"].predict(G["pca"].transform(h.reshape(len(h), -1).astype(np.float32)))
    # step 0 = last prompt token, step k = generated token k (generate_answers.py); the text is re-tokenized only for display
    gen_tok = tok.tokenize(gen[i]["generated"])
    tokens = [tok.tokenize(question_prompt(gen[i]["question"]))[-1]] + gen_tok[:len(s) - 1]
    return dict(idx=i, q=gen[i]["question"], label=label[i], s=s, tokens=tokens, score=float(theta[s].mean()),
                retokenized_ok=len(gen_tok) == gen[i]["n_tokens"])


hal = [hal_trace(i) for i in test_h]
sc_hal = {"true": np.array([r["score"] for r in hal if r["label"] == 1]), "false": np.array([r["score"] for r in hal if r["label"] == 0])}
at = auc(sc_hal["true"], sc_hal["false"])
print(f"HAL recomputed test AUC {at:.4f}   stored {G['auc_truth']:.4f}   ({len(sc_hal['true'])} true / {len(sc_hal['false'])} false test answers)")
assert abs(at - G["auc_truth"]) < 0.01, "HAL model not reproduced - refusing to write"

# ---- Lab 1: PCA(2) view of concrete states (train / test normal / test abnormal), OOD model's text ----
rng = np.random.RandomState(SEED)
tr_recs = [embed(qa(r["Question"], r["Best Answer"])) for r in train[:150]]
# Is token position 0 an outlier (GPT-2 "attention sink")?  Measured, not assumed: if its vector is >3x longer than the median of the others it is left out of THIS 2-D picture only.
_n0 = np.median([np.linalg.norm(h[0]) for _, h in tr_recs]); _nr = np.median(np.concatenate([np.linalg.norm(h[1:], axis=1) for _, h in tr_recs]))
SKIP0 = bool(_n0 / _nr > 3); SK = 1 if SKIP0 else 0
print(f"token-0 norm / median norm of other tokens = {_n0 / _nr:.2f} -> {'excluded from' if SKIP0 else 'kept in'} the Lab 1 picture")
tr_h = np.concatenate([h[SK:] for _, h in tr_recs])
pca2 = PCA(n_components=2, random_state=SEED).fit(tr_h)
pick = lambda H, k: H[rng.choice(len(H), min(k, len(H)), replace=False)]
te_h = np.concatenate([r["h"][SK:] for r in recs if r["kind"] == "true"])
ab_h = np.concatenate([r["h"][SK:] for r in recs if r["kind"] == "garb"])
pts = []
for split, H, k in (("train", tr_h, 500), ("test", te_h, 300), ("abn", ab_h, 300)):
    for x, y in pca2.transform(pick(H, k)): pts.append({"x": round(float(x), 3), "y": round(float(y), 3), "split": split})
print("PCA(2) explained variance:", pca2.explained_variance_ratio_.round(3))

# ---- Lab 3: example traces — CHOSEN to show the method clearly (not a random sample; Lab 4 shows the overall result) ----
N_GOOD = 4                             # clear cases + one counter-example per task (shows the FP/FN trade-off honestly)
traces = []
by = {}
for r in recs: by.setdefault((r["q"], r["kind"]), r)
cand = []
for it in test_items:                  # OOD: pairs (original, garbled) of the same question, by score gap
    t, o = by[(it["q"], "true")], by[(it["q"], "garb")]
    if all(12 <= len(x["tokens"]) <= 40 for x in (t, o)): cand.append((plaus(t) - plaus(o), it))      # short enough to read on screen
cand.sort(key=lambda x: -x[0])
for k, (g, it, why) in enumerate([(g, it, "clear") for g, it in cand[:N_GOOD]] + [(g, it, "counter-example") for g, it in cand[-1:]]):
    print(f"OOD {why:16s} gap {g:+.3f}  {it['q'][:60]}")
    for kind, lab, nm in (("true", "normal", "normal"), ("garb", "abnormal", "garbled text")):
        r = by[(it["q"], kind)]; s = r["s"]
        traces.append({"id": f"OOD {why} #{k + 1} · {nm}", "mode": "transition", "label": lab, "prompt": r["text"], "tokens": [clean(x) for x in r["tokens"]],
                       "states": [f"s{int(x)}" for x in s], "semantics": [None] + [round(float(P[a_, b_]), 4) for a_, b_ in zip(s[:-1], s[1:])]})
# HAL: one answer per question (GPT-2's own), so normal and abnormal examples come from different questions.
# clear = highest-scoring true answers / lowest-scoring false answers; counter-example = lowest true / highest false.
hal_ok = [r for r in hal if r["retokenized_ok"]]
nor = sorted([r for r in hal_ok if r["label"] == 1], key=lambda r: -r["score"])
abn = sorted([r for r in hal_ok if r["label"] == 0], key=lambda r: r["score"])
for k, (rn, ra, why) in enumerate([(nor[j], abn[j], "clear") for j in range(N_GOOD)] + [(nor[-1], abn[-1], "counter-example")]):
    print(f"HAL {why:16s} true {rn['score']:.3f} / false {ra['score']:.3f}")
    for r, lab, nm in ((rn, "normal", "judged true"), (ra, "abnormal", "judged false")):
        traces.append({"id": f"HAL {why} #{k + 1} · {nm}", "mode": "state", "label": lab, "prompt": f"Q: {r['q']}", "tokens": [clean(x) for x in r["tokens"]],
                       "states": [f"s{int(x)}" for x in r["s"]], "semantics": [round(float(theta[x]), 4) for x in r["s"]]})

# ---- Lab 4: per-sample semantic scores on the whole test split ----
r4 = lambda v: [round(float(x), 4) for x in v]
scores_out = {"GPT2-OOD": {"normal": r4(sc_ood["true"]), "abnormal": r4(sc_ood["garb"])},
              "GPT2-TQA": {"normal": r4(sc_hal["true"]), "abnormal": r4(sc_hal["false"])}}
th = {"transition": youden(sc_ood["true"], sc_ood["garb"]), "state": youden(sc_hal["true"], sc_hal["false"])}
print("Lab 3 default thresholds (Youden on test scores):", {k: round(v, 3) for k, v in th.items()})

old = json.load(open(OUT, encoding="utf-8"))
meta = old["meta"]; meta["llms"] = ["gpt2 (our own, Lab 1/3/4 + Section 09)"]
meta["ownModel"] = {"model": M["model"],
    "ood": {"version": M["version"], "pca": int(M["pca"].n_components_), "states": int(M["K"]), "testQuestions": len(test), "auc": round(ap, 4)},
    "hal": {"version": G["version"], "pca": int(G["pca"].n_components_), "states": int(G["K"]), "testQuestions": len(test_h),
            "testTrue": int(len(sc_hal["true"])), "testFalse": int(len(sc_hal["false"])), "auc": round(at, 4)},
    "aucPlaus": round(ap, 4), "aucTruth": round(at, 4), "lab3Threshold": {k: round(v, 3) for k, v in th.items()},
    "lab3Note": "Lab 3 examples are hand-picked by rule, not a random sample. OOD: the 4 test questions with the largest original-vs-garbled score gap (texts of 12-40 tokens) plus 1 counter-example. HAL: the 4 highest-scoring answers judged true and the 4 lowest-scoring answers judged false, plus the lowest true / highest false as counter-example. Overall performance is in Lab 4.",
    "versions": VERSIONS,
    "lab1Note": (f"Lab 1 scatter leaves out token position 0 of each text: its vector is {_n0 / _nr:.1f}x longer than the other tokens (GPT-2 attention sink) and would squash the picture; the model itself uses all tokens." if SKIP0 else "Lab 1 scatter shows all tokens."),
    "generatedAt": datetime.date.today().isoformat()}
_js = json.dumps({"meta": meta, "states2d": {"points": pts}, "configs": old["configs"], "traces": traces, "scores": scores_out}, ensure_ascii=False, separators=(",", ":"))
open(OUT, "w", encoding="utf-8").write(_js)
open(OUT[:-2], "w", encoding="utf-8").write("window.LUNA_DATA=" + _js + ";\n")   # luna-data.js: same data for pages opened as file://
print("wrote", os.path.abspath(OUT), len(pts), "points,", len(traces), "traces")
