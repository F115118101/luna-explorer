"""Export real GPT-2 data for the web page (Lab 1, 3, 4) into web/luna-data.json, using luna_model.pkl.

Usage:  python live/export_luna_data.py
Re-creates tune_luna.py's seeded 60/20/20 question split, so TEST questions were never used to fit PCA / KMeans / DTMC.
Checks that the re-computed test AUCs equal the ones stored in luna_model.pkl before writing anything.
"""
import csv, json, os, pickle, random, datetime
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from transformers import AutoModel, AutoTokenizer
from luna_grid import GridPartition  # noqa: F401  (needed to unpickle)

SEED = 0
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "web", "luna-data.json")
M = pickle.load(open(os.path.join(HERE, "luna_model.pkl"), "rb"))
LAYERS = M["layers"]
_V = json.load(open(os.path.join(HERE, "luna_versions.json"), encoding="utf-8"))["versions"]      # written by tune_luna.py: both versions' scores
VERSIONS = [dict(version=v["version"], dimsPerToken=v["dims_per_token"], partition=v["partition"], pca=v["pca"], states=v["states"],
                 valPlaus=v["val_plaus"], valTruth=v["val_truth"], testPlaus=v["test_plaus"], testTruth=v["test_truth"], selected=v["selected"]) for v in _V]

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
tok = AutoTokenizer.from_pretrained(M["model"])
lm = AutoModel.from_pretrained(M["model"], output_hidden_states=True).eval()


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


def clean(t): return t.replace("Ġ", " ").replace("Ċ", "↵")


def states_of(h): return M["km"].predict(M["pca"].transform(h))


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
    t = qa(r["Question"], r["Best Answer"]); test_items.append(dict(q=r["Question"], true=t, false=qa(r["Question"], r["Best Incorrect Answer"]), garb=garble(t)))
print(len(train), len(val), len(test), "questions (train/val/test)")

P, theta = M["P"], M["theta"]
recs = []                              # test traces
for it in test_items:
    for kind in ("true", "false", "garb"):
        tokens, h = embed(it[kind]); s = states_of(h)
        recs.append(dict(q=it["q"], kind=kind, text=it[kind], tokens=tokens, h=h, s=s, start=answer_start(tokens)))


def scores(r):
    s, st = r["s"], r["start"]
    tr = P[s[:-1], s[1:]][max(st - 1, 0):]
    return (float(tr.mean()) if len(tr) else 0.0), float(theta[s[st:]].mean() if len(s[st:]) else theta[s].mean())


sc = {k: np.array([scores(r) for r in recs if r["kind"] == k]) for k in ("true", "false", "garb")}
auc = lambda p, q: roc_auc_score([1] * len(p) + [0] * len(q), list(p) + list(q))
ap, at = auc(sc["true"][:, 0], sc["garb"][:, 0]), auc(sc["true"][:, 1], sc["false"][:, 1])
print(f"recomputed test AUC: plaus {ap:.4f} truth {at:.4f}   stored: {M['auc_plaus']:.4f} {M['auc_truth']:.4f}")
assert abs(ap - M["auc_plaus"]) < 0.01 and abs(at - M["auc_truth"]) < 0.01, "split not reproduced - refusing to write"

# ---- Lab 1: PCA(2) view of concrete states (train / test normal / test abnormal) ----
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
def trace(r, mode, label, tag):
    s, st = r["s"], r["start"]
    sem = [None] + [round(float(P[a_, b_]), 4) for a_, b_ in zip(s[:-1], s[1:])] if mode == "transition"         else [None if i < st else round(float(theta[x]), 4) for i, x in enumerate(s)]
    return {"id": tag, "mode": mode, "label": label, "prompt": r["text"], "tokens": [clean(t) for t in r["tokens"]],
            "states": [f"s{int(x)}" for x in s], "semantics": sem}
by = {}
for r in recs: by.setdefault((r["q"], r["kind"]), r)
N_GOOD, N_BAD = 4, 1          # clear cases + one counter-example per mode (shows the FP/FN trade-off honestly)
def pick(mode, bad_kind, col):
    cand = []
    for it in test_items:
        t, o = by[(it["q"], "true")], by[(it["q"], bad_kind)]
        if not all(12 <= len(x["tokens"]) <= 40 for x in (t, o)): continue          # short enough to read on screen
        cand.append((scores(t)[col] - scores(o)[col], it))
    cand.sort(key=lambda x: -x[0])
    return [(g, it, "clear") for g, it in cand[:N_GOOD]] + [(g, it, "counter-example") for g, it in cand[-N_BAD:]]
traces = []
for mode, bad_kind, col, nm, bl in (("state", "false", 1, "TruthfulQA", "false answer"), ("transition", "garb", 0, "OOD", "garbled text")):
    for k, (g, it, why) in enumerate(pick(mode, bad_kind, col)):
        print(f"{nm} {why:16s} gap {g:+.3f}  {it['q'][:60]}")
        tag = f"{nm} {why} #{k + 1}"
        traces.append(trace(by[(it["q"], "true")], mode, "normal", tag + " · normal"))
        traces.append(trace(by[(it["q"], bad_kind)], mode, "abnormal", tag + f" · {bl}"))

# ---- Lab 4: per-sample semantic scores on the whole test split ----
scores_out = {"GPT2-OOD": {"normal": [round(float(v), 4) for v in sc["true"][:, 0]], "abnormal": [round(float(v), 4) for v in sc["garb"][:, 0]]},
              "GPT2-TQA": {"normal": [round(float(v), 4) for v in sc["true"][:, 1]], "abnormal": [round(float(v), 4) for v in sc["false"][:, 1]]}}

def youden(pos, neg):                  # threshold t (flag abnormal if score <= t) maximising TPR - FPR on the test scores
    ts = np.unique(np.concatenate([pos, neg])); best = max(ts, key=lambda t: (np.mean(neg <= t) - np.mean(pos <= t)))
    return float(best)
th = {"transition": youden(sc["true"][:, 0], sc["garb"][:, 0]), "state": youden(sc["true"][:, 1], sc["false"][:, 1])}
print("Lab 3 default thresholds (Youden on test scores):", {k: round(v, 3) for k, v in th.items()})
old = json.load(open(OUT, encoding="utf-8"))
meta = old["meta"]; meta["llms"] = ["gpt2 (our own, Lab 1/3/4 + Section 10)"]; meta["ownModel"] = {"model": M["model"], "layers": len(LAYERS), "pca": int(M["pca"].n_components_), "states": int(M["K"]), "testQuestions": len(test), "aucPlaus": round(ap, 4), "aucTruth": round(at, 4), "lab3Threshold": {k: round(v, 3) for k, v in th.items()}, "lab3Note": "Lab 3 examples are hand-picked by rule: the 4 test questions with the largest normal-vs-abnormal score gap (texts of 12-40 tokens) plus 1 counter-example per mode; not a random sample. Overall performance is in Lab 4.", "versions": VERSIONS, "lab1Note": (f"Lab 1 scatter leaves out token position 0 of each text: its vector is {_n0 / _nr:.1f}x longer than the other tokens (GPT-2 attention sink) and would squash the picture; the model itself uses all tokens." if SKIP0 else "Lab 1 scatter shows all tokens."), "generatedAt": datetime.date.today().isoformat()}
json.dump({"meta": meta, "states2d": {"points": pts}, "configs": old["configs"], "traces": traces, "scores": scores_out}, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
print("wrote", os.path.abspath(OUT), len(pts), "points,", len(traces), "traces")
