"""Improve the live LUNA model: all 817 TruthfulQA questions, last decoder block only, config sweep on a validation split.

Usage:  python live/tune_luna.py     (writes luna_model.pkl, which server.py loads)

Split by question: 60% train / 20% val / 20% test.  Configs are picked on VAL; TEST is scored once at the end.
"""
import csv, itertools, json, os, pickle, random, shutil, time, urllib.request
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoConfig, AutoModel, AutoTokenizer

import sys
from luna_grid import GridPartition
MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
# The paper (Sec. III-B) only says the concrete state is "the decoder block outputs" and does not say how blocks are combined.
# The authors' released code (reproduce/*.py, demo.py) runs every experiment with hidden_states_block_idx = "31" = the LAST block
# of Llama-7b (one 4096-d vector per token); its parse_hidden_states() can also concatenate several blocks, but no experiment uses that.
# We build BOTH versions and compare them on a validation split:
#   "last"   = output of the last decoder block only (what the authors' experiments do)            -> 768 dims for GPT-2
#   "concat" = outputs of all decoder blocks concatenated (our reading of "32 blocks x 4096 dims")  -> 12 x 768 = 9216 dims
LAST = AutoConfig.from_pretrained(MODEL).n_layer
ALL_LAYERS = list(range(1, LAST + 1))
LAYER_SETS = {"concat": ALL_LAYERS, "last": [LAST]}
# Table III (paper): Cluster/DTMC = PCA {512,1024,2048} x #state {200,400,600} (KMeans);  Grid/DTMC = PCA {3,5,10} x m {5,10,15} x history {1,2,3}.
# Our choices: "concat" uses the paper's PCA dims as they are (input is 9216-d). For "last" (768-d) PCA 1024/2048 is impossible, so we keep the
# same fractions of the input as the authors' 512/1024/2048 of 4096 (1/8, 1/4, 1/2) = 96/192/384. Grid uses PCA 3 and history step 1 only.
PCA_DIMS = {"concat": [512, 1024, 2048], "last": [96, 192, 384]}
SCALES = [False]            # no standardisation (neither the paper nor the authors' luna/*.py mention a scaler)
K_LIST = [200, 400, 600]
GRID_M = [5, 10, 15]
MIN_PLAUS_VAL = 0.80      # a config must keep OOD detection at least this good to be eligible
SEED = 0
HERE = os.path.dirname(os.path.abspath(__file__))
CSV_URL = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
tok = AutoTokenizer.from_pretrained(MODEL)
lm = AutoModel.from_pretrained(MODEL, output_hidden_states=True).eval()


def embed(text):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        hs = lm(**ids).hidden_states
    h = np.stack([hs[l][0].numpy() for l in ALL_LAYERS], 1).astype(np.float16)   # [len, n_layer, 768]
    return tok.convert_ids_to_tokens(ids["input_ids"][0]), h


def answer_start(tokens):
    for i in range(len(tokens) - 2, -1, -1):
        if tokens[i] == "A" and tokens[i + 1] == ":": return i + 2
    return 0


def garble(text):
    w = text.split(); random.shuffle(w)
    junk = ["zxq", "@@#", "qwrtp", "0xFF", "ñññ", "~~~"]
    for _ in range(max(1, len(w) // 3)):
        w.insert(random.randrange(len(w) + 1), random.choice(junk))
    return " ".join(w)


path = os.path.join(HERE, "TruthfulQA.csv")
if not os.path.exists(path):
    urllib.request.urlretrieve(CSV_URL, path)
with open(path, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
random.shuffle(rows)
n = len(rows); a, b = int(n * 0.6), int(n * 0.8)
parts = {"train": rows[:a], "val": rows[a:b], "test": rows[b:]}
print({k: len(v) for k, v in parts.items()}, "questions")

qa = lambda q, ans: f"Q: {q}\nA: {ans}"
traces = []   # dict(split, kind, emb, start)
t0 = time.time()
for split, rs in parts.items():
    for r in rs:
        items = [("true", qa(r["Question"], r["Best Answer"])), ("false", qa(r["Question"], r["Best Incorrect Answer"]))]
        if split != "train": items.append(("garb", garble(items[0][1])))
        for kind, text in items:
            tokens, h = embed(text)
            traces.append(dict(split=split, kind=kind, h=h, start=answer_start(tokens)))
print(f"embedded {len(traces)} traces in {time.time() - t0:.0f}s")

H = np.concatenate([t["h"] for t in traces])                      # [Ntok, n_layer, 768] float16
offs = np.cumsum([0] + [len(t["h"]) for t in traces])
tr_idx = [i for i, t in enumerate(traces) if t["split"] == "train"]
tr_tok = np.concatenate([np.arange(offs[i], offs[i + 1]) for i in tr_idx])
print("tokens:", H.shape[0], " train tokens:", len(tr_tok))


def auc(pos, neg): return roc_auc_score([1] * len(pos) + [0] * len(neg), list(pos) + list(neg))


def evaluate(S, K, split):
    """S: state id per token (global). Build DTMC/theta from train, return (plaus AUC, truth AUC, P, theta)."""
    counts = np.zeros((K, K)); tot = np.zeros(K); good = np.zeros(K)
    for i in tr_idx:
        s = S[offs[i]:offs[i + 1]]; y = traces[i]["kind"] == "true"
        if y:
            np.add.at(counts, (s[:-1], s[1:]), 1)
        np.add.at(tot, s, 1)
        if y: np.add.at(good, s, 1)
    P = counts / np.maximum(counts.sum(1, keepdims=True), 1)   # #transitions s->s' / #outgoing transitions of s (no smoothing applied; the paper does not mention any)
    theta = good / np.maximum(tot, 1)                          # paper: average of all concrete semantics inside the abstract state

    def score(i):
        s = S[offs[i]:offs[i + 1]]; st = traces[i]["start"]
        tr = P[s[:-1], s[1:]]                     # transition into token j+1
        tr = tr[max(st - 1, 0):]                  # transitions landing on answer tokens
        return (float(tr.mean()) if len(tr) else 0.0), float(theta[s[st:]].mean() if len(s[st:]) else theta[s].mean())
    sc = {k: np.array([score(i) for i, t in enumerate(traces) if t["split"] == split and t["kind"] == k]) for k in ("true", "false", "garb")}
    return auc(sc["true"][:, 0], sc["garb"][:, 0]), auc(sc["true"][:, 1], sc["false"][:, 1]), P, theta



def calib_refs(S, P, theta):
    ref = []
    for i in tr_idx:                      # scores of truthful TRAIN traces = reference distribution for the percentile shown in the page
        if traces[i]["kind"] != "true": continue
        s = S[offs[i]:offs[i + 1]]; st = traces[i]["start"]
        tr = P[s[:-1], s[1:]][max(st - 1, 0):]
        ref.append(((float(tr.mean()) if len(tr) else 0.0), float(theta[s[st:]].mean() if len(s[st:]) else theta[s].mean())))
    return np.array(ref)


summary = []
for lname, layers in LAYER_SETS.items():
    results, cache = [], {}
    cols = [ALL_LAYERS.index(l) for l in layers]
    X = H[:, cols, :].reshape(len(H), -1).astype(np.float32)
    print(f"\n===== version '{lname}': layers {layers}, {X.shape[1]} dims per token =====", flush=True)
    for nd in PCA_DIMS[lname]:
        pca = PCA(n_components=nd, random_state=SEED, svd_solver="randomized").fit(X[tr_tok])
        Z = pca.transform(X)
        for K in K_LIST:
            km = KMeans(n_clusters=K, n_init=2, random_state=SEED).fit(Z[tr_tok])
            S = km.predict(Z)
            ap, at, _, _ = evaluate(S, K, "val")
            results.append(dict(name=f"{lname}-kmeans", nd=nd, K=K, val_plaus=ap, val_truth=at))
            cache[(f"{lname}-kmeans", nd, K)] = (pca, km, S)
            print(f"{lname:6s} kmeans pca={nd:4d} K={K:4d}  val plaus {ap:.3f}  truth {at:.3f}", flush=True)
    pca3 = PCA(n_components=3, random_state=SEED, svd_solver="randomized").fit(X[tr_tok])
    Z3 = pca3.transform(X); del X
    for m in GRID_M:
        gp = GridPartition(m).fit(Z3[tr_tok]); S = gp.predict(Z3); K = m ** 3
        ap, at, _, _ = evaluate(S, K, "val")
        results.append(dict(name=f"{lname}-grid", nd=3, K=K, val_plaus=ap, val_truth=at))
        cache[(f"{lname}-grid", 3, K)] = (pca3, gp, S)
        print(f"{lname:6s} grid   pca=   3 K={K:4d}  val plaus {ap:.3f}  truth {at:.3f}", flush=True)

    ok = [r for r in results if r["val_plaus"] >= MIN_PLAUS_VAL] or results      # (no config reaches 0.80 for "last": falls back to all)
    best = max(ok, key=lambda r: r["val_truth"])
    pca, km, S = cache[(best["name"], best["nd"], best["K"])]
    tp, tt, P, theta = evaluate(S, best["K"], "test")
    print(f"best on VAL for '{lname}': {best['name']} pca{best['nd']} K{best['K']}  ->  TEST plaus {tp:.3f}  truth {tt:.3f}", flush=True)
    ref = calib_refs(S, P, theta)
    model = dict(model=MODEL, version=lname, layer=layers[0], layers=layers, hs_layers=layers, scaler=None, pca=pca, km=km,
                 P=P, theta=theta, K=best["K"], ref_plaus=np.sort(ref[:, 0]), ref_truth=np.sort(ref[:, 1]),
                 auc_plaus=tp, auc_truth=tt, counts=None, n_train=len(tr_idx))
    with open(os.path.join(HERE, f"luna_model_{lname}.pkl"), "wb") as f: pickle.dump(model, f)
    summary.append(dict(version=lname, layers=layers, dims_per_token=len(layers) * H.shape[2], partition=best["name"].split("-")[1],
                        pca=best["nd"], states=best["K"], val_plaus=round(best["val_plaus"], 4), val_truth=round(best["val_truth"], 4),
                        test_plaus=round(tp, 4), test_truth=round(tt, 4)))
    del S, cache

# the version served by the page = higher mean of the two VALIDATION AUCs (the test AUCs are only reported, not used to choose)
for r in summary: r["val_mean"] = round((r["val_plaus"] + r["val_truth"]) / 2, 4)
chosen = max(summary, key=lambda r: r["val_mean"])
for r in summary: r["selected"] = r is chosen
shutil.copyfile(os.path.join(HERE, f"luna_model_{chosen['version']}.pkl"), os.path.join(HERE, "luna_model.pkl"))
json.dump(dict(model=MODEL, nTrain=len(tr_idx), versions=summary), open(os.path.join(HERE, "luna_versions.json"), "w", encoding="utf-8"), indent=1)
print("\n" + "\n".join(f"{r['version']:6s} val {r['val_plaus']:.3f}/{r['val_truth']:.3f}  TEST plaus {r['test_plaus']:.3f}  truth {r['test_truth']:.3f}" + ("   <- used by the page" if r["selected"] else "") for r in summary))
print("(test has ~160 questions -> AUC noise about +-0.03)")
