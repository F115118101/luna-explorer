"""LUNA hallucination detection on GPT-2's OWN TruthfulQA answers, following the authors' pipeline.

Usage:  python live/tune_luna_gen.py     (needs gpt2_answers.jsonl + gen_hidden/ from generate_answers.py, gpt2_judged.jsonl from judge_answers.py)
Writes  luna_gen_results.json   all settings
        luna_model_gen.pkl      the setting picked on VAL (server.py and export_luna_data.py use it for truthfulness)
        luna_gen_versions.json  best setting per version ("concat" / "last"), shown in the page's section 09 table

Differences from tune_luna.py (which uses TruthfulQA's human-written Best Answer / Best Incorrect Answer):
  - one trace per question = GPT-2's own generated answer, hidden states recorded during generation (generate_answers.py)
  - label = Gemini judge's 0/1 (judge_answers.py), standing in for the authors' truth_prob > 0.5 (luna/data_loader.py l.19, 52)
  - split as the authors do (data_loader.py): train_test_split(test_size=0.2, random_state=42), then the same on train for val
  - theta(state) = pos / (pos + neg) over every token of every train trace, unseen state -> 0.0
    (construct_dtmc_state_model l.1088, get_sentence_scores_by_state_binding l.1129 in probabilistic_abstraction_model.py)
  - trace score = mean theta over the WHOLE trace (all generation steps), AUC vs the 0/1 label
  - the authors report AUC on val + test together for every setting; we additionally pick one setting on val and report test
Only the truthfulness task is done here; the OOD (garbled text) task stays in tune_luna.py.
"""
import json, os, pickle, time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from luna_grid import GridPartition

HERE = os.path.dirname(os.path.abspath(__file__))
N_LAYER = 12
LAYER_SETS = {"concat": list(range(N_LAYER)), "last": [N_LAYER - 1]}          # indices into the saved [steps, 12, 768] arrays
PCA_DIMS = {"concat": [512, 1024, 2048], "last": [96, 192, 384]}               # same choices as tune_luna.py
K_LIST = [200, 400, 600]
GRID_M = [5, 10, 15]
SEED = 0

with open(os.path.join(HERE, "gpt2_answers.jsonl"), encoding="utf-8") as f:
    gen = {d["idx"]: d for d in map(json.loads, f)}
with open(os.path.join(HERE, "gpt2_judged.jsonl"), encoding="utf-8") as f:
    label = {d["idx"]: d["label"] for d in map(json.loads, f)}
ids = sorted(i for i in gen if i in label)
print(f"{len(ids)} questions, {sum(label[i] for i in ids)} labelled true", flush=True)

tr_val, test = train_test_split(ids, test_size=0.2, random_state=42)
train, val = train_test_split(tr_val, test_size=0.2, random_state=42)
split = {**{i: "train" for i in train}, **{i: "val" for i in val}, **{i: "test" for i in test}}
print({s: len(v) for s, v in (("train", train), ("val", val), ("test", test))}, "questions", flush=True)

t0 = time.time()
hs = [np.load(os.path.join(HERE, "gen_hidden", f"{i}.npy")) for i in ids]   # [steps, 12, 768] float16
H = np.concatenate(hs)
offs = np.cumsum([0] + [len(h) for h in hs]); del hs
pos = {i: k for k, i in enumerate(ids)}
tr_tok = np.concatenate([np.arange(offs[pos[i]], offs[pos[i] + 1]) for i in train])
print(f"loaded {H.shape[0]} token states ({len(tr_tok)} train) in {time.time() - t0:.0f}s", flush=True)


def evaluate(S, K):
    """S: state id per token. theta from train traces, return {split: AUC} and theta."""
    p, n = np.zeros(K), np.zeros(K)
    for i in train:
        s = S[offs[pos[i]]:offs[pos[i] + 1]]
        np.add.at(p if label[i] else n, s, 1)
    theta = p / (p + n + 1e-20)                    # never-visited states stay 0.0, as in the authors' code
    score = {i: float(theta[S[offs[pos[i]]:offs[pos[i] + 1]]].mean()) for i in ids}
    auc = lambda part: roc_auc_score([label[i] for i in part], [score[i] for i in part])
    return dict(val=auc(val), test=auc(test), val_test=auc(val + test)), theta


results, best_fit = [], {}       # best_fit[version] = (val AUC, result row, pca, partition, S) of that version's best setting on VAL


def keep(lname, row, pca, part, S):
    results.append(row)
    if lname not in best_fit or row["val"] > best_fit[lname][0]: best_fit[lname] = (row["val"], row, pca, part, S)


for lname, cols in LAYER_SETS.items():
    X = H[:, cols, :].reshape(len(H), -1).astype(np.float32)
    print(f"\n===== version '{lname}': {X.shape[1]} dims per token =====", flush=True)
    for nd in PCA_DIMS[lname]:
        pca = PCA(n_components=nd, random_state=SEED, svd_solver="randomized").fit(X[tr_tok]); Z = pca.transform(X)
        for K in K_LIST:
            km = KMeans(n_clusters=K, n_init=2, random_state=SEED).fit(Z[tr_tok]); S = km.predict(Z)
            a, _ = evaluate(S, K)
            keep(lname, dict(version=lname, partition="kmeans", pca=nd, states=K, **{k: round(v, 4) for k, v in a.items()}), pca, km, S)
            print(f"{lname:6s} kmeans pca={nd:4d} K={K:4d}  val {a['val']:.3f}  test {a['test']:.3f}  val+test {a['val_test']:.3f}", flush=True)
    pca3 = PCA(n_components=3, random_state=SEED, svd_solver="randomized").fit(X[tr_tok]); Z3 = pca3.transform(X); del X
    for m in GRID_M:
        gp = GridPartition(m).fit(Z3[tr_tok]); S = gp.predict(Z3); K = m ** 3
        a, _ = evaluate(S, K)
        keep(lname, dict(version=lname, partition="grid", pca=3, states=K, **{k: round(v, 4) for k, v in a.items()}), pca3, gp, S)
        print(f"{lname:6s} grid   pca=   3 K={K:4d}  val {a['val']:.3f}  test {a['test']:.3f}  val+test {a['val_test']:.3f}", flush=True)

best = max(results, key=lambda r: r["val"])
chosen = best["version"]
_, row, pca, part, S = best_fit[chosen]
_, theta = evaluate(S, row["states"])
ref = np.sort([float(theta[S[offs[pos[i]]:offs[pos[i] + 1]]].mean()) for i in train if label[i]])   # truthful TRAIN traces = reference for the page's percentile
# block_idx 0..11 = decoder blocks 1..12 = hidden_states[1..12]
model = dict(model="gpt2", task="hal", version=chosen, block_idx=LAYER_SETS[chosen], pca=pca, km=part, theta=theta, K=row["states"], ref_truth=ref,
             auc_truth=row["test"], auc_truth_val=row["val"], n_train=len(train), test_ids=test)
with open(os.path.join(HERE, "luna_model_gen.pkl"), "wb") as f: pickle.dump(model, f)
versions = [dict(version=v, dims_per_token=len(LAYER_SETS[v]) * H.shape[2], partition=r["partition"], pca=r["pca"], states=r["states"],
                 val=r["val"], test=r["test"], selected=v == chosen) for v, (_, r, *_rest) in best_fit.items()]
json.dump(dict(model="gpt2", task="hal", nTrain=len(train), nTest=len(test), versions=versions),
          open(os.path.join(HERE, "luna_gen_versions.json"), "w", encoding="utf-8"), indent=1)
vt = np.array([r["val_test"] for r in results])
summary = dict(n_questions=len(ids), n_true=int(sum(label[i] for i in ids)), split={"train": len(train), "val": len(val), "test": len(test)},
               picked_on_val=best, authors_style_val_test=dict(mean=round(float(vt.mean()), 4), median=round(float(np.median(vt)), 4),
                                                              max=round(float(vt.max()), 4), min=round(float(vt.min()), 4)), settings=results)
json.dump(summary, open(os.path.join(HERE, "luna_gen_results.json"), "w", encoding="utf-8"), indent=1)
print(f"\npicked on VAL: {best['version']}-{best['partition']} pca{best['pca']} K{best['states']}  ->  TEST AUC {best['test']:.3f}")
print(f"authors-style (val+test, all {len(results)} settings): mean {vt.mean():.3f}  median {np.median(vt):.3f}  max {vt.max():.3f}  min {vt.min():.3f}")
print(f"(test has {len(test)} questions -> AUC noise roughly +-0.04)")
