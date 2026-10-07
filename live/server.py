"""Live LUNA analysis server.   python live/server.py   ->  http://127.0.0.1:8000/
Serves luna.html and POST /analyze {question?, answer}. Needs luna_model.pkl (run tune_luna.py first)."""
import os, pickle
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer
from luna_grid import GridPartition  # noqa: F401  (needed to unpickle grid-partition models)

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "..", "web")   # luna.html + luna-data.json
M = pickle.load(open(os.path.join(HERE, "luna_model.pkl"), "rb"))
tok = AutoTokenizer.from_pretrained(M["model"])
lm = AutoModel.from_pretrained(M["model"], output_hidden_states=True).eval()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Req(BaseModel):
    question: str = ""
    answer: str


def pct(ref, v):
    """share of reference (normal) scores that are <= v"""
    return float(np.searchsorted(ref, v, side="right") / len(ref))


def clean(t):
    return t.replace("Ġ", " ").replace("Ċ", "↵")


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB, "luna.html"))


@app.get("/{name}.html")
def page(name: str):
    path = os.path.join(WEB, name + ".html")
    return FileResponse(path) if name.replace("-", "").replace("_", "").isalnum() and os.path.exists(path) else JSONResponse({}, status_code=404)


@app.get("/luna-data.json")
def luna_data():
    path = os.path.join(WEB, "luna-data.json")
    return FileResponse(path) if os.path.exists(path) else JSONResponse({}, status_code=404)


@app.get("/health")
def health():
    return {"ok": True, "model": M["model"], "states": M["K"], "auc_plaus": M["auc_plaus"], "auc_truth": M["auc_truth"]}


TOKENIZERS = {"gpt2": "gpt2", "bert": "bert-base-uncased"}   # whitelist for tokenization.html
_tok_cache = {}


class TokReq(BaseModel):
    text: str
    tokenizer: str = "gpt2"


@app.post("/tokenize")
def tokenize(r: TokReq):
    name = TOKENIZERS.get(r.tokenizer)
    if name is None:
        return JSONResponse({"error": "unknown tokenizer"}, status_code=400)
    if name not in _tok_cache:
        _tok_cache[name] = tok if name == M["model"] else AutoTokenizer.from_pretrained(name)
    t = _tok_cache[name]
    text = r.text[:2000]
    ids = t(text, add_special_tokens=False)["input_ids"]
    return {"tokenizer": name, "tokens": t.convert_ids_to_tokens(ids), "ids": ids, "chars": len(text)}


@app.post("/analyze")
def analyze(r: Req):
    text = f"Q: {r.question.strip()}\nA: {r.answer.strip()}" if r.question.strip() else r.answer.strip()
    if not text:
        return {"error": "empty"}
    ids = tok(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        hs = lm(**ids).hidden_states
    h = np.concatenate([hs[l][0].numpy() for l in M.get("layers", [M["layer"]])], axis=1)
    if M.get("scaler") is not None:
        h = M["scaler"].transform(h)
    s = M["km"].predict(M["pca"].transform(h))
    P, theta = M["P"], M["theta"]
    trans = [None] + [float(P[a, b]) for a, b in zip(s[:-1], s[1:])]
    th = [float(theta[i]) for i in s]
    tokens = [clean(t) for t in tok.convert_ids_to_tokens(ids["input_ids"][0])]
    a = next((i + 2 for i in range(len(tokens) - 2, -1, -1) if tokens[i] == "A" and tokens[i + 1] == ":"), 0)
    valid = [t for t in trans[a:] if t is not None]
    plaus = float(np.mean(valid)) if valid else 0.0
    truth = float(np.mean(th[a:] or th))
    pp, tp = pct(M["ref_plaus"], plaus), pct(M["ref_truth"], truth)
    return {
        "text": text, "tokens": tokens, "states": [int(i) for i in s], "trans": trans, "theta": th,
        "plaus": plaus, "truth": truth, "plaus_pct": pp, "truth_pct": tp,
        "plaus_verdict": "abnormal" if pp < 0.05 else ("borderline" if pp < 0.20 else "normal"),
        "truth_verdict": "suspect" if tp < 0.30 else "ok",
        "auc_plaus": M["auc_plaus"], "auc_truth": M["auc_truth"], "K": M["K"], "answer_start": a,
        "theta_all": [float(x) for x in theta],
        "first_low": next((i for i, t in enumerate(trans) if i >= a and t is not None and t < 0.02), None),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
