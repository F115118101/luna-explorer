"""Live LUNA analysis server.   python live/server.py   ->  http://127.0.0.1:8000/
Serves luna.html and POST /analyze {question?, answer}.
Needs luna_model.pkl (OOD / plausibility, tune_luna.py) and luna_model_gen.pkl (truthfulness, tune_luna_gen.py)."""
import os, pickle
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer
from luna_grid import GridPartition  # noqa: F401  (needed to unpickle grid-partition models)
from luna_prompt import question_prompt

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "..", "web")   # luna.html + luna-data.json
M = pickle.load(open(os.path.join(HERE, "luna_model.pkl"), "rb"))        # OOD
G = pickle.load(open(os.path.join(HERE, "luna_model_gen.pkl"), "rb"))    # truthfulness
tok = AutoTokenizer.from_pretrained(M["model"])
lm = AutoModel.from_pretrained(M["model"], output_hidden_states=True).eval()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
# static assets (images, the pre-rendered intro video, …); StaticFiles handles Range requests
# itself, which is what lets <video> seeking jump straight to the dragged frame instead of
# re-buffering from the start.
app.mount("/assets", StaticFiles(directory=os.path.join(WEB, "assets")), name="assets")


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
    return {"ok": True, "model": M["model"], "states": M["K"], "auc_plaus": M["auc_plaus"], "auc_truth": G["auc_truth"]}


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


def truthfulness(question, answer):
    """Score the answer as if GPT-2 had generated it: same prompt as generate_answers.py, then "\nA: <answer>" (GPT-2 writes
    that "\nA: " itself). GPT-2 is causal, so one forward pass gives exactly the states generation would have produced:
    step 0 = last prompt token, step k = answer token k; the last token is never fed back during generation, so it is dropped."""
    p_ids = tok(question_prompt(question))["input_ids"]
    ids = tok(question_prompt(question) + "\nA: " + answer, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        hs = lm(**ids).hidden_states
    lo, hi = len(p_ids) - 1, ids["input_ids"].shape[1] - 1
    h = np.concatenate([hs[b + 1][0, lo:hi].numpy() for b in G["block_idx"]], axis=1)
    s = G["km"].predict(G["pca"].transform(h))
    th = [float(G["theta"][i]) for i in s]
    tokens = [clean(t) for t in tok.convert_ids_to_tokens(ids["input_ids"][0][lo:hi])]
    return tokens, [int(i) for i in s], th


@app.post("/analyze")
def analyze(r: Req):
    text = f"Q: {r.question.strip()}\nA: {r.answer.strip()}" if r.question.strip() else r.answer.strip()
    if not r.answer.strip():
        return {"error": "empty"}
    ids = tok(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        hs = lm(**ids).hidden_states
    h = np.concatenate([hs[l][0].numpy() for l in M.get("layers", [M["layer"]])], axis=1)
    if M.get("scaler") is not None:
        h = M["scaler"].transform(h)
    s = M["km"].predict(M["pca"].transform(h))
    P = M["P"]
    trans = [None] + [float(P[a, b]) for a, b in zip(s[:-1], s[1:])]
    tokens = [clean(t) for t in tok.convert_ids_to_tokens(ids["input_ids"][0])]
    a = next((i + 2 for i in range(len(tokens) - 2, -1, -1) if tokens[i] == "A" and tokens[i + 1] == ":"), 0)
    valid = [t for t in trans[a:] if t is not None]
    plaus = float(np.mean(valid)) if valid else 0.0
    t_tokens, t_states, th = truthfulness(r.question.strip(), r.answer.strip())
    truth = float(np.mean(th))
    pp, tp = pct(M["ref_plaus"], plaus), pct(G["ref_truth"], truth)
    return {
        "text": text, "tokens": tokens, "states": [int(i) for i in s], "trans": trans,
        "truth_tokens": t_tokens, "truth_states": t_states, "theta": th,
        "plaus": plaus, "truth": truth, "plaus_pct": pp, "truth_pct": tp,
        "plaus_verdict": "abnormal" if pp < 0.05 else ("borderline" if pp < 0.20 else "normal"),
        "truth_verdict": "suspect" if tp < 0.30 else "ok",
        "auc_plaus": M["auc_plaus"], "auc_truth": G["auc_truth"], "K": M["K"], "answer_start": a,
        "first_low": next((i for i, t in enumerate(trans) if i >= a and t is not None and t < 0.02), None),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
