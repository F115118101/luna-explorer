"""Let GPT-2 answer all 817 TruthfulQA questions itself, following the authors' code (luna-author-code/reproduce/generate_nlp.py).

Usage:  python live/generate_answers.py [model]
Writes  gpt2_answers.jsonl         one line per question (text only); resumable, answered questions are skipped
        gen_hidden/<idx>.npy       hidden states recorded DURING generation, [n_steps, n_layer, 768] float16

What is copied from the authors (line numbers in generate_nlp.py):
  - questions: HF "truthful_qa" / generation / validation, 817 questions (l.528)
  - prompt = system_prompt + "Q: {question}", no few-shot examples, no trailing "\nA:" (l.17, l.77)
  - sampling: do_sample, top_k=10, top_p=0.9, temperature=0.1, max_new_tokens=100 (l.84-96)
  - trace: per generation step the LAST position of each block output (parse_hidden_states, l.25-58), i.e.
    step 0 = last prompt token, step i = generated token i (the final generated token is never fed back, so it has no state)
Our differences: GPT-2 small instead of Llama-2/Alpaca 7B; we keep all blocks (authors keep only the last, "31") so that
tune_luna.py can still build both the "concat" and the "last" version; seed fixed per question so a resumed run is identical.
"""
import json, os, sys, time
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from luna_prompt import question_prompt

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "gpt2_answers.jsonl")
HS_DIR = os.path.join(HERE, "gen_hidden")
DATA = os.path.join(HERE, "truthfulqa_hf_generation.parquet")
DATA_URL = "https://huggingface.co/datasets/truthfulqa/truthful_qa/resolve/main/generation/validation-00000-of-00001.parquet"

if not os.path.exists(DATA):
    import urllib.request
    urllib.request.urlretrieve(DATA_URL, DATA)
rows = pd.read_parquet(DATA).to_dict("records")
os.makedirs(HS_DIR, exist_ok=True)
tok = AutoTokenizer.from_pretrained(MODEL)
lm = AutoModelForCausalLM.from_pretrained(MODEL).eval()

done = set()
if os.path.exists(OUT):
    with open(OUT, encoding="utf-8") as f:
        done = {json.loads(l)["idx"] for l in f if l.strip()}
print(f"{len(rows)} questions, {len(done)} already answered", flush=True)

t0, n = time.time(), 0
with open(OUT, "a", encoding="utf-8") as f:
    for i, r in enumerate(rows):
        if i in done: continue
        torch.manual_seed(i)
        ids = tok(question_prompt(r["question"]), return_tensors="pt")
        with torch.no_grad():
            out = lm.generate(**ids, max_new_tokens=100, top_k=10, do_sample=True, top_p=0.9, temperature=0.1,
                              output_hidden_states=True, return_dict_in_generate=True, pad_token_id=tok.eos_token_id)
        # out.hidden_states: one tuple per step, each (embedding, block 1 .. block n_layer) of shape [1, seq, 768]
        h = np.stack([np.stack([layer[0, -1].numpy() for layer in step[1:]]) for step in out.hidden_states]).astype(np.float16)
        new = out.sequences[0, ids["input_ids"].shape[1]:]
        text = tok.decode(new, skip_special_tokens=True)
        np.save(os.path.join(HS_DIR, f"{i}.npy"), h)
        f.write(json.dumps(dict(idx=i, question=r["question"], generated=text, n_tokens=len(new), n_states=len(h), model=MODEL), ensure_ascii=False) + "\n"); f.flush()
        n += 1
        if n % 25 == 0:
            el = time.time() - t0
            print(f"{len(done) + n}/{len(rows)}  {el / n:.2f}s/q  eta {el / n * (len(rows) - len(done) - n) / 60:.0f} min", flush=True)
print(f"done: {n} new answers in {(time.time() - t0) / 60:.1f} min -> {OUT}")
