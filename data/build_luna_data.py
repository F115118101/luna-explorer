"""Convert the LUNA authors' result CSVs (data/raw/<TASK>.csv) into luna-data.json for the page (Lab 5).

Source: https://github.com/dasong-sd/LUNA-LLM-Model-Based-Analysis  (eval/*/rq23_all_settings.csv, GPL-3.0)
Metric definitions follow the authors' own notebook eval/RQ_plots.ipynb (mapping_practitioner_metrics, n-gram = 4).
Usage:  python data/build_luna_data.py   (writes web/luna-data.json)
"""
import ast, csv, json, math, os, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
OUT = os.path.join(HERE, "..", "web", "luna-data.json")
TASKS = ["OOD-A", "ADV-A", "TQA-A", "OOD-L", "ADV-L", "TQA-L", "HE-C", "MBPP-C", "CSNJ-C", "TCS-C"]
N_GRAM = 4
csv.field_size_limit(10 ** 9)


def num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return 0.0 if math.isinf(v) else v        # authors replace inf by 0


def gram(x):
    try:
        return float(ast.literal_eval(x).get(N_GRAM))
    except Exception:
        return float("nan")


def flip(v):
    return v if v > 0.5 else 1 - v             # authors' AUC / preciseness normalisation


def clean(v):
    return None if v is None or math.isnan(v) else round(v, 6)


def row_to_config(task, r):
    p = r["settings"].split("_")              # e.g. Grid_5_3_DTMC_0_1 / KMeans_600_2048_HMM_200_0
    method, states, pca, model = p[0], int(p[1]), int(p[2]), p[3]
    hmm = int(p[4]) if model == "HMM" else None
    hist = int(p[5]) if method == "Grid" and len(p) > 5 else None
    g = lambda k: num(r[k])
    sde = abs((g("upper_bound_entropy") + g("lower_bound_entropy")) / 2 - g("stationary_distribution_entropy"))
    perp = (abs(g("abstract_model_perplexity_good") - g("abstract_model_perplexity_bad"))
            + abs(g("abstract_model_smoothed_perplexity_2_good") - g("abstract_model_smoothed_perplexity_2_bad"))) / 2
    ndt = (abs(gram(r["derivative_diversity_n_gram_level_val_increasing"]) - gram(r["derivative_diversity_n_gram_level_test_increasing"]))
           + abs(gram(r["derivative_diversity_n_gram_level_val_decreasing"]) - gram(r["derivative_diversity_n_gram_level_test_decreasing"]))) / 2
    m = {
        "SUC": g("succinctness"), "COV": g("coverage"), "SEN": g("sensitivity"), "SS": g("sink_state"),
        "PERP": perp, "SDE": sde,
        "PRE": flip(g("preciseness_mean")),
        "ENT": (g("entropy_val") + g("entropy_test")) / 2,
        "SL": g("probabilistic_reasoning_divergence"),
        "IVT": abs(g("value_diversity_instant_level_val") - g("value_diversity_instant_level_test")),
        "NVT": abs(gram(r["value_diversity_n_gram_level_val"]) - gram(r["value_diversity_n_gram_level_test"])),
        "NDT": ndt,
    }
    return {"task": task, "partition": method.lower() if method == "Grid" else method, "model": model,
            "pca": pca, "states": states, "history": hist, "hmmComp": hmm,
            "auc": clean(flip(g("aucroc"))), "metrics": {k: clean(v) for k, v in m.items()}}


configs, counts = [], {}
for t in TASKS:
    path = os.path.join(RAW, t + ".csv")
    if not os.path.exists(path):
        print("missing", t); continue
    with open(path, encoding="utf-8") as f:
        rows = [row_to_config(t, r) for r in csv.DictReader(f)]
    rows = [c for c in rows if c["auc"] is not None]
    configs += rows; counts[t] = len(rows)

out = {"meta": {"source": "LUNA authors' released results (github.com/dasong-sd/LUNA-LLM-Model-Based-Analysis)",
                "generatedAt": datetime.date.today().isoformat(), "tasks": counts},
       "configs": configs}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
with open(OUT[:-2], "w", encoding="utf-8") as f:   # luna-data.js: same data for pages opened as file://
    f.write("window.LUNA_DATA=" + json.dumps(out, ensure_ascii=False, separators=(",", ":")) + ";\n")
print("configs:", len(configs), counts)
print("wrote", os.path.normpath(OUT), round(os.path.getsize(OUT) / 1e3), "KB")
