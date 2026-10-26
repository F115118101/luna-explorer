# LUNA 互動導讀

論文：*LUNA: A Model-Based Universal Analysis Framework for Large Language Models*（IEEE TSE 2024，[DOI: 10.1109/TSE.2024.3411928](https://doi.org/10.1109/TSE.2024.3411928)）。論文 PDF 與全文受 IEEE 版權保護，不包含在本 repo 中，請從 IEEE Xplore 取得。

## 啟動

```bash
pip install torch transformers scikit-learn numpy fastapi uvicorn
# OOD（合理度）模型
python live/tune_luna.py          # 正常 = TruthfulQA 的問題 + Best Answer，異常 = 同一題但答案被打亂、插入亂詞（約 15～20 分鐘，會自動下載 GPT-2 與 TruthfulQA）
                                  #   同時訓練兩個版本：「所有 decoder block 串接」與「只取最後一個 decoder block（作者程式碼的做法）」，
                                  #   各自 PCA → KMeans／Grid → DTMC，以驗證集分數挑一個存成 luna_model.pkl，兩版分數寫入 luna_versions.json
# 幻覺（真實度）模型：照作者流程，用 GPT-2 自己的回答
python live/generate_answers.py   # GPT-2 回答 817 題 TruthfulQA，生成時記錄 hidden states（gpt2_answers.jsonl、gen_hidden/）
python live/judge_answers.py      # Gemini 把每個回答標成真／假，取代作者的 GPT-judge（需要在 live/.env 設 GEMINI_API_KEY）
python live/tune_luna_gen.py      # 照作者的切分與語意綁定訓練，以驗證集挑一個存成 luna_model_gen.pkl
python live/export_luna_data.py   # 產生 Lab 1、3、4 的真實資料（寫入 web/luna-data.json，含兩個版本的分數）
python live/server.py         # 開啟 http://127.0.0.1:8000/
```

只看導讀、不需要第 09 節的實作時，可以在 `web/` 資料夾執行 `python -m http.server`。

## 資料夾

| 資料夾 | 內容 |
|---|---|
| `web/` | 導讀網頁 `luna.html`、背景概念補充頁 `concepts.html`（導讀裡的名詞點擊後開啟）、各階段測驗 `quiz.html`，以及 Lab 5 用的真實資料 `luna-data.json` |
| `live/` | 第 09 節實作：`tune_luna.py`（OOD）、`generate_answers.py` → `judge_answers.py` → `tune_luna_gen.py`（幻覺）訓練，`server.py` 提供分析服務 |
| `data/` | 論文作者釋出的原始結果（`raw/`）與轉換腳本 `build_luna_data.py` |
| `notes/` | 概念與公式整理、時間估算、報告準備筆記（論文全文與翻譯只留在本機，不上傳） |

## 資料來源與限制

- Lab 5（9 個任務）的真實資料來自作者的 [GitHub 倉庫](https://github.com/dasong-sd/LUNA-LLM-Model-Based-Analysis)（GPL-3.0）；ADV-A 沒有資料，維持模擬。
- Lab 1、3、4 是自己的 GPT-2 small 實跑的**真實資料**（`python live/export_luna_data.py` 產生 `web/luna-data.json`，不是論文的模型）；Lab 2 的矩陣仍由頁面內的模擬軌跡建立，標 `SIMULATED`。
- 第 09 節用 GPT-2 small 實作這套方法，兩個版本（串接／最後一層）的分數都列在該節的表格裡。幻覺任務和作者一樣用模型自己生成的回答，但標籤改由 Gemini 判定（prompt 附上參考答案、只有 0／1 沒有機率值）；OOD 任務用打亂的 TruthfulQA 答案，不是論文的 SST-2 風格改寫。模型、judge 與資料都不同，不能和論文的數字直接比，也不是事實查核器。完整差異見 `web/concepts.html#errata` 第 7 段。

## 重新產生資料

```bash
python data/build_luna_data.py    # 1. data/raw/*.csv → web/luna-data.json（Lab 5；會覆蓋整個檔案）
python live/export_luna_data.py   # 2. 再補上 Lab 1、3、4（會保留步驟 1 的 Lab 5 資料，所以順序不能反）
```
