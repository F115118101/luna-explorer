# LUNA 互動導讀

論文：*LUNA: A Model-Based Universal Analysis Framework for Large Language Models*（IEEE TSE 2024）

## 啟動

```bash
pip install torch transformers scikit-learn numpy fastapi uvicorn
python live/tune_luna.py      # 第一次：訓練實作用的模型（約 15～20 分鐘，會自動下載 GPT-2 與 TruthfulQA）
                              #   同時訓練兩個版本：「所有 decoder block 串接」與「只取最後一個 decoder block（作者程式碼的做法）」，
                              #   各自 PCA → KMeans／Grid → DTMC，以驗證集分數挑一個存成 luna_model.pkl，兩版分數寫入 luna_versions.json
python live/export_luna_data.py   # 產生 Lab 1、3、4 的真實資料（寫入 web/luna-data.json，含兩個版本的分數）
python live/server.py         # 開啟 http://127.0.0.1:8000/
```

只看導讀、不需要第 10 節的實作時，可以在 `web/` 資料夾執行 `python -m http.server`。

## 資料夾

| 資料夾 | 內容 |
|---|---|
| `web/` | 導讀網頁 `luna.html`、背景概念補充頁 `concepts.html`（導讀裡的名詞點擊後開啟）、各階段測驗 `quiz.html`，以及 Lab 5 用的真實資料 `luna-data.json` |
| `live/` | 第 10 節實作：`tune_luna.py` 訓練、`server.py` 提供分析服務 |
| `data/` | 論文作者釋出的原始結果（`raw/`）與轉換腳本 `build_luna_data.py` |
| `notes/` | 論文英文原文 `LUNA_原始.md`（對應 `LUNA_翻譯.md`，查證用）、論文翻譯、概念與公式整理、時間估算 |
| `paper/` | 論文 PDF |

## 資料來源與限制

- Lab 5（9 個任務）的真實資料來自作者的 [GitHub 倉庫](https://github.com/dasong-sd/LUNA-LLM-Model-Based-Analysis)（GPL-3.0）；ADV-A 沒有資料，維持模擬。
- Lab 1、3、4 是自己的 GPT-2 small 實跑的**真實資料**（`python live/export_luna_data.py` 產生 `web/luna-data.json`，不是論文的模型）；Lab 2 的矩陣仍由頁面內的模擬軌跡建立，標 `SIMULATED`。
- 第 10 節用 GPT-2 small 實作這套方法，兩個版本（串接／最後一層）的分數都列在該節的表格裡。評分對象與論文不同（資料集的標準答案 vs 錯誤答案，不是模型自己生成的答案），不能和論文的數字直接比，也不是事實查核器。

## 重新產生資料

```bash
python data/build_luna_data.py    # 1. data/raw/*.csv → web/luna-data.json（Lab 5；會覆蓋整個檔案）
python live/export_luna_data.py   # 2. 再補上 Lab 1、3、4（會保留步驟 1 的 Lab 5 資料，所以順序不能反）
```
