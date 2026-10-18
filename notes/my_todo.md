# web
- 3 core equations in luna.html

# LUNA HAL 換成 GPT-2 自己的答案（2026-10-19 未完成，接著做）
現況：程式都改好了，網頁資料還沒重新匯出 → **luna.html 目前會報錯（§09 版本表），匯出後才會好**。
1. 確認 `live/luna_model_gen.pkl` 有 `theta`、`km`、`K`、`ref_truth`：
   `python -c "import pickle,sys; sys.path.insert(0,'live'); print(list(pickle.load(open('live/luna_model_gen.pkl','rb'))))"`
   沒有的話重建（只擬合挑中的設定，約 10～15 分鐘；AUC 不等於 0.7701 / 0.6526 會拒絕存檔）：
   `python C:\Users\user\AppData\Local\Temp\claude\E------------\dc525412-411a-4122-8dd9-1279895259ac\scratchpad\refit_hal.py live`
   （暫存檔不見的話：`python live/tune_luna_gen.py`，全部 24 種設定重跑，約 40 分鐘）
2. `python live/export_luna_data.py`（約 4 分鐘，會檢查 OOD / HAL 的 test AUC）
3. 開 luna.html 檢查 §03 §05 §06 §09 沒有錯誤
4. commit 所有修改（不含截圖 jpg）

# me
- undertstand every toy、lab's graph meaning, such as concepts.html#roc graph
- grid vs kmeans/gmm