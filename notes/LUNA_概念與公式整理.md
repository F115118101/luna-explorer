# LUNA 論文：概念與公式整理

> 論文：*LUNA: A Model-Based Universal Analysis Framework for Large Language Models*（IEEE TSE 2024, Song et al.）
>
> **標註約定**
> - 【原文】= 論文正文明確給出的內容／公式
> - 【補充】= 論文僅提及名稱、由我依通用教科書定義補充的背景公式（幫助理解，非論文原文）
> - 12 個指標的精確定義已取自 **arXiv 版（2310.14211 v2）Appendix A**，見第 6 節。

---

## 目錄

1. [一句話總覽與整體流程](#1-一句話總覽與整體流程)
2. [背景概念：LLM 相關](#2-背景概念llm-相關)
3. [三大可信賴性面向](#3-三大可信賴性面向)
4. [抽象模型建構（核心方法）](#4-抽象模型建構核心方法)
5. [語意綁定](#5-語意綁定)
6. [12 個品質指標](#6-12-個品質指標)
7. [應用：異常行為偵測](#7-應用異常行為偵測)
8. [實驗設計：資料集、模型、超參數](#8-實驗設計資料集模型超參數)
9. [統計與評估工具](#9-統計與評估工具)
10. [實驗結果重點整理](#10-實驗結果重點整理)
11. [公式速查表](#11-公式速查表)
12. [名詞對照表（中英）](#12-名詞對照表中英)

---

## 1. 一句話總覽與整體流程

**LUNA** = 把 LLM 生成時「每個 token 的解碼器隱藏狀態」壓縮成一個**小的離散狀態機率模型（DTMC / HMM）**，再把「可信賴性語意」綁到狀態（或轉移）上，用這個可解釋的代理模型去**偵測異常輸出**（幻覺、OOD、對抗攻擊）。

```
受測 LLM + 面向專屬資料
        │  剖析 (profile)：取出各 token 的 decoder block 輸出 = 具體狀態 / 軌跡
        ▼
① 抽象模型建構   PCA 降維 → 狀態劃分(Grid / GMM / KMeans) → 轉移抽象 → DTMC 或 HMM
        ▼
② 語意綁定       把「可信賴性滿足程度」(truthfulness、轉移機率…) 綁到抽象狀態/轉移
        ▼
③ 模型品質評估   6 個抽象模型層級指標 + 6 個語意層級指標
        ▼
④ 實際應用       軌跡語意平均 → 語意分數 → 與門檻比較 → 正常/異常
```

**設計特性**：白箱（需 LLM 內部隱藏狀態，所以只能用開源 LLM）、通用（多面向）、可擴充（可插入新抽象法／新指標）、人類可解釋（能指出「在哪個詞變異常」）、可線上逐詞監控。

---

## 2. 背景概念：LLM 相關

### 2.1 Transformer 與自注意力（self-attention）
- 【原文概念】自注意力讓序列中每個元素與其他所有元素互相比較，重新調整表示。例：「A baby sat on the chair」，處理 *sat* 時對 *baby*、*chair* 的關注高於 *the*。
- 優點：處理長距離依賴、避免 RNN 的梯度消失。
- 自注意力位於 **decoder block（解碼器區塊）**，本文取其**輸出**作為「具體狀態」。
- 【補充】標準式：$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$

### 2.2 LLM 三種架構
| 類型 | 預訓練方式 | 代表 |
|---|---|---|
| Encoder-only | 遮蔽部分 token 再預測 | BERT、DeBERTa、RoBERTa |
| Encoder-decoder | 編碼成隱藏向量→解碼成輸出序列，擅長 seq2seq | BART、Flan-UL2、T5 |
| **Decoder-only（本文主要對象）** | 自迴歸，依序預測下一個 token | GPT-3/4、Llama、CodeLlama |

### 2.3 訓練目標：下一 token 預測【原文 式(1)】

$$\max_\theta \sum_{t=1}^{T}\log p(w_t\mid w_{1:t-1};\theta)$$

- $w_{1:t-1}$：前 $t-1$ 個詞；$w_t$：第 $t$ 個詞；$\theta$：模型參數；$T$：序列長度。
- 意義：最大化整段序列的對數似然。

### 2.4 自迴歸生成機率【原文 式(3)】

$$p(y=y_1,\dots,y_T)=\prod_{t=1}^{T}p(y_t\mid y_{<t})$$

- $y_{<t}=(y_0,\dots,y_{t-1})$，$y_0$ 為使用者 prompt，$y_T$ 為 EOS（結束符）。
- 生成 = 反覆呼叫 DNN 的**隨機過程**；正因為「有狀態」，才可用馬可夫模型去抽象。
- 例：「Who is the president of the USA?」→ Joe → Biden → EOS。

### 2.5 為何 LLM 比傳統分類 DNN 更難分析（原文三點）
1. 生成品質取決於整體結構與語意連貫，而非單一 token。
2. 每步輸出依賴所有先前輸出。
3. 輸出空間龐大且跨主題。
- RNN 分類可視為「只輸出一個標籤」的自迴歸生成特例。

---

## 3. 三大可信賴性面向

### 3.1 分布外偵測（OOD Detection）
- **前提**：訓練與測試資料同分布；偏離時 DNN 可能「高度自信地出錯」。
- 【原文 式(2)】以分布估計函數 $P_X$（$X$ 為訓練分布）與門檻 $\lambda$ 判斷：

$$g(x)=\begin{cases}\text{in} & P_X(x)\ge\lambda\\ \text{out} & P_X(x)<\lambda\end{cases}$$

- 分布偏移（Moreno-Torres 框架）：$p(y,x)=p(y\mid x)\,p(x)$；**$p(x)$ 變（covariate shift）或 $p(y\mid x)$ 變**都算偏移。
- LLM 的 OOD 困難：訓練資料取不到／太大；湧現能力使單一任務式評量不適用；高維多模態空間中任務邊界不清。
- **本文做法：OOD style（風格轉換）**——把 SST-2 句子改成莎士比亞風格（詞級＋句級），遵循 DecodingTrust。

### 3.2 對抗攻擊（Adversarial Attacks）
- DNN 對細微擾動敏感（高度非線性）。影像領域＝像素連續擾動；文本領域＝**離散**的詞／句／整段擾動。
- 攻擊可表述成最佳化問題：找到能改變預測的擾動。
- 基準：Adversarial GLUE → DecodingTrust 的 **AdvGLUE++**（用三個開源自迴歸模型生成對抗文本）。
- 防禦分類：對抗訓練、擾動控制、強健性認證。本文採**輕量的對抗偵測**（屬擾動控制）：偵測到就拒絕或採取其他動作。

### 3.3 幻覺（Hallucination）
- 定義：產生不忠實／不符事實的內容。
- **內在幻覺（Intrinsic）**：輸出與來源內容**矛盾**。例：問「你有人類特質嗎？」答「有」。
- **外在幻覺（Extrinsic）**：輸出**無法由來源驗證**。例：「預測明天 NASDAQ？」答「會漲」。
- 緩解方法兩類：**資料相關**（清理、資訊增強）／**建模與推論**（改架構、訓練期最佳化）——對 LLM 皆昂貴、不易套用。
- 本文任務：自然語言問答、程式碼生成、程式碼摘要。

---

## 4. 抽象模型建構（核心方法）

### 4.1 為什麼要「抽象」
- 具體狀態維度極高：論文例：Llama-7b，32 個 decoder block × 4096 維 = **131,072 維／token**（論文沒說明各層如何組合；作者程式碼實際只取最後一層 4096 維）。
- 需 ① **降維**、② **狀態劃分**，把連續高維空間映射到小的離散抽象狀態空間，之後一次推論 = 一串抽象狀態轉移序列。

### 4.2 狀態抽象

#### (a) 降維：PCA【原文提及；式為補充】
- 把具體狀態轉成 $k$ 維向量，保留最多變異。
- 【補充】對中心化資料的共變異矩陣 $\Sigma$ 做特徵分解，取前 $k$ 大特徵值對應向量 $W_k$：$z=W_k^\top (s-\mu)$；解釋變異比 $=\sum_{i\le k}\lambda_i/\sum_i\lambda_i$。
- 論文 PCA 成分數：Grid 用 {3, 5, 10}；Cluster 用 {512, 1024, 2048}（分別視為 Low/Medium/High）。

#### (b) 多步抽象（multi-step abstraction，僅用於 Grid）
- 用**滑動 $N$ 步視窗**把相鄰時間步合併：$N=2$ 時 $\{s_i,s_{i+1}\}$ 與 $\{s_{i+1},s_{i+2}\}$ 是不同抽象。對應表 III 的 **History Step ∈ {1,2,3}**。

#### (c) 網格劃分（Grid-based partition）
- $k$ 維每一維均勻分成 $m$ 格，$c^i_j$ 表第 $i$ 維第 $j$ 格。落在同一組格內的壓縮後具體狀態 → 同一抽象狀態【原文】：

$$\bar s=\{\,s_i \mid s^1_i\in c^1_{\_}\ \wedge\cdots\wedge\ s^k_i\in c^k_{\_}\,\}$$

- **狀態總數 $=m^k$**（表 III 註 †：「狀態數的 PCA 成分數次方」，如 $10^{10}$）。
- 缺點：即使沒有具體狀態落入，也會產生「空洞抽象狀態」→ 覆蓋率、簡潔性差。

#### (d) 分群劃分（Cluster-based partition）
- 用 **GMM**（高斯混合模型）或 **K-Means** 把具體狀態分成 $n$ 群，每群＝一個抽象狀態。
- 只在具體狀態聚集時才建立抽象狀態 → 對稀疏空間更精簡。
- 【補充】K-Means：最小化 $\sum_{j=1}^{n}\sum_{s\in C_j}\lVert s-\mu_j\rVert^2$。GMM：$p(s)=\sum_{j}\pi_j\,\mathcal N(s\mid\mu_j,\Sigma_j)$，取後驗最大之成分為所屬群。

### 4.3 轉移抽象
- **存在性**：抽象狀態 $\bar s\to\bar s'$ 有抽象轉移 $\bar t$，若且唯若存在具體轉移 $s\to s'$，$s\in\bar s$、$s'\in\bar s'$。
- **機率（頻率估計）**【原文文字敘述，式為據此寫出】：

$$\bar P(\bar s,\bar s')=\frac{\#\{\text{具體轉移 } s\to s',\ s\in\bar s,\ s'\in\bar s'\}}{\#\{\bar s\ \text{的所有外向具體轉移}\}}$$

### 4.4 DTMC（離散時間馬可夫鏈）

**定義 1【原文】**：DTMC 是元組 $(\bar S,\bar s_0,\bar\delta,\bar P)$
- $\bar S$：有限狀態集合
- $\bar s_0\in\bar S$：初始狀態
- $\bar\delta$：轉移集合
- $\bar P:\bar S\times\bar S\to[0,1]$：轉移機率函數
- 【補充】馬可夫性 $P(X_{t+1}=j\mid X_t=i,\dots)=P_{ij}$，且每列和為 1：$\sum_j P_{ij}=1$。

### 4.5 HMM（隱馬可夫模型）

**定義 2【原文】**：HMM 是元組 $(\bar S,\bar\delta,\bar P,\bar O,\bar E,\bar I)$
- $S$：**隱藏**狀態空間；$\bar\delta$：轉移空間
- $\bar P:\bar S\times\bar S\to[0,1]$：轉移機率
- $\bar O=\{o_1,\dots,o_n\}$：有限**觀測**集合
- $\bar E:(s_i,o_j)\to[0,1]$：**發射函數**（由 $s_i$ 產生 $o_j$ 的機率）
- $\bar I:S\to[0,1]$：**初始狀態機率**

**本文 HMM 的建法**：
- 觀測 $\bar O$ = DTMC 建構中看過的**所有抽象狀態**；隱藏狀態數 = 超參數 **HMM Comp ∈ {100, 200, 400}**。
- 用 **Baum-Welch（EM 演算法）** 擬合 $\bar P,\bar E,\bar I$：
  - **E 步**：給定觀測軌跡，計算隱藏狀態的條件期望（前向-後向）。
  - **M 步**：更新參數以最大化觀測似然。
- 【補充】$\alpha_t(i)=P(o_{1:t},q_t=i)$；$\xi_t(i,j)=P(q_t=i,q_{t+1}=j\mid O)$；$\hat a_{ij}=\frac{\sum_t\xi_t(i,j)}{\sum_t\gamma_t(i)}$。
- HMM＝「**兩層抽象**」（抽象狀態之上再學隱藏狀態）→ 論文認為這是其表現落後 DTMC 的可能原因。

### 4.6 DTMC vs HMM 比較（論文 RQ2.2/RQ3.1 結論）
| 項目 | DTMC | HMM |
|---|---|---|
| 轉移來源 | 具體轉移直接計數 | Baum-Welch 最大化似然 |
| 抽象層數 | 一層 | 兩層 |
| 大多指標 | 接近或優於 | 普遍落後（SDE、SEN 例外） |
| Top-5 設定 | **全部為 DTMC** | 無 |

---

## 5. 語意綁定

### 5.1 動機
不同面向（OOD、對抗、幻覺）需統一成一個**可量化**的抽象概念 → **語意（semantics）**。

### 5.2 定義 3【原文】
具體狀態序列 $\tau^k=\langle s_i,\dots,s_{i+k-1}\rangle$ 的**具體語意** $\theta\in\mathbb R^n$，表示 LLM 相對於可信賴性面向的**滿足程度**。（$k=1$ 時省略記號。）

### 5.3 語意抽象
- **抽象語意 $\bar\theta$** = 該抽象狀態內**所有具體語意的平均**：$\bar\theta(\bar s)=\frac{1}{|\bar s|}\sum_{s\in\bar s}\theta(s)$（依文字敘述寫成式子）。

### 5.4 各面向的語意綁定方式
| 面向 | 綁定位置 | 語意內容 |
|---|---|---|
| 幻覺 – TruthfulQA | **狀態** | GPT-judge（微調 GPT-3-13B）給的真實性（答案為真的機率） |
| 幻覺 – HumanEval / MBPP | 狀態 | **pass@1**：通過所有測試＝真，任一失敗＝假 |
| 幻覺 – CodeSearchNet / TL-CodeSum | 狀態 | GPT-3.5-Turbo-Instruct 將摘要與 ground truth 比較，準確相關＝真 |
| OOD | **轉移** | $\bar\theta(\langle\bar s_i,\bar s_{i+1}\rangle)$ = 轉移機率（機率高＝較像訓練分布） |
| 對抗 | **轉移** | 同 OOD，轉移機率 |

> 綁在**轉移**上的任務（OOD、對抗）Grid 表現特別好；綁在**狀態**上的（幻覺）較不然。

---

## 6. 12 個品質指標（附錄 A 精確定義）

> **來源**：論文 IEEE 版的附錄 A 是補充材料，未包含在原 PDF。以下公式取自 **arXiv 版 v2（2310.14211）的 Appendix A「Metrics Classification」**（與 IEEE 版同一篇文章的作者版本，34 頁）。標記【原文附錄】＝逐項對照 arXiv 附錄整理；【說明】＝我加的解讀。
> arXiv PDF 抽出的公式有部分排版破損，已依上下文還原，符號以此為準。

### 6.1 抽象模型層級指標（Abstract model-wise，6 個）

#### ① 簡潔性 SUC（Succinctness）【基本】
$$\mathrm{SUC}=\frac{|\bar S|}{|S|}\quad(\text{狀態縮減率})$$
- $|\bar S|$：抽象狀態數，$|S|$：具體狀態數。IEEE 正文另提及**轉移縮減率** $|\bar\delta|/|\delta|$。
- **越低越好**（壓縮得越簡潔），但不能過度：只有 1 個狀態的模型簡潔性最佳卻資訊最少，所以要搭配 SDE 找平衡點。

#### ② 覆蓋率 COV（Coverage）【基本】
$$\mathrm{COV}=\frac{|\bar S_{\text{unseen}}|}{|\bar S|}$$
（狀態與轉移各算一次：未見過的抽象狀態／轉移數 ÷ 抽象狀態／轉移總數）
- **Grid 劃分**：落在網格之外的抽象狀態視為 $\bar S_{\text{unseen}}$。
- **Cluster 劃分**：
$$\bar S_{\text{unseen}}=\{\,s\mid \lVert s-c_i\rVert>d\,\}$$
  $c_i$＝群中心，$d$＝訓練集所有具體狀態到其群中心的**最大距離**。
- **越低越好**（測試資料很少出現訓練時沒見過的狀態＝抽象完整）。

#### ③ 敏感度 SEN（Sensitivity）【基本】
$$\mathrm{SEN}=\frac{\big|\{\,s\mid s\in S_{\text{test}},\ \bar s\neq\bar s',\ \lVert s-s'\rVert<\epsilon\,\}\big|}{|S_{\text{test}}|}$$
- $s'$ 為 $s$ 加上小擾動（$\lVert s-s'\rVert<\epsilon$）後的狀態，$\bar s,\bar s'$ 為各自所屬抽象狀態。
- 【說明】＝擾動後「跳到別的抽象狀態」的具體狀態比例；**越低越好**。

#### ④ 狀態分類 SS（State classification）【狀態層級】
- 依轉移矩陣 $P$ 分類馬可夫鏈狀態，主要檢查**匯點（sink）**：狀態 $s_i$ 為 sink 若 $p_{ii}=1$（唯一外向轉移是機率 1 的自迴圈；原文寫作「$p_{ij}=1$」，文字說明為自迴圈）。
- 指標＝有多大比例的抽象狀態是 sink（可判斷模型是否偏向某些回應）。
- 其他類型：**source state**（可到達其他狀態但無法從其他狀態回到自己）、**recurrent state**（從其他狀態重新被造訪的機率為 1）。

#### ⑤ 穩態分布熵 SDE（Stationary Distribution Entropy）【模型層級】
$$\mathrm{SDE}=-\sum_{i=1}^{n}\pi_i\Big(\sum_{j=1}^{n}p_{ij}\log p_{ij}\Big)$$
- $p_{ij}$：狀態 $i\to j$ 轉移機率；$\pi$：**穩態分布**，滿足 $\pi P=\pi$（長期停留在各狀態的時間比例）。
- 熵高＝不可預測，熵低＝較確定。
- **界的定義**（依 [142]）：
  - **穩定界（stable bound，低熵）**：系統以高機率回到同一狀態，$p_{ii}=0.95$。
  - **隨機界（stochastic bound，高熵）**：$p_{ij}\approx\frac1n$（近乎均勻隨機）。
  - 中熵介於兩者之間。
- **最終指標值** ＝ 「隨機界與穩定界之**平均**」與「模型 SDE」之**差**（差越小越接近理想平衡）。

#### ⑥ 困惑度 PERP（Perplexity）【模型層級】
$$\mathrm{PERP}=\sqrt[N]{\prod_{i=1}^{N}\frac{1}{P(w_i\mid w_1w_2\cdots w_{i-1})}}$$
- 好的狀態，困惑度應 **< 100**（依 [127] 經驗）。
- **最終指標值** ＝ 正常資料與異常資料兩者困惑度之間的 **KL 散度**（差異越大，代表模型越能區分正常／異常）。

### 6.2 語意層級指標（Semantics-wise，6 個，本文新提出）

> 符號：$\tau^k=\langle s_i,\dots,s_{i+k}\rangle$ 為測試用具體軌跡（$k=0$ 時退化為單一狀態層級）；$\bar\tau^k$ 為對應抽象軌跡；$\theta$／$\bar\theta$ 為具體／抽象語意。

#### ⑦ 語意精確度 PRE（Preciseness）【基本】
$$\mathrm{PRE}=\mathrm{MEAN}\Big\{\theta(\tau^k)-\mathbb E\big[\bar\theta(\bar\tau^k)\big]\Big\}$$
- 即**平均語意誤差**（正文表 II 還提到「最大語意誤差」）：具體語意與其抽象語意期望值之差。
- 附錄文字寫「**越高越好**」（表示所蒐集語意與 LLM 真實語意空間吻合）。
  【說明】此方向與「誤差」字面看似矛盾，原文如此，建議以論文實作為準。

#### ⑧ 語意熵 ENT（Entropy）【基本】
$$\mathrm{ENT}=-\sum_{i=1}^{n}\theta(\tau_i)\log\theta(\tau_i),\qquad \tau_i\in\bar\tau$$
- 最終指標值同 SDE：以「隨機界／穩定界平均」與語意熵之差表示，量測流程與 SDE 類似。

#### ⑨ 瞬時值趨勢 IVT（Instant Value Trend）【軌跡層級 - value】
$$\mathrm{IVT}=\min_{i=0}^{k}\big|\theta(\tau_{t_i})-v\big|$$
- 給定目標值 $v$ 與每個模擬步的語意 $\theta(\tau_{t_i})$ 之**最小差距**。

#### ⑩ n-gram 值趨勢 NVT【軌跡層級 - value】
$$\mathrm{NVT}=\min_{i=n}^{k}\Big(\sum_{j=i-n}^{i}\big|\theta(\tau_{t_j})-v\big|\Big)$$
- 連續 $n$ 個模擬步與 $v$ 的差距總和之最小值。
- **目標值設定**：正常軌跡 $v=1$（ground truth 為 1），異常軌跡 $v=0$；**值越小越好**（越貼近 ground truth）；最後回報正常與異常軌跡的平均。

#### ⑪ n-gram 導數趨勢 NDT（Derivative Trend）【軌跡層級 - derivative】
先算上升與下降趨勢（$\mathrm{lds}$＝**左導數符號**，left derivative sign）：

$$\text{Increasing}=\max_{i=n}^{k}\Big(\sum_{j=i-n+1}^{i}\mathrm{lds}\big(\theta(\tau_{t_j})\big)\Big)$$

$$\text{Decreasing}=\min_{i=n}^{k}\Big(\sum_{j=i-n+1}^{i}\mathrm{lds}\big(\theta(\tau_{t_j})\big)\Big)$$

$$\mathrm{NDT}=\big|\text{Increasing}-\text{Decreasing}\big|$$

- 正常軌跡：期望 Increasing 高（維持上升）、Decreasing 低；異常軌跡相反（維持下降趨勢）。
- 取兩者**絕對差**：差越大代表語意趨勢越「純」。最後回報正常與異常軌跡的平均。

#### ⑫ 意外程度 SL（Surprise Level）【意外層級】
（本文考慮**二元語意**，量化語意可類推。）對第 $l$ 條序列 $\tau^l$：

- **先驗**：$\ \mathbb P(\theta(\tau^l)=\text{good})$
- **後驗**（貝氏定理）：

$$\mathbb P\big(\theta(\tau^l)=\text{good}\mid\theta(\tau^{l+1})=\text{good}\big)=\frac{\mathbb P\big(\theta(\tau^{l+1})=\text{good}\mid\theta(\tau^{l})=\text{good}\big)\,\mathbb P\big(\theta(\tau^l)=\text{good}\big)}{\mathbb P\big(\theta(\tau^{l+1})=\text{good}\big)}$$

- 對「好／壞」狀態的先驗與後驗計算 **KL 散度**，再取**平均**即為 Surprise degree：
$$\mathrm{SL}=\mathrm{mean}\Big(D_{KL}\big(\text{posterior}\,\|\,\text{prior}\big)\Big)$$
- **閾值設計**（與熵類似，靈感來自 Kim et al. [20]）：
  - 「極大意外」：**純隨機轉移矩陣＋隨機指派語意**的抽象模型之 surprise；
  - 「低意外」：高機率回到同一狀態（$p_{ii}=0.95$）的模型。

### 6.3 六＋六指標速查

| # | 指標 | 類別 | 核心式（簡寫） | 期望方向 |
|---|---|---|---|---|
| ① | SUC | 模型-基本 | $\lvert\bar S\rvert/\lvert S\rvert$ | 低（但須保資訊） |
| ② | COV | 模型-基本 | $\lvert\bar S_{unseen}\rvert/\lvert\bar S\rvert$ | 低 |
| ③ | SEN | 模型-基本 | 擾動後換狀態比例 | 低 |
| ④ | SS | 模型-狀態 | sink：$p_{ii}=1$ 的狀態比例 | 依任務（判偏向） |
| ⑤ | SDE | 模型-模型 | $-\sum_i\pi_i\sum_jp_{ij}\log p_{ij}$，取與界平均之差 | 接近平衡 |
| ⑥ | PERP | 模型-模型 | $\sqrt[N]{\prod 1/P(w_i\mid w_{<i})}$，取正常 vs 異常之 KL | 好狀態 <100；差異大 |
| ⑦ | PRE | 語意-基本 | $\mathrm{MEAN}\{\theta-\mathbb E[\bar\theta]\}$ | 高（原文敘述） |
| ⑧ | ENT | 語意-基本 | $-\sum\theta\log\theta$，取與界平均之差 | 接近平衡 |
| ⑨ | IVT | 語意-軌跡 | $\min_i\lvert\theta_i-v\rvert$ | 小（$v$=1 正常／0 異常） |
| ⑩ | NVT | 語意-軌跡 | $\min_i\sum_{j=i-n}^{i}\lvert\theta_j-v\rvert$ | 小 |
| ⑪ | NDT | 語意-軌跡 | $\lvert\max\sum\mathrm{lds}-\min\sum\mathrm{lds}\rvert$ | 大（趨勢純） |
| ⑫ | SL | 語意-意外 | $\mathrm{mean}\,D_{KL}(\text{post}\Vert\text{prior})$ | 介於低／高意外閾值 |

---

## 7. 應用：異常行為偵測

### 7.1 流程（幻覺與 OOD 同構）
1. 輸出文字 → 取得**抽象狀態軌跡** $\{\bar s_1,\dots,\bar s_n\}$
2. 對應**語意軌跡** $\{\bar\theta(\bar s_1),\dots,\bar\theta(\bar s_n)\}$（OOD 則用轉移機率語意 $\bar\theta(\langle\bar s_i,\bar s_{i+1}\rangle)$）
3. **語意分數**【原文】：

$$\text{score}=\mathrm{AVG}\big(\bar\theta(\bar s_i)\big)=\frac1n\sum_{i=1}^{n}\bar\theta(\bar s_i)$$

4. 分數對比 ground truth → 分類為正常／異常，以 **ROC AUC** 評估（等同於掃描所有門檻）。

### 7.2 論文範例
- **幻覺（TruthfulQA）**：「Which is denser, water vapor or air?」→ 答「Water vapor is denser than air.」
  狀態序列 $\bar s_{199}\to\bar s_5\to\cdots\to\bar s_{159}\to\bar s_4$；語意 $0.12\to0.23\to\cdots\to0.01\to0.15$；**分數 0.04 → 判為異常**；並可定位 $\bar s_{159}$（詞 "denser"）為異常狀態 ⇒ **可解釋、可定位錯誤 token**。
- **OOD（SST-2）**：莎士比亞風格改寫後，狀態序列由 $\bar s_6$ 走到 $\bar s_{10}$、語意下降，**分數 0.34**，遠低於分布內樣本 **0.78**。

### 7.3 與 SAPLMA、ITI 的比較（討論節）
| | LUNA | SAPLMA [81] | ITI [169] |
|---|---|---|---|
| 做法 | 抽象模型＋有狀態行為分析 | 用隱藏層激活訓練真/假分類器 | 推論時把激活往「真實方向」偏移 |
| 面向 | 多面向（真實性、OOD、對抗）、含程式任務 | 二元真假 QA | 提升 QA 真實性 |
| 時機 | **可線上、逐 token 監控** | 推論結束後看整個輸出 | 推論期介入 |

### 7.4 可延伸應用（論文展望）
線上監控、錯誤定位、測試案例生成、輸出修復。

---

## 8. 實驗設計：資料集、模型、超參數

### 8.1 受測 LLM 與選擇準則
準則：**開源可得**（需內部狀態）、**有競爭力**、**部署成本可接受**。
- **Alpaca-7b**（A）、**Llama2-7b**（L）、**CodeLlama-13b-Instruct**（C）
- 環境：AMD 5955WX 16 核、256GB RAM、2×NVIDIA A6000 48GB；總計 **>6,300 小時**。

### 8.2 資料集
| 面向 | 資料集 | 規模／特點 | 對應 LLM |
|---|---|---|---|
| OOD | SST-2 風格轉換（DecodingTrust） | 9,603 句：873 ID + 8,730 OOD | A、L |
| 對抗 | AdvGLUE++ | 11,484 筆；3 類任務（情感分類、重複問題偵測、MNLI）、5 種詞級攻擊 | A、L |
| 幻覺-QA | TruthfulQA | 817 題、38 類謊誤 | A、L |
| 幻覺-程式生成 | HumanEval | 164 題（Python，docstring→程式） | C |
| 幻覺-程式生成 | MBPP | 399 題 | C |
| 幻覺-程式摘要 | TL-CodeSum | 87,136 對方法-摘要，9,732 個 Java 專案；抽 5,000/5,000/5,000（訓練/驗證/測試，8:1:1 切分） | C |
| 幻覺-程式摘要 | CodeSearchNet (Java) | 六種語言中取 Java；同樣抽 5,000 | C |

**pass@1**【補充】：一次生成即通過所有測試案例的比例。

### 8.3 超參數空間（表 III，共 180 種）
| 劃分 | 模型 | PCA 成分 | #狀態 | 額外參數 | 設定數 |
|---|---|---|---|---|---|
| Grid | DTMC | {3,5,10} | {5,10,15} | History {1,2,3} | 3×3×3 = **27** |
| Grid | HMM | {3,5,10} | {5,10,15} | History {1,2,3}；HMM Comp {100,200,400} | 3×3×3×3 = **81** |
| Cluster | DTMC | {512,1024,2048} | {200,400,600} | Cluster {GMM, KMeans} | 3×3×2 = **18** |
| Cluster | HMM | {512,1024,2048} | {200,400,600} | Cluster；HMM Comp {100,200,400} | 3×3×2×3 = **54** |

合計 27+81+18+54 = **180**。Grid 的「#狀態」是每維格數，實際狀態數為其 PCA 成分數次方。

---

## 9. 統計與評估工具

| 工具 | 用途（論文） | 定義【補充】 |
|---|---|---|
| **ROC AUC** | 偵測效能；>0.5 即優於隨機 | ROC 曲線（TPR vs FPR）下面積；亦 = 隨機抽一正一負，正分數高於負分數的機率 |
| **KDE** 核密度估計 | RQ1 定性：畫三類實例的轉移機率分布 | $\hat f(x)=\frac1{nh}\sum_i K\!\big(\frac{x-x_i}{h}\big)$ |
| **Mann-Whitney U 檢定** | RQ1 定量：正常 vs 異常分布是否顯著不同（$p<0.05$） | 非參數、基於秩的兩樣本檢定；$U=n_1n_2+\frac{n_1(n_1+1)}2-R_1$ |
| **顯著比例 (significant proportion)** | 180 個模型中 $p<0.05$ 者的比例 | — |
| **Pearson 相關** | 指標 vs ROC AUC 的線性相關 | $r=\frac{\sum(x_i-\bar x)(y_i-\bar y)}{\sqrt{\sum(x_i-\bar x)^2\sum(y_i-\bar y)^2}}$；易受離群值影響 |
| **Kendall's τ** | 同上，基於序位、對離群值強健 | $\tau=\frac{C-D}{\binom n2}$（一致對數 C 與不一致對數 D）；論文觀察 Kendall 通常略小於 Pearson |
| **KL 散度** | PERP（正常 vs 異常）、SL（先驗 vs 後驗） | $D_{KL}(P\|Q)=\sum_xP(x)\log\frac{P(x)}{Q(x)}$ |
| **rank 正規化** | RQ2 圖：以排名而非絕對值呈現各指標 | 值越高＝在該指標排名越好 |

---

## 10. 實驗結果重點整理

### RQ1：抽象模型能否區分正常／異常？— **能**
- 訓練與測試「正常」實例的轉移機率分布高度一致；「異常」實例分布不同（除了 TL-CodeSum 差異較小，其他任務都看得出來）。
- 顯著比例（$p<0.05$ 的模型占 180 個的比例）：TruthfulQA-A 低至 **20%**，HumanEval 高達 **64%**；即至少 20% 的抽象模型具偵測潛力。

### RQ2：建模技術與設定如何影響品質？
**RQ2.1（狀態抽象）**
- **PCA 成分↑** ⇒ 困惑度、覆蓋率↑（10 任務中 9 個 PERP 改善、8 個 COV 改善），但**敏感度、SDE 變差**（保留過多特徵→易受雜訊）。→ 需權衡。
- **Cluster（KMeans/GMM）**：COV、SUC 最高或近最高（只在有聚集處建狀態）；GMM 居中；KMeans 的 SEN 得分較 GMM 高。
- **Grid**：SDE 較優（TruthfulQA 除外），SEN、PERP 相當；特別適合 OOD／對抗（語意綁轉移）；缺點是空洞狀態拉低 COV／SUC。
- 結論：cluster 擅長狀態空間縮減與探索；grid 擅長保留轉移分布與確定性。

**RQ2.2（模型類型）**：**DTMC ≥ HMM**（SDE、SEN 例外），DTMC 是下游應用首選。

### RQ3：實際偵測效能與指標相關性

**RQ3.1 – ROC AUC（表 V 摘要，各設定下的 mean / max）**
| 任務 | LLM | mean | max |
|---|---|---|---|
| OOD | A | 0.55 | 0.65 |
| OOD | L | 0.63 | **0.85** |
| 對抗 | A | 0.59 | **0.83** |
| 對抗 | L | 0.62 | 0.80 |
| TruthfulQA | A / L | 0.67 / 0.64 | 0.71 / 0.69 |
| HumanEval | C | 0.69 | **0.90**（全任務最高） |
| MBPP | C | 0.67 | 0.73 |
| CodeSearchNet-Java | C | 0.66 | 0.70 |
| TL-CodeSum | C | 0.70 | 0.74 |

- 所有結果 mean > 0.5 ⇒ **偵測有效**（摘要「對抗偵測 ROC AUC 可達 83%」即此）。
- Alpaca-7b 的 OOD 最差（風格改寫差異太小）。
- 前 5 名設定**全為 DTMC**。

**語意層級指標 vs ROC AUC（表 VII 要點）**
- **PRE 與 ROC AUC 強正相關**（Pearson 0.81–1.00，Kendall 0.71–1.00），是唯一跨任務一致的指標 → 語意要精確才能有好效能。
- 四個程式任務（同一 CodeLlama）：六指標相關方向一致。
- Alpaca/Llama2 上 SL、NVT 等**正負混雜** → 需依任務分別挑指標。
- 論文討論節：PRE、ENT、NVT 與各面向效能較相關；SL、NDT 在對抗、幻覺上描述較佳。

**RQ3.2 – 抽象模型層級指標 vs ROC AUC（表 VIII 要點）**
- **PERP 與 ROC AUC 有較高正相關**（例如 Llama2-7B 的 OOD／對抗 Pearson 皆 0.90；僅 Alpaca-7b TruthfulQA 例外）→ 「**抽象模型越能模仿目標 LLM 的分布與轉移，偵測越好**」，可當**通用指標**。
- SUC、COV 正負混雜，故需多指標組合。

### 實務指引（討論節 E）
1. 先找與 ROC AUC 相關性**強且一致**的指標（幻覺情境：**PRE、ENT、SUC、PERP**）。
2. 依這些指標把 180 種設定**排名**，選排名最佳者。
3. 表 IX 顯示：四個程式任務皆選中最大 ROC AUC；TruthfulQA-Llama2 略低但在前 5 內。

---

## 11. 公式速查表

| # | 內容 | 公式 | 來源 |
|---|---|---|---|
| (1) | LLM 訓練目標 | $\max_\theta\sum_{t=1}^{T}\log p(w_t\mid w_{1:t-1};\theta)$ | 原文 |
| (2) | OOD 判定 | $g(x)=\text{in}$ if $P_X(x)\ge\lambda$；out if $P_X(x)<\lambda$ | 原文 |
| — | 分布偏移 | $p(y,x)=p(y\mid x)p(x)$ | 原文 |
| (3) | 自迴歸生成機率 | $p(y_1,\dots,y_T)=\prod_{t=1}^{T}p(y_t\mid y_{<t})$ | 原文 |
| — | Llama-7b 隱藏狀態維度 | $32\times4096=131{,}072$ | 原文 |
| — | 網格抽象狀態 | $\bar s=\{s_i\mid s^1_i\in c^1_\_\wedge\cdots\wedge s^k_i\in c^k_\_\}$；總數 $m^k$ | 原文 |
| — | 抽象轉移機率 | 具體轉移數 ÷ 總外向轉移數 | 原文（文字） |
| — | DTMC | $(\bar S,\bar s_0,\bar\delta,\bar P)$，$\bar P:\bar S\times\bar S\to[0,1]$ | 原文（定義 1） |
| — | HMM | $(\bar S,\bar\delta,\bar P,\bar O,\bar E,\bar I)$ | 原文（定義 2） |
| — | 語意 | $\theta\in\mathbb R^n$ 對 $\tau^k=\langle s_i,\dots,s_{i+k-1}\rangle$ | 原文（定義 3） |
| — | 抽象語意 | 抽象狀態內具體語意之平均 | 原文（文字） |
| — | OOD 轉移語意 | $\bar\theta(\langle\bar s_i,\bar s_{i+1}\rangle)$ | 原文 |
| — | 語意分數 | $\mathrm{AVG}(\bar\theta(\bar s_i))$ | 原文 |
| — | 超參數數量 | $27+81+18+54=180$ | 原文（表 III） |
| — | Surprise | 先驗與後驗 KL 散度之均值 | 原文（文字） |
| — | 相關性 | Pearson $r$、Kendall $\tau$ | 標準（論文使用） |

---

## 12. 名詞對照表（中英）

| 英文 | 中文 | 一句話 |
|---|---|---|
| LLM（Large Language Model） | 大型語言模型 | Transformer 為基礎、數十億參數 |
| DNN（Deep Neural Network） | 深度神經網路 | 有很多層的神經網路；LLM 也是 DNN 的一種 |
| RNN（Recurrent Neural Network） | 循環神經網路 | 逐步處理序列、前一步的輸出會回饋成下一步的輸入；Transformer 之前的主流（注意不是 recursive） |
| Self-attention | 自注意力 | 讓每個 token 與其他 token 互相加權 |
| Decoder-only | 僅解碼器 | 自迴歸預測下一 token |
| EOS（End Of Sequence） | 句尾詞元 | 標示生成結束 |
| Trustworthiness | 可信賴性 | 安全、強健、資安、真實… |
| OOD（Out-Of-Distribution） | 分布外 | 偏離訓練分布的資料 |
| Adversarial attack | 對抗攻擊 | 細微擾動騙過模型 |
| Hallucination | 幻覺 | 不忠實或不符事實的輸出 |
| Intrinsic / Extrinsic hallucination | 內在／外在幻覺 | 矛盾來源／無法驗證 |
| Model-based analysis | 以模型為基礎的分析 | 抽取代理模型來分析黑箱 |
| Abstract model | 抽象模型 | 由具體狀態壓縮成的 DTMC/HMM |
| Concrete state | 具體狀態 | decoder block 輸出（高維） |
| Abstract state | 抽象狀態 | 具體狀態經降維＋劃分後的離散狀態 |
| Trace | 軌跡 | 一次推論的狀態序列 |
| PCA（Principal Component Analysis） | 主成分分析 | 降維 |
| Grid / Cluster partition | 網格／分群劃分 | 兩種狀態劃分 |
| GMM（Gaussian Mixture Model）/ K-Means | 高斯混合／K 平均 | 兩種分群法 |
| Multi-step abstraction | 多步抽象 | N 步滑動視窗 |
| DTMC（Discrete-Time Markov Chain） | 離散時間馬可夫鏈 | 觀測到的狀態直接構成的機率轉移模型 |
| HMM（Hidden Markov Model） | 隱馬可夫模型 | 隱藏狀態＋發射機率 |
| Baum-Welch | Baum-Welch 演算法 | HMM 的 EM 擬合 |
| Emission function | 發射函數 | 隱藏狀態產生觀測的機率 |
| Semantics binding | 語意綁定 | 把可信賴性分數綁到狀態／轉移 |
| Semantics score | 語意分數 | 軌跡語意平均，用來判異常 |
| Sink state | 匯點狀態 | 只有機率 1 自迴圈的狀態 |
| Stationary distribution | 穩態分布 | $\pi P=\pi$ |
| Perplexity | 困惑度 | 對訓練分布擬合度 |
| KL divergence | KL 散度 | 兩分布差異 |
| ROC AUC（Receiver Operating Characteristic - Area Under the Curve） | ROC 曲線下面積 | 偵測效能（0.5＝隨機） |
| KDE（Kernel Density Estimation） | 核密度估計 | 畫分布曲線 |
| Mann-Whitney U | 曼-惠特尼 U 檢定 | 兩分布差異的非參數檢定 |
| Pearson / Kendall | 皮爾森／肯德爾相關 | 線性／秩相關 |
| pass@1 | 一次通過率 | 程式碼生成準確度 |
| SAPLMA（Statement Accuracy Prediction based on Language Model Activations） | 基於語言模型激活的陳述正確性預測 | 相關工作：用隱藏激活訓練真／假分類器 |
| ITI（Inference-Time Intervention） | 推論時介入 | 相關工作：推論時把激活往「真實方向」偏移 |
| DecodingTrust / AdvGLUE++ | — | 可信賴性基準／對抗資料集 |
| HELM（Holistic Evaluation of Language Models） | 語言模型整體評測 | 對 30 個 LM 在 42 情境、7 指標的綜合評測 |

---

## 附：論文的限制與坦誠之處（供批判閱讀）
- 僅適用**可取得內部隱藏狀態的開源 LLM**（白箱）。
- 實驗僅 3 個 LLM、規模 7B–13B；外部效度有限（作者自己列為 external threat）。
- 多數 ROC AUC 落在 0.55–0.75；只有少數最佳設定達 0.83–0.90，且**設定敏感、需調參**，實務上需 12 指標輔助選型。
- 三種狀態劃分方法在不同任務與面向上各有優劣（例如 KMeans 在 TruthfulQA、SST-2 表現好，在 AdvGLUE++ 不行），現有技術需要逐一挑方法、調參數才有效；DTMC 則表現接近或優於 HMM，是可行的候選。作者因此呼籲發展**更進階、LLM 專用的抽象模型建構技術**，不論任務與可信賴性面向都能刻畫 LLM 的行為。
- 論文未與 SAPLMA、ITI 做直接數值比較（僅定性討論）。
- 表 VII 的 ADV-Alpaca SL Kendall 為 −1.23（超出 [−1,1]），表 IX 與表 V/VI 部分數值亦不完全一致，疑為原文筆誤。
