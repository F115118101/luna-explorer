# 附錄左側公式解釋（luna_v3_appendix.html，第 01–07 單元）

照左側公式欄的順序：每個單元先寫「輸入 → 運算 → 輸出」，再一條一條講公式分頁。
頁面上公式裡的數字是玩具資料，這裡只寫一般式。

---

## 01 LLM：自迴歸

**輸入 → 輸出**：提示 $x$（文字）→ 回答 $y=(y_1,\dots,y_T)$

### 自迴歸分解

$$
P(y\mid x)=\prod_{t=1}^{T}P(y_t\mid x,\,y_{<t}),\qquad y_{<t}=(y_1,\dots,y_{t-1})
$$

- 整句回答的機率 = 每一步「在題目和前面已生成的 token 條件下，選到 $y_t$」的機率連乘。
- 模型只要學會右邊這個條件機率，就能一步一步寫出整句。

### 取對數

$$
\log P(y\mid x)=\sum_{t=1}^{T}\log P(y_t\mid x,\,y_{<t})
$$

- 很多個小於 1 的數連乘會小到電腦存不下，所以改成對數相加。
- 訓練時最小化的就是它的負數（07 的交叉熵）。

---

## 02 分詞：BPE

**輸入 → 輸出**：文字 → $\mathbf x=(x_1,\dots,x_n)$，$x_t\in\{0,\dots,31999\}$

### BPE 合併規則

$$
\mathrm{count}(a,b)=\sum_{w\in\text{語料}} f(w)\cdot\#\{\,i:\ w_i=a,\ w_{i+1}=b\,\}
$$

$$
(a^{*},b^{*})=\arg\max_{(a,b)}\ \mathrm{count}(a,b)
$$

- $f(w)$：詞 $w$ 在語料中出現幾次；$w_i$：這個詞目前切出來的第 $i$ 個片段。
- 找出相鄰出現次數最多的一對 $(a^*,b^*)$，把它們合併成一個新片段 $a^*b^*$。

### 詞彙表成長

$$
V\leftarrow V\cup\{a^{*}b^{*}\},\qquad |V|\leftarrow|V|+1
$$

- 每合併一次，詞彙表多一個 token；重複到 $|V|=32{,}000$（Llama-2）為止。
- 推論時不再數次數，只照學到的合併順序套用。

---

## 03 嵌入：查表與訓練

**輸入 → 輸出**：$x_t$ → $\mathbf e_t\in\mathbb R^{4096}$；整句疊成 $H^{(0)}\in\mathbb R^{n\times4096}$

頁面上的玩具模型：只用目前這個詞預測下一個詞，嵌入 2 維、11 個詞。

### 查表

$$
\mathbf e_t=E[x_t,:]^{\top}=E^{\top}\,\mathrm{onehot}(x_t)
$$

- $E$：詞彙數 × 維度。取第 $x_t$ 列再轉置成直的向量。
- 等價於拿 one-hot 向量（只有第 $x_t$ 格是 1）去乘 $E^\top$。

### 交叉熵損失

$$
z_j=W_j\cdot\mathbf e_t+b_j,\qquad
p_j=\frac{e^{z_j}}{\sum_{k}e^{z_k}},\qquad
\mathcal L=-\log p_{y}=-\log\frac{e^{z_{y}}}{\sum_{k}e^{z_k}}
$$

- $W_j$：輸出矩陣 $W$ 的第 $j$ 列；$b_j$：偏差；$z_j$：第 $j$ 個詞的分數。
- $y$：正確的下一個詞。它的機率越高，損失越小。

### 梯度更新（只看 $\mathbf e_t$）

$$
\frac{\partial\mathcal L}{\partial\mathbf e_t}=W^{\top}(\mathbf p-\mathbf y),\qquad
\mathbf e_t\leftarrow\mathbf e_t-\eta\,W^{\top}(\mathbf p-\mathbf y)
$$

- $\mathbf y$：正確答案的 one-hot 向量；$\mathbf p-\mathbf y$：預測和答案差多少。
- $\eta$：學習率（頁面上的滑桿）。
- $E$ 裡只有這一步用到的那一列會被更新。

### 全部參數更新

$$
\mathbf g=\mathbf p-\mathbf y,\qquad
\nabla_{\mathbf e}\mathcal L=W^{\top}\mathbf g,\qquad
\nabla_{W}\mathcal L=\mathbf g\,\mathbf e_t^{\top},\qquad
\nabla_{\mathbf b}\mathcal L=\mathbf g
$$

$$
E[x_t,:]\leftarrow E[x_t,:]-\eta\,(W^{\top}\mathbf g)^{\top},\qquad
W\leftarrow W-\eta\,\mathbf g\,\mathbf e_t^{\top},\qquad
\mathbf b\leftarrow\mathbf b-\eta\,\mathbf g
$$

- 嵌入、輸出矩陣、偏差三者一起更新。
- 後面常接一樣的詞，收到的梯度就相似，所以會被推到一起。

---

## 04 位置：RoPE

**輸入 → 輸出**：$Q=H^{(0)}W_Q,\ K=H^{(0)}W_K$ → 旋轉後的 $\widetilde Q,\widetilde K\in\mathbb R^{n\times128}$（每個頭）

### 旋轉

$$
R(m\theta_i)=\begin{pmatrix}\cos m\theta_i & -\sin m\theta_i\\ \sin m\theta_i & \cos m\theta_i\end{pmatrix},\qquad
\theta_i=10000^{-2i/128},\quad i=0,\dots,63
$$

- 128 維切成 64 組，每組 2 維；位置 $m$ 的那組旋轉 $m\theta_i$ 弧度。
- $i$ 越大，$\theta_i$ 越小，轉得越慢：前面的組分辨近距離，後面的組分辨遠距離。

### 只看相對距離

$$
\big(R(m\theta)\mathbf q\big)^{\top}\big(R(n\theta)\mathbf k\big)
=\mathbf q^{\top}R(m\theta)^{\top}R(n\theta)\,\mathbf k
=\mathbf q^{\top}R(-m\theta)\,R(n\theta)\,\mathbf k
=\mathbf q^{\top}R\big((n-m)\theta\big)\,\mathbf k
$$

- 旋轉矩陣的轉置 = 反方向旋轉；兩個旋轉相乘 = 角度相加。
- 所以內積只跟 $n-m$ 有關，跟 $m$、$n$ 本身無關（「兩個一起 +1」內積不變）。

---

## 05 自注意力

**輸入 → 輸出**：$\widetilde Q,\widetilde K,\ V=H^{(0)}W_V$ → $\Delta A\in\mathbb R^{n\times4096}$

### 縮放點積注意力

$$
\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\Big(\frac{QK^{\top}}{\sqrt{d_k}}+M\Big)V,\qquad \sqrt{d_k}=\sqrt{128}\approx11.31
$$

- $QK^\top$：每個 token 的 $\mathbf q$ 和每個 token 的 $\mathbf k$ 的內積，得到 $n\times n$ 分數表。
- 除以 $\sqrt{d_k}$ 讓分數不要太大；$M$ 是遮罩；softmax 逐列做。

### 選中這一列（單一元素）

$$
s_{ij}=\tilde{\mathbf q}_i^{\top}\tilde{\mathbf k}_j,\qquad
\alpha_{ij}=\frac{\exp\big(s_{ij}/\sqrt{d_k}+M_{ij}\big)}{\sum_{j'=1}^{n}\exp\big(s_{ij'}/\sqrt{d_k}+M_{ij'}\big)}
$$

- $\alpha_{ij}$：第 $i$ 個 token 看第 $j$ 個 token 的權重，一列加起來 = 1。
- 熱圖的一列就是一個 $i$ 的 $(\alpha_{i1},\dots,\alpha_{in})$。

### 因果遮罩

$$
M_{ij}=\begin{cases}0, & j\le i\\ -\infty, & j>i\end{cases}
\qquad\Rightarrow\qquad e^{\,s/\sqrt{d_k}-\infty}=0
$$

- 未來位置的分數變 $-\infty$，softmax 後權重剛好是 0，看不到未來。

### 投影權重與多頭輸出

$$
\mathbf q_t^{(r)}=W_Q^{(r)}\mathbf h_t,\quad
\mathbf k_t^{(r)}=W_K^{(r)}\mathbf h_t,\quad
\mathbf v_t^{(r)}=W_V^{(r)}\mathbf h_t,\qquad
W_Q^{(r)},W_K^{(r)},W_V^{(r)}\in\mathbb R^{128\times4096}
$$

$$
\mathbf o_t^{(r)}=\sum_{j=1}^{n}\alpha_{tj}^{(r)}\mathbf v_j^{(r)}\in\mathbb R^{128},\qquad
\Delta\mathbf a_t=W_O\,[\mathbf o_t^{(1)};\dots;\mathbf o_t^{(32)}],\qquad W_O\in\mathbb R^{4096\times4096}
$$

- $r=1,\dots,32$：32 個頭各自算一套；串接 32 × 128 = 4,096 維，再乘 $W_O$。
- 這是 token 之間唯一交換資訊的地方。

---

## 06 解碼器區塊

**輸入 → 輸出**：$\mathbf h^{(l-1)}$、$\Delta\mathbf a$ → $\mathbf h^{(l)}$；32 層後得到 $H^{(32)}\in\mathbb R^{n\times4096}$

### RMSNorm

$$
\mathrm{RMSNorm}(\mathbf h)=\frac{\mathbf h}{\sqrt{\dfrac1d\displaystyle\sum_{i=1}^{d}h_i^2+\epsilon}}\odot\mathbf g
$$

- 分母是各座標的均方根，除掉後向量大小穩定；$\epsilon$ 防止除以 0（Llama-2 是 $10^{-5}$）。
- $\mathbf g$：學出來的增益，兩個子層各有一組。
- 比 LayerNorm 少了「減平均」。

### 殘差連接

$$
\mathbf a=\mathbf h^{(l-1)}+\mathrm{Attn}\big(\mathrm{RMSNorm}_1(\mathbf h^{(l-1)})\big),\qquad
\mathbf h^{(l)}=\mathbf a+\mathrm{FFN}\big(\mathrm{RMSNorm}_2(\mathbf a)\big)
$$

- 每個子層都是「正規化 → 計算 → 加回原本的向量」。
- 合起來就是主頁的 $\mathbf h^{(l)}=\mathbf h^{(l-1)}+\Delta\mathbf a+\Delta\mathbf m$。

### SwiGLU 前饋層

$$
\mathrm{FFN}(\mathbf x)=W_2\big(\mathrm{SiLU}(W_1\mathbf x)\odot W_3\mathbf x\big),\qquad
\mathrm{SiLU}(u)=u\,\sigma(u)=\frac{u}{1+e^{-u}}
$$

$$
W_1,W_3\in\mathbb R^{11008\times4096},\qquad W_2\in\mathbb R^{4096\times11008}
$$

- 放大到 11,008 維 → SiLU 那一路當閘門逐元素乘另一路 → 縮回 4,096 維。
- 一次只處理一個 token。

---

## 07 輸出層、取樣與預訓練

**輸入 → 輸出**：$\mathbf h^{(32)}_t$ → $\mathbf p_t\in[0,1]^{32000}$

### 輸出層 + softmax（含溫度）

$$
\widehat{\mathbf h}=\mathrm{RMSNorm}_{\text{final}}(\mathbf h^{(32)}),\qquad
\mathbf z=W_{\text{out}}\,\widehat{\mathbf h}\in\mathbb R^{32000},\qquad
p_i(T)=\frac{e^{z_i/T}}{\sum_{j=1}^{32000}e^{z_j/T}}
$$

- $z_i$：第 $i$ 個 token 的分數（logit）。
- 溫度 $T$：$T$ 小 → 分布尖，幾乎只選最高的；$T$ 大 → 分布平。$T=1$ 就是一般 softmax。

### 預訓練損失

$$
\mathcal L_{\text{train}}=-\frac1N\sum_{t=1}^{N}\log p_\theta(x_{t+1}\mid x_{\le t};\,T=1)
$$

- 拿真實文字，每個位置取「正確下一個 token」機率的 $-\log$，對約 2 兆個 token 取平均。
- 頁面上的 $\mathcal L_{\text{demo}}(T)=-\log p(\text{目標})$ 只算一個位置，會跟著溫度滑桿變。

### 困惑度

$$
\mathrm{PPL}=\exp\!\Big(-\frac1N\sum_{t=1}^{N}\log p(y_t\mid\cdot)\Big)
$$

- 平均損失取 exp。PPL = 5 的意思是：模型平均每一步像在 5 個選項裡猜。

### top-p（核取樣）

$$
p_{(1)}\ge p_{(2)}\ge\dots\ge p_{(32000)},\qquad
k^{*}=\min\Big\{k:\ \sum_{i=1}^{k}p_{(i)}\ge P\Big\},\qquad
S_P=\{(1),\dots,(k^{*})\}
$$

$$
p'_i=\frac{p_i}{\sum_{j\in S_P}p_j}\quad(i\in S_P),\qquad y_{t+1}\sim\mathbf p'
$$

- 機率由大到小排，留下累積到 $P$ 的最短前段，重新歸一化後再抽。

### 全模型參數與更新

$$
\Theta_{\text{global}}=\{E,\ \mathbf g_{\text{final}},\ W_{\text{out}}\},\qquad
\Theta^{(l)}_{\text{attn}}=\{W_Q^{(l)},W_K^{(l)},W_V^{(l)},W_O^{(l)}\},\qquad
\Theta^{(l)}_{\text{FFN}}=\{W_1^{(l)},W_2^{(l)},W_3^{(l)}\},\qquad
\Theta^{(l)}_{\text{norm}}=\{\mathbf g_1^{(l)},\mathbf g_2^{(l)}\},\qquad l=1,\dots,32
$$

AdamW 更新（第 $k$ 步，$\mathbf G=\nabla_\Theta\mathcal L_{\text{train}}$）：

$$
\mathbf m\leftarrow\beta_1\mathbf m+(1-\beta_1)\mathbf G,\qquad
\mathbf v\leftarrow\beta_2\mathbf v+(1-\beta_2)\,\mathbf G\odot\mathbf G
$$

$$
\hat{\mathbf m}=\frac{\mathbf m}{1-\beta_1^{k}},\qquad
\hat{\mathbf v}=\frac{\mathbf v}{1-\beta_2^{k}},\qquad
\Theta\leftarrow\Theta-\eta\Big(\frac{\hat{\mathbf m}}{\sqrt{\hat{\mathbf v}}+\epsilon}+\lambda\,\Theta\Big)
$$

- 所有參數一起用反向傳播的梯度更新；Llama-2 的線性層沒有偏差項。
- $\mathbf m,\mathbf v$：梯度的移動平均與平方的移動平均；$\lambda$：權重衰減。
