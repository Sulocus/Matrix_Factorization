# Showcase Results

このディレクトリには、`dev` ブランチで実行した局所実験から選んだ軽量な図だけを置いている。
大きな tensor payload や checkpoint は含めない。

## ディレクトリ

- `01_warm_start`: warm start の初期 overlap を変えた比較。
- `02_fixed_onsager_vs_no_onsager`: fixed Onsager と no Onsager の比較。
- `03_warm_start_onsager`: 初期 overlap `0.2` の warm start Onsager。

## 基本 overlap

ここでの基本的な overlap は、teacher 方向への absolute projection である。
正規化には teacher の二乗ノルムを使う。

$$
Q_Y =
\frac{|\langle Y_s,Y_t\rangle|}{\langle Y_t,Y_t\rangle},
\qquad
Q_W =
\frac{|\langle W_s,W_t\rangle|}{\langle W_t,W_t\rangle}.
$$

この値は `1` で clip しない。したがって student の scale が teacher と異なる場合、
`1` を超える値も数値エラーではなく scale 情報として残る。

## Sign Gauge と Scale Gauge

この行列分解には、潜在次元に対応する $M$ 次元の gauge vector
$\boldsymbol{k} = (k_1,\ldots,k_M)$ がある。

$$
W_{:m} \mapsto k_m W_{:m},
\qquad
X_{m:} \mapsto k_m^{-1} X_{m:}.
$$

sign 補正では、各 $k_m$ の大きさは使わず、符号だけを見る。
W 側では、各 latent channel の teacher/student 内積の符号を合わせることに相当する。

$$
Q_W^{\mathrm{sign}} =
\frac{\sum_{m=1}^M |\langle W_{s,:m}, W_{t,:m}\rangle|}
{\sum_{m=1}^M \|W_{t,:m}\|^2}.
$$

scale-gauge 補正では、各 channel ごとに連続値の scalar $g_m$ を fitting する。

$$
g_m^\star =
\arg\min_{g\ne 0}
\left(
\|g W_{s,:m}-W_{t,:m}\|^2
+ \|g^{-1} X_{s,m:}-X_{t,m:}\|^2
\right).
$$

ここで
$a_m=\|W_{s,:m}\|^2$,
$b_m=\langle W_{s,:m},W_{t,:m}\rangle$,
$c_m=\|X_{s,m:}\|^2$,
$d_m=\langle X_{s,m:},X_{t,m:}\rangle$
とおくと、最小化する $g$ 依存部分は

$$
a_m g^2 - 2b_m g + \frac{c_m}{g^2} - \frac{2d_m}{g}.
$$

実装では、次の四次方程式の実数かつ非零の根を候補にする。

$$
a_m g^4 - b_m g^3 + d_m g - c_m = 0
$$

さらに fallback candidate として $\pm\sqrt{c_m/a_m}$ と $\pm 1$ も加え、
目的関数が最小になる候補を選ぶ。この処理は $|g_m|$ を `1` に近づける制約ではない。
表示される gauge magnitude は median $\left|\log |g_m|\right|$ であり、
scale ずれの大きさを表す補助量である。

alignment 後の projection は

$$
Q_W^{\mathrm{gauge}} =
\frac{\sum_m g_m^\star \langle W_{s,:m},W_{t,:m}\rangle}
{\sum_m \|W_{t,:m}\|^2},
\qquad
Q_X^{\mathrm{gauge}} =
\frac{\sum_m (g_m^\star)^{-1} \langle X_{s,m:},X_{t,m:}\rangle}
{\sum_m \|X_{t,m:}\|^2}.
$$

`qwqx_gauge` の図では

$$
\frac{1}{2}(Q_W^{\mathrm{gauge}}+Q_X^{\mathrm{gauge}})
$$

を表示している。

これらの補正表示は、training 初期の物理的正しさを保証するものではない。
後段で生じる symmetry breaking や teacher/student 間の gauge mismatch を取り除いて、
同じ解を異なる gauge で見ている可能性を確認するための表示である。
