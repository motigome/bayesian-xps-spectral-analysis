# Bayesian XPS Spectral Analysis Tutorials

XPSスペクトル解析を題材に、ベイズ推論とベイズ自由エネルギーを手元で再現するための日本語チュートリアルです。
技術書「第3章第3節」の数値実験に対応し、同じ記号・人工データ条件を使います。

本リポジトリには、次の2つのNotebookがあります。

1. [`y=ax+b` のベイズ推論](tutorials/01_linear_model/linear_model_bayesian_inference.ipynb)
   - 最小二乗法とベイズ推論の違い
   - 交換モンテカルロ法（EMC）
   - EMC温度列から求める自由エネルギー $F(\sigma^2)$
   - 雑音分散 $\sigma^{2*}$ の選択
   - $a,b$ の信用区間、相関、事後予測
2. [XPSスペクトル分解](tutorials/02_xps_spectral_decomposition/xps_spectral_decomposition.ipynb)
   - 3本のガウスピークからなる人工XPSスペクトル
   - $F(K)$ によるピーク数選択
   - $A_k,\mu_k,w_k$ の事後分布
   - 各ピークと合成スペクトルの不確かさ

[![Open linear tutorial in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/motigome/bayesian-xps-spectral-analysis/blob/main/tutorials/01_linear_model/linear_model_bayesian_inference.ipynb)
[![Open XPS tutorial in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/motigome/bayesian-xps-spectral-analysis/blob/main/tutorials/02_xps_spectral_decomposition/xps_spectral_decomposition.ipynb)

## 対象読者

- XPSのピーク位置、幅、強度、背景処理には馴染みがある
- ベイズ推論、事前分布、事後分布、周辺尤度はこれから学ぶ
- Pythonの基本的なコードを上から順に実行できる

数式は使いますが、各節を「考え方 → 数式 → 実行セル → 図の読み方」の順にしています。

## ローカルでの実行

C++20対応コンパイラとPython 3.10以上が必要です。macOSではApple Clang、Linuxでは最近のGCC/Clangを利用できます。

```bash
git clone https://github.com/motigome/bayesian-xps-spectral-analysis.git
cd bayesian-xps-spectral-analysis
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[tutorial]"
jupyter lab
```

JupyterLabが開いたら、`tutorials/01_linear_model` から順にNotebookを実行してください。

Notebookは通常設定ではEMCとモデル比較を実行するため、計算に時間がかかります。動作だけを短く確認する場合は、
JupyterLabを起動する前に次を設定します。

```bash
export BAYES_XPS_FAST=1
jupyter lab
```

高速設定は配線確認用です。自由エネルギーや信用区間を報告するときは通常設定を使い、seed、サンプル数、
温度列を変えた再実行も行ってください。

## Google Colabでの実行

上のColabバッジからNotebookを開き、メニューの「ランタイム」から「すべてのセルを実行」を選びます。
先頭セルがColab環境を検出すると、このリポジトリと描画依存関係を自動で取得します。

## 書籍条件との対応

線形モデルは $N=50$, $x\in[-1,1]$, $a=0.8$, $b=0.1$, $\sigma^2=0.01$、
$a,b\sim\mathrm{Uniform}(-1,1)$ を使います。

XPS例は背景なし、$K_{\mathrm{true}}=3$, $N=300$, $E\in[0,3]$, $\sigma^2=0.01$ とし、

- $A=(0.6,1.5,1.2)$
- $\mu=(1.2,1.45,1.7)$
- $w=(0.10,0.08,0.07)$

を使います。ピーク形状は

$$
\phi(E\mid\mu,w)=\exp\left[-\frac{(E-\mu)^2}{2w^2}\right]
$$

です。ここで $A$ はピーク面積ではなくピーク高さです。面積は $A w\sqrt{2\pi}$ になります。

原稿ではXPSパラメータの事前分布を区間一様分布としていますが、具体的な上下限は定めていません。
本Notebookで採用した再現用範囲を表で明示し、範囲を変える感度解析を演習に含めています。

## 交換モンテカルロ法

推論Coreは、逆温度の異なる複数レプリカを交換しながらサンプリングします。
温度経路には熱力学積分の恒等式があります。同梱Coreは単純な台形則ではなく、隣接温度間の
正規化定数比を両方向の指数再重み付けで推定して累積し、周辺尤度とベイズ自由エネルギーを評価します。

- 線形モデル: $F(\sigma^2)$ を比較し、$\sigma^{2*}$ を選ぶ
- XPSモデル: 固定した既知雑音のもとで $F(K)$ を比較し、$K^*$ を選ぶ

`diagnostics.tsv` にはパラメータ更新の採択率と隣接レプリカの交換率が保存されます。
自由エネルギーだけでなく、診断、複数seed、サンプル数依存性も確認してください。

## ディレクトリ構成

```text
bayes_emc/                         Python CLI
cpp/include/bayes_emc/             C++20 EMC Core
tutorials/01_linear_model/         線形モデルの教材
tutorials/02_xps_spectral_decomposition/
                                    XPSスペクトル分解の教材
tests/                              Core/CLIテスト
```

Notebookの編集元は同名のJupytext `py:percent` ファイルです。Notebookを再生成するときは次を使います。

```bash
jupytext --to ipynb tutorials/01_linear_model/linear_model_bayesian_inference.py
jupytext --to ipynb tutorials/02_xps_spectral_decomposition/xps_spectral_decomposition.py
```

## 実測XPSへ進む前に

この教材は教育用の人工データです。実測XPSでは、少なくとも次を別途検討してください。

- Shirley/Tougaardなどの背景
- Voigtまたはpseudo-Voigtピーク
- doublet、面積比、ピーク間隔などの物理制約
- エネルギー校正、装置関数、前処理履歴
- Poisson性を含む観測雑音
- 事前分布とモデル候補に対する感度解析

モデルを増やすほど自動的に正しくなるわけではありません。生成モデル、事前分布、診断結果を解析条件として記録してください。
