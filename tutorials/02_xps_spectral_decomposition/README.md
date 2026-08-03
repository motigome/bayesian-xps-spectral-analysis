# XPSスペクトル分解チュートリアル

背景なしの3本のガウスピークを使い、交換モンテカルロ法による事後推論と、ベイズ自由エネルギー
$F(K)$ によるピーク数選択を順に再現します。人工データの条件は原稿の4.3節に合わせています。

## 実行方法

リポジトリ直下で依存関係を導入し、実行済みNotebookを開きます。C++20対応コンパイラも必要です。

```bash
python -m pip install -e ".[tutorial]"
jupyter lab tutorials/02_xps_spectral_decomposition/xps_spectral_decomposition.ipynb
```

短い動作確認には次を使います。FAST結果はモデル比較の報告には使わず、通常設定で再実行してください。

```bash
BAYES_XPS_FAST=1 python tutorials/02_xps_spectral_decomposition/xps_spectral_decomposition.py
```

## 主なファイル

- `xps_spectral_decomposition.ipynb`: 通常設定で実行済みの日本語チュートリアル
- `xps_spectral_decomposition.py`: Jupytext `py:percent` 形式の編集元
- `config.json`: $K=3$ の基本設定
- `config.tuned.json`: 提案幅と温度列のチューニング済み設定
- `config.model_selection.json`: $K=1,\ldots,6$ の通常設定（実行時は無視対象のruntime版を生成）
- `data/synthetic_xps.csv`: 固定seedで生成した人工データ（Notebookから同じ内容を再生成可能）
- `src/target.hpp`: $A_k,\mu_k,w_k$ からガウスピーク和を返す生成モデル

実行時の `result/`, `config.runtime.json` はこのディレクトリ内に生成され、Git管理から除外されます。

## 解釈上の注意

この例は、既知の雑音分散、背景なし、ガウスピークという教育用モデルです。通常設定では $K=3$ が
自由エネルギー最小になりますが、$K=4$ との差は小さいため、複数seed、計算量、事前範囲への感度も確認してください。
実測XPSでは背景、Voigt系形状、doublet制約、装置関数、雑音モデルを別途検討する必要があります。
