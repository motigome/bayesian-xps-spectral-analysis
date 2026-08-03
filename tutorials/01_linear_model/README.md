# 線形モデルのベイズ推論チュートリアル

`y = ax + b` を使い、最小二乗法、交換モンテカルロ法の温度列による雑音分散選択、事後分布、事後予測を順に確認します。人工データの条件は原稿の4.2節に合わせています。

## 実行方法

リポジトリ直下でチュートリアル用依存関係を導入します。C++20対応コンパイラも必要です。

```bash
python -m pip install -e ".[tutorial]"
```

percent形式のソースは、そのままPythonスクリプトとして実行できます。

```bash
python tutorials/01_linear_model/linear_model_bayesian_inference.py
```

Jupyterで読む場合は、Jupytextで同じディレクトリ内にNotebookを生成します。

```bash
jupytext --to ipynb tutorials/01_linear_model/linear_model_bayesian_inference.py
jupyter lab tutorials/01_linear_model/linear_model_bayesian_inference.ipynb
```

短い動作確認には次を使います。FAST結果はサンプル数が少ないため、最終的な数値や図には通常モードを使ってください。

```bash
BAYES_XPS_FAST=1 python tutorials/01_linear_model/linear_model_bayesian_inference.py
```

## ファイルと生成物

- `linear_model_bayesian_inference.py`: 日本語解説つきJupytext Notebook source
- `config.json`: 一様事前分布、EMC温度列、ガウス雑音設定
- `data/data.csv`: 固定seedで生成した人工データ（Notebookから同じ内容を再生成可能）
- `src/target.hpp`: 平均関数 `a*x+b`
- `src/main.cpp`: C++ Coreの実行入口
- `result/`, `config.runtime.json`, `config.run.json`: 実行時にこのディレクトリ内へ生成

既存の `config.tuned.json` は再利用されます。モデル、事前分布、温度列を変えた場合は、このファイルを削除して再チューニングしてください。
