# Bayes EMC V2 C++ Core

このディレクトリは、V2 Coreの最小骨格です。
現在の主系統のC++実装で、CLIの `bayes-emc run` はこのCoreを使います。

## 構成

- `include/bayes_emc/data.hpp`: 入出力データ。
- `include/bayes_emc/model_spec.hpp`: 複数モデル、基底数、パラメータ定義。
- `include/bayes_emc/prior.hpp`: 事前分布。
- `include/bayes_emc/parameter.hpp`: `model/basis/layer/parameter` の添字と値の保持。
- `include/bayes_emc/likelihood.hpp`: 尤度計算。
- `include/bayes_emc/emc_engine.hpp`: 最小EMCエンジン。
- `include/bayes_emc/free_energy.hpp`: ベイズ自由エネルギーによるガウスノイズ強度推定。
- `include/bayes_emc/sample_writer.hpp`: `sample.json` 出力。
- `examples/linear_1d/`: 1次元線形モデル。特定対象に寄せないための最小例。
- `examples/linear_1d_synthetic/`: 真値つき人工データを生成し、MAP推定を確認する例。
- `examples/spectral_basis/`: 基底を複数持つスペクトル分解例。
- `examples/linear_plus_spectral/`: 線形背景モデルとガウスピークモデルを同時に持つ複数モデル例。

## 考え方

V2では、解析対象を次の階層で扱います。

```text
model -> basis -> layer -> parameter
```

通常は `layer` を意識せず、`model -> basis -> parameter` として書けます。
スペクトル分解のピーク、混合モデルの成分、局所基底展開の基底関数などを `basis` として扱うことで、
事前分布と標的関数の実装を対応させやすくします。

## 最小検証

```bash
python -m bayes_emc init linear /tmp/linear
python -m bayes_emc run /tmp/linear/config.json
python -m bayes_emc plot /tmp/linear/result/sample.json

python -m bayes_emc init spectral /tmp/spectral
python -m bayes_emc run /tmp/spectral/config.json
python -m bayes_emc plot /tmp/spectral/result/sample.json
python -m bayes_emc select-peaks /tmp/spectral/config.json --min 1 --max 5

python -m bayes_emc init background-spectral /tmp/background_spectral
python -m bayes_emc run /tmp/background_spectral/config.json
python -m bayes_emc plot /tmp/background_spectral/result/sample.json
```

`init linear` は `intercept=1.25`, `slope=-0.80` の100点人工データを固定シードのガウスノイズ付きで生成します。
`run` は `config.json` から `generated_v2_config.hpp` を生成し、
`src/target.hpp` のモデル式を使ってV2 Coreを実行します。
データは `data.format` で `csv`, `tsv`, `whitespace` を選べます。
公開テンプレートは `x,y` ヘッダ付きCSVを使い、`input_columns` と `output_columns` で列を指定します。
`bayes-emc plot` はV2の列指向 `samples.values`、`samples.log_posterior`、
トップレベルのパラメータラベルを読み取り、
`matplotlib` がある場合はcorner plotまたはヒストグラムを、ない場合は簡易SVGを保存します。
MAPサンプルはどちらの場合も `posterior_max.json` に保存します。
`run` は全レプリカのエネルギー履歴からベイズ自由エネルギーを評価し、
`noise_estimation.txt` にノイズ強度候補と自由エネルギーを書き出します。
`model.noise.estimate_sigma2` が `true` なら候補からノイズ分散を選び、
`false` なら `model.noise.sigma2_min` を既知値として固定します。
ノイズ推定時の `sample.json` は、選ばれたノイズ分散に対応する温度層のサンプルを書き出します。
トップレベルの `posterior_replica_id`, `posterior_inverse_temperature`, `posterior_sigma2` で確認できます。
`log.txt` は実行要約に絞り、各レプリカのMH採択率と隣接レプリカ間の交換率は
`diagnostics.tsv` に書き出します。採択率はパラメータごとの列として保存されます。
採択率または交換率が10%未満、または99%超になった項目は `diagnostics_warnings.tsv` に書き出します。
`bayes-emc tune` では、最初の有限温度層の採択率を `C` の90%下限と99%上限、
隣接交換率を `gamma` と `replica_num`、最低温度層の採択率30%を `d` の調整目安として使います。
最後に、まだ10%未満の採択率が残る温度層・パラメータだけ `replica_step_scales` で提案幅を縮めます。
列名は `model_name[basis_id].layer_name.parameter_name` です。
`Inv Temp` の先頭に `*` が付く行は `data_size * beta < 1` の温度層です。
`beta=0` の最高温度レプリカは事前分布から直接サンプルし、採択率は100%として記録します。
本推論前に `bayes-emc tune config.json` を実行すると、短いEMC試行で
`C`, `gamma`, `d` と局所ステップ幅補正を調整した `config.tuned.json` を作れます。`replica_num=8` から始め、
`C` は最初の有限温度層の採択率が90%を超えるまで事前分布幅由来の値から調整します。
次に `gamma` を1の直上から上げ、`beta=0` との交換率が90%超、非最高温側の交換率が10%以上に
なる温度列を探します。必要なら `replica_num` を増やします。その後、最初の有限温度層の
採択率が99%以上なら `C` を大きくします。最後に `d` を0.5へ戻し、最低温度層のMH採択率が
30%程度になるように、低すぎる場合は `d` を大きく、高すぎる場合は `d` を小さくします。
それでも10%未満の箇所は、該当する温度層・パラメータの `replica_step_scales` だけを小さくします。
EMCの各温度レプリカのMH更新は `std::thread` で並列実行できます。
CLIテンプレートでは `emc.progress` が有効で、実行中は標準エラーに進捗バーを表示します。
`config.json` では `emc.parallel_workers: 0` が自動設定、`1` が逐次実行です。
レプリカ交換は同期点として逐次に行います。
大きなデータでは `emc.likelihood_workers` を増やすことで、1レプリカ内の尤度計算を
データ点方向にも並列化できます。CLIテンプレートの `TargetFunction` は出力バッファへ直接書き込むため、
尤度計算の内側で一時 `std::vector` を毎回作らない形です。
`spectral` も同じV2 Coreを使い、`config.json` のピーク数と事前分布、
`src/target.hpp` の基底ループだけで動きます。
`select-peaks` は同じスペクトル設定の `basis_count` を候補数に差し替えて複数回実行し、
`result/model_selection/peak_count/` に自由エネルギー比較を保存します。

`linear_plus_spectral` は、同じ解析内に `linear_background` と `spectral_peaks` の2モデルを置く例です。
線形背景とピーク群を別モデルとして持つことで、基底を持たない成分と基底を繰り返す成分を同じCoreで扱えます。
CLIテンプレートの `background-spectral` は、この考え方を公開入口から確認できる形にしたものです。
