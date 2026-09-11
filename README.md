# レポート評価システム: Baseline vs Improved

授業要件である以下を満たすための比較実験用プロジェクトです。

- ベースラインを用意する
- 言語処理技術を2つ以上含める
- システムの評価を行う
- 誤り分析を行う

## 1. システム構成

### Baseline

意味表現モデルを使わない単純な比較対象です。

- 冗長性
  - 文字 bigram / trigram の反復率
  - 文字 n-gram TF-IDF による文間類似度
- 論理整合性
  - 隣接文のTF-IDF類似度
  - 否定表現の有無が反転した場合を矛盾候補とするルール
- 要求網羅度
  - 必須・推奨キーワードの完全一致

### Improved

Baselineに意味的な言語処理を追加した提案手法です。

- 冗長性
  - n-gram反復率
  - Sentence-BERTによる文の意味類似度
- 論理整合性
  - 日本語NLIによる contradiction 確率
- 要求網羅度
  - 完全一致
  - Sentence-BERTによる意味的一致
- 補助解析
  - GiNZAによるNER
  - TF-IDFによる設問・模範解答との差分語候補

このため、提案システムには少なくとも以下の言語処理技術が含まれます。

1. TF-IDF
2. Sentence-BERT
3. Natural Language Inference (NLI)
4. Named Entity Recognition (NER)

## 2. ディレクトリ

```text
report_evaluator_comparison/
├── .python-version      # uvが利用するPython（3.11）
├── pyproject.toml       # uvの依存関係定義
├── UV_GUIDE.md          # uv実行手順
├── report_evaluator/
│   ├── baseline.py      # ベースライン
│   ├── improved.py      # 改善版
│   ├── common.py
│   └── schemas.py
├── examples/
│   ├── input.json
│   ├── evaluation_sample.jsonl
│   └── evaluation_dataset_50.jsonl
├── run.py               # 片方のシステムを実行
├── compare.py           # Baseline / Improved を同じ入力で比較
└── evaluate_dataset.py  # 定量評価 + 誤り分析
```

## 3. セットアップ（uv）

このプロジェクトは `uv` と `pyproject.toml` を使って依存関係を管理します。従来の `pip install -r ...` は不要です。詳しくは [`UV_GUIDE.md`](UV_GUIDE.md) を参照してください。

### Baselineだけ試す

```bash
uv sync
uv run python run.py examples/input.json --system baseline --output baseline_result.json
```

`uv run` は必要に応じて環境を同期するため、初回から次の1コマンドでも実行できます。

```bash
uv run python run.py examples/input.json --system baseline
```

### Improvedまで利用する

改善版の依存関係は `improved` extra として分離しています。

```bash
uv sync --extra improved
uv run --extra improved python run.py examples/input.json --system improved --output improved_result.json
```

Improvedの初回実行時にはSentence-BERTとNLIのモデルがHugging Faceからダウンロードされます。

## 4. 実行

### Baseline

```bash
uv run python run.py examples/input.json --system baseline --output baseline_result.json
```

### Improved

```bash
uv run --extra improved python run.py examples/input.json --system improved --output improved_result.json
```

### 両方を一度に比較

1件の入力を比較する場合:

```bash
uv run --extra improved python compare.py examples/input.json --output comparison_result.json
```

50件データセットを比較する場合:

```bash
uv run --extra improved python compare.py \
  examples/evaluation_dataset_50.pretty.json \
  --output comparison_result.json
```

データセット比較では、各ケースについて `Gold / Baseline / Improved` の総合点・冗長性・論理整合性・要求網羅度を1〜5点でターミナルに一覧表示します。`ΔB` と `ΔI` は、それぞれBaseline/Improvedの総合点から正解総合点を引いた誤差です。完全な検出結果や各軸の誤差は `comparison_result.json` に保存されます。

保存JSON全体も標準出力に表示したい場合は `--json-stdout` を付けます。

```bash
uv run --extra improved python compare.py \
  examples/evaluation_dataset_50.pretty.json \
  --output comparison_result.json \
  --json-stdout
```

## 5. 共通出力

両システムとも以下を返します。

```json
{
  "system": "baseline",
  "overall_score_100": 80.0,
  "overall_score_5": 4.2,
  "redundancy_score": 75.0,
  "consistency_score": 90.0,
  "coverage_score": 80.0,
  "highlights": [],
  "keywords": [],
  "missing_required_keywords": [],
  "named_entities": [],
  "diagnostics": {}
}
```

`highlights`には文字位置と`sentence_index`を含むため、UIで次の色分けができます。

- redundancy: 黄色
- contradiction: 赤色
- keyword: 緑色

## 6. システム評価

### 正解データ

`examples/evaluation_sample.jsonl`は評価スクリプト動作確認用のtoyデータです。研究・授業レポートで最終結果を報告するときは、このファイルを実際の人手アノテーションに置き換えてください。

1行につき1レポートです。

```json
{
  "case_id": "report001",
  "request": { "...": "通常の入力JSON" },
  "gold": {
    "redundancy_sentences": [2, 5],
    "contradiction_sentences": [7],
    "matched_required_keywords": ["過学習", "検証データ"],
    "redundancy_score_5": 3.0,
    "consistency_score_5": 4.0,
    "coverage_score_5": 4.0,
    "overall_score_5": 3.8
  }
}
```

文番号は0始まりです。

### Baselineを評価

```bash
uv run python evaluate_dataset.py examples/evaluation_sample.jsonl \
  --systems baseline \
  --output baseline_evaluation.json
```

### BaselineとImprovedを評価

```bash
uv run --extra improved python evaluate_dataset.py examples/evaluation_sample.jsonl \
  --systems baseline improved \
  --output evaluation_result.json
```

## 7. 評価指標

箇所検出には以下を出します。

- Precision
- Recall
- F1-score

対象:

- 冗長文
- 矛盾文
- 必須キーワード検出

1〜5点の人間評価との一致には以下を出します。

- MAE
- RMSE
- Spearman順位相関

これにより、例えば最終レポートでは次の形式で比較できます。

| System | Redundancy F1 | Contradiction F1 | Keyword F1 | Overall MAE |
|---|---:|---:|---:|---:|
| Baseline | 実験値 | 実験値 | 実験値 | 実験値 |
| Improved | 実験値 | 実験値 | 実験値 | 実験値 |

## 8. 誤り分析

`evaluate_dataset.py`は`error_analysis`として次を自動保存します。

- false_positive
- false_negative
- 1点以上ずれた総合点

例:

```json
{
  "case_id": "report012",
  "category": "redundancy",
  "error": "false_positive",
  "sentence_index": 4,
  "sentence": "..."
}
```

この出力を見ながら、誤りを例えば以下に分類できます。

- 意味は似ているが新情報を含む文を冗長と判定
- 異なる研究結果の紹介をNLIが文章の矛盾と判定
- 専門用語そのものを書かず説明だけしたケースをBaselineが未検出
- Sentence-BERTが関連概念をキーワードそのものと誤判定

最終レポートでは、単に件数を示すだけでなく、各種類について実例を2〜3件示し、原因と改善策を考察するとまとまりやすくなります。

## 9. 注意点

このコードの閾値は初期値です。実際の人手評価データを用いて閾値と重みを調整してください。またNLIの`contradiction`は文章全体の論理破綻を保証するものではないため、本システムでは「矛盾候補」として扱っています。

## 10. 日本語デスクトップGUI

TkinterによるGUIを追加しました。ローカルのデスクトップ上で実行します。

```bash
uv sync
uv run python gui.py
```

GUIに追加のPythonパッケージは不要ですが、**Tk対応のPythonとデスクトップ環境**が必要です。
`uv run python -m tkinter` で確認用ウィンドウが開くことを確認してください。
Tkが利用できない場合は、Windows / macOSではTkを含むPython公式インストーラー、
Ubuntu / Debianでは `python3-tk` を備えたシステムPythonなどを使用します。
uvが選ぶPythonにもTkが含まれる必要があります。必要に応じて、対応バージョン
（Python 3.11〜3.13）のTk対応Pythonを `uv venv --python /path/to/python` で指定してから同期してください。
SSHのみの環境や画面のないサーバーでは既存のCLIをご利用ください。

### 使い方

1. 「入力」にレポート本文と問題文を入力します。本文はUTF-8の `.txt` からも開けます。
2. 必須・推奨キーワードをカンマ、読点、改行で区切って入力します。空欄も可能です。同一語の重複は除き、必須を優先します。
3. 「評価する」を押すと、総合点と3項目の結果を表示します。
4. 各タブの本文で、冗長性は黄色、矛盾候補は赤い下線、キーワードは緑色で表示します。下段には理由と推敲時の確認事項が表示されます。
5. 入力タブで文章を修正し、再評価できます。結果画面は評価時の本文を保持します。
6. 「入力・結果をJSONで保存」で入力と評価結果を保存できます（`request` と `result` を含むGUI用形式）。

baselineは問題文自体を採点に使用しません。キーワード未指定の場合は網羅度を「未評価」と表示しますが、
既存エンジンの仕様により総合点には網羅度100点が含まれます。完全一致は既存処理の空白除去・casefold後の照合です。
本文中の表記が異なる場合、キーワード一覧で検出と表示されてもマーカーが付かないことがあります。
マーカーは既存の文類似度による指摘箇所を示し、bigram / trigram反復率は本文全体の指標として別途表示します。
改善案は推敲のための確認事項であり、文章の自動書き換え機能ではありません。

improvedを選択する場合は次のように起動してください。初回はモデル取得のため通信が必要です。
評価は別スレッドで実行し、処理中もウィンドウを操作できます。

```bash
uv sync --extra improved
uv run --extra improved python gui.py
```

### GUIからシステム性能を測定

「システム評価」で既存形式の正解データ（JSON / JSONL）を選ぶと、選択中の評価方式について
Precision・Recall・F1・Accuracyと、人手採点に対するSpearman順位相関・MAE・RMSEを表示します。
結果とケース別の誤り分析はJSONで保存できます。`examples/evaluation_sample.jsonl` で動作を試せます。
正解データ形式は本READMEの「正解データ」を参照してください。

Accuracyは `(TP + TN) / (TP + TN + FP + FN)` です。冗長文・矛盾文では全文、必須キーワードでは
指定された全必須キーワードを母集団としてマイクロ集計します。評価対象が0件なら算出不可です。
CLIの `evaluate_dataset.py` にも `accuracy` と `tn` を追加しています。
Spearmanは2件未満または採点が一定の場合は算出不可です。正解データなしに性能指標は算出しません。

### テスト

```bash
uv run python -m unittest discover -s tests -v
```

テストは入力整形、空入力、Accuracyの真陰性、短文、文字位置、既存データの評価表示を確認します。
