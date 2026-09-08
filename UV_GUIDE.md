# uv での実行方法

このプロジェクトは `pyproject.toml` を依存関係の定義元として使います。`pip install -r ...` は不要です。

## 1. ベースラインだけ実行

初回は次のどちらかで環境を作れます。

```bash
uv sync
```

その後、1件を評価します。

```bash
uv run python run.py examples/input.json --system baseline --output baseline_result.json
```

`uv run` は必要に応じて環境同期も行うため、最初から次の1コマンドでも構いません。

```bash
uv run python run.py examples/input.json --system baseline
```

## 2. 改善版を実行

Sentence-BERT、NLI、GiNZAなどは `improved` extra に分離しています。

```bash
uv sync --extra improved
```

評価:

```bash
uv run --extra improved python run.py examples/input.json --system improved --output improved_result.json
```

初回実行時は Hugging Face から Sentence-BERT / NLI のモデルがダウンロードされます。

## 3. Baseline と Improved を比較

```bash
uv run --extra improved python compare.py examples/input.json --output comparison_result.json
```

## 4. 50件データセットを評価

Baselineのみ:

```bash
uv run python evaluate_dataset.py examples/evaluation_dataset_50.jsonl \
  --systems baseline \
  --output baseline_evaluation.json
```

Baseline / Improvedの比較:

```bash
uv run --extra improved python evaluate_dataset.py examples/evaluation_dataset_50.jsonl \
  --systems baseline improved \
  --output evaluation_result.json
```

## 5. lockファイル

初回の `uv sync` / `uv run` で依存関係が解決されると `uv.lock` が生成されます。以後はそのlockファイルをGitに含めてください。

再現性を優先してlockを更新せず同期したい場合:

```bash
uv sync --locked
```

Improved込み:

```bash
uv sync --locked --extra improved
```

## 6. Pythonバージョン

`.python-version` では Python 3.11 を指定しています。uvが管理可能なPython 3.11がない場合は、通常uvが必要なPythonを取得します。

```bash
uv python install 3.11
```

## 7. 依存関係を変更する場合

Baseline側へ追加:

```bash
uv add <package>
```

Improved側へ追加:

```bash
uv add --optional improved <package>
```

変更後は `pyproject.toml` と `uv.lock` の両方をコミットします。


## データセット比較

`compare.py` は次の3形式を自動判別します。

- 1件分の入力JSON（`examples/input.json`）
- JSON配列（`examples/evaluation_dataset_50.pretty.json`）
- JSONL（`examples/evaluation_dataset_50.jsonl`）

したがって、50件をまとめてBaseline / Improvedで比較する場合は次のどちらでも実行できます。

```bash
uv run --extra improved python compare.py \
  examples/evaluation_dataset_50.pretty.json \
  --output comparison_result.json
```

または

```bash
uv run --extra improved python compare.py \
  examples/evaluation_dataset_50.jsonl \
  --output comparison_result.json
```

`evaluate_dataset.py` もJSON配列とJSONLの両方を受け付けます。

### 各ケースの採点結果表示

データセットを `compare.py` に渡すと、各ケースについて次の情報を1行ずつ表示します。

- Gold: 正解ラベルの総合点 / 冗長性 / 論理整合性 / 要求網羅度
- BASE: Baselineの予測点
- IMPR: Improvedの予測点
- ΔB: Baseline総合点 - Gold総合点
- ΔI: Improved総合点 - Gold総合点

すべて1〜5点尺度です。表示例:

```text
CASE           PATTERN                     GOLD O/R/C/K           BASE O/R/C/K       ΔB  IMPR O/R/C/K       ΔI
synthetic_001  exemplary_exact             5.00/5.00/5.00/5.00    4.88/...          -0.12 4.95/...          -0.05
```

詳細は `comparison_result.json` の `cases` にケース単位で保存されます。標準出力にもJSON全体を出したい場合は `--json-stdout` を指定してください。
