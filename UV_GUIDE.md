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

## 2. 改善後のシステムを実行

Sentence-BERT、NLI、GiNZAなどは `improved` extra に分離しています。

```bash
uv sync --extra improved
```

評価:

```bash
uv run --extra improved python run.py examples/input.json --system improved --output improved_result.json
```

初回実行時は Hugging Face から Sentence-BERT / NLI のモデルがダウンロードされます。

## 3. ベースラインのシステムと改善後のシステムを比較

```bash
uv run --extra improved python compare.py examples/input.json --output comparison_result.json
```

## 4. 50件のデータセットを評価

ベースラインのシステムのみ:

```bash
uv run python evaluate_dataset.py examples/evaluation_dataset_50.jsonl \
  --systems baseline \
  --output baseline_evaluation.json
```

ベースラインのシステム / 改善後のシステムの比較:

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

改善後のシステム込み:

```bash
uv sync --locked --extra improved
```

## 6. Pythonバージョン

`.python-version` では Python 3.11 を指定しています。uvが管理可能なPython 3.11がない場合は、通常uvが必要なPythonを取得します。

```bash
uv python install 3.11
```

## 7. 依存関係を変更する場合

ベースラインのシステム側へ追加:

```bash
uv add <package>
```

改善後のシステム側へ追加:

```bash
uv add --optional improved <package>
```

変更後は `pyproject.toml` と `uv.lock` の両方をコミットします。
