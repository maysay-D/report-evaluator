# 合成評価データセット（50件）

このデータセットは `evaluate_dataset.py` でそのまま利用できる形式です。

## 内容

- 50件
- 5課題テーマ × 10パターン
- 各テーマに以下を収録
  1. 模範例（必須語を明示）
  2. 模範例（概念的な言い換えを含む）
  3. 完全一致に近い冗長例
  4. 意味的な言い換えによる冗長例
  5. 直接的な論理矛盾
  6. 離れた文との論理矛盾
  7. 必須概念の欠落
  8. 必須語を使わない意味的網羅
  9. 複合エラー
  10. 境界例（キーワードの見かけ上の一致、または異なる研究結果の比較）

## 課題テーマ

- 教師あり学習と過学習
- 自然言語処理の文書表現
- 分類器の評価指標
- 関係データベースの正規化
- 情報検索システム

## 主な正解ラベル

`gold` 内の以下の項目は、既存の `evaluate_dataset.py` が直接使用します。

- `redundancy_sentences`
- `contradiction_sentences`
- `matched_required_keywords`
- `redundancy_score_5`
- `consistency_score_5`
- `coverage_score_5`
- `overall_score_5`

さらに誤り分析用として以下も付与しています。

- `sentences`
- `redundant_sentence_pairs`
- `contradiction_sentence_pairs`
- `keyword_annotations`
  - `exact`: 専門語が明示され、内容も扱っている
  - `semantic`: 専門語そのものはないが概念を説明している
  - `missing`: 言及がない
  - `incidental_only`: 文字列はあるが概念を説明していない
- `semantic_required_keywords`
- `missing_required_keywords`
- `incidental_only_keywords`

## 実行例

ベースラインだけ:

```bash
uv run python evaluate_dataset.py examples/evaluation_dataset_50.jsonl \
  --systems baseline \
  --output baseline_result.json
```

ベースラインと改善版を比較:

```bash
uv run --extra improved python evaluate_dataset.py examples/evaluation_dataset_50.jsonl \
  --systems baseline improved \
  --output comparison_result.json
```

## 注意

これは人間が実際に採点した実データではなく、システムの単体テスト・比較実験・誤り分析を始めるための**合成データ**です。
最終的な研究評価では、このデータだけで性能を主張せず、可能であれば実際の学生レポートを匿名化した上で複数人にアノテーションして評価してください。

特に境界例は重要です。

- キーワードが文字列として存在しても、課題要求を満たしているとは限らない
- NLI上で矛盾しそうな二文でも、異なる研究結果を比較しているだけなら文章自体は論理破綻ではない

というFalse Positiveを確認できます。
