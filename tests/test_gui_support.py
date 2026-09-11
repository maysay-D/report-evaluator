import unittest
from pathlib import Path

from evaluate_dataset import Counts, evaluate_system, safe_spearman
from report_evaluator import BaselineEvaluator
from report_evaluator.dataset_io import load_dataset_records
from report_evaluator.gui_helpers import format_metrics, make_request, parse_keywords


class GuiSupportTests(unittest.TestCase):
    def test_keywords_and_request(self):
        self.assertEqual(parse_keywords(' 必須、推奨，必須\n別語, '), ['必須', '推奨', '別語'])
        req = make_request('  本文😀\n', '問題', 'A,A', 'A,B')
        self.assertEqual(req.report.text, '  本文😀\n')
        self.assertEqual(req.assignment.required_keywords, ['A'])
        self.assertEqual(req.assignment.recommended_keywords, ['B'])

    def test_blank_input(self):
        for text, prompt in [(' \n', '問題'), ('本文', ' ' )]:
            with self.assertRaises(ValueError):
                make_request(text, prompt, '', '')

    def test_accuracy_includes_true_negatives(self):
        counts = Counts()
        counts.add({0, 1}, {0, 2}, set(range(5)))
        self.assertEqual(counts.metrics()['accuracy'], 0.6)
        self.assertEqual(counts.metrics()['tn'], 2)
        self.assertEqual(counts.metrics()['f1'], 0.5)
        self.assertIsNone(Counts().metrics()['accuracy'])
        with self.assertRaises(ValueError):
            counts.add(set(), {8}, {0})

    def test_small_report(self):
        result = BaselineEvaluator().evaluate(make_request('あ\nい', '問題', '', ''))
        self.assertEqual(result.consistency_score, 100)

    def test_highlights_refer_to_original_text(self):
        text = '😀 太陽光発電は有効である。\n太陽光発電は有効ではない。\n太陽光発電は有効ではない。'
        result = BaselineEvaluator().evaluate(make_request(text, '説明せよ', '太陽光発電', '蓄電池'))
        self.assertEqual({h.type for h in result.highlights}, {'redundancy', 'contradiction', 'keyword'})
        for h in result.highlights:
            self.assertEqual(text[h.start:h.end], h.text)
        self.assertFalse(result.keywords[1].matched)

    def test_dataset_formatting_and_spearman(self):
        records = load_dataset_records(Path(__file__).resolve().parents[1] / 'examples/evaluation_sample.jsonl')
        result = evaluate_system('baseline', records)
        rendered = format_metrics(result)
        for name in ['Precision', 'Recall', 'F1', 'Accuracy', 'Spearman']:
            self.assertIn(name, rendered)
        self.assertEqual(result['n_cases'], 3)
        self.assertIsNone(safe_spearman([1, 1], [2, 3]))
        self.assertEqual(safe_spearman([1, 2, 3], [3, 2, 1]), -1)


if __name__ == '__main__':
    unittest.main()
