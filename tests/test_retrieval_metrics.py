"""Regression checks for missing labels, honest denominators, and CLI failures."""
from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from graph_agent.retrieval import GraphIndex, RetrievalCase, benchmark_retrieval
from graph_agent.schemas import Question, Triple

spec = importlib.util.spec_from_file_location("evaluate_retrieval_cli", ROOT / "scripts/evaluate_retrieval.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class RetrievalMetricTests(unittest.TestCase):
    def setUp(self):
        self.edge = Triple("a", "r", "b")
        self.index = GraphIndex([self.edge])
        self.q = Question("q", "test", ("a",))

    def test_empty_dataset_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty QA"):
            benchmark_retrieval(self.index, iter(()))

    def test_unlabeled_questions_have_null_recall_and_skipped_counts(self):
        for gold in (None, ()):
            with self.subTest(gold=gold):
                result = benchmark_retrieval(self.index, [RetrievalCase(self.q, gold, gold)])
                for metric in ("answer", "evidence"):
                    self.assertIsNone(result[f"{metric}_recall_macro"])
                    self.assertIsNone(result[f"{metric}_recall_micro"])
                    self.assertEqual(result[f"{metric}_evaluated_questions"], 0)
                    self.assertEqual(result[f"{metric}_skipped_questions"], 1)
                self.assertEqual(result["questions"], 1)
                self.assertGreaterEqual(result["latency_mean_ms"], 0)
                self.assertIn("null", json.dumps(result, allow_nan=False))

    def test_missing_evidence_does_not_raise_macro_recall(self):
        missed = Triple("x", "r", "y")
        result = benchmark_retrieval(self.index, [
            RetrievalCase(self.q, ("b",), (missed,)), RetrievalCase(self.q, ("b",)),
        ])
        self.assertEqual(result["evidence_recall_macro"], 0)
        self.assertEqual(result["evidence_recall_micro"], 0)
        self.assertEqual(result["evidence_evaluated_questions"], 1)
        self.assertEqual(result["evidence_skipped_questions"], 1)
        self.assertEqual(result["answer_evaluated_questions"], 2)
        self.assertEqual(result["answer_recall_macro"], 1)

    def test_answer_and_evidence_denominators_are_independent(self):
        result = benchmark_retrieval(self.index, [
            RetrievalCase(self.q, ("b",), None),
            RetrievalCase(self.q, None, (self.edge,)),
            RetrievalCase(self.q, ("b", "missing"), (Triple("x", "r", "y"),)),
        ])
        self.assertAlmostEqual(result["answer_recall_macro"], .75)
        self.assertAlmostEqual(result["answer_recall_micro"], 2 / 3)
        self.assertEqual(result["evidence_recall_macro"], .5)
        self.assertEqual(result["evidence_recall_micro"], .5)
        self.assertEqual(result["answer_evaluated_questions"], 2)
        self.assertEqual(result["evidence_evaluated_questions"], 2)

    def test_gold_is_read_after_retrieval(self):
        called = []
        original = self.index.retrieve_with_stats
        def retrieve(question, **kwargs):
            self.assertIs(question, self.q)
            called.append(True)
            return original(question, **kwargs)
        self.index.retrieve_with_stats = retrieve
        parent = self
        class GuardedCase:
            question = self.q
            @property
            def gold_answer_ids(self):
                parent.assertTrue(called)
                return ("b",)
            @property
            def gold_evidence(self):
                parent.assertTrue(called)
                return (parent.edge,)
        self.assertEqual(benchmark_retrieval(self.index, [GuardedCase()])["answer_recall_macro"], 1)


class RetrievalEvaluationInputTests(unittest.TestCase):
    def row(self):
        return {"question_id": "q", "text": "test", "topic_entity_ids": ["a"], "answer_ids": ["b"]}

    def read(self, row):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qa.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            return tuple(cli._read_cases(path))

    def test_required_answer_ids_reject_missing_null_empty_and_blank(self):
        rows = []
        for value in (None, [], [""], ["  "], "b", [1]):
            row = self.row()
            row["answer_ids"] = value
            rows.append(row)
        row = self.row()
        del row["answer_ids"]
        rows.append(row)
        for row in rows:
            with self.subTest(row=row), self.assertRaisesRegex(ValueError, "answer_ids"):
                self.read(row)

    def test_optional_evidence_is_skipped_when_missing_null_or_empty(self):
        for value in ("missing", None, []):
            row = self.row()
            if value != "missing":
                row["evidence"] = value
            case = self.read(row)[0]
            result = benchmark_retrieval(GraphIndex([Triple("a", "r", "b")]), [case])
            self.assertIsNone(result["evidence_recall_macro"])
            self.assertEqual(result["evidence_skipped_questions"], 1)

    def test_invalid_evidence_is_rejected(self):
        for value in ("bad", [["a", "r"]], [["a", "", "b"]], [["a", "r", 3]]):
            row = self.row()
            row["evidence"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "evidence"):
                self.read(row)

    def test_blank_question_or_topic_is_rejected(self):
        for field, value in (("question_id", " "), ("text", ""), ("topic_entity_ids", [""])):
            row = self.row()
            row[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.read(row)

    def run_cli(self, raw, output_exists=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "g.tsv").write_text("a\tr\tb\n", encoding="utf-8")
            (root / "qa.jsonl").write_text(raw, encoding="utf-8")
            report = root / "nested/report.json"
            if output_exists:
                report.parent.mkdir()
                report.write_text("existing report", encoding="utf-8")
            result = subprocess.run([
                sys.executable, str(ROOT / "scripts/evaluate_retrieval.py"), "--triples", str(root / "g.tsv"),
                "--qa", str(root / "qa.jsonl"), "--output", str(report),
            ], capture_output=True, text=True)
            return result, report.read_text(encoding="utf-8") if report.exists() else None

    def test_cli_refuses_empty_dataset_and_writes_no_report(self):
        result, report = self.run_cli("\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("empty QA", result.stderr)
        self.assertIsNone(report)

    def test_bad_later_row_preserves_existing_report(self):
        raw = json.dumps(self.row()) + '\n{"question_id":"bad","text":"bad"}\n'
        result, report = self.run_cli(raw, output_exists=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("QA line 2", result.stderr)
        self.assertEqual(report, "existing report")

    def test_cli_creates_parent_and_serializes_undefined_recall_as_null(self):
        result, report = self.run_cli(json.dumps(self.row()) + "\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        metrics = json.loads(report)
        self.assertEqual(json.loads(result.stdout), metrics)
        self.assertIsNone(metrics["evidence_recall_macro"])
        self.assertEqual(metrics["answer_recall_macro"], 1)


if __name__ == "__main__":
    unittest.main()
