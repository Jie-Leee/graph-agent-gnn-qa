"""Behavioral checks for provenance, failed runs and group-level inference."""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graph_agent.evaluation.reporting import (
    ReportError, build_report, paired_group_bootstrap, summarize, validate_records,
)


def record(qid="q1", group="g1", variant="llm", predicted=("a",), **changes):
    value = {
        "question_id": qid, "group_id": group,
        "mode": "gnn" if variant.startswith("gat_") else variant,
        "variant": variant, "split": "validation", "execution_backend": "live",
        "experiment_fingerprint": "fixed-config",
        "gold_answer_ids": ["a"], "predicted_answer_ids": list(predicted),
        "answer_text": "a" if predicted else "", "status": "ok", "error_type": None,
        "logical_input_tokens": 20, "logical_output_tokens": 3,
        "actual_input_tokens": 20, "actual_output_tokens": 3,
        "logical_elapsed_seconds": .2, "actual_elapsed_seconds": .2,
        "usage_reported": True, "served_models": ["fixed-model"], "cited_evidence": [],
    }
    value.update(changes)
    return value


class ReportingTests(unittest.TestCase):
    def test_errors_stay_in_question_and_group_denominators(self):
        rows, _ = validate_records([
            record("q1", "g1"),
            record("q2", "g1", status="error", error_type="timeout", f1=1, exact_match=1),
        ])
        row = summarize(rows)[0]
        self.assertEqual(row["questions"], 2)
        self.assertEqual(row["groups"], 1)
        self.assertEqual(row["f1"], .5)
        self.assertEqual(row["group_f1"], .5)
        self.assertEqual(row["error_count"], 1)
        self.assertEqual(row["answer_rate"], .5)

    def test_missing_tokens_never_turn_into_zero(self):
        rows, _ = validate_records([
            record("q1"), record("q2", "g2", actual_input_tokens=None,
                                 logical_output_tokens=None, usage_reported=False),
        ])
        row = summarize(rows)[0]
        self.assertIsNone(row["actual_input_tokens_total"])
        self.assertEqual(row["actual_input_tokens_known_total"], 20)
        self.assertEqual(row["actual_input_tokens_reported_count"], 1)
        self.assertIsNone(row["logical_output_tokens_total"])
        self.assertEqual(row["usage_reported_count"], 1)

    def test_rejects_backend_split_or_fingerprint_mix(self):
        for changes in (
            {"execution_backend": "mock"},
            {"split": "test"},
            {"experiment_fingerprint": "different-config"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ReportError):
                validate_records([record(), record("q2", "g2", **changes)], allow_mock=True)

    def test_mock_requires_explicit_opt_in(self):
        with self.assertRaises(ReportError):
            validate_records([record(execution_backend="mock")])
        rows, meta = validate_records([record(execution_backend="mock")], allow_mock=True)
        self.assertEqual(meta["execution_backend"], "mock")
        self.assertEqual(summarize(rows)[0]["coverage_status"], "mock_only")

    def test_mismatch_duplicate_or_missing_group_rejected(self):
        invalid = [
            [record(), record()],
            [record(mode="gnn")],
            [record(group_id="")],
            [record(experiment_fingerprint="")],
            [record(), record(variant="text", gold_answer_ids=["b"])],
        ]
        for rows in invalid:
            with self.subTest(rows=rows), self.assertRaises(ReportError):
                validate_records(rows)

    def test_manifest_must_match(self):
        with self.assertRaises(ReportError):
            validate_records([record()], {"experiment_fingerprint": "other"})
        with self.assertRaises(ReportError):
            validate_records([record()], {"split": "test"})

    def test_paired_bootstrap_uses_groups_and_keeps_failed_predictions(self):
        source = []
        for qid, group, good_left in (
            ("q1", "g1", False), ("q2", "g1", False),
            ("q3", "g2", True), ("q4", "g2", True),
        ):
            source += [
                record(qid, group, "llm", predicted=("a",) if good_left else ("b",)),
                record(qid, group, "text", predicted=("b",) if good_left else ("a",)),
            ]
        source.append(record("left_only", "g3", "llm"))
        rows, _ = validate_records(source)
        one = paired_group_bootstrap(rows, "llm", "text", samples=200, seed=7)
        two = paired_group_bootstrap(rows, "llm", "text", samples=200, seed=7)
        self.assertEqual(one, two)
        self.assertEqual(one["paired_questions"], 4)
        self.assertEqual(one["paired_groups"], 2)
        self.assertEqual(one["reference_only_questions"], 1)
        self.assertEqual(one["candidate_only_questions"], 0)
        self.assertEqual(one["difference"], 0)
        self.assertEqual(one["ci_low"], -1)
        self.assertEqual(one["ci_high"], 1)
        failed = [
            record("q1", "g1", "llm"),
            record("q1", "g1", "text", status="error"),
            record("q2", "g2", "llm"),
            record("q2", "g2", "text", status="error"),
        ]
        rows, _ = validate_records(failed)
        pair = paired_group_bootstrap(rows, "llm", "text", samples=50)
        self.assertEqual(pair["difference"], -1)
        self.assertEqual(pair["paired_questions"], 2)

    def test_insufficient_groups_does_not_claim_an_interval(self):
        rows, _ = validate_records([record(), record(variant="text")])
        pair = paired_group_bootstrap(rows, "llm", "text", samples=30)
        self.assertIsNone(pair["ci_low"])
        self.assertEqual(pair["status"], "insufficient_groups")

    def test_no_citation_is_not_hallucination_and_empty_is_distinct(self):
        rows, _ = validate_records([
            record("correct"),
            record("empty", "g2", predicted=()),
            record("mapping", "g3", predicted=(), answer_text="unmapped name",
                   mapping_failed=True),
            record("unclear", "g4", predicted=(), answer_text="cannot answer"),
        ])
        by_id = {row["question_id"]: row for row in rows}
        self.assertEqual(by_id["correct"]["graph_unsupported_answer_fraction"], 0)
        self.assertIsNone(by_id["empty"]["graph_unsupported_answer_fraction"])
        summary = summarize(rows)[0]
        self.assertEqual(summary["mapping_failure_count"], 1)
        self.assertEqual(summary["empty_ids_with_text_count"], 1)
        self.assertEqual(summary["empty_prediction_count"], 3)

    def test_partial_report_has_no_full_test_claim_and_errors_are_written(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "predictions.jsonl"
            data = [
                record("q1", "g1"),
                record("q2", "g2", status="error", error_type="timeout"),
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in data), encoding="utf-8")
            (root / "manifest.json").write_text(json.dumps({
                "experiment_fingerprint": "fixed-config", "execution_backend": "live",
                "split": "validation", "status": "smoke",
                "dataset_question_count": 200, "selected_question_count": 2,
            }), encoding="utf-8")
            result = build_report(path, root / "report", plots=False, bootstrap_samples=20)
            report = (root / "report" / "E实验评测报告.md").read_text(encoding="utf-8")
            self.assertIn("smoke_or_partial", report)
            self.assertIn("missing", report)
            self.assertEqual(result["record_count"], 2)
            errors = (root / "report" / "error_cases.jsonl").read_text(encoding="utf-8")
            self.assertIn("timeout", errors)
            self.assertTrue((root / "report" / "main_results.csv").is_file())

    def test_mock_report_is_visibly_marked(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "predictions.jsonl"
            path.write_text(json.dumps(record(execution_backend="mock")) + "\n", encoding="utf-8")
            result = build_report(path, root / "report", allow_mock=True,
                                  plots=False, bootstrap_samples=20)
            report = (root / "report" / "E实验评测报告.md").read_text(encoding="utf-8")
            self.assertIn("MOCK", report)
            self.assertIn("不能支持任何正式性能结论", report)
            self.assertEqual(result["execution_backend"], "mock")

    def test_missing_plot_dependency_keeps_tables_and_surfaces_error(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "predictions.jsonl"
            path.write_text(json.dumps(record()) + "\n", encoding="utf-8")
            with patch("graph_agent.evaluation.reporting._write_plots",
                       side_effect=ReportError("matplotlib missing")):
                result = build_report(path, root / "report", plots=True, bootstrap_samples=20)
            self.assertIn("matplotlib", result["plot_error"])
            self.assertTrue((root / "report" / "variant_summary.csv").exists())
            report = (root / "report" / "E实验评测报告.md").read_text(encoding="utf-8")
            self.assertIn("图表未生成", report)


class RunnerContractReportingTests(unittest.TestCase):
    def _report(self, rows, manifest):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        path = root / "predictions.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        result = build_report(path, root / "report", plots=False, bootstrap_samples=40)
        with (root / "report" / "variant_summary.csv").open(encoding="utf-8-sig", newline="") as handle:
            summaries = {row["variant"]: row for row in csv.DictReader(handle)}
        return root, result, summaries

    def test_actual_runner_resolution_fields_are_counted_separately(self):
        values = [
            record("q1", "g1", predicted=(), answer_text="unknown name",
                   unresolved_answer_names=["unknown name"], model_attempted_answer=True),
            record("q2", "g2", predicted=(), answer_text="Springfield",
                   ambiguous_answer_names={"Springfield": ["m1", "m2"]},
                   model_attempted_answer=True),
            record("q3", "g3", predicted=(), answer_text="bad id",
                   rejected_answer_ids=["/m/not-in-catalog"], model_attempted_answer=True),
            record("q4", "g4", unresolved_answer_names=["extra unknown"],
                   model_attempted_answer=True),
        ]
        rows, _ = validate_records(values)
        summary = summarize(rows)[0]
        self.assertEqual(summary["unresolved_name_question_count"], 2)
        self.assertEqual(summary["ambiguous_name_question_count"], 1)
        self.assertEqual(summary["rejected_id_question_count"], 1)
        self.assertEqual(summary["normalization_unresolved_count"], 3)
        self.assertEqual(summary["mapping_failure_count"], 4)
        self.assertEqual(summary["empty_ids_with_text_count"], 0)
        self.assertEqual(summary["model_attempted_answer_count"], 4)
        self.assertEqual(summary["f1"], .25)

    def test_error_cases_preserve_text_resolution_and_trace_contract(self):
        row = record(
            predicted=(), answer_text="Example alias", question_text="Who directed the film?",
            topic_entity_ids=["/m/topic"], unresolved_answer_names=["Example alias"],
            ambiguous_answer_names={"Name": ["/m/1", "/m/2"]},
            rejected_answer_ids=["/m/bad"], model_attempted_answer=True,
            trace_file="traces/example.json",
        )
        root, _, _ = self._report([row], {"experiment_fingerprint": "fixed-config"})
        case = json.loads((root / "report" / "error_cases.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(case["category"], "normalization_unresolved")
        self.assertEqual(case["unresolved_answer_names"], ["Example alias"])
        self.assertEqual(case["ambiguous_answer_names"]["Name"], ["/m/1", "/m/2"])
        self.assertEqual(case["rejected_answer_ids"], ["/m/bad"])
        self.assertEqual(case["trace_file"], "traces/example.json")
        self.assertEqual(case["topic_entity_ids"], ["/m/topic"])
        md = (root / "report" / "error_cases.md").read_text(encoding="utf-8")
        self.assertIn("Who directed the film?", md)
        self.assertIn("traces/example.json", md)
        self.assertEqual(case["f1"], 0)

    def test_manifest_pending_scores_zero_in_question_and_group_denominators(self):
        rows = [record("q1", "g1"), record("q1", "g1", variant="text")]
        manifest = {
            "experiment_fingerprint": "fixed-config", "split": "validation",
            "execution_backend": "live", "status": "partial",
            "dataset_question_count": 200, "selected_question_count": 2,
            "selected_question_ids": ["q1", "q2"],
            "selected_question_groups": {"q1": "g1", "q2": "g1"},
            "variants": ["llm", "text"],
        }
        root, result, summaries = self._report(rows, manifest)
        self.assertEqual(result["record_count"], 2)
        self.assertEqual(result["pending_record_count"], 2)
        for summary in summaries.values():
            self.assertEqual(summary["questions"], "2")
            self.assertEqual(summary["observed_questions"], "1")
            self.assertEqual(summary["pending_count"], "1")
            self.assertEqual(float(summary["f1"]), .5)
            self.assertEqual(float(summary["group_f1"]), .5)
            self.assertEqual(summary["error_count"], "0")
            self.assertEqual(summary["logical_input_tokens_total"], "")
        cases = [json.loads(line) for line in
                 (root / "report" / "error_cases.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(cases), 2)
        self.assertTrue(all(c["category"] == "pending" and c["gold_answer_ids"] is None for c in cases))
        pair = json.loads((root / "report" / "paired_f1_comparisons.json").read_text())[0]
        self.assertEqual(pair["paired_questions"], 2)
        self.assertEqual(pair["paired_groups"], 1)

    def test_selected_count_without_ids_still_keeps_pending_in_denominator(self):
        rows = [record(), record(variant="text")]
        manifest = {"experiment_fingerprint": "fixed-config", "selected_question_count": 3,
                    "dataset_question_count": 200, "status": "partial"}
        root, result, summaries = self._report(rows, manifest)
        self.assertEqual(result["pending_without_ids"], 4)
        for summary in summaries.values():
            self.assertAlmostEqual(float(summary["f1"]), 1 / 3)
            self.assertEqual(summary["group_f1"], "")
            self.assertEqual(summary["pending_count"], "2")
        pair = json.loads((root / "report" / "paired_f1_comparisons.json").read_text())[0]
        self.assertIsNone(pair["difference"])
        self.assertIsNone(pair["ci_low"])
        self.assertEqual(pair["status"], "pending_question_identity_missing")

    def test_pending_without_group_mapping_does_not_fabricate_group_interval(self):
        manifest = {
            "experiment_fingerprint": "fixed-config", "selected_question_count": 2,
            "identity": {"selected_question_ids": ["q1", "q2"], "variants": ["llm", "text"]},
        }
        root, _, summaries = self._report([record(), record(variant="text")], manifest)
        self.assertTrue(all(s["group_f1"] == "" for s in summaries.values()))
        pair = json.loads((root / "report" / "paired_f1_comparisons.json").read_text())[0]
        self.assertEqual(pair["status"], "pending_group_mapping_missing")
        self.assertIsNone(pair["ci_low"])

    def test_manifest_question_identity_is_enforced(self):
        with self.assertRaises(ReportError):
            self._report([record()], {"selected_question_ids": ["other"]})
        with self.assertRaises(ReportError):
            self._report([record()], {"selected_question_count": 2, "selected_question_ids": ["q1"]})
        with self.assertRaises(ReportError):
            self._report([record()], {"selected_question_ids": ["q1"],
                                      "selected_question_groups": {"q1": "wrong-group"}})


if __name__ == "__main__":
    unittest.main()
