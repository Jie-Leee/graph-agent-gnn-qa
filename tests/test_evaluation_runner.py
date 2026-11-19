"""E's evaluator contracts: mock-only; no network, torch, or real credentials."""
from __future__ import annotations
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from threading import Barrier

from graph_agent.evaluation.runner import EvaluationOptions, run_evaluation
from graph_agent.schemas import Prediction, Question


class FakeEngine:
    requested_model = "unit-test-model"
    last_trace = None
    def __init__(self, *, fail=False, unknown=False, interrupt_on=None):
        self.fail, self.unknown, self.interrupt_on = fail, unknown, interrupt_on
        self.inputs = []
    def identity(self):
        return {"engine": "unit-test", "requested_model": self.requested_model}
    def preflight(self):
        pass
    def run(self, question, variant):
        assert type(question) is Question
        assert not hasattr(question, "answer_ids")
        assert not hasattr(question, "relation_path")
        assert not hasattr(question, "evidence")
        self.inputs.append((question.question_id, variant))
        if question.question_id == self.interrupt_on:
            raise KeyboardInterrupt()
        hit = variant not in ("llm", "text")
        self.last_trace = {"run_id": question.question_id + "-" + variant, "usage_reported": not self.unknown,
            "actual_input_tokens": 1 if hit else 11, "actual_output_tokens": 2,
            "actual_elapsed_seconds": 0.1, "route_cache_hit": hit,
            "retrieved_node_ids": ["answer"], "context_candidate_ids": ["answer"],
            "shared_route_events": [{"stage": "route", "plan": "same"}],
            "events": [{"served_model": "unit-served", "usage_reported": not self.unknown}]}
        if self.fail and question.question_id == "q2":
            raise RuntimeError("SECRET_SHOULD_NOT_BE_WRITTEN")
        return Prediction(question.question_id, ("answer",), "answer", tool_calls=1,
                          elapsed_seconds=0.2, input_tokens=11, output_tokens=2)


class EvaluationRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "data").mkdir()
        examples = [{"question_id": "q" + str(i), "text": "question " + str(i),
            "topic_entity_ids": ["topic"], "answer_ids": ["answer"],
            "relation_path": ["r1", "r2"], "evidence": [["topic", "r1", "middle"], ["middle", "r2", "answer"]]}
            for i in (1, 2)]
        (self.root / "data/validation.jsonl").write_text("".join(json.dumps(r) + "\n" for r in examples), encoding="utf-8")
        (self.root / "data/validation.questions.jsonl").write_text("".join(json.dumps({k: r[k] for k in
            ("question_id", "text", "topic_entity_ids")}) + "\n" for r in examples), encoding="utf-8")
        (self.root / "config.json").write_text(json.dumps({"seed": 42}), encoding="utf-8")
        self.options = EvaluationOptions(data=self.root / "data", config=self.root / "config.json",
            output=self.root / "result", backend="mock", limit=2, variants=("llm", "text", "gat_2_mean"), repo_root=self.root)
    def tearDown(self):
        self.temp.cleanup()
    def rows(self):
        return [json.loads(line) for line in (self.options.output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    def test_gold_isolation_and_group_results(self):
        engine = FakeEngine()
        summary = run_evaluation(self.options, engine=engine)
        self.assertEqual(len(engine.inputs), 6)
        self.assertEqual(summary["variants"]["llm"]["exact_match"], 1)
        self.assertEqual(summary["variants"]["llm"]["query_group_count"], 1)
        self.assertEqual(summary["variants"]["llm"]["group_macro_f1"], 1)
        self.assertEqual(len({r["group_id"] for r in self.rows()}), 1)
    def test_failure_remains_in_denominator_and_errors_are_safe(self):
        summary = run_evaluation(self.options, engine=FakeEngine(fail=True))
        self.assertEqual(summary["variants"]["llm"]["failure_count"], 1)
        self.assertEqual(summary["variants"]["llm"]["f1"], 0.5)
        self.assertEqual(len(self.rows()), 6)
        self.assertNotIn("SECRET_SHOULD_NOT_BE_WRITTEN", (self.options.output / "predictions.jsonl").read_text())
    def test_unknown_usage_is_null(self):
        summary = run_evaluation(self.options, engine=FakeEngine(unknown=True))
        self.assertIsNone(self.rows()[0]["logical_input_tokens"])
        self.assertIsNone(summary["variants"]["llm"]["actual_input_tokens"])
    def test_known_current_usage_survives_unknown_cached_route(self):
        class CachedUnknownRoute(FakeEngine):
            def run(self, question, variant):
                result = super().run(question, variant)
                self.last_trace.update(usage_reported=False, events=[
                    {"stage": "answer", "usage_reported": True, "served_model": "unit-served"}])
                return result
        options = replace(self.options, variants=("gat_2_mean",))
        summary = run_evaluation(options, engine=CachedUnknownRoute())
        row = self.rows()[0]
        self.assertIsNone(row["logical_input_tokens"])
        self.assertEqual(row["actual_input_tokens"], 1)
        self.assertFalse(row["usage_reported"])
        self.assertTrue(row["actual_usage_reported"])
        self.assertEqual(summary["variants"]["gat_2_mean"]["actual_input_tokens"], 2)

    def test_logical_and_actual_costs_are_not_added(self):
        summary = run_evaluation(self.options, engine=FakeEngine())
        gnn = summary["variants"]["gat_2_mean"]
        self.assertEqual(gnn["logical_input_tokens"], 22)
        self.assertEqual(gnn["actual_input_tokens"], 2)
        self.assertTrue(all(r["route_cache_hit"] for r in self.rows() if r["mode"] == "gnn"))
    def test_mock_is_marked_and_not_formal(self):
        summary = run_evaluation(self.options)
        self.assertFalse(summary["formal_results"])
        self.assertEqual(summary["execution_backend"], "mock")
        self.assertTrue(all(r["execution_backend"] == "mock" for r in self.rows()))
        self.assertIsNone(self.rows()[0]["actual_input_tokens"])
    def test_parallel_questions_keep_routes_and_persist_whole_batches(self):
        barrier = Barrier(2)

        class ParallelEngine(FakeEngine):
            def new_worker(self):
                return ParallelEngine()

            def run(self, question, variant):
                if variant == "llm":
                    barrier.wait(timeout=3)
                    self.question_id = question.question_id
                else:
                    assert self.question_id == question.question_id
                result = super().run(question, variant)
                self.last_trace["shared_route_events"] = [{"stage": "route", "qid": self.question_id}]
                return result

        engine = ParallelEngine()
        summary = run_evaluation(replace(self.options, workers=2), engine=engine)
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["variants"]["llm"]["exact_match"], 1)
        rows = self.rows()
        self.assertEqual(len(rows), 6)
        for index in (0, 3):
            batch = rows[index:index + 3]
            self.assertEqual(len({row["question_id"] for row in batch}), 1)
            self.assertEqual([row["variant"] for row in batch], list(self.options.variants))
            self.assertTrue(all(row["shared_route_consistent"] for row in batch))
        self.assertNotEqual(rows[1]["route_fingerprint"], rows[4]["route_fingerprint"])

        resumed = ParallelEngine()
        run_evaluation(replace(self.options, resume=True), engine=resumed)
        self.assertEqual(resumed.inputs, [])
        self.assertEqual(rows, self.rows())

    def test_resume_skips_complete_questions(self):
        run_evaluation(self.options, engine=FakeEngine())
        resumed = FakeEngine()
        run_evaluation(replace(self.options, resume=True), engine=resumed)
        self.assertEqual(resumed.inputs, [])
        self.assertEqual(len(self.rows()), 6)
    def test_resume_after_interruption_uses_whole_questions(self):
        with self.assertRaises(KeyboardInterrupt):
            run_evaluation(self.options, engine=FakeEngine(interrupt_on="q2"))
        self.assertEqual(len(self.rows()), 3)
        resumed = FakeEngine()
        summary = run_evaluation(replace(self.options, resume=True), engine=resumed)
        self.assertEqual({qid for qid, _ in resumed.inputs}, {"q2"})
        self.assertTrue(summary["complete"])
        self.assertEqual(len(self.rows()), 6)
    def test_resume_rejects_changed_config(self):
        run_evaluation(self.options, engine=FakeEngine())
        (self.root / "config.json").write_text('{"seed":43}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "fingerprint changed"):
            run_evaluation(replace(self.options, resume=True), engine=FakeEngine())
    def test_resume_rejects_changed_dataset(self):
        run_evaluation(self.options, engine=FakeEngine())
        path = self.root / "data/validation.jsonl"
        path.write_text(path.read_text() + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "fingerprint changed"):
            run_evaluation(replace(self.options, resume=True), engine=FakeEngine())
    def test_existing_output_requires_resume(self):
        run_evaluation(self.options, engine=FakeEngine())
        with self.assertRaisesRegex(ValueError, "already exists"):
            run_evaluation(self.options, engine=FakeEngine())
    def test_unknown_variant_is_rejected(self):
        with self.assertRaises(ValueError):
            run_evaluation(replace(self.options, variants=("not-a-model",)), engine=FakeEngine())
    def test_empty_answers_have_no_unsupported_fraction(self):
        run_evaluation(self.options)
        self.assertIsNone(self.rows()[0]["graph_unsupported_answer_fraction"])
    def test_default_five_validation_questions_are_development(self):
        gold_path = self.root / "data/validation.jsonl"
        base = [json.loads(line) for line in gold_path.read_text().splitlines()]
        examples = [dict(base[0], question_id="q" + str(i), text="question " + str(i)) for i in range(1, 9)]
        gold_path.write_text("".join(json.dumps(row) + "\n" for row in examples), encoding="utf-8")
        (self.root / "data/validation.questions.jsonl").write_text("".join(json.dumps({k: row[k] for k in
            ("question_id", "text", "topic_entity_ids")}) + "\n" for row in examples), encoding="utf-8")
        options = replace(self.options, backend="live", limit=EvaluationOptions().limit)
        summary = run_evaluation(options, engine=FakeEngine())
        self.assertEqual(summary["dataset_question_count"], 8)
        self.assertEqual(summary["selected_question_count"], 5)
        self.assertTrue(summary["complete"])
        self.assertFalse(summary["full_split_complete"])
        self.assertFalse(summary["formal_results"])
        self.assertEqual(summary["result_kind"], "development")
        self.assertEqual(summary["status"], "smoke")
        manifest = json.loads((options.output / "manifest.json").read_text())
        self.assertEqual(manifest["dataset_question_count"], 8)
        self.assertEqual(manifest["status"], "smoke")

    def test_only_complete_full_live_test_is_formal(self):
        for suffix in ("jsonl", "questions.jsonl"):
            source = self.root / ("data/validation." + suffix)
            (self.root / ("data/test." + suffix)).write_text(source.read_text(), encoding="utf-8")
        options = replace(self.options, backend="live", split="test")
        summary = run_evaluation(options, engine=FakeEngine())
        self.assertTrue(summary["formal_results"])
        self.assertTrue(summary["full_split_complete"])
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["result_kind"], "formal")
        manifest = json.loads((options.output / "manifest.json").read_text())
        self.assertTrue(manifest["formal_results"])
        self.assertEqual(manifest["split"], "test")

    def test_manifest_ids_groups_and_row_question_metadata(self):
        summary = run_evaluation(self.options, engine=FakeEngine())
        manifest = json.loads((self.options.output / "manifest.json").read_text())
        self.assertEqual(manifest["selected_question_ids"], ["q1", "q2"])
        self.assertEqual(manifest["variants"], list(self.options.variants))
        self.assertEqual(manifest["selected_question_groups"], summary["selected_question_groups"])
        self.assertEqual(set(manifest["selected_question_groups"]), {"q1", "q2"})
        for row in self.rows():
            self.assertEqual(row["group_id"], manifest["selected_question_groups"][row["question_id"]])
            self.assertTrue(row["question_text"].startswith("question "))
            self.assertEqual(row["topic_entity_ids"], ["topic"])

    def test_trace_and_manifest_have_matching_fingerprints(self):
        run_evaluation(self.options, engine=FakeEngine())
        manifest = json.loads((self.options.output / "manifest.json").read_text())
        for row in self.rows():
            trace = json.loads((self.options.output / row["trace_file"]).read_text())
            self.assertEqual(row["experiment_fingerprint"], manifest["experiment_fingerprint"])
            self.assertEqual(trace["experiment_fingerprint"], manifest["experiment_fingerprint"])


if __name__ == "__main__":
    unittest.main()
