"""Boundary checks for A handoff without real data downloads or LLM calls."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("accept_data_handoff", ROOT / "scripts" / "accept_data_handoff.py")
handoff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handoff)

from graph_agent.data.prepare import prepare_dataset  # noqa: E402
from graph_agent.schemas import QAExample, Question, Triple  # noqa: E402


class UnimplementedAgent:
    """Local fixture; real D development must not break this unit contract."""

    def run(self, question, mode):
        raise NotImplementedError("Synthetic Agent fixture")


class QuestionBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.question = Question("test-1", "List every time zone of any birthplace of Ada.", ("/m/a",))

    def test_payload_has_exact_three_fields(self):
        payload = handoff.question_payload(self.question)
        self.assertEqual(set(payload), {"question_id", "text", "topic_entity_ids"})
        self.assertEqual(payload["topic_entity_ids"], ["/m/a"])

    def test_supervised_example_cannot_be_used_as_inference_input(self):
        gold = QAExample(self.question, ("/m/b",), ("r1", "r2"), ())
        with self.assertRaises(handoff.HandoffError):
            handoff.question_payload(gold)

    def test_probe_catches_consumer_that_forwards_gold_fields(self):
        def bad_consumer(question):
            return {**handoff.question_payload(question), "answer_ids": ["secret"]}

        with self.assertRaisesRegex(handoff.HandoffError, "extra gold"):
            handoff.contract_probe([self.question], bad_consumer)

    def test_probe_catches_consumer_that_changes_question(self):
        def changed_consumer(question):
            return {**handoff.question_payload(question), "text": "modified"}

        with self.assertRaises(handoff.HandoffError):
            handoff.contract_probe([self.question], changed_consumer)

    def test_poisoned_gold_does_not_change_question_exports(self):
        result = handoff.poison_gold_probe([self.question])
        self.assertTrue(result["question_objects_unchanged"])
        self.assertEqual(result["question_count"], 1)

    def test_scaffold_D_is_reported_pending(self):
        result = handoff.downstream_agent_status(self.question, UnimplementedAgent)
        self.assertEqual(result["status"], "pending_downstream_implementation")
        self.assertFalse(result["actual_llm_or_tool_loop_verified"])

    def test_returning_agent_does_not_automatically_become_accepted(self):
        class ReturningAgent:
            def run(self, question, mode):
                return object()

        self.assertEqual(handoff.downstream_agent_status(self.question, ReturningAgent)["status"],
                         "requires_downstream_review")

    def test_changed_real_D_module_is_not_invoked(self):
        with tempfile.TemporaryDirectory() as directory:
            changed_source = Path(directory) / "runner.py"
            changed_source.write_text("# future real API implementation", encoding="utf-8")
            with patch.object(handoff, "AGENT_SOURCE", changed_source), patch.object(
                handoff.AgentRunner, "run", side_effect=AssertionError("must not invoke changed D")
            ):
                result = handoff.downstream_agent_status(self.question)
                self.assertEqual(result["status"], "requires_downstream_review")


class SyntheticHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.raw = Path(self.temp.name) / "raw"
        self.processed = Path(self.temp.name) / "processed"
        graph_dir = self.raw / "FB15K-237"
        graph_dir.mkdir(parents=True)
        birth = "/people/person/place_of_birth"
        zone = "/location/location/time_zones"
        triples = []
        entities = {}
        for number in range(8):
            topic, middle, answer = f"/m/person{number}", f"/m/city{number}", f"/m/zone{number}"
            triples.extend([f"{topic}\t{birth}\t{middle}\n", f"{middle}\t{zone}\t{answer}\n"])
            entities.update({topic: f"Person {number}", middle: f"City {number}", answer: f"Zone {number}"})
        (graph_dir / "train.txt").write_text("".join(triples), encoding="utf-8")
        (graph_dir / "valid.txt").write_text(f"/m/city0\t{zone}\t/m/heldout\n", encoding="utf-8")
        (graph_dir / "test.txt").write_text(f"/m/city1\t{zone}\t/m/heldout\n", encoding="utf-8")
        entities["/m/heldout"] = "Heldout zone"
        (self.raw / "entity2text.txt").write_text("".join(f"{key}\t{value}\n" for key, value in entities.items()), encoding="utf-8")
        (self.raw / "relation2text.txt").write_text(f"{birth}\tplace of birth\n{zone}\ttime zone\n", encoding="utf-8")
        prepare_dataset(self.raw, self.processed, targets={"train": 4, "validation": 2, "test": 2},
                        min_path_support=1, max_dominant_share=1, max_answer_set_share=1)

    def test_synthetic_train_only_index_and_honest_statuses(self):
        report = handoff.run_acceptance(self.raw, self.processed, agent_factory=UnimplementedAgent)
        self.assertEqual(report["A_export"]["split_items"], {"train": 4, "validation": 2, "test": 2})
        self.assertEqual(report["B_scaffold"]["indexed_training_edges"], 16)
        self.assertEqual(report["B_scaffold"]["heldout_exclusive_edges"], 2)
        self.assertEqual(report["B_scaffold"]["heldout_exclusive_edges_indexed"], 0)
        self.assertEqual(report["B_scaffold"]["diagnostic"]["mean_gold_node_recall"], 1.0)
        self.assertEqual(report["D_contract_probe"]["status"], "passed")
        self.assertEqual(report["D_actual_integration"]["status"], "pending_downstream_implementation")
        self.assertEqual(report["human_acceptance"]["status"], "pending_no_confirmation")

    def test_changed_export_fails_checksum_before_receiving(self):
        with (self.processed / "test.questions.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("{}\n")
        with self.assertRaisesRegex(handoff.HandoffError, "checksum"):
            handoff.run_acceptance(self.raw, self.processed)

    def test_bad_index_with_heldout_edge_is_rejected(self):
        class BadIndex(handoff.GraphIndex):
            def __init__(self, triples):
                super().__init__(triples)
                self.outgoing["/m/city0"].append(Triple("/m/city0", "/location/location/time_zones", "/m/heldout"))

        with self.assertRaisesRegex(handoff.HandoffError, "not exactly"):
            handoff.run_acceptance(self.raw, self.processed, graph_index_factory=BadIndex)

    def test_bad_serializing_receiver_fails_acceptance(self):
        with self.assertRaises(handoff.HandoffError):
            handoff.run_acceptance(self.raw, self.processed,
                                   consumer=lambda question: {**handoff.question_payload(question), "evidence": []})

    def test_retriever_cannot_return_fabricated_edges(self):
        class FabricatingRetriever(handoff.GraphIndex):
            def retrieve(self, question, **options):
                subgraph = super().retrieve(question, **options)
                fake = Triple("/m/a", "made_up", "/m/b")
                return replace(subgraph, triples=subgraph.triples + (fake,))

        with self.assertRaisesRegex(handoff.HandoffError, "outside the visible graph"):
            handoff.run_acceptance(self.raw, self.processed, graph_index_factory=FabricatingRetriever)


if __name__ == "__main__":
    unittest.main()
