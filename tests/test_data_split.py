"""Regression checks for exact grouped splits and graph-grounded QA audits."""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graph_agent.data.builder import QAGroup  # noqa: E402
from graph_agent.data.split import (  # noqa: E402
    SplitError,
    assign_records,
    audit_records,
    validate_targets,
)
from graph_agent.schemas import Triple  # noqa: E402


class SplitValidationTests(unittest.TestCase):
    def test_rejects_odd_counts_before_trying_to_assign_groups(self) -> None:
        with self.assertRaises(SplitError):
            assign_records([], {"train": 1, "validation": 2, "test": 2}, entity_text={})

    def test_rejects_missing_or_unknown_splits(self) -> None:
        for targets in (
            {"train": 2, "test": 2},
            {"train": 2, "validation": 2, "test": 2, "extra": 2},
        ):
            with self.subTest(targets=targets), self.assertRaises(SplitError):
                validate_targets(targets, 2)

    def test_rejects_negative_noninteger_and_boolean_counts(self) -> None:
        for count in (-2, 2.0, True, "2"):
            with self.subTest(count=count), self.assertRaises(SplitError):
                validate_targets({"train": count, "validation": 2, "test": 2}, 2)

    def test_rejects_unavailable_paraphrase_counts(self) -> None:
        for count in (0, -1, 5, 1.5, True):
            with self.subTest(count=count), self.assertRaises(SplitError):
                validate_targets({"train": 12, "validation": 12, "test": 12}, count)

    def test_zero_size_splits_are_supported(self) -> None:
        assignment = assign_records(
            [], {"train": 0, "validation": 0, "test": 0}, entity_text={}
        )
        self.assertEqual(assignment.records, {"train": [], "validation": [], "test": []})

    def test_exact_counts_and_group_integrity_for_each_supported_count(self) -> None:
        groups = [
            QAGroup(
                f"/m/topic{i}",
                ("/people/person/place_of_birth", "/location/location/time_zones"),
                (f"/m/answer{i}",),
                (),
            )
            for i in range(10)
        ]
        for count in (1, 2, 3, 4):
            targets = {"train": 2 * count, "validation": count, "test": count}
            with self.subTest(paraphrases=count):
                assignment = assign_records(groups, targets, entity_text={}, paraphrases=count)
                self.assertEqual({k: len(v) for k, v in assignment.records.items()}, targets)
                split_topics = [
                    {row["topic_entity_ids"][0] for row in rows}
                    for rows in assignment.records.values()
                ]
                for i, topics in enumerate(split_topics):
                    for other in split_topics[i + 1:]:
                        self.assertFalse(topics & other)


class EvidenceAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.triples = (
            Triple("/m/a", "r1", "/m/middle"),
            Triple("/m/middle", "r2", "/m/b"),
            Triple("/m/middle", "r2", "/m/c"),
        )
        self.item = {
            "question_id": "train-00000",
            "text": "List all answers.",
            "topic_entity_ids": ["/m/a"],
            "answer_ids": ["/m/b", "/m/c"],
            "relation_path": ["r1", "r2"],
            "evidence": [[t.head, t.relation, t.tail] for t in self.triples],
        }

    def records(self) -> dict[str, list[dict]]:
        return {"train": [copy.deepcopy(self.item)], "validation": [], "test": []}

    def test_complete_connected_visible_evidence_passes(self) -> None:
        result = audit_records(self.records(), train_triples=self.triples)
        self.assertEqual(result["violations"], [])
        self.assertTrue(result["visible_graph_checked"])

    def test_adjacency_and_triples_give_the_same_audit(self) -> None:
        adjacency = {"/m/a": [("r1", "/m/middle")],
                     "/m/middle": [("r2", "/m/b"), ("r2", "/m/c")]}
        self.assertEqual(
            audit_records(self.records(), train_triples=self.triples),
            audit_records(self.records(), adjacency=adjacency),
        )

    def test_answer_as_unrelated_tail_does_not_count_as_evidence(self) -> None:
        records = self.records()
        records["train"][0]["evidence"] = [["/m/unrelated", "unrelated", "/m/b"]]
        result = audit_records(records)
        self.assertEqual(result["evidence_incomplete_items"], 1)
        self.assertEqual(result["disconnected_evidence_items"], 1)
        self.assertTrue(result["violations"])

    def test_connected_fabricated_edge_is_rejected(self) -> None:
        records = self.records()
        records["train"][0]["answer_ids"].append("/m/fake")
        records["train"][0]["evidence"].append(["/m/middle", "r2", "/m/fake"])
        result = audit_records(records, train_triples=self.triples)
        self.assertEqual(result["evidence_not_in_visible_graph_items"], 1)
        self.assertEqual(result["unreachable_answer_items"], 1)

    def test_omitting_reachable_gold_is_rejected(self) -> None:
        records = self.records()
        records["train"][0]["answer_ids"] = ["/m/b"]
        records["train"][0]["evidence"].pop()
        result = audit_records(records, train_triples=self.triples)
        self.assertEqual(result["evidence_incomplete_items"], 0)
        self.assertEqual(result["incomplete_answer_set_items"], 1)
        self.assertTrue(result["violations"])

    def test_duplicate_question_and_answer_ids_are_rejected(self) -> None:
        records = self.records()
        records["train"][0]["answer_ids"].append("/m/b")
        duplicate = copy.deepcopy(self.item)
        duplicate["text"] = "Another wording."
        records["test"] = [duplicate]
        result = audit_records(records)
        self.assertEqual(result["duplicate_question_ids"], 1)
        self.assertEqual(result["duplicate_answer_id_items"], 1)

    def test_visible_graph_sources_are_mutually_exclusive(self) -> None:
        with self.assertRaises(ValueError):
            audit_records(self.records(), train_triples=self.triples, adjacency={})


if __name__ == "__main__":
    unittest.main()
