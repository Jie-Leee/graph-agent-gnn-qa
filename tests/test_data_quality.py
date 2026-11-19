"""Quality flags are metadata and never alter frozen QA gold supervision."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("audit_data_quality", ROOT / "scripts" / "audit_data_quality.py")
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)

from graph_agent.data.loader import RawGraph  # noqa: E402
from graph_agent.schemas import Triple  # noqa: E402


def fixture():
    labels = {"topic": "Same  Name", "topic_outside": " same name ",
              "middle": "Middle", "a": "Salzburg", "b": "SALZBURG",
              "c": "Actor-GB", "d": "City-US"}
    triples = (Triple("topic", "r1", "middle"), Triple("middle", "r2", "a"),
               Triple("middle", "r2", "b"), Triple("middle", "r2", "c"),
               Triple("topic", "other_relation", "a"))
    graph = RawGraph({"train": triples, "valid": (), "test": ()}, labels, {"r1": "first", "r2": "second"})
    item = {"question_id": "train-0", "text": "A question", "topic_entity_ids": ["topic"],
            "answer_ids": ["a", "b", "c"], "relation_path": ["r1", "r2"],
            "evidence": [["topic", "r1", "middle"], ["middle", "r2", "a"],
                         ["middle", "r2", "b"], ["middle", "r2", "c"]]}
    return graph, {"train": [item, {**copy.deepcopy(item), "question_id": "train-1", "text": "A rephrasing"}],
                   "validation": [], "test": []}


class QualityFlagTests(unittest.TestCase):
    def test_collisions_distinguish_full_mapping_from_visible_graph(self):
        graph, records = fixture()
        rows, _ = quality.annotate_groups(records, graph)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertTrue(row["flags"]["topic_label_collision_all_mapping"])
        self.assertFalse(row["flags"]["topic_label_collision_visible_graph"])
        self.assertEqual(row["topic_label_candidate_ids_all_mapping"], ["topic", "topic_outside"])
        self.assertEqual(row["topic_label_candidate_ids_visible_graph"], ["topic"])

    def test_visible_collision_uses_entities_at_either_edge_end(self):
        graph, records = fixture()
        graph = RawGraph({**graph.triples, "train": graph.train_triples +
                          (Triple("middle", "another", "topic_outside"),)}, graph.entity_text, graph.relation_text)
        row = quality.annotate_groups(records, graph)[0][0]
        self.assertTrue(row["flags"]["topic_label_collision_visible_graph"])

    def test_distinct_same_name_answer_ids_are_preserved(self):
        graph, records = fixture()
        original = copy.deepcopy(records)
        row = quality.annotate_groups(records, graph)[0][0]
        self.assertTrue(row["flags"]["same_label_answer_ids"])
        self.assertEqual(row["answer_label_collisions"][0]["entity_ids"], ["a", "b"])
        self.assertEqual(row["answer_count"], 3)
        self.assertEqual(records, original)

    def test_partial_direct_neighbor_is_not_all_answers_or_invalid_gold(self):
        graph, records = fixture()
        row = quality.annotate_groups(records, graph)[0][0]
        self.assertEqual(row["direct_neighbor_answer_ids"], ["a"])
        self.assertTrue(row["flags"]["any_gold_direct_neighbor"])
        self.assertFalse(row["flags"]["all_gold_direct_neighbors"])
        self.assertNotIn("invalid", row["flags"])
        self.assertEqual(row["topic_out_degree"], 2)
        self.assertEqual(row["answer_size_bucket"], "2-5")

    def test_all_direct_neighbors_is_topological_and_relation_agnostic(self):
        graph, records = fixture()
        extra = (Triple("topic", "different", "b"), Triple("topic", "different", "c"))
        graph = RawGraph({**graph.triples, "train": graph.train_triples + extra}, graph.entity_text, graph.relation_text)
        row = quality.annotate_groups(records, graph)[0][0]
        self.assertTrue(row["flags"]["all_gold_direct_neighbors"])
        self.assertEqual(row["direct_neighbor_answer_ids"], ["a", "b", "c"])

    def test_source_suffixes_remain_exact_and_do_not_normalize_identity(self):
        graph, records = fixture()
        row = quality.annotate_groups(records, graph)[0][0]
        self.assertEqual(row["source_suffix_labels"], [{"entity_id": "c", "raw_label": "Actor-GB"}])
        self.assertNotEqual(quality.normalized_label("Actor-GB"), quality.normalized_label("Actor"))
        self.assertEqual(graph.entity_text["c"], "Actor-GB")

    def test_renaming_paraphrase_does_not_change_group_identity(self):
        graph, records = fixture()
        first = quality.annotate_groups(records, graph)[0][0]
        records["train"].reverse()
        second = quality.annotate_groups(records, graph)[0][0]
        self.assertEqual(first, second)
        self.assertEqual(first["question_ids"], ["train-0", "train-1"])

    def test_cross_split_same_group_and_inconsistent_gold_fail(self):
        graph, records = fixture()
        bad = copy.deepcopy(records)
        bad["test"] = [{**bad["train"][0], "question_id": "test-0"}]
        with self.assertRaisesRegex(ValueError, "crosses"):
            quality.annotate_groups(bad, graph)
        bad = copy.deepcopy(records)
        bad["train"][1]["answer_ids"] = ["a"]
        with self.assertRaisesRegex(ValueError, "conflicting"):
            quality.annotate_groups(bad, graph)

    def test_review_preserves_gold_and_declares_review_pending(self):
        graph, records = fixture()
        rows, representatives = quality.annotate_groups(records, graph)
        row = quality.review_record(rows[0], representatives[rows[0]["group_id"]], graph.entity_text)
        self.assertEqual(row["review_status"], "pending_semantic_review")
        self.assertEqual([answer["entity_id"] for answer in row["answers"]], ["a", "b", "c"])
        self.assertEqual(row["evidence"][0]["head_name"], "Same  Name")


class QualitySamplingTests(unittest.TestCase):
    def test_data_module_code_mismatch_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            processed = root / "processed"
            processed.mkdir()
            (processed / "manifest.json").write_text(json.dumps({
                "files": {}, "raw_inputs": {}, "code_sha256": {"loader.py": "0" * 64}
            }), encoding="utf-8")
            with patch.object(quality, "resolve_raw_files", return_value={}), patch.object(
                quality, "load_raw"
            ) as load_raw, self.assertRaisesRegex(ValueError, "Data code differs"):
                quality.audit(root / "raw", processed, root / "quality")
            load_raw.assert_not_called()
            self.assertFalse((root / "quality").exists())

    def test_raw_graph_mismatch_is_rejected_before_annotation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            processed = root / "processed"
            processed.mkdir()
            source = root / "changed_train.txt"
            source.write_text("changed raw graph", encoding="utf-8")
            (processed / "manifest.json").write_text(json.dumps({
                "files": {}, "raw_inputs": {"train": {"sha256": "0" * 64}}
            }), encoding="utf-8")
            with patch.object(quality, "resolve_raw_files", return_value={"train": source}), patch.object(
                quality, "load_raw"
            ) as load_raw, self.assertRaisesRegex(ValueError, "Raw input differs"):
                quality.audit(root / "raw", processed, root / "quality")
            load_raw.assert_not_called()
            self.assertFalse((root / "quality").exists())

    def test_tampered_frozen_data_is_rejected_before_graph_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            processed = root / "processed"
            processed.mkdir()
            (processed / "train.jsonl").write_text("changed", encoding="utf-8")
            (processed / "manifest.json").write_text(json.dumps({
                "files": {"train.jsonl": {"sha256": "0" * 64}}
            }), encoding="utf-8")
            with patch.object(quality, "load_raw") as load_raw, self.assertRaisesRegex(
                ValueError, "checksum mismatch"
            ):
                quality.audit(root / "raw", processed, root / "quality")
            load_raw.assert_not_called()
            self.assertFalse((root / "quality").exists())

    def test_quality_output_cannot_be_the_frozen_dataset_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "separate output"):
                quality.audit(root / "raw", root / "processed", root / "processed")

    def test_sample_covers_paths_and_prefers_another_split(self):
        rows = [{"group_id": f"g_{index}", "split": split, "relation_path": ["r1", "r2"]}
                for index, split in enumerate(["train", "train", "train", "test"])]
        rows.append({"group_id": "unique", "split": "train", "relation_path": ["r2", "r3"]})
        first = quality.select_review_groups(rows, seed=7)
        second = quality.select_review_groups(list(reversed(rows)), seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        repeated = [row for row in first if row["relation_path"] == ["r1", "r2"]]
        self.assertEqual({row["split"] for row in repeated}, {"train", "test"})
        self.assertEqual(len({row["group_id"] for row in first}), len(first))

    def test_sampling_rejects_unavailable_count(self):
        for count in (0, -1, True, 1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                quality.select_review_groups([], per_path=count)

    def test_summary_counts_groups_not_rephrasing_rows_and_writes_lf(self):
        graph, records = fixture()
        rows, reps = quality.annotate_groups(records, graph)
        reviews = [quality.review_record(rows[0], reps[rows[0]["group_id"]], graph.entity_text)]
        summary = quality.build_summary(rows, reviews, seed=1, per_path=2)
        self.assertEqual(summary["group_count"], 1)
        self.assertEqual(summary["row_count"], 2)
        self.assertEqual(summary["flag_group_counts"]["any_gold_direct_neighbor"], 1)
        self.assertEqual(summary["per_split"]["train"]["rows"], 2)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "summary.json"
            quality.write_json(path, summary)
            self.assertNotIn(b"\r\n", path.read_bytes())
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), summary)


if __name__ == "__main__":
    unittest.main()
