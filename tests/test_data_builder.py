"""Regression tests for complete answer sets and strict raw text loading."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graph_agent.data.builder import enumerate_groups, render_group_items
from graph_agent.data.loader import RawDataError, load_raw, read_text_map

BIRTH = "/people/person/place_of_birth"
ZONE = "/location/location/time_zones"
NATION = "/people/person/nationality"


class CompleteAnswerTests(unittest.TestCase):
    def test_mixed_return_to_topic_rejects_entire_group(self):
        graph = {"h": ((BIRTH, "m"),), "m": ((ZONE, "h"), (ZONE, "a"))}
        self.assertEqual(enumerate_groups(graph), [])

    def test_mixed_second_hop_self_loop_rejects_entire_group(self):
        graph = {"h": ((BIRTH, "m"),), "m": ((ZONE, "m"), (ZONE, "a"))}
        self.assertEqual(enumerate_groups(graph), [])

    def test_first_hop_self_loop_rejects_entire_group(self):
        graph = {
            "h": ((BIRTH, "h"), (BIRTH, "m"), (ZONE, "a")),
            "m": ((ZONE, "b"),),
        }
        groups = enumerate_groups(graph)
        self.assertNotIn(("h", (BIRTH, ZONE)), {group.key for group in groups})

    def test_complete_answers_union_all_intermediates(self):
        graph = {
            "h": ((BIRTH, "m"), (BIRTH, "n")),
            "m": ((ZONE, "a"), (ZONE, "b")),
            "n": ((ZONE, "b"), (ZONE, "c")),
        }
        group, = enumerate_groups(graph)
        self.assertEqual(group.answer_ids, ("a", "b", "c"))
        self.assertEqual(len(group.evidence), len(set(group.evidence)))

    def test_max_groups_is_exact_even_with_many_paths_for_one_head(self):
        graph = {"h": ((BIRTH, "m"),), "m": ((ZONE, "a"), (NATION, "b"))}
        self.assertEqual(len(enumerate_groups(graph, max_groups=1)), 1)
        self.assertEqual(len(enumerate_groups(graph, max_groups=2)), 2)

    def test_invalid_limits_fail_explicitly(self):
        for parameter in ("max_answers", "max_groups"):
            for value in (0, -1, 1.5, True):
                with self.subTest(parameter=parameter, value=value):
                    with self.assertRaises(ValueError):
                        enumerate_groups({}, **{parameter: value})

    def test_one_to_four_paraphrases_are_distinct_and_deterministic(self):
        for tails in (("a",), ("a", "b")):
            graph = {"h": ((BIRTH, "m"),), "m": tuple((ZONE, t) for t in tails)}
            group, = enumerate_groups(graph)
            for count in range(1, 5):
                with self.subTest(tails=tails, count=count):
                    items = render_group_items(group, count, {"h": "Example"})
                    self.assertEqual(len({text for _, text in items}), count)
                    self.assertEqual(items, render_group_items(group, count, {"h": "Example"}))
            for count in (0, -1, 5, True):
                with self.assertRaises(ValueError):
                    render_group_items(group, count, {})


class TextMapTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / "entity2text.txt"

    def test_malformed_or_empty_maps_fail(self):
        for text in ("", "\n", "id without tab\n", "id\t \n", "\tName\n", "id\tA\tB\n"):
            with self.subTest(text=text):
                self.path.write_text(text, encoding="utf-8")
                with self.assertRaises(RawDataError):
                    read_text_map(self.path)

    def test_conflicting_duplicate_id_fails_with_line_number(self):
        self.path.write_text("id\tAlpha\nid\tBeta\n", encoding="utf-8")
        with self.assertRaisesRegex(RawDataError, r"entity2text.txt:2"):
            read_text_map(self.path)

    def test_identical_duplicate_and_utf8_bom_are_supported(self):
        self.path.write_text("id\tAlpha\nid\tAlpha\n", encoding="utf-8-sig")
        self.assertEqual(read_text_map(self.path), {"id": "Alpha"})

    def test_load_raw_rejects_missing_train_entity_labels(self):
        for split in ("train", "valid", "test"):
            (self.root / f"{split}.txt").write_text(f"h\t{BIRTH}\tm\n", encoding="utf-8")
        self.path.write_text("h\tHead\n", encoding="utf-8")
        (self.root / "relation2text.txt").write_text(f"{BIRTH}\tBirthplace\n", encoding="utf-8")
        with self.assertRaisesRegex(RawDataError, "实体缺少文本映射"):
            load_raw(self.root)

    def test_load_raw_rejects_missing_train_relation_labels(self):
        for split in ("train", "valid", "test"):
            (self.root / f"{split}.txt").write_text(f"h\t{BIRTH}\tm\n", encoding="utf-8")
        self.path.write_text("h\tHead\nm\tMiddle\n", encoding="utf-8")
        (self.root / "relation2text.txt").write_text("other\tOther\n", encoding="utf-8")
        with self.assertRaisesRegex(RawDataError, "关系缺少文本映射"):
            load_raw(self.root)


if __name__ == "__main__":
    unittest.main()
