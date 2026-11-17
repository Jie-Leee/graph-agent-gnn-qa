"""End-to-end preparation, provenance and safe consumer-interface checks."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graph_agent.data.dataset import DatasetError, load_examples, load_questions  # noqa: E402
from graph_agent.data.prepare import prepare_dataset  # noqa: E402
from graph_agent.data.split import SplitError  # noqa: E402
from graph_agent.schemas import QAExample, Question, Triple  # noqa: E402

BIRTH = "/people/person/place_of_birth"
ZONE = "/location/location/time_zones"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PreparationIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw"
        graph = self.raw / "FB15K-237"
        graph.mkdir(parents=True)
        train = []
        valid = []
        test = []
        labels = {}
        for i in range(8):
            topic, middle, answer = f"/m/topic{i}", f"/m/city{i}", f"/m/zone{i}"
            train.extend([(topic, BIRTH, middle), (middle, ZONE, answer)])
            # Every training query would gain extra answers if held-out edges
            # accidentally became visible, regardless of the sampled topics.
            valid.append((middle, ZONE, f"/m/valid_only{i}"))
            test.append((middle, ZONE, f"/m/test_only{i}"))
            for entity in (topic, middle, answer, f"/m/valid_only{i}", f"/m/test_only{i}"):
                labels[entity] = entity.rsplit("/", 1)[-1].replace("_", " ")
        valid.extend([("/m/heldout", BIRTH, "/m/heldout_city"),
                      ("/m/heldout_city", ZONE, "/m/heldout_zone")])
        for entity in ("/m/heldout", "/m/heldout_city", "/m/heldout_zone"):
            labels[entity] = entity.rsplit("/", 1)[-1]
        for name, rows in (("train", train), ("valid", valid), ("test", test)):
            (graph / f"{name}.txt").write_text(
                "".join("\t".join(row) + "\n" for row in rows), encoding="utf-8"
            )
        (self.raw / "entity2text.txt").write_text(
            "".join(f"{key}\t{value}\n" for key, value in sorted(labels.items())),
            encoding="utf-8",
        )
        (self.raw / "relation2text.txt").write_text(
            f"{BIRTH}\tplace of birth\n{ZONE}\ttime zone\n", encoding="utf-8"
        )
        self.options = {
            "seed": 11,
            "targets": {"train": 4, "validation": 2, "test": 2},
            "paraphrases": 2,
            "max_answers": 4,
            "max_groups": 32,
            "min_path_support": 1,
            "max_dominant_share": 1.0,
            "max_answer_set_share": 1.0,
        }

    def build(self, name: str = "processed") -> tuple[Path, dict]:
        output = self.root / name
        return output, prepare_dataset(self.raw, output, **self.options)

    def test_outputs_round_trip_and_heldout_edges_never_supply_answers(self) -> None:
        output, stats = self.build()
        self.assertEqual(stats["audit"]["violations"], [])
        self.assertTrue(stats["audit"]["visible_graph_checked"])
        for split, expected_count in self.options["targets"].items():
            examples = load_examples(output / f"{split}.jsonl")
            questions = load_questions(output / f"{split}.questions.jsonl")
            self.assertEqual(len(examples), expected_count)
            self.assertEqual(tuple(example.question for example in examples), questions)
            for example in examples:
                self.assertIsInstance(example, QAExample)
                self.assertIsInstance(example.question, Question)
                self.assertTrue(all(isinstance(edge, Triple) for edge in example.evidence))
                topic, = example.question.topic_entity_ids
                self.assertNotEqual(topic, "/m/heldout")
                index = topic.removeprefix("/m/topic")
                self.assertEqual(example.answer_ids, (f"/m/zone{index}",))
                self.assertEqual(example.relation_path, (BIRTH, ZONE))
                self.assertEqual(example.evidence, (
                    Triple(topic, BIRTH, f"/m/city{index}"),
                    Triple(f"/m/city{index}", ZONE, f"/m/zone{index}"),
                ))

    def test_manifest_records_all_parameters_raw_hashes_and_output_hashes(self) -> None:
        output, _ = self.build()
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        for name, value in self.options.items():
            key = "paraphrases_per_group" if name == "paraphrases" else name
            self.assertEqual(manifest[key], value)
        self.assertEqual(set(manifest["raw_inputs"]),
                         {"train", "valid", "test", "entity_text", "relation_text"})
        for entry in manifest["raw_inputs"].values():
            self.assertEqual(entry["sha256"], sha256(self.raw / entry["file"]))
        self.assertIn("prepare.py", manifest["code_sha256"])
        required_data_files = {
            f"{split}{suffix}.jsonl" for split in self.options["targets"]
            for suffix in ("", ".questions")
        }
        self.assertTrue(required_data_files <= manifest["files"].keys())
        for filename, entry in manifest["files"].items():
            self.assertEqual(entry["sha256"], sha256(output / filename))
            if filename.endswith(".jsonl"):
                self.assertEqual(entry["items"], len((output / filename).read_text().splitlines()))

    def test_inference_jsonl_and_loader_expose_only_question_fields(self) -> None:
        output, _ = self.build()
        for split in self.options["targets"]:
            rows = [json.loads(line) for line in
                    (output / f"{split}.questions.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all(set(row) == {"question_id", "text", "topic_entity_ids"}
                                for row in rows))
            # The loader is also safe if a caller supplies the supervision file.
            for question in load_questions(output / f"{split}.jsonl"):
                self.assertEqual(set(asdict(question)), {"question_id", "text", "topic_entity_ids"})
                self.assertFalse(hasattr(question, "answer_ids"))
                self.assertFalse(hasattr(question, "relation_path"))
                self.assertFalse(hasattr(question, "evidence"))

    def test_same_inputs_and_parameters_reproduce_outputs_in_separate_directories(self) -> None:
        first, first_stats = self.build("first")
        second, second_stats = self.build("second")
        self.assertEqual(first_stats, second_stats)
        for filename in ("train.jsonl", "validation.jsonl", "test.jsonl",
                         "train.questions.jsonl", "validation.questions.jsonl", "test.questions.jsonl",
                         "stats.json", "stats.md", "samples.md"):
            with self.subTest(filename=filename):
                self.assertEqual((first / filename).read_bytes(), (second / filename).read_bytes())
        manifests = [json.loads((path / "manifest.json").read_text(encoding="utf-8"))
                     for path in (first, second)]
        for manifest in manifests:
            manifest.pop("created_at")
        self.assertEqual(*manifests)

    def test_invalid_parameters_are_rejected_before_raw_loading(self) -> None:
        invalid = (
            {"targets": {}},
            {"targets": {"train": 1, "validation": 2, "test": 2}},
            {"targets": {"train": -2, "validation": 2, "test": 2}},
            {"paraphrases": 0}, {"paraphrases": 5},
            {"max_answers": 0}, {"max_groups": 0}, {"min_path_support": 0},
            {"max_dominant_share": 1.1}, {"max_answer_set_share": -0.1},
        )
        for overrides in invalid:
            options = {**self.options, **overrides}
            output = self.root / "invalid_output"
            with self.subTest(overrides=overrides), patch(
                "graph_agent.data.prepare.load_raw"
            ) as load_raw:
                with self.assertRaises((ValueError, SplitError)):
                    prepare_dataset(self.root / "missing_raw", output, **options)
                load_raw.assert_not_called()
                self.assertFalse(output.exists())

    def test_failed_audit_writes_no_outputs_and_preserves_existing_files(self) -> None:
        for existing in (False, True):
            output = self.root / ("existing" if existing else "new")
            if existing:
                output.mkdir()
                (output / "train.jsonl").write_text("existing data", encoding="utf-8")
            with self.subTest(existing=existing), patch(
                "graph_agent.data.prepare.audit_records", return_value={"violations": ["broken evidence"]}
            ):
                with self.assertRaisesRegex(ValueError, "audit failed"):
                    prepare_dataset(self.raw, output, **self.options)
            if existing:
                self.assertEqual(list(output.iterdir()), [output / "train.jsonl"])
                self.assertEqual((output / "train.jsonl").read_text(), "existing data")
            else:
                self.assertFalse(output.exists())


class DatasetConsumerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "records.jsonl"
        self.item = {
            "question_id": "test-00000", "text": "A question?",
            "topic_entity_ids": ["/m/a"], "answer_ids": ["/m/b"],
            "relation_path": ["r1", "r2"],
            "evidence": [["/m/a", "r1", "/m/m"], ["/m/m", "r2", "/m/b"]],
        }

    def write(self, rows: list[dict]) -> None:
        self.path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_malformed_evidence_is_rejected_with_file_and_line(self) -> None:
        for evidence in ([], None, "bad", [["a", "r"]], [["a", "r", ""]], [[1, "r", "b"]]):
            item = {**self.item, "evidence": evidence}
            self.write([item])
            with self.subTest(evidence=evidence), self.assertRaises(DatasetError) as caught:
                load_examples(self.path)
            self.assertIn(f"{self.path}:1:", str(caught.exception))

    def test_duplicate_question_ids_are_rejected_by_both_consumers(self) -> None:
        duplicate = {**self.item, "text": "Another question?"}
        self.write([self.item, duplicate])
        for consumer in (load_examples, load_questions):
            with self.subTest(consumer=consumer.__name__), self.assertRaisesRegex(
                DatasetError, "duplicate question_id"
            ):
                consumer(self.path)

    def test_duplicate_entity_ids_are_rejected(self) -> None:
        for field in ("answer_ids", "topic_entity_ids"):
            item = copy.deepcopy(self.item)
            item[field] *= 2
            self.write([item])
            with self.subTest(field=field), self.assertRaisesRegex(DatasetError, "duplicate IDs"):
                load_examples(self.path)

    def test_question_loader_discards_even_malformed_gold_supervision(self) -> None:
        self.write([{**self.item, "answer_ids": "secret answer", "relation_path": None,
                     "evidence": "secret evidence", "extra": "secret metadata"}])
        questions = load_questions(self.path)
        self.assertEqual(questions, (Question("test-00000", "A question?", ("/m/a",)),))
        self.assertEqual(set(asdict(questions[0])), {"question_id", "text", "topic_entity_ids"})


if __name__ == "__main__":
    unittest.main()
