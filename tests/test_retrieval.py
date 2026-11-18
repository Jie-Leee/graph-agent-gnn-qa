"""Independent acceptance checks; fixtures are synthetic, not experiment results."""
from pathlib import Path
import json
import random
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from graph_agent.retrieval import (
    EntityLinker, GraphIndex, RetrievalCase, benchmark_retrieval,
    format_paths, format_triples, to_mermaid,
)
from graph_agent.schemas import Question, Triple


class MemberBChecks(unittest.TestCase):
    def setUp(self):
        self.triples = (Triple("film", "directed_by", "person"),
                        Triple("person", "place_of_birth", "city"),
                        Triple("film", "genre", "drama"))
        self.index = GraphIndex(self.triples, {"film": ("Sample Film", "The Movie"),
                                              "person": "Some Director", "city": "Sample City"})
        self.q = Question("q", "Where was the director of Sample Film born?", ())

    def test_case_and_punctuation_alias(self):
        self.assertEqual(self.index.link_entities("SAMPLE-FILM?"), ("film",))

    def test_alternate_alias(self):
        self.assertEqual(self.index.link_entities("Who directed The Movie?"), ("film",))

    def test_longest_alias(self):
        linker = EntityLinker({"long": "New York City", "short": "New York"})
        self.assertEqual(linker.link("New York City"), ("long",))

    def test_alias_collision(self):
        self.assertEqual(EntityLinker({"a": "Name", "b": "Name"}).link("Name"), ("a", "b"))

    def test_alias_deduplication(self):
        self.assertEqual(self.index.link_entities("Sample Film and The Movie"), ("film",))

    def test_unknown_name(self):
        self.assertEqual(self.index.retrieve(Question("q", "Unknown", ())).node_ids, ())

    def test_explicit_topics_override_names(self):
        graph = self.index.retrieve(Question("q", "Sample Film", ("city",)), max_hops=0)
        self.assertEqual(graph.topic_entity_ids, ("city",))

    def test_multiple_topics_deduplicated(self):
        graph = self.index.retrieve(Question("q", "", ("film", "city", "film")), max_hops=0)
        self.assertEqual(graph.node_ids, ("film", "city"))

    def test_forward_two_hop_path(self):
        graph = self.index.retrieve(self.q)
        self.assertEqual(graph.paths["city"], self.triples[:2])
        self.assertFalse(any(step.is_reverse for step in graph.path_steps["city"]))

    def test_inverse_path_preserves_original_triples(self):
        graph = self.index.retrieve(Question("q", "", ("city",)))
        self.assertEqual(graph.paths["film"], (self.triples[1], self.triples[0]))
        self.assertTrue(all(step.is_reverse for step in graph.path_steps["film"]))
        self.assertEqual(graph.path_steps["film"][0].source_entity_id, "city")
        self.assertEqual(graph.path_steps["film"][-1].target_entity_id, "film")

    def test_directed_only(self):
        graph = self.index.retrieve(Question("q", "", ("city",)), allow_inverse=False)
        self.assertEqual(graph.node_ids, ("city",))
        self.assertEqual(graph.triples, ())

    def test_zero_hops(self):
        graph = self.index.retrieve(self.q, max_hops=0)
        self.assertEqual(graph.node_ids, ("film",))
        self.assertEqual(graph.triples, ())

    def test_zero_edges(self):
        result = self.index.retrieve_with_stats(self.q, max_edges=0)
        self.assertEqual(result.subgraph.node_ids, ("film",))
        self.assertTrue(result.stats.truncated_by_edges)

    def test_node_cap(self):
        result = self.index.retrieve_with_stats(self.q, max_nodes=2)
        self.assertEqual(len(result.subgraph.node_ids), 2)
        self.assertTrue(result.stats.truncated_by_nodes)

    def test_edge_cap(self):
        result = self.index.retrieve_with_stats(self.q, max_edges=1)
        self.assertEqual(len(result.subgraph.triples), 1)
        self.assertTrue(result.stats.truncated_by_edges)

    def test_invalid_limits(self):
        for kwargs in ({"max_hops": -1}, {"max_edges": -1}, {"max_nodes": 0}, {"max_entities": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.index.retrieve(self.q, **kwargs)

    def test_topics_exceed_budget(self):
        with self.assertRaises(ValueError):
            self.index.retrieve(Question("q", "", ("film", "city")), max_nodes=1)

    def test_duplicate_triples(self):
        graph = GraphIndex(self.triples * 3).retrieve(Question("q", "", ("film",)))
        self.assertEqual(len(graph.triples), 3)

    def test_question_ranking_before_file_order(self):
        index = GraphIndex((self.triples[2], self.triples[0], self.triples[1]))
        graph = index.retrieve(Question("q", "director born", ("film",)), max_nodes=2)
        self.assertIn("person", graph.node_ids)

    def test_deterministic_results(self):
        self.assertEqual(self.index.retrieve(self.q, max_nodes=3), self.index.retrieve(self.q, max_nodes=3))

    def test_tsv_bom_comments_and_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "g.tsv").write_text("# header\na\tr\tb\na\tr\tb\n", encoding="utf-8-sig")
            (root / "names.tsv").write_text("a\tAlpha\na\tAlternative\n", encoding="utf-8-sig")
            index = GraphIndex.from_tsv(root / "g.tsv", root / "names.tsv")
            self.assertEqual(len(index.triples), 1)
            self.assertEqual(index.link_entities("Alternative"), ("a",))

    def test_invalid_tsv_has_line_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.tsv"
            path.write_text("a\tr\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, ":1"):
                GraphIndex.from_tsv(path)

    def test_text_and_mermaid_export(self):
        graph = self.index.retrieve(Question("q", "", ("city",)))
        self.assertIn("reverse", format_paths(graph))
        self.assertIn("film --directed_by--> person", format_triples(graph.triples))
        mermaid = to_mermaid(graph)
        self.assertTrue(mermaid.startswith("flowchart LR"))
        self.assertIn("-.->", mermaid)
        self.assertIn("topic", mermaid)

    def test_labeled_recall(self):
        cases = [RetrievalCase(self.q, ("city",), self.triples[:2]),
                 RetrievalCase(self.q, ("city", "absent"), (self.triples[0], Triple("x", "r", "y")))]
        result = benchmark_retrieval(self.index, cases)
        self.assertEqual(result["questions"], 2)
        self.assertAlmostEqual(result["answer_recall_macro"], .75)
        self.assertAlmostEqual(result["answer_recall_micro"], 2 / 3)
        self.assertAlmostEqual(result["evidence_recall_macro"], .75)
        self.assertAlmostEqual(result["evidence_recall_micro"], .75)
        self.assertGreaterEqual(result["latency_p95_ms"], result["latency_p50_ms"])

    def test_gold_fields_do_not_enter_retriever(self):
        calls = []
        original = self.index.retrieve_with_stats
        def spy(question, **kwargs):
            calls.append((question, kwargs))
            return original(question, **kwargs)
        self.index.retrieve_with_stats = spy
        benchmark_retrieval(self.index, [RetrievalCase(self.q, ("city",), self.triples[:2])])
        self.assertEqual(calls[0][0], self.q)
        self.assertFalse({"answer_ids", "gold_answer_ids", "relation_path", "evidence"} & calls[0][1].keys())

    def test_query_cli_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "g.tsv").write_text("a\tr\tb\nb\ts\tc\n", encoding="utf-8")
            proc = subprocess.run([sys.executable, str(ROOT / "scripts/query_graph.py"),
                                   "--triples", str(root / "g.tsv"), "--question", "test",
                                   "--topic-entity", "a"], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("flowchart LR", proc.stdout)
            self.assertIn("b --s--> c", proc.stdout)

    def test_evaluation_cli_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "g.tsv").write_text("a\tr\tb\nb\ts\tc\n", encoding="utf-8")
            row = {"question_id": "q", "text": "test", "topic_entity_ids": ["a"],
                   "answer_ids": ["c"], "evidence": [["a", "r", "b"], ["b", "s", "c"]]}
            (root / "qa.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            proc = subprocess.run([sys.executable, str(ROOT / "scripts/evaluate_retrieval.py"),
                                   "--triples", str(root / "g.tsv"), "--qa", str(root / "qa.jsonl"),
                                   "--output", str(root / "report.json")], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            result = json.loads(proc.stdout)
            self.assertEqual(result["answer_recall_macro"], 1)
            self.assertEqual(result["evidence_recall_macro"], 1)
            self.assertEqual(result, json.loads((root / "report.json").read_text(encoding="utf-8")))

    def test_random_graph_invariants_400_cases(self):
        rng = random.Random(42)
        for trial in range(400):
            triples = [Triple(str(rng.randrange(20)), "r" + str(rng.randrange(4)), str(rng.randrange(20)))
                       for _ in range(rng.randrange(1, 150))]
            topics = tuple(rng.sample([str(i) for i in range(20)], rng.randint(1, 3)))
            hops, nodes, edges = rng.randrange(5), rng.randint(len(topics), 20), rng.randrange(40)
            inverse = bool(rng.randrange(2))
            graph = GraphIndex(triples).retrieve(Question("q", "", topics), max_hops=hops,
                                                max_nodes=nodes, max_edges=edges, allow_inverse=inverse)
            with self.subTest(trial=trial):
                self.assertLessEqual(len(graph.node_ids), nodes)
                self.assertLessEqual(len(graph.triples), edges)
                self.assertEqual(len(graph.triples), len(set(graph.triples)))
                self.assertTrue(set(graph.triples) <= set(triples))
                for triple in graph.triples:
                    self.assertIn(triple.head, graph.node_ids)
                    self.assertIn(triple.tail, graph.node_ids)
                for target, steps in graph.path_steps.items():
                    self.assertLessEqual(len(steps), hops)
                    self.assertEqual(graph.paths[target], tuple(step.triple for step in steps))
                    if not steps:
                        self.assertIn(target, topics)
                        continue
                    cursor = steps[0].source_entity_id
                    self.assertIn(cursor, topics)
                    for step in steps:
                        self.assertEqual(step.source_entity_id, cursor)
                        self.assertIn(step.triple, graph.triples)
                        if step.is_reverse:
                            self.assertTrue(inverse)
                            self.assertEqual((step.source_entity_id, step.target_entity_id),
                                             (step.triple.tail, step.triple.head))
                        else:
                            self.assertEqual((step.source_entity_id, step.target_entity_id),
                                             (step.triple.head, step.triple.tail))
                        cursor = step.target_entity_id
                    self.assertEqual(cursor, target)


def defect_probes():
    print("\nKNOWN-GAP PROBES (observations, not acceptance passes):")
    alias_triples = [Triple("film", "cast", "actor"), Triple("film", "director", "director")]
    alias_question = Question("q", "Who directed Test Film?", ("film",))
    alias_results = {}
    for label, aliases in [("actual_name_only", ("Test Film",)),
                           ("with_unmentioned_alias", ("Test Film", "The Director's Cut"))]:
        graph = GraphIndex(alias_triples, {"film": aliases}).retrieve(alias_question, max_nodes=2)
        alias_results[label] = graph.node_ids
    print(json.dumps({"alias_removes_relation_word": alias_results}, indent=2))
    triples = [Triple("film", "directed_by", "director"),
               Triple("director", "place_of_birth", "city")]
    triples += [Triple("film", "cast", f"actor:{i}") for i in range(200)]
    index = GraphIndex(triples)
    q = Question("q", "Where was the director born?", ("film",))
    bounded = index.retrieve_with_stats(q)
    enlarged = index.retrieve(q, max_nodes=203)
    print(json.dumps({"high_degree_two_hop": {
        "default_nodes": len(bounded.subgraph.node_ids),
        "director_retrieved": "director" in bounded.subgraph.node_ids,
        "city_retrieved": "city" in bounded.subgraph.node_ids,
        "city_retrieved_max_nodes_203": "city" in enlarged.node_ids,
        "node_truncation": bounded.stats.truncated_by_nodes,
    }}, indent=2))
    empty_index = GraphIndex([])
    empty_q = Question("q", "Unknown", ())
    for label, cases in [("unlabeled", [RetrievalCase(empty_q)]), ("empty_dataset", []),
                         ("mixed_evidence", [RetrievalCase(empty_q, ("x",), (Triple("x", "r", "y"),)),
                                             RetrievalCase(empty_q, ("x",))])]:
        metrics = benchmark_retrieval(empty_index, cases)
        print(json.dumps({label: {k: v for k, v in metrics.items() if "recall" in k or k == "questions"}}, indent=2))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(MemberBChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    defect_probes()
    raise SystemExit(not result.wasSuccessful())
