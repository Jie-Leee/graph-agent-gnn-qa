"""D/B boundary tests using the real retrieval implementation, no live API."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graph_agent.agent.demo import DEMO_CATALOG, DEMO_QUESTION, DEMO_TRIPLES, ScriptedProvider, answer_message, query_message
from graph_agent.agent.evidence import EntityCatalog, build_context, lexical_tokens, text_rank
from graph_agent.agent.retrieval import GraphRetrievalTool
from graph_agent.agent.runner import AgentRunner, AgentSettings
from graph_agent.retrieval.graph import GraphIndex
from graph_agent.schemas import Candidate, PathStep, Question, Subgraph, Triple


READY = {"role": "assistant", "content": "Ready"}


class AgentRetrievalTests(unittest.TestCase):
    def test_train_only_factory_passes_names_and_records_graph_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            graph_dir = raw / "FB15K-237"
            graph_dir.mkdir()
            (graph_dir / "train.txt").write_text("s\tdirected_by\td\nd\tplace_of_birth\tc\n", encoding="utf-8")
            (graph_dir / "valid.txt").write_text("s\tgenre\theldout\n", encoding="utf-8")
            (graph_dir / "test.txt").write_text("c\tgenre\theldout\n", encoding="utf-8")
            (raw / "entity2text.txt").write_text("s\tSample Film\nd\tDirector\nc\tCity\nheldout\tHeldout\n", encoding="utf-8")
            (raw / "relation2text.txt").write_text("directed_by\tdirected by\nplace_of_birth\tplace of birth\ngenre\tgenre\n", encoding="utf-8")
            tool, catalog = GraphRetrievalTool.from_raw(raw)
            self.assertEqual(len(tool.index.triples), 2)
            self.assertEqual(tool.index.link_entities("Sample Film"), ("s",))
            self.assertEqual(tool.index.relation_names["place_of_birth"], ("place of birth",))
            self.assertEqual(catalog.entity_names["c"], "City")
            graph = tool.retrieve(Question("q", "Where was the director born?", ("s",)))
            self.assertNotIn("heldout", graph.node_ids)
            self.assertEqual(tool.describe()["visible_split"], "train")
            self.assertEqual(len(tool.describe()["train_sha256"]), 64)
            self.assertIsInstance(tool.last_stats["truncated_by_nodes"], bool)

    def test_reverse_paths_reach_model_with_original_triples_and_traversal_flags(self):
        triple = Triple("director", "directed", "film")
        catalog = EntityCatalog({"film": "Film", "director": "Director"}, {"directed": "directed"})
        tool = GraphRetrievalTool(GraphIndex([triple], catalog.entity_names, catalog.relation_names))
        provider = ScriptedProvider([query_message(), READY,
                                    answer_message(ids=("director",), text="Director", evidence=(triple,))])
        runner = AgentRunner(provider, retriever=tool, catalog=catalog)
        result = runner.run(Question("reverse", "Who directed Film?", ("film",)), "text")
        self.assertEqual(result.answer_ids, ("director",))
        self.assertEqual(result.cited_evidence, (triple,))
        path = json.loads(provider.requests[-1]["messages"][1]["content"])["evidence"]["candidates"][0]["path"][0]
        self.assertEqual(path["triple"], ["director", "directed", "film"])
        self.assertEqual((path["source_id"], path["target_id"], path["is_reverse"]), ("film", "director", True))
        events = [event for event in runner.last_trace["events"] if event["stage"] == "retrieval"]
        self.assertEqual(events[0]["stats"]["retrieved_edges"], 1)

    def test_directed_only_setting_is_forwarded(self):
        triple = Triple("director", "directed", "film")
        tool = GraphRetrievalTool(GraphIndex([triple]))
        provider = ScriptedProvider([query_message(), READY, answer_message(text="Unknown")])
        runner = AgentRunner(provider, retriever=tool, settings=AgentSettings(allow_inverse=False))
        runner.run(Question("directed", "Who directed Film?", ("film",)), "text")
        self.assertEqual(tool.last_stats["retrieved_edges"], 0)
        self.assertEqual(runner.last_trace["context_candidate_ids"], [])

    def test_linked_topics_are_preserved_for_topicless_optional_input(self):
        index = GraphIndex(DEMO_TRIPLES, DEMO_CATALOG.entity_names, DEMO_CATALOG.relation_names)
        tool = GraphRetrievalTool(index)
        provider = ScriptedProvider([query_message(), READY, answer_message(ids=("city:sample",), text="Sample City")])
        runner = AgentRunner(provider, retriever=tool, catalog=DEMO_CATALOG)
        result = runner.run(Question("linked", DEMO_QUESTION.text, ()), "text")
        self.assertEqual(result.answer_ids, ("city:sample",))
        expected_topics = list(index.link_entities(DEMO_QUESTION.text))
        self.assertEqual(runner.last_trace["resolved_topic_entity_ids"], expected_topics)
        actual_topics = json.loads(provider.requests[-1]["messages"][1]["content"])["topics"]
        self.assertEqual([topic["id"] for topic in actual_topics], expected_topics)

    def test_nonnested_hop_queries_keep_latest_graph_within_budget_including_cache_hits(self):
        a = Triple("s", "r", "a")
        b = Triple("s", "r", "b")
        c = Triple("b", "r", "c")
        class NonNestedRetriever:
            def __init__(self):
                self.calls = []
            def retrieve(self, question, *, max_hops, **kwargs):
                self.calls.append(max_hops)
                if max_hops == 1:
                    return Subgraph(("s",), ("s", "a"), (a,), {"a": (a,)})
                return Subgraph(("s",), ("s", "b", "c"), (b, c), {"b": (b,), "c": (b, c)})
        retriever = NonNestedRetriever()
        provider = ScriptedProvider([query_message("one", 2), query_message("two", 1), query_message("three", 2),
                                    answer_message(ids=("c",), text="C")])
        runner = AgentRunner(provider, retriever=retriever,
                             settings=AgentSettings(max_tool_calls=3, max_nodes=3, max_edges=2))
        result = runner.run(Question("different", "Question", ("s",)), "text")
        self.assertEqual(result.answer_ids, ("c",))
        self.assertEqual(retriever.calls, [2, 1])
        self.assertEqual(set(runner.last_trace["retrieved_node_ids"]), {"s", "b", "c"})
        self.assertNotIn("a", runner.last_trace["context_candidate_ids"])

    def test_wrong_direction_metadata_is_rejected(self):
        triple = Triple("a", "r", "b")
        graph = Subgraph(("a",), ("a", "b"), (triple,), {"b": (triple,)},
                         {"b": (PathStep(triple, "b", "a", True),)})
        with self.assertRaisesRegex(ValueError, "direction metadata"):
            build_context([Candidate("b", 1, (triple,))], graph, EntityCatalog(), 2000)

    def test_invalid_inverse_setting_fails_before_inference(self):
        with self.assertRaises(ValueError):
            AgentSettings(allow_inverse="false")

    def test_compact_context_shares_long_names_and_keeps_paths_atomic(self):
        triples = (Triple("topic", "relationship", "one"), Triple("topic", "relationship", "two"))
        graph = GraphIndex(triples).retrieve(Question("q", "Question", ("topic",)))
        catalog = EntityCatalog({"topic": "Topic " * 20, "one": "One", "two": "Two"},
                                {"relationship": "Relationship " * 20})
        context = build_context([Candidate("one", 2, (triples[0],)), Candidate("two", 1, (triples[1],))],
                                graph, catalog, 2000)
        data = json.loads(context.text)
        self.assertEqual(context.candidate_ids, ("one", "two"))
        self.assertEqual(set(data["relations"]), {"relationship"})
        self.assertEqual(context.text.count(catalog.relation_names["relationship"]), 1)
        self.assertEqual(context.triples, triples)
        self.assertLessEqual(len(context.text.encode("utf-8")), 2000)

    def test_relation_inflections_are_fixed_without_supervision(self):
        self.assertEqual(lexical_tokens("contained contains"), {"contain"})
        self.assertEqual(lexical_tokens("born birthplace"), {"birth"})

    def test_text_evidence_uses_relevant_real_path_instead_of_shortcut(self):
        edges = (Triple("person", "birth", "city"), Triple("city", "contains", "university"),
                 Triple("person", "studied_at", "university"))
        question = Question("shortcut", "Return every contained location of any birthplace of Ada.", ("person",))
        graph = GraphIndex(edges).retrieve(question)
        self.assertEqual(len(graph.paths["university"]), 1)
        catalog = EntityCatalog(relation_names={"birth": "place of birth", "contains": "location contains",
                                                "studied_at": "education student"})
        candidates = text_rank(question, graph, catalog)
        candidate = next(c for c in candidates if c.entity_id == "university")
        self.assertEqual(candidate.evidence, edges[:2])
        context = build_context([candidate], graph, catalog, 2000)
        self.assertEqual(context.triples, edges[:2])
        one_hop = text_rank(question, graph, catalog, max_hops=1)
        self.assertEqual(len(next(c for c in one_hop if c.entity_id == "university").evidence), 1)


if __name__ == "__main__":
    unittest.main()
