from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from graph_agent.agent.demo import (
    DEMO_CATALOG, DEMO_QUESTION, DEMO_TRIPLES, ScriptedProvider, answer_message, query_message,
)
from graph_agent.agent.evidence import EntityCatalog, build_context
from graph_agent.agent.provider import Completion, ProviderError
from graph_agent.agent.runner import AgentError, AgentRunner, AgentSettings
from graph_agent.data.dataset import load_questions
from graph_agent.retrieval.graph import GraphIndex
from graph_agent.schemas import Candidate, QAExample, Question, Triple


READY = {"role": "assistant", "content": "Ready"}


class CountingRetriever(GraphIndex):
    def __init__(self):
        super().__init__(DEMO_TRIPLES)
        self.calls = []

    def retrieve(self, question, **kwargs):
        self.calls.append((question, kwargs))
        return super().retrieve(question, **kwargs)


class RunnerTests(unittest.TestCase):
    def runner(self, messages, **kwargs):
        provider = ScriptedProvider(messages)
        retriever = CountingRetriever()
        return AgentRunner(provider, retriever=retriever, catalog=DEMO_CATALOG, **kwargs), provider, retriever

    def test_llm_never_routes_or_receives_topic_ids_graph_or_catalog(self):
        runner, provider, retriever = self.runner([answer_message(names=("Sample City",), text="Sample City")])
        result = runner.run(DEMO_QUESTION, "llm")
        self.assertEqual(result.answer_ids, ("city:sample",))
        self.assertEqual(result.tool_calls, 0)
        self.assertFalse(retriever.calls)
        user = json.loads(provider.requests[0]["messages"][1]["content"])
        self.assertEqual(user, {"question": DEMO_QUESTION.text})
        self.assertIsNone(provider.requests[0]["tools"])

    def test_query_loop_returns_only_real_evidence(self):
        runner, provider, retriever = self.runner([
            query_message(), READY, answer_message(ids=("city:sample",), text="Sample City",
                evidence=(DEMO_TRIPLES[0], DEMO_TRIPLES[2]))])
        result = runner.run(DEMO_QUESTION, "text")
        self.assertEqual(result.answer_ids, ("city:sample",))
        self.assertEqual(result.tool_calls, 1)
        self.assertEqual(len(retriever.calls), 1)
        self.assertEqual(provider.requests[1]["messages"][-1]["role"], "tool")
        self.assertEqual(result.cited_evidence, (DEMO_TRIPLES[0], DEMO_TRIPLES[2]))

    def test_model_can_skip_retrieval(self):
        runner, _, retriever = self.runner([READY, answer_message(text="4")])
        result = runner.run(Question("skip", "2+2?", ()), "text")
        self.assertEqual(result.tool_calls, 0)
        self.assertFalse(retriever.calls)

    def test_repeated_query_is_cached_and_max_calls_force_answer_stage(self):
        runner, provider, retriever = self.runner([
            query_message("one"), query_message("two"), answer_message(text="Unknown")])
        result = runner.run(DEMO_QUESTION, "text")
        self.assertEqual(result.tool_calls, 2)
        self.assertEqual(len(retriever.calls), 1)
        self.assertEqual(len(provider.requests), 3)
        self.assertIsNone(provider.requests[-1]["tools"])

    def test_invalid_tool_batch_causes_no_retrieval(self):
        bad_calls = []
        for args in ('{"max_hops":3}', '{"max_hops":true}', '{"max_hops":2,"answer_ids":["x"]}', "bad"):
            message = query_message()
            message["tool_calls"][0]["function"]["arguments"] = args
            bad_calls.append(message)
        extra = query_message()
        extra["tool_calls"] *= 3
        bad_calls.append(extra)
        unknown = query_message()
        unknown["tool_calls"][0]["function"]["name"] = "read_file"
        bad_calls.append(unknown)
        for message in bad_calls:
            with self.subTest(message=message):
                runner, _, retriever = self.runner([message])
                with self.assertRaises(AgentError):
                    runner.run(DEMO_QUESTION, "text")
                self.assertFalse(retriever.calls)

    def test_missing_dependencies_fail_before_provider_call(self):
        provider = ScriptedProvider([])
        for mode in ("text", "gnn"):
            with self.assertRaises(AgentError):
                AgentRunner(provider).run(DEMO_QUESTION, mode)
        runner, provider, _ = self.runner([])
        with self.assertRaisesRegex(AgentError, "trained"):
            runner.run(DEMO_QUESTION, "gnn")
        self.assertFalse(provider.requests)

    def test_shared_route_used_in_both_modes_without_text_context_leaking_to_gnn(self):
        class Scorer:
            def rank(self, question, graph):
                return [Candidate("city:sample", 9, (DEMO_TRIPLES[0], DEMO_TRIPLES[2]))]
        runner, provider, retriever = self.runner([
            query_message(), READY, answer_message(text="text"), answer_message(text="gnn")], scorer=Scorer())
        first = runner.run(DEMO_QUESTION, "text")
        second = runner.run(DEMO_QUESTION, "gnn")
        self.assertTrue(runner.last_trace["route_cache_hit"])
        self.assertEqual(first.tool_calls, second.tool_calls)
        self.assertEqual(len(retriever.calls), 1)
        self.assertEqual(len(provider.requests), 4)
        evidence = json.loads(provider.requests[-1]["messages"][1]["content"])["evidence"]
        self.assertEqual([row["entity"]["id"] for row in evidence["candidates"]], ["city:sample"])
        runner.clear_cache()
        self.assertFalse(runner._plans)

    def test_poisoned_supervision_never_enters_actual_prompts(self):
        public = asdict(DEMO_QUESTION)
        poison = {**public, "answer_ids": ["GOLD_SECRET"], "relation_path": ["PATH_SECRET"],
                  "evidence": "EVIDENCE_SECRET"}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "questions.jsonl"
            source.write_text(json.dumps(poison), encoding="utf-8")
            runner, provider, _ = self.runner([query_message(), READY, answer_message(text="Unknown")])
            runner.run(load_questions(source)[0], "text")
            self.assertNotIn("SECRET", json.dumps(provider.requests))
        runner, _, _ = self.runner([])
        with self.assertRaises(TypeError):
            runner.run(QAExample(DEMO_QUESTION, ("GOLD_SECRET",)), "llm")

    def test_real_A_validation_inputs_are_accepted_without_gold_fields(self):
        questions = load_questions(ROOT / "data/processed/validation.questions.jsonl")
        runner, provider, _ = self.runner([answer_message(text="Unknown") for _ in questions])
        for question in questions:
            runner.run(question, "llm")
        self.assertEqual(len(provider.requests), 200)
        for request in provider.requests:
            self.assertEqual(set(json.loads(request["messages"][1]["content"])), {"question"})

    def test_same_name_different_ids_remain_ambiguous(self):
        catalog = EntityCatalog({"a": "Salzburg", "b": "Salzburg"})
        provider = ScriptedProvider([answer_message(names=("SALZBURG",), text="Salzburg")])
        runner = AgentRunner(provider, catalog=catalog)
        result = runner.run(DEMO_QUESTION, "llm")
        self.assertEqual(result.answer_ids, ())
        self.assertEqual(runner.last_trace["resolution"]["ambiguous_names"]["SALZBURG"], ["a", "b"])

    def test_hallucinated_ids_are_not_accepted_and_names_are_recorded(self):
        runner, _, _ = self.runner([answer_message(ids=("invented",), names=("Unknown Place",), text="uncertain")])
        result = runner.run(DEMO_QUESTION, "llm")
        self.assertEqual(result.answer_ids, ())
        self.assertEqual(runner.last_trace["resolution"]["rejected_ids"], ["invented"])
        self.assertEqual(runner.last_trace["resolution"]["unresolved_names"], ["Unknown Place"])

    def test_malformed_answer_or_unseen_citation_fails_instead_of_becoming_prediction(self):
        for message in ({"role": "assistant", "content": "not JSON"},
                        answer_message(evidence=(DEMO_TRIPLES[0],))):
            runner, _, _ = self.runner([message])
            with self.assertRaises(AgentError):
                runner.run(DEMO_QUESTION, "llm")

    def test_fabricated_gnn_evidence_is_rejected(self):
        class BadScorer:
            def rank(self, question, graph):
                return [Candidate("city:sample", 1, (Triple("film:sample", "fake", "city:sample"),))]
        runner, _, _ = self.runner([query_message(), READY], scorer=BadScorer())
        with self.assertRaisesRegex(ValueError, "fabricated"):
            runner.run(DEMO_QUESTION, "gnn")

    def test_context_budget_preserves_whole_paths_and_original_direction(self):
        graph = GraphIndex(DEMO_TRIPLES).retrieve(DEMO_QUESTION)
        candidate = Candidate("city:sample", 1, (DEMO_TRIPLES[0], DEMO_TRIPLES[2]))
        small = build_context([candidate], graph, DEMO_CATALOG, 20)
        self.assertEqual(small.candidate_ids, ())
        full = build_context([candidate], graph, DEMO_CATALOG, 2000)
        self.assertEqual(full.triples, candidate.evidence)
        self.assertLessEqual(len(full.text.encode("utf-8")), 2000)

    def test_error_trace_survives_failure_without_exception_secret(self):
        class FailingProvider:
            def complete(self, *args, **kwargs):
                raise ProviderError("secret-key")
        with tempfile.TemporaryDirectory() as directory:
            runner = AgentRunner(FailingProvider(), trace_dir=Path(directory))
            question = Question("../../unsafe", "Question", ())
            with self.assertRaises(ProviderError):
                runner.run(question, "llm")
            paths = list(Path(directory).glob("*.json"))
            self.assertEqual(len(paths), 1)
            trace = json.loads(paths[0].read_text(encoding="utf-8"))
            self.assertEqual(trace["status"], "error")
            self.assertNotIn("secret-key", paths[0].read_text())

    def test_disabled_tools_skip_router(self):
        runner, provider, retriever = self.runner([answer_message(text="Unknown")],
                                                settings=AgentSettings(max_tool_calls=0))
        result = runner.run(DEMO_QUESTION, "text")
        self.assertEqual(result.tool_calls, 0)
        self.assertEqual(len(provider.requests), 1)
        self.assertFalse(retriever.calls)

    def test_invalid_settings_and_mode_fail_before_network(self):
        for kwargs in ({"max_tool_calls": -1}, {"max_hops": 3}, {"max_nodes": True}, {"context_token_budget": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AgentSettings(**kwargs)
        with self.assertRaises(ValueError):
            AgentRunner(ScriptedProvider([])).run(DEMO_QUESTION, "invalid")


if __name__ == "__main__":
    unittest.main()
