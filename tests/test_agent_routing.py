from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("agent_routing_checks", ROOT / "scripts/check_agent_routing.py")
routing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(routing)

from graph_agent.agent.demo import DEMO_CATALOG, DEMO_TRIPLES, ScriptedProvider, answer_message, query_message
from graph_agent.agent.provider import Completion
from graph_agent.agent.runner import AgentRunner
from graph_agent.retrieval.graph import GraphIndex


class RoutingTests(unittest.TestCase):
    def test_malformed_stop_structure_is_a_bounded_Agent_error(self):
        from graph_agent.agent.runner import AgentError
        provider = ScriptedProvider([{"role": "assistant", "content": None,
                                     "tool_calls": [{"id": "bad", "type": "function", "function": None}]}])
        runner = AgentRunner(provider, retriever=GraphIndex(DEMO_TRIPLES))
        question, _ = routing.load_cases(ROOT / "docs/agent_routing_cases.json")[0]
        with self.assertRaises(AgentError):
            runner.run(question, "text")

    def test_declared_finish_action_stops_without_graph_query(self):
        finish = {"role": "assistant", "content": None, "tool_calls": [{"id": "stop-1", "type": "function",
                  "function": {"name": "finish", "arguments": "{}"}}]}
        provider = ScriptedProvider([finish, answer_message(text="120 minutes")])
        runner = AgentRunner(provider, retriever=GraphIndex(DEMO_TRIPLES), catalog=DEMO_CATALOG)
        question, _ = routing.load_cases(ROOT / "docs/agent_routing_cases.json")[16]
        result = runner.run(question, "text")
        self.assertEqual(result.tool_calls, 0)
        self.assertIn("finish", [tool["function"]["name"] for tool in provider.requests[0]["tools"]])
        self.assertEqual(runner.last_trace["events"][1]["stage"], "route_stop")

    def test_all_24_reference_labels_stay_outside_actual_inference(self):
        cases = routing.load_cases(ROOT / "docs/agent_routing_cases.json")
        self.assertEqual(len(cases), 24)
        self.assertEqual(sum(expected for _, expected in cases), 12)
        provider = ScriptedProvider([
            message for _ in cases for message in (
                {"role": "assistant", "content": "No query needed"}, answer_message(text="Unknown"))
        ])
        runner = AgentRunner(provider, retriever=GraphIndex(DEMO_TRIPLES), catalog=DEMO_CATALOG)
        rows = [routing.run_case(runner, question, expected) for question, expected in cases]
        self.assertEqual(sum(row["agreement"] for row in rows), 12)
        self.assertNotIn("expected_query", json.dumps(provider.requests))

    def test_shared_cost_is_in_predictions_but_cached_route_is_not_double_billed(self):
        class UsageProvider(ScriptedProvider):
            def complete(self, messages, *, tools=None):
                result = super().complete(messages, tools=tools)
                return Completion(result.message, 10, 2, True)

        class Scorer:
            def rank(self, question, graph):
                return []

        provider = UsageProvider([query_message(), {"role": "assistant", "content": "Ready"},
                                  answer_message(text="Unknown"), answer_message(text="Unknown")])
        runner = AgentRunner(provider, retriever=GraphIndex(DEMO_TRIPLES), catalog=DEMO_CATALOG, scorer=Scorer())
        question, _ = routing.load_cases(ROOT / "docs/agent_routing_cases.json")[0]
        first = runner.run(question, "text")
        second = runner.run(question, "gnn")
        self.assertEqual((first.input_tokens, first.output_tokens), (30, 6))
        self.assertEqual((second.input_tokens, second.output_tokens), (30, 6))
        self.assertEqual(runner.last_trace["actual_input_tokens"], 10)
        self.assertEqual(runner.last_trace["actual_output_tokens"], 2)
        self.assertTrue(runner.last_trace["usage_reported"])


if __name__ == "__main__":
    unittest.main()
