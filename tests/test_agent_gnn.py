"""C/D loading and shared-evidence tests; synthetic checkpoints are NOT results."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from graph_agent.agent.demo import DEMO_CATALOG, DEMO_QUESTION, DEMO_TRIPLES, ScriptedProvider, answer_message, query_message
from graph_agent.agent.evidence import EntityCatalog
from graph_agent.agent.gnn import CheckpointScorer
from graph_agent.agent.retrieval import GraphRetrievalTool
from graph_agent.agent.runner import AgentRunner
from graph_agent.retrieval.graph import GraphIndex
from graph_agent.schemas import Candidate, Question


class SharedEvidenceTests(unittest.TestCase):
    def test_missing_weight_never_silently_becomes_text_mode(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(FileNotFoundError):
            CheckpointScorer(Path(directory) / "best.pt")

    def test_modes_share_paths_even_if_scorer_returns_an_unrelated_shortcut(self):
        from graph_agent.schemas import Triple
        triples = (Triple("s", "birth", "city"), Triple("city", "contains", "u"), Triple("s", "student", "u"))
        question = Question("shared", "List contained locations of any birthplace.", ("s",))
        catalog = EntityCatalog(relation_names={"birth": "place of birth", "contains": "location contains", "student": "education student"})
        class Scorer:
            def rank(self, question, graph):
                self.graph = graph
                return [Candidate("u", 10, (triples[2],))]
        scorer = Scorer()
        provider = ScriptedProvider([query_message(), {"role": "assistant", "content": "Ready"},
                                     answer_message(text="Text"), answer_message(text="GNN")])
        runner = AgentRunner(provider, retriever=GraphRetrievalTool(GraphIndex(triples)), catalog=catalog, scorer=scorer)
        runner.run(question, "text")
        text_evidence = json.loads(provider.requests[-1]["messages"][1]["content"])["evidence"]
        runner.run(question, "gnn")
        gnn_evidence = json.loads(provider.requests[-1]["messages"][1]["content"])["evidence"]
        text_path = next(row["path"] for row in text_evidence["candidates"] if row["entity"]["id"] == "u")
        self.assertEqual(gnn_evidence["candidates"][0]["path"], text_path)
        self.assertEqual(scorer.graph.paths["u"], triples[:2])
        self.assertTrue(runner.last_trace["route_cache_hit"])

    def test_all_mode_cli_writes_three_rows_and_reuses_route(self):
        spec = importlib.util.spec_from_file_location("run_agent_cli", ROOT / "scripts/run_agent.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        class Scorer:
            def rank(self, question, graph):
                return [Candidate("city:sample", 1, graph.paths["city:sample"])]
        provider = ScriptedProvider([answer_message(text="LLM"), query_message(),
            {"role": "assistant", "content": "Ready"}, answer_message(ids=("city:sample",), text="Text"),
            answer_message(ids=("city:sample",), text="GNN")])
        tool = GraphRetrievalTool(GraphIndex(DEMO_TRIPLES))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions = root / "questions.jsonl"
            questions.write_text(json.dumps({"question_id": DEMO_QUESTION.question_id, "text": DEMO_QUESTION.text,
                                            "topic_entity_ids": list(DEMO_QUESTION.topic_entity_ids)}), encoding="utf-8")
            output = root / "predictions.jsonl"
            args = ["run_agent.py", "--mode", "all", "--questions", str(questions), "--output", str(output),
                    "--trace-dir", str(root / "traces")]
            with patch.object(sys, "argv", args), patch.object(cli, "CheckpointScorer", return_value=Scorer()), \
                    patch.object(cli.GraphRetrievalTool, "from_raw", return_value=(tool, DEMO_CATALOG)), \
                    patch.object(cli.LLMSettings, "from_env", return_value=object()), \
                    patch.object(cli, "LangChainClient", return_value=provider):
                cli.main()
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["mode"] for row in rows], ["llm", "text", "gnn"])
            self.assertEqual([row["status"] for row in rows], ["ok"] * 3)
            self.assertEqual(len(provider.requests), 5)


@unittest.skipUnless(importlib.util.find_spec("torch_geometric"), "GNN dependencies required")
class CheckpointLoadTests(unittest.TestCase):
    def test_real_C_scorer_loads_synthetic_fixture_and_is_deterministic_on_cpu(self):
        import torch
        from graph_agent.gnn.model import CandidateGAT, GraphEncoder, question_tokens
        # Random parameters only verify serialization/inference, never quality.
        torch.manual_seed(42)
        encoder = GraphEncoder(DEMO_CATALOG.entity_names, DEMO_CATALOG.relation_names,
                               sorted(set(question_tokens(DEMO_QUESTION.text))))
        architecture = {"hidden_dim": 8, "layers": 1, "graph_pool": "mean"}
        model = CandidateGAT(encoder, **architecture)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test-only.pt"
            torch.save({"format": 1, "architecture": architecture, "vocabulary": encoder.vocabulary(),
                        "weights": model.state_dict()}, path)
            scorer = CheckpointScorer(path)
            graph = GraphIndex(DEMO_TRIPLES).retrieve(DEMO_QUESTION)
            first = scorer.rank(DEMO_QUESTION, graph)
            second = scorer.rank(DEMO_QUESTION, graph)
            self.assertEqual(first, second)
            self.assertEqual(len(first), len(graph.node_ids) - 1)
            self.assertEqual(len(scorer.describe()["checkpoint_sha256"]), 64)
            self.assertTrue(all(c.evidence and all(t in graph.triples for t in c.evidence) for c in first))


if __name__ == "__main__":
    unittest.main()
