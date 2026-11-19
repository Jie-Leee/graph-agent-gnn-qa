"""Offline D/C acceptance: real checkpoint, scripted routing, posthoc coverage only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from accept_agent_retrieval import ContractProbeProvider  # noqa: E402
from graph_agent.agent.gnn import CheckpointScorer  # noqa: E402
from graph_agent.agent.retrieval import GraphRetrievalTool  # noqa: E402
from graph_agent.agent.runner import AgentRunner  # noqa: E402
from graph_agent.data.dataset import load_examples, load_questions  # noqa: E402
from graph_agent.evaluation.metrics import candidate_recall  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/gat_2_mean/best.pt")
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--questions", type=Path, default=ROOT / "data/processed/validation.questions.jsonl")
    parser.add_argument("--gold", type=Path, help="Loaded AFTER all inference, for coverage only")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/agent-final/gnn_acceptance.json")
    args = parser.parse_args()
    try:
        if args.limit is not None and args.limit < 1:
            raise ValueError("--limit must be positive")
        if args.output.resolve().is_relative_to((ROOT / "data").resolve()):
            raise ValueError("Output must not overwrite dataset files")
        scorer = CheckpointScorer(args.checkpoint)
        tool, catalog = GraphRetrievalTool.from_raw(args.raw)
        runner = AgentRunner(ContractProbeProvider(), retriever=tool, catalog=catalog, scorer=scorer)
        questions = load_questions(args.questions)
        if args.limit is not None:
            questions = questions[:args.limit]
        rows = []
        for question in questions:
            pair = {"question_id": question.question_id}
            routes = []
            for mode in ("text", "gnn"):
                runner.run(question, mode)
                trace = runner.last_trace
                pair[mode] = {"shared_candidate_ids": trace["shared_candidate_ids"],
                              "ranked_candidate_ids": trace["ranked_candidate_ids"],
                              "context_candidate_ids": trace["context_candidate_ids"],
                              "context_utf8_bytes": trace["context_utf8_bytes"],
                              "route_cache_hit": trace["route_cache_hit"]}
                routes.append(trace["shared_route_events"])
            if routes[0] != routes[1] or set(pair["text"]["shared_candidate_ids"]) != set(pair["gnn"]["shared_candidate_ids"]):
                raise ValueError("Evidence modes did not share routing/candidate preparation")
            if set(pair["text"]["ranked_candidate_ids"]) != set(pair["gnn"]["ranked_candidate_ids"]):
                raise ValueError("The trained GNN did not score the shared candidate set")
            rows.append(pair)
        if not rows:
            raise ValueError("No questions")
        report = {"kind": "offline_real_GNN_with_scripted_LLM_not_QA_accuracy", "questions": len(rows),
                  "scorer": scorer.describe(), "retriever": tool.describe(),
                  "shared_route_and_candidate_contract": "passed", "rows": rows}
        if args.gold:
            gold = {e.question.question_id: e.answer_ids for e in load_examples(args.gold)}
            report["context_recall_macro"] = {mode: sum(candidate_recall(row[mode]["context_candidate_ids"], gold[row["question_id"]])
                for row in rows) / len(rows) for mode in ("text", "gnn")}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False, indent=2))
    except (RuntimeError, ValueError, OSError, KeyError) as error:
        parser.exit(1, f"D/C acceptance failed: {error}\n")


if __name__ == "__main__":
    main()
