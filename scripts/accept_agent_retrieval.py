"""D/B offline contract acceptance on real questions; scripted LLM, not QA accuracy."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graph_agent.agent.demo import answer_message, query_message  # noqa: E402
from graph_agent.agent.provider import Completion  # noqa: E402
from graph_agent.agent.retrieval import GraphRetrievalTool  # noqa: E402
from graph_agent.agent.runner import AgentRunner, AgentSettings  # noqa: E402
from graph_agent.data.dataset import load_examples, load_questions  # noqa: E402
from graph_agent.evaluation.metrics import candidate_recall  # noqa: E402


class ContractProbeProvider:
    """Always requests 2 hops, then abstains. No model routing or answer claims."""
    def complete(self, messages, *, tools=None):
        if tools:
            return Completion(query_message() if messages[-1]["role"] == "user" else
                              {"role": "assistant", "content": "Retrieval ready (scripted probe)."})
        return Completion(answer_message(text="Offline contract probe; no model answer generated."))


def accept(runner: AgentRunner, questions) -> dict:
    rows = []
    for question in questions:
        prediction = runner.run(question, "text")
        trace = runner.last_trace
        rows.append({"question_id": question.question_id, "status": trace["status"],
                     "tool_calls": prediction.tool_calls,
                     "retrieved_node_ids": trace.get("retrieved_node_ids", []),
                     "ranked_candidate_ids": trace.get("ranked_candidate_ids", []),
                     "context_candidate_ids": trace["context_candidate_ids"],
                     "context_utf8_bytes": trace["context_utf8_bytes"],
                     "retrieval_stats": [e["stats"] for e in trace["events"] if e["stage"] == "retrieval"]})
    if not rows:
        raise ValueError("Cannot accept an empty question set")
    return {"kind": "offline_D_B_contract_with_scripted_provider", "not_model_QA_accuracy": True,
            "question_count": len(rows), "settings": asdict(runner.settings),
            "context_format": "compact_paths_v2",
            "evidence_code_sha256_lf": hashlib.sha256((ROOT / "src/graph_agent/agent/evidence.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
            "retriever": runner.retriever.describe(),
            "mean_context_candidates": sum(len(row["context_candidate_ids"]) for row in rows) / len(rows),
            "all_contexts_within_byte_budget": all(row["context_utf8_bytes"] <= runner.settings.context_token_budget for row in rows),
            "rows": rows}


def add_posthoc_coverage(report: dict, gold_path: Path) -> None:
    """Evaluation only, after inference has finished. Gold never enters runner."""
    gold = {example.question.question_id: example for example in load_examples(gold_path)}
    coverage = {"retrieved_node_ids": [], "ranked_candidate_ids": [], "context_candidate_ids": []}
    for row in report["rows"]:
        if row["question_id"] not in gold:
            raise ValueError("Gold file does not cover the accepted questions")
        answers = gold[row["question_id"]].answer_ids
        for field in coverage:
            coverage[field].append(candidate_recall(row[field], answers))
    report["posthoc_coverage_macro"] = {name: sum(values) / len(values) for name, values in coverage.items()}
    report["posthoc_note"] = "Gold loaded only after all Agent runs; coverage diagnostics, not answer accuracy."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--questions", type=Path, default=ROOT / "data/processed/validation.questions.jsonl")
    parser.add_argument("--gold", type=Path, help="Optional post-inference coverage check only")
    parser.add_argument("--limit", type=int, help="Default all validation questions")
    parser.add_argument("--allow-inverse", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/agent-b-integration/offline_acceptance.json")
    args = parser.parse_args()
    try:
        if args.limit is not None and args.limit < 1:
            raise ValueError("--limit must be positive")
        if args.output.resolve().is_relative_to((ROOT / "data").resolve()):
            raise ValueError("Acceptance output must not overwrite data files")
        tool, catalog = GraphRetrievalTool.from_raw(args.raw)
        questions = load_questions(args.questions)
        if args.limit is not None:
            questions = questions[:args.limit]
        runner = AgentRunner(ContractProbeProvider(), retriever=tool, catalog=catalog,
                             settings=AgentSettings(allow_inverse=args.allow_inverse))
        report = accept(runner, questions)
        report["questions_sha256"] = hashlib.sha256(args.questions.read_bytes()).hexdigest()
        if args.gold:
            add_posthoc_coverage(report, args.gold)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in report.items() if key != "rows"}, ensure_ascii=False, indent=2))
        print(f"Report: {args.output}")
    except (RuntimeError, ValueError, OSError) as error:
        parser.exit(1, f"Agent/retrieval acceptance failed: {error}\n")


if __name__ == "__main__":
    main()
