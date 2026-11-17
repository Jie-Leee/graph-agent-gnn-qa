"""Separate, synthetic routing checks; never part of FB15k-237 QA scores."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graph_agent.agent.demo import DEMO_CATALOG, DEMO_TRIPLES  # noqa: E402
from graph_agent.agent.provider import LangChainClient, LLMSettings  # noqa: E402
from graph_agent.agent.runner import AgentRunner  # noqa: E402
from graph_agent.retrieval.graph import GraphIndex  # noqa: E402
from graph_agent.schemas import Question  # noqa: E402


def load_cases(path: Path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Routing cases must be a nonempty array")
    seen = set()
    result = []
    for row in rows:
        if (not isinstance(row, dict) or set(row) != {
            "question_id", "text", "topic_entity_ids", "expected_query"
        } or type(row["expected_query"]) is not bool):
            raise ValueError("Invalid routing case fields")
        if any(not isinstance(row[key], str) or not row[key].strip() for key in ("question_id", "text")):
            raise ValueError("Routing case needs nonempty ID and text")
        if not isinstance(row["topic_entity_ids"], list) or not all(
            isinstance(v, str) and v for v in row["topic_entity_ids"]
        ) or row["question_id"] in seen:
            raise ValueError("Invalid topics or duplicate routing question ID")
        seen.add(row["question_id"])
        # Only these fields cross into inference; expected_query stays here.
        result.append((Question(row["question_id"], row["text"], tuple(row["topic_entity_ids"])), row["expected_query"]))
    return result


def run_case(runner: AgentRunner, question: Question, expected: bool) -> dict:
    try:
        prediction = runner.run(question, "text")
        queried = prediction.tool_calls > 0
        return {"question_id": question.question_id, "status": "ok", "queried": queried,
                "expected_query": expected, "agreement": queried == expected,
                "tool_calls": prediction.tool_calls, "run_id": runner.last_trace["run_id"]}
    except (RuntimeError, ValueError, OSError) as error:
        return {"question_id": question.question_id, "status": "error", "error_type": type(error).__name__,
                "expected_query": expected, "agreement": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "docs/agent_routing_cases.json")
    parser.add_argument("--limit", type=int, default=1, help="Use 24 for the complete synthetic set")
    parser.add_argument("--dry-run", action="store_true", help="Validate cases only; no API calls")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--trace-dir", type=Path, default=ROOT / "logs/agent-routing")
    args = parser.parse_args()
    try:
        cases = load_cases(args.cases)
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        if args.dry_run:
            print(json.dumps({"status": "validated_without_model_calls", "cases": len(cases),
                              "query_reference": sum(expected for _, expected in cases),
                              "skip_reference": sum(not expected for _, expected in cases)}))
            return
        runner = AgentRunner(LangChainClient(LLMSettings.from_env(args.env_file)),
                             retriever=GraphIndex(DEMO_TRIPLES), catalog=DEMO_CATALOG,
                             trace_dir=args.trace_dir)
        rows = [run_case(runner, question, expected) for question, expected in cases[:args.limit]]
        for row in rows:
            print(json.dumps(row, ensure_ascii=False))
        print(json.dumps({"kind": "synthetic_routing_only_not_QA", "cases": len(rows),
                          "errors": sum(row["status"] == "error" for row in rows),
                          "agreement_including_errors_as_failures": sum(row["agreement"] for row in rows) / len(rows)}))
    except (RuntimeError, ValueError, OSError) as error:
        parser.exit(1, f"Routing checks failed: {error}\n")


if __name__ == "__main__":
    main()
