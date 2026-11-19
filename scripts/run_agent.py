"""D's small development runner. E owns the formal batch evaluation entry point."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graph_agent.agent.evidence import EntityCatalog  # noqa: E402
from graph_agent.agent.gnn import CheckpointScorer  # noqa: E402
from graph_agent.agent.provider import LangChainClient, LLMSettings  # noqa: E402
from graph_agent.agent.runner import AgentRunner, AgentSettings  # noqa: E402
from graph_agent.agent.retrieval import GraphRetrievalTool  # noqa: E402
from graph_agent.data.dataset import load_questions  # noqa: E402
from graph_agent.data.loader import read_text_map  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("llm", "text", "gnn", "all"), default="llm")
    parser.add_argument("--questions", type=Path, default=ROOT / "data/processed/validation.questions.jsonl")
    parser.add_argument("--limit", type=int, default=1, help="Default 1 to keep development API usage small")
    parser.add_argument("--question-id")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/gat_2_mean/best.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--continue-on-error", action="store_true", help="Keep failed rows and continue; exits nonzero if any failed")
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--allow-inverse", action=argparse.BooleanOptionalAction, default=True,
                        help="Use B's reverse traversal (default enabled); fix across evidence modes")
    parser.add_argument("--entity-map", type=Path, help="Pure LLM: optional complete entity2text.txt, used AFTER generation")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment.json")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--trace-dir", type=Path, default=ROOT / "logs/agent")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/agent/predictions.jsonl")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    try:
        if args.output.resolve() == args.questions.resolve() or args.output.resolve().is_relative_to((ROOT / "data").resolve()):
            raise ValueError("Prediction output must not overwrite dataset files")
        questions = load_questions(args.questions)
        if args.question_id:
            questions = tuple(q for q in questions if q.question_id == args.question_id)
        questions = questions[:args.limit]
        if not questions:
            raise ValueError("No matching questions")
        config = json.loads(args.config.read_text(encoding="utf-8"))
        settings = AgentSettings(**{key: config["retrieval"][key] for key in (
            "max_hops", "max_nodes", "max_edges", "context_token_budget")},
            max_tool_calls=config["agent"]["max_tool_calls"], allow_inverse=args.allow_inverse)
        scorer = CheckpointScorer(args.checkpoint, device=args.device) if args.mode in ("gnn", "all") else None
        retriever, catalog = None, EntityCatalog()
        if args.mode != "llm":
            retriever, catalog = GraphRetrievalTool.from_raw(args.raw)
            print(f"Using B GraphIndex with names, train edges only, allow_inverse={args.allow_inverse}.", file=sys.stderr)
        elif args.entity_map:
            catalog = EntityCatalog(read_text_map(args.entity_map))
        else:
            print("No entity map: answer text/names will be logged; entity-ID metrics are unavailable.", file=sys.stderr)
        provider = LangChainClient(LLMSettings.from_env(args.env_file))
        runner = AgentRunner(provider, retriever=retriever, scorer=scorer, catalog=catalog,
                             settings=settings, trace_dir=args.trace_dir)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Append rather than destroying previous predictions. Run IDs are in traces.
        modes = ("llm", "text", "gnn") if args.mode == "all" else (args.mode,)
        failures = 0
        with args.output.open("a", encoding="utf-8") as handle:
            for question in questions:
                for mode in modes:
                    try:
                        prediction = runner.run(question, mode)
                        row = {"run_id": runner.last_trace["run_id"], "mode": mode,
                               "status": "ok", **asdict(prediction)}
                    except (RuntimeError, ValueError, OSError) as error:
                        failures += 1
                        row = {"run_id": runner.last_trace["run_id"], "mode": mode,
                               "question_id": question.question_id, "status": "error", "error_type": type(error).__name__}
                        if not args.continue_on_error:
                            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                            raise
                    line = json.dumps(row, ensure_ascii=False)
                    handle.write(line + "\n")
                    handle.flush()
                    print(line)
        print(f"Traces: {args.trace_dir}", file=sys.stderr)
        if failures:
            parser.exit(1, f"{failures} Agent rows failed; inspect error rows and traces.\n")
    except (RuntimeError, ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"Agent run failed: {error}\n")


if __name__ == "__main__":
    main()
