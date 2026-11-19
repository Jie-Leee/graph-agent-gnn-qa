"""Supplementary no-tool LLM prompt sensitivity check; separate from the formal run."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


PROMPT = """Answer the question using your own knowledge. No knowledge graph,
evidence, or retrieval tools will be provided. Give your best plausible answer;
if you know at least one likely answer, return the ones you know even if the list
may be incomplete. If you have no plausible answer, leave the lists empty and
explain the uncertainty. Use exact entity names unless you know their canonical
entity IDs. Never invent IDs, and leave cited_evidence empty.
Return only a JSON object with these four fields:
{"answer_ids": [], "answer_names": [], "answer_text": "concise answer or uncertainty", "cited_evidence": []}
"""
PROTOCOL = "supplemental-pure-llm-prompt-v1"


def replace_answer_prompt(messages, tools, stage):
    if tools or stage != "answer" or [m.get("role") for m in messages] != ["system", "user"]:
        raise ValueError("Prompt sensitivity is limited to one no-tool answer call")
    payload = json.loads(messages[1]["content"])
    if not isinstance(payload, dict) or set(payload) != {"question"} or not isinstance(payload["question"], str):
        raise ValueError("Prompt sensitivity accepts question text only")
    return [{"role": "system", "content": PROMPT}, messages[1]]


def run(root, split, limit, workers, resume, tag):
    sys.path.insert(0, str(root / "src"))
    from graph_agent.agent.provider import LangChainClient
    from graph_agent.agent.runner import AgentRunner
    from graph_agent.evaluation.runner import EvaluationOptions, LiveEngine, _atomic_json, run_evaluation

    class SensitivityAgent(AgentRunner):
        def _complete(self, messages, tools, trace, stage):
            # Change the actual prompt before the normal logger and provider see it.
            return super()._complete(replace_answer_prompt(messages, tools, stage), tools, trace, stage)

    class SensitivityEngine(LiveEngine):
        def _initialize(self):
            if self.runner is None:
                self._prepare_resources()
                retriever, catalog, settings = self.resources
                if retriever is not None or self.options.variants != ("llm",):
                    raise ValueError("Prompt sensitivity must have no retriever and exactly one variant")
                self.runner = SensitivityAgent(LangChainClient(self.llm), catalog=catalog, settings=settings)

        def identity(self):
            return {
                **super().identity(),
                "supplemental_protocol": PROTOCOL,
                "supplemental_system_prompt": PROMPT,
                "supplemental_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            }

    config = root / "configs" / "experiment.json"
    output = root / "outputs" / "evaluation" / f"llm-prompt-sensitivity-{split}-{limit}-{tag}"
    options = EvaluationOptions(
        data=root / "data" / "processed", raw=root / "data" / "raw", config=config, env=root / ".env",
        output=output, split=split, limit=limit, variants=("llm",), backend="live",
        resume=resume, repo_root=root, workers=workers,
    )
    engine = SensitivityEngine(options, json.loads(config.read_text(encoding="utf-8-sig")))
    summary = run_evaluation(options, engine=engine)
    for name in ("manifest.json", "summary.json"):
        path = output / name
        record = json.loads(path.read_text(encoding="utf-8"))
        record.update(formal_results=False, result_kind="supplemental_prompt_sensitivity",
                      supplemental_protocol=PROTOCOL)
        _atomic_json(path, record)
    item = summary["variants"]["llm"]
    print(json.dumps({
        "result_kind": "supplemental_prompt_sensitivity",
        "split": split,
        "output": str(output),
        "selected_questions": summary["selected_question_count"],
        "successes": item["success_count"], "failures": item["failure_count"],
        "exact_match": item["exact_match"], "f1": item["f1"], "answer_rate": item["answer_rate"],
        "served_models": item["served_models"],
    }, ensure_ascii=False, indent=2))
    return 0 if summary["complete"] and item["success_count"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tag", default="v1", help="Label for a distinct supplemental output directory")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.limit < 1 or args.workers < 1 or not args.tag.isascii() or not args.tag.isalnum():
        parser.error("limit/workers must be positive and tag must be ASCII letters or digits")
    try:
        return run(args.repo.resolve(), args.split, args.limit, args.workers, args.resume, args.tag)
    except Exception as error:
        parser.exit(1, f"Prompt sensitivity failed: {type(error).__name__}; see local outputs if created.\n")


if __name__ == "__main__":
    raise SystemExit(main())
