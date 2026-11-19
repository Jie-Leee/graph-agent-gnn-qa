"""E's formal batch evaluation entry point; defaults to five validation questions."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from graph_agent.evaluation.runner import EvaluationOptions, VARIANTS, run_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed", help="Processed directory or explicit gold JSONL")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment.json")
    parser.add_argument("--env", "--env-file", dest="env", type=Path, default=ROOT / ".env")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--limit", type=int, default=5, help="Positive number; test is never selected implicitly")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/evaluation")
    parser.add_argument("--variants", nargs="+", default=["llm", "text", "gat_2_mean"],
                        help="Space/comma separated: " + ",".join(VARIANTS))
    parser.add_argument("--resume", action="store_true", help="Requires identical data/config/model/weights/source/git fingerprints")
    parser.add_argument("--backend", choices=("live", "mock"), default="live")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    variants = tuple(part for item in args.variants for part in item.split(",") if part)
    options = EvaluationOptions(data=args.data, raw=args.raw, config=args.config, env=args.env,
        output=args.output, split=args.split, limit=args.limit, variants=variants,
        backend=args.backend, resume=args.resume, device=args.device, repo_root=ROOT)
    try:
        summary = run_evaluation(options)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"Evaluation preflight/output failed: {error}\n")
    except Exception as error:
        parser.exit(1, f"Evaluation failed: {type(error).__name__}; no provider error details were printed.\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    failures = sum(v["failure_count"] for v in summary["variants"].values())
    if failures or not summary["complete"]:
        parser.exit(1, f"Evaluation retained {failures} failed predictions; inspect predictions.jsonl and traces.\n")


if __name__ == "__main__":
    main()
