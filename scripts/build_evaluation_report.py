"""Create E's independent tables, figures and Chinese report from predictions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graph_agent.evaluation.reporting import ReportError, build_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--allow-mock", action="store_true",
                        help="Permit explicitly marked mock-only development reports")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    try:
        if args.bootstrap_samples < 1:
            raise ReportError("--bootstrap-samples must be positive")
        result = build_report(
            args.predictions, args.output, manifest_path=args.manifest,
            allow_mock=args.allow_mock, bootstrap_samples=args.bootstrap_samples,
            seed=args.seed, plots=not args.no_plots,
        )
    except (ReportError, OSError, ValueError) as error:
        print(f"REPORT ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2 if result["plot_error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
