"""Two fictional examples; --offline uses a scripted test double, no API calls."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graph_agent.agent.demo import (  # noqa: E402
    DEMO_CATALOG, DEMO_QUESTION, DEMO_TRIPLES, SKIP_QUESTION, offline_provider,
)
from graph_agent.agent.provider import LangChainClient, LLMSettings  # noqa: E402
from graph_agent.agent.runner import AgentRunner  # noqa: E402
from graph_agent.retrieval.graph import GraphIndex  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Scripted responses; not a real Agent routing experiment")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--trace-dir", type=Path, default=ROOT / "logs" / "agent-demo")
    args = parser.parse_args()
    try:
        provider = offline_provider() if args.offline else LangChainClient(LLMSettings.from_env(args.env_file))
        runner = AgentRunner(provider, retriever=GraphIndex(DEMO_TRIPLES),
                             catalog=DEMO_CATALOG, trace_dir=args.trace_dir)
        print("SCRIPTED OFFLINE DEMO (no LLM decisions)" if args.offline else "LIVE LLM DEMO (fictional graph, not formal QA)")
        for question in (DEMO_QUESTION, SKIP_QUESTION):
            print(json.dumps(asdict(runner.run(question, "text")), ensure_ascii=False, indent=2))
        print(f"Traces: {args.trace_dir}")
    except (RuntimeError, ValueError, OSError) as error:
        parser.exit(1, f"Agent demo failed: {error}\n")


if __name__ == "__main__":
    main()
