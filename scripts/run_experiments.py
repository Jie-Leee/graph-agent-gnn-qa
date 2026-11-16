"""Reserved for E. Refuse to generate fabricated experiment results."""
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal evaluation - scaffold only")
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    raise SystemExit(
        "NOT IMPLEMENTED: connect real QA data, the Agent and trained GNN first. "
        f"Requested modes: {config['modes']}. No experiment results were written."
    )


if __name__ == "__main__":
    main()
