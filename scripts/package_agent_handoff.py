"""Package D's source, verified C main checkpoint, reports and replay; exclude keys."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/agent-final/Agent-D-权重联调完成.zip")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".zip" or not args.output.resolve().is_relative_to((ROOT / "outputs").resolve()):
        parser.error("Package output must be a .zip inside the project's outputs directory")
    secrets = []
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            if "=" in line:
                name, value = line.split("=", 1)
                if name.strip() in ("LLM_API_KEY", "DEEPSEEK_API_KEY") and value.strip():
                    secrets.append(value.strip().strip("\"'").encode())
    files = {}

    def add(path: Path, name: str | None = None):
        path = ROOT / path
        data = path.read_bytes()
        if any(secret in data for secret in secrets) or re.search(rb"sk-[A-Za-z0-9_-]{24,}", data):
            raise ValueError("Potential credential found; packaging aborted")
        files[name or path.relative_to(ROOT).as_posix()] = data

    for folder in ("src", "scripts"):
        for path in (ROOT / folder).rglob("*"):
            if path.suffix in (".py", ".cjs") and "__pycache__" not in path.parts:
                add(path)
    for path in (ROOT / "tests").glob("test_agent*.py"):
        add(path)
    for name in (".env.example", "README.md", "pyproject.toml", "requirements.txt", "requirements-agent.txt",
                 "requirements-agent-gnn.txt", "requirements-gnn.txt", "configs/experiment.json",
                 "docs/C-GNN交接.md", "docs/agent_routing_cases.json", "checkpoints/gat_2_mean/best.pt"):
        add(Path(name))
    for path in (ROOT / "docs").glob("D*.md"):
        add(path)
    for path in (ROOT / "docs/agent").glob("*"):
        if path.suffix in (".md", ".pptx"):
            add(path)
    for name in ("routing.json", "routing_initial_protocol.json", "three_modes.jsonl",
                 "three_mode_smoke_summary.json", "gnn_acceptance.json"):
        add(Path("outputs/agent-final") / name, "reports/" + name)
    for directory in ("three-modes", "routing-v2"):
        for path in (ROOT / "logs/agent-final" / directory).glob("*.json"):
            add(path, f"reports/traces/{directory}/{path.name}")
    for name in ("Agent三模式真实日志回放.html", "Agent三模式备用演示.webm"):
        add(Path("outputs/agent-final/deliverables") / name, "deliverables/" + name)
    files["README-D.txt"] = (
        "D Agent handoff after real C checkpoint integration.\n"
        "Merge these source paths into the existing A/B/C project checkout.\n"
        "Use Anaconda langchain and your OWN local .env key.\n"
        "See docs/D-Agent完成说明.md for verified results and limits.\n"
        "Includes C's published gat_2_mean checkpoint; other ablations remain in Git.\n"
        "Partial local training, raw data and credentials are excluded.\n"
        "Replay/video use saved REAL logs, not live API calls.\n"
        "200-question coverage and first-5-question QA are development checks, not a final benchmark.\n"
    ).encode("utf-8")
    files["MANIFEST.sha256.json"] = json.dumps(
        {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
        ensure_ascii=False, indent=2).encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            if name == ".env" or "gnn-reproduced" in name:
                raise ValueError("Private/intermediate artifact is not a handoff input")
            archive.writestr(name, data)
    with zipfile.ZipFile(args.output) as archive:
        if archive.testzip() is not None:
            raise ValueError("Corrupt handoff archive")
    print(json.dumps({"package": str(args.output), "files": len(files), "bytes": args.output.stat().st_size,
                      "checkpoint_sha256": hashlib.sha256(files["checkpoints/gat_2_mean/best.pt"]).hexdigest(),
                      "credential_check": "passed"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
