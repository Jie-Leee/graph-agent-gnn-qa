"""Independently verify an A handoff against raw graph and manifest hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graph_agent.data.dataset import load_examples, load_questions  # noqa: E402
from graph_agent.data.loader import load_raw  # noqa: E402
from graph_agent.data.split import SPLIT_ORDER, audit_records  # noqa: E402


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _digest_chunks(handle)


def _digest_chunks(handle) -> str:
    checksum = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1 << 20), b""):
        checksum.update(chunk)
    return checksum.hexdigest()


def child(directory: Path, name: str) -> Path:
    path = (directory / name).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError(f"Manifest path escapes its directory: {name}")
    return path


def check(raw: Path, processed: Path, *, retrieval_diagnostic: bool = False) -> dict:
    manifest = json.loads((processed / "manifest.json").read_text(encoding="utf-8"))
    for name, info in manifest["raw_inputs"].items():
        if digest(child(raw, info["file"])) != info["sha256"]:
            raise ValueError(f"Raw checksum mismatch: {name}")
    for name, info in manifest["files"].items():
        if digest(child(processed, name)) != info["sha256"]:
            raise ValueError(f"Generated checksum mismatch: {name}")
    for name, expected in manifest["code_sha256"].items():
        source = child(ROOT / "src" / "graph_agent" / "data", name).read_bytes()
        if hashlib.sha256(source.replace(b"\r\n", b"\n")).hexdigest() != expected:
            raise ValueError(f"Data code changed since preparation: {name}; regenerate the data")

    records = {}
    examples_by_split = {}
    questions_by_split = {}
    public_fields = {"question_id", "text", "topic_entity_ids"}
    for split in SPLIT_ORDER:
        full_path = processed / f"{split}.jsonl"
        public_path = processed / f"{split}.questions.jsonl"
        examples = load_examples(full_path)
        questions = load_questions(public_path)
        if tuple(example.question for example in examples) != questions:
            raise ValueError(f"{split}: inference inputs do not match gold records")
        if len(examples) != manifest["targets"][split]:
            raise ValueError(f"{split}: target count mismatch")
        records[split] = [json.loads(row) for row in full_path.read_text(encoding="utf-8").splitlines() if row.strip()]
        for row in public_path.read_text(encoding="utf-8").splitlines():
            if row.strip() and set(json.loads(row)) != public_fields:
                raise ValueError(f"{split}: inference file includes missing or forbidden fields")
        examples_by_split[split] = examples
        questions_by_split[split] = questions

    graph = load_raw(raw)
    audit = audit_records(records, train_triples=graph.train_triples)
    if audit["violations"]:
        raise ValueError("Audit failed: " + "; ".join(audit["violations"]))
    result = {"items": {name: len(rows) for name, rows in records.items()}, "audit": audit}
    if retrieval_diagnostic:
        # Diagnostic only: the scaffold BFS sees Question, never gold fields.
        from graph_agent.retrieval.graph import GraphIndex

        index = GraphIndex(graph.train_triples)
        recalls = []
        for question, example in zip(questions_by_split["test"], examples_by_split["test"]):
            subgraph = index.retrieve(question, max_hops=2, max_nodes=200, max_edges=800)
            recalls.append(len(set(example.answer_ids) & set(subgraph.node_ids)) / len(example.answer_ids))
        result["scaffold_bfs_diagnostic"] = {
            "description": "test gold-node recall at 2 hops / 200 nodes / 800 edges; not QA accuracy",
            "mean_gold_node_recall": round(sum(recalls) / max(len(recalls), 1), 6),
            "all_gold_nodes_present_fraction": round(sum(r == 1 for r in recalls) / max(len(recalls), 1), 6),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--processed", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--retrieval-diagnostic", action="store_true")
    args = parser.parse_args()
    try:
        result = check(args.raw, args.processed, retrieval_diagnostic=args.retrieval_diagnostic)
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        parser.exit(1, f"Data verification failed: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
