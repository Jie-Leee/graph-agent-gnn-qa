"""A-side executable handoff checks; downstream/team acceptance is separate.

No LLM is called. D's recording consumer is an explicit contract probe, not an
Agent implementation. Gold is compared by the evaluator only after retrieval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graph_agent.agent.runner import AgentRunner  # noqa: E402
from graph_agent.data.dataset import load_examples, load_questions  # noqa: E402
from graph_agent.data.loader import load_raw  # noqa: E402
from graph_agent.retrieval.graph import GraphIndex  # noqa: E402
from graph_agent.schemas import Question, Subgraph  # noqa: E402

PUBLIC_FIELDS = {"question_id", "text", "topic_entity_ids"}
SPLITS = ("train", "validation", "test")
# Only this already-inspected NotImplemented scaffold may be invoked by the
# default check. A later D implementation can perform network/API calls and
# must be tested deliberately by its owner, outside this A-only probe.
AGENT_SOURCE = ROOT / "src" / "graph_agent" / "agent" / "runner.py"
INSPECTED_AGENT_SCAFFOLD_SHA256_LF = "10d5006ca663c3256b6f3f99f1831bec47706bef3d41f73ded94a9c063a33951"


class HandoffError(ValueError):
    """An export or receiving contract did not meet the A handoff rules."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def question_payload(question: Question) -> dict:
    """Serialize only the exact shared Question contract, never QAExample."""
    if type(question) is not Question:
        raise HandoffError("Inference receiver requires an exact Question object")
    payload = json.loads(json.dumps(asdict(question), ensure_ascii=False))
    if set(payload) != PUBLIC_FIELDS:
        raise HandoffError("Question serialization contains forbidden or missing fields")
    return payload


def contract_probe(questions, consumer=None) -> dict:
    """Exercise a recording consumer; this does not exercise D's LLM prompt."""
    recorded = []

    def recording_consumer(question: Question) -> dict:
        payload = question_payload(question)
        recorded.append(payload)
        return payload

    receiver = recording_consumer if consumer is None else consumer
    count = 0
    for question in questions:
        expected = question_payload(question)
        actual = receiver(question)
        if not isinstance(actual, dict) or set(actual) != PUBLIC_FIELDS or actual != expected:
            raise HandoffError("Contract probe consumer serialized extra gold fields or changed Question inputs")
        count += 1
    return {
        "status": "passed",
        "kind": "recording_consumer_contract_probe",
        "question_count": count,
        "serialized_fields": sorted(PUBLIC_FIELDS),
        "gold_fields_forwarded": False,
        "not_an_actual_agent_or_llm_run": True,
    }


def poison_gold_probe(questions) -> dict:
    """Arbitrary supervised values must not alter exported Question objects."""
    expected = tuple(questions)
    with tempfile.TemporaryDirectory(prefix="a-handoff-") as directory:
        source = Path(directory) / "poisoned_supervision.jsonl"
        with source.open("w", encoding="utf-8", newline="\n") as handle:
            for question in expected:
                item = question_payload(question)
                item.update({
                    "answer_ids": "__GOLD_ANSWER_POISON__",
                    "relation_path": {"__GOLD_PATH_POISON__": True},
                    "evidence": "__GOLD_EVIDENCE_POISON__",
                    "extra_supervision": ["__EXTRA_POISON__"],
                })
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        reloaded = load_questions(source)
    if reloaded != expected:
        raise HandoffError("Changing gold supervision changed inference inputs")
    return {
        "status": "passed",
        "question_count": len(expected),
        "gold_values_replaced_with_invalid_sentinels": True,
        "question_objects_unchanged": True,
    }


def downstream_agent_status(question: Question, agent_factory=AgentRunner) -> dict:
    """Report the real implementation status without inventing D acceptance."""
    if agent_factory is AgentRunner and hashlib.sha256(
        AGENT_SOURCE.read_bytes().replace(b"\r\n", b"\n")
    ).hexdigest() != INSPECTED_AGENT_SCAFFOLD_SHA256_LF:
        return {
            "status": "requires_downstream_review",
            "reason": "D module changed from the inspected scaffold; its run method was not invoked",
            "actual_llm_or_tool_loop_verified": False,
        }
    try:
        prediction = agent_factory().run(question, "llm")
    except NotImplementedError:
        return {
            "status": "pending_downstream_implementation",
            "reason": "AgentRunner.run currently raises NotImplementedError",
            "actual_llm_or_tool_loop_verified": False,
        }
    return {
        "status": "requires_downstream_review",
        "reason": "AgentRunner returned; this probe alone does not audit its prompt/tool-loop implementation",
        "returned_type": type(prediction).__name__,
        "actual_llm_or_tool_loop_verified": False,
    }


def _safe_child(directory: Path, relative: str) -> Path:
    candidate = (directory / relative).resolve()
    if not candidate.is_relative_to(directory.resolve()):
        raise HandoffError(f"Manifest path escapes data directory: {relative}")
    return candidate


def run_acceptance(raw_dir: Path, processed_dir: Path, *, consumer=None,
                   graph_index_factory=GraphIndex, agent_factory=AgentRunner) -> dict:
    raw_dir, processed_dir = Path(raw_dir), Path(processed_dir)
    manifest = json.loads((processed_dir / "manifest.json").read_text(encoding="utf-8"))
    for relative, entry in manifest["files"].items():
        if sha256(_safe_child(processed_dir, relative)) != entry["sha256"]:
            raise HandoffError(f"Processed file checksum mismatch: {relative}")
    for name, entry in manifest["raw_inputs"].items():
        if sha256(_safe_child(raw_dir, entry["file"])) != entry["sha256"]:
            raise HandoffError(f"Raw input checksum mismatch: {name}")
    for name, expected in manifest["code_sha256"].items():
        source = _safe_child(ROOT / "src" / "graph_agent" / "data", name).read_bytes()
        if hashlib.sha256(source.replace(b"\r\n", b"\n")).hexdigest() != expected:
            raise HandoffError(f"Data source changed since dataset preparation: {name}")

    examples = {}
    inputs = {}
    for split in SPLITS:
        examples[split] = load_examples(processed_dir / f"{split}.jsonl")
        inputs[split] = load_questions(processed_dir / f"{split}.questions.jsonl")
        if len(examples[split]) != manifest["targets"][split]:
            raise HandoffError(f"Export count mismatch: {split}")
        if tuple(example.question for example in examples[split]) != inputs[split]:
            raise HandoffError(f"Question and supervision exports differ: {split}")
        public_path = processed_dir / f"{split}.questions.jsonl"
        for line in public_path.read_text(encoding="utf-8").splitlines():
            if line.strip() and set(json.loads(line)) != PUBLIC_FIELDS:
                raise HandoffError(f"Public export contains extra or missing fields: {split}")
    questions = inputs["test"]
    if not questions:
        raise HandoffError("Acceptance needs at least one test Question")

    graph = load_raw(raw_dir)
    visible = set(graph.train_triples)
    heldout_exclusive = (set(graph.triples["valid"]) | set(graph.triples["test"])) - visible
    index = graph_index_factory(graph.train_triples)
    indexed = {triple for edges in index.outgoing.values() for triple in edges}
    if indexed != visible:
        raise HandoffError("Retrieval index is not exactly the visible training graph")
    if indexed & heldout_exclusive:
        raise HandoffError("Heldout-exclusive graph edges entered the retrieval index")

    retrieved = []
    for question in questions:
        question_payload(question)
        subgraph = index.retrieve(question, max_hops=2, max_nodes=200, max_edges=800)
        if not isinstance(subgraph, Subgraph):
            raise HandoffError("B scaffold did not return a Subgraph")
        if not set(subgraph.triples) <= visible:
            raise HandoffError("Retrieved evidence includes edges outside the visible graph")
        retrieved.append(subgraph)
    # Gold is joined only here, after every retrieval call has finished.
    recalls = [
        len(set(example.answer_ids) & set(subgraph.node_ids)) / len(example.answer_ids)
        for example, subgraph in zip(examples["test"], retrieved)
    ]
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    return {
        "report_kind": "A_side_export_and_contract_acceptance",
        "dataset_manifest_sha256": sha256(processed_dir / "manifest.json"),
        "dataset_file_sha256": {
            name: entry["sha256"] for name, entry in manifest["files"].items()
            if name.endswith(".jsonl")
        },
        "repository_revision_at_execution": revision,
        "acceptance_script_sha256_lf": hashlib.sha256(
            Path(__file__).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest(),
        "A_export": {
            "status": "passed",
            "split_items": {split: len(examples[split]) for split in SPLITS},
            "public_exports_match_supervised_questions": True,
            "C_training_QAExample_readable": True,
            "E_gold_QAExample_readable": True,
            "known_topic_ids_provided": True,
        },
        "B_scaffold": {
            "status": "compatible",
            "indexed_training_edges": len(indexed),
            "heldout_exclusive_edges": len(heldout_exclusive),
            "heldout_exclusive_edges_indexed": 0,
            "questions_retrieved": len(questions),
            "all_returned_evidence_edges_in_train": True,
            "retrieval_input_type": "Question",
            "gold_joined_only_after_retrieval": True,
            "diagnostic": {
                "max_hops": 2, "max_nodes": 200, "max_edges": 800,
                "mean_gold_node_recall": round(sum(recalls) / len(recalls), 6),
                "all_gold_nodes_present_fraction": round(sum(value == 1 for value in recalls) / len(recalls), 6),
                "is_QA_accuracy": False,
            },
            "actual_B_implementation_acceptance": "pending_teammate_confirmation",
        },
        "D_contract_probe": contract_probe(questions, consumer),
        "gold_poison_probe": poison_gold_probe(questions),
        "D_actual_integration": downstream_agent_status(questions[0], agent_factory),
        "human_acceptance": {
            "status": "pending_no_confirmation",
            "reason": "No confirmation has been received from B, C, D or E; no teammate was contacted",
        },
        "scope": (
            "A exports and current scaffold compatibility only. No actual LLM prompt or tool loop was verified. "
            "Gold supervision fields must not be forwarded; evidence retrieved from the visible graph is allowed "
            "for text/gnn modes."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--processed", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "quality" / "handoff_acceptance.json")
    args = parser.parse_args()
    try:
        report = run_acceptance(args.raw, args.processed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8", newline="\n")
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        parser.exit(1, f"A-side handoff check failed: {error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
