"""Create deterministic supervision-only quality annotations for frozen A QA.

Flags identify cases to inspect; they are not automatic error labels. The tool
does not change questions, answers, evidence, source names or existing manifests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graph_agent.data.loader import RawGraph, load_raw, resolve_raw_files  # noqa: E402
from graph_agent.data.surface import relation_surface  # noqa: E402

SPLITS = ("train", "validation", "test")
FLAG_NAMES = (
    "topic_label_collision_all_mapping",
    "topic_label_collision_visible_graph",
    "same_label_answer_ids",
    "source_label_suffix",
    "any_gold_direct_neighbor",
    "all_gold_direct_neighbors",
)
NORMALIZER = "Unicode str.casefold plus split/join whitespace collapse; no suffix stripping"
SUFFIX = re.compile(r"-(?:GB|US)$")


def normalized_label(label: str) -> str:
    return " ".join(label.casefold().split())


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_group_id(topic_id: str, relation_path: tuple[str, ...]) -> str:
    key = json.dumps([topic_id, list(relation_path)], ensure_ascii=False, separators=(",", ":"))
    return "g_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def annotate_groups(records: dict[str, list[dict]], graph: RawGraph) -> tuple[list[dict], dict[str, dict]]:
    """Return annotations and one original QA record per stable group ID."""
    adjacency = graph.adjacency()
    visible_entities = {entity for t in graph.train_triples for entity in (t.head, t.tail)}
    labels: dict[str, list[str]] = defaultdict(list)
    for entity_id, label in graph.entity_text.items():
        labels[normalized_label(label)].append(entity_id)
    for candidates in labels.values():
        candidates.sort()

    grouped: dict[tuple, list[tuple[str, dict]]] = defaultdict(list)
    seen_ids = set()
    for split in SPLITS:
        for item in records.get(split, []):
            if len(item["topic_entity_ids"]) != 1 or len(item["relation_path"]) != 2:
                raise ValueError("Quality audit expects one topic and a two-relation path")
            if item["question_id"] in seen_ids:
                raise ValueError(f"Duplicate question ID: {item['question_id']}")
            seen_ids.add(item["question_id"])
            if not item["answer_ids"] or len(set(item["answer_ids"])) != len(item["answer_ids"]):
                raise ValueError(f"Empty or duplicate answer IDs: {item['question_id']}")
            key = (item["topic_entity_ids"][0], tuple(item["relation_path"]))
            grouped[key].append((split, item))

    annotations = []
    representatives = {}
    for (topic, path), members in sorted(grouped.items()):
        split_names = {split for split, _ in members}
        if len(split_names) != 1:
            raise ValueError(f"Group crosses QA splits: {topic}, {path}")
        split = next(iter(split_names))
        items = sorted((item for _, item in members), key=lambda item: item["question_id"])
        item = items[0]
        answers = sorted(item["answer_ids"])
        if any(sorted(other["answer_ids"]) != answers for other in items):
            raise ValueError(f"Paraphrases have conflicting gold answers: {topic}, {path}")
        group_id = stable_group_id(topic, path)
        topic_label = graph.entity_text[topic]
        all_candidates = labels[normalized_label(topic_label)]
        visible_candidates = sorted(set(all_candidates) & visible_entities)
        answer_labels: dict[str, list[str]] = defaultdict(list)
        for answer in answers:
            answer_labels[normalized_label(graph.entity_text[answer])].append(answer)
        collisions = [
            {"normalized_label": label, "entity_ids": ids,
             "raw_labels": [graph.entity_text[entity] for entity in ids]}
            for label, ids in sorted(answer_labels.items()) if len(ids) > 1
        ]
        evidence_nodes = {node for head, _, tail in item["evidence"] for node in (head, tail)}
        named_nodes = evidence_nodes | set(answers) | {topic}
        suffix_labels = [
            {"entity_id": entity, "raw_label": graph.entity_text[entity]}
            for entity in sorted(named_nodes) if SUFFIX.search(graph.entity_text[entity])
        ]
        direct_neighbors = {tail for _, tail in adjacency.get(topic, ())}
        direct_answers = sorted(set(answers) & direct_neighbors)
        degree = len(adjacency.get(topic, ()))
        flags = {
            "topic_label_collision_all_mapping": len(all_candidates) > 1,
            "topic_label_collision_visible_graph": len(visible_candidates) > 1,
            "same_label_answer_ids": bool(collisions),
            "source_label_suffix": bool(suffix_labels),
            "any_gold_direct_neighbor": bool(direct_answers),
            "all_gold_direct_neighbors": len(direct_answers) == len(answers),
        }
        annotations.append({
            "group_id": group_id,
            "split": split,
            "question_ids": [other["question_id"] for other in items],
            "topic_entity_id": topic,
            "topic_name": topic_label,
            "relation_path": list(path),
            "answer_count": len(answers),
            "answer_size_bucket": "1" if len(answers) == 1 else "2-5" if len(answers) <= 5 else "6-20" if len(answers) <= 20 else "21+",
            "topic_out_degree": degree,
            "degree_bucket": "0-20" if degree <= 20 else "21-100" if degree <= 100 else "101+",
            "topic_label_candidate_ids_all_mapping": all_candidates,
            "topic_label_candidate_ids_visible_graph": visible_candidates,
            "answer_label_collisions": collisions,
            "source_suffix_labels": suffix_labels,
            "direct_neighbor_answer_ids": direct_answers,
            "flags": flags,
        })
        representatives[group_id] = item
    return annotations, representatives


def select_review_groups(annotations: list[dict], *, seed: int = 20261117,
                         per_path: int = 2) -> list[dict]:
    """Stable hash ranks; after first pick prefer an unrepresented split."""
    if type(per_path) is not int or per_path < 1:
        raise ValueError("per_path must be a positive integer")
    by_path = defaultdict(list)
    for annotation in annotations:
        by_path[tuple(annotation["relation_path"])].append(annotation)
    selected = []
    for path in sorted(by_path):
        pool = sorted(by_path[path], key=lambda row: hashlib.sha256(
            f"{seed}|{row['group_id']}".encode("utf-8")
        ).hexdigest())
        picked_splits = set()
        for _ in range(min(per_path, len(pool))):
            different = [row for row in pool if row["split"] not in picked_splits]
            row = (different or pool)[0]
            selected.append(row)
            picked_splits.add(row["split"])
            pool.remove(row)
    return selected


def review_record(annotation: dict, item: dict, entity_text: dict[str, str]) -> dict:
    return {
        "group_id": annotation["group_id"],
        "split": annotation["split"],
        "question_id": item["question_id"],
        "all_question_ids": annotation["question_ids"],
        "text": item["text"],
        "topic": {"entity_id": annotation["topic_entity_id"], "name": annotation["topic_name"]},
        "relation_path": item["relation_path"],
        "relation_surface": [relation_surface(relation) for relation in item["relation_path"]],
        "answers": [{"entity_id": entity, "name": entity_text[entity]} for entity in item["answer_ids"]],
        "evidence": [{"head": head, "relation": relation, "tail": tail,
                      "head_name": entity_text[head], "tail_name": entity_text[tail]}
                     for head, relation, tail in item["evidence"]],
        "flags": annotation["flags"],
        "review_status": "pending_semantic_review",
    }


def _flag_counts(rows: list[dict]) -> dict[str, int]:
    return {flag: sum(row["flags"][flag] for row in rows) for flag in FLAG_NAMES}


def build_summary(annotations: list[dict], reviews: list[dict], *, seed: int,
                  per_path: int) -> dict:
    by_path = defaultdict(list)
    for annotation in annotations:
        by_path[tuple(annotation["relation_path"])].append(annotation)
    return {
        "artifact_role": "supervision_and_quality_review_only; never Agent or retriever inputs",
        "dataset_status": "frozen QA unchanged; flags are review signals, not confirmed errors",
        "normalizer": NORMALIZER,
        "definitions": {
            "topic_label_collision_all_mapping": "topic name maps to multiple IDs in the complete source name map",
            "topic_label_collision_visible_graph": "topic name maps to multiple IDs present in train graph",
            "same_label_answer_ids": "distinct gold IDs in one answer set share a normalized source name; do not merge",
            "source_label_suffix": "at least one named topic/answer/evidence node ends in exact -GB or -US; preserve source label",
            "any_gold_direct_neighbor": "at least one gold answer is also an outgoing one-hop neighbor under any relation; does not invalidate relation composition",
            "all_gold_direct_neighbors": "all gold answers are outgoing one-hop neighbors; this is topological, not proof the question can be semantically answered in one hop",
            "topic_out_degree": "number of unique directed relation-tail edges in visible train graph",
        },
        "group_count": len(annotations),
        "row_count": sum(len(row["question_ids"]) for row in annotations),
        "path_count": len(by_path),
        "flag_group_counts": _flag_counts(annotations),
        "answer_size_group_counts": dict(sorted(Counter(row["answer_size_bucket"] for row in annotations).items())),
        "topic_degree_group_counts": dict(sorted(Counter(row["degree_bucket"] for row in annotations).items())),
        "per_split": {
            split: {"groups": sum(row["split"] == split for row in annotations),
                    "rows": sum(len(row["question_ids"]) for row in annotations if row["split"] == split),
                    "flag_group_counts": _flag_counts([row for row in annotations if row["split"] == split])}
            for split in SPLITS
        },
        "per_path": [
            {"relation_path": list(path), "relation_surface": [relation_surface(r) for r in path],
             "groups": len(rows), "flag_group_counts": _flag_counts(rows)}
            for path, rows in sorted(by_path.items())
        ],
        "selection": {"seed": seed, "per_path_max": per_path,
                      "policy": "SHA256(seed|group_id) ranking within each relation path; prefer a different split after the first pick",
                      "review_group_count": len(reviews),
                      "review_path_count": len({tuple(row["relation_path"]) for row in reviews}),
                      "selected_group_ids": [row["group_id"] for row in reviews]},
    }


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8", newline="\n")


def audit(raw: Path, processed: Path, output: Path, *, seed: int = 20261117,
          per_path: int = 2) -> dict:
    if output.resolve() in (raw.resolve(), processed.resolve()):
        raise ValueError("Quality annotations require a separate output directory")
    manifest = json.loads((processed / "manifest.json").read_text(encoding="utf-8"))
    input_names = sorted(set(manifest["files"]) | {"manifest.json"})
    if any(Path(name).name != name for name in input_names):
        raise ValueError("Manifest file names must be direct children of processed directory")
    for name, entry in manifest["files"].items():
        if checksum(processed / name) != entry["sha256"]:
            raise ValueError(f"Frozen artifact checksum mismatch: {name}")
    raw_files = resolve_raw_files(raw)
    for name, entry in manifest["raw_inputs"].items():
        if checksum(raw_files[name]) != entry["sha256"]:
            raise ValueError(f"Raw input differs from frozen QA manifest: {name}")
    for name, expected in manifest["code_sha256"].items():
        if Path(name).name != name:
            raise ValueError("Data module names must be direct children of src/graph_agent/data")
        source = (ROOT / "src" / "graph_agent" / "data" / name).read_bytes()
        if hashlib.sha256(source.replace(b"\r\n", b"\n")).hexdigest() != expected:
            raise ValueError(f"Data code differs from frozen QA manifest: {name}")
    graph = load_raw(raw)
    records = {split: [json.loads(line) for line in (processed / f"{split}.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()] for split in SPLITS}
    annotations, representatives = annotate_groups(records, graph)
    selected = select_review_groups(annotations, seed=seed, per_path=per_path)
    reviews = [review_record(row, representatives[row["group_id"]], graph.entity_text) for row in selected]
    summary = build_summary(annotations, reviews, seed=seed, per_path=per_path)
    summary["input_sha256"] = {
        "processed": {name: checksum(processed / name) for name in input_names},
        "raw": {name: {"file": path.relative_to(raw).as_posix(), "sha256": checksum(path)}
                for name, path in raw_files.items()},
    }
    summary["audit_script_sha256_lf"] = hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "group_annotations.jsonl", annotations)
    write_jsonl(output / "review_samples.jsonl", reviews)
    write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--processed", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "quality")
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--per-path", type=int, default=2)
    args = parser.parse_args()
    try:
        summary = audit(args.raw, args.processed, args.output, seed=args.seed, per_path=args.per_path)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"Quality audit failed: {error}\n")
    display = {key: summary[key] for key in
               ("group_count", "row_count", "path_count", "flag_group_counts")}
    display["review_group_count"] = summary["selection"]["review_group_count"]
    display["review_path_count"] = summary["selection"]["review_path_count"]
    print(json.dumps(display, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
