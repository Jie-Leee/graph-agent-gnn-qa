"""Data preparation entry point owned by A.

Pipeline
--------
1. parse raw FB15k-237 (train / valid / test + entity & relation text)
2. index the **training** triples as the visible graph
3. enumerate two-hop ``(topic entity, relation path)`` groups with complete
   answer sets and the evidence paths that justify them
4. drop degenerate relation paths (too rare, or one answer dominates)
5. assign whole groups to splits, two paraphrases per group
6. audit for leakage and write JSONL + manifest + reports

Run it through ``scripts/prepare_data.py``; the signature below is the stable
one declared by the project scaffold.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .builder import enumerate_groups, filter_degenerate_paths
from .loader import count_duplicates, load_raw, resolve_raw_files
from .report import render_samples_markdown, render_stats_markdown, write_text
from .split import SPLIT_ORDER, assign_records, audit_records, validate_targets
from .surface import RELATION_SURFACE, TEMPLATE_VERSION, relation_surface

DEFAULT_TARGETS: dict[str, int] = {"train": 2000, "validation": 200, "test": 300}
SPLIT_FILENAMES = {
    "train": "train.jsonl",
    "validation": "validation.jsonl",
    "test": "test.jsonl",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_sha256(path: Path) -> str:
    # Git may convert source LF to CRLF on Windows checkout.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def prepare_dataset(
    raw_dir: Path,
    output_dir: Path,
    seed: int = 42,
    *,
    targets: dict[str, int] | None = None,
    paraphrases: int = 2,
    max_answers: int = 20,
    max_groups: int = 60_000,
    min_path_support: int = 25,
    max_dominant_share: float = 0.6,
    max_answer_set_share: float = 0.4,
) -> dict:
    """Build the QA dataset and return the statistics that were written."""
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    targets = dict(DEFAULT_TARGETS if targets is None else targets)
    validate_targets(targets, paraphrases)
    if not isinstance(max_answers, int) or isinstance(max_answers, bool) or max_answers < 1:
        raise ValueError("max_answers must be a positive integer")
    if not isinstance(max_groups, int) or isinstance(max_groups, bool) or max_groups < 1:
        raise ValueError("max_groups must be a positive integer")
    if not isinstance(min_path_support, int) or isinstance(min_path_support, bool) or min_path_support < 1:
        raise ValueError("min_path_support must be a positive integer")
    if not 0 <= max_dominant_share <= 1 or not 0 <= max_answer_set_share <= 1:
        raise ValueError("path share thresholds must be between 0 and 1")

    graph = load_raw(raw_dir)
    adjacency = graph.adjacency()

    candidate_groups = enumerate_groups(
        adjacency, max_answers=max_answers, max_groups=max_groups, seed=seed
    )
    kept_groups, path_stats = filter_degenerate_paths(
        candidate_groups,
        min_support=min_path_support,
        max_dominant_share=max_dominant_share,
        max_answer_set_share=max_answer_set_share,
    )
    assignment = assign_records(
        kept_groups,
        targets,
        entity_text=graph.entity_text,
        paraphrases=paraphrases,
        seed=seed,
    )

    audit = audit_records(assignment.records, adjacency=adjacency)
    if audit["violations"]:
        raise ValueError("Data audit failed before writing outputs: " + "; ".join(audit["violations"]))

    answer_histogram: Counter[int] = Counter()
    path_usage: Counter[tuple[str, str]] = Counter()
    path_types: dict[str, set[tuple[str, str]]] = {}
    answers_sum: dict[str, int] = {}
    answers_max: dict[str, int] = {}
    for name in SPLIT_ORDER:
        items = assignment.records[name]
        answers_sum[name] = sum(len(i["answer_ids"]) for i in items)
        answers_max[name] = max((len(i["answer_ids"]) for i in items), default=0)
        path_types[name] = set()
        for item in items:
            answer_histogram[len(item["answer_ids"])] += 1
            path = tuple(item["relation_path"])
            path_usage[path] += 1
            path_types[name].add(path)

    split_stats = {
        name: {
            "items": len(assignment.records[name]),
            "groups": assignment.group_counts[name],
            "paraphrases_per_group": paraphrases,
            "answers_mean": round(
                answers_sum[name] / max(len(assignment.records[name]), 1), 2
            ),
            "answers_max": answers_max[name],
            "path_types": len(path_types[name]),
        }
        for name in SPLIT_ORDER
    }

    dropped = [
        (
            stat.path[0],
            stat.path[1],
            stat.groups,
            stat.dominant_share,
            stat.answer_set_share,
            stat.reason,
        )
        for stat in path_stats
        if not stat.kept
    ]

    stats = {
        "dataset": "FB15k-237 two-hop template QA",
        "seed": seed,
        "template_version": TEMPLATE_VERSION,
        "paraphrases_per_group": paraphrases,
        "max_answers": max_answers,
        "max_groups": max_groups,
        "min_path_support": min_path_support,
        "max_dominant_share": max_dominant_share,
        "max_answer_set_share": max_answer_set_share,
        "visible_graph": "train split only",
        "whitelist_relations": len(RELATION_SURFACE),
        "raw": {**graph.summary(), "duplicates": count_duplicates(raw_dir)},
        "splits": split_stats,
        "answer_histogram": dict(sorted(answer_histogram.items())),
        "path_count": len(path_usage),
        "path_usage": [[list(path), count] for path, count in path_usage.most_common()],
        "template_usage": [
            [template, count]
            for template, count in assignment.template_counts.most_common()
        ],
        "dropped_paths": [list(row) for row in dropped],
        "path_stats": [
            {
                "path": list(stat.path),
                "path_text": " >> ".join(relation_surface(p) for p in stat.path),
                "groups": stat.groups,
                "answers_mean": stat.answers_mean,
                "dominant_share": stat.dominant_share,
                "answer_set_share": stat.answer_set_share,
                "kept": stat.kept,
                "reason": stat.reason,
            }
            for stat in sorted(path_stats, key=lambda s: -s.groups)
        ],
        "candidate_groups": len(candidate_groups),
        "kept_groups": len(kept_groups),
        "skipped_groups_on_collision": assignment.skipped_groups,
        "audit": audit,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_ORDER:
        _write_jsonl(output_dir / SPLIT_FILENAMES[name], assignment.records[name])
        _write_jsonl(
            output_dir / f"{name}.questions.jsonl",
            [{key: item[key] for key in ("question_id", "text", "topic_entity_ids")}
             for item in assignment.records[name]],
        )

    (output_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )
    write_text(output_dir / "stats.md", render_stats_markdown(stats))
    write_text(
        output_dir / "samples.md",
        render_samples_markdown(assignment.records, graph.entity_text),
    )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "template_version": TEMPLATE_VERSION,
        "paraphrases_per_group": paraphrases,
        "max_answers": max_answers,
        "max_groups": max_groups,
        "min_path_support": min_path_support,
        "max_dominant_share": max_dominant_share,
        "max_answer_set_share": max_answer_set_share,
        "targets": targets,
        "visible_graph": "FB15K-237 train.txt",
        "protocol": "transductive known-graph two-hop template QA; grouped query split",
        "topic_entities": "provided gold topic IDs; entity linking is evaluated separately",
        "raw_inputs": {
            name: {"file": path.relative_to(raw_dir).as_posix(), "sha256": _sha256(path)}
            for name, path in resolve_raw_files(raw_dir).items()
        },
        "code_sha256": {
            path.name: _code_sha256(path)
            for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        "code_hash_policy": "SHA256 of source bytes with CRLF normalized to LF",
        "files": {
            filename: {
                "items": len(assignment.records[name]) if filename.endswith("jsonl") else None,
                "sha256": _sha256(output_dir / filename),
            }
            for name in SPLIT_ORDER
            for filename in (SPLIT_FILENAMES[name], f"{name}.questions.jsonl")
        },
    }
    for filename in ("stats.json", "stats.md", "samples.md"):
        manifest["files"][filename] = {"sha256": _sha256(output_dir / filename)}
    source_record = raw_dir / "download_sources.json"
    if source_record.is_file():
        manifest["download_sources"] = json.loads(source_record.read_text(encoding="utf-8"))
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )

    return stats
