"""Independent, provenance-checked reporting for E's question-level records."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

MAIN_VARIANTS = ("llm", "text", "gat_2_mean")
GNN_VARIANTS = ("gat_1_mean", "gat_2_mean", "gat_3_mean", "gat_2_sum", "gat_2_max")
VARIANT_MODES = {"llm": "llm", "text": "text", **{v: "gnn" for v in GNN_VARIANTS}}
OK_STATUSES = {"ok", "success", "completed", "cached"}
TOKEN_FIELDS = (
    "logical_input_tokens", "logical_output_tokens",
    "actual_input_tokens", "actual_output_tokens",
)
TIME_FIELDS = ("logical_elapsed_seconds", "actual_elapsed_seconds")


class ReportError(ValueError):
    """The records cannot be combined into one valid experiment report."""


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _ids(value: Any, field: str) -> set[str]:
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(x, str) and x.strip() for x in value
    ):
        raise ReportError(f"{field} must be a list of nonempty entity IDs")
    return set(value)


def _fingerprint(value: dict) -> str | None:
    for field in ("experiment_fingerprint", "config_fingerprint",
                  "run_fingerprint", "manifest_fingerprint"):
        if isinstance(value.get(field), str) and value[field].strip():
            return value[field]
    return None


def _effective(row: dict) -> dict:
    result = dict(row)
    gold = _ids(row.get("gold_answer_ids"), "gold_answer_ids")
    predicted = _ids(row.get("predicted_answer_ids"), "predicted_answer_ids")
    success = str(row.get("status", "")).lower() in OK_STATUSES
    overlap = len(gold & predicted)
    precision = overlap / len(predicted) if predicted else float(not gold)
    recall = overlap / len(gold) if gold else float(not predicted)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    result.update(
        exact_match=float(gold == predicted) if success else 0.0,
        precision=precision if success else 0.0,
        recall=recall if success else 0.0,
        f1=f1 if success else 0.0,
        successful=success,
        answered=bool(predicted) and success,
        graph_unsupported_answer_fraction=(
            len(predicted - gold) / len(predicted) if success and predicted else None
        ),
    )
    error_type = str(row.get("error_type") or "").lower()
    unresolved = row.get("unresolved_answer_names") or []
    ambiguous = row.get("ambiguous_answer_names") or {}
    rejected = row.get("rejected_answer_ids") or []
    diagnostics = bool(unresolved or ambiguous or rejected)
    mapping_failed = bool(row.get("mapping_failed")) or diagnostics or any(
        word in error_type for word in ("mapping", "normaliz", "unmapped")
    )
    result.update(
        mapping_failed=mapping_failed, unresolved_answer_names=unresolved,
        ambiguous_answer_names=ambiguous, rejected_answer_ids=rejected,
        model_attempted_answer=bool(row.get("model_attempted_answer", predicted)),
        normalization_unresolved=success and not predicted and diagnostics,
    )
    result["empty_ids_with_text"] = (
        success and not predicted and bool(str(row.get("answer_text") or "").strip())
        and not mapping_failed
    )
    for field in (*TOKEN_FIELDS, *TIME_FIELDS):
        result[field] = _number(row.get(field))
    for field in ("candidate_recall", "context_answer_recall"):
        value = _number(row.get(field))
        if value is not None and value > 1:
            raise ReportError(f"{field} must be between zero and one")
        result[field] = value
    return result


def validate_records(records: Iterable[dict], manifest: dict | None = None,
                     *, allow_mock: bool = False) -> tuple[list[dict], dict]:
    """Reject incompatible runs; retain unsuccessful observations as zero scores."""
    source = list(records)
    if not source:
        raise ReportError("No prediction records; no results can be inferred")
    if not all(isinstance(row, dict) for row in source):
        raise ReportError("Each prediction record must be a JSON object")
    manifest = manifest or {}
    seen, fingerprints, backends, splits = set(), set(), set(), set()
    questions: dict[str, tuple[str, frozenset[str]]] = {}
    output = []
    for number, row in enumerate(source, 1):
        qid, group = row.get("question_id"), row.get("group_id")
        if not isinstance(qid, str) or not qid or not isinstance(group, str) or not group:
            raise ReportError(f"Record {number}: question_id and group_id are required")
        variant = row.get("variant")
        if variant not in VARIANT_MODES or row.get("mode") != VARIANT_MODES[variant]:
            raise ReportError(f"Record {number}: incompatible mode/variant")
        key = (variant, qid)
        if key in seen:
            raise ReportError(f"Duplicate prediction: {variant}/{qid}")
        seen.add(key)
        fingerprint = _fingerprint(row)
        if not fingerprint:
            raise ReportError(f"Record {number}: experiment_fingerprint is required")
        fingerprints.add(fingerprint)
        backend, split = row.get("execution_backend"), row.get("split")
        if backend not in ("live", "mock") or not isinstance(split, str) or not split:
            raise ReportError(f"Record {number}: explicit live/mock backend and split required")
        backends.add(backend)
        splits.add(split)
        effective = _effective(row)
        identity = (group, frozenset(row["gold_answer_ids"]))
        if qid in questions and questions[qid] != identity:
            raise ReportError(f"Gold answers/group disagree for {qid}")
        questions[qid] = identity
        output.append(effective)
    if len(backends) != 1:
        raise ReportError("Cannot mix mock and live predictions")
    if len(splits) != 1:
        raise ReportError("Cannot mix different data splits")
    if len(fingerprints) != 1:
        raise ReportError("Cannot mix different experiment/config fingerprints")
    backend, split, fingerprint = next(iter(backends)), next(iter(splits)), next(iter(fingerprints))
    if backend == "mock" and not allow_mock:
        raise ReportError("Mock results require --allow-mock and cannot support formal conclusions")
    manifest_fingerprint = _fingerprint(manifest)
    if manifest_fingerprint and manifest_fingerprint != fingerprint:
        raise ReportError("Manifest fingerprint differs from prediction records")
    for field, expected in (("split", split), ("execution_backend", backend)):
        if manifest.get(field) is not None and manifest[field] != expected:
            raise ReportError(f"Manifest {field} differs from prediction records")
    return output, {
        "execution_backend": backend, "split": split,
        "experiment_fingerprint": fingerprint,
        "record_count": len(output), "question_count": len(questions),
        "group_count": len({v[0] for v in questions.values()}),
        "manifest_status": manifest.get("status", "not_provided"),
    }



def _selection(manifest: dict) -> tuple[list[str] | None, int | None, dict[str, str], list[str]]:
    identity = manifest.get("identity") or {}
    ids = manifest.get("selected_question_ids", identity.get("selected_question_ids"))
    if ids is not None:
        if not isinstance(ids, list) or not all(isinstance(q, str) and q for q in ids):
            raise ReportError("selected_question_ids must contain string IDs")
        if len(ids) != len(set(ids)):
            raise ReportError("selected_question_ids contains duplicates")
    count = manifest.get("selected_question_count", identity.get("selected_question_count"))
    if count is None and ids is not None:
        count = len(ids)
    if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count < 1):
        raise ReportError("selected_question_count must be positive")
    if ids is not None and count is not None and len(ids) != count:
        raise ReportError("Selected ID count differs from selected_question_count")
    groups = manifest.get("selected_question_groups", identity.get("selected_question_groups")) or {}
    if not isinstance(groups, dict) or not all(
        isinstance(q, str) and isinstance(g, str) and g for q, g in groups.items()
    ):
        raise ReportError("selected_question_groups must map question IDs to group IDs")
    variants = manifest.get("variants", identity.get("variants")) or []
    if isinstance(variants, dict):
        variants = list(variants)
    if not isinstance(variants, list) or any(v not in VARIANT_MODES for v in variants):
        raise ReportError("Manifest variants must be known variants")
    return ids, count, groups, variants


def _expand_pending(records: list[dict], manifest: dict) -> tuple[list[dict], dict]:
    """Pending is zero for scoring, not a fabricated prediction or answer."""
    ids, selected_count, groups, planned = _selection(manifest)
    groups, result = dict(groups), list(records)
    by_question = {}
    for row in records:
        qid = row["question_id"]
        if ids is not None and qid not in ids:
            raise ReportError(f"Prediction is outside selected manifest IDs: {qid}")
        if qid in groups and groups[qid] != row["group_id"]:
            raise ReportError(f"Manifest group differs for {qid}")
        groups[qid] = row["group_id"]
        by_question[qid] = row
    observed = list(dict.fromkeys(r["variant"] for r in records))
    variants = planned or observed
    if set(observed) - set(variants):
        raise ReportError("Predictions include variants outside manifest")
    pending, missing_groups = 0, set()
    if ids is not None:
        keys = {(r["variant"], r["question_id"]) for r in records}
        template = records[0]
        for variant in variants:
            for qid in ids:
                if (variant, qid) in keys:
                    continue
                original, group = by_question.get(qid), groups.get(qid)
                if group is None:
                    missing_groups.add(qid)
                result.append({
                    "question_id": qid, "group_id": group, "variant": variant,
                    "mode": VARIANT_MODES[variant], "split": template["split"],
                    "execution_backend": template["execution_backend"],
                    "experiment_fingerprint": _fingerprint(template),
                    "gold_answer_ids": original["gold_answer_ids"] if original else None,
                    "predicted_answer_ids": [], "answer_text": "", "status": "pending",
                    "error_type": None, "synthetic_pending": True, "successful": False,
                    "answered": False, "mapping_failed": False, "model_attempted_answer": False,
                    "unresolved_answer_names": [], "ambiguous_answer_names": {},
                    "rejected_answer_ids": [], "normalization_unresolved": False,
                    "empty_ids_with_text": False, "exact_match": 0.0, "precision": 0.0,
                    "recall": 0.0, "f1": 0.0, "candidate_recall": None,
                    "context_answer_recall": None, "graph_unsupported_answer_fraction": None,
                    "usage_reported": None, "route_cache_hit": False, "served_models": [],
                    "cited_evidence": [], "trace_file": None,
                    **{key: None for key in (*TOKEN_FIELDS, *TIME_FIELDS)},
                })
                pending += 1
    without_ids = 0
    if ids is None and selected_count is not None:
        for variant in variants:
            n = sum(r["variant"] == variant for r in records)
            if n > selected_count:
                raise ReportError("Recorded questions exceed selected count")
            without_ids += selected_count - n
    return result, {
        "selected_question_count": selected_count,
        "pending_record_count": pending + without_ids,
        "pending_without_ids": without_ids,
        "pending_group_ids_missing": len(missing_groups),
    }

def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def _sum_field(rows: list[dict], field: str) -> dict:
    values = [_number(row.get(field)) for row in rows]
    known = [value for value in values if value is not None]
    return {
        field + "_total": sum(known) if len(known) == len(rows) else None,
        field + "_known_total": sum(known) if known else None,
        field + "_reported_count": len(known),
    }


def _dataset_count(manifest: dict) -> int | None:
    for field in ("dataset_question_count", "full_split_question_count",
                  "dataset_total_questions"):
        value = manifest.get(field)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def summarize(records: list[dict], manifest: dict | None = None) -> list[dict]:
    """Selected-question denominators retain pending observations as zero."""
    manifest = manifest or {}
    _, selected_count, _, _ = _selection(manifest)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        grouped[row["variant"]].append(row)
    expected = _dataset_count(manifest)
    table = []
    for variant, rows in sorted(grouped.items()):
        observed = [r for r in rows if not r.get("synthetic_pending")]
        denominator = selected_count or len(rows)
        if len(rows) > denominator or len(observed) > denominator:
            raise ReportError("Prediction count exceeds selected-question denominator")
        pending = denominator - len(observed)
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            if row.get("group_id") is not None:
                groups[row["group_id"]].append(row)
        incomplete_groups = any(r.get("group_id") is None for r in rows) or len(rows) < denominator
        state = "processed_subset"
        if expected:
            state = "complete_split" if denominator == expected and not pending else "partial_split"
        if pending:
            state = "partial_pending"
        elif str(manifest.get("status", "")).lower() in ("smoke", "partial", "incomplete", "running"):
            state = "smoke_or_partial"
        if rows[0]["execution_backend"] == "mock":
            state = "mock_only"
        summary = {
            "variant": variant, "mode": VARIANT_MODES[variant],
            "execution_backend": rows[0]["execution_backend"], "split": rows[0]["split"],
            "experiment_fingerprint": _fingerprint(rows[0]),
            "questions": denominator, "observed_questions": len(observed),
            "pending_count": pending, "groups": None if incomplete_groups else len(groups),
            "dataset_questions": expected, "coverage_status": state,
            **{key: sum(r[key] for r in rows) / denominator
               for key in ("exact_match", "precision", "recall", "f1")},
            "group_exact_match": None if incomplete_groups else _mean(
                _mean(r["exact_match"] for r in group) for group in groups.values()),
            "group_f1": None if incomplete_groups else _mean(
                _mean(r["f1"] for r in group) for group in groups.values()),
            "answer_rate": sum(r["answered"] for r in rows) / denominator,
            "error_rate": sum(not r["successful"] for r in observed) / denominator,
            "error_count": sum(not r["successful"] for r in observed),
            "empty_prediction_count": sum(r["successful"] and not r["answered"] for r in observed),
            "mapping_failure_count": sum(r["mapping_failed"] for r in observed),
            "unresolved_name_question_count": sum(bool(r.get("unresolved_answer_names")) for r in observed),
            "ambiguous_name_question_count": sum(bool(r.get("ambiguous_answer_names")) for r in observed),
            "rejected_id_question_count": sum(bool(r.get("rejected_answer_ids")) for r in observed),
            "normalization_unresolved_count": sum(r.get("normalization_unresolved", False) for r in observed),
            "model_attempted_answer_count": sum(r.get("model_attempted_answer", False) for r in observed),
            "empty_ids_with_text_count": sum(r["empty_ids_with_text"] for r in observed),
            "graph_unsupported_answer_fraction": _mean(
                r["graph_unsupported_answer_fraction"] for r in observed
                if r["graph_unsupported_answer_fraction"] is not None),
            "nonempty_scored_questions": sum(
                r["graph_unsupported_answer_fraction"] is not None for r in observed),
            "route_cache_hits": sum(bool(r.get("route_cache_hit")) for r in observed),
            "usage_reported_count": sum(r.get("usage_reported") is True for r in observed),
        }
        for field in ("candidate_recall", "context_answer_recall"):
            values = [r[field] for r in observed if r[field] is not None]
            summary[field] = _mean(values)
            summary[field + "_reported_count"] = len(values)
        for field in (*TOKEN_FIELDS, *TIME_FIELDS):
            summary.update(_sum_field(rows, field))
            if pending:
                summary[field + "_total"] = None
            summary[field + "_expected_count"] = denominator
            if field in TIME_FIELDS:
                total = summary[field + "_total"]
                summary["mean_" + field] = total / denominator if total is not None else None
        models = set()
        for row in observed:
            model = row.get("served_models") or []
            models.update([model] if isinstance(model, str) else model)
        summary["served_models"] = sorted(str(model) for model in models)
        table.append(summary)
    return table


def _quantile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def paired_group_bootstrap(records: list[dict], reference: str, candidate: str,
                           *, samples: int = 2000, seed: int = 42) -> dict:
    """Resample paired query groups, not correlated paraphrases."""
    if samples < 1:
        raise ReportError("bootstrap samples must be positive")
    left = {r["question_id"]: r for r in records if r["variant"] == reference}
    right = {r["question_id"]: r for r in records if r["variant"] == candidate}
    shared = sorted(left.keys() & right.keys())
    group_mapping_missing = any(left[qid].get("group_id") is None or
                                right[qid].get("group_id") is None for qid in shared)
    groups: dict[str, list[float]] = defaultdict(list)
    for qid in shared:
        if left[qid]["group_id"] != right[qid]["group_id"]:
            raise ReportError(f"Group mismatch in paired comparison: {qid}")
        if left[qid].get("group_id") is not None:
            groups[left[qid]["group_id"]].append(right[qid]["f1"] - left[qid]["f1"])
    differences = [_mean(group) for _, group in sorted(groups.items())]
    result = {
        "reference": reference, "candidate": candidate, "metric": "paired_group_f1",
        "paired_questions": len(shared), "paired_groups": len(groups),
        "reference_only_questions": len(left.keys() - right.keys()),
        "candidate_only_questions": len(right.keys() - left.keys()),
        "difference": _mean(differences), "ci_low": None, "ci_high": None,
        "bootstrap_samples": samples, "seed": seed,
        "status": "no_overlap" if not shared else "insufficient_groups",
    }
    if group_mapping_missing:
        result.update(status="pending_group_mapping_missing", difference=None)
        return result
    if len(differences) >= 2:
        stable = int(hashlib.sha256((reference + "|" + candidate).encode()).hexdigest()[:12], 16)
        rng = random.Random(seed + stable)
        values = [sum(rng.choice(differences) for _ in differences) / len(differences)
                  for _ in range(samples)]
        result.update(ci_low=_quantile(values, 0.025), ci_high=_quantile(values, 0.975),
                      status="available")
    return result


def _dump_csv(path: Path, rows: list[dict]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list))
                             else v for k, v in row.items()})


def _format(value: Any, *, percent: bool = False) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, (int, float)):
        return f"{value * 100:.2f}%" if percent else f"{value:.4f}"
    return str(value).replace("|", "/").replace("\n", " ")


def _error_cases(records: list[dict]) -> list[dict]:
    cases = []
    for row in records:
        if row["exact_match"] == 1 and not row["mapping_failed"]:
            continue
        if row.get("synthetic_pending") or row.get("status") == "pending":
            category = "pending"
        elif not row["successful"]:
            category = "execution_error"
        elif row.get("normalization_unresolved"):
            category = "normalization_unresolved"
        elif row["mapping_failed"]:
            category = "mapping_failure"
        elif not row["answered"]:
            category = "empty_ids_with_text" if row["empty_ids_with_text"] else "empty_answer"
        elif row["candidate_recall"] == 0:
            category = "retrieval_miss"
        elif row["candidate_recall"] is not None and row["candidate_recall"] < 1:
            category = "partial_retrieval"
        else:
            category = "wrong_or_incomplete_answer"
        fields = ("question_id", "question_text", "topic_entity_ids", "group_id", "mode", "variant", "split", "execution_backend",
                  "gold_answer_ids", "predicted_answer_ids", "answer_text", "f1", "exact_match",
                  "status", "error_type", "candidate_recall", "context_answer_recall",
                  "graph_unsupported_answer_fraction", "cited_evidence",
                  "unresolved_answer_names", "ambiguous_answer_names",
                  "rejected_answer_ids", "model_attempted_answer",
                  "normalization_unresolved", "trace_file", "synthetic_pending")
        case = {field: row.get(field) for field in fields}
        case["category"] = category
        cases.append(case)
    return sorted(cases, key=lambda x: (x["category"] != "execution_error",
                                       x["f1"], x["variant"], x["question_id"]))


def _write_plots(output: Path, summaries: list[dict], meta: dict) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ReportError("matplotlib is unavailable; CSV/Markdown were generated. "
                          "Install matplotlib or use --no-plots.") from error
    lookup = {r["variant"]: r for r in summaries}
    marker = "MOCK / NOT EXPERIMENT RESULTS" if meta["execution_backend"] == "mock" else "LIVE"
    files = []
    def label(variant):
        row = lookup[variant]
        return (f"{variant}\nN={row['questions']}, M={row['groups']}"
                f"\nRecorded={row['observed_questions']}, pending={row['pending_count']}")
    def save(fig, stem):
        fig.tight_layout()
        for suffix in ("png", "svg"):
            name = stem + "." + suffix
            fig.savefig(output / name, dpi=180, bbox_inches="tight")
            files.append(name)
        plt.close(fig)
    def bars(stem, variants, title):
        present = [name for name in variants if name in lookup]
        if not present:
            return
        fig, axis = plt.subplots(figsize=(max(7, 1.65 * len(present)), 4.8))
        xs = list(range(len(present)))
        axis.bar([x - .18 for x in xs], [lookup[v]["f1"] for v in present], .36, label="Set F1")
        axis.bar([x + .18 for x in xs], [lookup[v]["exact_match"] for v in present], .36, label="Exact match")
        axis.set_xticks(xs, [label(v) for v in present])
        axis.set_ylim(0, 1)
        axis.set_ylabel("Score")
        axis.set_title(title + "\n" + marker + " | " + meta["split"])
        axis.legend()
        axis.grid(axis="y", alpha=.2)
        save(fig, stem)
    bars("main_results", MAIN_VARIANTS, "Predefined baselines")
    bars("ablation_layers", ("gat_1_mean", "gat_2_mean", "gat_3_mean"), "GAT layer ablation")
    bars("ablation_pool", ("gat_2_mean", "gat_2_sum", "gat_2_max"), "Graph pooling ablation")
    present = [v for v in (*MAIN_VARIANTS, *GNN_VARIANTS) if v in lookup]
    present = list(dict.fromkeys(present))
    # Seven variants need compact labels and enough horizontal space per panel.
    def compact_label(variant):
        row = lookup[variant]
        groups = row["groups"] if row["groups"] is not None else "?"
        return (f"{variant}\nN={row['questions']} M={groups}"
                f"\nR={row['observed_questions']} P={row['pending_count']}")
    fig, axes = plt.subplots(1, 2, figsize=(max(14, len(present) * 2.4), 6.1))
    latency = [(i, lookup[v]["mean_logical_elapsed_seconds"]) for i, v in enumerate(present)
               if lookup[v]["mean_logical_elapsed_seconds"] is not None]
    if latency:
        axes[0].bar([i for i, value in latency], [value for i, value in latency], color="#547AA5")
    else:
        axes[0].text(.5, .5, "Latency unknown", ha="center", transform=axes[0].transAxes)
    axes[0].set_ylabel("Mean logical latency (s)")
    axes[0].set_ylim(bottom=0)
    axes[1].bar(range(len(present)), [lookup[v]["answer_rate"] for v in present], color="#498467")
    axes[1].set_ylabel("Nonempty entity-answer rate")
    axes[1].set_ylim(0, 1)
    for axis in axes:
        axis.set_xticks(range(len(present)), [compact_label(v) for v in present],
                        rotation=0, ha="center", fontsize=9)
        axis.grid(axis="y", alpha=.2)
    fig.suptitle("Latency and answer coverage\n" + marker + " | " + meta["split"])
    fig.text(.5, .025, "N = selected questions; M = query groups; R = recorded; P = pending",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .10, 1, .94))
    for suffix in ("png", "svg"):
        name = "latency_answer_rate." + suffix
        fig.savefig(output / name, dpi=180, bbox_inches="tight")
        files.append(name)
    plt.close(fig)
    return files


def build_report(predictions_path: str | Path, output_dir: str | Path, *,
                 manifest_path: str | Path | None = None, allow_mock: bool = False,
                 bootstrap_samples: int = 2000, seed: int = 42,
                 plots: bool = True) -> dict:
    """Build independent result tables, paired intervals, cases and a Chinese report."""
    predictions_path, output = Path(predictions_path), Path(output_dir)
    if manifest_path is None:
        sibling = predictions_path.with_name("manifest.json")
        manifest_path = sibling if sibling.exists() else None
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8-sig")) if manifest_path else {}
    if not isinstance(manifest, dict):
        raise ReportError("Manifest must contain an object")
    source = []
    with predictions_path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    source.append(json.loads(line))
                except ValueError as error:
                    raise ReportError(f"Invalid JSON on prediction line {number}") from error
    records, meta = validate_records(source, manifest, allow_mock=allow_mock)
    records, pending_meta = _expand_pending(records, manifest)
    meta.update(pending_meta)
    summaries = summarize(records, manifest)
    meta["scoring_question_count"] = max(r["questions"] for r in summaries)
    lookup = {r["variant"]: r for r in summaries}
    comparisons = [("llm", "text"), ("text", "gat_2_mean")]
    comparisons += [("gat_2_mean", v) for v in GNN_VARIANTS if v != "gat_2_mean"]
    paired = [paired_group_bootstrap(records, left, right, samples=bootstrap_samples, seed=seed)
              for left, right in comparisons if left in lookup and right in lookup]
    if pending_meta["pending_without_ids"]:
        for pair in paired:
            pair.update(status="pending_question_identity_missing", difference=None,
                        ci_low=None, ci_high=None)
    cases = _error_cases(records)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report_metadata.json").write_text(
        json.dumps({**meta, "bootstrap_seed": seed, "bootstrap_samples": bootstrap_samples,
                    "source": str(predictions_path), "manifest": str(manifest_path) if manifest_path else None},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _dump_csv(output / "variant_summary.csv", summaries)
    _dump_csv(output / "main_results.csv", [lookup[v] for v in MAIN_VARIANTS if v in lookup])
    _dump_csv(output / "gnn_ablation.csv", [lookup[v] for v in GNN_VARIANTS if v in lookup])
    group_rows = []
    for variant in lookup:
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in records:
            if row["variant"] == variant:
                if row.get("group_id") is not None:
                    groups[row["group_id"]].append(row)
        for group, rows in sorted(groups.items()):
            group_rows.append({"variant": variant, "group_id": group, "questions": len(rows),
                               "f1": _mean(r["f1"] for r in rows),
                               "exact_match": _mean(r["exact_match"] for r in rows),
                               "errors": sum(not r["successful"] and not r.get("synthetic_pending") for r in rows),
                               "pending": sum(bool(r.get("synthetic_pending")) for r in rows)})
    _dump_csv(output / "group_scores.csv", group_rows)
    _dump_csv(output / "paired_f1_comparisons.csv", paired)
    (output / "paired_f1_comparisons.json").write_text(
        json.dumps(paired, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "error_cases.jsonl").write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases), encoding="utf-8")
    case_lines = ["# 逐题错误与待复核案例", "",
                  "这里只分类观察到的现象；名称解析未完成不等于自然语言答案正确，检索与错误的因果关系仍需复核。",
                  "pending 为未执行的选中题，按规则计零分，不是生成的回答。", "",
                  "| 模式 | 问题ID | 问题文字 | 类别 | F1 | 错误类型 | 追踪文件 |",
                  "|---|---|---|---|---:|---|---|"]
    for case in cases[:50]:
        text = str(case.get("question_text") or "")[:220]
        values = (case["variant"], case["question_id"], text, case["category"],
                  case["f1"], case["error_type"], case.get("trace_file"))
        case_lines.append("| " + " | ".join(_format(value) for value in values) + " |")
    case_lines += ["", f"共 {len(cases)} 条；本页展示前 {min(50, len(cases))} 条，完整记录见 error_cases.jsonl。"]
    (output / "error_cases.md").write_text("\n".join(case_lines) + "\n", encoding="utf-8")
    title = "# E 实验评测报告"
    if meta["execution_backend"] == "mock":
        title += " MOCK 仅用于程序验证"
    lines = [title, "",
             "**MOCK 数据不能支持任何正式性能结论。**" if meta["execution_backend"] == "mock"
             else "本报告仅汇总已保存的真实调用记录，结论范围以实际题数和完成状态为准。",
             "", f"划分：{meta['split']}；问题 {meta['question_count']} 条；查询组 {meta['group_count']} 组。",
             f"实验指纹：{meta['experiment_fingerprint']}。清单状态：{meta['manifest_status']}。",
             "主配置预先固定为 gat_2_mean，不按测试结果重新选择最佳配置。",
             "", "## 指标与解释", "",
             "EM 比较完整实体答案集合；F1 为每题集合 F1 的宏平均。运行异常及清单中尚未执行的 pending 题保留在选中题分母中并记零分。",
             f"评分分母题数：{meta['scoring_question_count']}；已保存记录：{meta['record_count']}；pending记录：{meta['pending_record_count']}。",
             "pending 的零分是完成度计分规则，不是模型已回答的性能；有pending的运行不能称为完整正式实验。",
             "组级结果先对同一逻辑查询的改写取平均，再对查询组取平均。",
             "图谱标准答案不支持比例 = 预测实体中不属于该题金标集合的比例，只对成功且非空的预测定义。",
             "该比例是固定图谱问答的代理指标，不能当作现实事实幻觉率。纯 LLM 没有引用不会自动被判为幻觉。",
             "空回答、明确的映射失败、以及有文本但没有实体 ID 的未决情况分别统计；不能把它们合成同一个幻觉指标。",
             "", "## 三组基线", "",
             "| 模式 | 已记录/评分分母/pending题及组数 | 全划分题数 | 状态 | EM | F1 | 回答率 | 运行错误 |",
             "|---|---:|---:|---|---:|---:|---:|---:|"]
    def result_line(variant):
        row = lookup.get(variant)
        if row is None:
            return f"| {variant} | 0/0 | unknown | missing | unknown | unknown | unknown | unknown |"
        group_count = row["groups"] if row["groups"] is not None else "unknown"
        scope = f"{row['observed_questions']}/{row['questions']}/{row['pending_count']}, M={group_count}"
        return (f"| {variant} | {scope} | "
                f"{row['dataset_questions'] if row['dataset_questions'] is not None else 'unknown'} | "
                f"{row['coverage_status']} | {_format(row['exact_match'], percent=True)} | "
                f"{_format(row['f1'], percent=True)} | {_format(row['answer_rate'], percent=True)} | "
                f"{row['error_count']} |")
    lines += [result_line(v) for v in MAIN_VARIANTS]
    lines += ["", "只有明确记录完整划分规模且覆盖全部题目时才标为 complete_split。少量 smoke/partial 运行不能当作全量测试。",
              "", "## 五组 GNN 消融", "",
              "| 配置 | 已记录/评分分母/pending题及组数 | 全划分题数 | 状态 | EM | F1 | 回答率 | 运行错误 |",
              "|---|---:|---:|---|---:|---:|---:|---:|"]
    lines += [result_line(v) for v in GNN_VARIANTS]
    lines += ["", "缺失配置保持 unknown，不补造数值，不据不完整数据描述提升趋势。",
              "", "## 配对组级 F1 差与 95% bootstrap 区间", "",
              "比较使用双方共同 question_id，并按选中清单补入pending零分。组是抽样单位；少于两组或pending组映射不完整时不生成区间。",
              "不重叠题数明确列出，区间只适用于共同覆盖的样本；小样本区间属于探索性描述。",
              "", "| 对比 候选减参考 | 配对题/组 | 差值 | 95% 区间 | 参考独有题 | 候选独有题 |",
              "|---|---:|---:|---|---:|---:|"]
    for pair in paired:
        interval = (f"[{_format(pair['ci_low'])}, {_format(pair['ci_high'])}]"
                    if pair["ci_low"] is not None else pair["status"])
        lines.append(f"| {pair['candidate']} - {pair['reference']} | "
                     f"{pair['paired_questions']}/{pair['paired_groups']} | "
                     f"{_format(pair['difference'])} | {interval} | "
                     f"{pair['reference_only_questions']} | {pair['candidate_only_questions']} |")
    lines += ["", "## 实际开销与逻辑开销", "",
              "逻辑开销描述该题完整推理所需的历史请求；实际开销只统计本次执行新增调用。缓存命中不能重复算作本次调用。",
              "缺失 token 或耗时保持 unknown。known_total 只是已报告部分之和，不能称为完整总量。",
              "按题累计的请求耗时不等同于并发实验的墙钟时间。这里不猜测 API 单价或精确账单。",
              "", "| 模式 | 逻辑输入/输出 token | 实际输入/输出 token | 逻辑/实际耗时合计 秒 | usage已报告题 |",
              "|---|---|---|---|---:|"]
    for row in summaries:
        lines.append(f"| {row['variant']} | {_format(row['logical_input_tokens_total'])} / "
                     f"{_format(row['logical_output_tokens_total'])} | "
                     f"{_format(row['actual_input_tokens_total'])} / {_format(row['actual_output_tokens_total'])} | "
                     f"{_format(row['logical_elapsed_seconds_total'])} / "
                     f"{_format(row['actual_elapsed_seconds_total'])} | "
                     f"{row['usage_reported_count']}/{row['questions']} |")
    models = sorted({m for row in summaries for m in row["served_models"]})
    lines += ["", "已记录服务模型：" + (", ".join(models) if models else "unknown") + "。",
              "若实际服务模型不一致，需在最终结论中说明，不能宣称严格固定了同一模型。",
              "", "## 错误分析", "",
              f"共 {len(cases)} 条未完全正确或待复核记录，见 error_cases.md 和 error_cases.jsonl。",
              "未解析名称、歧义名称、被拒绝实体ID分别计数；空ID且有诊断时单列normalization_unresolved（名称解析未完成），不认定自然语言答案正确。mapping_failure_count表示出现解析问题的题数，可能包含部分解析成功的回答。",
              "", "## 名称解析诊断", "",
              "| 配置 | 未解析名称题数 | 歧义名称题数 | 拒绝ID题数 | 空ID且解析未完成 | 模型尝试回答题数 |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['variant']} | {row['unresolved_name_question_count']} | "
                     f"{row['ambiguous_name_question_count']} | {row['rejected_id_question_count']} | "
                     f"{row['normalization_unresolved_count']} | {row['model_attempted_answer_count']} |")
    lines += ["", "这些是诊断题数，可在同一道题上重叠；详细名称、候选ID、题面及trace_file保存在错误清单。", "", "## 图表", ""]
    plot_error = None
    plot_files = []
    if plots:
        try:
            plot_files = _write_plots(output, summaries, meta)
        except ReportError as error:
            plot_error = str(error)
            lines += ["图表未生成：" + plot_error]
    else:
        lines += ["本次使用 no-plots，仅生成可复核的 CSV、JSON 与 Markdown。"]
    for name in plot_files:
        if name.endswith(".png"):
            lines += [f"![{name}]({name})", ""]
    (output / "E实验评测报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {**meta, "output_dir": str(output), "variants": len(summaries),
            "error_cases": len(cases), "plot_files": plot_files, "plot_error": plot_error}
