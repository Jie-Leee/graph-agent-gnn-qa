"""Offline retrieval quality and latency measurements.

Gold answers and evidence are read only after each query has completed, so the
benchmark never passes supervision into the retrieval API.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math

from graph_agent.retrieval.graph import GraphIndex
from graph_agent.schemas import Question, Triple


@dataclass(frozen=True)
class RetrievalCase:
    question: Question
    # Recall is undefined for missing or empty gold sets. Those cases still
    # contribute to query-time statistics, but not to that recall denominator.
    gold_answer_ids: tuple[str, ...] | None = None
    gold_evidence: tuple[Triple, ...] | None = None


def _recall(found: set[object], expected: set[object]) -> float | None:
    return len(found & expected) / len(expected) if expected else None


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    rank = max(1, math.ceil(percentile * len(values)))
    return sorted(values)[rank - 1]


def benchmark_retrieval(
    index: GraphIndex,
    cases: Iterable[RetrievalCase],
    *,
    max_hops: int = 2,
    max_nodes: int = 200,
    max_edges: int = 800,
    max_entities: int = 5,
    allow_inverse: bool = True,
) -> dict[str, float | int | None]:
    """Measure recall over nonempty gold sets and latency over all questions.

    Answer and evidence recall have separate evaluated/skipped counts. If no
    case has a nonempty gold set for a metric, its macro and micro recall are
    ``None`` (JSON ``null``), not a successful retrieval. An empty dataset is an
    error. Empty gold sets are not an evaluation of unanswerable questions.
    """
    answer_recalls: list[float] = []
    evidence_recalls: list[float] = []
    answer_hits = answer_total = evidence_hits = evidence_total = 0
    latencies_ms: list[float] = []
    node_counts: list[int] = []
    edge_counts: list[int] = []
    node_truncations = edge_truncations = 0
    count = 0

    for case in cases:
        # Only the Question crosses into retrieval; gold fields stay in this
        # evaluation wrapper and are compared against the completed result.
        result = index.retrieve_with_stats(
            case.question,
            max_hops=max_hops,
            max_nodes=max_nodes,
            max_edges=max_edges,
            max_entities=max_entities,
            allow_inverse=allow_inverse,
        )
        answers = set(case.gold_answer_ids or ())
        retrieved_answers = set(result.subgraph.node_ids)
        evidence = set(case.gold_evidence or ())
        retrieved_evidence = set(result.subgraph.triples)
        answer_recall = _recall(retrieved_answers, answers)
        evidence_recall = _recall(retrieved_evidence, evidence)
        if answer_recall is not None:
            answer_recalls.append(answer_recall)
        if evidence_recall is not None:
            evidence_recalls.append(evidence_recall)
        answer_hits += len(retrieved_answers & answers)
        answer_total += len(answers)
        evidence_hits += len(retrieved_evidence & evidence)
        evidence_total += len(evidence)
        latencies_ms.append(result.stats.elapsed_seconds * 1000.0)
        node_counts.append(result.stats.retrieved_nodes)
        edge_counts.append(result.stats.retrieved_edges)
        node_truncations += int(result.stats.truncated_by_nodes)
        edge_truncations += int(result.stats.truncated_by_edges)
        count += 1

    if not count:
        raise ValueError("Cannot evaluate an empty QA dataset.")
    return {
        "questions": count,
        "answer_evaluated_questions": len(answer_recalls),
        "answer_skipped_questions": count - len(answer_recalls),
        "evidence_evaluated_questions": len(evidence_recalls),
        "evidence_skipped_questions": count - len(evidence_recalls),
        "answer_recall_macro": sum(answer_recalls) / len(answer_recalls) if answer_recalls else None,
        "answer_recall_micro": answer_hits / answer_total if answer_total else None,
        "evidence_recall_macro": sum(evidence_recalls) / len(evidence_recalls) if evidence_recalls else None,
        "evidence_recall_micro": evidence_hits / evidence_total if evidence_total else None,
        "latency_mean_ms": sum(latencies_ms) / count,
        "latency_p50_ms": _percentile(latencies_ms, 0.50),
        "latency_p95_ms": _percentile(latencies_ms, 0.95),
        "nodes_mean": sum(node_counts) / count,
        "edges_mean": sum(edge_counts) / count,
        "node_truncation_rate": node_truncations / count,
        "edge_truncation_rate": edge_truncations / count,
    }
