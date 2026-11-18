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
    gold_answer_ids: tuple[str, ...] = ()
    gold_evidence: tuple[Triple, ...] = ()


def _recall(found: set[object], expected: set[object]) -> float:
    return len(found & expected) / len(expected) if expected else 1.0


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
) -> dict[str, float | int]:
    """Measure answer/evidence recall and query time for a labeled QA split."""
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
        answers = set(case.gold_answer_ids)
        retrieved_answers = set(result.subgraph.node_ids)
        evidence = set(case.gold_evidence)
        retrieved_evidence = set(result.subgraph.triples)
        answer_recalls.append(_recall(retrieved_answers, answers))
        evidence_recalls.append(_recall(retrieved_evidence, evidence))
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

    divisor = count or 1
    return {
        "questions": count,
        "answer_recall_macro": sum(answer_recalls) / divisor,
        "answer_recall_micro": answer_hits / answer_total if answer_total else 1.0,
        "evidence_recall_macro": sum(evidence_recalls) / divisor,
        "evidence_recall_micro": evidence_hits / evidence_total if evidence_total else 1.0,
        "latency_mean_ms": sum(latencies_ms) / divisor,
        "latency_p50_ms": _percentile(latencies_ms, 0.50),
        "latency_p95_ms": _percentile(latencies_ms, 0.95),
        "nodes_mean": sum(node_counts) / divisor,
        "edges_mean": sum(edge_counts) / divisor,
        "node_truncation_rate": node_truncations / divisor,
        "edge_truncation_rate": edge_truncations / divisor,
    }
