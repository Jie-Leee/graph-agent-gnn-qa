"""Rank/serialize retrieved paths without access to QA supervision."""
from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol

from graph_agent.schemas import Candidate, Question, Subgraph, Triple


def encode(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def normalized_name(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


class EntityCatalog:
    def __init__(self, entity_names: dict[str, str] | None = None,
                 relation_names: dict[str, str] | None = None):
        self.entity_names = dict(entity_names or {})
        self.relation_names = dict(relation_names or {})
        self.by_name = defaultdict(list)
        for entity_id, name in self.entity_names.items():
            self.by_name[normalized_name(name)].append(entity_id)

    def entity(self, entity_id: str) -> dict:
        return {"id": entity_id, "name": self.entity_names.get(entity_id, entity_id)}

    def triple(self, triple: Triple) -> dict:
        return {"head": self.entity(triple.head),
                "relation": {"id": triple.relation,
                             "name": self.relation_names.get(triple.relation, triple.relation)},
                "tail": self.entity(triple.tail)}

    def resolve(self, name: str) -> tuple[str, ...]:
        """All exact normalized matches; ambiguity is never silently resolved."""
        return tuple(self.by_name.get(normalized_name(name), ()))


class CandidateScorer(Protocol):
    def rank(self, question: Question, subgraph: Subgraph) -> list[Candidate]: ...


def text_rank(question: Question, subgraph: Subgraph, catalog: EntityCatalog) -> list[Candidate]:
    """Fixed lexical relation-overlap baseline, with ID tie-breaking."""
    words = set(re.findall(r"\w+", question.text.casefold()))
    candidates = []
    for entity_id, path in subgraph.paths.items():
        if entity_id in subgraph.topic_entity_ids or not path:
            continue
        relations = " ".join(catalog.relation_names.get(t.relation, t.relation) for t in path)
        overlap = len(words & set(re.findall(r"\w+", relations.casefold())))
        candidates.append(Candidate(entity_id, float(overlap), path))
    return sorted(candidates, key=lambda c: (-c.score, c.entity_id))


def validate_candidate(candidate: Candidate, graph: Subgraph) -> None:
    if candidate.entity_id not in graph.node_ids or not math.isfinite(candidate.score):
        raise ValueError("Scorer returned an unknown candidate or non-finite score")
    if not candidate.evidence:
        raise ValueError("Scorer must supply a real evidence path")
    edges = set(graph.triples)
    current = set(graph.topic_entity_ids)
    # Traversal may go against a stored edge (B's future inverse retrieval), but
    # the original head/relation/tail direction must be preserved verbatim.
    for triple in candidate.evidence:
        if triple not in edges:
            raise ValueError("Scorer returned fabricated evidence")
        next_nodes = set()
        if triple.head in current:
            next_nodes.add(triple.tail)
        if triple.tail in current:
            next_nodes.add(triple.head)
        if not next_nodes:
            raise ValueError("Scorer returned a disconnected evidence path")
        current = next_nodes
    if candidate.entity_id not in current:
        raise ValueError("Evidence path does not reach candidate")


@dataclass(frozen=True)
class EvidenceContext:
    text: str
    candidate_ids: tuple[str, ...]
    triples: tuple[Triple, ...]


def build_context(candidates: list[Candidate], graph: Subgraph, catalog: EntityCatalog,
                  byte_budget: int) -> EvidenceContext:
    # Use UTF-8 bytes as a conservative tokenizer-independent cap. This is NOT
    # measured token usage; only the provider reports actual token counts.
    rows, ids, triples = [], [], {}
    for candidate in candidates:
        validate_candidate(candidate, graph)
        if candidate.entity_id in ids:
            continue
        # Scores live on different scales (lexical overlap vs learned GNN).
        # Only ordering/selection differs between answer contexts.
        row = {"entity": catalog.entity(candidate.entity_id),
               "path": [catalog.triple(t) for t in candidate.evidence]}
        if len(encode({"candidates": rows + [row]}).encode("utf-8")) > byte_budget:
            continue  # Never split a path or invent a shortened path.
        rows.append(row)
        ids.append(candidate.entity_id)
        triples.update(dict.fromkeys(candidate.evidence))
    return EvidenceContext(encode({"candidates": rows}), tuple(ids), tuple(triples))


def merge_subgraphs(graphs: tuple[Subgraph, ...], topics: tuple[str, ...]) -> Subgraph:
    nodes, edges, paths = dict.fromkeys(topics), {}, {}
    for graph in graphs:
        nodes.update(dict.fromkeys(graph.node_ids))
        edges.update(dict.fromkeys(graph.triples))
        for entity_id, path in graph.paths.items():
            if entity_id not in paths or len(path) < len(paths[entity_id]):
                paths[entity_id] = path
    return Subgraph(topics, tuple(nodes), tuple(edges), paths)
