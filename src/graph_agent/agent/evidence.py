"""Rank/serialize retrieved paths without access to QA supervision."""
from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol

from graph_agent.schemas import Candidate, PathStep, Question, Subgraph, Triple


def encode(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def normalized_name(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def lexical_tokens(text: str) -> set[str]:
    """Fixed inflection handling; no QA labels, learned weights or gold paths."""
    synonyms = {"born": "birth", "birthplace": "birth", "birthplaces": "birth",
                "director": "direct", "directors": "direct", "directed": "direct"}
    result = set()
    for word in re.findall(r"[^\W_]+", text.casefold()):
        if word in synonyms:
            word = synonyms[word]
        elif len(word) > 4 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 4 and word.endswith("ed"):
            word = word[:-2]
        elif len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        result.add(word)
    return result


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


def text_rank(question: Question, subgraph: Subgraph, catalog: EntityCatalog, *,
              allow_inverse: bool = True, max_hops: int = 2) -> list[Candidate]:
    """Fixed lexical ranking over real paths, independent of any gold labels.

    B returns shortest paths for display, which may use an unrelated shortcut.
    Select the best matching <=2-hop path INSIDE its already bounded subgraph.
    This does not retrieve extra nodes/edges or use the pre-labelled QA path.
    """
    stopwords = {"a", "an", "the", "of", "in", "on", "at", "to", "from", "for", "by",
                 "any", "every", "all", "list", "return", "what", "which", "who", "where",
                 "are", "is", "was", "were", "does", "do", "and", "or", "s"}
    words = lexical_tokens(question.text) - stopwords
    adjacency = defaultdict(list)
    relation_words = {}
    topics = set(subgraph.topic_entity_ids)
    for triple in subgraph.triples:
        adjacency[triple.head].append((triple.tail, triple))
        if allow_inverse:
            adjacency[triple.tail].append((triple.head, triple))
        relation_words.setdefault(triple.relation,
            lexical_tokens(catalog.relation_names.get(triple.relation, triple.relation)) - stopwords)
    best = {}

    def consider(entity_id, path):
        if entity_id in topics:
            return
        matched = set().union(*(relation_words[t.relation] for t in path)) & words
        score = float(len(matched))
        key = (-score, len(path), tuple((t.head, t.relation, t.tail) for t in path))
        if entity_id not in best or key < best[entity_id][0]:
            best[entity_id] = (key, Candidate(entity_id, score, path))

    for topic in subgraph.topic_entity_ids:
        for middle, first in adjacency[topic]:
            if middle == topic:
                continue
            consider(middle, (first,))
            if max_hops >= 2:
                for target, second in adjacency[middle]:
                    if target not in (topic, middle):
                        consider(target, (first, second))
    return sorted((value[1] for value in best.values()), key=lambda c: (-c.score, c.entity_id))


def candidate_steps(candidate: Candidate, graph: Subgraph) -> tuple[PathStep, ...]:
    if candidate.entity_id not in graph.node_ids or not math.isfinite(candidate.score):
        raise ValueError("Scorer returned an unknown candidate or non-finite score")
    if not candidate.evidence:
        raise ValueError("Scorer must supply a real evidence path")
    edges = set(graph.triples)
    if any(triple not in edges for triple in candidate.evidence):
        raise ValueError("Scorer returned fabricated evidence")
    for topic in graph.topic_entity_ids:
        node, steps = topic, []
        for triple in candidate.evidence:
            if triple.head == node:
                step = PathStep(triple, node, triple.tail, False)
            elif triple.tail == node:
                step = PathStep(triple, node, triple.head, True)
            else:
                break
            steps.append(step)
            node = step.target_entity_id
        else:
            if node == candidate.entity_id:
                result = tuple(steps)
                if graph.paths.get(candidate.entity_id) == candidate.evidence and candidate.entity_id in graph.path_steps:
                    if graph.path_steps[candidate.entity_id] != result:
                        raise ValueError("Retriever path direction metadata disagrees with evidence")
                return result
    raise ValueError("Evidence path is disconnected or does not reach candidate")


def validate_candidate(candidate: Candidate, graph: Subgraph) -> None:
    candidate_steps(candidate, graph)


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
    entities, relations = {}, {}
    for candidate in candidates:
        steps = candidate_steps(candidate, graph)
        if candidate.entity_id in ids:
            continue
        # Scores live on different scales (lexical overlap vs learned GNN).
        # Only ordering/selection differs between answer contexts.
        row = {"entity": catalog.entity(candidate.entity_id),
               "path": [{"triple": [step.triple.head, step.triple.relation, step.triple.tail],
                         "source_id": step.source_entity_id,
                         "target_id": step.target_entity_id, "is_reverse": step.is_reverse}
                        for step in steps]}
        next_entities, next_relations = dict(entities), dict(relations)
        for triple in candidate.evidence:
            for entity_id in (triple.head, triple.tail):
                next_entities[entity_id] = catalog.entity_names.get(entity_id, entity_id)
            next_relations[triple.relation] = catalog.relation_names.get(triple.relation, triple.relation)
        proposed = {"entities": next_entities, "relations": next_relations, "candidates": rows + [row]}
        if len(encode(proposed).encode("utf-8")) > byte_budget:
            continue  # Never split a path or invent a shortened path.
        rows.append(row)
        ids.append(candidate.entity_id)
        triples.update(dict.fromkeys(candidate.evidence))
        entities, relations = next_entities, next_relations
    payload = {"entities": entities, "relations": relations, "candidates": rows} if rows else {"candidates": []}
    return EvidenceContext(encode(payload), tuple(ids), tuple(triples))


def merge_subgraphs(graphs: tuple[Subgraph, ...], topics: tuple[str, ...]) -> Subgraph:
    nodes, edges, paths, steps = dict.fromkeys(topics), {}, {}, {}
    for graph in graphs:
        nodes.update(dict.fromkeys(graph.node_ids))
        edges.update(dict.fromkeys(graph.triples))
        for entity_id, path in graph.paths.items():
            if entity_id not in paths or len(path) < len(paths[entity_id]):
                paths[entity_id] = path
                if entity_id in graph.path_steps:
                    steps[entity_id] = graph.path_steps[entity_id]
                else:
                    steps.pop(entity_id, None)
    return Subgraph(topics, tuple(nodes), tuple(edges), paths, steps)
