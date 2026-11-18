"""Bounded question-aware retrieval over a directed knowledge graph."""
from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from itertools import count
from pathlib import Path
from time import perf_counter

from graph_agent.retrieval.entity_linking import (
    EntityLinker,
    aliases_by_entity,
    normalize_aliases,
    tokenize,
)
from graph_agent.schemas import PathStep, Question, Subgraph, Triple


_RELATION_EQUIVALENTS = {
    "born": "birth",
    "birth": "birth",
    "directed": "direct",
    "director": "direct",
    "directors": "direct",
    "directing": "direct",
}


def _relation_tokens(text: str) -> frozenset[str]:
    return frozenset(_RELATION_EQUIVALENTS.get(token, token) for token in tokenize(text))


@dataclass(frozen=True)
class _Traversal:
    triple: Triple
    neighbor: str
    is_reverse: bool


@dataclass(frozen=True)
class RetrievalStats:
    elapsed_seconds: float
    topic_count: int
    expanded_nodes: int
    retrieved_nodes: int
    retrieved_edges: int
    truncated_by_nodes: bool
    truncated_by_edges: bool


@dataclass(frozen=True)
class RetrievalResult:
    subgraph: Subgraph
    stats: RetrievalStats


def _metadata_aliases(
    values: Mapping[str, str | Iterable[str]] | None,
) -> dict[str, tuple[str, ...]]:
    return aliases_by_entity(values)


def _load_alias_table(path: str | Path | None) -> dict[str, tuple[str, ...]]:
    if path is None:
        return {}
    table: dict[str, list[str]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.rstrip("\r\n")
            if not line or line.lstrip().startswith("#"):
                continue
            fields = line.split("\t", 1)
            if len(fields) != 2 or not fields[0].strip() or not fields[1].strip():
                raise ValueError(f"Invalid name mapping at {path}:{line_number}; expected ID<TAB>name.")
            table[fields[0].strip()].append(fields[1].strip())
    return {key: normalize_aliases(values) for key, values in table.items()}


class GraphIndex:
    """Adjacency index with exact-name linking and bounded path-priority search.

    Traversal can follow both stored edges and their reverse direction. Returned
    triples are always the original KG records. A final BFS constructs shortest
    paths within the selected subgraph and records reverse steps explicitly.
    """

    def __init__(
        self,
        triples: Iterable[Triple],
        entity_names: Mapping[str, str | Iterable[str]] | None = None,
        relation_names: Mapping[str, str | Iterable[str]] | None = None,
    ) -> None:
        self.triples = tuple(dict.fromkeys(triples))
        self.outgoing: dict[str, list[Triple]] = defaultdict(list)
        self.incoming: dict[str, list[Triple]] = defaultdict(list)
        self._adjacency: dict[str, list[_Traversal]] = defaultdict(list)
        for triple in self.triples:
            self.outgoing[triple.head].append(triple)
            self.incoming[triple.tail].append(triple)
            self._adjacency[triple.head].append(_Traversal(triple, triple.tail, False))
            self._adjacency[triple.tail].append(_Traversal(triple, triple.head, True))

        self.entity_names = _metadata_aliases(entity_names)
        self.relation_names = _metadata_aliases(relation_names)
        self.entity_linker = EntityLinker(self.entity_names)
        self._relation_tokens: dict[str, tuple[frozenset[str], ...]] = {}
        for triple in self.triples:
            if triple.relation not in self._relation_tokens:
                aliases = (triple.relation, *self.relation_names.get(triple.relation, ()))
                self._relation_tokens[triple.relation] = tuple(
                    _relation_tokens(alias) for alias in aliases if tokenize(alias)
                )
        self._entity_tokens = {
            entity_id: tuple(_relation_tokens(alias) for alias in aliases if tokenize(alias))
            for entity_id, aliases in self.entity_names.items()
        }

    @classmethod
    def from_tsv(
        cls,
        triples_path: str | Path,
        entity_names_path: str | Path | None = None,
        relation_names_path: str | Path | None = None,
    ) -> GraphIndex:
        """Load a tab-separated ``head<TAB>relation<TAB>tail`` graph.

        Optional name files use ``entity-or-relation-ID<TAB>display name`` and
        may contain multiple rows for one ID. This matches common FB15k-237
        graph and text mapping files while keeping all raw data local.
        """
        triples: list[Triple] = []
        path = Path(triples_path)
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.rstrip("\r\n")
                if not line or line.lstrip().startswith("#"):
                    continue
                fields = line.split("\t")
                if len(fields) != 3 or any(not field.strip() for field in fields):
                    raise ValueError(
                        f"Invalid triple at {path}:{line_number}; expected head<TAB>relation<TAB>tail."
                    )
                triples.append(Triple(*(field.strip() for field in fields)))
        return cls(
            triples,
            entity_names=_load_alias_table(entity_names_path),
            relation_names=_load_alias_table(relation_names_path),
        )

    def link_entities(self, text: str, max_entities: int = 5) -> tuple[str, ...]:
        """Resolve exact entity-name mentions using the loaded alias table."""
        return self.entity_linker.link(text, max_entities=max_entities)

    def retrieve(
        self,
        question: Question,
        max_hops: int = 2,
        max_nodes: int = 200,
        max_edges: int = 800,
        max_entities: int = 5,
        allow_inverse: bool = True,
    ) -> Subgraph:
        """Retrieve a bounded local subgraph; explicit topic IDs take priority."""
        return self.retrieve_with_stats(
            question,
            max_hops=max_hops,
            max_nodes=max_nodes,
            max_edges=max_edges,
            max_entities=max_entities,
            allow_inverse=allow_inverse,
        ).subgraph

    def retrieve_with_stats(
        self,
        question: Question,
        max_hops: int = 2,
        max_nodes: int = 200,
        max_edges: int = 800,
        max_entities: int = 5,
        allow_inverse: bool = True,
    ) -> RetrievalResult:
        """Retrieve a subgraph and report bounded-search timing and truncation."""
        started = perf_counter()
        if max_hops < 0 or max_nodes < 1 or max_edges < 0 or max_entities < 1:
            raise ValueError("Expected nonnegative hops/edges and positive node/entity limits.")

        topics = tuple(dict.fromkeys(question.topic_entity_ids))
        if not topics:
            topics = self.link_entities(question.text, max_entities=max_entities)
        if len(topics) > max_nodes:
            raise ValueError("Topic count exceeds max_nodes.")

        question_tokens = self._question_tokens(question.text, topics)
        nodes: dict[str, None] = dict.fromkeys(topics)
        depths = {topic: 0 for topic in topics}
        edges: dict[Triple, None] = {}
        expanded: set[str] = set()
        truncated_by_nodes = False
        truncated_by_edges = False
        # Each layer initially gets a share of the remaining node/edge budget.
        # Overflow stays queued and can borrow unused capacity in another pass.
        # Within a layer, topics take turns and cumulative path relevance ranks
        # candidates. This preserves breadth without starving deeper evidence.
        queues: dict[int, list[list[tuple[float, int, int, str, _Traversal]]]] = {}
        next_owner: dict[int, int] = {}
        serial = count()
        relation_scores: dict[str, float] = {}
        entity_scores: dict[str, float] = {}

        def expand(node: str, depth: int, score: float, owner: int) -> None:
            nonlocal truncated_by_nodes
            if depth >= max_hops:
                return
            expanded.add(node)
            layer = queues.setdefault(depth + 1, [[] for _ in topics])
            for traversal in self._adjacency.get(node, ()):
                if traversal.is_reverse and not allow_inverse:
                    continue
                if traversal.neighbor not in nodes and len(nodes) >= max_nodes:
                    truncated_by_nodes = True
                    continue
                relation, neighbor = traversal.triple.relation, traversal.neighbor
                if relation not in relation_scores:
                    relation_scores[relation] = self._relation_relevance(relation, question_tokens)
                if neighbor not in entity_scores:
                    entity_scores[neighbor] = self._entity_relevance(neighbor, question_tokens)
                heappush(layer[owner], (
                    -(score + relation_scores[relation] + entity_scores[neighbor]),
                    -(depth + 1), next(serial), node, traversal,
                ))

        def prune_full_node_budget() -> None:
            nonlocal truncated_by_nodes
            # Once the node budget is full, discard impossible new-node edges
            # in one pass, retaining edges between selected nodes. This avoids
            # popping a hub's entire remainder one edge at a time.
            for layer in queues.values():
                for heap in layer:
                    kept = []
                    for entry in heap:
                        _, negative_depth, _, source, traversal = entry
                        if depths[source] != -negative_depth - 1:
                            continue
                        if traversal.neighbor not in nodes:
                            truncated_by_nodes = True
                        else:
                            kept.append(entry)
                    heap[:] = kept
                    heapify(heap)

        for owner, topic in enumerate(topics):
            expand(topic, 0, 0.0, owner)
        while any(heap for layer in queues.values() for heap in layer):
            for level in range(1, max_hops + 1):
                if level not in queues:
                    continue
                layer = queues[level]
                order = ((next_owner.get(level, 0) + offset) % len(topics) for offset in range(len(topics)))
                active = deque(owner for owner in order if layer[owner])
                start_nodes, start_edges = len(nodes), len(edges)
                remaining_hops = max_hops - level + 1
                node_share = (max_nodes - start_nodes + remaining_hops - 1) // remaining_hops
                edge_share = max(1, (max_edges - start_edges + remaining_hops - 1) // remaining_hops)
                while active:
                    owner = active.popleft()
                    heap = layer[owner]
                    while heap:
                        negative_score, negative_depth, _, node, traversal = heappop(heap)
                        depth = -negative_depth
                        if depths[node] != depth - 1:
                            continue  # A shorter route superseded this expansion.
                        triple, neighbor = traversal.triple, traversal.neighbor
                        if triple not in edges:
                            if neighbor not in nodes and len(nodes) >= max_nodes:
                                truncated_by_nodes = True
                                continue
                            if len(edges) >= max_edges:
                                truncated_by_edges = True
                                break
                            edges[triple] = None
                            nodes[neighbor] = None
                            added = True
                        else:
                            added = False

                        if depth < depths.get(neighbor, max_hops + 1):
                            depths[neighbor] = depth
                            expand(neighbor, depth, -negative_score, owner)
                            if len(nodes) == max_nodes:
                                prune_full_node_budget()
                        if added:
                            next_owner[level] = (owner + 1) % len(topics)
                            break
                    if truncated_by_edges:
                        break
                    if heap:
                        active.append(owner)
                    if ((node_share and len(nodes) - start_nodes >= node_share)
                            or len(edges) - start_edges >= edge_share):
                        break  # Defer remaining candidates; do not call it truncation yet.
                if truncated_by_edges:
                    break
            if truncated_by_edges:
                break

        # Search priority may discover a longer route first. Rebuild paths from
        # the selected original edges so every returned path is shortest,
        # continuous, and consistent with the other returned paths.
        adjacency: dict[str, list[_Traversal]] = defaultdict(list)
        for triple in edges:
            adjacency[triple.head].append(_Traversal(triple, triple.tail, False))
            if allow_inverse:
                adjacency[triple.tail].append(_Traversal(triple, triple.head, True))
        paths: dict[str, tuple[Triple, ...]] = {topic: () for topic in topics}
        path_steps: dict[str, tuple[PathStep, ...]] = {topic: () for topic in topics}
        queue = deque(topics)
        while queue:
            node = queue.popleft()
            for traversal in adjacency.get(node, ()):
                neighbor = traversal.neighbor
                if neighbor in paths:
                    continue
                step = PathStep(traversal.triple, node, neighbor, traversal.is_reverse)
                paths[neighbor] = paths[node] + (traversal.triple,)
                path_steps[neighbor] = path_steps[node] + (step,)
                queue.append(neighbor)

        subgraph = Subgraph(
            topic_entity_ids=topics,
            node_ids=tuple(nodes),
            triples=tuple(edges),
            paths=paths,
            path_steps=path_steps,
        )
        stats = RetrievalStats(
            elapsed_seconds=perf_counter() - started,
            topic_count=len(topics),
            expanded_nodes=len(expanded),
            retrieved_nodes=len(nodes),
            retrieved_edges=len(edges),
            truncated_by_nodes=truncated_by_nodes,
            truncated_by_edges=truncated_by_edges,
        )
        return RetrievalResult(subgraph, stats)

    def _question_tokens(self, text: str, topics: tuple[str, ...]) -> set[str]:
        """Remove only the actual topic-name spans before relation normalization."""
        tokens = tokenize(text)
        linker = EntityLinker({topic: self.entity_names.get(topic, ()) for topic in topics})
        mentions = linker.match(text, max_entities=max(1, len(topics)))
        covered = {position for match in mentions for position in range(match.start_token, match.end_token)}
        return {
            _RELATION_EQUIVALENTS.get(token, token)
            for position, token in enumerate(tokens) if position not in covered
        }

    def _relation_relevance(self, relation: str, question_tokens: set[str]) -> float:
        relation_variants = self._relation_tokens.get(relation, ())
        relation_overlap = max((len(tokens & question_tokens) for tokens in relation_variants), default=0)
        score = float(relation_overlap)
        if any(tokens and tokens <= question_tokens for tokens in relation_variants):
            score += 3.0
        return score

    def _entity_relevance(self, entity_id: str, question_tokens: set[str]) -> float:
        entity_variants = self._entity_tokens.get(entity_id, ())
        entity_overlap = max((len(tokens & question_tokens) for tokens in entity_variants), default=0)
        return min(entity_overlap, 3) * 0.25
