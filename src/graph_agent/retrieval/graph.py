"""Directed BFS baseline for the offline smoke demo.

No gold relation path or answer is used. Question-aware ranking, entity linking
and inverse-relation handling remain B's work.
"""
from collections import defaultdict, deque
from collections.abc import Iterable
from graph_agent.schemas import Question, Subgraph, Triple


class GraphIndex:
    def __init__(self, triples: Iterable[Triple]) -> None:
        self.outgoing: dict[str, list[Triple]] = defaultdict(list)
        for triple in dict.fromkeys(triples):
            self.outgoing[triple.head].append(triple)

    def retrieve(self, question: Question, max_hops: int = 2,
                 max_nodes: int = 200, max_edges: int = 800) -> Subgraph:
        if max_hops < 0 or max_nodes < 1 or max_edges < 0:
            raise ValueError("Expected nonnegative hops/edges and positive nodes.")
        topics = tuple(dict.fromkeys(question.topic_entity_ids))
        if len(topics) > max_nodes:
            raise ValueError("Topic count exceeds max_nodes.")
        nodes = dict.fromkeys(topics)
        paths: dict[str, tuple[Triple, ...]] = {topic: () for topic in topics}
        queue = deque((topic, 0) for topic in topics)
        edges: dict[Triple, None] = {}
        while queue and len(edges) < max_edges:
            node, depth = queue.popleft()
            if depth >= max_hops:
                continue
            for triple in self.outgoing.get(node, []):
                if len(edges) >= max_edges:
                    break
                if triple.tail not in nodes:
                    if len(nodes) >= max_nodes:
                        continue
                    nodes[triple.tail] = None
                    paths[triple.tail] = paths[node] + (triple,)
                    queue.append((triple.tail, depth + 1))
                edges[triple] = None
        return Subgraph(topics, tuple(nodes), tuple(edges), paths)
