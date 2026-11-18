"""D's adapter for B's named, bidirectional, measured retrieval tool."""
from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

from graph_agent.agent.evidence import EntityCatalog
from graph_agent.data.loader import load_raw, resolve_raw_files
from graph_agent.retrieval import graph as graph_module
from graph_agent.retrieval.graph import GraphIndex
from graph_agent.schemas import Question, Subgraph


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GraphRetrievalTool:
    """Keep train-only construction and retrieval provenance in one place."""
    def __init__(self, index: GraphIndex, *, provenance: dict | None = None):
        self.index = index
        self.last_stats: dict | None = None
        self.provenance = dict(provenance or {})

    @classmethod
    def from_raw(cls, raw_dir: Path) -> tuple[GraphRetrievalTool, EntityCatalog]:
        raw = load_raw(raw_dir)
        files = resolve_raw_files(raw_dir)
        tool = cls(GraphIndex(raw.train_triples, entity_names=raw.entity_text,
                              relation_names=raw.relation_text), provenance={
            "visible_split": "train", "train_sha256": _sha256(files["train"]),
            "entity_names_sha256": _sha256(files["entity_text"]),
            "relation_names_sha256": _sha256(files["relation_text"]),
        })
        return tool, EntityCatalog(raw.entity_text, raw.relation_text)

    def describe(self) -> dict:
        source = Path(graph_module.__file__).read_bytes().replace(b"\r\n", b"\n")
        return {"backend": "GraphIndex", "indexed_triples": len(self.index.triples),
                "retrieval_code_sha256_lf": hashlib.sha256(source).hexdigest(), **self.provenance}

    def retrieve(self, question: Question, max_hops: int = 2, max_nodes: int = 200,
                 max_edges: int = 800, allow_inverse: bool = True) -> Subgraph:
        self.last_stats = None
        result = self.index.retrieve_with_stats(question, max_hops=max_hops,
            max_nodes=max_nodes, max_edges=max_edges, allow_inverse=allow_inverse)
        self.last_stats = asdict(result.stats)
        return result.subgraph
