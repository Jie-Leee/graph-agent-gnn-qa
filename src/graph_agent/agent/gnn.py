"""Lazy, CPU-friendly adapter for C's real trained GAT checkpoint."""
from __future__ import annotations

import hashlib
from pathlib import Path

from graph_agent.schemas import Candidate, Question, Subgraph


class CheckpointScorer:
    def __init__(self, checkpoint: Path, *, device: str = "cpu"):
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                "GNN checkpoint missing: obtain C's trained gat_2_mean/best.pt "
                "and pass --checkpoint PATH. No text/random fallback is used.")
        try:
            import torch
            from graph_agent.gnn.model import GNNScorer
        except ImportError:
            raise RuntimeError("Activate conda langchain and install requirements-gnn.txt for GNN inference") from None
        torch.set_num_threads(min(torch.get_num_threads(), 4))
        self.checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        self.scorer = GNNScorer(checkpoint, device=device)
        self.device = str(self.scorer.device)
        self.checkpoint_name = checkpoint.name

    def describe(self) -> dict:
        return {"backend": "C.GNNScorer", "checkpoint_sha256": self.checkpoint_sha256,
                "checkpoint_name": self.checkpoint_name, "device": self.device,
                "architecture": {"hidden_dim": self.scorer.model.entity.embedding_dim,
                                 "layers": len(self.scorer.model.layers),
                                 "graph_pool": {"global_mean_pool": "mean", "global_add_pool": "sum",
                                                "global_max_pool": "max"}[self.scorer.model.pool.__name__]},
                "evidence_policy": "shared_question_relevant_paths_v1"}

    def rank(self, question: Question, subgraph: Subgraph) -> list[Candidate]:
        return self.scorer.rank(question, subgraph)
