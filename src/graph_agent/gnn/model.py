"""C owns this module. No random scores are used as fake GNN outputs."""
from pathlib import Path
from graph_agent.schemas import Candidate, Question, Subgraph


class GNNScorer:
    def __init__(self, checkpoint: Path) -> None:
        self.checkpoint = checkpoint

    def rank(self, question: Question, subgraph: Subgraph) -> list[Candidate]:
        """Use a trained GAT; score entities with question q, node h_v and graph g.

        Preserve relation type/direction and topic flags. Graph pooling g must
        participate in scoring for the pooling ablation to matter. Evidence
        paths must be real paths from the retrieved subgraph.
        """
        raise NotImplementedError("C: implement trained GAT scoring before use.")
