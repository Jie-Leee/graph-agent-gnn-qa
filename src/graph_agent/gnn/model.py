"""Question-conditioned graph attention and evidence-backed candidate ranking."""

import re
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, global_add_pool, global_max_pool, global_mean_pool

from graph_agent.schemas import Candidate, Question, Subgraph


def question_tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


class GraphEncoder:
    """Maps training vocabulary and original KG triples to numeric features."""

    def __init__(self, entities: Iterable[str], relations: Iterable[str],
                 words: Iterable[str]) -> None:
        self.entities = {value: i for i, value in enumerate(entities, start=1)}
        self.relations = {value: i for i, value in enumerate(relations, start=1)}
        self.words = {value: i for i, value in enumerate(words, start=1)}

    def encode(self, question: Question, graph: Subgraph,
               answer_ids: Iterable[str] = ()) -> Data:
        indices = {entity: i for i, entity in enumerate(graph.node_ids)}
        topics, answers = set(graph.topic_entity_ids), set(answer_ids)
        source, target, edge_type = [], [], []
        for triple in graph.triples:
            head, tail = indices[triple.head], indices[triple.tail]
            relation = self.relations.get(triple.relation, 0) * 2
            source.extend((head, tail))
            target.extend((tail, head))
            edge_type.extend((relation, relation + 1))
        return Data(
            num_nodes=len(indices),
            entity=torch.tensor([self.entities.get(node, 0) for node in indices], dtype=torch.long),
            topic=torch.tensor([int(node in topics) for node in indices], dtype=torch.long),
            edge_index=torch.tensor([source, target], dtype=torch.long),
            edge_type=torch.tensor(edge_type, dtype=torch.long),
            words=torch.tensor([self.words.get(word, 0) for word in question_tokens(question.text)]
                               or [0], dtype=torch.long),
            candidate=torch.tensor([node not in topics and bool(graph.paths.get(node))
                                    for node in indices], dtype=torch.bool),
            label=torch.tensor([node in answers for node in indices], dtype=torch.float32),
        )

    def vocabulary(self) -> dict[str, list[str]]:
        return {"entities": list(self.entities), "relations": list(self.relations),
                "words": list(self.words)}


def batch_graphs(graphs: list[Data]) -> Batch:
    return Batch.from_data_list(graphs, follow_batch=["words"])


class CandidateGAT(nn.Module):
    def __init__(self, encoder: GraphEncoder, hidden_dim: int, layers: int,
                 graph_pool: str) -> None:
        super().__init__()
        self.pool = {"mean": global_mean_pool, "sum": global_add_pool,
                     "max": global_max_pool}[graph_pool]
        self.entity = nn.Embedding(len(encoder.entities) + 1, hidden_dim)
        self.topic = nn.Embedding(2, hidden_dim)
        self.word = nn.Embedding(len(encoder.words) + 1, hidden_dim)
        self.edge = nn.Embedding(2 * (len(encoder.relations) + 1), 32)
        self.layers = nn.ModuleList(
            GATConv(hidden_dim, hidden_dim, heads=4, concat=False,
                    edge_dim=32, fill_value=0.0)
            for _ in range(layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(layers))
        self.score = nn.Sequential(nn.Linear(4 * hidden_dim, hidden_dim), nn.GELU(),
                                   nn.Linear(hidden_dim, 1))

    def forward(self, batch: Batch) -> Tensor:
        question = global_mean_pool(self.word(batch.words), batch.words_batch,
                                    size=batch.num_graphs)
        node = self.entity(batch.entity) + self.topic(batch.topic)
        edge = self.edge(batch.edge_type)
        for layer, norm in zip(self.layers, self.norms):
            update = F.elu(layer(node, batch.edge_index, edge))
            node = norm(node + F.dropout(update, p=0.1, training=self.training))
        # Compress extreme sums without normalizing away the sum/mean distinction.
        graph = torch.asinh(self.pool(node, batch.batch, size=batch.num_graphs))
        query = question[batch.batch]
        features = torch.cat([node, graph[batch.batch], query, node * query], dim=-1)
        return self.score(features).squeeze(-1)


class GNNScorer:
    def __init__(self, checkpoint: Path, device: str = "cpu") -> None:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved.get("format") != 1:
            raise ValueError("Unsupported GNN checkpoint format")
        self.device = torch.device(device)
        self.encoder = GraphEncoder(**saved["vocabulary"])
        self.model = CandidateGAT(self.encoder, **saved["architecture"])
        self.model.load_state_dict(saved["weights"])
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def rank(self, question: Question, subgraph: Subgraph) -> list[Candidate]:
        graph = self.encoder.encode(question, subgraph)
        eligible = graph.candidate.nonzero(as_tuple=True)[0].tolist()
        if not eligible:
            return []
        logits = self.model(batch_graphs([graph]).to(self.device)).cpu().tolist()
        results = [
            Candidate(subgraph.node_ids[i], logits[i], subgraph.paths[subgraph.node_ids[i]])
            for i in eligible
        ]
        return sorted(results, key=lambda candidate: (-candidate.score, candidate.entity_id))
