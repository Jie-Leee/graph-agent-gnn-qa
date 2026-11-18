"""Train and select evidence rankers on the fixed QA splits."""

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import torch
import torch_geometric
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data

from graph_agent.data.dataset import load_examples
from graph_agent.data.loader import load_raw
from graph_agent.gnn.model import CandidateGAT, GraphEncoder, batch_graphs, question_tokens
from graph_agent.retrieval.graph import GraphIndex
from graph_agent.schemas import QAExample


def encode_split(index: GraphIndex, encoder: GraphEncoder, examples: tuple[QAExample, ...],
                 budget: dict, split: str) -> list[Data]:
    started = time.perf_counter()
    graphs = []
    for count, example in enumerate(examples, 1):
        subgraph = index.retrieve(example.question, **budget)
        graph = encoder.encode(example.question, subgraph, example.answer_ids)
        graph.gold_count = len(set(example.answer_ids))
        graphs.append(graph)
        if count % 250 == 0 or count == len(examples):
            print(f"{split}: encoded {count}/{len(examples)} in "
                  f"{time.perf_counter() - started:.1f}s", flush=True)
    return graphs


def loader(graphs: list[Data], batch_size: int, seed: int, shuffle: bool) -> DataLoader:
    return DataLoader(graphs, batch_size=batch_size, shuffle=shuffle, collate_fn=batch_graphs,
                      generator=torch.Generator().manual_seed(seed))


def graph_losses(logits: torch.Tensor, batch: Batch) -> torch.Tensor:
    # Keep each graph and each available label class on a comparable scale.
    # A graph with no retrieved positives can still supervise its negatives.
    ids = batch.batch
    positive = batch.candidate & batch.label.bool()
    negative = batch.candidate & ~batch.label.bool()
    losses = F.binary_cross_entropy_with_logits(logits, batch.label, reduction="none")
    total = losses.new_zeros(batch.num_graphs)
    present = losses.new_zeros(batch.num_graphs)
    for mask in (positive, negative):
        weights = mask.to(losses.dtype)
        counts = losses.new_zeros(batch.num_graphs).index_add_(0, ids, weights)
        sums = losses.new_zeros(batch.num_graphs).index_add_(0, ids, losses * weights)
        total = total + sums / counts.clamp_min(1)
        present = present + (counts > 0).to(losses.dtype)
    return total / present.clamp_min(1)


@torch.inference_mode()
def evaluate(model: CandidateGAT, batches: DataLoader, device: torch.device) -> dict:
    model.eval()
    count = loss_sum = available = recall = map10 = hit1 = hitless = 0
    for batch in batches:
        batch.to(device)
        logits = model(batch)
        loss_sum += graph_losses(logits, batch).sum().item()
        scores = logits.cpu().tolist()
        labels = batch.label.cpu().tolist()
        eligible = batch.candidate.cpu().tolist()
        boundaries = batch.ptr.cpu().tolist()
        for start, end, gold in zip(boundaries, boundaries[1:], batch.gold_count.tolist()):
            indices = [i for i in range(start, end) if eligible[i]]
            ordered = sorted(indices, key=lambda i: (-scores[i], i))[:10]
            found = sum(labels[i] for i in indices)
            available += found / gold
            hitless += not found
            hit1 += bool(ordered and labels[ordered[0]])
            hits = precision_sum = 0
            for rank, i in enumerate(ordered, 1):
                if labels[i]:
                    hits += 1
                    precision_sum += hits / rank
            recall += hits / gold
            map10 += precision_sum / min(gold, 10)
            count += 1
    return {"loss": loss_sum / count, "candidate_recall": available / count,
            "recall_at_10": recall / count, "map_at_10": map10 / count,
            "hit_at_1": hit1 / count, "no_positive_questions": hitless, "questions": count}


def train(train_path: Path, validation_path: Path, output_dir: Path, *,
          raw_path: Path = Path("data/raw"), config_path: Path = Path("configs/experiment.json"),
          names: tuple[str, ...] = (), epochs: int = 20, patience: int = 5,
          batch_size: int = 24, learning_rate: float = 0.001,
          device_name: str = "auto") -> None:
    started = time.perf_counter()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    configurations = config["gnn"]["configs"]
    unknown = set(names) - {row["name"] for row in configurations}
    if unknown:
        raise ValueError(f"Unknown GAT configurations: {sorted(unknown)}")
    if min(epochs, patience, batch_size) < 1 or not 0 < learning_rate < float("inf"):
        raise ValueError("Epochs, patience, batch size and learning rate must be positive")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    seed = config["seed"]
    budget = {key: config["retrieval"][key] for key in ("max_hops", "max_nodes", "max_edges")}
    training, validation = load_examples(train_path), load_examples(validation_path)
    if not training or not validation or any(not example.answer_ids for example in (*training, *validation)):
        raise ValueError("Training and validation splits require questions with labeled answers")

    raw = load_raw(raw_path)
    triples = raw.train_triples
    entities = sorted({triple.head for triple in triples} | {triple.tail for triple in triples})
    relations = sorted({triple.relation for triple in triples})
    words = sorted({word for example in training for word in question_tokens(example.question.text)})
    encoder = GraphEncoder(entities, relations, words)
    index = GraphIndex(triples, raw.entity_text, raw.relation_text)
    print(f"device={device} torch={torch.__version__} pyg={torch_geometric.__version__}", flush=True)
    train_graphs = encode_split(index, encoder, training, budget, "train")
    val_graphs = encode_split(index, encoder, validation, budget, "validation")
    preprocess_seconds = time.perf_counter() - started

    provenance = {
        "seed": seed, "batch_size": batch_size, "learning_rate": learning_rate,
        "epochs_limit": epochs, "patience": patience, "retrieval": budget, "device": str(device),
        "torch": str(torch.__version__), "pyg": torch_geometric.__version__,
        "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "validation_sha256": hashlib.sha256(validation_path.read_bytes()).hexdigest(),
        "train_questions": len(train_graphs), "validation_questions": len(val_graphs),
        "preprocess_seconds": round(preprocess_seconds, 3),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    val_batches = loader(val_graphs, batch_size, seed, shuffle=False)
    summary = {}
    for item in configurations:
        name = item["name"]
        if names and name not in names:
            continue
        run_dir = output_dir / name
        run_dir.mkdir(exist_ok=True)
        random.seed(seed)
        torch.manual_seed(seed)
        train_batches = loader(train_graphs, batch_size, seed, shuffle=True)
        architecture = {"hidden_dim": config["gnn"]["hidden_dim"],
                        "layers": item["layers"], "graph_pool": item["graph_pool"]}
        model = CandidateGAT(encoder, **architecture).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        best, rows = None, []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()
            model.train()
            train_loss = 0.0
            for batch in train_batches:
                batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                losses = graph_losses(model(batch), batch)
                losses.mean().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                train_loss += losses.detach().sum().item()
            row = {"epoch": epoch, "train_loss": train_loss / len(train_graphs),
                   **evaluate(model, val_batches, device)}
            row["seconds"] = round(time.perf_counter() - epoch_start, 3)
            rows.append(row)
            print(f"{name} {json.dumps(row)}", flush=True)
            if best is None or row["map_at_10"] > best["map_at_10"]:
                best = row
                temporary = run_dir / "best.pt.tmp"
                torch.save({"format": 1, "architecture": architecture,
                            "vocabulary": encoder.vocabulary(), "weights": model.state_dict()},
                           temporary)
                temporary.replace(run_dir / "best.pt")
            with (run_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=row.keys())
                writer.writeheader()
                writer.writerows(rows)
            if epoch - best["epoch"] >= patience:
                break
        peak_mib = (round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
                    if device.type == "cuda" else None)
        result = {"architecture": architecture, "best": best, "epochs_run": len(rows),
                  "epoch_seconds_total": round(sum(row["seconds"] for row in rows), 3),
                  "peak_cuda_allocated_mib": peak_mib}
        (run_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        summary[name] = result
        (output_dir / "summary.json").write_text(
            json.dumps({"training": provenance, "configurations": summary}, indent=2) + "\n",
            encoding="utf-8",
        )
        del optimizer, model
    print(f"Completed in {time.perf_counter() - started:.1f}s: {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--validation", type=Path, default=Path("data/processed/validation.jsonl"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/gnn"))
    parser.add_argument("--name", action="append", default=[], help="Train a subset of configurations")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    train(args.train, args.validation, args.output, raw_path=args.raw, config_path=args.config,
          names=tuple(args.name), epochs=args.epochs, patience=args.patience,
          batch_size=args.batch_size, learning_rate=args.learning_rate, device_name=args.device)


if __name__ == "__main__":
    main()
