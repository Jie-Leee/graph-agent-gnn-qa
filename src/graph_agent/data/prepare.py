"""A owns this interface; it is not a finished QA generator."""
from pathlib import Path


def prepare_dataset(raw_dir: Path, output_dir: Path, seed: int = 42) -> None:
    """Prefer teacher QA, otherwise create verifiable two-hop template QA.

    Record the fixed visible graph, compute complete answer sets, and group
    paraphrases of (topic entity, relation path) into a single QA split.
    Distinguish QA splits from original link-prediction edge splits.
    """
    raise NotImplementedError("A: implement real data loading and QA preparation.")
