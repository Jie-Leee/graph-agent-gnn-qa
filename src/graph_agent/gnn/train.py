"""C: complete this entry point with real QA supervision."""
from pathlib import Path


def train(train_path: Path, validation_path: Path, output_dir: Path) -> None:
    """Train each declared configuration; never select on final test labels."""
    raise NotImplementedError("C: implement supervised candidate scoring training.")
