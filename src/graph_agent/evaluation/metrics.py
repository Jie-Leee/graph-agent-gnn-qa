"""Entity-set metrics; inputs must use canonical entity IDs."""
from collections.abc import Iterable


def answer_metrics(predicted: Iterable[str], gold: Iterable[str]) -> dict[str, float]:
    predicted_set, gold_set = set(predicted), set(gold)
    overlap = len(predicted_set & gold_set)
    precision = overlap / len(predicted_set) if predicted_set else float(not gold_set)
    recall = overlap / len(gold_set) if gold_set else float(not predicted_set)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"exact_match": float(predicted_set == gold_set),
            "precision": precision, "recall": recall, "f1": f1}


def candidate_recall(candidate_ids: Iterable[str], gold: Iterable[str]) -> float:
    candidates, answers = set(candidate_ids), set(gold)
    return len(candidates & answers) / len(answers) if answers else 1.0
