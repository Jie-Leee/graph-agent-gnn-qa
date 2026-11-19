"""Load A's JSONL files into the unchanged shared dataclasses.

Use ``load_questions`` at inference: it deliberately copies only the three
Question fields, even when the source also contains gold supervision.
"""
from __future__ import annotations

import json
from pathlib import Path

from graph_agent.schemas import QAExample, Question, Triple


class DatasetError(ValueError):
    """A processed record is malformed."""


def _rows(path: Path):
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError("expected a JSON object")
                yield number, item
            except (ValueError, TypeError) as error:
                raise DatasetError(f"{path}:{number}: {error}") from error


def _strings(value, field: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(part, str) and part.strip() for part in value
    ) or (not value and not allow_empty):
        raise DatasetError(f"{field} must be a list of nonempty strings")
    if len(value) != len(set(value)) and field != "relation_path":
        raise DatasetError(f"{field} contains duplicate IDs")
    return tuple(value)


def _question(item: dict) -> Question:
    for field in ("question_id", "text"):
        if not isinstance(item.get(field), str) or not item[field].strip():
            raise DatasetError(f"{field} must be a nonempty string")
    return Question(
        question_id=item["question_id"],
        text=item["text"],
        topic_entity_ids=_strings(item.get("topic_entity_ids"), "topic_entity_ids"),
    )


def load_questions(path: Path) -> tuple[Question, ...]:
    """Return inference inputs with no gold answers, paths or evidence."""
    questions = []
    seen = set()
    for number, item in _rows(path):
        try:
            question = _question(item)
            if question.question_id in seen:
                raise DatasetError(f"duplicate question_id {question.question_id}")
            seen.add(question.question_id)
            questions.append(question)
        except (ValueError, KeyError, TypeError) as error:
            raise DatasetError(f"{path}:{number}: {error}") from error
    return tuple(questions)


def load_examples(path: Path) -> tuple[QAExample, ...]:
    """Return supervision for training or evaluation, never for Agent prompts."""
    examples = []
    seen = set()
    for number, item in _rows(path):
        try:
            question = _question(item)
            if question.question_id in seen:
                raise DatasetError(f"duplicate question_id {question.question_id}")
            seen.add(question.question_id)
            answers = _strings(item.get("answer_ids"), "answer_ids")
            relation_path = _strings(item.get("relation_path"), "relation_path")
            evidence = item.get("evidence")
            if not isinstance(evidence, list) or not evidence:
                raise DatasetError("evidence must be a nonempty list of triples")
            triples = []
            for row in evidence:
                if not isinstance(row, list) or len(row) != 3 or not all(
                    isinstance(part, str) and part.strip() for part in row
                ):
                    raise DatasetError("each evidence triple needs three nonempty strings")
                triples.append(Triple(*row))
            examples.append(QAExample(question, answers, relation_path, tuple(triples)))
        except (ValueError, KeyError, TypeError) as error:
            raise DatasetError(f"{path}:{number}: {error}") from error
    return tuple(examples)
