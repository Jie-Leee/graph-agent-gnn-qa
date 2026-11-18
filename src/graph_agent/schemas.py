"""Shared contracts. Coordinate changes with the group leader.

Inference receives Question only. QAExample keeps supervision separate so that
gold answers and paths do not enter retrieval or Agent prompts.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Triple:
    head: str
    relation: str
    tail: str


@dataclass(frozen=True)
class Question:
    question_id: str
    text: str
    topic_entity_ids: tuple[str, ...]


@dataclass(frozen=True)
class QAExample:
    question: Question
    answer_ids: tuple[str, ...]
    relation_path: tuple[str, ...] = ()
    evidence: tuple[Triple, ...] = ()  # Supervision/evaluation only.


@dataclass(frozen=True)
class PathStep:
    """An original KG triple plus the direction used to traverse it."""
    triple: Triple
    source_entity_id: str
    target_entity_id: str
    is_reverse: bool = False


@dataclass(frozen=True)
class Subgraph:
    topic_entity_ids: tuple[str, ...]
    node_ids: tuple[str, ...]
    triples: tuple[Triple, ...]
    paths: dict[str, tuple[Triple, ...]] = field(default_factory=dict)
    path_steps: dict[str, tuple[PathStep, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class Candidate:
    entity_id: str
    score: float
    evidence: tuple[Triple, ...]


@dataclass(frozen=True)
class Prediction:
    question_id: str
    answer_ids: tuple[str, ...]
    answer_text: str
    cited_evidence: tuple[Triple, ...] = ()
    tool_calls: int = 0
    elapsed_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
