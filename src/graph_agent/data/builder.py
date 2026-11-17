"""Two-hop QA group construction.

A *group* is ``(topic entity, relation path)`` together with its **complete**
answer set inside the visible graph. Grouping matters because the split unit is
the group, not the question: every paraphrase of one group lands in the same
split, so a model cannot memorise a surface form from training and see it again
in test.
"""
from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from dataclasses import dataclass

from graph_agent.schemas import Triple

from .surface import (
    ALLOWED_RELATIONS,
    TEMPLATES_PLURAL,
    TEMPLATES_SINGULAR,
    render_question,
)


@dataclass(frozen=True)
class QAGroup:
    """One (topic entity, relation path) with all of its answers."""

    topic_entity: str
    relation_path: tuple[str, str]
    answer_ids: tuple[str, ...]
    evidence: tuple[Triple, ...]

    @property
    def key(self) -> tuple[str, tuple[str, str]]:
        return (self.topic_entity, self.relation_path)

    @property
    def path_label(self) -> str:
        return " >> ".join(self.relation_path)


def _stable_index(key: str, modulo: int, salt: str) -> int:
    """Deterministic index that survives across processes (unlike ``hash``)."""
    digest = hashlib.blake2b(f"{salt}|{key}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % modulo


def enumerate_groups(
    adjacency: dict[str, tuple[tuple[str, str], ...]],
    *,
    max_answers: int = 20,
    max_groups: int = 60_000,
    seed: int = 42,
) -> list[QAGroup]:
    """Walk every head entity and collect valid two-hop groups.

    Head order is shuffled with ``seed`` before taking at most ``max_groups``.
    This is a deterministic bounded enumeration, not a uniform group sample.
    A group is rejected whole if any matching walk contains a self-loop or
    returns to the topic: removing only that walk would truncate its answers.
    """
    for name, value in (("max_answers", max_answers), ("max_groups", max_groups)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} 必须为正整数")
    heads = sorted(adjacency)
    random.Random(seed).shuffle(heads)

    groups: list[QAGroup] = []
    for head in heads:
        # (r1, r2) -> {answer: first intermediate that reaches it}
        bucket: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
        invalid_paths: set[tuple[str, str]] = set()
        for first, middle in adjacency[head]:
            if first not in ALLOWED_RELATIONS:
                continue
            for second, tail in adjacency.get(middle, ()):
                if second not in ALLOWED_RELATIONS or second == first:
                    continue  # self-composition reads as "the X of the X"
                path = (first, second)
                bucket[path].setdefault(tail, middle)
                if middle == head or tail == head or tail == middle:
                    invalid_paths.add(path)

        for path, answers in bucket.items():
            if path in invalid_paths or not 1 <= len(answers) <= max_answers:
                continue
            first, second = path
            # One shared first hop means the intermediate repeats for every
            # answer; de-duplicate so the evidence list stays readable.
            evidence: list[Triple] = []
            seen: set[Triple] = set()
            for tail, middle in answers.items():
                for triple in (
                    Triple(head, first, middle),
                    Triple(middle, second, tail),
                ):
                    if triple not in seen:
                        seen.add(triple)
                        evidence.append(triple)
            groups.append(
                QAGroup(
                    topic_entity=head,
                    relation_path=path,
                    answer_ids=tuple(sorted(answers)),
                    evidence=tuple(evidence),
                )
            )
            if len(groups) == max_groups:
                return groups
    return groups


@dataclass(frozen=True)
class PathStat:
    path: tuple[str, str]
    groups: int
    answers_mean: float
    dominant_share: float
    answer_set_share: float
    kept: bool
    reason: str


def filter_degenerate_paths(
    groups: list[QAGroup],
    *,
    min_support: int = 25,
    max_dominant_share: float = 0.6,
    max_answer_set_share: float = 0.4,
) -> tuple[list[QAGroup], list[PathStat]]:
    """Drop relation paths that are too rare or effectively constant.

    Two degeneracy signals:

    * **dominant answer** — ``producer >> profession`` always answers "film
      producer", so a model can score without touching the graph;
    * **repeated answer set** — ``nationality >> time zone`` answers "the five
      US time zones" for most people, so the answer is guessable from the path
      alone even though no single entity dominates.

    Both would inflate every score and hide whether retrieval or the GNN did
    anything, so both are filtered and reported.
    """
    by_path: dict[tuple[str, str], list[QAGroup]] = defaultdict(list)
    for group in groups:
        by_path[group.relation_path].append(group)

    stats: list[PathStat] = []
    kept: list[QAGroup] = []
    for path, items in sorted(by_path.items()):
        answers = Counter(answer for item in items for answer in item.answer_ids)
        total = sum(answers.values())
        dominant = answers.most_common(1)[0][1] / total if total else 0.0
        answer_sets = Counter(item.answer_ids for item in items)
        repeated = answer_sets.most_common(1)[0][1] / len(items) if items else 0.0

        if len(items) < min_support:
            reason = f"support<{min_support}"
        elif dominant > max_dominant_share:
            reason = f"dominant answer {dominant:.0%}>{max_dominant_share:.0%}"
        elif repeated > max_answer_set_share:
            reason = f"repeated answer set {repeated:.0%}>{max_answer_set_share:.0%}"
        else:
            reason = "kept"
        ok = reason == "kept"
        stats.append(
            PathStat(
                path=path,
                groups=len(items),
                answers_mean=round(
                    sum(len(i.answer_ids) for i in items) / len(items), 2
                ),
                dominant_share=round(dominant, 3),
                answer_set_share=round(repeated, 3),
                kept=ok,
                reason=reason,
            )
        )
        if ok:
            kept.extend(items)
    return kept, stats


def render_group_items(
    group: QAGroup,
    count: int,
    entity_text: dict[str, str],
) -> list[tuple[str, str]]:
    """Render ``count`` distinct paraphrases, returning ``(template, text)`` pairs."""
    templates = TEMPLATES_SINGULAR if len(group.answer_ids) == 1 else TEMPLATES_PLURAL
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= len(templates):
        raise ValueError(f"每组问法数必须是 1 到 {len(templates)} 之间的整数")
    label = f"{group.topic_entity}|{group.relation_path[0]}|{group.relation_path[1]}"
    start = _stable_index(label, len(templates), "template")
    picked = [templates[(start + i) % len(templates)] for i in range(count)]
    return [
        (tpl, render_question(group.topic_entity, group.relation_path, tpl, entity_text))
        for tpl in picked
    ]


def render_group_questions(
    group: QAGroup,
    count: int,
    entity_text: dict[str, str],
) -> list[str]:
    """Render ``count`` distinct paraphrases for one group, deterministically."""
    return [text for _, text in render_group_items(group, count, entity_text)]
