"""Grouped assignment of QA items to splits, plus the leakage audit.

The unit of assignment is the (topic entity, relation path) group, never the
individual question. Two consequences:

* all paraphrases of one group stay together, so no surface form is seen in
  training and tested again;
* a topic entity may legitimately appear in several splits, but never with the
  same relation path — otherwise a model could answer by memorising a fact
  instead of retrieving it.

Assignment is greedy over a seeded shuffle: a group is appended whole to the
current split, and a group is skipped if any of its paraphrases collides with a
question already used anywhere. That keeps item counts exact while guaranteeing
globally unique question text.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from graph_agent.schemas import Triple

from .builder import QAGroup, render_group_items

SPLIT_ORDER = ("train", "validation", "test")


class SplitError(RuntimeError):
    """Raised when the requested split sizes cannot be satisfied."""


def validate_targets(targets: Mapping[str, int], paraphrases: int) -> None:
    """Reject settings that cannot preserve whole groups and exact counts."""
    if type(paraphrases) is not int or not 1 <= paraphrases <= 4:
        raise SplitError("paraphrases 必须是 1 到 4 的整数")
    if set(targets) != set(SPLIT_ORDER):
        raise SplitError("targets 必须且只能包含 train、validation、test")
    for name in SPLIT_ORDER:
        target = targets[name]
        if type(target) is not int or target < 0:
            raise SplitError(f"{name} 的目标条数必须是非负整数")
        if target % paraphrases:
            raise SplitError(
                f"{name} 的目标条数 {target} 必须是每组问法数 {paraphrases} 的倍数"
            )


@dataclass
class Assignment:
    records: dict[str, list[dict]] = field(default_factory=dict)
    group_counts: dict[str, int] = field(default_factory=dict)
    template_counts: Counter = field(default_factory=Counter)
    skipped_groups: int = 0


def assign_records(
    groups: list[QAGroup],
    targets: dict[str, int],
    *,
    entity_text: dict[str, str],
    paraphrases: int = 2,
    seed: int = 42,
) -> Assignment:
    """Fill each split to exactly ``targets[name]`` items."""
    validate_targets(targets, paraphrases)

    pool = sorted(groups, key=lambda g: (g.topic_entity, g.relation_path))
    random.Random(seed).shuffle(pool)

    assignment = Assignment(
        records={name: [] for name in SPLIT_ORDER},
        group_counts={name: 0 for name in SPLIT_ORDER},
    )
    used_texts: set[str] = set()
    cursor = 0

    for name in SPLIT_ORDER:
        target = targets[name]
        bucket = assignment.records[name]
        while len(bucket) < target:
            if cursor >= len(pool):
                raise SplitError(
                    f"{name} 只填到 {len(bucket)}/{target} 条：候选组已耗尽。"
                    "请放宽白名单、提高 max_answers 或降低目标条数。"
                )
            group = pool[cursor]
            cursor += 1
            items = render_group_items(group, paraphrases, entity_text)
            texts = [text for _, text in items]
            # Skip the whole group rather than partially using it, so every
            # group keeps the same number of paraphrases.
            if len(set(texts)) < paraphrases or any(t in used_texts for t in texts):
                assignment.skipped_groups += 1
                continue
            used_texts.update(texts)
            for template, text in items:
                bucket.append(
                    {
                        "question_id": f"{name}-{len(bucket):05d}",
                        "text": text,
                        "topic_entity_ids": [group.topic_entity],
                        "answer_ids": list(group.answer_ids),
                        "relation_path": list(group.relation_path),
                        "evidence": [
                            [t.head, t.relation, t.tail] for t in group.evidence
                        ],
                    }
                )
                assignment.template_counts[template] += 1
            assignment.group_counts[name] += 1

    return assignment


def audit_records(
    records: dict[str, list[dict]],
    *,
    train_triples: Iterable[Triple] | None = None,
    adjacency: Mapping[str, Iterable[tuple[str, str]]] | None = None,
) -> dict[str, object]:
    """Audit split leakage, connected evidence and (when supplied) visible graph.

    Supplying the visible training graph also verifies that every evidence edge
    exists and that gold answers equal the *complete* two-hop answer set. Without
    it, only internal evidence consistency can be checked; callers should pass
    one of ``train_triples`` or ``adjacency`` before publishing a dataset.
    """
    if train_triples is not None and adjacency is not None:
        raise ValueError("请只提供 train_triples 或 adjacency 之一")
    graph_edges: set[tuple[str, str, str]] | None = None
    graph_index: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    if train_triples is not None:
        graph_edges = {(t.head, t.relation, t.tail) for t in train_triples}
    elif adjacency is not None:
        graph_edges = {
            (head, relation, tail)
            for head, edges in adjacency.items()
            for relation, tail in edges
        }
    if graph_edges is not None:
        for head, relation, tail in graph_edges:
            graph_index[head][relation].add(tail)

    group_splits: dict[tuple, set[str]] = defaultdict(set)
    question_splits: dict[str, set[str]] = defaultdict(set)
    duplicated: dict[str, int] = {}
    topics_by_split: dict[str, set[str]] = {}
    answers_by_split: dict[str, set[str]] = {}
    self_answers = 0
    empty_answers = 0
    evidence_without_answer = 0
    disconnected_evidence = 0
    evidence_outside_graph = 0
    incomplete_answer_sets = 0
    unreachable_answers = 0
    duplicate_question_ids = 0
    duplicate_answer_ids = 0
    question_ids: set[str] = set()

    for name in SPLIT_ORDER:
        items = records.get(name, [])
        topics: set[str] = set()
        answers: set[str] = set()
        seen: set[str] = set()
        duplicates = 0
        for item in items:
            if item["question_id"] in question_ids:
                duplicate_question_ids += 1
            question_ids.add(item["question_id"])
            group_key = (item["topic_entity_ids"][0], tuple(item["relation_path"]))
            group_splits[group_key].add(name)
            question_splits[item["text"]].add(name)
            if item["text"] in seen:
                duplicates += 1
            seen.add(item["text"])
            topics.update(item["topic_entity_ids"])
            answers.update(item["answer_ids"])
            if not item["answer_ids"]:
                empty_answers += 1
            if set(item["topic_entity_ids"]) & set(item["answer_ids"]):
                self_answers += 1
            answer_set = set(item["answer_ids"])
            if len(answer_set) != len(item["answer_ids"]):
                duplicate_answer_ids += 1
            evidence = {tuple(triple) for triple in item["evidence"]}
            topic = item["topic_entity_ids"][0]
            first, second = item["relation_path"]
            middles = {tail for head, rel, tail in evidence if head == topic and rel == first}
            final_edges = {
                (head, rel, tail)
                for head, rel, tail in evidence
                if head in middles and rel == second
            }
            connected_middles = {head for head, _, _ in final_edges}
            connected_edges = final_edges | {
                (topic, first, middle) for middle in connected_middles
            }
            supported_answers = {tail for _, _, tail in final_edges}
            if not answer_set <= supported_answers:
                evidence_without_answer += 1
            if evidence - connected_edges:
                disconnected_evidence += 1
            if graph_edges is not None:
                if evidence - graph_edges:
                    evidence_outside_graph += 1
                expected = {
                    tail
                    for middle in graph_index.get(topic, {}).get(first, set())
                    for tail in graph_index.get(middle, {}).get(second, set())
                }
                if expected - answer_set:
                    incomplete_answer_sets += 1
                if answer_set - expected:
                    unreachable_answers += 1
        duplicated[name] = duplicates
        topics_by_split[name] = topics
        answers_by_split[name] = answers

    cross_groups = [key for key, names in group_splits.items() if len(names) > 1]
    cross_questions = [text for text, names in question_splits.items() if len(names) > 1]
    train_topics = topics_by_split.get("train", set())
    test_topics = topics_by_split.get("test", set())

    violations: list[str] = []
    violations += [f"组跨划分：{k}" for k in cross_groups[:5]]
    violations += [f"问句跨划分：{q[:60]}" for q in cross_questions[:5]]
    violations += [
        f"{name} 有 {count} 条重复问句"
        for name, count in duplicated.items()
        if count
    ]
    if self_answers:
        violations.append(f"{self_answers} 条的答案包含主题实体")
    if empty_answers:
        violations.append(f"{empty_answers} 条没有答案")
    if evidence_without_answer:
        violations.append(f"{evidence_without_answer} 条的证据不覆盖全部答案")
    if disconnected_evidence:
        violations.append(f"{disconnected_evidence} 条包含不属于主题实体指定两跳路径的证据")
    if evidence_outside_graph:
        violations.append(f"{evidence_outside_graph} 条包含可见图中不存在的证据边")
    if incomplete_answer_sets:
        violations.append(f"{incomplete_answer_sets} 条遗漏可见图中的两跳答案")
    if unreachable_answers:
        violations.append(f"{unreachable_answers} 条包含可见图中不可达的答案")
    if duplicate_question_ids:
        violations.append(f"{duplicate_question_ids} 条 question_id 重复")
    if duplicate_answer_ids:
        violations.append(f"{duplicate_answer_ids} 条 answer_ids 包含重复 ID")

    return {
        "cross_split_groups": len(cross_groups),
        "cross_split_questions": len(cross_questions),
        "duplicated_questions_within_split": duplicated,
        "self_answer_items": self_answers,
        "empty_answer_items": empty_answers,
        "evidence_incomplete_items": evidence_without_answer,
        "disconnected_evidence_items": disconnected_evidence,
        "evidence_not_in_visible_graph_items": evidence_outside_graph,
        "incomplete_answer_set_items": incomplete_answer_sets,
        "unreachable_answer_items": unreachable_answers,
        "duplicate_question_ids": duplicate_question_ids,
        "duplicate_answer_id_items": duplicate_answer_ids,
        "visible_graph_checked": graph_edges is not None,
        "topic_entity_overlap_train_test": len(train_topics & test_topics),
        "test_topic_entities_unseen_in_train": len(test_topics - train_topics),
        "answer_entity_overlap_train_test": len(
            answers_by_split.get("train", set()) & answers_by_split.get("test", set())
        ),
        "violations": violations,
    }
