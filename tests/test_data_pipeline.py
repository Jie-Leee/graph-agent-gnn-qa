"""Tests for A's data pipeline.

Runs on a synthetic graph, so no FB15k-237 download is required:

    python tests/test_data_pipeline.py        # plain asserts
    python -m pytest tests/ -q                # if pytest is installed
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graph_agent.data.builder import (  # noqa: E402
    enumerate_groups,
    filter_degenerate_paths,
    render_group_items,
)
from graph_agent.data.loader import read_text_map, read_triples  # noqa: E402
from graph_agent.data.split import assign_records, audit_records  # noqa: E402
from graph_agent.data.surface import relation_surface  # noqa: E402
from graph_agent.schemas import Triple  # noqa: E402

BIRTH = "/people/person/place_of_birth"
ZONE = "/location/location/time_zones"
NATION = "/people/person/nationality"
FILM = "/film/film/produced_by"


def make_graph(heads: int = 40) -> dict[str, tuple[tuple[str, str], ...]]:
    """A small but realistic-shaped graph.

    Each head has a birthplace that sits in two time zones (so the
    ``place_of_birth >> time_zones`` group has a multi-answer set) and a
    country in one (a single-answer group). Both shapes are needed to exercise
    the singular/plural templates.
    """
    adjacency: dict[str, list[tuple[str, str]]] = {}
    for index in range(heads):
        head = f"/m/h{index:03d}"
        city = f"/m/city{index:03d}"
        country = f"/m/country{index:03d}"
        adjacency[head] = [(BIRTH, city), (NATION, country)]
        adjacency[city] = [(ZONE, f"/m/zone{index % 5}"), (ZONE, f"/m/zone{(index + 1) % 5}")]
        adjacency[country] = [(ZONE, f"/m/zone{index % 7}")]
    return {k: tuple(v) for k, v in adjacency.items()}


ENTITY_TEXT = {}


def test_answers_are_complete_and_evidence_covers_them() -> None:
    adjacency = make_graph(1)
    groups = enumerate_groups(adjacency, max_groups=100)
    by_path = {g.relation_path: g for g in groups}
    assert set(by_path) == {(BIRTH, ZONE), (NATION, ZONE)}
    for group in groups:
        reachable = {triple.tail for triple in group.evidence if triple.relation == ZONE}
        assert set(group.answer_ids) <= reachable, "证据必须覆盖全部答案"
        assert group.topic_entity not in group.answer_ids, "答案不能是主题实体"


def test_evidence_is_deduplicated() -> None:
    adjacency = make_graph(1)
    group = next(g for g in enumerate_groups(adjacency, max_groups=100))
    assert len(group.evidence) == len(set(group.evidence)), "证据不应重复"


def test_self_composition_is_rejected() -> None:
    adjacency = {
        "/m/a": [(ZONE, "/m/b")],
        "/m/b": [(ZONE, "/m/c")],
    }
    assert enumerate_groups(adjacency, max_groups=100) == [], "同关系两跳应被排除"


def test_answer_cap_filters_hub_groups() -> None:
    adjacency = {"/m/a": [(BIRTH, "/m/b")], "/m/b": [(ZONE, f"/m/z{i}") for i in range(9)]}
    assert enumerate_groups(adjacency, max_answers=5, max_groups=100) == []
    assert len(enumerate_groups(adjacency, max_answers=20, max_groups=100)) == 1


def test_template_does_not_reveal_answer_count() -> None:
    adjacency = make_graph(10)
    groups = enumerate_groups(adjacency, max_groups=100)
    single = next(g for g in groups if len(g.answer_ids) == 1)
    plural = next(g for g in groups if len(g.answer_ids) > 1)
    single_texts = [t for _, t in render_group_items(single, 2, ENTITY_TEXT)]
    plural_texts = [t for _, t in render_group_items(plural, 2, ENTITY_TEXT)]
    assert all("every" in t and "any" in t for t in single_texts + plural_texts)
    assert len(set(single_texts)) == 2, "两条问法必须不同"


def test_split_hits_exact_counts_and_keeps_groups_together() -> None:
    groups = filter_degenerate_paths(
        enumerate_groups(make_graph(60), max_groups=500),
        min_support=1,
    )[0]
    assignment = assign_records(
        groups,
        {"train": 40, "validation": 10, "test": 10},
        entity_text=ENTITY_TEXT,
        paraphrases=2,
        seed=7,
    )
    assert {k: len(v) for k, v in assignment.records.items()} == {
        "train": 40,
        "validation": 10,
        "test": 10,
    }
    seen: dict[tuple, str] = {}
    for name, items in assignment.records.items():
        per_group: dict[tuple, int] = {}
        for item in items:
            key = (item["topic_entity_ids"][0], tuple(item["relation_path"]))
            per_group[key] = per_group.get(key, 0) + 1
            assert seen.setdefault(key, name) == name, "同一组不得跨划分"
        assert set(per_group.values()) == {2}, "每组应恰好两条问法"


def test_assignment_is_deterministic() -> None:
    groups = filter_degenerate_paths(
        enumerate_groups(make_graph(30), max_groups=500), min_support=1
    )[0]
    targets = {"train": 20, "validation": 6, "test": 6}
    first = assign_records(groups, targets, entity_text=ENTITY_TEXT, seed=42)
    second = assign_records(groups, targets, entity_text=ENTITY_TEXT, seed=42)
    assert first.records == second.records


def test_audit_flags_cross_split_question() -> None:
    clean = {
        "train": [{"question_id": "t0", "text": "Q1", "topic_entity_ids": ["/m/a"],
                   "answer_ids": ["/m/b"], "relation_path": ["r1", "r2"],
                   "evidence": [["/m/a", "r1", "/m/x"], ["/m/x", "r2", "/m/b"]]}],
        "validation": [],
        "test": [{"question_id": "s0", "text": "Q2", "topic_entity_ids": ["/m/c"],
                  "answer_ids": ["/m/d"], "relation_path": ["r1", "r2"],
                  "evidence": [["/m/c", "r1", "/m/y"], ["/m/y", "r2", "/m/d"]]}],
    }
    assert audit_records(clean)["violations"] == []

    leaked = {k: list(v) for k, v in clean.items()}
    leaked["test"] = leaked["test"] + [
        {**clean["train"][0], "question_id": "s1"}
    ]
    assert audit_records(leaked)["violations"], "重复问句必须被审计发现"


def test_loader_rejects_malformed_row(tmp_path: Path) -> None:
    path = tmp_path / "train.txt"
    path.write_text("/m/a\t/r\t/m/b\n/m/a\t/m/b\n", encoding="utf-8")
    try:
        read_triples(path)
    except Exception as error:  # noqa: BLE001
        assert "不是合法" in str(error)
    else:  # pragma: no cover
        raise AssertionError("坏行应当报错")


def test_loader_reads_text_map(tmp_path: Path) -> None:
    path = tmp_path / "entity2text.txt"
    path.write_text("/m/a\tAlpha\n/m/b\tBeta Gamma\n", encoding="utf-8")
    assert read_text_map(path) == {"/m/a": "Alpha", "/m/b": "Beta Gamma"}


def test_relation_surface_falls_back_to_last_segment() -> None:
    assert relation_surface("/people/person/place_of_birth") == "place of birth"
    assert relation_surface("/unknown/thing/my_custom_rel") == "my custom rel"


def test_triple_is_hashable() -> None:
    assert len({Triple("/m/a", "r", "/m/b"), Triple("/m/a", "r", "/m/b")}) == 1


def main() -> None:
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failures = 0
    for name, func in tests:
        try:
            if func.__code__.co_argcount:
                import tempfile

                with tempfile.TemporaryDirectory() as tmp:
                    func(Path(tmp))
            else:
                func()
        except Exception as error:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(error).__name__}: {error}")
        else:
            print(f"ok   {name}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
