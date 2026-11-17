"""Raw FB15k-237 loading and integrity checks.

Visible-graph policy
--------------------
QA is built on the **training split only**. Validation and test edges are never
used to derive questions or answers. That guarantees every gold answer is
reachable by a forward traversal of the visible graph at inference time, so the
benchmark measures "question answering over a known graph" and not the original
link-prediction protocol. See ``docs/数据说明.md`` for the full argument.
"""
from __future__ import annotations

import io
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from graph_agent.schemas import Triple

GRAPH_DIR = "FB15K-237"
SPLIT_FILES = {"train": "train.txt", "valid": "valid.txt", "test": "test.txt"}
ENTITY_TEXT_FILE = "entity2text.txt"
RELATION_TEXT_FILE = "relation2text.txt"


class RawDataError(RuntimeError):
    """Raised when the raw data directory is missing or malformed."""


@dataclass(frozen=True)
class RawGraph:
    """Everything the builder needs, already parsed and de-duplicated."""

    triples: dict[str, tuple[Triple, ...]]
    entity_text: dict[str, str]
    relation_text: dict[str, str]

    @property
    def train_triples(self) -> tuple[Triple, ...]:
        return self.triples["train"]

    def adjacency(self) -> dict[str, tuple[tuple[str, str], ...]]:
        """head entity -> tuple of (relation, tail entity), de-duplicated."""
        index: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for triple in self.train_triples:
            index[triple.head].append((triple.relation, triple.tail))
        return {head: tuple(dict.fromkeys(edges)) for head, edges in index.items()}

    def summary(self) -> dict[str, object]:
        entities: set[str] = set()
        relations: set[str] = set()
        for triples in self.triples.values():
            for triple in triples:
                entities.update((triple.head, triple.tail))
                relations.add(triple.relation)
        train_entities = {t.head for t in self.train_triples} | {
            t.tail for t in self.train_triples
        }
        train_relations = {t.relation for t in self.train_triples}
        out_degree = Counter(t.head for t in self.train_triples)
        in_degree = Counter(t.tail for t in self.train_triples)
        return {
            "triples": {name: len(rows) for name, rows in self.triples.items()},
            "entities_all_splits": len(entities),
            "relations_all_splits": len(relations),
            "entities_train": len(train_entities),
            "relations_train": len(train_relations),
            "entity_text_entries": len(self.entity_text),
            "relation_text_entries": len(self.relation_text),
            "entity_text_coverage_train": _coverage(train_entities, self.entity_text),
            "relation_text_coverage_train": _coverage(train_relations, self.relation_text),
            "out_degree_mean": round(
                sum(out_degree.values()) / max(len(out_degree), 1), 2
            ),
            "out_degree_max": max(out_degree.values(), default=0),
            "in_degree_max": max(in_degree.values(), default=0),
            "self_loops_train": sum(1 for t in self.train_triples if t.head == t.tail),
        }


def _coverage(subset: set[str], mapping: dict[str, str]) -> float:
    if not subset:
        return 0.0
    return round(len(subset & mapping.keys()) / len(subset), 4)


def _resolve_graph_dir(raw_dir: Path) -> Path:
    """Accept both ``raw/FB15K-237/`` and a flat ``raw/`` layout."""
    nested = raw_dir / GRAPH_DIR
    if (nested / SPLIT_FILES["train"]).exists():
        return nested
    if (raw_dir / SPLIT_FILES["train"]).exists():
        return raw_dir
    raise RawDataError(
        f"找不到 {SPLIT_FILES['train']}。请把 FB15k-237 解压到 "
        f"{raw_dir / GRAPH_DIR}（或直接放在 {raw_dir}），"
        "并确保 entity2text.txt / relation2text.txt 在同一目录下。"
    )


def read_triples(path: Path) -> tuple[Triple, ...]:
    if not path.exists():
        raise RawDataError(f"缺少三元组文件：{path}")
    triples: list[Triple] = []
    with io.open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            row = line.rstrip("\r\n")
            if not row:
                continue
            parts = row.split("\t")
            if len(parts) != 3 or not all(parts):
                raise RawDataError(
                    f"{path.name}:{line_no} 不是合法的 head<TAB>relation<TAB>tail：{row!r}"
                )
            triples.append(Triple(parts[0], parts[1], parts[2]))
    # dict.fromkeys keeps first-seen order, so the output is deterministic.
    return tuple(dict.fromkeys(triples))


def read_text_map(path: Path) -> dict[str, str]:
    if not path.exists():
        raise RawDataError(f"缺少文本映射文件：{path}")
    mapping: dict[str, str] = {}
    with io.open(path, encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            row = line.rstrip("\r\n")
            if not row:
                continue
            parts = row.split("\t")
            if len(parts) != 2 or not all(part.strip() for part in parts):
                raise RawDataError(
                    f"{path.name}:{line_no} 不是合法的 ID<TAB>非空文本：{row!r}"
                )
            key, value = (part.strip() for part in parts)
            if key in mapping and mapping[key] != value:
                raise RawDataError(
                    f"{path.name}:{line_no} 的 ID {key!r} 存在冲突文本映射"
                )
            mapping[key] = value
    if not mapping:
        raise RawDataError(f"文本映射文件为空：{path}")
    return mapping


def _find_text_file(graph_dir: Path, raw_dir: Path, filename: str) -> Path:
    """Text maps may sit next to the triples or one level up in ``raw/``."""
    for candidate in (graph_dir / filename, raw_dir / filename):
        if candidate.exists():
            return candidate
    raise RawDataError(
        f"缺少文本映射文件 {filename}：已查找 {graph_dir} 与 {raw_dir}。"
        "可从 https://github.com/yao8839836/kg-bert/tree/master/data/FB15k-237 获取。"
    )


def resolve_raw_files(raw_dir: Path) -> dict[str, Path]:
    """Locate the five input files in deterministic key order for provenance."""
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise RawDataError(f"原始数据目录不存在：{raw_dir}")
    graph_dir = _resolve_graph_dir(raw_dir)
    files = {name: graph_dir / filename for name, filename in SPLIT_FILES.items()}
    files["entity_text"] = _find_text_file(graph_dir, raw_dir, ENTITY_TEXT_FILE)
    files["relation_text"] = _find_text_file(graph_dir, raw_dir, RELATION_TEXT_FILE)
    for path in files.values():
        if not path.is_file():
            raise RawDataError(f"缺少原始数据文件：{path}")
    return files


def load_raw(raw_dir: Path) -> RawGraph:
    """Parse raw data; require labels for every visible entity and relation."""
    files = resolve_raw_files(raw_dir)
    triples = {name: read_triples(files[name]) for name in SPLIT_FILES}
    graph = RawGraph(
        triples=triples,
        entity_text=read_text_map(files["entity_text"]),
        relation_text=read_text_map(files["relation_text"]),
    )
    train_entities = {t.head for t in graph.train_triples} | {
        t.tail for t in graph.train_triples
    }
    train_relations = {t.relation for t in graph.train_triples}
    for label, ids, mapping in (
        ("实体", train_entities, graph.entity_text),
        ("关系", train_relations, graph.relation_text),
    ):
        missing = sorted(ids - mapping.keys())
        if missing:
            raise RawDataError(
                f"训练图有 {len(missing)} 个{label}缺少文本映射，例如：{missing[:5]}"
            )
    return graph


def count_duplicates(raw_dir: Path) -> dict[str, int]:
    """Count duplicate rows per split (reported, not silently dropped)."""
    graph_dir = _resolve_graph_dir(Path(raw_dir))
    counts: dict[str, int] = {}
    for name, filename in SPLIT_FILES.items():
        seen: set[tuple[str, ...]] = set()
        duplicates = 0
        with io.open(graph_dir / filename, encoding="utf-8") as handle:
            for line in handle:
                row = line.rstrip("\r\n")
                if not row:
                    continue
                key = tuple(row.split("\t"))
                if key in seen:
                    duplicates += 1
                else:
                    seen.add(key)
        counts[name] = duplicates
    return counts
