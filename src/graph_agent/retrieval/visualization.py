"""Text and Mermaid renderers for retrieved evidence subgraphs."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import html

from graph_agent.retrieval.entity_linking import aliases_by_entity
from graph_agent.schemas import Subgraph, Triple


def _labels(values: Mapping[str, str | Iterable[str]] | None) -> dict[str, str]:
    return {key: aliases[0] for key, aliases in aliases_by_entity(values).items() if aliases}


def _display(value: str, names: Mapping[str, str]) -> str:
    return names.get(value, value)


def format_paths(
    subgraph: Subgraph,
    entity_names: Mapping[str, str | Iterable[str]] | None = None,
    relation_names: Mapping[str, str | Iterable[str]] | None = None,
) -> str:
    """Render shortest retrieved paths, preserving forward/reverse traversal."""
    entities = _labels(entity_names)
    relations = _labels(relation_names)
    lines: list[str] = []
    for target, steps in subgraph.path_steps.items():
        if not steps:
            continue
        rendered = []
        for step in steps:
            relation = _display(step.triple.relation, relations)
            direction = ", reverse" if step.is_reverse else ""
            rendered.append(
                f"{_display(step.source_entity_id, entities)} "
                f"--[{relation}{direction}]--> {_display(step.target_entity_id, entities)}"
            )
        lines.append(f"{_display(target, entities)}: " + " ; ".join(rendered))
    return "\n".join(lines)


def to_mermaid(
    subgraph: Subgraph,
    entity_names: Mapping[str, str | Iterable[str]] | None = None,
    relation_names: Mapping[str, str | Iterable[str]] | None = None,
) -> str:
    """Return a Mermaid flowchart with topic nodes highlighted.

    Edges retain the source KG direction. Dashed edges indicate that the query
    path traversed that original triple in reverse.
    """
    entities = _labels(entity_names)
    relations = _labels(relation_names)
    node_keys = {entity_id: f"n{index}" for index, entity_id in enumerate(subgraph.node_ids)}
    topics = set(subgraph.topic_entity_ids)
    reverse_edges = {
        step.triple
        for path in subgraph.path_steps.values()
        for step in path
        if step.is_reverse
    }

    def quote(value: str) -> str:
        return html.escape(value, quote=True).replace("[", "&#91;").replace("]", "&#93;")

    lines = ["flowchart LR"]
    for entity_id, key in node_keys.items():
        label = quote(_display(entity_id, entities))
        lines.append(f'  {key}["{label}"]')
        if entity_id in topics:
            lines.append(f"  class {key} topic")
    for triple in subgraph.triples:
        head = node_keys.get(triple.head)
        tail = node_keys.get(triple.tail)
        if head is None or tail is None:
            continue
        relation = quote(_display(triple.relation, relations)).replace("|", "&#124;")
        edge = "-.->" if triple in reverse_edges else "-->"
        lines.append(f"  {head} {edge}|{relation}| {tail}")
    lines.extend(
        (
            "  classDef topic fill:#ffe3a3,stroke:#9a6700,stroke-width:2px",
            "  %% Solid arrows are original KG direction; dashed means reverse query traversal.",
        )
    )
    return "\n".join(lines)


def format_triples(
    triples: Iterable[Triple],
    entity_names: Mapping[str, str | Iterable[str]] | None = None,
    relation_names: Mapping[str, str | Iterable[str]] | None = None,
) -> str:
    """Render raw evidence triples with optional display names."""
    entities = _labels(entity_names)
    relations = _labels(relation_names)
    return "\n".join(
        f"{_display(triple.head, entities)} --{_display(triple.relation, relations)}--> "
        f"{_display(triple.tail, entities)}"
        for triple in triples
    )
