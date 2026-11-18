"""Small, deterministic entity linker for names shipped with a graph."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import re


_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str) -> tuple[str, ...]:
    """Return case-folded word tokens while ignoring punctuation and IDs' slashes."""
    return tuple(match.group(0).casefold() for match in _WORD.finditer(text))


@dataclass(frozen=True)
class EntityMatch:
    entity_id: str
    alias: str
    start_token: int
    end_token: int
    score: int


class EntityLinker:
    """Find exact, contiguous name mentions; never consult QA answers or paths.

    When aliases collide, every entity with that alias is returned. The caller
    can inspect matches and apply a task-specific disambiguation policy.
    """

    _END = object()

    def __init__(self, entity_names: Mapping[str, str | Iterable[str]] | None = None) -> None:
        self._trie: dict[object, object] = {}
        self._alias_count = 0
        for entity_id, values in (entity_names or {}).items():
            aliases = (values,) if isinstance(values, str) else values
            for alias in aliases:
                if not isinstance(alias, str) or not alias.strip():
                    continue
                tokens = tokenize(alias)
                if not tokens:
                    continue
                node = self._trie
                for token in tokens:
                    node = node.setdefault(token, {})  # type: ignore[assignment]
                terminal = node.setdefault(self._END, [])
                terminal.append((entity_id, alias, len(tokens) * 100 + len(alias)))  # type: ignore[union-attr]
                self._alias_count += 1

    def match(self, text: str, max_entities: int = 5) -> tuple[EntityMatch, ...]:
        """Return longest non-overlapping name matches in question order."""
        if max_entities < 1:
            raise ValueError("max_entities must be positive.")
        tokens = tokenize(text)
        matches: list[EntityMatch] = []
        position = 0
        while position < len(tokens):
            node = self._trie
            best: tuple[int, list[tuple[str, str, int]]] | None = None
            cursor = position
            while cursor < len(tokens) and tokens[cursor] in node:
                node = node[tokens[cursor]]  # type: ignore[assignment]
                cursor += 1
                terminal = node.get(self._END)  # type: ignore[union-attr]
                if terminal:
                    best = (cursor, terminal)  # type: ignore[assignment]
            if best is None:
                position += 1
                continue
            end, entries = best
            for entity_id, alias, score in entries:
                matches.append(EntityMatch(entity_id, alias, position, end, score))
            position = end

        # Keep the strongest mention for an entity if multiple aliases refer to it.
        strongest: dict[str, EntityMatch] = {}
        for match in matches:
            prior = strongest.get(match.entity_id)
            if prior is None or (match.score, -match.start_token) > (prior.score, -prior.start_token):
                strongest[match.entity_id] = match
        return tuple(sorted(strongest.values(), key=lambda m: (m.start_token, -m.score, m.entity_id))[:max_entities])

    def link(self, text: str, max_entities: int = 5) -> tuple[str, ...]:
        """Return entity IDs for the exact name mentions in ``text``."""
        return tuple(match.entity_id for match in self.match(text, max_entities=max_entities))

    @property
    def alias_count(self) -> int:
        return self._alias_count


def normalize_aliases(values: str | Iterable[str] | None) -> tuple[str, ...]:
    """Normalize metadata values into a de-duplicated tuple of non-empty aliases."""
    if values is None:
        return ()
    candidates = (values,) if isinstance(values, str) else values
    result: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        candidate = candidate.strip()
        key = " ".join(tokenize(candidate))
        if candidate and key and key not in seen:
            result.append(candidate)
            seen.add(key)
    return tuple(result)


def aliases_by_entity(values: Mapping[str, str | Iterable[str]] | None) -> dict[str, tuple[str, ...]]:
    """Convert a string-or-iterable mapping to normalized alias tuples."""
    return {entity_id: normalize_aliases(aliases) for entity_id, aliases in (values or {}).items()}
