"""Fictional graph and explicitly scripted offline responses, never experiment data."""
from __future__ import annotations

import copy
from collections import deque

from graph_agent.agent.evidence import EntityCatalog, encode
from graph_agent.agent.provider import Completion
from graph_agent.schemas import Question, Triple


DEMO_TRIPLES = (
    Triple("film:sample", "directed_by", "person:director"),
    Triple("film:sample", "genre", "genre:drama"),
    Triple("person:director", "place_of_birth", "city:sample"),
    Triple("person:director", "profession", "job:director"),
)
DEMO_CATALOG = EntityCatalog(
    {"film:sample": "Sample Film", "person:director": "Sample Director",
     "genre:drama": "Drama", "city:sample": "Sample City", "job:director": "Director"},
    {"directed_by": "directed by", "genre": "genre", "place_of_birth": "place of birth",
     "profession": "profession"},
)
DEMO_QUESTION = Question("demo-query", "Where was the director of Sample Film born?", ("film:sample",))
SKIP_QUESTION = Question("demo-skip", "What is 2 + 2?", ())


def answer_message(*, ids=(), names=(), text="Unknown", evidence=()) -> dict:
    return {"role": "assistant", "content": encode({"answer_ids": list(ids),
        "answer_names": list(names), "answer_text": text,
        "cited_evidence": [[t.head, t.relation, t.tail] for t in evidence]})}


def query_message(call_id="query-1", hops=2) -> dict:
    return {"role": "assistant", "content": None, "tool_calls": [{"id": call_id,
        "type": "function", "function": {"name": "query_graph", "arguments": encode({"max_hops": hops})}}]}


class ScriptedProvider:
    """Test double: makes no model decisions, reports no fabricated token usage."""
    def __init__(self, messages):
        self.messages = deque(messages)
        self.requests = []

    def complete(self, messages, *, tools=None):
        self.requests.append({"messages": copy.deepcopy(messages), "tools": copy.deepcopy(tools)})
        if not self.messages:
            raise RuntimeError("Scripted provider has no response left")
        return Completion(copy.deepcopy(self.messages.popleft()))


def offline_provider() -> ScriptedProvider:
    return ScriptedProvider([
        query_message(), {"role": "assistant", "content": "Retrieval is ready."},
        answer_message(ids=("city:sample",), text="Sample City (fictional).",
                       evidence=(DEMO_TRIPLES[0], DEMO_TRIPLES[2])),
        {"role": "assistant", "content": "No graph is needed for arithmetic."},
        answer_message(text="4"),
    ])
