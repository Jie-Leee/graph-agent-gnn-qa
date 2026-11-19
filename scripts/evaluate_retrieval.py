"""Evaluate labeled QA JSONL; missing evidence is excluded from evidence recall."""
import argparse
import json
from collections.abc import Iterator
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graph_agent.retrieval.benchmark import RetrievalCase, benchmark_retrieval
from graph_agent.retrieval.graph import GraphIndex
from graph_agent.schemas import Question, Triple


def _string_tuple(
    value: Any, field: str, line_number: int, *, allow_empty: bool = True,
) -> tuple[str, ...]:
    if (not isinstance(value, list)
            or any(not isinstance(item, str) or not item.strip() for item in value)
            or (not value and not allow_empty)):
        requirement = "a list of nonempty strings" if allow_empty else "a nonempty list of nonempty strings"
        raise ValueError(f"QA line {line_number}: {field} must be {requirement}.")
    return tuple(value)


def _read_cases(path: str | Path) -> Iterator[RetrievalCase]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error.msg}") from error
            if not isinstance(row, dict):
                raise ValueError(f"QA line {line_number}: each row must be a JSON object.")
            question_id = row.get("question_id", f"line-{line_number}")
            text = row.get("text")
            if (not isinstance(question_id, str) or not question_id.strip()
                    or not isinstance(text, str) or not text.strip()):
                raise ValueError(f"QA line {line_number}: question_id and text must be nonempty strings.")
            topics = _string_tuple(row.get("topic_entity_ids", []), "topic_entity_ids", line_number)
            answers = _string_tuple(row.get("answer_ids"), "answer_ids", line_number, allow_empty=False)
            raw_evidence = row.get("evidence")
            if raw_evidence is None:
                raw_evidence = []
            if not isinstance(raw_evidence, list):
                raise ValueError(f"QA line {line_number}: evidence must be a list of triples.")
            evidence: list[Triple] = []
            for triple in raw_evidence:
                if (not isinstance(triple, list) or len(triple) != 3
                        or any(not isinstance(part, str) or not part.strip() for part in triple)):
                    raise ValueError(
                        f"QA line {line_number}: each evidence item must be [head, relation, tail] "
                        "with nonempty strings."
                    )
                evidence.append(Triple(*triple))
            yield RetrievalCase(
                question=Question(question_id, text, topics),
                gold_answer_ids=answers,
                gold_evidence=tuple(evidence),
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triples", required=True, help="TSV file: head<TAB>relation<TAB>tail")
    parser.add_argument("--qa", required=True,
                        help="Labeled QA JSONL with nonempty answer_ids; not a .questions.jsonl file")
    parser.add_argument("--entity-names", help="Optional TSV file: entity ID<TAB>display name")
    parser.add_argument("--relation-names", help="Optional TSV file: relation ID<TAB>display name")
    parser.add_argument("--max-hops", type=int, default=2)
    parser.add_argument("--max-nodes", type=int, default=200)
    parser.add_argument("--max-edges", type=int, default=800)
    parser.add_argument("--directed-only", action="store_true",
                        help="Do not traverse incoming edges in reverse")
    parser.add_argument("--output", help="Optional path for the JSON metrics report")
    args = parser.parse_args()

    try:
        # Validate the entire labeled input before querying or creating a report.
        cases = tuple(_read_cases(args.qa))
        if not cases:
            raise ValueError("Cannot evaluate an empty QA dataset.")
        index = GraphIndex.from_tsv(args.triples, args.entity_names, args.relation_names)
        metrics = benchmark_retrieval(
            index,
            cases,
            max_hops=args.max_hops,
            max_nodes=args.max_nodes,
            max_edges=args.max_edges,
            allow_inverse=not args.directed_only,
        )
        report = json.dumps(metrics, ensure_ascii=False, indent=2)
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(report + "\n", encoding="utf-8")
    except (ValueError, OSError) as error:
        parser.exit(1, f"Retrieval evaluation failed: {error}\n")
    print(report)


if __name__ == "__main__":
    main()
