"""Query a local TSV knowledge graph and print evidence plus a Mermaid graph."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graph_agent.retrieval.graph import GraphIndex
from graph_agent.retrieval.visualization import format_paths, format_triples, to_mermaid
from graph_agent.schemas import Question


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triples", required=True, help="TSV file: head<TAB>relation<TAB>tail")
    parser.add_argument("--entity-names", help="Optional TSV file: entity ID<TAB>display name")
    parser.add_argument("--relation-names", help="Optional TSV file: relation ID<TAB>display name")
    parser.add_argument("--question", required=True, help="Question text used for entity/relation matching")
    parser.add_argument("--topic-entity", action="append", default=[],
                        help="Known topic entity ID; may be passed more than once")
    parser.add_argument("--max-hops", type=int, default=2)
    parser.add_argument("--max-nodes", type=int, default=200)
    parser.add_argument("--max-edges", type=int, default=800)
    parser.add_argument("--directed-only", action="store_true",
                        help="Do not traverse incoming edges in reverse")
    args = parser.parse_args()

    index = GraphIndex.from_tsv(args.triples, args.entity_names, args.relation_names)
    question = Question("cli-query", args.question, tuple(args.topic_entity))
    result = index.retrieve_with_stats(
        question,
        max_hops=args.max_hops,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        allow_inverse=not args.directed_only,
    )
    graph = result.subgraph

    print("Topics:", ", ".join(graph.topic_entity_ids) or "(no entity name matched)")
    print("Retrieval stats:")
    print(json.dumps(asdict(result.stats), ensure_ascii=False, indent=2))
    print("\nTriples:")
    print(format_triples(graph.triples, index.entity_names, index.relation_names) or "(none)")
    print("\nEvidence paths:")
    print(format_paths(graph, index.entity_names, index.relation_names) or "(none)")
    print("\nMermaid subgraph:")
    print(to_mermaid(graph, index.entity_names, index.relation_names))


if __name__ == "__main__":
    main()
