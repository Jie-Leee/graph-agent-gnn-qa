"""Python standard-library demo; not an LLM/GNN experiment."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graph_agent.retrieval.graph import GraphIndex
from graph_agent.retrieval.visualization import format_paths, format_triples, to_mermaid
from graph_agent.schemas import Question, Triple


def main() -> None:
    triples = [
        Triple("film:sample", "directed_by", "person:director"),
        Triple("film:sample", "genre", "genre:drama"),
        Triple("person:director", "place_of_birth", "city:sample"),
        Triple("person:director", "profession", "job:director"),
    ]
    entity_names = {
        "film:sample": "sample film",
        "person:director": "sample director",
        "genre:drama": "drama",
        "city:sample": "sample city",
        "job:director": "film director",
    }
    relation_names = {
        "directed_by": "directed by",
        "place_of_birth": "place of birth born in",
        "genre": "film genre",
        "profession": "profession",
    }
    question = Question(
        "demo-001", "Where was the director of the sample film born?", ())
    index = GraphIndex(triples, entity_names=entity_names, relation_names=relation_names)
    result = index.retrieve_with_stats(question, max_hops=2)
    graph = result.subgraph
    print("OFFLINE RETRIEVAL DEMO - fictional sample graph; no LLM or GNN")
    print(f"Question: {question.text}")
    print(f"Linked topic entities: {', '.join(graph.topic_entity_ids) or '(none)'}")
    print(f"Retrieved nodes={len(graph.node_ids)}, edges={len(graph.triples)}")
    print(f"Retrieval time: {result.stats.elapsed_seconds * 1000:.3f} ms")
    print("Retrieved triples:")
    print(format_triples(graph.triples, entity_names, relation_names))
    print("Shortest evidence paths (not a predicted answer):")
    print(format_paths(graph, entity_names, relation_names))
    print("Mermaid subgraph:")
    print(to_mermaid(graph, entity_names, relation_names))
    print("Next: real FB15k-237 data, LLM Agent, and trained GAT.")


if __name__ == "__main__":
    main()
