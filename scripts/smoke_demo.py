"""Python standard-library demo; not an LLM/GNN experiment."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graph_agent.retrieval.graph import GraphIndex
from graph_agent.schemas import Question, Triple


def main() -> None:
    triples = [
        Triple("film:sample", "directed_by", "person:director"),
        Triple("film:sample", "genre", "genre:drama"),
        Triple("person:director", "place_of_birth", "city:sample"),
        Triple("person:director", "profession", "job:director"),
    ]
    question = Question("demo-001",
        "Where was the director of the sample film born?", ("film:sample",))
    graph = GraphIndex(triples).retrieve(question, max_hops=2)
    print("OFFLINE RETRIEVAL DEMO - fictional sample graph; no LLM or GNN")
    print(f"Question: {question.text}")
    print(f"Retrieved nodes={len(graph.node_ids)}, edges={len(graph.triples)}")
    for triple in graph.triples:
        print(f"  {triple.head} --{triple.relation}--> {triple.tail}")
    print("Retrieved paths (not a predicted answer):")
    for entity_id, path in graph.paths.items():
        if path:
            relations = " / ".join(triple.relation for triple in path)
            print(f"  {entity_id}: {relations}")
    print("Next: real FB15k-237 data, entity linking, LLM Agent, and trained GAT.")


if __name__ == "__main__":
    main()
