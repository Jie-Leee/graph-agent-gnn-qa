"""D owns this module. Tool routing must be an actual model decision."""
from typing import Literal
from graph_agent.schemas import Prediction, Question

Mode = Literal["llm", "text", "gnn"]


class AgentRunner:
    def run(self, question: Question, mode: Mode) -> Prediction:
        """Use the same LLM and three evidence modes in a bounded query loop.

        llm: no KG evidence. text: fixed text evidence ranking. gnn: trained
        GNN ranking plus structured evidence. The LLM chooses query/answer.
        Cache routing/retrieval across evidence variants for fair comparisons.
        """
        if mode not in ("llm", "text", "gnn"):
            raise ValueError(f"Unknown mode: {mode}")
        raise NotImplementedError("D: connect the LLM and tool-calling loop.")
