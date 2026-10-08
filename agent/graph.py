"""6-node LangGraph pipeline wiring.

intake -> classification -> extraction -> validation -> matching -> human_review -> END

`human_review` may call `interrupt()`, which pauses the graph mid-node;
the caller must supply a checkpointer (see agent/api.py) so the paused
state survives across process boundaries and can be resumed later via
`graph.ainvoke(Command(resume=...), config={"configurable": {"thread_id": ...}})`.
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from agent.nodes.classification import classification_node
from agent.nodes.extraction import extraction_node
from agent.nodes.human_review import human_review_node
from agent.nodes.intake import intake_node
from agent.nodes.matching import matching_node
from agent.nodes.validation import validation_node
from agent.state import InvoiceState
from agent.telemetry import traced_node


def build_graph(checkpointer=None):
    graph = StateGraph(InvoiceState)

    graph.add_node("intake", traced_node("intake")(intake_node))
    graph.add_node("classification", traced_node("classification")(classification_node))
    graph.add_node("extraction", traced_node("extraction")(extraction_node))
    graph.add_node("validation", traced_node("validation")(validation_node))
    graph.add_node("matching", traced_node("matching")(matching_node))
    graph.add_node("human_review", traced_node("human_review")(human_review_node))

    graph.set_entry_point("intake")
    graph.add_edge("intake", "classification")
    graph.add_edge("classification", "extraction")
    graph.add_edge("extraction", "validation")
    graph.add_edge("validation", "matching")
    graph.add_edge("matching", "human_review")
    graph.add_edge("human_review", END)

    return graph.compile(checkpointer=checkpointer)


async def pending_review(graph, config) -> dict | None:
    """The review payload if this run is paused at human_review, else None.

    Reads the checkpointed state instead of looking for "__interrupt__" in the ainvoke() result: that key only
    exists in newer LangGraph releases, so checking for it silently reports every paused run as finished on
    older ones (the cause of the harness mis-reporting that earlier versions of this README disclosed)."""
    state = await graph.aget_state(config)
    for task in state.tasks:
        if task.interrupts:
            return task.interrupts[0].value
    return None
