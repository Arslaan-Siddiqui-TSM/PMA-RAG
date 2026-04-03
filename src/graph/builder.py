from typing import Callable

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from psycopg_pool import AsyncConnectionPool

from src.graph.edges import (
    route_after_intent,
    route_after_quality_gate,
    route_after_reflection,
    route_after_reformulate,
)
from src.graph.nodes import (
    casual_response_node,
    classify_intent_node,
    generate_node,
    help_response_node,
    quality_gate_node,
    reformulate_query_node,
    rerank_node,
    retrieve_node,
)
from src.graph.planner import plan_retrieval_node
from src.graph.reflection import reflect_on_retrieval_node
from src.graph.state import RAGState

NodeMap = dict[str, Callable]


def _default_node_map() -> NodeMap:
    return {
        "classify_intent": classify_intent_node,
        "casual_response": casual_response_node,
        "help_response": help_response_node,
        "reformulate_query": reformulate_query_node,
        "plan_retrieval": plan_retrieval_node,
        "retrieve": retrieve_node,
        "rerank": rerank_node,
        "reflect_on_retrieval": reflect_on_retrieval_node,
        "generate": generate_node,
        "quality_gate": quality_gate_node,
    }


def build_graph(node_map: NodeMap | None = None) -> StateGraph:
    nodes = node_map or _default_node_map()
    graph = StateGraph(RAGState)

    for name, fn in nodes.items():
        graph.add_node(name, fn)

    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges("classify_intent", route_after_intent)
    graph.add_conditional_edges("reformulate_query", route_after_reformulate)
    graph.add_edge("plan_retrieval", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "reflect_on_retrieval")
    graph.add_conditional_edges("reflect_on_retrieval", route_after_reflection)
    graph.add_edge("generate", "quality_gate")
    graph.add_conditional_edges(
        "quality_gate",
        route_after_quality_gate,
        {
            "generate": "generate",
            "reformulate_query": "reformulate_query",
            "end": END,
        },
    )

    graph.add_edge("casual_response", END)
    graph.add_edge("help_response", END)

    return graph


async def compile_graph(
    pool: AsyncConnectionPool,
    node_map: NodeMap | None = None,
) -> CompiledStateGraph:
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()
    graph = build_graph(node_map)
    return graph.compile(checkpointer=checkpointer)
