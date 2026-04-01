from typing import Callable

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from psycopg_pool import AsyncConnectionPool

from src.graph.edges import (
    route_after_intent,
    route_after_reformulate,
    route_after_relevance_check,
    route_after_validation,
)
from src.graph.nodes import (
    casual_response_node,
    check_relevance_node,
    classify_intent_node,
    decompose_query_node,
    generate_node,
    help_response_node,
    reformulate_query_node,
    rerank_node,
    retrieve_node,
    validate_answer_node,
)
from src.graph.state import RAGState

NodeMap = dict[str, Callable]


def _default_node_map() -> NodeMap:
    return {
        "classify_intent": classify_intent_node,
        "casual_response": casual_response_node,
        "help_response": help_response_node,
        "reformulate_query": reformulate_query_node,
        "decompose_query": decompose_query_node,
        "retrieve": retrieve_node,
        "rerank": rerank_node,
        "check_relevance": check_relevance_node,
        "generate": generate_node,
        "validate_answer": validate_answer_node,
    }


def build_graph(node_map: NodeMap | None = None) -> StateGraph:
    nodes = node_map or _default_node_map()
    graph = StateGraph(RAGState)

    for name, fn in nodes.items():
        graph.add_node(name, fn)

    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges("classify_intent", route_after_intent)
    graph.add_conditional_edges("reformulate_query", route_after_reformulate)
    graph.add_edge("decompose_query", "retrieve")

    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "check_relevance")
    graph.add_conditional_edges("check_relevance", route_after_relevance_check)
    graph.add_edge("generate", "validate_answer")
    graph.add_conditional_edges(
        "validate_answer",
        route_after_validation,
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
