from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from psycopg_pool import AsyncConnectionPool

from src.graph.edges import route_after_relevance_check
from src.graph.nodes import (
    check_relevance_node,
    generate_node,
    no_answer_node,
    rerank_node,
    retrieve_node,
)
from src.graph.state import RAGState


def build_graph() -> StateGraph:
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("check_relevance", check_relevance_node)
    graph.add_node("generate", generate_node)
    graph.add_node("no_answer", no_answer_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "check_relevance")
    graph.add_conditional_edges("check_relevance", route_after_relevance_check)
    graph.add_edge("generate", END)
    graph.add_edge("no_answer", END)

    return graph


async def compile_graph(pool: AsyncConnectionPool) -> CompiledStateGraph:
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()
    graph = build_graph()
    return graph.compile(checkpointer=checkpointer)
