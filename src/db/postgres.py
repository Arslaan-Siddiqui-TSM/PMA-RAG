from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from config import settings

_pool: AsyncConnectionPool | None = None


async def get_pool() -> AsyncConnectionPool:
    global _pool
    if _pool is None:
        _pool = AsyncConnectionPool(
            conninfo=settings.postgres_uri,
            max_size=settings.postgres_pool_max_size,
            kwargs={"autocommit": True, "row_factory": dict_row},
            open=False,
        )
        await _pool.open()
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


async def delete_thread_checkpoints(thread_id: str) -> None:
    """Remove all LangGraph checkpoint data for a given thread."""
    pool = await get_pool()
    async with pool.connection() as conn:
        await conn.execute(
            "DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,)
        )
        await conn.execute(
            "DELETE FROM checkpoint_blobs WHERE thread_id = %s", (thread_id,)
        )
        await conn.execute(
            "DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,)
        )
