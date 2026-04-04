import asyncio
import sys

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.api.app import app
from src.api.dependencies import init_components, shutdown_components

# Psycopg async cannot use Windows' default ProactorEventLoop; set policy before any
# event loop exists (pytest-asyncio 1.x may ignore a custom event_loop fixture).
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


@pytest_asyncio.fixture(scope="session")
async def components():
    c = await init_components()
    yield c
    await shutdown_components()


@pytest_asyncio.fixture(scope="session")
async def client(components):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
