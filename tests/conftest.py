import asyncio
import sys

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.api.app import app
from src.api.dependencies import init_components, shutdown_components


@pytest.fixture(scope="session")
def event_loop():
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


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
