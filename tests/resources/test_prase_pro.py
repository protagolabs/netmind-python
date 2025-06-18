import os
import pytest
import time
import asyncio
from netmind import NetMind, AsyncNetMind
from netmind.types.parse_pro import TaskStatus


FILE_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "demo",
    "table.pdf"
)
WAIT_TIME = 10
FINAL_STATUSES = [
    TaskStatus.success,
    TaskStatus.failed,
    TaskStatus.pending
]


class TestNetMindParsePro:
    @pytest.fixture
    def sync_client(self) -> NetMind:
        return NetMind(api_key=os.getenv("NETMIND_API_KEY"))

    def test_parse(self, sync_client: NetMind):
        result = sync_client.parse_pro.parse(FILE_PATH, format="json")

        assert isinstance(result, list)
        assert len(result) > 0

    def test_aparse(self, sync_client: NetMind):
        result = sync_client.parse_pro.aparse(FILE_PATH, format="markdown")
        assert hasattr(result, "task_id")
        assert hasattr(result, "status")
        assert result.status == "PENDING"

        time.sleep(WAIT_TIME)

        task_result = sync_client.parse_pro.aresult(result.task_id)
        assert task_result.status in FINAL_STATUSES
        assert isinstance(task_result.data, str)


@pytest.mark.asyncio
class TestAsyncNetMindParsePro:
    @pytest.fixture
    def async_client(self) -> AsyncNetMind:
        return AsyncNetMind(api_key=os.getenv("NETMIND_API_KEY"))

    async def test_parse(self, async_client: AsyncNetMind):
        result = await async_client.parse_pro.parse(FILE_PATH, format="json")
        assert isinstance(result, list)
        assert len(result) > 0

    async def test_aparse(self, async_client: AsyncNetMind):
        result = await async_client.parse_pro.aparse(FILE_PATH, format="markdown")
        assert hasattr(result, "task_id")
        assert hasattr(result, "status")
        assert result.status == "PENDING"

        await asyncio.sleep(WAIT_TIME)

        task_result = await async_client.parse_pro.aresult(result.task_id)
        assert task_result.status in FINAL_STATUSES
        assert isinstance(task_result.data, str)
