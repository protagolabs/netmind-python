import os
import openai
import pytest
from netmind import NetMind, AsyncNetMind


FILE_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "demo",
    "table.pdf")
PURPOSE = "inference"


class TestNetMindFiles:
    @pytest.fixture
    def sync_client(self) -> NetMind:
        return NetMind(api_key=os.getenv("NETMIND_API_KEY"))

    def test_list_files(self, sync_client: NetMind):
        files = sync_client.files.list()
        assert isinstance(files, list)
        if len(files) > 0:
            assert hasattr(files[0], "id")
            assert hasattr(files[0], "file_name")
            assert hasattr(files[0], "purpose")

    def test_file_lifecycle(self, sync_client: NetMind):
        create_resp = sync_client.files.create(
            file=FILE_PATH,
            purpose=PURPOSE
        )
        file_id = create_resp.id
        assert file_id.startswith("file-")

        retrieve_resp = sync_client.files.retrieve(file_id)
        assert retrieve_resp.id == file_id
        assert retrieve_resp.purpose == PURPOSE

        sync_client.files.delete(file_id)
        with pytest.raises(openai.NotFoundError):
            sync_client.files.retrieve(file_id)


@pytest.mark.asyncio
class TestAsyncNetMindFiles:
    @pytest.fixture
    def async_client(self) -> AsyncNetMind:
        return AsyncNetMind(api_key=os.getenv("NETMIND_API_KEY"))

    async def test_list_files(self, async_client: AsyncNetMind):
        files = await async_client.files.list()
        assert isinstance(files, list)
        if len(files) > 0:
            assert hasattr(files[0], "id")
            assert hasattr(files[0], "file_name")
            assert hasattr(files[0], "purpose")

    async def test_file_lifecycle(self, async_client: AsyncNetMind):
        create_resp = await async_client.files.create(
            file=FILE_PATH,
            purpose=PURPOSE
        )
        file_id = create_resp.id
        assert file_id.startswith("file-")

        retrieve_resp = await async_client.files.retrieve(file_id)
        assert retrieve_resp.id == file_id

        await async_client.files.delete(file_id)
        with pytest.raises(openai.NotFoundError):
            await async_client.files.retrieve(file_id)

