import os
import pytest

from netmind import NetMind, AsyncNetMind
from openai.types.create_embedding_response import CreateEmbeddingResponse


MODEL = "nvidia/NV-Embed-v2"
INPUT = ['The food was delicious and the waiter...']
DIMENSION = 4096


def assert_embeddings(response: CreateEmbeddingResponse):
    assert isinstance(response, CreateEmbeddingResponse)
    assert response.model == MODEL
    assert isinstance(response.data, list)
    assert len(response.data) == len(INPUT)
    assert len(response.data[0].embedding) == DIMENSION


class TestNetMindEmbeddings:
    @pytest.fixture
    def sync_client(self) -> NetMind:
        return NetMind(api_key=os.getenv("NETMIND_API_KEY"))

    def test_create(self, sync_client: NetMind):
        response = sync_client.embeddings.create(
            model=MODEL,
            input=INPUT,
        )
        assert_embeddings(response)


@pytest.mark.asyncio
class TestAsyncNetMindEmbeddings:
    @pytest.fixture
    def async_client(self) -> AsyncNetMind:
        return AsyncNetMind(api_key=os.getenv("NETMIND_API_KEY"))

    async def test_create(self, async_client: AsyncNetMind):
        response = await async_client.embeddings.create(
            model=MODEL,
            input=INPUT,
        )
        assert_embeddings(response)
