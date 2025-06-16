import os
import pytest

from netmind import NetMind, AsyncNetMind
from openai.types import CompletionUsage
from openai.types.chat.chat_completion import Choice, ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage


MODEL = "Qwen/Qwen3-8B"
MESSAGES = [
    {"role": "system", "content": "Act like you are a helpful assistant."},
    {"role": "user", "content": "Hi there!"},
]
MAX_TOKENS = 512


def assert_chat_completion(response: ChatCompletion):
    assert isinstance(response, ChatCompletion)
    assert isinstance(response.id, str)
    assert isinstance(response.created, int)
    assert isinstance(response.choices, list)
    assert isinstance(response.choices[0], Choice)
    assert isinstance(response.choices[0].message, ChatCompletionMessage)

    usage = response.usage
    assert isinstance(usage, CompletionUsage)
    assert isinstance(usage.prompt_tokens, int)
    assert isinstance(usage.completion_tokens, int)
    assert isinstance(usage.total_tokens, int)
    assert usage.prompt_tokens + usage.completion_tokens == usage.total_tokens


class TestNetMindChat:
    @pytest.fixture
    def sync_client(self) -> NetMind:
        return NetMind(api_key=os.getenv("NETMIND_API_KEY"))

    def test_create(self, sync_client: NetMind):
        response = sync_client.chat.completions.create(
            model=MODEL,
            messages=MESSAGES,
            max_tokens=MAX_TOKENS,
        )
        assert_chat_completion(response)


@pytest.mark.asyncio
class TestAsyncNetMindChat:
    @pytest.fixture
    def async_client(self) -> AsyncNetMind:
        return AsyncNetMind(api_key=os.getenv("NETMIND_API_KEY"))

    async def test_create(self, async_client: AsyncNetMind):
        response = await async_client.chat.completions.create(
            model=MODEL,
            messages=MESSAGES,
            max_tokens=MAX_TOKENS,
        )
        assert_chat_completion(response)
