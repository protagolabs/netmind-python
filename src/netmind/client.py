import os
from functools import cached_property
from openai import OpenAI, AsyncOpenAI

from netmind.exceptions import NetMindError
from netmind.constants import BASE_URL
from netmind.types import NetMindClient
from netmind.resources import (
    Chat, AsyncChat,
    Embeddings, AsyncEmbeddings,
    Files, AsyncFiles,
    ParsePro, AsyncParsePro
)


class NetMind:
    def __init__(
            self,
            *,
            api_key: str | None = None,
            base_url: str | None = None,
            **kwargs,
    ):

        # get api key
        if not api_key:
            api_key = os.environ.get("NETMIND_API_KEY")

        if not api_key:
            raise NetMindError(
                "The api_key client option must be set either by passing api_key to the client or by setting the "
                "NETMIND_API_KEY environment variable"
            )

        # get base url
        if not base_url:
            base_url = os.environ.get("NETMIND_BASE_URL")
        if not base_url:
            base_url = BASE_URL
        if not base_url.endswith("/inference-api/openai/v1"):
            inference_url = base_url + "/inference-api/openai/v1"
        else:
            inference_url = base_url

        if base_url.endswith("/inference-api/openai/v1"):
            base_url = base_url.replace("/inference-api/openai/v1", "")
        else:
            base_url = base_url

        self.client = NetMindClient(
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

        self._openai_client: OpenAI = OpenAI(
            api_key=api_key,
            base_url=base_url, **kwargs
        )
        self._inference_client: OpenAI = OpenAI(
            api_key=api_key,
            base_url=inference_url, **kwargs
        )

    @cached_property
    def chat(self):
        return Chat(self._inference_client)

    @cached_property
    def embeddings(self):
        return Embeddings(self._inference_client)

    @cached_property
    def files(self):
        return Files(self._openai_client)

    @cached_property
    def parse_pro(self):
        return ParsePro(self, self._openai_client)


class AsyncNetMind:
    def __init__(
            self,
            *,
            api_key: str | None = None,
            base_url: str | None = None,
            **kwargs,
    ):

        # get api key
        if not api_key:
            api_key = os.environ.get("NETMIND_API_KEY")

        if not api_key:
            raise NetMindError(
                "The api_key client option must be set either by passing api_key to the client or by setting the "
                "NETMIND_API_KEY environment variable"
            )

        # get base url
        if not base_url:
            base_url = os.environ.get("NETMIND_BASE_URL")

        if not base_url:
            base_url = BASE_URL

        if not base_url.endswith("/inference-api/openai/v1"):
            inference_url = base_url + "/inference-api/openai/v1"
        else:
            inference_url = base_url

        if base_url.endswith("/inference-api/openai/v1"):
            base_url = base_url.replace("/inference-api/openai/v1", "")
        else:
            base_url = base_url

        self.client = NetMindClient(
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

        self._openai_client: AsyncOpenAI = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url, **kwargs
        )

        self._inference_client: AsyncOpenAI = AsyncOpenAI(
            api_key=api_key,
            base_url=inference_url, **kwargs
        )

    @cached_property
    def chat(self):
        return AsyncChat(self._inference_client)

    @cached_property
    def embeddings(self):
        return AsyncEmbeddings(self._inference_client)

    @cached_property
    def files(self):
        return AsyncFiles(self._openai_client)

    @cached_property
    def parse_pro(self):
        return AsyncParsePro(self, self._openai_client)