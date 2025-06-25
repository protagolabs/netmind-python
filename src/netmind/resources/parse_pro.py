import os
import re
from pathlib import Path
from urllib.parse import urlparse
from typing import List, Union, overload, TYPE_CHECKING
from openai._resource import SyncAPIResource, AsyncAPIResource

from netmind.types.files import FilePurpose
from netmind.types.parse_pro import (
    Formt, JsonFormat, MarkdownFormat,
    ParseTask, ParseTaskResult
)

if TYPE_CHECKING:
    from netmind import Netmind, AsyncNetmind
    from openai import OpenAI, AsyncOpenAI


def is_url(s: str) -> bool:
    parsed = urlparse(s)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


class ParsePro(SyncAPIResource):

    def __init__(self, netmind_clinet: 'Netmind', openai_client: 'OpenAI'):
        self.client = netmind_clinet
        super().__init__(openai_client)

    def _prepare_source(self, source: Union[str, Path]) -> str:
        if isinstance(source, Path):
            source = str(source)
        if source.startswith("file-"):
            file = self.client.files.retrieve(source)
            if file.purpose != FilePurpose.inference:
                raise ValueError(
                    f"File {source} is not suitable for inference, expected purpose: {FilePurpose.inference}"
                )
            source = self.client.files.retrieve_url(file.id).presigned_url
        elif is_url(source):
            pass
        else:
            file = self.client.files.create(source, purpose=FilePurpose.inference)
            source = self.client.files.retrieve_url(file.id).presigned_url
        if not isinstance(source, str):
            raise ValueError("Source must be a string or a Path object.")
        return source

    @overload
    def parse(self, source: str) -> Union[JsonFormat, MarkdownFormat]: ...

    @overload
    def parse(self, source: Path) -> Union[JsonFormat, MarkdownFormat]: ...

    def parse(
            self,
            source: Union[str, Path],
            format: Formt = Formt.markdown,
            timeout: float = 5 * 60
    ) -> Union[JsonFormat, MarkdownFormat]:
        source = self._prepare_source(source)
        response = self._post(
            "/inference-api/agent/v1/parse-pdf",
            body={"url": source, "format": format},
            options={'timeout': timeout},
            cast_to = Union[JsonFormat, MarkdownFormat],
        )
        return response

    @overload
    def aparse(self, source: str) -> Union[JsonFormat, MarkdownFormat]: ...

    @overload
    def aparse(self, source: Path) -> Union[JsonFormat, MarkdownFormat]: ...

    def aparse(
            self,
            source: Union[str, Path],
            format: Formt = Formt.markdown,
            timeout: float = 5 * 60
    ) -> ParseTask:
        source = self._prepare_source(source)
        response = self._post(
            "/inference-api/agent/v1/parse-pdf/async",
            body={"url": source, "format": format},
            options={'timeout': timeout},
            cast_to=ParseTask,
        )
        return response

    def aresult(self, task_id: str) -> ParseTaskResult:
        if not task_id:
            raise ValueError(f"Expected a non-empty value for `task_id` but received {task_id!r}")

        response = self._get(
            f"/inference-api/agent/v1/parse-pdf/async/{task_id}",
            cast_to=ParseTaskResult,
        )
        return response


class AsyncParsePro(AsyncAPIResource):
    def __init__(self, netmind_clinet: 'AsyncNetmind', openai_client: 'AsyncOpenAI'):
        self.client = netmind_clinet
        super().__init__(openai_client)


    async def _prepare_source(self, source: Union[str, Path]) -> str:
        if isinstance(source, Path):
            source = str(source)
        if source.startswith("file-"):
            file = await self.client.files.retrieve(source)
            if file.purpose != FilePurpose.inference:
                raise ValueError(
                    f"File {source} is not suitable for inference, expected purpose: {FilePurpose.inference}"
                )
            source = (await self.client.files.retrieve_url(file.id)).presigned_url
        elif is_url(source):
            pass
        else:
            file = await self.client.files.create(source, purpose=FilePurpose.inference)
            source = (await self.client.files.retrieve_url(file.id)).presigned_url
        if not isinstance(source, str):
            raise ValueError("Source must be a string or a Path object.")
        return source

    @overload
    async def parse(self, source: str) -> Union[JsonFormat, MarkdownFormat]: ...

    @overload
    async def parse(self, source: Path) -> Union[JsonFormat, MarkdownFormat]: ...

    async def parse(
            self,
            source: Union[str, Path],
            format: Formt = Formt.markdown,
            timeout: float = 5 * 60
    ) -> Union[JsonFormat, MarkdownFormat]:
        source = await self._prepare_source(source)
        response = await self._post(
            "/inference-api/agent/v1/parse-pdf",
            body={"url": source, "format": format},
            options={'timeout': timeout},
            cast_to=Union[JsonFormat, MarkdownFormat],
        )
        return response

    @overload
    async def aparse(self, source: str) -> Union[JsonFormat, MarkdownFormat]: ...

    @overload
    async def aparse(self, source: Path) -> Union[JsonFormat, MarkdownFormat]: ...

    async def aparse(
            self,
            source: Union[str, Path],
            format: Formt = Formt.markdown,
            timeout: float = 5 * 60
    ) -> ParseTask:
        source = await self._prepare_source(source)
        response = await self._post(
            "/inference-api/agent/v1/parse-pdf/async",
            body={"url": source, "format": format},
            options={'timeout': timeout},
            cast_to=ParseTask,
        )
        return response

    async def aresult(self, task_id: str) -> ParseTaskResult:
        if not task_id:
            raise ValueError(f"Expected a non-empty value for `task_id` but received {task_id!r}")

        response = await self._get(
            f"/inference-api/agent/v1/parse-pdf/async/{task_id}",
            cast_to=ParseTaskResult,
        )
        return response
