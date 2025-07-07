from netmind.types.code_interpreter import CodeInterpreterCodeRequest, CodeInterpreterCodeResponse
from openai._resource import SyncAPIResource, AsyncAPIResource
from openai import OpenAI, AsyncOpenAI


class CodeInterpreter(SyncAPIResource):

    def __init__(self, openai_client: OpenAI):
        super().__init__(openai_client)

    def run(self, request_data: CodeInterpreterCodeRequest) -> CodeInterpreterCodeResponse | None:
        return self._post(
            "/inference-api/agent/code-interpreter/v1/execute",
            body=request_data.model_dump(),
            options={'timeout': 30, "max_retries": 3},
            cast_to=CodeInterpreterCodeResponse
        )


class AsyncCodeInterpreter(AsyncAPIResource):

    def __init__(self, openai_client: AsyncOpenAI):
        super().__init__(openai_client)

    async def arun(self, request_data: CodeInterpreterCodeRequest) -> CodeInterpreterCodeResponse | None:
        return await self._post(
            "/inference-api/agent/code-interpreter/v1/execute",
            body=request_data.model_dump(),
            options={'timeout': 30, "max_retries": 3},
            cast_to=CodeInterpreterCodeResponse
        )
