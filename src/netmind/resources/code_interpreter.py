import httpx
from netmind.types.files import CodeInterpreterCodeRequest, CodeInterpreterCodeRunResponse


class CodeInterpreter():

    def __init__(self, netmind: "NetMind"):
        self.netmind = netmind

    def run(self, request_data: CodeInterpreterCodeRequest) -> CodeInterpreterCodeRunResponse | None:
        request_body = request_data.model_dump()
        response = httpx.post(
            f"{self.netmind.client.base_url}/inference-api/agent/code-interpreter/v1/execute",
            json=request_body,
            headers={"Authorization": self.netmind.client.api_key}
        )
        response.raise_for_status()
        result = response.json()
        if result.get("run"):
            return CodeInterpreterCodeRunResponse(**result["run"])
        return None


class AsyncCodeInterpreter():

    def __init__(self, netmind: "NetMind"):
        self.netmind = netmind

    async def run_code(self, request_data: CodeInterpreterCodeRequest) -> CodeInterpreterCodeRunResponse | None:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.netmind.client.base_url}/inference-api/agent/code-interpreter/v1/execute",
                json=request_data.model_dump(),
                headers={"Authorization": self.netmind.client.api_key}
            )
            response.raise_for_status()
            result = response.json()
            if result.get("run"):
                return CodeInterpreterCodeRunResponse(**result["run"])
            return None
