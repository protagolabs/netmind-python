

import pytest
from unittest.mock import Mock, AsyncMock, patch

from netmind.resources.code_interpreter import CodeInterpreter, AsyncCodeInterpreter
from netmind.types.files import (
    CodeInterpreterCodeRequest,
    CodeInterpreterCodeRunResponse,
    CodeInterpreterCodeFile,
    CodeInterpreterCodeRunData,
)


# Test data
SAMPLE_CODE_REQUEST = CodeInterpreterCodeRequest(
    language="python",
    files=[
        CodeInterpreterCodeFile(
            name="main.py",
            content="print('Hello, World!')\nprint(2 + 2)"
        )
    ],
    stdin="",
    args=[],
    file_id_usage=["file-123"]
)

SAMPLE_CODE_RUN_RESPONSE_DATA = {
    "run": {
        "signal": None,
        "stdout": "Hello, World!\n4\n",
        "stderr": "",
        "code": 0,
        "output": "Hello, World!\n4\n",
        "memory": 1024,
        "message": "Execution completed successfully",
        "status": "success",
        "cpu_time": 100,
        "wall_time": 150,
        "data": [
            {
                "generated_file_name": "output.txt",
                "id": "file-abc",
                "mime_type": "text/plain"
            }
        ]
    }
}


class TestCodeInterpreter:
    @pytest.fixture
    def mock_netmind(self):
        mock_netmind = Mock()
        mock_netmind.client.base_url = "https://api.netmind.ai/inference-api/agent/code-interpreter/v1"
        mock_netmind.client.api_key = "mock-api-key"
        return mock_netmind

    @pytest.fixture
    def code_interpreter(self, mock_netmind):
        return CodeInterpreter(mock_netmind)

    @patch('httpx.post')
    def test_run_code_success(self, mock_post, code_interpreter):
        # Mock successful response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = SAMPLE_CODE_RUN_RESPONSE_DATA
        mock_post.return_value = mock_response

        # Test code execution
        response = code_interpreter.run(SAMPLE_CODE_REQUEST)

        # Assertions
        assert isinstance(response, CodeInterpreterCodeRunResponse)
        assert response.stdout == "Hello, World!\n4\n"
        assert response.code == 0
        assert response.status == "success"
        assert response.memory == 1024
        assert len(response.data) == 1
        assert isinstance(response.data[0], CodeInterpreterCodeRunData)
        assert response.data[0].generated_file_name == "output.txt"

        # Verify the request was made correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]['headers']['Authorization'] == 'mock-api-key'
        assert 'json' in call_args[1]
        assert call_args[1]['json']['file_id_usage'] == ["file-123"]

    @patch('httpx.post')
    def test_run_code_no_run_in_response(self, mock_post, code_interpreter):
        # Mock response without 'run' key
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"status": "failed"}
        mock_post.return_value = mock_response

        # Test code execution
        response = code_interpreter.run(SAMPLE_CODE_REQUEST)

        # Should return None when no 'run' key
        assert response is None


@pytest.mark.asyncio
class TestAsyncCodeInterpreter:
    @pytest.fixture
    def mock_netmind(self):
        mock_netmind = Mock()
        mock_netmind.client.base_url = "https://api.netmind.ai/inference-api/agent/code-interpreter/v1"
        mock_netmind.client.api_key = "mock-api-key"
        return mock_netmind

    @pytest.fixture
    def async_code_interpreter(self, mock_netmind):
        return AsyncCodeInterpreter(mock_netmind)

    @patch('httpx.AsyncClient')
    async def test_run_code_success(self, mock_async_client, async_code_interpreter):
        # Mock async client and response
        mock_client_instance = AsyncMock()
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = SAMPLE_CODE_RUN_RESPONSE_DATA
        mock_client_instance.post.return_value = mock_response
        mock_async_client.return_value.__aenter__.return_value = mock_client_instance

        # Test code execution
        response = await async_code_interpreter.run_code(SAMPLE_CODE_REQUEST)

        # Assertions
        assert isinstance(response, CodeInterpreterCodeRunResponse)
        assert response.stdout == "Hello, World!\n4\n"
        assert response.code == 0
        assert response.status == "success"
        assert len(response.data) == 1
        assert isinstance(response.data[0], CodeInterpreterCodeRunData)
        assert response.data[0].id == "file-abc"

    @patch('httpx.AsyncClient')
    async def test_run_code_no_run_in_response(self, mock_async_client, async_code_interpreter):
        # Mock async client and response without 'run' key
        mock_client_instance = AsyncMock()
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"status": "failed"}
        mock_client_instance.post.return_value = mock_response
        mock_async_client.return_value.__aenter__.return_value = mock_client_instance

        # Test code execution
        response = await async_code_interpreter.run_code(SAMPLE_CODE_REQUEST)

        # Should return None when no 'run' key
        assert response is None


# Test utility functions
def test_code_interpreter_code_file():
    """Test CodeInterpreterCodeFile model"""
    code_file = CodeInterpreterCodeFile(
        name="test.py",
        content="print('test')"
    )
    assert code_file.name == "test.py"
    assert code_file.content == "print('test')"


def test_code_interpreter_code_request():
    """Test CodeInterpreterCodeRequest model"""
    request = CodeInterpreterCodeRequest(
        language="python",
        files=[
            CodeInterpreterCodeFile(name="main.py", content="print('hello')")
        ],
        stdin="input_text",
        args=["--verbose"],
        file_id_usage=["file-xyz"]
    )
    assert request.language == "python"
    assert len(request.files) == 1
    assert request.stdin == "input_text"
    assert request.args == ["--verbose"]
    assert request.file_id_usage == ["file-xyz"]


def test_code_interpreter_code_run_response():
    """Test CodeInterpreterCodeRunResponse model"""
    response = CodeInterpreterCodeRunResponse(
        stdout="output",
        stderr="",
        code=0,
        output="output",
        memory=512,
        cpu_time=50,
        wall_time=100,
        data=[
            CodeInterpreterCodeRunData(
                generated_file_name="result.csv",
                id="file-res-123",
                mime_type="text/csv"
            )
        ]
    )
    assert response.stdout == "output"
    assert response.code == 0
    assert response.memory == 512
    assert len(response.data) == 1
    assert response.data[0].generated_file_name == "result.csv"
