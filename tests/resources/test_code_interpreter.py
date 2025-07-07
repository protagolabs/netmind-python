import os
import pytest
from netmind import NetMind, AsyncNetMind
from netmind.types.code_interpreter import (
    CodeInterpreterCodeRequest,
    CodeInterpreterCodeRunResponse,
    CodeInterpreterCodeFile,
    CodeInterpreterCodeResponse,
)


# Test data
SAMPLE_CODE_REQUEST = CodeInterpreterCodeRequest(
    language="python",
    files=[
        CodeInterpreterCodeFile(
            name="main.py",
            content="print('Hello, World!')\nprint(2 + 2)\nresult = 5 * 3\nprint(f'Result: {result}')"
        )
    ]
)

SAMPLE_CODE_REQUEST_WITH_ERROR = CodeInterpreterCodeRequest(
    language="python",
    files=[
        CodeInterpreterCodeFile(
            name="error.py",
            content="print('This will work')\nundefined_variable_that_will_cause_error"
        )
    ],
    stdin="",
    args=[],
    file_id_usage=[]
)


class TestNetMindCodeInterpreter:
    @pytest.fixture
    def sync_client(self) -> NetMind:
        return NetMind(api_key=os.getenv("NETMIND_API_KEY"))

    def test_run_code_success(self, sync_client: NetMind):
        """Test successful code execution"""
        result = sync_client.code_interpreter.run(SAMPLE_CODE_REQUEST)
        
        assert isinstance(result, CodeInterpreterCodeResponse)
        assert result.run is not None
        assert isinstance(result.run, CodeInterpreterCodeRunResponse)
        assert "Hello, World!" in result.run.stdout
        assert "4" in result.run.stdout
        assert "Result: 15" in result.run.stdout
        assert result.run.code == 0  # Success exit code

    def test_run_code_with_error(self, sync_client: NetMind):
        """Test code execution with error"""
        result = sync_client.code_interpreter.run(SAMPLE_CODE_REQUEST_WITH_ERROR)
        
        assert isinstance(result, CodeInterpreterCodeResponse)
        assert result.run is not None
        assert isinstance(result.run, CodeInterpreterCodeRunResponse)
        assert "This will work" in result.run.stdout
        assert result.run.code != 0  # Error exit code
        assert len(result.run.stderr) > 0  # Should have error output

    def test_run_code_multiple_files(self, sync_client: NetMind):
        """Test code execution with multiple files"""
        multi_file_request = CodeInterpreterCodeRequest(
            language="python",
            files=[
                CodeInterpreterCodeFile(
                    name="main.py",
                    content="from utils import add, multiply\n\nresult1 = add(3, 4)\nresult2 = multiply(5, 6)\nprint(f'Addition: {result1}')\nprint(f'Multiplication: {result2}')"
                ),
                CodeInterpreterCodeFile(
                    name="utils.py",
                    content="def add(a, b):\n    return a + b\n\ndef multiply(a, b):\n    return a * b"
                ),
            ]
        )
        
        result = sync_client.code_interpreter.run(multi_file_request)
        
        assert isinstance(result, CodeInterpreterCodeResponse)
        assert result.run is not None
        print(result.run)
        assert "Addition: 7" in result.run.stdout
        assert "Multiplication: 30" in result.run.stdout
        assert result.run.code == 0


@pytest.mark.asyncio
class TestAsyncNetMindCodeInterpreter:
    @pytest.fixture
    def async_client(self) -> AsyncNetMind:
        return AsyncNetMind(api_key=os.getenv("NETMIND_API_KEY"))

    async def test_arun_code_success(self, async_client: AsyncNetMind):
        """Test successful async code execution"""
        result = await async_client.code_interpreter.arun(SAMPLE_CODE_REQUEST)
        
        assert isinstance(result, CodeInterpreterCodeResponse)
        assert result.run is not None
        assert isinstance(result.run, CodeInterpreterCodeRunResponse)
        assert "Hello, World!" in result.run.stdout
        assert "4" in result.run.stdout
        assert "Result: 15" in result.run.stdout
        assert result.run.code == 0  # Success exit code

    async def test_arun_code_with_error(self, async_client: AsyncNetMind):
        """Test async code execution with error"""
        result = await async_client.code_interpreter.arun(SAMPLE_CODE_REQUEST_WITH_ERROR)
        
        assert isinstance(result, CodeInterpreterCodeResponse)
        assert result.run is not None
        assert isinstance(result.run, CodeInterpreterCodeRunResponse)
        assert "This will work" in result.run.stdout
        assert result.run.code != 0  # Error exit code
        assert len(result.run.stderr) > 0  # Should have error output

    async def test_arun_code_with_input(self, async_client: AsyncNetMind):
        """Test async code execution with stdin input"""
        input_request = CodeInterpreterCodeRequest(
            language="python",
            files=[
                CodeInterpreterCodeFile(
                    name="input_test.py",
                    content="name = input('Enter your name: ')\nprint(f'Hello, {name}!')"
                )
            ],
            stdin="NetMind",
            args=[],
            file_id_usage=[]
        )
        
        result = await async_client.code_interpreter.arun(input_request)
        
        assert isinstance(result, CodeInterpreterCodeResponse)
        assert result.run is not None
        assert "Hello, NetMind!" in result.run.stdout
        assert result.run.code == 0

    async def test_arun_code_with_args(self, async_client: AsyncNetMind):
        """Test async code execution with command line arguments"""
        args_request = CodeInterpreterCodeRequest(
            language="python",
            files=[
                CodeInterpreterCodeFile(
                    name="args_test.py",
                    content="import sys\nprint(f'Script name: {sys.argv[0]}')\nfor i, arg in enumerate(sys.argv[1:], 1):\n    print(f'Arg {i}: {arg}')"
                )
            ],
            stdin="",
            args=["arg1", "arg2", "test"],
            file_id_usage=[]
        )
        
        result = await async_client.code_interpreter.arun(args_request)
        
        assert isinstance(result, CodeInterpreterCodeResponse)
        assert result.run is not None
        assert "Arg 1: arg1" in result.run.stdout
        assert "Arg 2: arg2" in result.run.stdout
        assert "Arg 3: test" in result.run.stdout
        assert result.run.code == 0

