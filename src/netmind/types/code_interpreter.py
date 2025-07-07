from typing import List, Optional
from netmind.types.abstract import BaseModel
from pydantic import Field


class CodeInterpreterCodeFile(BaseModel):
    name: str
    content: str


class CodeInterpreterCodeRequest(BaseModel):
    language: str
    files: List[CodeInterpreterCodeFile]
    stdin: Optional[str] = ""
    args: Optional[List[str]] = []
    file_id_usage: Optional[List[str]] = Field(
        default=[],
        description="List of file id to use in the code run, use file.create to upload files first"
    )


class CodeInterpreterCodeRunData(BaseModel):
    generated_file_name: str
    id: str
    mime_type: str


class CodeInterpreterCodeRunResponse(BaseModel):
    signal: Optional[str] = None
    stdout: str
    stderr: str
    code: int
    output: str
    memory: int
    message: Optional[str] = None
    status: Optional[str] = None
    cpu_time: int
    wall_time: int
    data: List[CodeInterpreterCodeRunData] = []


class CodeInterpreterCodeResponse(BaseModel):
    run: CodeInterpreterCodeRunResponse
    language: str
    version: str
