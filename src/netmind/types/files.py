from enum import Enum
from httpx import URL
from pydantic import HttpUrl
from typing import Optional, List
from netmind.types.abstract import BaseModel, ConfigDict


class FilePurpose(str, Enum):
    fine_tune = "fine-tune"
    batch = "batch"
    inference = "inference"
    code_interpreter = "code-interpreter"


class FileId(BaseModel):
    id: str


class FilePresigned(FileId):
    presigned_url: HttpUrl | URL

    model_config = ConfigDict(arbitrary_types_allowed=True)


class FileObject(FileId):
    file_name: str
    purpose: FilePurpose
    created_at: int
    bytes: Optional[int] = None
    length: Optional[int] = None
    token: Optional[int] = None


class FileList(BaseModel):
    data: List[FileObject]