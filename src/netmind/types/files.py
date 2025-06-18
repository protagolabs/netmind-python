from enum import Enum
from httpx import URL
from pydantic import HttpUrl
from typing import Optional, List
from netmind.types.abstract import BaseModel


class FilePurpose(str, Enum):
    fine_tune = 'fine-tune'
    batch = 'batch'
    inference = 'inference'


class FileId(BaseModel):
    id: str


class FilePresigned(FileId):
    presigned_url: HttpUrl | URL


class FileObject(FileId):
    file_name: str
    purpose: FilePurpose
    created_at: int
    bytes: Optional[int] = None
    length: Optional[int] = None
    token: Optional[int] = None


class FileList(BaseModel):
    data: List[FileObject]