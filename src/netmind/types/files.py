from enum import Enum
from httpx import URL
from pydantic import BaseModel, HttpUrl

from netmind.types.abstract import BaseModel


class FilePurpose(str, Enum):
    fine_tune = 'fine-tune'
    batch = 'batch'
    inference = 'inference'


class FilePresigned(BaseModel):
    id: str
    presigned_url: HttpUrl | URL
