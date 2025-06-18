from enum import Enum
from netmind.types.abstract import BaseModel
from pydantic import PrivateAttr
from typing import List, Dict, Annotated, Any


class Formt(str, Enum):
    json = "json"
    markdown = "markdown"


JsonFormat = Annotated[List[Dict[str, Any]], "List of blocks in structured JSON format"]
MarkdownFormat = Annotated[str, "Markdown-formatted plain text"]


class TaskStatus(str, Enum):
    pending = "PENDING"
    received = "RECEIVED"
    started = "STARTED"
    success = "SUCCESS"
    failed = "FAILED"
    revoked = "REVOKED"
    rejected = "REJECTED"
    retry = "RETRY"
    ignored = "IGNORED"


class ParseTask(BaseModel):
    task_id: str
    status: str


class ParseTaskResult(BaseModel):
    task_id: str
    status: TaskStatus
    data: JsonFormat | MarkdownFormat | None = None
    error: str | None = None
    page_size: int | None = None

    def is_successful(self) -> bool:
        return self.status == TaskStatus.success and self.data is not None

    def is_failed(self) -> bool:
        return self.status == TaskStatus.failed and self.error is not None