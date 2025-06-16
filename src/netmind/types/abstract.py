import pydantic
from typing import Dict, Any
from pydantic import ConfigDict
from typing_extensions import ClassVar

from netmind.constants import BASE_URL


class NetMindClient:
    def __init__(
            self,
            api_key: str | None = None,
            base_url: str | None = BASE_URL,
            **kwargs: Dict[str, Any]
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.kwargs = kwargs


class BaseModel(pydantic.BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")
