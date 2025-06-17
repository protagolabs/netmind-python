import pydantic
from typing import Dict, Any
from pydantic import ConfigDict
from typing_extensions import ClassVar
from openai import BaseModel as OpenAIBaseModel

from netmind.constants import BASE_URL
from netmind.version import VERSION


class NetMindClient:
    def __init__(
            self,
            api_key: str | None = None,
            base_url: str | None = BASE_URL,
            **kwargs
    ):
        self.api_key = api_key
        self.base_url = base_url
        kwargs.setdefault("version", VERSION)
        kwargs.setdefault("_strict_response_validation", False)
        self.kwargs = kwargs


class BaseModel(OpenAIBaseModel):
    pass
