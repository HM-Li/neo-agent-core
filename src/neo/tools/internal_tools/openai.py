from pydantic import BaseModel, Field
from .base import BaseInternalTool
from typing import Optional, ClassVar, Literal


class UserLocation(BaseModel):
    """
    {
        "type": "approximate",
        "country": "GB",
        "city": "London",
        "region": "London",
    }
    """

    type: str = "approximate"
    country: str
    city: str
    region: str


class WebSearch(BaseInternalTool):
    """
    {
        "type": "web_search_preview",
        "user_location": {
            "type": "approximate",
            "country": "GB",
            "city": "London",
            "region": "London",
        },
         "search_context_size": "low",
    }
    """

    name: ClassVar[str] = "WebSearch"
    provider: ClassVar[str] = "OpenAIResponseModel"
    type: str = "web_search_preview"
    user_location: Optional[UserLocation] = None
    search_context_size: Literal["low", "medium", "high"] = None
