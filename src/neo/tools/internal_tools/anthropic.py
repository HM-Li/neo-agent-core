from typing import ClassVar, List, Optional

from pydantic import BaseModel, Field

from .base import BaseInternalTool


class UserLocation(BaseModel):
    """
    User location for localizing search results.

    Example:
    {
        "type": "approximate",
        "city": "San Francisco",
        "country": "US"
    }
    """

    type: str = "approximate"
    city: str
    country: str


class WebSearch(BaseInternalTool):
    """
    Anthropic Web Search internal tool based on official documentation.

    Tool definition format:
    {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": 5,
        "allowed_domains": ["example.com"],
        "user_location": {
            "type": "approximate",
            "city": "San Francisco",
            "country": "US"
        }
    }

    Features:
    - Real-time web content access
    - Automatic source citations
    - Localization support
    - Domain filtering
    - Usage limits
    """

    name: str = "web_search"
    provider: ClassVar[str] = "AnthropicModel"
    type: str = "web_search_20250305"
    max_uses: Optional[int] = Field(
        default=None, description="Maximum number of searches allowed"
    )
    allowed_domains: Optional[List[str]] = Field(
        default=None, description="List of allowed domains to search"
    )
    user_location: Optional[UserLocation] = Field(
        default=None, description="User location for localizing search results"
    )
