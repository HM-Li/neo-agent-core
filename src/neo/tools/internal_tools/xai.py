from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from .base import BaseInternalTool


class SearchParameters(BaseModel):
    mode: Literal["on", "off", "auto"] = "auto"
    return_citations: bool = True


class WebSearch(BaseInternalTool):
    """XAI web search tool."""

    name: ClassVar[str] = "WebSearch"
    provider: ClassVar[str] = "XAIModel"
    search_parameters: SearchParameters = Field(
        default_factory=SearchParameters,
        description="Parameters for the web search tool.",
    )
