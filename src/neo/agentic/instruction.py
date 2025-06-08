from typing import Any, List, Optional

from pydantic import BaseModel, Field, field_validator

from neo.types.tool_codes import StandardToolCode


class ModelConfigs(BaseModel, extra="allow"):
    """
    The ModelConfigs class encapsulates model configurations.
    """

    model: str = Field(
        ...,
        description="The model name, e.g. gpt-4",
    )

    temperature: float = Field(
        default=None,
        description="The temperature for the model's response.",
    )

    max_tokens: int = Field(
        default=None,
        description="The maximum number of tokens for the model's response.",
    )

    top_p: float = Field(
        default=None,
        description="The top-p sampling parameter for the model's response.",
    )


class OtherConfigs(BaseModel):
    """
    The OtherConfigs class encapsulates other configurations that aren't related to the GenAI model.
    """

    timeaware: Optional[bool] = Field(
        default=None,
        description="Whether adding context providing time to the thread.",
    )

    tools: Optional[List[StandardToolCode | str]] = Field(
        default=None,
        description="Tool codes to be enabled for the model. Must be pre-registered in the ToolRegistry.",
    )

    custom_api_key: Optional[str] = Field(
        default=None,
        description="Custom API key for the model.",
    )


class Instruction(BaseModel):
    """
    The Instruction class encapsulates user or system instructions.
    """

    content: Optional[str] = None

    model_configs: Optional[ModelConfigs] = None

    other_configs: Optional[OtherConfigs] = None

    @field_validator("content", mode="after")
    @classmethod
    def _validate_content(cls, v):
        if isinstance(v, str):
            v = v.strip()
            if not v:
                raise ValueError("Instruction content must be a non-empty string.")
        return v

    def __repr__(self):
        return f"Instruction({self.content!r})"

    def __str__(self):
        return self.content if self.content else "<No instruction provided.>"
