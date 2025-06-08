import datetime
import uuid
from typing import Any, List, Literal, Optional, TypeVar, Union

from pydantic import BaseModel, Field, field_validator, model_serializer

from neo.types.contents import BaseContent, RawContent, TextContent
from neo.types.modalities import Modality
from neo.types.roles import Role
from neo.utils.common import get_current_utc_timestamp
from neo.utils.ids import IDMixin

AcceptableContent = TypeVar("AcceptableContent", str, BaseContent)


class Context(BaseModel, IDMixin):
    """A context is a collection of data input into a model or returned by a model."""

    contents: Optional[List[BaseContent]] = []
    id: str = Field(
        default_factory=lambda x: Context.generate_id(),
        description="A unique identifier for the context.",
    )
    provider_role: Role = Role.USER
    provider_name: Optional[str] = None
    provider_context_id: Optional[str] = (
        None  # track external system id like tool call id.
    )
    time_provided: str = Field(
        default_factory=get_current_utc_timestamp,
        description="The time the context was provided in ISO 8601 format.",
    )

    @field_validator("contents", mode="before")
    @classmethod
    def validate_contents(cls, contents: AcceptableContent) -> List[BaseContent]:
        """Validate the contents of the context."""
        if contents is None:
            return []
        if isinstance(contents, str):
            return [TextContent(data=contents)]
        if isinstance(contents, BaseContent):
            return [contents]
        return contents

    @field_validator("provider_role", mode="before")
    @classmethod
    def validate_provider_role(cls, provider_role: str | Role) -> Role:
        """Validate the provider role."""
        if isinstance(provider_role, str):
            return Role(provider_role)
        return provider_role

    def _display_contents(self) -> str:
        def display(c):
            match c:
                case TextContent() | RawContent():
                    return c.data
                case BaseContent():
                    return f"<Type: {c.type}>"
                case _:
                    return str(c)

        contents = [display(c) for c in self.contents]
        return ", ".join(contents)

    def __repr__(self):
        contents = self._display_contents()
        return (
            f"Context(\n"
            f"    contents={contents},\n"
            f"    provider_name={self.provider_name},\n"
            f"    provider_role={self.provider_role}\n"
            f")"
        )

    def __str__(self):
        contents = self._display_contents()
        return contents

    @property
    def require_multimodal(self):
        return any(
            [
                c.modality not in [Modality.TEXT, Modality.STRUCTURED]
                for c in self.contents
            ]
        )

    @model_serializer(mode="wrap", when_used="always")
    def customise_all_serialization(self, original_serializer, **kwargs):
        """Override the serialization method to include the contents.
        This is a workaround for the issue with Pydantic v2.0.0b1 where model_dump
        does not handle ClassVar fields correctly.
        """
        data = original_serializer(self, **kwargs)
        data["contents"] = [c.model_dump() for c in self.contents]
        return data


# %%
# %%
