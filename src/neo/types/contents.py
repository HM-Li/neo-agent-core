from __future__ import annotations  # for TYPE_CHECKING

from typing import Any, ClassVar, List, Optional

from pydantic import BaseModel, Field, field_validator

from neo.tools import Tool
from neo.types.modalities import Modality


class BaseContent(BaseModel):
    modality: ClassVar[Modality]

    @property
    def type(self):
        name = self.__class__.__name__
        return name

    def __str__(self):
        return str(self.__dict__)

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "modality"):
            raise TypeError(
                "Subclasses of Content must define a 'modality' class attribute"
            )


class RawContent(BaseContent):
    """
    Handle raw API content block to provide flexibility
    """

    data: Any
    modality: ClassVar[Modality] = Modality.UNDEFINED


class TextContent(BaseContent):
    data: str
    modality: ClassVar[Modality] = Modality.TEXT


class ThoughtContent(BaseContent):
    """In some cases, maintain a stream of thoughts or reasoning is important for the model. E.g. tool use."""

    thought_id: str
    data: Optional[List[str]] = Field(
        default=None, description="List of thoughts. None for redacted thoughts."
    )
    modality: ClassVar[Modality] = Modality.THOUGHT


class _BytesUrlContentMixin:
    @field_validator("data", mode="before")
    @classmethod
    def validate_data(cls, value):
        # check if the value is bytes or a URL
        if isinstance(value, str):
            # check if it's a valid URL
            if not value.startswith(("http://", "https://")):
                raise ValueError("data must be a valid URL")

        elif not isinstance(value, bytes):
            raise ValueError("data must be of type bytes or a valid URL")

        return value

    @field_validator("mime_type", mode="after")
    @classmethod
    def validate_mime_type(cls, value):
        if value is not None:
            value = value.lower()
        return value

    def model_post_init(self, __context: Any) -> None:
        # mime_type and file_name are required when data is bytes
        missing_metadata = self.mime_type is None or self.file_name is None
        if isinstance(self.data, bytes) and missing_metadata:
            raise ValueError(
                "mime_type and file_name must be provided when data is bytes"
            )


class AudioContent(_BytesUrlContentMixin, BaseContent):
    mime_type: Optional[str] = None
    data: bytes | str = Field(
        description="Data in binary bytes or a URL.",
    )
    file_name: Optional[str] = None
    modality: ClassVar[Modality] = Modality.AUDIO


class AudioTextContent(_BytesUrlContentMixin, BaseContent):
    mime_type: Optional[str] = None
    data: bytes | str = Field(
        description="Data in binary bytes or a URL.",
    )
    file_name: Optional[str] = None
    modality: ClassVar[Modality] = Modality.TEXT
    text: Optional[str] = Field(
        default=None,
        description="Transcription of the audio content.",
    )
    transcription_handler: Tool = Field(
        description="Transcription handler that takes in AudioLikeContent and returns TextContent."
    )


class ImageContent(_BytesUrlContentMixin, BaseContent):
    mime_type: Optional[str] = None
    data: bytes | str = Field(
        description="Data in binary bytes or a URL.",
    )
    file_name: Optional[str] = None
    modality: ClassVar[Modality] = Modality.IMAGE


class DocumentContent(_BytesUrlContentMixin, BaseContent):
    mime_type: Optional[str] = None
    data: bytes | str = Field(
        description="Data in binary bytes or a URL.",
    )
    file_name: Optional[str] = None
    modality: ClassVar[Modality] = Modality.DOCUMENT


class DocumentTextContent(_BytesUrlContentMixin, BaseContent):
    mime_type: Optional[str] = None
    data: bytes | str = Field(
        description="Data in binary bytes or a URL.",
    )
    file_name: Optional[str] = None
    modality: ClassVar[Modality] = Modality.TEXT
    text: Optional[str] = Field(
        default=None,
        description="Text extraction of the document content.",
    )


class ToolInputContent(BaseContent):
    params: dict
    tool_name: str
    tool_use_id: str
    modality: ClassVar[Modality] = Modality.STRUCTURED


class ToolOutputContent(BaseContent):
    tool_use_id: str
    contents: List[BaseContent] | None  # can be a Content or a list of Content
    is_error: bool
    modality: ClassVar[Modality] = Modality.STRUCTURED


class BooleanContent(BaseContent):
    data: bool
    modality: ClassVar[Modality] = Modality.STRUCTURED
