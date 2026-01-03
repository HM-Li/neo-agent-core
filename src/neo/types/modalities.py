from enum import Enum


class Modality(Enum):
    """
    Enum representing different modalities.
    """

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"  # pdf, text, json, etc.
    STRUCTURED = "structured"  # dict, etc.
    UNDEFINED = "undefined"  # used when the modality is not defined

    def __repr__(self):
        return self.value
