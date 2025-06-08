from enum import Enum


class StandardToolCode(str, Enum):
    """
    Canonical, model‑agnostic codes for common tools.

    Extend this enum as new cross‑model tools are introduced.
    """

    IMAGE_GEN = "image_gen"
    WEB_SEARCH = "web_search"
    TRANSCRIBE = "transcribe"
