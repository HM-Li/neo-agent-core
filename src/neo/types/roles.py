import enum


class Role(enum.Enum):
    """enum for different roles in the system"""

    USER = "user"
    ASSISTANT = "assistant"
    DEVELOPER = "developer"
    SYSTEM = "system"
    TOOL = "tool"
    UNDEFINED = (
        "undefined"  # Use this for cases where the role is not defined or is unknown
    )
