import uuid
from pydantic import model_validator


class IDMixin:
    """mixin for generating unique identifiers. Ensures no two classes share the same ID prefix."""

    @model_validator(mode="before")
    @classmethod
    def _no_user_provided_id(cls, values):
        if isinstance(values, dict) and values.get("id", None) is not None:
            raise ValueError("User provided ID is not allowed.")

        return values

    @classmethod
    def generate_id(cls) -> str:
        """Generate a unique identifier.

        Returns:
            str: A unique identifier as a string.
        """
        prefix = cls.__name__.lower()
        return f"{prefix}_{uuid.uuid4()}"
