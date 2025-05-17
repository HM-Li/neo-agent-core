from neo.tools.tool import Tool
from neo.types.errors import ToolError
from enum import Enum

from typing import List, Optional

from neo.utils.singleton import Singleton


class StandardToolCode(str, Enum):
    """
    Canonical, model‑agnostic codes for common tools.

    Extend this enum as new cross‑model tools are introduced.
    """

    IMAGE_GEN = "image_gen"
    WEB_SEARCH = "web_search"
    TRANSCRIBE = "transcribe"


class ToolRegistry(metaclass=Singleton):
    """
    Registry that maps a *model_class class* (e.g. `GoogleAIModel`) to a
    dictionary of **standard tool codes** → concrete `Tool` instances.
    """

    def __init__(self) -> None:
        self._registry: dict[type, dict[str, Tool]] = {}

    def register_tool(
        self,
        model_class: type,
        code: str | StandardToolCode,
        tool: Tool,
    ) -> None:
        """
        Register a `Tool` under a model_class with the given standard code.
        """
        if model_class is None:
            raise ToolError("Provider must be specified when registering a tool.")

        model_dict = self._registry.setdefault(model_class, {})
        code_key = str(code)

        if code_key in model_dict:
            raise ValueError(
                f"Tool code '{code_key}' already registered for model_class "
                f"'{model_class.__name__}'."
            )

        model_dict[code_key] = tool

    def get_tool(self, model_class: type, code: str | StandardToolCode) -> Tool:
        """
        Retrieve a registered tool.
        """
        try:
            return self._registry[model_class][str(code)]
        except KeyError as exc:
            raise KeyError(
                f"No tool '{code}' registered for model_class '{model_class.__name__}'."
            ) from exc

    def list_tools(self, model_class: type) -> List[str]:
        """
        List all tools registered for a specific model_class.
        """
        return list(self._registry.get(model_class, {}).keys())

    @property
    def all_tools(self) -> dict[type, dict[str, Tool]]:
        """Return the entire registry structure."""
        return self._registry


def make_tool(
    _func=None,
    *,
    code: Optional[StandardToolCode] = None,
    models: Optional[List[type]] = None,
):
    """
    Decorator that wraps a callable into a `Tool` **and** registers it
    in the global :class:`ToolRegistry`.

    It may be used with or without arguments:

    ```
    @make_tool(models=[GoogleAIModel], code=StandardToolCode.IMAGE_GEN)
    def generate_image(...):
        ...
    ```

    or, without arguments (defaults to function name as code **and** requires
    explicit `models`):

    ```
    @make_tool
    def image_gen(...):
        ...
    ```
    """

    def decorator(func):
        if isinstance(func, Tool):
            this_tool = func
        else:
            if not callable(func):
                raise ToolError("Function is not callable.")

            this_tool = Tool(func=func)

        if models is None:
            return this_tool

        # Use provided code or fall back to the (possibly overridden) tool name.
        code_key = str(code or this_tool.name)

        # Register the tool.
        for model_class in models:
            ToolRegistry().register_tool(
                model_class=model_class,
                code=code_key,
                tool=this_tool,
            )

        return this_tool

    # Support both @make_tool and @make_tool(...) syntaxes.
    if _func is None:
        return decorator
    return decorator(_func)


# ==========
from neo.tools.image_gen.openai import agenerate_image
from neo.tools.internal_tools.openai import WebSearch
from neo.tools.speech.openai import aspeech2text
from neo.models.providers.openai import OpenAIResponseModel
from neo.models.providers.anthropic import AnthropicModel

tools = [
    (
        agenerate_image,
        StandardToolCode.IMAGE_GEN,
        [OpenAIResponseModel, AnthropicModel],
    ),
    (WebSearch, StandardToolCode.WEB_SEARCH, [OpenAIResponseModel]),
    (aspeech2text, StandardToolCode.TRANSCRIBE, [OpenAIResponseModel, AnthropicModel]),
]

for func, code, models in tools:
    make_tool(
        _func=func,
        code=code,
        models=models,
    )
