from typing import List, Optional

from neo.tools.tool import Tool
from neo.types.errors import ToolError
from neo.types.tool_codes import StandardToolCode
from neo.utils.singleton import Singleton


class ToolRegistry(metaclass=Singleton):
    """
    Registry that maps a *model_class class* (e.g. `GoogleAIModel`) to a
    dictionary of **standard tool codes** → concrete `Tool` instances.
    """

    def __init__(self) -> None:
        self._registry: dict[type, dict[str, Tool]] = {}
        self._model_agnostic_registry: dict[str, Tool] = {}

    def register_tool(
        self,
        model_class: type,
        code: StandardToolCode | str,
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

    def register_model_agnostic_tool(
        self,
        code: StandardToolCode | str,
        tool: Tool,
    ) -> None:
        """
        Register a model-agnostic tool that can be used by any model.
        """
        code_key = str(code)

        if code_key in self._model_agnostic_registry:
            raise ValueError(
                f"Model-agnostic tool code '{code_key}' already registered."
            )

        self._model_agnostic_registry[code_key] = tool

    def get_tool(self, model_class: type, code: str | StandardToolCode) -> Tool:
        """
        Retrieve a registered tool. First checks model-specific tools,
        then falls back to model-agnostic tools.
        """
        # Convert StandardToolCode enum to its value for lookup
        lookup_key = code.value if isinstance(code, StandardToolCode) else str(code)

        # First try model-specific tools
        try:
            return self._registry[model_class][lookup_key]
        except KeyError:
            pass

        # Fall back to model-agnostic tools
        if lookup_key in self._model_agnostic_registry:
            return self._model_agnostic_registry[lookup_key]

        # Tool not found anywhere
        raise KeyError(
            f"No tool '{code}' registered for model_class '{model_class.__name__}' "
            f"or as model-agnostic tool."
        )

    def list_tools(self, model_class: type) -> List[str]:
        """
        List all tools registered for a specific model_class, including model-agnostic tools.
        """
        model_specific = list(self._registry.get(model_class, {}).keys())
        model_agnostic = list(self._model_agnostic_registry.keys())
        return model_specific + model_agnostic

    def list_model_agnostic_tools(self) -> List[str]:
        """
        List all model-agnostic tools.
        """
        return list(self._model_agnostic_registry.keys())

    @property
    def model_supported_tools(self) -> dict[type, dict[str, Tool]]:
        """Return the entire registry structure."""
        return self._registry

    @property
    def model_agnostic_tools(self) -> dict[str, Tool]:
        """Return all model-agnostic tools."""
        return self._model_agnostic_registry


def make_tool(
    _func=None,
    *,
    code: Optional[StandardToolCode | str] = None,
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
            # Check if it's an internal tool class (subclass of BaseInternalTool)
            if (
                isinstance(func, type)
                and hasattr(func, "__bases__")
                and any("BaseInternalTool" in str(base) for base in func.__mro__)
            ):
                # For internal tools, instantiate and return the instance directly
                this_tool = func()
            elif callable(func):
                # For regular functions, wrap in Tool
                this_tool = Tool(func=func)
            else:
                raise ToolError("Function is not callable.")

        # Use provided code or fall back to the (possibly overridden) tool name.
        # For StandardToolCode enums, use the value instead of the string representation
        if code is not None:
            code_key = code.value if isinstance(code, StandardToolCode) else str(code)
        else:
            code_key = this_tool.name

        if models is None:
            # Register as model-agnostic tool when no models are provided
            ToolRegistry().register_model_agnostic_tool(
                code=code_key,
                tool=this_tool,
            )
            return this_tool

        # Register the tool for specific models.
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


from neo.models.providers.anthropic import AnthropicModel
from neo.models.providers.openai import OpenAIResponseModel
from neo.models.providers.xai import XAIModel

# ==========
from neo.tools.image_gen.openai import agenerate_image
from neo.tools.internal_tools.anthropic import WebSearch as AnthropicWebSearch
from neo.tools.internal_tools.openai import WebSearch as OpenAIWebSearch
from neo.tools.internal_tools.xai import WebSearch as XAIWebSearch
from neo.tools.speech.openai import aspeech2text

tools = [
    (
        agenerate_image,
        StandardToolCode.IMAGE_GEN,
        [OpenAIResponseModel, AnthropicModel],
    ),
    (OpenAIWebSearch, StandardToolCode.WEB_SEARCH, [OpenAIResponseModel]),
    (AnthropicWebSearch, StandardToolCode.WEB_SEARCH, [AnthropicModel]),
    (XAIWebSearch, StandardToolCode.WEB_SEARCH, [XAIModel]),
    (aspeech2text, StandardToolCode.TRANSCRIBE, None),  # model-agnostic
]

for func, code, models in tools:
    make_tool(
        _func=func,
        code=code,
        models=models,
    )
