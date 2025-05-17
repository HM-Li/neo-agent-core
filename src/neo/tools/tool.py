import inspect
from pydantic import BaseModel, Field
from typing import Optional, ClassVar, Callable, Any
from neo.types.errors import ToolError


class BaseTool(BaseModel):
    pass


class Tool(BaseTool):
    """Wrapper for a callable function that can be used as a tool."""

    func: Optional[Callable] = Field(default=None, exclude=True)
    tool_usage_instructions: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    params: Optional[dict] = None
    provider: Optional[Any] = Field(
        default=None,
        description="The provider of the function.",
    )

    @staticmethod
    def get_name(func: Callable) -> str:
        return func.__name__

    @staticmethod
    def get_description(func: Callable) -> str:
        """Get the description of the function.
        Returns
        -------
        str
            The docstring of the function.
        """
        return func.__doc__

    @staticmethod
    def get_params(func: Callable) -> dict:
        """Get the parameters of the function as a dictionary.
        Returns
        -------
        dict
            A dictionary with the parameter names as keys and the parameter types as values.
        """
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
            type(None): "null",
        }

        try:
            signature = inspect.signature(func)
        except ValueError as e:
            name = Tool.get_name(func=func)
            raise ToolError(f"Failed to get signature for function {name}: {str(e)}")

        params = {}
        for param in signature.parameters.values():
            param_type = type_map.get(param.annotation, "string")
            params[param.name] = {"type": param_type}

        required = [
            param.name
            for param in signature.parameters.values()
            if param.default == inspect._empty
        ]

        schema = {"type": "object", "properties": params, "required": required}
        schema["additionalProperties"] = False
        return schema

    def model_post_init(self, __context: Any) -> None:
        if self.func is not None:
            if self.name is None:
                self.name = self.get_name(func=self.func)
            if self.description is None:
                self.description = self.get_description(func=self.func)
            if self.params is None:
                self.params = self.get_params(func=self.func)

    @property
    def is_coroutine(self) -> bool:
        """Check if the function is a coroutine."""
        return inspect.iscoroutinefunction(self.func)
