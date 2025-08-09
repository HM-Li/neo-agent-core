import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Literal, Optional

from pydantic import BaseModel

from neo.contexts.context import Context
from neo.contexts.thread import Thread
from neo.mcp.client import MCPClient
from neo.tools import BaseTool, Tool
from neo.types.contents import (
    BaseContent,
    RawContent,
    TextContent,
    ToolInputContent,
    ToolOutputContent,
)
from neo.types.errors import (
    ModelModalityError,
    ToolError,
    ToolNotFoundError,
    ToolRuntimeError,
)
from neo.types.modalities import Modality
from neo.types.roles import Role
from neo.utils.common import get_current_utc_timestamp
from neo.utils.logger import get_logger


class BaseChatModel(ABC):
    """
    BaseChatModel encapsulates common functionality for interacting with any model API.
    It includes prompt building, configuration preparation, and response wrapping.

    Attributes
    ----------
    model : str
        The name of the model to be used.
    input_modalities : List[Modality], optional
        List of input modalities supported by the model.
    output_modalities : List[Modality], optional
        List of output modalities supported by the model.
    instruction : str, optional
        System message to be sent to the model.
    configs : dict, optional
        Model configurations, including temperature and other settings.
    custom_api_key : str, optional
        Custom API key for authentication.
    json_mode : bool, default False
        If True, the model will return a JSON response.
    boolean_response : bool, default False
        If True, the model will return a boolean response.
    structured_response_model : BaseModel, optional
        Pydantic model for structured responses.
    tools : List[Tool | Callable], optional
        List of tools to be used with the model.
    mcp_clients : List[MCPClient], optional
        List of MCP clients to be used with the model.
    tool_choice : Literal["auto", "required"], default "auto"
        If "auto", the model will automatically choose the tool to use. If "required", the tool must be specified.
    timeaware : bool, default False
        If True, the model will be time-aware and include the current UTC time in the instruction.
    enable_thinking : bool, default False
        If True, enables thinking mode for the model (currently supported by Anthropic models).
    thinking_budget_tokens : int, default 1024
        The number of tokens allocated for thinking when thinking mode is enabled.
    auto_tool_run : bool, default True
        If True, automatically executes tools when they are called by the model. If False, returns tool calls without executing them.
    """

    PROMPT_TEMPLATE = {
        "text": None,
        "image": None,
        "audio": None,
        "tool_output": None,
        "tool_input": None,
    }

    def __init__(
        self,
        model: str,
        input_modalities: Optional[List[Modality]] = None,
        output_modalities: Optional[List[Modality]] = None,
        instruction: Optional[str] = None,
        configs: Optional[dict] = None,
        custom_api_key: Optional[str] = None,
        json_mode: bool = False,
        boolean_response: bool = False,
        structured_response_model: Optional[BaseModel] = None,
        tools: Optional[List[Tool | Callable]] = None,
        mcp_clients: Optional[List[MCPClient]] = None,
        tool_choice: Literal["auto", "required"] = "auto",
        timeaware: bool = False,
        enable_thinking: bool = None,
        thinking_budget_tokens: int = None,
        tool_preamble: bool = False,
        auto_tool_run: bool = True,
    ):
        if (
            sum(
                [
                    json_mode,
                    boolean_response,
                    structured_response_model is not None,
                ]
            )
            > 1
        ):
            raise ValueError(
                "json_mode, boolean_response, and structured_response_model cannot be used together"
            )

        self.model = model
        self.instruction = instruction
        self.input_modalities = (
            [Modality(m) for m in input_modalities] if input_modalities else None
        )
        self.output_modalities = (
            [Modality(m) for m in output_modalities] if output_modalities else None
        )
        self.configs = self._init_configs(configs)
        self.custom_api_key = custom_api_key
        self.json_mode = json_mode
        self.boolean_response = boolean_response
        self.structured_response_model = structured_response_model
        self.tools = tools
        self.mcp_clients = mcp_clients
        self.tool_choice = tool_choice
        self.timeaware = timeaware
        self.enable_thinking = enable_thinking
        self.tool_preamble = tool_preamble
        self.auto_tool_run = auto_tool_run

        if self.enable_thinking is not None:
            self.thinking_budget_tokens = thinking_budget_tokens or 1024

        self._logger = None
        self.client = self.create_client()
        self.tool_registry = {}
        self.mcp_client_registry = {}

        self._check_params()

    @property
    @abstractmethod
    def unsupported_params(self) -> List[str]:
        """
        List of parameter names that are not supported by this model.
        Must be implemented by subclasses.
        """
        pass

    def _check_params(self) -> None:
        """
        Check if the parameters are supported using the unsupported_params property.
        """
        for param in self.unsupported_params:
            if hasattr(self, param) and getattr(self, param) is not None:
                raise ValueError(
                    f"{self.__class__.__name__} does not support `{param}` parameter. Please remove it."
                )

    @property
    def logger(self):
        if self._logger is None:
            self._logger = get_logger(self.__class__.__name__)
        return self._logger

    def _init_configs(self, configs):
        default = {
            "model": self.model,
            "temperature": 1,  # enforce default temperature
        }

        if configs is not None:
            default.update(configs)
        return default

    def set_configs(self, configs: dict) -> None:
        """
        Set the model configurations.
        """
        self.configs = self._init_configs(configs)

    def set_instruction(self, instruction: str) -> None:
        """
        Set the system message.
        """
        self.instruction = instruction

    @abstractmethod
    def create_client(self):
        """
        Instantiate the appropriate API client.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def get_base_url(self) -> str:
        """
        Returns the base URL for the API.
        Must be implemented by subclasses.
        """
        pass

    async def acheck_context_modality(self, context: Context) -> None:
        """
        Check if the modalities of the context match the model's input modalities.
        """
        for content in context.contents:
            if content.modality == Modality.UNDEFINED:
                self.logger.warning(
                    f"Content modality is undefined in context id {context.id}. "
                    "This may lead to unexpected behavior. "
                    "Consider setting the modality explicitly."
                )
            elif content.modality not in self.input_modalities:
                raise ModelModalityError(
                    f"Unsupported modality {content.modality} for model {self.model}."
                    "Supported modalities are: "
                    f"{self.input_modalities}"
                )

    async def acheck_thread_modalities(self, thread: Thread) -> None:
        """
        Check if the modalities of the thread match the model's input modalities.
        """
        async for context in thread:
            await self.acheck_context_modality(context)

    async def prepare_thread(
        self, user_input: Optional[str | Context | Thread | List]
    ) -> Thread:
        """
        Prepares a Thread object from the input.

        Parameters
        ----------
        user_input : Union[str, Context, Thread]
            Input can be a string (converted to Context), a Context object, or a Thread object

        Returns
        -------
        Thread
            A Thread object containing the input context(s)

        Raises
        ------
        ValueError
            If input type is not str, Context, or Thread
        """
        match user_input:
            case str() | BaseContent():
                context = Context(contents=user_input, provider_role=Role.USER)
                thread = Thread(contexts=[context])
            case Context():
                thread = Thread(contexts=[user_input])
            case Thread():
                thread = user_input
            case list():
                thread = Thread()
                for block in user_input:
                    if not isinstance(block, dict):
                        raise ValueError(
                            "When passing a list, each item must be a dictionary "
                            "representing a Context block."
                        )
                    if "role" in block:
                        role = Role(block["role"])
                    else:
                        role = Role.UNDEFINED

                    # for some providers, like OpenAI, a tool call doesn't include content block
                    content = block.get("content", [block])
                    content = RawContent(data=content)

                    context = Context(contents=content, provider_role=role)
                    await thread.append_context(context)
            case None:
                thread = Thread()
            case _:
                raise ValueError(
                    f"Expected str, Context, or Thread, got {type(user_input).__name__}"
                )

        return thread

    @abstractmethod
    def prepare_config(
        self, user_input: str | Context | Thread, base_thread: Thread
    ) -> dict:
        """
        Prepares the API request configuration by converting the input Context into a prompt
        and merging it with default and custom settings.
        This implementation assumes remote (non-local) defaults.

        Parameters
        ----------
        user_input : str | Context | Thread
            The user input to be processed.
        base_thread : Thread
            The thread object containing the contexts.


        Returns
        -------
        messages : list
            A list of messages to be sent to the model API.
        configs : dict
            A dictionary of configuration options for the model API.
        thread : Thread
            A Thread object containing the contexts.
        """

    @abstractmethod
    def add_response_to_thread(self, thread: Thread, response: Any) -> Thread:
        """Convert model API's response to a Content object and add to the Thread.

        Parameters
        ----------
        thread : Thread
            Thread object to which the response will be added
        response : Any
            model API response
        """

    @abstractmethod
    async def acreate(
        self,
        user_input: str | Context | Thread,
        base_thread: Thread = None,
        return_generated_thread: bool = False,
        return_response_object: bool = False,
    ) -> Thread:
        """
        Sends a completion request to the model API and processes the response.

        Parameters
        ----------
        user_input : str | Context | Thread
            The input to be processed. Can be a string, Context object, or Thread object.
        base_thread : Thread, optional
            An optional base thread to be used for the request.
        return_generated_thread : bool, default False
            If True, returns the generated thread. Otherwise, returns the base thread.
            Useful for agentic tasks.
        return_response_object : bool, default False
            If True, returns the response object.

        Notes
        -----
        It is important to ensure that this function is neo.Thread-safe, as one thread might be
        used by multiple models at the same time.
        """

    @abstractmethod
    def context_to_prompt(self, context: Context) -> dict:
        """
        Converts a Context object into a prompt dictionary.
        """

    async def thread_to_prompt(
        self, thread: Thread, base_thread: Thread = None
    ) -> list:
        """
        Converts a Thread object into a list of messages.
        """
        new_messages = []
        async for context in thread:
            if self.input_modalities is not None:
                await self.acheck_context_modality(context)

            if inspect.iscoroutinefunction(self.context_to_prompt):
                msg = await self.context_to_prompt(context)
            else:
                msg = self.context_to_prompt(context)

            new_messages.append(msg)

        if base_thread is not None:
            base_messages = []
            async for context in base_thread:
                if self.input_modalities is not None:
                    await self.acheck_context_modality(context)

                if inspect.iscoroutinefunction(self.context_to_prompt):
                    msg = await self.context_to_prompt(context)
                else:
                    msg = self.context_to_prompt(context)
                base_messages.append(msg)

            new_messages = base_messages + new_messages

        return new_messages

    def get_augmented_instruction(self, timestamp: str = None) -> str:
        """
        Add time-aware instruction and tool preamble instruction to the system message.
        """
        instruction = self.instruction if self.instruction else ""
        if self.timeaware is True:
            if timestamp is None:
                timestamp = get_current_utc_timestamp()

            instruction += f" | Current UTC Time: {timestamp}."
            instruction += (
                " You can use this information to provide more accurate responses. "
            )
        if self.tool_preamble is True:
            instruction += " Before you call a tool, explain why you are calling it."
        return instruction

    async def aclear_registries(self) -> None:
        """Clear the tool registry."""
        # close all mcp clients
        try:
            for client in self.mcp_client_registry.values():
                await client.aclose()
        finally:
            self.tool_registry.clear()
            self.mcp_client_registry.clear()
        self.logger.info("Cleared tool registry and closed all MCP clients.")

    def tool_to_json_schema(self, tool: Tool | Callable) -> dict:
        """
        Convert a tool to JSON schema.
        """
        raise NotImplementedError("This method should be implemented in the subclass.")

    def register_tool(self, tool: Tool | Callable):
        """Register a tool with the model."""
        # Handle internal tool classes by instantiating them
        if self.is_internal_tool(tool):
            if not isinstance(tool, BaseTool):
                # It's a class, instantiate it
                tool = tool()
            tool_schema = self.tool_to_json_schema(tool)
            self.logger.info(f"Registered tool '{tool.name}' to the registry. ")
            return tool_schema

        tool_schema = self.tool_to_json_schema(tool)

        if not isinstance(tool, Tool):
            tool = Tool(func=tool)

        if tool.name in self.tool_registry:
            raise ToolError(f"Tool with name {tool.name} already registered")

        # mcp client must be provided if the tool doesn't include a func
        if not tool.func and not isinstance(tool.provider, MCPClient):
            raise ToolError(
                "Tool must either provide a function or provide a provider that is an MCPClient instance."
            )

        self.tool_registry[tool.name] = tool

        self.logger.info(f"Registered tool '{tool.name}' to the registry. ")
        return tool_schema

    async def bind_mcp_client(self, client: MCPClient) -> List:
        """Connect and register MCP clients"""
        if not isinstance(client, MCPClient):
            raise ToolError("Only MCPClient instances can be connected.")
        if client.name in self.mcp_client_registry:
            raise ToolError(f"MCP client with name {client.name} already registered")
        await client.aconnect()
        self.mcp_client_registry[client.name] = client

        # Register all tools from the MCP client
        tool_schemas = []
        for tool in client.tools.values():
            tool_schema = self.register_tool(tool)
            tool_schemas.append(tool_schema)

        return tool_schemas

    @classmethod
    def is_internal_tool(cls, tool: BaseTool | Callable) -> bool:
        """
        Check if the tool is an internal tool provided by this model.
        """
        if callable(tool) and not hasattr(tool, "provider"):
            return False

        # If the tool is a BaseTool instance, check its provider
        if isinstance(tool, BaseTool):
            # Check if the provider of the tool is the same as the class name
            return tool.provider == cls.__name__

        # If the tool is a class (type), check if it's a subclass of BaseTool and has the right provider
        if isinstance(tool, type) and issubclass(tool, BaseTool):
            # Check if the provider of the tool class is the same as the class name
            return hasattr(tool, "provider") and tool.provider == cls.__name__

        return False

    @staticmethod
    def is_mcp_tool(tool: BaseTool | Callable) -> bool:
        """
        Check if the tool is an MCP tool.
        """
        if isinstance(tool, BaseTool):
            # Check if the provider of the tool is an MCPClient instance
            return isinstance(tool.provider, MCPClient)
        return False

    async def handle_single_tool_response(
        self, content: ToolInputContent
    ) -> ToolOutputContent:
        """Handle single tool response and return the tool result or the callable instance."""
        is_error = False

        try:
            if content.tool_name not in self.tool_registry:
                raise ToolNotFoundError(
                    "Unknown tool name: {}".format(content.tool_name)
                )

            tool = self.tool_registry[content.tool_name]

            try:
                if self.is_mcp_tool(tool):
                    out = await tool.provider.call_tool(
                        tool_name=tool.name, tool_args=content.params
                    )
                else:
                    if tool.is_coroutine:
                        out = await tool.func(**content.params)
                    else:
                        out = tool.func(**content.params)

            except ToolError as e:
                raise
            except Exception as e:
                raise ToolRuntimeError(e) from e

            if not isinstance(out, list):
                out = [out]

            _out = []
            for o in out:
                if o is not None and not isinstance(o, (BaseContent, str)):
                    # this is an error msg to developers
                    raise TypeError("Tool output is not a valid Content object.")

                if isinstance(o, str):
                    o = TextContent(data=o)

                _out.append(o)
        except ToolError as e:
            is_error = True
            _out = [
                TextContent(data="Tool Error: " + str(e))
            ]  # return the error message as a string in TextContent

        out = ToolOutputContent(
            tool_use_id=content.tool_use_id, contents=_out, is_error=is_error
        )
        return out
