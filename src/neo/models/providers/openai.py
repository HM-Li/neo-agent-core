import base64
import copy
import json
from typing import Any, Callable, List, Optional, Union

import openai
from openai import AsyncOpenAI  # Assumes an async client interface
from pydantic import BaseModel

from neo.contexts import Thread
from neo.contexts.context import Context
from neo.mcp.client import MCPClient
from neo.models.providers.base import BaseChatModel
from neo.tools import BaseTool, Tool
from neo.types.contents import (
    AudioContent,
    AudioTextContent,
    BooleanContent,
    DocumentContent,
    DocumentTextContent,
    ImageContent,
    RawContent,
    TextContent,
    ToolInputContent,
    ToolOutputContent,
)
from neo.types.errors import ContextLengthExceededError, ModelServiceError, ToolError
from neo.types.roles import Role
from neo.utils.file_handling import (
    base64_str_to_binary,
    binary_to_base64_str,
    extract_text_from_pdf,
    fetch_url_as_base64_str,
    reformat_audio_bytes,
)


class OpenAICompleteModel(BaseChatModel):
    """
    OpenaiModel encapsulates OpenAI-specific adjustments.
    """

    PROMPT_TEMPLATE = {
        "text": {"type": "text", "text": "{data}"},
        "image": {
            "type": "image_url",
            "image_url": {"url": "data:{mime_type};base64,{data}"},
        },
        "audio": {
            "type": "input_audio",
            "input_audio": {"data": "{data}", "format": "{format}"},
        },
        "tool_input": {
            "type": "function",
            "id": "{call_id}",
            "function": {
                "name": "{tool_name}",
                "arguments": "{input}",
            },
        },
    }

    @property
    def unsupported_params(self) -> List[str]:
        return []

    def context_to_prompt(self, context, add_role: bool = True):
        """convert context to a user prompt message following the default api template"""
        prompt = []
        tool_inputs = []
        extra_message_body = {}

        for c in context.contents:
            is_tool_input = False
            match c:
                case TextContent():
                    t = copy.deepcopy(self.PROMPT_TEMPLATE["text"])
                    t["text"] = c.data

                case ImageContent():
                    t = copy.deepcopy(self.PROMPT_TEMPLATE["image"])
                    if isinstance(c.data, bytes):
                        url = t["image_url"]["url"]
                        data = binary_to_base64_str(c.data)
                        t["image_url"]["url"] = url.format(
                            data=data, mime_type=c.mime_type
                        )
                    else:
                        # url
                        t["image_url"]["url"] = c.data

                case AudioContent():
                    t = copy.deepcopy(self.PROMPT_TEMPLATE["audio"])
                    if isinstance(c.data, bytes):
                        data = c.data
                    else:
                        fetched = fetch_url_as_base64_str(c.data)
                        data = base64_str_to_binary(fetched["data"])
                        c.mime_type = fetched["mime_type"]

                    # OpenAI only supports wav and mp3 formats
                    supported_mime = ["audio/wav", "audio/mp3"]
                    mime = c.mime_type
                    if mime not in supported_mime:
                        reformatted = reformat_audio_bytes(
                            audio_data=data,
                            mime_type=mime,
                            target_format="wav",
                        )
                        data = reformatted["data"]
                        mime = reformatted["mime_type"]

                    mime = mime.split("/")[1]

                    # binary to base64
                    data = binary_to_base64_str(data)

                    t["input_audio"]["data"] = data
                    t["input_audio"]["format"] = mime

                case ToolInputContent():
                    t = copy.deepcopy(self.PROMPT_TEMPLATE["tool_input"])
                    t["function"]["name"] = c.tool_name
                    t["function"]["arguments"] = json.dumps(c.params)
                    t["id"] = c.tool_use_id
                    is_tool_input = True  # OpenAI completion API treats tool inputs differently from other content types

                case ToolOutputContent():
                    extra_message_body["tool_call_id"] = c.tool_use_id

                    # OpenAI only supports one string output per tool call
                    if len(c.contents) != 1:
                        raise ValueError(
                            "OpenAI only supports one output per tool call for the output."
                        )

                    output = c.contents[0]
                    if output is not None:
                        output = output.data

                    # OpenAI expects the output to be a string
                    t = str(output)

                case _:
                    t = copy.deepcopy(self.PROMPT_TEMPLATE["text"])
                    t["text"] = str(c)

            if is_tool_input:
                tool_inputs.append(t)
            else:
                prompt.append(t)

        if len(prompt) == 1 and isinstance(prompt[0], str):
            # if only one content, return it directly
            prompt = prompt[0]

        # post processing
        if add_role or tool_inputs or extra_message_body:
            prompt = {"role": context.provider_role.value, "content": prompt}

            if tool_inputs:
                prompt["tool_calls"] = tool_inputs

            if extra_message_body:
                prompt.update(extra_message_body)
        return prompt

    def create_client(self):
        base_url = self.get_base_url()
        return AsyncOpenAI(base_url=base_url, api_key=self.custom_api_key)

    def get_base_url(self) -> str:
        return "https://api.openai.com/v1"

    @classmethod
    def tool_to_json_schema(cls, tool: BaseTool | Callable) -> dict:
        """Convert a tool to a json schema"""
        if callable(tool):
            tool = Tool(func=tool)

        if not isinstance(tool, BaseTool):
            raise ToolError(
                f"The provided tool is not a callable or a BaseTool instance: {tool}"
            )

        if cls.is_internal_tool(tool):
            tool_schema = tool.model_dump(exclude_none=True)
        else:
            tool_schema = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.params,
                    "strict": True,
                },
            }

            # OpenAI requires 'additionalProperties': False
            tool_schema["function"]["parameters"]["additionalProperties"] = False

        return tool_schema

    async def prepare_config(
        self, user_input: str | Context | Thread, base_thread: Thread
    ) -> tuple:

        thread = await self.prepare_thread(user_input)
        config = copy.deepcopy(self.configs)

        # add system message atomically
        if base_thread is None:
            base_thread = Thread()

        temp_thread = await base_thread.afork()
        instruction = self.get_augmented_instruction()

        if instruction is not None:
            system_msg = Context(contents=instruction, provider_role=Role.DEVELOPER)
            await temp_thread.add_context_to_beginning(system_msg)

        messages = await self.thread_to_prompt(thread=thread, base_thread=temp_thread)

        if self.json_mode:
            config["response_format"] = {"type": "json_object"}
        elif self.boolean_response:
            config["response_format"] = BooleanContent
        elif self.structured_response_model:
            config["response_format"] = self.structured_response_model

        # OpenAI expects "max_completion_tokens" instead of "max_tokens".
        if "max_tokens" in config:
            config["max_completion_tokens"] = config.pop("max_tokens")

        # add tools
        tools = []
        if self.tools is not None:
            for tool in self.tools:
                tool_schema = self.register_tool(tool)
                tools.append(tool_schema)

        # add mcp clients
        if self.mcp_clients is not None:
            for client in self.mcp_clients:
                client_tool_schemas = await self.bind_mcp_client(client)
                tools.extend(client_tool_schemas)

        if len(tools) > 0:
            config["tools"] = tools
            config["tool_choice"] = self.tool_choice

        return messages, config, thread

    async def add_response_to_thread(self, thread: Thread, response: Any) -> Thread:

        msg = response.choices[0].message
        if getattr(msg, "refusal", None):
            raise ModelServiceError(msg.refusal)

        contexts = []
        # Handle text content
        if msg.content:
            if getattr(msg, "parsed", None):
                result_text = msg.parsed
            else:
                result_text = msg.content

            if not isinstance(result_text, str):
                result_text = str(result_text)

            output_context = Context(
                contents=TextContent(data=result_text),
                provider_role=Role.ASSISTANT,
                provider_name=self.model,
                provider_context_id=response.id,
            )
            contexts.append(output_context)

        # Handle tool calls first
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            # For openai completion API, tool calls are considered as one context/ message and tool outputs are separate contexts.
            tool_call_context = Context(
                contents=[],
                provider_role=Role.ASSISTANT,
                provider_name=self.model,
                provider_context_id=response.id,
            )
            tool_output_contexts = []

            for tool_call in msg.tool_calls:
                # Create tool input context
                tool_input_content = ToolInputContent(
                    tool_name=tool_call.function.name,
                    tool_use_id=tool_call.id,
                    params=json.loads(tool_call.function.arguments),
                )
                tool_call_context.contents.append(tool_input_content)

                # Execute tool and create tool output context if auto_tool_run is enabled
                if self.auto_tool_run:
                    tool_output_content = await self.handle_single_tool_response(
                        content=tool_input_content
                    )
                    tool_output_contexts.append(
                        Context(
                            contents=tool_output_content,
                            provider_role=Role.TOOL,
                            provider_name=None,
                            provider_context_id=response.id,
                        )
                    )
            # Add tool call context and tool output contexts to the main contexts
            contexts.append(tool_call_context)
            contexts.extend(tool_output_contexts)

        # add response to thread
        if contexts:
            await thread.append_contexts(contexts)

    async def acreate(
        self,
        user_input: str | Context | Thread,
        base_thread: Thread = None,
        return_response_object: bool = False,
        return_generated_thread: bool = False,
    ) -> Thread:
        try:
            messages, config, thread = await self.prepare_config(
                user_input=user_input, base_thread=base_thread
            )

            self.logger.info(
                f"Sending Model API Request to ({self.get_base_url()}) with Configs: {config}"
            )

            # Use parse API if structured/boolean response is requested.
            parse_api = (
                self.boolean_response or self.structured_response_model is not None
            )

            try:
                if not parse_api:
                    response = await self.client.chat.completions.create(
                        messages=messages, **config
                    )
                else:
                    response = await self.client.beta.chat.completions.parse(
                        messages=messages, **config
                    )
            except openai.BadRequestError as e:
                err_msg = str(e)
                if (
                    "context_length_exceeded" in err_msg
                    or "string_above_max_length" in err_msg
                ):
                    raise ContextLengthExceededError(e) from e

                raise ModelServiceError(e) from e
            except Exception as e:
                raise ModelServiceError(e) from e

            self.logger.info(
                f"Model API Request Completed. Usage: {getattr(response, 'usage', {})}"
            )

            if return_response_object:
                return response

            # add response to thread
            await self.add_response_to_thread(thread, response)

            # extend the base thread
            if base_thread is not None:
                await base_thread.extend_thread(thread=thread)
            else:
                base_thread = thread

            if return_generated_thread is True:
                return thread

            return base_thread
        finally:
            # clear tool registry
            await self.aclear_registries()


class OpenAIResponseModel(BaseChatModel):
    """
    OpenaiModel encapsulates OpenAI-specific adjustments.
    Updated to use the new Response API.
    """

    PROMPT_TEMPLATE = {
        "text": {"type": "input_text", "text": "{data}"},
        "image": {
            "type": "input_image",
            "image_url": "data:{mime_type};base64,{data}",
        },
        "audio": {
            "type": "input_audio",
            "input_audio": {"data": "{data}", "format": "{format}"},
        },
        "document": {
            "type": "input_file",
            "filename": "{filename}",
            "file_data": "data:application/pdf;base64,{data}",
        },
        "tool_input": {
            "type": "function_call",
            "name": "{tool_name}",
            "call_id": "{tool_use_id}",
            "arguments": "{input}",
        },
        "tool_output": {
            "type": "function_call_output",
            "call_id": "{call_id}",
            "output": "{output}",
        },
    }

    @property
    def unsupported_params(self) -> List[str]:
        # OpenAI Response API doesn't support thinking yet
        return ["enable_thinking", "thinking_budget_tokens"]

    async def context_to_prompt(self, context, add_role: bool = True):
        """convert context to a user prompt message following the default api template"""
        prompt = []

        def create_text_block(text_data):
            t = copy.deepcopy(self.PROMPT_TEMPLATE["text"])
            t["text"] = text_data
            # input_text vs output_text
            if context.provider_role == Role.ASSISTANT:
                t["type"] = "output_text"
            else:
                t["type"] = "input_text"
            return t

        for c in context.contents:
            match c:
                case TextContent():
                    t = create_text_block(c.data)

                case AudioContent():
                    t = copy.deepcopy(self.PROMPT_TEMPLATE["audio"])
                    t["input_audio"]["data"] = c.data
                    t["input_audio"]["format"] = c.mime_type

                case ImageContent():
                    t = copy.deepcopy(self.PROMPT_TEMPLATE["image"])
                    if isinstance(c.data, bytes):
                        data = binary_to_base64_str(c.data)
                        t["image_url"] = t["image_url"].format(
                            data=data, mime_type=c.mime_type
                        )
                    else:
                        # url
                        t["image_url"] = c.data

                case DocumentContent():

                    if isinstance(c.data, bytes):
                        data = binary_to_base64_str(c.data)
                    else:
                        fetched = fetch_url_as_base64_str(c.data)
                        data = fetched["data"]
                        c.file_name = fetched["file_name"]
                        c.mime_type = fetched["mime_type"]

                    t = copy.deepcopy(self.PROMPT_TEMPLATE["document"])
                    t["file_data"] = t["file_data"].format(data=data)
                    t["filename"] = c.file_name

                case DocumentTextContent():
                    if c.text is not None:
                        text_data = c.text
                    else:

                        if isinstance(c.data, bytes):
                            data = binary_to_base64_str(c.data)
                        else:
                            fetched = fetch_url_as_base64_str(c.data)
                            data = fetched["data"]
                            c.file_name = fetched["file_name"]
                            c.mime_type = fetched["mime_type"]

                        if c.mime_type == "application/pdf":
                            self.logger.info(
                                f"Extracting text from PDF file: {c.file_name}"
                            )
                            text_data = extract_text_from_pdf(data)
                        else:
                            text_data = base64_str_to_binary(data).decode(
                                "utf-8"
                            )  # text
                        c.text = text_data

                    t = create_text_block(text_data)

                case AudioTextContent():
                    if c.text is not None:
                        text_data = c.text
                    else:
                        self.logger.info(f"Transcribing audio file: {c.file_name}")
                        if c.transcription_handler.is_coroutine:
                            text_data = await c.transcription_handler.func(c)
                        else:
                            text_data = c.transcription_handler.func(c)

                        if not isinstance(text_data, TextContent):
                            raise ValueError(
                                "Transcription handler must return TextContent"
                            )
                        text_data = text_data.data
                        c.text = text_data
                    t = create_text_block(text_data)

                case ToolInputContent():
                    t = copy.deepcopy(self.PROMPT_TEMPLATE["tool_input"])
                    t["call_id"] = c.tool_use_id
                    t["name"] = c.tool_name
                    t["arguments"] = json.dumps(c.params)

                    # no role for tool call
                    add_role = False

                case ToolOutputContent():
                    t = copy.deepcopy(self.PROMPT_TEMPLATE["tool_output"])
                    t["call_id"] = c.tool_use_id

                    # openai only support one string output per tool call
                    if len(c.contents) != 1:
                        raise ValueError(
                            "OpenAI only supports one output per tool call for the output."
                        )

                    output = c.contents[0]

                    if output is not None:
                        output = output.data

                    # OpenAI expects the output to be a string
                    t["output"] = str(output)

                    # no role for tool call
                    add_role = False

                case RawContent():
                    t = c.data

                case _:
                    t = copy.deepcopy(self.PROMPT_TEMPLATE["text"])
                    t["text"] = str(c)

                    # input_text vs output_text
                    if context.provider_role == Role.ASSISTANT:
                        t["type"] = "output_text"
                    else:
                        t["type"] = "input_text"
            prompt.append(t)

        # post processing
        if add_role and context.provider_role != Role.UNDEFINED:
            prompt = {"role": context.provider_role.value, "content": prompt}
        else:
            # if return the content directly, say for tool call, the content should be a single object
            if len(prompt) == 1:
                prompt = prompt[0]

        return prompt

    def create_client(self):
        base_url = self.get_base_url()
        return AsyncOpenAI(base_url=base_url, api_key=self.custom_api_key)

    def get_base_url(self) -> str:
        return "https://api.openai.com/v1"

    @staticmethod
    def base_model_to_json_schema(model: BaseModel) -> dict:
        """Convert a base model to text format schema

        Parameters
        ----------
        model : BaseModel
            Pydantic basemodel

        Returns
        -------
        str
            dict
        """
        schema = model.model_json_schema()

        title = schema.pop("title")
        # required
        schema["additionalProperties"] = False

        formatted_schema = {
            "type": "json_schema",
            "name": title,
            "schema": schema,
            "strict": True,
        }
        return formatted_schema

    @classmethod
    def tool_to_json_schema(cls, tool: BaseTool | Callable) -> dict:
        """Convert a tool to a json schema"""
        if callable(tool):
            tool = Tool(func=tool)

        if not isinstance(tool, BaseTool):
            raise ToolError(
                f"The provided tool is not a callable or a BaseTool instance: {tool}"
            )

        if cls.is_internal_tool(tool):
            # For internal tools like WebSearch, return the tool's direct schema
            # This will be something like {"type": "web_search_preview", "user_location": {...}, ...}
            tool_schema = tool.model_dump(exclude_none=True)
        else:
            # For regular function tools, wrap in OpenAI function schema
            tool_schema = {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.params,
                "strict": True,
            }

            # openai requires 'additionalProperties': False
            tool_schema["parameters"]["additionalProperties"] = False

        return tool_schema

    async def prepare_config(
        self, user_input: str | Context | Thread, base_thread: Thread = None
    ) -> tuple:

        thread = await self.prepare_thread(user_input)

        # Convert the thread to a list of messages.
        messages = await self.thread_to_prompt(thread, base_thread=base_thread)

        if len(messages) == 0:
            raise ValueError("No valid content provided")

        configs = copy.deepcopy(self.configs)

        format_configs = {}
        if self.json_mode:
            format_configs = {"type": "json_object"}
        elif self.boolean_response:
            format_configs = self.base_model_to_json_schema(BooleanContent)
        elif self.structured_response_model:
            format_configs = self.base_model_to_json_schema(
                self.structured_response_model
            )
            
        if format_configs:
            text_config = configs.get("text", {}) or {}
            text_config["format"] = format_configs  
            configs["text"] = text_config

        # OpenAI expects "max_output_tokens" instead of "max_tokens" for the response API.
        if "max_tokens" in configs:
            configs["max_output_tokens"] = configs.pop("max_tokens")

        # add system message
        configs["instructions"] = self.get_augmented_instruction()

        # add tools
        tools = []
        if self.tools is not None:
            for tool in self.tools:
                tool_schema = self.register_tool(tool)
                tools.append(tool_schema)

        # add mcp clients
        if self.mcp_clients is not None:
            for client in self.mcp_clients:
                client_tool_schemas = await self.bind_mcp_client(client)
                tools.extend(client_tool_schemas)

        if len(tools) > 0:
            configs["tools"] = tools
            configs["tool_choice"] = self.tool_choice

        return messages, configs, thread

    async def add_response_to_thread(
        self,
        thread: Thread,
        response: Any,
    ) -> Thread:
        # one response might contain multiple tool calls or messages
        contexts = []
        for item in response.output:
            if item.type == "message":
                contents = []
                # one message might contain multiple content types
                for c in item.content:
                    if c.type == "output_text":
                        result_text = c.text

                        if self.boolean_response is True:
                            self.structured_response_model = BooleanContent

                        if self.structured_response_model is not None:
                            # openai returns a json string
                            _params = json.loads(result_text)

                            # check params
                            self.structured_response_model(**_params)

                        result_content = TextContent(data=str(result_text))

                        contents.append(result_content)
                    else:
                        raise ValueError(f"Unknown content type: {c.type}")

                # create context for the assistant
                output_context = Context(
                    contents=contents,
                    provider_role=Role.ASSISTANT,
                    provider_name=self.model,
                    provider_context_id=item.id,
                )
                contexts.append(output_context)

            elif item.type == "function_call":
                _params = json.loads(item.arguments)

                tool_input_content = ToolInputContent(
                    tool_name=item.name, tool_use_id=item.call_id, params=_params
                )
                contexts.append(
                    Context(
                        contents=tool_input_content,
                        provider_role=Role.ASSISTANT,
                        provider_name=self.model,
                        provider_context_id=item.id,
                    )
                )
                # handle tool input if auto_tool_run is enabled
                if self.auto_tool_run:
                    tool_output_content = await self.handle_single_tool_response(
                        content=tool_input_content
                    )

                    contexts.append(
                        Context(
                            contents=tool_output_content,
                            provider_role=Role.USER,
                            provider_name=None,
                            provider_context_id=item.id,
                        )
                    )
            elif item.type == "reasoning":
                # Handle reasoning (thinking) blocks from OpenAI
                for summary in item.summary:
                    self.logger.thinking(summary.text)
        # add response to thread
        await thread.append_contexts(contexts)

    async def acreate(
        self,
        user_input: str | Context | Thread,
        base_thread: Thread = None,
        return_response_object: bool = False,
        return_generated_thread: bool = False,
    ) -> Thread:
        try:
            messages, configs, thread = await self.prepare_config(
                user_input=user_input, base_thread=base_thread
            )

            self.logger.info(
                f"Sending Model API Request to ({self.get_base_url()}) with Configs: {configs}"
            )

            try:
                response = await self.client.responses.create(input=messages, **configs)
            except openai.BadRequestError as e:
                err_msg = str(e)
                if (
                    "context_length_exceeded" in err_msg
                    or "string_above_max_length" in err_msg
                ):
                    raise ContextLengthExceededError(e) from e

                raise ModelServiceError(e) from e
            except Exception as e:
                raise ModelServiceError(e) from e

            self.logger.info(
                f"Model API Request Completed. Usage: {getattr(response, 'usage', {})}"
            )

            if return_response_object:
                return response

            # check for errors
            if response.error is not None:
                raise ModelServiceError(response.error)

            if (
                response.status == "incomplete"
                and response.incomplete_details.reason == "max_output_tokens"
            ):
                # could be partial response
                if response.output_text is None:
                    raise ContextLengthExceededError(response.incomplete_details.reason)

            ## add response to thread
            await self.add_response_to_thread(thread, response)

            ## extend the base thread
            if base_thread is not None:
                await base_thread.extend_thread(thread=thread)
            else:
                base_thread = thread

            if return_generated_thread is True:
                return thread

            return base_thread
        finally:
            # clear tool registry
            await self.aclear_registries()
