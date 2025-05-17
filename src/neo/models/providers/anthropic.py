# %%
import copy
from typing import Any, Callable, List, Optional

import anthropic
from pydantic import BaseModel

from neo.contexts.context import Context
from neo.contexts.thread import Thread
from neo.models.providers.base import BaseChatModel
from neo.tools import Tool
from neo.types.contents import (
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
from neo.types.errors import ModelServiceError, ToolError
from neo.types.modalities import Modality
from neo.types.roles import Role
from neo.utils.file_handling import (
    base64_str_to_binary,
    binary_to_base64_str,
    extract_text_from_pdf,
    fetch_url_as_base64_str,
)


class AnthropicModel(BaseChatModel):
    """
    Anthropic models
    """

    PROMPT_TEMPLATE = {
        "text": {"type": "text", "text": "{data}"},
        "image_base64": {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "{mime_type}",
                "data": "{data}",
            },
        },
        "image_url": {
            "type": "image",
            "source": {
                "type": "url",
                "url": "{data}",
            },
        },
        "document_base64": {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": "{data}",
            },
        },
        "document_url": {
            "type": "document",
            "source": {
                "type": "url",
                "url": "{data}",
            },
        },
        "tool_input": {
            "type": "tool_use",
            "id": "{tool_use_id}",
            "name": "{tool_name}",
            "input": "{data}",
        },
        "tool_output": {
            "type": "tool_result",
            "tool_use_id": "{tool_use_id}",
            "content": "{data}",
            "is_error": False,
        },
    }

    def _check_params(self):
        pass

    def create_client(self):
        return anthropic.AsyncAnthropic(api_key=self.custom_api_key)

    def get_base_url(self) -> str:
        # Anthropic's client manages endpoints internally.
        return "https://api.anthropic.com/v1"

    async def context_to_prompt(
        self, context: Context | List, add_role: bool = True
    ) -> dict:
        prompt = []
        contents = context.contents if isinstance(context, Context) else context

        for c in contents:

            match c:
                case TextContent():
                    t = copy.deepcopy(self.PROMPT_TEMPLATE["text"])
                    t["text"] = c.data

                case ImageContent():
                    if isinstance(c.data, bytes):
                        t = copy.deepcopy(self.PROMPT_TEMPLATE["image_base64"])
                        t["source"]["data"] = binary_to_base64_str(c.data)
                        t["source"]["media_type"] = c.mime_type
                    elif isinstance(c.data, str):
                        t = copy.deepcopy(self.PROMPT_TEMPLATE["image_url"])
                        t["source"]["url"] = c.data

                case ToolInputContent():
                    t = copy.deepcopy(self.PROMPT_TEMPLATE["tool_input"])
                    t["input"] = c.params
                    t["id"] = c.tool_use_id
                    t["name"] = c.tool_name

                case ToolOutputContent():
                    t = copy.deepcopy(self.PROMPT_TEMPLATE["tool_output"])
                    p = await self.context_to_prompt(c.contents, add_role=False)
                    t["content"] = p
                    t["tool_use_id"] = t["tool_use_id"].format(
                        tool_use_id=c.tool_use_id
                    )
                    t["is_error"] = c.is_error

                case DocumentContent():
                    if isinstance(c.data, bytes):
                        t = copy.deepcopy(self.PROMPT_TEMPLATE["document_base64"])
                        t["source"]["data"] = binary_to_base64_str(c.data)
                        t["source"]["media_type"] = c.mime_type
                    elif isinstance(c.data, str):
                        t = copy.deepcopy(self.PROMPT_TEMPLATE["document_url"])
                        t["source"]["url"] = c.data

                case DocumentTextContent():
                    if c.text is not None:
                        text_data = c.text
                    else:
                        if isinstance(c.data, str):
                            fetched = fetch_url_as_base64_str(c.data)
                            data = fetched["data"]
                            c.mime_type = fetched["mime_type"]
                            c.file_name = fetched["file_name"]
                        else:  # bytes
                            data = binary_to_base64_str(c.data)

                        # load text
                        if c.mime_type == "application/pdf":
                            self.logger.info(
                                f"Extracting text from PDF file: {c.file_name}"
                            )
                            text_data = extract_text_from_pdf(data=data)
                        else:
                            text_data = base64_str_to_binary(data=data).decode("utf-8")
                        c.text = text_data

                    t = copy.deepcopy(self.PROMPT_TEMPLATE["text"])
                    t["text"] = text_data

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
                    t = copy.deepcopy(self.PROMPT_TEMPLATE["text"])
                    t["text"] = text_data

                case RawContent():
                    t = c.data

                case _:
                    t = copy.deepcopy(self.PROMPT_TEMPLATE["text"])
                    t["text"] = str(c)

            prompt.append(t)

        if len(prompt) == 0:
            raise ValueError("No valid content provided")

        if add_role:
            prompt = {"role": context.provider_role.value, "content": prompt}
        return prompt

    async def prepare_config(
        self, user_input: str | Context | Thread, base_thread: Thread
    ) -> dict:
        """Anthropic requires a max_token parameter in the config."""

        thread = await self.prepare_thread(user_input)
        messages = await self.thread_to_prompt(thread, base_thread=base_thread)
        configs = copy.deepcopy(self.configs)

        # Add system message
        instruction = self.get_augmented_instruction()
        if instruction is not None:
            configs["system"] = instruction

        # handle boolean response
        if self.boolean_response is True:
            self.structured_response_model = BooleanContent

        if self.structured_response_model is not None:
            bool_content_schema = self.base_model_to_json_schema(
                self.structured_response_model
            )
            tools = [bool_content_schema]
            tool_choice = {"type": "tool", "name": bool_content_schema["name"]}

            configs["tools"] = tools
            configs["tool_choice"] = tool_choice

        tools = []
        if self.tools is not None:
            for tool in self.tools:
                # Convert callable tools to JSON schema format
                tool_schema = self.register_tool(tool=tool)
                tools.append(tool_schema)

        # add mcp clients
        if self.mcp_clients is not None:
            for client in self.mcp_clients:
                client_tool_schemas = await self.bind_mcp_client(client)
                tools.extend(client_tool_schemas)

        if len(tools) > 0:
            configs["tools"] = tools
            configs["tool_choice"] = (
                {"type": "any"} if self.tool_choice == "required" else {"type": "auto"}
            )

        # anthropic requires a max_token parameter in the config.
        if "max_tokens" not in configs:
            # set default max_tokens
            configs["max_tokens"] = 2048
            self.logger.warning(
                "No max_tokens provided, using default value of 2048 since Anthropic requires `max_tokens` parameter."
            )

        return messages, configs, thread

    @staticmethod
    def base_model_to_json_schema(model: BaseModel):
        input_schema = model.model_json_schema()

        title = input_schema.pop("title")

        schema = {"name": title, "description": "", "input_schema": input_schema}
        return schema

    @staticmethod
    def tool_to_json_schema(tool: Tool | Callable) -> dict:
        """
        Convert a Tool object to a JSON schema format that can be used by Anthropic.
        """
        if callable(tool):
            # If it's a callable, wrap it in a Tool object
            tool = Tool(func=tool)

        if not isinstance(tool, Tool):
            raise ToolError(
                f"The provided tool is not a callable or a Tool instance: {tool}"
            )

        input_schema = tool.params

        # Create a schema dictionary
        schema = {
            "name": tool.name,
            "description": tool.description if tool.description else "",
            "input_schema": input_schema,
        }

        return schema

    async def add_response_to_thread(
        self, thread: Thread, response: anthropic.types.message.Message
    ) -> Thread:
        _id = response.id
        contents = []
        tool_output = None

        # important: unlike OpenAI, Anthropic's response is a single tool call or a sequence of tool call
        for item in response.content:
            if item.type == "text":
                contents.append(TextContent(data=item.text))

            elif item.type == "tool_use":
                # handle strucrured response differently
                if self.boolean_response is True:
                    self.structured_response_model = BooleanContent

                if self.structured_response_model is not None:
                    # handle structured response
                    params = item.input
                    # check params
                    self.structured_response_model(**params)
                    content = TextContent(data=str(params))
                    contents.append(content)
                else:
                    tool_input = ToolInputContent(
                        params=item.input, tool_name=item.name, tool_use_id=item.id
                    )
                    contents.append(tool_input)

                    # handle tool call
                    tool_output = await self.handle_single_tool_response(tool_input)

            else:
                raise TypeError(f"Unsupported API response content type: {item.type}")

        # handle tool output
        contexts = []
        c = Context(
            contents=contents,
            provider_role=Role.ASSISTANT,
            provider_name=self.model,
            provider_context_id=_id,
        )
        contexts.append(c)

        if tool_output is not None:
            c = Context(
                contents=tool_output,
                provider_role=Role.USER,
            )
            contexts.append(c)

        # add to thread
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
                user_input, base_thread=base_thread
            )

            self.logger.info(f"Sending Claude API Request with Configs: {configs}")

            try:
                response = await self.client.messages.create(
                    messages=messages,
                    **configs,
                )
            except Exception as e:
                raise ModelServiceError(e) from e

            self.logger.info(
                f"Model API Request Completed. Usage: {getattr(response, 'usage', {})}"
            )

            if return_response_object is True:
                return response

            await self.add_response_to_thread(thread=thread, response=response)

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


# %%
