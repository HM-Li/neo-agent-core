import copy
import os
import json
from typing import Any, List, Union, Callable

import httpx
from google import genai
from google.genai import types
from pydantic import BaseModel

from neo.contexts.context import Context
from neo.contexts.thread import Thread
from neo.models.base import BaseChatModel
from neo.tools import BaseTool, Tool
from neo.types.contents import (
    BooleanContent,
    TextContent,
    ThoughtContent,
    ToolInputContent,
    ToolOutputContent,
    BaseContent,
    StructuredContent,
)
from neo.types.errors import ModelServiceError, ModelConfigError, ToolError
from neo.types.roles import Role


class GoogleAIModel(BaseChatModel):
    """
    GoogleAIModel uses Google's Generative AI SDK.

    Note: this model is still under development and may not be fully functional.
    """

    @property
    def unsupported_params(self) -> List[str]:
        return []

    def create_client(self):
        # by default the client prioritize GOOGLE_API_KEY instead of GEMINI_API_KEY
        # since neo requires GOOGLE_API_KEY for other services, here we use GEMINI_API_KEY by default
        api_key = None
        if self.custom_api_key is not None:
            api_key = self.custom_api_key
        else:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Neither GEMINI_API_KEY nor GOOGLE_API_KEY is located in the environment."
            )

        client = genai.Client(api_key=api_key)
        return client

    def get_base_url(self) -> str:
        raise NotImplementedError("GoogleAIModel does not have a base URL")

    @classmethod
    def tool_to_json_schema(cls, tool: BaseTool | Callable) -> dict:
        """Convert a tool to Google GenAI function declaration format."""
        if callable(tool):
            tool = Tool(func=tool)

        if not isinstance(tool, BaseTool):
            raise ToolError(
                f"The provided tool is not a callable or a BaseTool instance: {tool}"
            )

        if cls.is_internal_tool(tool):
            tool_schema = tool.model_dump(exclude_none=True)
        else:
            # Google doesn't accept additionalProperties in parameters
            params = copy.deepcopy(tool.params)
            params.pop("additionalProperties", None)

            tool_schema = {
                "name": tool.name,
                "description": tool.description if tool.description else "",
                "parameters": params,
            }

        return tool_schema

    def context_to_prompt(self, context: Context | List[BaseContent], add_role: bool = True) -> dict:
        """
        Convert a single Context to Google GenAI Parts.
        Required by base class but not used in our thread_to_prompt override.
        """
        is_provider_context = False
        
        if isinstance(context, Context):
            # Detect if the same provider's context
            is_provider_context = context.provider_name == __class__.__name__
            contents = context.contents
        else:
            contents = context

        # Convert context contents to Google GenAI Parts
        parts = []
        for content in contents:
            match content:
                case TextContent():
                    parts.append(types.Part.from_text(text=content.data))

                case ToolInputContent():
                    # treat as raw_data if same provider context
                    if is_provider_context:
                        parts.append(content.raw_data)
                    else:
                        parts.append(types.Part.from_text(text=f"<Tool Input>: {str(content.raw_data)}"))

                case ToolOutputContent():
                    # treat as raw_data if same provider context
                    if is_provider_context:
                        parts.append(content.raw_data)
                    else:
                        parts.append(types.Part.from_text(text=f"<Tool Output>: {str(content.raw_data)}"))

                case ThoughtContent():
                    # treat as raw_data if same provider context
                    if is_provider_context:
                        parts.append(content.raw_data)
                    else:
                        parts.append(types.Part.from_text(text=f"<Thought Content>: {str(content.raw_data)}"))

                # TODO: Add support for other content types (images, documents, etc.)
                # case ImageContent():
                #     parts.append(types.Part.from_uri(...))

                case _:
                    _str = str(content) if not hasattr(content, "data") else str(content.data)
                    parts.append(types.Part.from_text(text=_str))

        if not parts:
            raise ValueError("No valid content found in context")

        if not add_role:
            return {"parts": parts}
        return {"parts": parts, "role": context.provider_role}

    async def thread_to_prompt(
        self, thread: Thread, base_thread: Thread = None
    ) -> list[types.Content]:
        """
        Convert Neo threads to Google GenAI Content objects.
        Supports consecutive messages from the same role by merging them.
        """
        # Combine base_thread and thread
        all_contexts = []
        if base_thread is not None:
            async for context in base_thread:
                all_contexts.append(context)
        async for context in thread:
            all_contexts.append(context)

        if not all_contexts:
            return []

        # Group consecutive messages by role
        native_contents = []
        current_role = None
        current_parts = []

        for context in all_contexts:
            if self.input_modalities is not None:
                await self.acheck_context_modality(context)

            role = context.provider_role

            # If role changes, finalize current content and start new one
            if current_role is not None and role != current_role:
                content = self._create_content(current_parts, current_role)
                if content:
                    native_contents.append(content)
                current_parts = []

            # Convert context to parts
            context_data = self.context_to_prompt(context)
            current_parts.extend(context_data["parts"])
            current_role = role

        # Finalize the last content
        if current_parts and current_role is not None:
            content = self._create_content(current_parts, current_role)
            if content:
                native_contents.append(content)

        return native_contents

    def _create_content(self, parts: list[types.Part], role: Role) -> types.Content:
        """Create native Google GenAI Content from parts and role."""
        if not parts:
            return None

        if role == Role.USER:
            return types.UserContent(parts=parts)
        elif role == Role.ASSISTANT:
            return types.ModelContent(parts=parts)
        else:
            raise ValueError(f"Unsupported role: {role}")

    async def prepare_config(
        self, user_input: str | Context | Thread, base_thread: Thread
    ) -> dict:
        # prepare a thread from user_input
        thread = await self.prepare_thread(user_input)
        # convert thread to a list of Content objects
        native_contents = await self.thread_to_prompt(thread, base_thread=base_thread)

        if len(native_contents) == 0:
            raise ValueError("No valid content provided")

        # response format
        json_mode = (
            self.json_mode
            or self.boolean_response
            or self.structured_response_model is not None
        )

        response_mime_type = None
        if json_mode is True:
            response_mime_type = "application/json"

        response_schema = None
        if self.boolean_response is True:
            response_schema = list[BooleanContent]
        elif self.structured_response_model is not None:
            response_schema = list[self.structured_response_model]

        configs_copy = copy.deepcopy(self.configs)

        # model will be set by the client
        configs_copy.pop("model", None)

        instruction = self.get_augmented_instruction()

        # Handle timeout configuration
        http_options = None
        timeout = configs_copy.pop("timeout", None)
        if timeout is not None:
            timeout_ms = None
            if isinstance(timeout, (int, float)):
                # Convert seconds to milliseconds
                timeout_ms = int(timeout * 1000)
            elif isinstance(timeout, httpx.Timeout):
                # Handle httpx.Timeout object - use read timeout as primary
                if timeout.read is None:
                    # httpx.Timeout(None) means infinite wait - set no timeout
                    timeout_ms = 0  # 0 means no timeout in Google's API
                else:
                    timeout_ms = int(timeout.read * 1000)
            
            # Create HttpOptions with the timeout value
            if timeout_ms is not None:
                http_options = types.HttpOptions(timeout=timeout_ms)

        gen_config_args = {
            "temperature": configs_copy.pop("temperature", None),
            "max_output_tokens": configs_copy.pop("max_tokens", None),
            "response_mime_type": response_mime_type,
            "response_schema": response_schema,
            **configs_copy,  # Add remaining configs
        }

        if http_options is not None:
            gen_config_args["http_options"] = http_options

        if instruction:  # Only add system_instruction if it's not None and not empty
            gen_config_args["system_instruction"] = instruction

        # Add tools configuration
        function_declarations = []
        if self.tools is not None:
            for tool in self.tools:
                tool_schema = self.register_tool(tool)
                function_declarations.append(tool_schema)

        # Add mcp clients
        if self.mcp_clients is not None:
            for client in self.mcp_clients:
                client_tool_schemas = await self.bind_mcp_client(client)
                function_declarations.extend(client_tool_schemas)

        if len(function_declarations) > 0:
            tools = types.Tool(function_declarations=function_declarations)
            gen_config_args["tools"] = [tools]

        final_configs = types.GenerateContentConfig(**gen_config_args)

        return native_contents, final_configs, thread

    async def add_response_to_thread(self, thread, response):
        contexts = []

        # Handle response parts
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and hasattr(candidate.content, "parts") and candidate.content.parts:
                assistant_contents = []
                tool_output_contexts = []

                for part in candidate.content.parts:
                    if hasattr(part, "thought") and part.thought:
                        # Log and store thinking content
                        self.logger.thinking(part.text)
                        assistant_contents.append(ThoughtContent(raw_data=part))

                    elif hasattr(part, "function_call") and part.function_call:
                        # Handle function call
                        tool_input = ToolInputContent(raw_data=part)
                        assistant_contents.append(tool_input)

                        # Execute tool if auto_tool_run is enabled
                        if self.auto_tool_run:
                            # Extract function call details
                            func_call = part.function_call
                            params = dict(func_call.args)

                            output, is_error = await self.handle_single_tool_response(
                                tool_name=func_call.name, params=params
                            )

                            # Package tool output as Google expects
                            context_data = self.context_to_prompt(output, add_role=False)
                            function_response_part = types.Part.from_function_response(
                                name=func_call.name,
                                response={"result": context_data["parts"]},
                            )

                            tool_output = ToolOutputContent(raw_data=function_response_part)
                            tool_output_contexts.append(
                                Context(
                                    contents=tool_output,
                                    provider_role=Role.USER,
                                    provider_name=__class__.__name__,
                                )
                            )

                    elif hasattr(part, "text") and part.text:
                        text = part.text.strip()
                        
                        if self.boolean_response is True:
                            # gemini returns a list of json objects
                            content = BooleanContent(**json.loads(text)[0])
                            assistant_contents.append(content)
                        elif self.structured_response_model is not None:
                            params = json.loads(text)[0]
                            content = self.structured_response_model(**params)
                            content = StructuredContent(data=content)
                            assistant_contents.append(content)
                        else:
                            assistant_contents.append(TextContent(data=text))

                    # log thought signature if available (this is for gemini-3-pro and beyond)
                    if hasattr(part, "thought_signature") and part.thought_signature:
                        self.logger.thinking(
                            f"Encrypted thought signature found..."
                        )

                # Create assistant context
                if assistant_contents:
                    contexts.append(
                        Context(
                            contents=assistant_contents,
                            provider_role=Role.ASSISTANT,
                            provider_name=__class__.__name__,
                        )
                    )

                # Add tool output contexts
                contexts.extend(tool_output_contexts)

        # Fallback to existing logic if no parts found
        if not contexts:
            if response.parsed is not None:
                response_text = response.parsed[0]
            else:
                response_text = response.text

            if not isinstance(response_text, str):
                response_text = str(response_text)

            contexts.append(
                Context(
                    contents=TextContent(data=response_text),
                    provider_role=Role.ASSISTANT,
                    provider_name=__class__.__name__,
                )
            )
        
        # add raw response to contexts
        for c in contexts:
            c.raw_response = response

        # Add all contexts to thread
        await thread.extend(contexts)

    async def acreate(
        self,
        user_input: str | Context | Thread,
        base_thread: Thread = None,
        return_response_object: bool = False,
        return_generated_thread: bool = False,
    ) -> Thread:
        try:
            native_contents, configs, thread = await self.prepare_config(
                user_input, base_thread=base_thread
            )
            self.logger.info(f"Sending Google API Request with Configs: {configs}")

            try:
                response = await self.client.aio.models.generate_content(
                    model=self.model,
                    contents=native_contents,
                    config=configs,
                )
            except Exception as e:
                raise ModelServiceError(e) from e

            self.logger.info(
                f"Model API Request Completed. Usage: {getattr(response, 'usage_metadata', {})}"
            )

            if return_response_object is True:
                return response

            await self.add_response_to_thread(thread, response)

            if base_thread is not None:
                await base_thread.extend(thread=thread)
            else:
                base_thread = thread

            if return_generated_thread is True:
                return thread

            return base_thread
        finally:
            await self.aclear_registries()
