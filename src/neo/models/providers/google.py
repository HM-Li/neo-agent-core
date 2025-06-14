import copy
import os
from typing import Any, List, Union

import httpx
from google import genai
from google.genai import types
from pydantic import BaseModel

from neo.contexts.context import Context
from neo.contexts.thread import Thread
from neo.models.providers.base import BaseChatModel
from neo.types.contents import BooleanContent, TextContent
from neo.types.errors import ModelServiceError
from neo.types.roles import Role


class GoogleAIModel(BaseChatModel):
    """
    GoogleAIModel uses Google's Generative AI SDK.

    Note: this model is still under development and may not be fully functional.
    """

    @property
    def unsupported_params(self) -> List[str]:
        return ["tools", "mcp_clients"]

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

    def context_to_prompt(self, context: Context) -> dict:
        """
        Convert a single Context to Google GenAI Parts.
        Required by base class but not used in our thread_to_prompt override.
        """
        # Only support user and assistant roles
        if context.provider_role not in [Role.USER, Role.ASSISTANT]:
            raise ValueError(
                "Only user and assistant roles are supported by GoogleAIModel"
            )

        # Convert context contents to Google GenAI Parts
        parts = []
        for content in context.contents:
            if isinstance(content, TextContent):
                parts.append(types.Part.from_text(text=content.data))
            # TODO: Add support for other content types (images, documents, etc.)
            # elif isinstance(content, ImageContent):
            #     parts.append(types.Part.from_uri(...))

        if not parts:
            raise ValueError("No valid content found in context")

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

        # Add thinking configuration if enabled
        if self.enable_thinking:
            gen_config_args["thinking_config"] = types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=self.thinking_budget_tokens,
            )

        if instruction:  # Only add system_instruction if it's not None and not empty
            gen_config_args["system_instruction"] = instruction

        final_configs = types.GenerateContentConfig(**gen_config_args)

        return native_contents, final_configs, thread

    async def add_response_to_thread(self, thread, response):
        contents = []

        # Handle thinking content if present
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                for part in candidate.content.parts:
                    if hasattr(part, "thought") and part.thought:
                        # Log thinking content similar to Anthropic implementation
                        self.logger.thinking(part.text)
                    elif hasattr(part, "text") and part.text:
                        contents.append(TextContent(data=part.text))

        # Fallback to existing logic if no parts found
        if not contents:
            if response.parsed is not None:
                response_text = response.parsed[0]
            else:
                response_text = response.text

            if not isinstance(response_text, str):
                response_text = str(response_text)

            contents.append(TextContent(data=response_text))

        context = Context(contents=contents, provider_role=Role.ASSISTANT)
        await thread.append_context(context)

    async def acreate(
        self,
        user_input: str | Context | Thread,
        base_thread: Thread = None,
        return_response_object: bool = False,
        return_generated_thread: bool = False,
    ) -> Thread:

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
            await base_thread.extend_thread(thread=thread)
        else:
            base_thread = thread

        if return_generated_thread is True:
            return thread

        return base_thread
