import copy
import os
from typing import Any, List, Union

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
        # only support user and assistant roles
        if context.provider_role not in [Role.USER, Role.ASSISTANT]:
            raise ValueError(
                "Only user and assistant roles are supported by GoogleAIModel"
            )

        # currently only text is supported
        if context.require_multimodal:
            raise ValueError("Multimodal context is not supported by GoogleAIModel")

        data = [c.data for c in context.contents if isinstance(c, TextContent)]

        if len(data) == 0:
            raise ValueError("No text content provided")

        if len(data) > 1:
            raise ValueError("Only one text content is supported")

        return data[0]

    async def thread_to_prompt(
        self, thread: Thread, base_thread: Thread
    ) -> list:  # google api requires a list of messages with alternate roles
        messages = []

        previous_role = None
        # must start with a user message
        if len(thread) > 0:
            first_context = await thread.aget_context(0)
            if first_context.provider_role != Role.USER:
                raise ValueError("The first context must be from the user")

            async for context in thread:
                if context.provider_role == previous_role:
                    raise ValueError("Consecutive contexts cannot have the same role")

                if self.input_modalities is not None:
                    await self.acheck_context_modality(context)

                msg = self.context_to_prompt(context)
                messages.append(msg)
                previous_role = context.provider_role

        if base_thread is not None and len(base_thread) > 0:
            # the last context of the base thread must be from the assistant
            last_context = await base_thread.aget_context(-1)
            if last_context.provider_role != Role.ASSISTANT:
                raise ValueError(
                    "The last context of the base thread must be from the assistant"
                )

            # convert base thread to a list of messages
            base_messages = await self.thread_to_prompt(base_thread, None)

            # append the last message of the base thread to the current thread
            messages = base_messages + messages
        return messages

    async def prepare_config(
        self, user_input: str | Context | Thread, base_thread: Thread
    ) -> dict:
        # prepare a thread from user_input
        thread = await self.prepare_thread(user_input)
        # convert thread to a list of messages
        messages = await self.thread_to_prompt(thread, base_thread=base_thread)

        if len(messages) == 0:
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

        gen_config_args = {
            "temperature": configs_copy.pop("temperature", None),
            "max_output_tokens": configs_copy.pop("max_tokens", None),
            "response_mime_type": response_mime_type,
            "response_schema": response_schema,
            **configs_copy,  # Add remaining configs
        }

        # Add thinking configuration if enabled
        if self.enable_thinking:
            gen_config_args["thinking_config"] = types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=self.thinking_budget_tokens,
            )

        if instruction:  # Only add system_instruction if it's not None and not empty
            gen_config_args["system_instruction"] = instruction

        final_configs = types.GenerateContentConfig(**gen_config_args)

        return messages, final_configs, thread

    async def add_response_to_thread(self, thread, response):
        contents = []

        # Handle thinking content if present
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                for part in candidate.content.parts:
                    if hasattr(part, "thought") and part.thought:
                        # Log thinking content similar to Anthropic implementation
                        self.logger.info(f"Thinking: {part.text}")
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

        messages, configs, thread = await self.prepare_config(
            user_input, base_thread=base_thread
        )
        self.logger.info(f"Sending Google API Request with Configs: {configs}")

        try:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=messages,
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
