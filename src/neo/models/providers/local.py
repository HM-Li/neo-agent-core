from typing import Optional

from openai import AsyncOpenAI  # Assumes an async client interface
from pydantic import BaseModel

from neo.contexts import Context, Thread
from neo.models.providers.openai import OpenAICompleteModel


class LocalModel(OpenAICompleteModel):
    """
    LocalModel is for models running on a local server using OpenAI-compatible API endpoints.
    All local-related logic (including defaults and client creation) is encapsulated here.
    """

    def create_client(self) -> AsyncOpenAI:
        if not self.custom_api_key:
            raise ValueError("Custom API key must be provided for local mode")
        return AsyncOpenAI(base_url=self.get_base_url(), api_key=self.custom_api_key)

    def get_base_url(self) -> str:
        return "http://localhost:4891/v1"

    async def prepare_config(
        self,
        user_input: str,
        base_thread: Thread = None,
    ) -> tuple:
        # Use local-specific defaults.
        _configs = {
            "model": self.model,
            "max_completion_tokens": 50,
            "temperature": 0,
            "top_p": 0.1,
        }
        messages, config, thread = await super().prepare_config(
            user_input=user_input, base_thread=base_thread
        )
        config.update(_configs)

        return messages, config, thread
