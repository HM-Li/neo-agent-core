import os

from openai import AsyncOpenAI

from neo.models.providers.openai import OpenAICompleteModel
from neo.types.roles import Role


class MistralModel(OpenAICompleteModel):
    """
    MistralModel uses the OpenAI-compatible API interface.
    Mistral AI provides OpenAI-compatible endpoints for easy integration.
    """

    def create_client(self):
        api_key = None
        if self.custom_api_key is not None:
            api_key = self.custom_api_key
        else:
            api_key = os.getenv("MISTRAL_API_KEY")
        
        if not api_key:
            raise ValueError("MISTRAL_API_KEY must be provided for Mistral AI API")
        
        return AsyncOpenAI(base_url=self.get_base_url(), api_key=api_key)

    def get_base_url(self) -> str:
        return "https://api.mistral.ai/v1"

    async def prepare_config(self, user_input, base_thread=None) -> tuple:
        messages, configs, thread = await super().prepare_config(
            user_input, base_thread=base_thread
        )

        # Mistral uses `system` role instead of `developer`
        for m in messages:
            if Role(m.get("role")) == Role.DEVELOPER:
                m["role"] = Role.SYSTEM.value

        return messages, configs, thread