import os

from openai import AsyncOpenAI

from neo.models.providers.openai import OpenAICompleteModel
from neo.types.roles import Role


class XAIModel(OpenAICompleteModel):
    """
    XaiModel also uses the OpenAI Complete API.
    """

    def create_client(self):
        self.logger.warning("Ignoring custom API key for x.ai API")
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY must be provided for x.ai API")
        return AsyncOpenAI(base_url=self.get_base_url(), api_key=api_key)

    def get_base_url(self) -> str:
        return "https://api.x.ai/v1"

    async def prepare_config(self, user_input, base_thread=None) -> tuple:
        messages, configs, thread = await super().prepare_config(
            user_input, base_thread=base_thread
        )

        # xai still uses `system` role instead of `developer`
        for m in messages:
            if Role(m.get("role")) == Role.DEVELOPER:
                m["role"] = Role.SYSTEM.value

        return messages, configs, thread