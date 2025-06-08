import os

from openai import AsyncOpenAI

from neo.models.providers.openai import OpenAICompleteModel
from neo.tools.internal_tools.xai import WebSearch
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
        # hajack web search tool and turn it into a config
        extra_body = {}
        if self.tools is not None:
            # find web search tool and turn into a extra_body config
            web_search_tool = None
            for tool in self.tools:
                if isinstance(tool, WebSearch):
                    web_search_tool = tool
                    self.tools.remove(tool)
                    break

            if web_search_tool:
                extra_body["search_parameters"] = (
                    web_search_tool.search_parameters.model_dump()
                )

                self.logger.info(
                    f"Converted web search tool to extra body config: {extra_body}"
                )

        messages, configs, thread = await super().prepare_config(
            user_input, base_thread=base_thread
        )

        # add extra body
        if extra_body:
            if not configs:
                configs = {}
            if "extra_body" not in configs:
                configs["extra_body"] = {}
            configs["extra_body"].update(extra_body)

        # xai still uses `system` role instead of `developer`
        for m in messages:
            if Role(m.get("role")) == Role.DEVELOPER:
                m["role"] = Role.SYSTEM.value

        return messages, configs, thread

    async def add_response_to_thread(self, thread, response):
        await super().add_response_to_thread(thread, response)

        # log thinking
        msg = response.choices[0].message
        if hasattr(msg, "reasoning_content") and msg.reasoning_content:
            self.logger.thinking(msg.reasoning_content)
