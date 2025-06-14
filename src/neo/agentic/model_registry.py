from typing import Callable, List, Literal, Optional

from pydantic import BaseModel

from neo.mcp.client import MCPClient
from neo.models.providers.anthropic import AnthropicModel
from neo.models.providers.base import BaseChatModel
from neo.models.providers.google import GoogleAIModel
from neo.models.providers.openai import OpenAICompleteModel, OpenAIResponseModel
from neo.models.providers.xai import XAIModel
from neo.tools import Tool
from neo.types.modalities import Modality
from neo.utils.singleton import Singleton


class ModelRegistry(metaclass=Singleton):
    """
    A simple namespace for model registrations.
    """

    def __init__(self):
        # internal registry dict
        self._registry = {}

    def register_model(
        self, model: str, cls: BaseChatModel, input_modalities: List[Modality]
    ) -> None:
        """
        Register a model under the given model name.

        Parameters
        ----------
        model : str
            The string key you want to associate with the model.
        cls : BaseChatModel
            The actual model you want to store.
        input_modalities : List[Modality]
            The input modalities for the model.

        Raises
        ------
        ValueError
            If a model is already registered under the given model name.
        """

        if not issubclass(cls, BaseChatModel):
            raise ValueError("Model must be a subclass of BaseChatModel")

        model = model.lower()

        if model in self._registry:
            raise ValueError(f"A model is already registered under the model '{model}'")
        self._registry[model] = {"class": cls, "input_modalities": input_modalities}

    def get_model_registry(self, model: str) -> dict:
        """
        Retrieve a previously registered model by model name.

        Parameters
        ----------
        model : str
            The string key associated with the model.

        Returns
        -------
        type
            The model if found.

        Raises
        ------
        KeyError
            If no model has been registered under the given model name.
        """
        model = model.lower()
        if model not in self._registry:
            raise KeyError(f"No model has been registered under the model '{model}'")
        return self._registry[model]

    def create_model(
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
    ) -> BaseChatModel:
        """
        Create an instance of a registered model.

        Parameters
        ----------
        model : str
            The string key associated with the model.
        output_modalities : List[Modality], optional
            The output modalities for the model, by default [Modality.TEXT]
        system : str, optional
            The system message for the model, by default None
        configs : dict, optional
            Additional configurations for the model, by default None
        custom_api_key : str, optional
            Custom API key for the model, by default None

        Returns
        -------
        BaseChatModel
            An instance of the registered model.
        """
        model = model.lower()
        if model not in self._registry:
            raise KeyError(f"No model has been registered under the name '{model}'")

        model_registry = self._registry[model]
        cls = model_registry["class"]
        input_modalities = model_registry["input_modalities"]

        model_instance = cls(
            model=model,
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            instruction=instruction,
            configs=configs,
            custom_api_key=custom_api_key,
            json_mode=json_mode,
            boolean_response=boolean_response,
            structured_response_model=structured_response_model,
            tools=tools,
            mcp_clients=mcp_clients,
            tool_choice=tool_choice,
            timeaware=timeaware,
            enable_thinking=enable_thinking,
            thinking_budget_tokens=thinking_budget_tokens,
        )
        return model_instance

    @property
    def supported_models(self) -> dict:
        """
        Return all registered models.

        Returns
        -------
        dict
            A dictionary of all registered models.
        """
        return self._registry

    def check_model_input_modalities(self, model: str) -> List[Modality]:
        """
        Get the input modalities for a registered model.

        Parameters
        ----------
        model : str
            The string key associated with the model.

        Returns
        -------
        List[Modality]
            The input modalities for the model.
        """
        model = model.lower()
        if model not in self._registry:
            raise KeyError(f"No model has been registered under the name '{model}'")
        return self._registry[model]["input_modalities"]


# ===========================

mr = ModelRegistry()

# model, cls, input_modalities
models = [
    (
        "gpt-4o",
        OpenAIResponseModel,
        [Modality.TEXT, Modality.IMAGE, Modality.STRUCTURED, Modality.DOCUMENT],
    ),
    (
        "gpt-4.1",
        OpenAIResponseModel,
        [Modality.TEXT, Modality.IMAGE, Modality.STRUCTURED, Modality.DOCUMENT],
    ),
    (
        "gpt-4.1-mini",
        OpenAIResponseModel,
        [Modality.TEXT, Modality.IMAGE, Modality.STRUCTURED, Modality.DOCUMENT],
    ),
    (
        "gpt-4.1-nano",
        OpenAIResponseModel,
        [Modality.TEXT, Modality.IMAGE, Modality.STRUCTURED, Modality.DOCUMENT],
    ),
    (
        "gpt-4o-audio-preview",
        OpenAICompleteModel,
        [Modality.TEXT, Modality.AUDIO],
    ),
    (
        "o3-pro",
        OpenAIResponseModel,
        [Modality.TEXT, Modality.IMAGE, Modality.STRUCTURED, Modality.DOCUMENT],
    ),
    (
        "o3",
        OpenAIResponseModel,
        [Modality.TEXT, Modality.IMAGE, Modality.STRUCTURED, Modality.DOCUMENT],
    ),
    ("o3-mini", OpenAIResponseModel, [Modality.TEXT]),
    (
        "o4-mini",
        OpenAIResponseModel,
        [Modality.TEXT, Modality.IMAGE, Modality.STRUCTURED, Modality.DOCUMENT],
    ),
    (
        "claude-3-7-sonnet-latest",
        AnthropicModel,
        [Modality.TEXT, Modality.IMAGE, Modality.STRUCTURED, Modality.DOCUMENT],
    ),
    (
        "claude-3-5-haiku-latest",
        AnthropicModel,
        [Modality.TEXT, Modality.STRUCTURED, Modality.DOCUMENT],
    ),
    (
        "claude-sonnet-4-20250514",
        AnthropicModel,
        [Modality.TEXT, Modality.IMAGE, Modality.STRUCTURED, Modality.DOCUMENT],
    ),
    (
        "claude-opus-4-20250514",
        AnthropicModel,
        [Modality.TEXT, Modality.IMAGE, Modality.STRUCTURED, Modality.DOCUMENT],
    ),
    (
        "gemini-2.0-flash",
        GoogleAIModel,
        [Modality.TEXT],
    ),
    (
        "gemini-2.0-flash-lite",
        GoogleAIModel,
        [Modality.TEXT],
    ),
    (
        "gemini-2.5-flash-preview-04-17",
        GoogleAIModel,
        [Modality.TEXT],
    ),
    (
        "gemini-2.5-pro-preview-05-06",
        GoogleAIModel,
        [Modality.TEXT],
    ),
    ("grok-2", XAIModel, [Modality.TEXT, Modality.STRUCTURED]),
    ("grok-3-mini", XAIModel, [Modality.TEXT, Modality.STRUCTURED]),
    ("grok-3", XAIModel, [Modality.TEXT, Modality.STRUCTURED]),
]

# Register models
for model, cls, input_modalities in models:
    mr.register_model(model, cls, input_modalities)
