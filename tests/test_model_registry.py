from inspect import signature

from neo.agentic.model_registry import ModelRegistry
from neo.models.providers.anthropic import AnthropicModel
from neo.models.base import BaseChatModel
from neo.models.providers.openai import OpenAIResponseModel
from neo.types.modalities import Modality


def test_create_model_signature_matches_base_chat_model():
    # Get the signatures
    registry = ModelRegistry()
    create_model_sig = signature(registry.create_model)
    base_model_sig = signature(BaseChatModel.__init__)

    ignore_params = ["self", "input_modalities", "fuzzy_mode"]

    # Get parameters excluding 'self', 'input_modalities', and 'fuzzy_mode'
    create_params = {
        k: v for k, v in create_model_sig.parameters.items() if k not in ignore_params
    }
    base_params = {
        k: v for k, v in base_model_sig.parameters.items() if k not in ignore_params
    }

    # Compare parameters
    assert (
        create_params == base_params
    ), "create_model signature doesn't match BaseChatModel.__init__"


def test_new_claude_models_registered():
    """Test that new Claude models are properly registered."""
    registry = ModelRegistry()

    # Test Claude Sonnet 4 registration
    assert "claude-sonnet-4-20250514" in registry.supported_models

    # Test Claude Opus 4 registration
    assert "claude-opus-4-20250514" in registry.supported_models

    # Test that they map to AnthropicModel
    all_models = registry.supported_models
    sonnet4_info = all_models["claude-sonnet-4-20250514"]
    opus4_info = all_models["claude-opus-4-20250514"]

    assert sonnet4_info.get("class") == AnthropicModel
    assert opus4_info.get("class") == AnthropicModel


def test_new_claude_models_support_multimodal():
    """Test that new Claude models support expected modalities."""
    registry = ModelRegistry()

    # Test Claude Sonnet 4 modalities
    sonnet4_modalities = registry.check_model_input_modalities(
        "claude-sonnet-4-20250514"
    )
    expected_modalities = [
        Modality.TEXT,
        Modality.IMAGE,
        Modality.STRUCTURED,
        Modality.DOCUMENT,
    ]

    assert set(sonnet4_modalities) == set(expected_modalities)

    # Test Claude Opus 4 modalities
    opus4_modalities = registry.check_model_input_modalities("claude-opus-4-20250514")
    assert set(opus4_modalities) == set(expected_modalities)


def test_gpt5_pro_model_registered():
    """Test that gpt-5-pro is properly registered."""
    registry = ModelRegistry()

    # Test gpt-5-pro registration
    assert "gpt-5-pro" in registry.supported_models

    # Test that it maps to OpenAIResponseModel
    all_models = registry.supported_models
    gpt5_pro_info = all_models["gpt-5-pro"]

    assert gpt5_pro_info.get("class") == OpenAIResponseModel


def test_gpt5_pro_supports_multimodal():
    """Test that gpt-5-pro supports expected modalities."""
    registry = ModelRegistry()

    # Test gpt-5-pro modalities
    gpt5_pro_modalities = registry.check_model_input_modalities("gpt-5-pro")
    expected_modalities = [
        Modality.TEXT,
        Modality.IMAGE,
        Modality.STRUCTURED,
        Modality.DOCUMENT,
    ]

    assert set(gpt5_pro_modalities) == set(expected_modalities)
