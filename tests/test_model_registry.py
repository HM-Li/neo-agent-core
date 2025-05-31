from inspect import signature

from neo.agentic.model_registry import ModelRegistry
from neo.models.providers.anthropic import AnthropicModel
from neo.models.providers.base import BaseChatModel
from neo.types.modalities import Modality


def test_create_model_signature_matches_base_chat_model():
    # Get the signatures
    registry = ModelRegistry()
    create_model_sig = signature(registry.create_model)
    base_model_sig = signature(BaseChatModel.__init__)

    ignore_params = ["self", "input_modalities"]

    # Get parameters excluding 'self' and 'name'/'model'
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
    assert "claude-sonnet-4-20250514" in registry.get_all_models()

    # Test Claude Opus 4 registration
    assert "claude-opus-4-20250514" in registry.get_all_models()

    # Test that they map to AnthropicModel
    all_models = registry.get_all_models()
    sonnet4_info = all_models["claude-sonnet-4-20250514"]
    opus4_info = all_models["claude-opus-4-20250514"]

    assert (
        sonnet4_info.get("model_class") == AnthropicModel
        or sonnet4_info.get("cls") == AnthropicModel
    )
    assert (
        opus4_info.get("model_class") == AnthropicModel
        or opus4_info.get("cls") == AnthropicModel
    )


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


def test_create_new_claude_models_with_thinking():
    """Test that new Claude models can be created with thinking parameters."""
    registry = ModelRegistry()

    # Test creating Claude Sonnet 4 with thinking enabled
    model = registry.create_model(
        model="claude-sonnet-4-20250514",
        enable_thinking=True,
        thinking_budget_tokens=512,
    )

    assert isinstance(model, AnthropicModel)
    assert model.enable_thinking is True
    assert model.thinking_budget_tokens == 512

    # Test creating Claude Opus 4 with thinking enabled
    model = registry.create_model(
        model="claude-opus-4-20250514",
        enable_thinking=True,
        thinking_budget_tokens=1024,
    )

    assert isinstance(model, AnthropicModel)
    assert model.enable_thinking is True
    assert model.thinking_budget_tokens == 1024
