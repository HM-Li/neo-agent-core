from inspect import signature
from neo.agentic.model_registry import ModelRegistry
from neo.models.providers.base import BaseChatModel


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
