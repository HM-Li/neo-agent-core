import pytest
from neo.types.contents import TextContent, ThoughtContent
from neo.types.modalities import Modality


def test_text_content():
    """Test TextContent functionality."""
    text_content = TextContent(data="Hello, world!")
    
    assert text_content.data == "Hello, world!"
    assert text_content.modality == Modality.TEXT


def test_thought_content():
    """Test ThoughtContent functionality."""
    # Test with thought data
    thought_content = ThoughtContent(
        thought_id="thought_123",
        data=["First thought", "Second thought", "Third thought"]
    )
    
    assert thought_content.thought_id == "thought_123"
    assert thought_content.data == ["First thought", "Second thought", "Third thought"]
    assert thought_content.modality == Modality.THOUGHT
    
    # Test with None data (redacted thoughts)
    redacted_thought = ThoughtContent(
        thought_id="thought_456",
        data=None
    )
    
    assert redacted_thought.thought_id == "thought_456"
    assert redacted_thought.data is None
    assert redacted_thought.modality == Modality.THOUGHT


def test_thought_modality():
    """Test that THOUGHT modality is properly defined."""
    assert Modality.THOUGHT.value == "thought"
    assert repr(Modality.THOUGHT) == "thought"
    
    # Test that it's included in the enum
    modalities = list(Modality)
    assert Modality.THOUGHT in modalities