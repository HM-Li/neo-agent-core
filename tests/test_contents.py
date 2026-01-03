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
        raw_data=["First thought", "Second thought", "Third thought"]
    )

    assert thought_content.raw_data == ["First thought", "Second thought", "Third thought"]
    assert thought_content.modality == Modality.TEXT

    # Test with None data (redacted thoughts)
    redacted_thought = ThoughtContent(
        raw_data=None
    )

    assert redacted_thought.raw_data is None
    assert redacted_thought.modality == Modality.TEXT


