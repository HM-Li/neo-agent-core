import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from neo.tools import Tool
from neo.tools.image_gen.openai import agenerate_image
from neo.tools.speech.openai import aspeech2text
from neo.types.contents import AudioContent, ImageContent, TextContent


def test_tool_creation():
    # Test basic tool creation
    def example_function(param1: str, param2: int) -> str:
        """Example function docstring."""
        return f"{param1}: {param2}"

    tool = Tool(func=example_function)

    # Check attributes
    assert tool.name == "example_function"
    assert tool.description == "Example function docstring."
    assert len(tool.params["properties"]) == 2

    # Check function call
    result = tool.func("test", 123)
    assert result == "test: 123"


@pytest.mark.asyncio
@patch("neo.tools.image_gen.openai.AsyncOpenAI")
async def test_agenerate_image(mock_async_openai):
    """Test the agenerate_image function with mocked OpenAI API."""
    # Set up the mock client and response
    mock_client = MagicMock()
    mock_async_openai.return_value = mock_client

    mock_images = MagicMock()
    mock_client.images = mock_images

    mock_generate = AsyncMock()
    mock_images.generate = mock_generate

    mock_generate.return_value = MagicMock(
        data=[MagicMock(b64_json=None, url="http://example.com/image.png")]
    )

    result = await agenerate_image(prompt="a cat", model="dall-e-3")

    # Check API was called with correct params
    mock_generate.assert_called_once()
    call_args = mock_generate.call_args[1]
    assert call_args["prompt"] == "a cat"
    assert call_args["model"] == "dall-e-3"

    # Check returned content
    assert isinstance(result, ImageContent)


@pytest.mark.asyncio
@patch("neo.tools.speech.openai.speech2text.AsyncOpenAI")
async def test_aspeech2text(mock_async_openai):
    mock_client = MagicMock()
    mock_async_openai.return_value = mock_client

    mock_audio = MagicMock()
    mock_client.audio = mock_audio

    mock_transcriptions = MagicMock()
    mock_audio.transcriptions = mock_transcriptions

    mock_create = AsyncMock()
    mock_transcriptions.create = mock_create

    transcription = "This is a transcription of the audio."
    mock_create.return_value = MagicMock(text=transcription)

    # Create audio content
    audio = AudioContent(
        data=b"audio data", mime_type="audio/wav", file_name="test.wav"
    )

    result = await aspeech2text(audio=audio)

    # Check API was called
    mock_create.assert_called_once()

    # Check result
    assert isinstance(result, TextContent)
    assert result.data == transcription
