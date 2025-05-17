import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock

from neo.contexts import Thread, Context
from neo.types import contents as C
from neo.types.roles import Role

# Import model providers
from neo.models.providers.openai import OpenAIResponseModel, OpenAICompleteModel
from neo.models.providers.anthropic import AnthropicModel
from neo.models.providers.google import GoogleAIModel
from neo.models.providers.xai import XAIModel


# Fixtures for common test data
@pytest.fixture
def simple_prompt():
    return "What is the capital of France?"


@pytest.fixture
def thread_with_history():
    return Thread(
        contexts=[
            "What is the capital of France?",
            Context(role=Role.ASSISTANT, contents="The capital of France is Paris."),
            "What is the population of Paris?",
        ]
    )


@pytest.fixture
def image_context():
    return Context(
        contents=[
            C.TextContent(data="Describe this image"),
            C.ImageContent(data="https://example.com/image.jpg"),
        ]
    )


@pytest.fixture
def document_context():
    return Context(
        contents=[
            C.DocumentTextContent(data="https://example.com/document.pdf"),
            C.TextContent(data="Summarize this document"),
        ]
    )


# OpenAI Tests
@pytest.mark.asyncio
@patch("neo.models.providers.openai.AsyncOpenAI")
async def test_openai_response_model(mock_async_openai, simple_prompt):
    mock_client = MagicMock()
    mock_async_openai.return_value = mock_client

    mock_responses = MagicMock()
    mock_client.responses = mock_responses

    mock_create = AsyncMock()
    mock_responses.create = mock_create

    # Setup mock response
    mock_response = MagicMock(
        id="resp_67ccd2bed1ec8190b14f964abc0542670bb6a6b452d3795b",
        object="response",
        created_at=1741476542,
        status="completed",
        error=None,
        incomplete_details=None,
        model="gpt-4.1-2025-04-14",
        output=[
            MagicMock(
                type="message",
                id="msg_67ccd2bf17f0819081ff3bb2cf6508e60bb6a6b452d3795b",
                status="completed",
                role="assistant",
                content=[
                    MagicMock(
                        type="output_text",
                        text="Paris is the capital of France.",
                        annotations=[],
                    )
                ],
            )
        ],
        usage=MagicMock(
            input_tokens=36,
            output_tokens=87,
            total_tokens=123,
        ),
    )
    mock_create.return_value = mock_response

    model = OpenAIResponseModel(model="gpt-4o")
    result = await model.acreate(simple_prompt)

    # Check model was called with correct params
    mock_create.assert_called_once()
    args = mock_create.call_args[1]
    assert args["model"] == "gpt-4o"
    assert args["input"][0]["content"][0]["text"] == simple_prompt

    # Check response handling
    assert isinstance(result, Thread)
    assert len(result._contexts) == 2
    assert result._contexts[0].contents[0].data == simple_prompt
    assert result._contexts[1].contents[0].data == "Paris is the capital of France."
    assert result._contexts[1].provider_role == Role.ASSISTANT


# Anthropic Tests
@pytest.mark.asyncio
@patch("anthropic.AsyncAnthropic")
async def test_anthropic_model(mock_async_anthropic, simple_prompt):
    mock_client = MagicMock()
    mock_async_anthropic.return_value = mock_client

    mock_messages = MagicMock()
    mock_client.messages = mock_messages

    mock_create = AsyncMock()
    mock_messages.create = mock_create

    # Setup mock response
    mock_response = MagicMock(
        id="msg_01AvZNfhqQKUhCTn8NKBMcNG",
        content=[MagicMock(type="text", text="Paris is the capital of France.")],
        usage=MagicMock(input_tokens=10, output_tokens=8),
    )
    mock_create.return_value = mock_response

    model = AnthropicModel(model="claude-3-7-sonnet-20250219")
    result = await model.acreate(simple_prompt)

    # Check model was called with correct params
    mock_create.assert_called_once()
    args = mock_create.call_args[1]
    assert args["model"] == "claude-3-7-sonnet-20250219"
    assert args["messages"][0]["content"][0]["text"] == simple_prompt

    # Check response handling
    assert isinstance(result, Thread)
    assert len(result._contexts) == 2
    assert result._contexts[0].contents[0].data == simple_prompt
    assert result._contexts[1].contents[0].data == "Paris is the capital of France."
    assert result._contexts[1].provider_role == Role.ASSISTANT


# Google Tests
@pytest.mark.asyncio
@patch("google.genai.Client")
async def test_google_model(mock_google, simple_prompt):
    """Test that GoogleAIModel correctly processes inputs and parses responses."""
    mock_client = MagicMock()
    mock_google.return_value = mock_client

    # Setup mock instance
    aio = MagicMock()
    mock_client.aio = aio

    mock_models = MagicMock()
    aio.models = mock_models

    mock_generate_content = AsyncMock()
    mock_models.generate_content = mock_generate_content

    # Setup mock response
    mock_response = MagicMock(
        text="Paris is the capital of France.",
        usage_metadata=MagicMock(prompt_token_count=10, candidates_token_count=8),
        parsed=None,
    )
    mock_generate_content.return_value = mock_response

    model = GoogleAIModel(model="gemini-2.0-flash")
    result = await model.acreate(simple_prompt)

    # Check model was called with correct params
    mock_generate_content.assert_called_once()

    # Verify the contents parameter was passed correctly
    call_args = mock_generate_content.call_args[1]
    assert "contents" in call_args
    assert call_args["contents"][0] == simple_prompt

    # Check response handling
    assert isinstance(result, Thread)
    assert len(result._contexts) == 2
    assert result._contexts[0].contents[0].data == simple_prompt
    assert result._contexts[-1].contents[0].data == "Paris is the capital of France."
    assert result._contexts[-1].provider_role == Role.ASSISTANT
