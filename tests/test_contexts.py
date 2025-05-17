import pytest
import json
from unittest.mock import patch, MagicMock
from neo.contexts import Thread, Context
from neo.types import contents as C
from neo.types.roles import Role
from neo.types.contents import TextContent


def test_context_initialization():
    # Test with string
    ctx = Context(contents="hello")
    assert ctx.provider_role == Role.USER
    assert len(ctx.contents) == 1
    assert isinstance(ctx.contents[0], TextContent)
    assert ctx.contents[0].data == "hello"

    # Test with content object
    text = C.TextContent(data="world")
    ctx = Context(contents=text)
    assert len(ctx.contents) == 1
    assert ctx.contents[0] == text

    # Test with content list
    ctx = Context(contents=[C.TextContent(data="hello"), C.TextContent(data="world")])
    assert len(ctx.contents) == 2

    # Test with role
    ctx = Context(contents="hello", provider_role=Role.ASSISTANT)
    assert ctx.provider_role == Role.ASSISTANT


def test_thread_content_conversion():
    # Test that thread properly converts different input types

    # String to Context with TextContent
    thread = Thread(contexts="hello")
    assert isinstance(thread._contexts[0], Context)
    assert thread._contexts[0].contents[0].data == "hello"

    # List of strings
    thread = Thread(contexts=["hello", "world"])
    assert len(thread._contexts) == 2
    assert thread._contexts[0].contents[0].data == "hello"
    assert thread._contexts[1].contents[0].data == "world"

    # Mix of Context objects and strings
    ctx = Context(contents="from context", role=Role.ASSISTANT)
    thread = Thread(contexts=[ctx, "direct string"])
    assert len(thread._contexts) == 2
    assert thread._contexts[0] == ctx
    assert thread._contexts[1].contents[0].data == "direct string"

    # Content objects
    img = C.ImageContent(data="http://example.com/img.jpg")
    thread = Thread(contexts=[img])
    assert thread._contexts[0].contents[0] == img


@pytest.mark.asyncio
async def test_thread_context_manipulation():
    """Test adding and retrieving contexts from a Thread object."""
    thread = Thread()
    assert len(thread._contexts) == 0  # Using public property instead

    # Append string
    await thread.append_context("first")
    assert len(thread._contexts) == 1
    assert thread._contexts[0].contents[0].data == "first"

    # Append context
    ctx = Context(contents="second", role=Role.ASSISTANT)
    await thread.append_context(ctx)
    assert len(thread._contexts) == 2
    assert thread._contexts[1] == ctx

    # Get context
    assert thread.get_context(0).contents[0].data == "first"
    assert thread.get_context(1) == ctx
    assert thread.get_context(-1) == ctx

    with pytest.raises(IndexError):
        thread.get_context(2)


def test_thread_serialization():
    # Create a thread with just text and image content
    thread = Thread(
        contexts=[
            Context(
                role=Role.USER,
                contents=[
                    C.TextContent(data="Hello"),
                    C.ImageContent(data="http://example.com/img.jpg"),
                ],
            ),
            Context(
                role=Role.ASSISTANT,
                contents=[
                    C.TextContent(data="I see an image you sent"),
                ],
            ),
        ]
    )

    # Test dumps and loads
    json_str = thread.dumps()
    data = json.loads(json_str)

    assert len(data) == 2
    assert data[0]["provider_role"] == "user"


def test_thread_display(capsys):
    # Simple test to ensure display doesn't crash
    thread = Thread(
        contexts=[
            "User message",
            Context(role=Role.ASSISTANT, contents="Assistant reply"),
        ]
    )
    thread.display()
    captured = capsys.readouterr()

    # Basic check that output contains some content
    assert "User message" in captured.out
    assert "Assistant reply" in captured.out
