"""Test module for MCP client functionality."""

import os
import pytest_asyncio
import pytest
from neo.mcp.client import MCPClient


@pytest_asyncio.fixture
def mcp_client_factory():
    """Returns a factory function for creating an MCPClient."""

    def _create_client():
        module_dir = os.path.dirname(__import__("neo").__file__)
        example_dir = os.path.join(module_dir, "../../examples")
        return MCPClient(
            name="demo",
            server_script_path=f"{example_dir}/mcp/mcp-server-demo/main.py",
        )

    return _create_client


@pytest.mark.asyncio
async def test_client_connection(mcp_client_factory):
    """Test that client connection establishes successfully."""
    mcp_client = mcp_client_factory()
    async with mcp_client:
        assert mcp_client.tools is not None
        assert isinstance(mcp_client.tools, dict)


@pytest.mark.asyncio
async def test_call_tool(mcp_client_factory):
    """Test that tool calling works properly."""
    mcp_client = mcp_client_factory()
    async with mcp_client:
        result = await mcp_client.call_tool("echo", {"text": "haha"})
        assert result is not None


@pytest.mark.asyncio
async def test_invalid_tool_call(mcp_client_factory):
    """Test that calling invalid tool raises exception."""
    mcp_client = mcp_client_factory()
    async with mcp_client:
        with pytest.raises(Exception):
            await mcp_client.call_tool("nonexistent_tool", {})


@pytest.mark.asyncio
async def test_client_name(mcp_client_factory):
    """Test that client name is set correctly."""
    mcp_client = mcp_client_factory()
    async with mcp_client:
        assert mcp_client.name == "demo"


@pytest.mark.asyncio
async def test_client_reconnection(mcp_client_factory):
    """Test that client can reconnect successfully."""
    mcp_client = mcp_client_factory()
    async with mcp_client:
        await mcp_client.aclose()
        await mcp_client.aconnect()
        assert mcp_client.tools is not None
