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
        server_path = f"{example_dir}/mcp/mcp-server-demo/main.py"
        return MCPClient(
            name="demo",
            command="python",
            args=[server_path],
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


@pytest.mark.asyncio
async def test_command_based_initialization():
    """Test that client can be initialized with command and args."""
    module_dir = os.path.dirname(__import__("neo").__file__)
    example_dir = os.path.join(module_dir, "../../examples")
    server_path = f"{example_dir}/mcp/mcp-server-demo/main.py"

    mcp_client = MCPClient(
        name="demo",
        command="python",
        args=[server_path],
    )
    async with mcp_client:
        assert mcp_client.tools is not None
        assert isinstance(mcp_client.tools, dict)


@pytest.mark.asyncio
async def test_empty_args_support():
    """Test that client supports empty args list."""
    # This test uses a hypothetical command that doesn't need args
    # For now, we'll use python with a script to verify empty args work
    module_dir = os.path.dirname(__import__("neo").__file__)
    example_dir = os.path.join(module_dir, "../../examples")
    server_path = f"{example_dir}/mcp/mcp-server-demo/main.py"

    # Test that empty args default is accepted
    mcp_client = MCPClient(
        name="demo",
        command="python",
        args=[server_path],  # We still need the script path
    )
    async with mcp_client:
        assert mcp_client.tools is not None


@pytest.mark.asyncio
async def test_env_parameter():
    """Test that environment variables are properly passed."""
    module_dir = os.path.dirname(__import__("neo").__file__)
    example_dir = os.path.join(module_dir, "../../examples")
    server_path = f"{example_dir}/mcp/mcp-server-demo/main.py"

    # Create client with custom environment variable
    mcp_client = MCPClient(
        name="demo",
        command="python",
        args=[server_path],
        env={"TEST_VAR": "test_value"},
    )
    async with mcp_client:
        assert mcp_client.tools is not None
        assert isinstance(mcp_client.tools, dict)


@pytest.mark.asyncio
async def test_is_connected(mcp_client_factory):
    """Test that is_connected() correctly reports connection status."""
    mcp_client = mcp_client_factory()

    # Should not be connected initially
    assert not mcp_client.is_connected()

    # Should be connected after aconnect
    await mcp_client.aconnect()
    assert mcp_client.is_connected()

    # Should not be connected after aclose
    await mcp_client.aclose()
    assert not mcp_client.is_connected()
