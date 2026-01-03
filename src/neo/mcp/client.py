import asyncio
import os
from typing import Optional, Dict
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult as MCPCallToolResult

from neo.utils.logger import get_logger
from neo.tools import Tool
from neo.types.errors import ToolError, ToolRuntimeError
from neo.types import contents


class MCPClient:
    def __init__(
        self,
        name: str,
        command: str,
        args: list[str] = [],
        env: Optional[Dict[str, str]] = None,
    ):
        """Initialize the MCP client

        Args:
            name: Name of the client
            command: Command to run the MCP server (e.g., 'python', 'node', 'docker', 'npx')
            args: Arguments to pass to the command (default: [])
            env: Environment variables to merge with system environment (default: None)
        """
        self.name = name
        self.session: Optional[ClientSession] = None
        self.exit_stack = None
        self.command: str = command
        self.args: list[str] = args
        self.env: Optional[Dict[str, str]] = env
        self.stdio = None
        self.write = None
        self.tools: Dict[Tool] = {}

        self._cleanup_lock = asyncio.Lock()
        self.logger = get_logger(f"MCPClient-{name}")

    def is_connected(self) -> bool:
        """Check if the client is currently connected to a server"""
        return self.exit_stack is not None and self.session is not None

    async def aconnect(self):
        """Connect to an MCP server using the configured command and args"""
        try:
            if self.exit_stack is not None:
                # prevent exiting context manager if already connected and not closed
                raise ValueError(
                    "Client is already connected. Please close it before reconnecting."
                )

            self.exit_stack = AsyncExitStack()

            # Merge custom env with system environment if provided
            merged_env = os.environ.copy() if self.env else None
            if self.env and merged_env:
                merged_env.update(self.env)

            server_params = StdioServerParameters(
                command=self.command, args=self.args, env=merged_env
            )

            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.stdio, self.write)
            )

            await self.session.initialize()

            # List available tools
            response = await self.session.list_tools()
            tools = response.tools

            # Initialize tools
            self.tools = {
                tool.name: Tool(
                    name=tool.name,
                    description=tool.description,
                    params=tool.inputSchema,
                    provider=self,
                )
                for tool in tools
            }

            self.logger.info(
                f"\nConnected to MCP server {self.name} with tools: {[tool.name for tool in tools]}",
            )
        except Exception as e:
            await self.aclose()
            raise

    async def aclose(self):
        """Close the connection to the server"""
        async with self._cleanup_lock:
            try:
                if self.exit_stack is not None:
                    await self.exit_stack.aclose()
                    self.exit_stack = None
                self.session = None
                self.stdio = None
                self.write = None
                self.tools = {}

                self.logger.info(f"Closed connection to MCP server {self.name}")
            except RuntimeError as e:
                # Handle Jupyter notebook context switching issues
                if "cancel scope" in str(e) or "different task" in str(e):
                    self.logger.warning(
                        f"Clean shutdown not possible (likely Jupyter notebook context): {e}. "
                        "Resources will be cleaned up by the kernel."
                    )
                    # Force cleanup of references
                    self.exit_stack = None
                    self.session = None
                    self.stdio = None
                    self.write = None
                    self.tools = {}
                else:
                    self.logger.error(
                        f"Error closing connection to MCP server {self.name}: {e}"
                    )
                    raise
            except Exception as e:
                self.logger.error(
                    f"Error closing connection to MCP server {self.name}: {e}"
                )
                raise

    async def call_tool(self, tool_name: str, tool_args: dict):
        """Call a tool with the given arguments"""
        if self.session is None:
            raise ValueError("Session is not connected for client {self.name}")

        if tool_name not in self.tools:
            raise ToolError(f"Tool {tool_name} not found in client {self.name}")

        response: MCPCallToolResult = await self.session.call_tool(tool_name, tool_args)

        if response.isError:
            raise ToolRuntimeError(
                f"Error calling tool {tool_name}: {response.content[0].text}"
            )

        # only support text content for now
        response_contents = []
        for c in response.content:
            if c.type == "text":
                response_contents.append(contents.TextContent(data=c.text))
            else:
                raise TypeError(f"Unsupported API response content type: {c.type}")

        return response_contents

    async def __aenter__(self):
        """Async context manager enter."""
        # current_task = asyncio.current_task()
        # self.logger.info(f"aconnect in task: {current_task}")
        await self.aconnect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # current_task = asyncio.current_task()
        # self.logger.info(f"aclose in task: {current_task}")
        """Async context manager exit."""
        await self.aclose()
