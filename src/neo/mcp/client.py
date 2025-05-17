import asyncio
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
    def __init__(self, name: str, server_script_path: str):
        """Initialize the MCP client
        Args:
            name: Name of the client
        """
        self.name = name
        self.session: Optional[ClientSession] = None
        self.exit_stack = None
        self.server_script_path: str = server_script_path
        self.stdio = None
        self.write = None
        self.tools: Dict[Tool] = {}

        self._cleanup_lock = asyncio.Lock()
        self.logger = get_logger(f"MCPClient-{name}")

    async def aconnect(self):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        try:
            if self.exit_stack is not None:
                # prevent exiting context manager if already connected and not closed
                raise ValueError(
                    "Client is already connected. Please close it before reconnecting."
                )

            self.exit_stack = AsyncExitStack()
            is_python = self.server_script_path.endswith(".py")
            is_js = self.server_script_path.endswith(".js")
            if not (is_python or is_js):
                raise ValueError("Server script must be a .py or .js file")

            command = "python" if is_python else "node"
            server_params = StdioServerParameters(
                command=command, args=[self.server_script_path], env=None
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
