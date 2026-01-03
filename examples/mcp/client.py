from neo.mcp.client import MCPClient
import asyncio
import os


async def main():

    current_dir = os.path.dirname(os.path.abspath(__file__))
    server_path = f"{current_dir}/mcp-server-demo/main.py"

    # Basic usage with Python script
    client = MCPClient(
        name="demo",
        command="python",
        args=[server_path],
    )

    # Alternative examples:
    # Docker-based MCP server
    # client = MCPClient(
    #     name="docker-server",
    #     command="docker",
    #     args=["run", "-i", "mcp-server-image"],
    # )
    #
    # npx-based MCP server
    # client = MCPClient(
    #     name="npx-server",
    #     command="npx",
    #     args=["-y", "@modelcontextprotocol/server-package"],
    # )
    #
    # With custom environment variables
    # client = MCPClient(
    #     name="demo",
    #     command="python",
    #     args=[server_path],
    #     env={"API_KEY": "your-key", "DEBUG": "true"},
    # )

    await client.aconnect()

    print(client.tools)

    test = await client.call_tool("test", {"text": "haha"})

    print(test)

    await client.aclose()


if __name__ == "__main__":

    asyncio.run(main())
