from neo.mcp.client import MCPClient
import asyncio
import os


async def main():

    current_dir = os.path.dirname(os.path.abspath(__file__))
    client = MCPClient(
        name="demo",
        server_script_path=f"{current_dir}/mcp-server-demo/main.py",
    )

    await client.aconnect()

    print(client.tools)

    test = await client.call_tool("test", {"text": "haha"})

    print(test)

    await client.close()


if __name__ == "__main__":

    asyncio.run(main())
