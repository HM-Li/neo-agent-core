# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mcp[cli]>=1.6.0",
# ]
# ///

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo")


@mcp.tool()
def echo(text: str) -> str:
    """echo"""
    return text


@mcp.resource("resource://weather")
def weather() -> str:
    """weather"""
    return "sunny"


@mcp.prompt()
def location() -> str:
    """location"""
    return "New York"


if __name__ == "__main__":
    mcp.run()
