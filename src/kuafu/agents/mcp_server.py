"""The official MCP SDK adapts the same tools exposed to Python and JSON clients."""

import asyncio
import json

from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, ImageContent, ListToolsResult, TextContent, Tool, ToolAnnotations

from kuafu import __version__

from .service import ToolService


def create_server(service=None):
    service = service or ToolService()
    definitions = service.list_tools()

    async def list_tools(context, params):
        return ListToolsResult(
            tools=[
                Tool(
                    name=tool["name"],
                    description=tool["description"],
                    input_schema=tool["inputSchema"],
                    annotations=ToolAnnotations(
                        read_only_hint=tool["readOnly"],
                        destructive_hint=False,
                        open_world_hint=tool["name"] == "kuafu_start_stage",
                    ),
                )
                for tool in definitions
            ]
        )

    async def call_tool(context, params):
        result = await asyncio.to_thread(service.call, params.name, params.arguments)
        content = []
        if result["ok"] and "image_base64" in result["result"]:
            data = result["result"].pop("image_base64")
            content.append(ImageContent(type="image", data=data, mime_type=result["result"]["mime_type"]))
        content.append(TextContent(type="text", text=json.dumps(result)))
        return CallToolResult(content=content, structured_content=result, is_error=not result["ok"])

    return Server(
        "Kuafu",
        version=__version__,
        instructions="Discover existing tasks with kuafu_list_tasks and kuafu_get_task. "
        "Create a task with kuafu_create_run and pass its run_id to tools. "
        "Read kuafu_contract, propose candidates, and inspect kuafu_run_status after reconnecting. "
        "Poll long jobs; a successful tool call does not mean physical task success.",
        on_list_tools=list_tools,
        on_call_tool=call_tool,
        get_tool_input_schema=lambda name: next(
            (tool["inputSchema"] for tool in definitions if tool["name"] == name), None
        ),
    )


def serve():
    async def run():
        server = create_server()
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    asyncio.run(run())
