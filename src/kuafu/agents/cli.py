"""JSON command interface and optional stdio MCP server for external agents."""

import argparse
import json
import sys
from pathlib import Path

from .service import ToolService


def main(argv=None):
    parser = argparse.ArgumentParser(prog="kuafu agent", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("tools", help="Print the discoverable tools and their JSON Schemas")
    call = sub.add_parser("call", help="Call a tool with JSON from a file or stdin (-)")
    call.add_argument("name")
    call.add_argument("--input", default="-", help="JSON arguments file; default: stdin")
    sub.add_parser("serve", help="Run the optional MCP server over local stdio")
    args = parser.parse_args(argv)
    if args.command == "serve":
        from .mcp_server import serve

        serve()
        return 0
    service = ToolService()
    if args.command == "tools":
        print(json.dumps(service.list_tools(), indent=2))
        return 0
    try:
        raw = sys.stdin.read() if args.input == "-" else Path(args.input).read_text()
        result = service.call(args.name, json.loads(raw))
    except (ValueError, OSError) as exc:
        result = {
            "protocol_version": 1,
            "tool": args.name,
            "ok": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["ok"] else 2
