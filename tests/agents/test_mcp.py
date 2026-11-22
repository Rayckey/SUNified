"""Exercise the actual stdio protocol, not just Python handler calls."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

from mcp import Client
from mcp.client.stdio import StdioServerParameters
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def test_stdio_client_discovers_tools_and_receives_structured_data_and_images(tmp_path):
    (tmp_path / "result.json").write_text('{"passed": false, "actual_travel_m": 0.02}')
    Image.new("RGB", (4, 4), color="yellow").save(tmp_path / "frame.png")

    async def run():
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "kuafu", "agent", "serve"],
            cwd=str(tmp_path),
            env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "KUAFU_ARTIFACT_ROOT": str(tmp_path)},
        )
        async with Client(parameters) as client:
            assert client.server_info.name == "Kuafu"
            tools = await client.list_tools()
            names = {tool.name for tool in tools.tools}
            assert all(name.startswith("kuafu_") for name in names)
            assert {"kuafu_contract", "kuafu_submit_program", "kuafu_verify_program", "kuafu_start_stage"} <= names
            assert {"kuafu_list_tasks", "kuafu_get_task"} <= names
            catalog = await client.call_tool("kuafu_list_tasks", {"robot_id": "kinova_gen3"})
            assert not catalog.is_error
            task_ids = {task["id"] for task in catalog.structured_content["result"]["tasks"]}
            assert {"stack2", "stack3", "can", "pen_cup", "hang_cup", "estop_open", "estop_close"} <= task_ids
            task = await client.call_tool("kuafu_get_task", {"task_id": "stack2", "robot_id": "kinova_gen3"})
            assert task.structured_content["result"]["environments"][0]["gym_id"] == "Kinova-Table-Gen3-Stack-2"
            result = await client.call_tool("kuafu_read_artifact", {"path": str(tmp_path / "result.json")})
            assert not result.is_error
            assert result.structured_content["result"]["data"]["passed"] is False
            visual = await client.call_tool("kuafu_read_artifact", {"path": str(tmp_path / "frame.png")})
            assert any(block.type == "image" for block in visual.content)
            assert "image_base64" not in visual.structured_content["result"]
            failure = await client.call_tool("kuafu_job_status", {"job_id": "invalid"})
            assert failure.is_error and not failure.structured_content["ok"]

    asyncio.run(run())


def test_json_cli_from_other_directory_reports_parse_errors(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "kuafu", "agent", "call", "kuafu_contract"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        input="not json",
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 2
    assert json.loads(result.stdout)["ok"] is False
