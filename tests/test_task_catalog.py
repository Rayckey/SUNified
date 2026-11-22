"""Keep published task references complete as registrations and programs change."""

import ast
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from kuafu.paths import LEARNED_PROGRAM_ROOT, PROJECT_ROOT, TASK_ROOT
from kuafu.pipeline.task_catalog import get_task, list_tasks


# Cabinet-door runtime support remains available to existing experiments;
# its single-, either- and both-door examples are excluded from the catalog.
UNLISTED_FORMULATION_IDS = {
    "Formulate-Panda-Cabinet-Door-Either-v0",
    "Formulate-Panda-Cabinet-Doors-Both-v0",
}


def test_catalog_covers_registered_robot_environments():
    gym = pytest.importorskip("gymnasium")
    from kuafu.rl.isaac.task_registry import ROBOT_TASK_REGISTRY

    for robot in ROBOT_TASK_REGISTRY.values():
        for module in robot.import_modules:
            importlib.import_module(module)
    expected = {name for name in gym.registry if "-Cabinet-Door" not in name and any(
        name.startswith(prefix + "-") for robot in ROBOT_TASK_REGISTRY.values() for prefix in robot.task_prefixes
    )}
    published = set()
    for summary in list_tasks():
        for env in get_task(summary["id"])["environments"]:
            if env.get("gym_id"):
                published.add(env["gym_id"])
                published.update(env["variants"])
                assert gym.spec(env["gym_id"]).entry_point == env["entry_point"]
    assert published == expected


def test_catalog_covers_formulation_scenes_and_saved_program_families():
    source = PROJECT_ROOT / "src/kuafu/isaac/formulation/panda.py"
    tree = ast.parse(source.read_text())
    mapping = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "_ENV_SPECS" for t in n.targets))
    expected = {ast.literal_eval(key) for key in mapping.keys} - UNLISTED_FORMULATION_IDS
    tasks = [get_task(row["id"]) for row in list_tasks()]
    actual = {item["gym_id"] for task in tasks for item in task["formulation"]}
    assert actual == expected
    families = {PROJECT_ROOT / path for task in tasks for path in task["program_families"]}
    assert families == {p for p in LEARNED_PROGRAM_ROOT.iterdir()
                        if p.is_dir() and p.name != "manipulation-door-either"}


def test_declared_assets_sources_and_program_examples_exist():
    covered_sources = set()
    for summary in list_tasks():
        task = get_task(summary["id"])
        assert (TASK_ROOT / summary["spec"]).is_file()
        paths = [task["scene_source"], task["success_check"]["source"],
                 *task["object_assets"], *task["program_families"], *task.get("related_sources", [])]
        paths.extend(item["runner"] for item in task["formulation"])
        for entry in task["environments"]:
            if "configuration" not in entry:
                continue
            inputs = entry["configuration"]
            paths.extend(value for value in (*inputs["assets"].values(), *inputs["controller_inputs"].values())
                         if "://" not in value and not value.startswith("${"))
            if inputs["default_program"]:
                paths.append(inputs["default_program"])
        for path in paths:
            assert (PROJECT_ROOT / path).exists(), (task["id"], path)
        covered_sources.update(paths)
        for saved in task["saved_programs"]:
            stages = PROJECT_ROOT / saved["path"] / "program"
            assert len(list(stages.glob("subtask_*"))) == saved["stages"]
            assert saved["stages"] > 0
        example = task["example_program"]
        if example:
            assert example in task["saved_programs"]
    base = PROJECT_ROOT / "src/kuafu/isaac/base/manipulation"
    task_sources = {str(path.relative_to(PROJECT_ROOT)) for path in base.glob("*_env.py")
                    if path.name not in {"panda_manipulation_env.py", "cabinet_door_env.py"}}
    assert task_sources <= covered_sources


def test_bundled_scene_asset_references_exist():
    from kuafu.paths import OBJECT_ROOT, ROBOT_ASSET_ROOT

    roots = {"OBJECT_ROOT": OBJECT_ROOT, "ROBOT_ASSET_ROOT": ROBOT_ASSET_ROOT}
    for source in (PROJECT_ROOT / "src/kuafu/isaac").rglob("*.py"):
        for node in ast.walk(ast.parse(source.read_text())):
            if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
                    and isinstance(node.left, ast.Name) and node.left.id in roots
                    and isinstance(node.right, ast.Constant) and isinstance(node.right.value, str)):
                assert (roots[node.left.id] / node.right.value).exists(), (source, node.right.value)


def test_bundled_usd_dependencies_resolve():
    sdf = pytest.importorskip("pxr.Sdf")
    usd_utils = pytest.importorskip("pxr.UsdUtils")
    missing = {}
    for path in (PROJECT_ROOT / "assets").rglob("*"):
        if path.suffix not in {".usd", ".usda", ".usdc"}:
            continue
        _, _, unresolved = usd_utils.ComputeAllDependencies(sdf.AssetPath(str(path)))
        # OmniPBR is provided by Isaac Sim's material library.
        local = set(unresolved) - {"OmniPBR.mdl"}
        if local:
            missing[str(path.relative_to(PROJECT_ROOT))] = sorted(local)
    assert not missing, missing


def test_task_selection_preserves_estop_direction_and_robot_layout():
    close = get_task("estop_close", "fr3_finray")
    opened = get_task("estop_open", "ur5_robotiq")
    assert close["task"] == "press the e-stop button"
    assert close["environments"][0]["gym_id"] == "FR3-FinRay-Table-EStop-Close"
    assert opened["task"] == "unlock the e-stop"
    assert opened["environments"][0]["gym_id"] == "UR5-Table-EStop-Closed"
    assert get_task("generated_cabinet")["target_travel_m"] == 0.1
    assert get_task("drawer")["task"] == "open the top drawer by 14 cm"
    panda = get_task("hang_cup", "franka_panda")
    fr3 = get_task("hang_cup", "fr3_finray")
    assert panda["environments"][0]["configuration"]["assets"]["cup_hanger"] != fr3["environments"][0]["configuration"]["assets"]["cup_hanger"]
    assert panda["interaction_frames"] == panda["environments"][0]["configuration"]["interaction_frames"]
    assert panda["scene_source"] == panda["environments"][0]["source"]
    with pytest.raises(ValueError, match="Unknown task"):
        get_task("../../pyproject.toml")
    with pytest.raises(ValueError, match="no registered"):
        get_task("generated_cabinet", "kinova_gen3")


def test_robot_selection_returns_its_configured_program():
    for task in list_tasks():
        if task["status"] != "registered":
            continue
        for robot in task["robots"]:
            selected = get_task(task["id"], robot)
            expected = selected["environments"][0]["configuration"]["default_program"]
            program = selected["example_program"]
            assert (program["path"] if program else None) == expected
    assert get_task("can", "franka_panda")["example_program"] != get_task("can")["example_program"]


def test_catalog_cli_works_without_loading_the_simulator(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
    command = [sys.executable, "-m", "kuafu", "tasks", "show", "can", "--robot", "kinova_gen3"]
    result = subprocess.run(command, cwd=tmp_path, env=env, text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["environments"][0]["gym_id"] == "Kinova-Table-Gen3-Can"
