"""Check program manifests and collection bookkeeping across their consumers."""

import json
from pathlib import Path

import pytest

from kuafu.data.collection import audit_mpc1600, monitor_mpc1600
from kuafu.data.collection import run_first2_new_programs_mpc1600 as collector


ROOT = Path(__file__).resolve().parents[1]


def test_bundled_campaign_is_usable_by_collector_and_auditor():
    manifest = json.loads(
        (ROOT / "assets/configs/campaigns/mpc_first_two_programs_1600.json").read_text()
    )
    collector.validate_manifest(manifest)
    tasks = audit_mpc1600.validate_manifest(manifest)
    assert len(tasks) == 7
    for task in tasks.values():
        for program in task["programs"]:
            assert collector.validate_program(program).is_dir()
            audit_mpc1600.validate_learned_program(program)


@pytest.mark.parametrize("index", [1, 2])
def test_monitor_keeps_program_numbers_in_worker_groups(tmp_path, index):
    program = f"program_{index:02d}"
    attempt = f"{program}_2026-10-19"
    log = tmp_path / "drawer/tranche_02/runs" / attempt / "logs/worker_003.txt"
    assert monitor_mpc1600.classify(tmp_path, log) == (
        "drawer", "tranche_02", program, attempt, 3,
    )
