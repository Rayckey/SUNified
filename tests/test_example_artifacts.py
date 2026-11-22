"""Keep the published end-to-end evidence attached to its exact inputs and outputs."""

import hashlib
import json
from kuafu.paths import PROJECT_ROOT


def test_video_to_policy_evidence_is_complete_and_unchanged():
    example = PROJECT_ROOT / "examples/04_video_to_policy"
    manifest = json.loads((example / "output/manifest.json").read_text())
    for base, entries in ((PROJECT_ROOT, manifest["inputs"]), (example, manifest["outputs"])):
        for name, expected in entries.items():
            path = (base / name).resolve()
            assert path.is_relative_to(base.resolve())
            with path.open("rb") as handle:
                assert hashlib.file_digest(handle, "sha256").hexdigest() == expected, name
    collection = json.loads((example / "output/collection.json").read_text())
    evaluation = json.loads((example / "output/evaluation.json").read_text())
    training = json.loads((example / "output/training.json").read_text())
    assert collection["episodes"] == len(collection["episode_results"]) >= 2
    assert all(episode["passed"] and episode["solver_passed"] for episode in collection["episode_results"])
    assert not set(training["train_episodes"]) & set(training["validation_episodes"])
    assert training["training_transitions"] + training["validation_transitions"] == collection["transitions"]
    assert evaluation["controller"] == "bc_policy"
    assert evaluation["mpc_solves"] == evaluation["model_requests"] == 0
    assert evaluation["passed"] == (evaluation["program_passed"] and evaluation["physical"]["passed"])
    assert (example / "output/rollout.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
