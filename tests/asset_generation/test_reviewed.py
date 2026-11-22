import json
import zipfile

import pytest

from kuafu.asset_generation.io import digest, write_json
from kuafu.asset_generation.reviewed import build_reviewed, package_reviewed, verify_physics_report
from kuafu.asset_generation.video import prepare


@pytest.fixture
def reviewed(tmp_path, video, holder, monkeypatch):
    workspace = tmp_path / "example"
    prepare(video, "hollow test holder", workspace, 2)
    write_json(workspace / "scene_v1.json", holder)
    manifest = tmp_path / "batch.json"
    write_json(
        manifest,
        {
            "version": 1,
            "source_mode": "interactive_assistant",
            "jobs": [
                {
                    "workspace": "example",
                    "scene": "scene_v1.json",
                    "physics": {"mass_kg": 0.35, "mass_is_estimate": True},
                    "review": "Test fixture: a reviewed hollow holder.",
                    "probe": None,
                }
            ],
        },
    )
    monkeypatch.setattr("kuafu.asset_generation.export.render_views", lambda *args: None)
    monkeypatch.setattr("kuafu.asset_generation.export.write_viewer", lambda *args: None)
    return manifest, workspace


def synthetic_report(out):
    """Synthetic validator fixture only; NOT an actual physics simulation."""
    return {
        "passed": True,
        "asset": str(out / "asset_dynamic.usda"),
        "steps": 240,
        "physics_dt": 1 / 120,
        "scope": "Synthetic test fixture; no simulation.",
        "asset_layers": [
            {"path": str(out / name), "sha256": digest(out / name)}
            for name in ["asset_dynamic.usda", "asset_geometry.usda"]
        ],
    }


def test_saved_workflow_preserves_source_and_requires_physics_before_publishing(reviewed):
    manifest, workspace = reviewed
    (out,) = build_reviewed(manifest)
    source = json.loads((workspace / "scene_v1.json").read_text())
    assert json.loads((out / "scene.json").read_text()) == source
    metadata = json.loads((out / "asset.json").read_text())
    assert metadata["reconstruction"]["standalone_model_api_calls"] == 0
    assert metadata["mass_basis"] == "estimated"
    assert not metadata["isaac_physics_tested"]
    with pytest.raises(ValueError, match="physics_smoke"):
        package_reviewed(manifest)
    with pytest.raises(ValueError, match="new or empty"):
        build_reviewed(manifest)


def test_publisher_matches_physics_hashes_and_keeps_archive_portable(reviewed):
    manifest, _ = reviewed
    (out,) = build_reviewed(manifest)
    write_json(out / "physics_smoke.json", synthetic_report(out))
    (archive,) = package_reviewed(manifest)
    assert json.loads((out / "asset.json").read_text())["isaac_physics_tested"]
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        assert "asset_dynamic.usda" in bundle.namelist()
        assert "asset_geometry.usda" in bundle.namelist()
        assert "scene.json" in bundle.namelist()
        assert "README.md" in bundle.namelist()
    with pytest.raises(ValueError, match="Archive already exists"):
        package_reviewed(manifest)


def test_stale_geometry_never_inherits_a_passing_physics_result(reviewed):
    manifest, _ = reviewed
    (out,) = build_reviewed(manifest)
    write_json(out / "physics_smoke.json", synthetic_report(out))
    path = out / "asset_geometry.usda"
    path.write_text(path.read_text() + "\n# Modified after the test\n")
    with pytest.raises(ValueError, match="stale"):
        verify_physics_report(out)
    assert not json.loads((out / "asset.json").read_text())["isaac_physics_tested"]


def test_recipe_cannot_escape_or_duplicate_workspaces(reviewed):
    manifest, _ = reviewed
    data = json.loads(manifest.read_text())
    data["jobs"] *= 2
    write_json(manifest, data)
    with pytest.raises(ValueError, match="unique"):
        build_reviewed(manifest)
    data["jobs"] = data["jobs"][:1]
    data["jobs"][0]["workspace"] = "../"
    write_json(manifest, data)
    with pytest.raises(ValueError, match="within"):
        build_reviewed(manifest)
    with pytest.raises(ValueError, match="simple directory"):
        build_reviewed(manifest, "../../escape")


def test_output_directory_cannot_redirect_writes_through_a_symlink(reviewed):
    manifest, workspace = reviewed
    alternate = workspace / "alternate"
    alternate.mkdir()
    (workspace / "final").symlink_to(alternate, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic links"):
        build_reviewed(manifest)
    assert not list(alternate.iterdir())
