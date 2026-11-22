"""Canonical filesystem locations for the unified project.

All modules use these constants instead of inferring the locations of former
sibling repositories from their own file depth.
"""

from __future__ import annotations

from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
PACKAGE_ROOT = SOURCE_ROOT / "kuafu"
DEFAULT_EXTERNAL_ARTIFACT_ROOT = PROJECT_ROOT.parent / "KuafuArtifacts"

CONTROL_ROOT = PACKAGE_ROOT / "mpc"
ASSET_GENERATION_ROOT = PACKAGE_ROOT / "asset_generation"
FORMULATION_ROOT = PACKAGE_ROOT / "programs"
ENVIRONMENT_ROOT = PACKAGE_ROOT / "isaac"
DATA_ROOT = PACKAGE_ROOT / "data"
PRIOR_ROOT = PACKAGE_ROOT / "bc"
RESIDUAL_RL_ROOT = PACKAGE_ROOT / "rl"

RESOURCE_ROOT = PROJECT_ROOT / "resources"
ROBOT_RESOURCE_ROOT = RESOURCE_ROOT / "robots"
ASSET_ROOT = PROJECT_ROOT / "assets"
PROMPT_ROOT = ASSET_ROOT / "prompts"
SKILL_ROOT = ASSET_ROOT / "skills"
TASK_ROOT = ASSET_ROOT / "tasks"
OBJECT_ROOT = ASSET_ROOT / "objects"
ROBOT_ASSET_ROOT = ASSET_ROOT / "robots"
SKILLS_ROOT = PACKAGE_ROOT / "skills"
LEARNED_PROGRAM_ROOT = RESOURCE_ROOT / "learned_programs"
EXPERIMENT_DEFINITION_ROOT = RESOURCE_ROOT / "experiment_definitions"

CONFIG_ROOT = PROJECT_ROOT / "assets/configs"


def project_root() -> Path:
    """Return the absolute project root."""

    return PROJECT_ROOT


def require_external_artifact_path(path: str | Path, label: str = "Artifact path") -> Path:
    """Resolve an artifact path and reject writes beneath the source project."""

    resolved = Path(path).expanduser().resolve()
    if resolved == PROJECT_ROOT or PROJECT_ROOT in resolved.parents:
        raise ValueError(f"{label} must be outside the source project: {resolved}")
    return resolved


def external_artifact_root() -> Path:
    """Return the configured external artifact root, or the sibling default."""

    configured = os.environ.get("KUAFU_ARTIFACT_ROOT", "").strip()
    root = Path(configured) if configured else DEFAULT_EXTERNAL_ARTIFACT_ROOT
    return require_external_artifact_path(root, "KUAFU artifact root")


def resolve_artifact_path(path: str | Path, label: str = "Artifact path") -> Path:
    """Resolve a relative path beneath the external artifact root."""

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = external_artifact_root() / candidate
    return require_external_artifact_path(candidate, label)



def program_output_root() -> Path:
    """Store new programs alongside their pipeline run or standalone artifacts."""
    configured = os.environ.get("KUAFU_PROGRAM_OUTPUT_ROOT")
    return require_external_artifact_path(configured) if configured else external_artifact_root() / "programs"
