"""Model conversations and offline replay stay outside the source checkout."""

from pathlib import Path

import pytest

from kuafu.paths import PROJECT_ROOT
from kuafu.programs.llm.prompt_manager import PromptManager


def test_transcripts_are_external_and_separate(monkeypatch, tmp_path):
    monkeypatch.chdir(PROJECT_ROOT)
    monkeypatch.setenv("KUAFU_ARTIFACT_ROOT", str(tmp_path))
    first = PromptManager(model_version="offline")
    second = PromptManager(model_version="offline")
    first.setSystemMessage("First program")
    second.setSystemMessage("Second program")
    first._saveGPTresponses("First response")

    transcripts = list((tmp_path / "llm/transcripts").glob("*.md"))
    assert len(transcripts) == 2
    assert "First response" in Path(first.transcript_path).read_text()
    assert "First response" not in Path(second.transcript_path).read_text()


def test_named_replay_uses_external_response_buffer(monkeypatch, tmp_path):
    monkeypatch.chdir(PROJECT_ROOT)
    monkeypatch.setenv("KUAFU_ARTIFACT_ROOT", str(tmp_path))
    responses = tmp_path / "llm/responses/cabinet"
    responses.mkdir(parents=True)
    (responses / "gpt_res_-1.txt").write_text("Saved program")
    manager = PromptManager(model_version="offline", use_buffer="cabinet")
    manager.setSystemMessage("Replay the cabinet program")
    assert manager.callGPT([{"type": "text", "text": "Open the drawer"}]) == "Saved program"
    assert manager.total_api_calls == 0
    assert Path(manager.transcript_path).is_relative_to(tmp_path)


def test_transcripts_reject_artifact_root_inside_checkout(monkeypatch):
    monkeypatch.setenv("KUAFU_ARTIFACT_ROOT", str(PROJECT_ROOT / "buffer"))
    with pytest.raises(ValueError, match="outside the source project"):
        PromptManager(model_version="offline")
