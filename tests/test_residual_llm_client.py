import sys
from types import SimpleNamespace

import pytest

from kuafu.rl.llm.client import OpenAIChatClient


@pytest.mark.parametrize("model, expected_temperature", [("gpt-5.5", None), ("other-model", 0.2)])
def test_usage_and_reasoning_preserve_request_defaults(monkeypatch, model, expected_temperature):
    calls = []
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="result"))],
        usage=SimpleNamespace(model_dump=lambda **_: {"prompt_tokens": 20, "completion_tokens": 5}),
    )
    def create(**kwargs):
        calls.append(kwargs)
        return response
    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda: fake))
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("KUAFU_LLM_REASONING_EFFORT", "medium")
    client = OpenAIChatClient(model=model, temperature=0.2, num_suggestions=1, system_prompt="test", reasoning_effort="high")
    result = client.prompt(user_prompt="test")
    assert calls[0]["reasoning_effort"] == "high"
    assert calls[0].get("temperature") == expected_temperature
    assert result.usage == {"prompt_tokens": 20, "completion_tokens": 5}
    assert result.raw_outputs == ["result"]
    response.usage = None
    client = OpenAIChatClient(model=model, temperature=0.2, num_suggestions=1, system_prompt="test")
    assert client.prompt(user_prompt="test").usage == {}
    assert calls[-1]["reasoning_effort"] == "medium"
    monkeypatch.delenv("KUAFU_LLM_REASONING_EFFORT")
    client = OpenAIChatClient(model=model, temperature=0.2, num_suggestions=1, system_prompt="test")
    client.prompt(user_prompt="test")
    assert "reasoning_effort" not in calls[-1]
