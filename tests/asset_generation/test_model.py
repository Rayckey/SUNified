import json

import httpx
import pytest
from openai import OpenAI
from PIL import Image

from kuafu.asset_generation.model import ModelOutputError, OpenAIModel
from kuafu.asset_generation.schema import RunConfig, SceneSpec


def test_actual_sdk_serialization_and_cache_without_paid_requests(tmp_path, mug):
    image = tmp_path / "frame.jpg"
    Image.new("RGB", (32, 32), "blue").save(image)
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        assert body["model"] == "gpt-6-astra"
        assert body["store"] is False
        assert body["text"]["format"]["type"] == "json_schema"
        assert body["text"]["format"]["strict"] is True
        assert body["input"][0]["content"][-1]["image_url"].startswith("data:image/jpeg;base64,")
        return httpx.Response(
            200,
            json={
                "id": "resp_test",
                "object": "response",
                "created_at": 1,
                "model": "gpt-6-astra",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_test",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {"type": "output_text", "text": mug.model_dump_json(), "annotations": []}
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 100,
                    "total_tokens": 200,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            },
        )

    client = OpenAI(
        api_key="test-placeholder", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    model = OpenAIModel(RunConfig(), tmp_path / "cache", client)
    first = model.complete(SceneSpec, "instructions", "mug", [image])
    second = model.complete(SceneSpec, "instructions", "mug", [image])
    assert first == second == mug
    assert model.api_calls == 1 and model.cache_hits == 1
    model.complete(SceneSpec, "instructions", "different description", [image])
    assert len(requests) == 2
    for path in (tmp_path / "cache").glob("*.json"):
        assert "test-placeholder" not in path.read_text()
        assert "base64," not in path.read_text()
    client.close()


def test_missing_credentials_explains_offline_options(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="demo and build-scene work offline"):
        OpenAIModel(RunConfig(), tmp_path).complete(SceneSpec, "instructions", "mug", [])


def test_refusal_is_not_cached_as_success(tmp_path):
    def handler(request):
        return httpx.Response(
            200,
            json={
                "id": "resp_test",
                "object": "response",
                "created_at": 1,
                "model": "gpt-6-astra",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_test",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "refusal", "refusal": "test refusal"}],
                    }
                ],
            },
        )

    client = OpenAI(
        api_key="test-placeholder", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    model = OpenAIModel(RunConfig(), tmp_path / "cache", client)
    with pytest.raises(ModelOutputError):
        model.complete(SceneSpec, "instructions", "mug", [])
    record = json.loads(next((tmp_path / "cache").glob("*.json")).read_text())
    assert record["status"] == "invalid"
    client.close()


def test_malformed_sdk_output_is_recorded_as_invalid_not_pending(tmp_path):
    def handler(request):
        return httpx.Response(
            200,
            json={
                "id": "resp_invalid",
                "object": "response",
                "created_at": 1,
                "model": "gpt-6-astra",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_invalid",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": '{"score": ', "annotations": []}],
                    }
                ],
            },
        )

    client = OpenAI(
        api_key="test-placeholder", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    model = OpenAIModel(RunConfig(), tmp_path / "cache", client)
    with pytest.raises(ModelOutputError, match="malformed structured output"):
        model.complete(SceneSpec, "instructions", "cabinet", [])
    record = json.loads(next((tmp_path / "cache").glob("*.json")).read_text())
    assert record["status"] == "invalid" and "Invalid JSON" in record["error"]
    assert model.api_calls == 1 and model.cache_hits == 0
    client.close()
