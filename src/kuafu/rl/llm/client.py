"""OpenAI chat client wrapper with lightweight message stack management."""

from __future__ import annotations

import base64
import mimetypes
import os
from dataclasses import dataclass, field
from typing import Any
from kuafu.common.prompt_context import content_for_model, text_for_model

ChatMessage = dict[str, Any]


@dataclass
class LlmResponse:
    messages: list[ChatMessage]
    raw_outputs: list[str]
    usage: dict[str, Any] = field(default_factory=dict)


def _image_data_url(image_path: str) -> str:
    mime_type, _encoding = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/png"
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _sanitize_message_for_log(message: ChatMessage) -> ChatMessage:
    sanitized = dict(message)
    content = sanitized.get("content")
    if not isinstance(content, list):
        return sanitized

    sanitized_parts: list[Any] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "image_url":
            sanitized_part = dict(part)
            image_url = sanitized_part.get("image_url")
            clean_image_url = dict(image_url) if isinstance(image_url, dict) else {}
            url = clean_image_url.get("url")
            if isinstance(url, str) and url.startswith("data:"):
                clean_image_url["url"] = "<attached image omitted from log>"
            sanitized_part["image_url"] = clean_image_url
            sanitized_parts.append(sanitized_part)
        else:
            sanitized_parts.append(part)
    sanitized["content"] = sanitized_parts
    return sanitized


class OpenAIChatClient:
    """Stateful chat client that keeps a short conversation window."""

    def __init__(
        self, *, model: str, temperature: float, num_suggestions: int,
        system_prompt: str, reasoning_effort: str | None = None,
    ):
        self._model = model
        self._temperature = float(temperature)
        self._num_suggestions = int(num_suggestions)
        self._reasoning_effort = (
            reasoning_effort if reasoning_effort is not None else os.environ.get("KUAFU_LLM_REASONING_EFFORT", "")
        ).strip() or None
        self._prompts: list[ChatMessage] = [{"role": "system", "content": text_for_model(system_prompt)}]

        try:
            import openai  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Missing dependency `openai`. Install it (in your conda env) via: `pip install openai`."
            ) from e

        if "AZURE_OPENAI_API_KEY" in os.environ:
            self._client = openai.AzureOpenAI(api_version="2024-02-01")
        elif "OPENAI_API_KEY" in os.environ:
            self._client = openai.OpenAI()
        else:
            raise RuntimeError("No OpenAI API key found in environment variables")

    def prompt(
        self,
        *,
        user_prompt: str,
        assistant_prompt: str | None = None,
        image_paths: list[str] | None = None,
    ) -> LlmResponse:
        user_prompt = text_for_model(user_prompt)
        if assistant_prompt is not None:
            self._prompts.append({"role": "assistant", "content": text_for_model(assistant_prompt)})
        self._prompts.append({"role": "user", "content": user_prompt})

        # Retain the system instruction and latest exchange when trimming history.
        if len(self._prompts) > 5:
            # Remove the oldest assistant+user pair after the system prompt.
            # prompts layout: [system, assistant, user, assistant, user, ...]
            self._prompts.pop(1)
            self._prompts.pop(1)

        messages_to_send = content_for_model(self._prompts)
        if image_paths:
            content_parts: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
            for image_path in image_paths:
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_data_url(image_path), "detail": "low"},
                    }
                )
            messages_to_send[-1] = {"role": "user", "content": content_parts}

        request: dict[str, Any] = {
            "model": self._model,
            "messages": messages_to_send,
            "n": self._num_suggestions,
        }
        # Match the formulation client: GPT-5 models use their default sampling
        # settings and reject the non-default temperature used by init sweeps.
        if not self._model.lower().startswith("gpt-5"):
            request["temperature"] = self._temperature
        if self._reasoning_effort:
            request["reasoning_effort"] = self._reasoning_effort

        try:
            resp = self._client.chat.completions.create(**request)
        except Exception as e:
            raise RuntimeError("An error occurred while prompting the LLM") from e

        raw_outputs = [c.message.content or "" for c in resp.choices]
        logged_messages = [_sanitize_message_for_log(m) for m in messages_to_send]
        usage_obj = getattr(resp, "usage", None)
        usage = usage_obj.model_dump(exclude_none=True) if usage_obj is not None else {}
        return LlmResponse(messages=logged_messages, raw_outputs=raw_outputs, usage=usage)
