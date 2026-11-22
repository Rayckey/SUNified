"""Replaceable vision-model interface with content-addressed API response caching."""

import base64
import hashlib
import json
import mimetypes
import os
from pathlib import Path
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from .io import digest, write_json
from .schema import RunConfig

T = TypeVar("T", bound=BaseModel)


class ModelOutputError(ValueError):
    pass


class VisionModel(Protocol):
    def complete(self, schema: type[T], instructions: str, prompt: str, images: list[Path]) -> T: ...


class OpenAIModel:
    def __init__(self, config: RunConfig, cache: Path, client=None):
        from kuafu.paths import resolve_artifact_path

        cache = resolve_artifact_path(cache, "Asset reconstruction response cache")
        self.config, self.cache, self.client = config, cache, client
        self.api_calls = 0
        self.cache_hits = 0

    def complete(self, schema: type[T], instructions: str, prompt: str, images: list[Path]) -> T:
        request = {
            "version": 1,
            "model": self.config.model,
            "instructions": instructions,
            "prompt": prompt,
            "schema": schema.model_json_schema(),
            "max_output_tokens": self.config.max_output_tokens,
            "reasoning_effort": self.config.reasoning_effort,
            "images": [{"name": path.name, "sha256": digest(path)} for path in images],
        }
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        path = self.cache / f"{key}.json"
        if path.exists():
            cached = json.loads(path.read_text())
            if cached.get("status") == "completed":
                result = schema.model_validate(cached["parsed"])
                self.cache_hits += 1
                return result
        if self.client is None:
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError(
                    "Set OPENAI_API_KEY to run the vision agent. The demo and build-scene work offline."
                )
            from openai import OpenAI

            self.client = OpenAI(timeout=self.config.timeout_s, max_retries=2)
        content = [{"type": "input_text", "text": prompt}]
        for index, image in enumerate(images, start=1):
            mime = mimetypes.guess_type(image.name)[0] or "image/png"
            encoded = base64.b64encode(image.read_bytes()).decode("ascii")
            content.extend(
                [
                    {"type": "input_text", "text": f"Image {index}: {image.name}"},
                    {"type": "input_image", "image_url": f"data:{mime};base64,{encoded}", "detail": "high"},
                ]
            )
        self.api_calls += 1
        write_json(path, {"status": "requested", "request": request})
        try:
            response = self.client.responses.parse(
                model=self.config.model,
                instructions=instructions,
                input=[{"role": "user", "content": content}],
                text_format=schema,
                reasoning={"effort": self.config.reasoning_effort},
                max_output_tokens=self.config.max_output_tokens,
                store=False,
            )
        except ValidationError as error:
            write_json(path, {"status": "invalid", "request": request, "error": str(error)})
            raise ModelOutputError("Provider returned malformed structured output; see cache") from error
        record = {
            "status": response.status,
            "request": request,
            "response_id": response.id,
            "usage": response.usage.model_dump() if response.usage else None,
            "output_text": response.output_text,
            "parsed": response.output_parsed.model_dump() if response.output_parsed else None,
        }
        if response.status != "completed" or response.output_parsed is None:
            record["status"] = "invalid"
            record["output"] = [item.model_dump() for item in response.output]
            write_json(path, record)
            raise ModelOutputError(
                "Model returned a refusal, incomplete response, or no structured output; see cache"
            )
        write_json(path, record)
        return response.output_parsed
