#!/usr/bin/env python3
import datetime
import json
import logging
import os
import time
import uuid
from pathlib import Path

from openai import OpenAI
try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

from kuafu.paths import external_artifact_root, resolve_artifact_path
logger = logging.getLogger(__name__)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if DEEPSEEK_API_KEY is None:
    logger.warning(
        "No DEEPSEEK_API_KEY found in environment variables. "
        "DeepSeek models will be unavailable."
    )
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if os.getenv("OPENAI_API_KEY") is None:
    logger.warning(
        "No OPENAI_API_KEY found in environment variables. " "Searching for text file"
    )
    try:
        OPENAI_API_KEY = Path("OPENAI_API_KEY").read_text(encoding="utf-8").strip()
    except OSError:
        OPENAI_API_KEY = None
else:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class PromptManager:
    def __init__(
        self,
        model_version="gpt-5.5",
        use_buffer=None,
        temperature=0.1,
        max_tokens=10000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    ) -> None:
        self.prompts = []
        self.api_called_times = -1
        self.model_version = model_version
        self.provider = self._infer_provider(model_version)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.responses = []
        self.total_api_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.last_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.deepseek_client = None
        self.gpt_client = None
        self.claude_client = None
        if self.provider == "offline":
            pass
        elif self.provider == "deepseek":
            if not DEEPSEEK_API_KEY:
                raise RuntimeError(
                    "A DeepSeek model was requested but DEEPSEEK_API_KEY "
                    "is not set in the environment."
                )
            self.deepseek_client = OpenAI(
                api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com"
            )
        elif self.provider == "claude":
            if Anthropic is None:
                raise RuntimeError(
                    "A Claude model was requested but the 'anthropic' package is not installed."
                )
            if not ANTHROPIC_API_KEY:
                raise RuntimeError(
                    "A Claude model was requested but ANTHROPIC_API_KEY is not set in the environment."
                )
            self.claude_client = Anthropic(api_key=ANTHROPIC_API_KEY)
        else:
            if not OPENAI_API_KEY:
                raise RuntimeError(
                    "An OpenAI model was requested but OPENAI_API_KEY is not set in the environment."
                )
            self.gpt_client = OpenAI(api_key=OPENAI_API_KEY)
        self.resetBufferPath(use_buffer)

    @staticmethod
    def _infer_provider(model_version):
        model_name = str(model_version).strip().lower()
        if model_name in {"offline", "local", "none", "noop"}:
            return "offline"
        if "deepseek" in model_name:
            return "deepseek"
        if "claude" in model_name:
            return "claude"
        return "openai"

    @staticmethod
    def _is_json_response(response_format):
        return response_format.get("type") == "json_object"

    @staticmethod
    def _strip_code_fence(text):
        stripped = text.strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            lines = stripped.splitlines()
            if len(lines) >= 3:
                return "\n".join(lines[1:-1]).strip()
        return stripped

    def _extract_system_message(self):
        system_messages = [
            prompt["content"]
            for prompt in self.prompts
            if prompt.get("role") == "system"
        ]
        if not system_messages:
            return None
        return "\n\n".join(str(message) for message in system_messages)

    @staticmethod
    def _convert_image_url_to_anthropic(block):
        url = block["image_url"]["url"]
        if not url.startswith("data:"):
            raise ValueError("Claude image support requires base64 data URLs.")
        header, data = url.split(",", 1)
        media_type = header.split(";")[0].split(":", 1)[1]
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": data,
            },
        }

    def _convert_content_to_anthropic(self, content):
        if isinstance(content, str):
            return content

        converted = []
        for block in content:
            block_type = block.get("type")
            if block_type == "text":
                converted.append({"type": "text", "text": block["text"]})
            elif block_type == "image_url":
                converted.append(self._convert_image_url_to_anthropic(block))
            else:
                raise ValueError(f"Unsupported Claude content block type: {block_type}")
        return converted

    def _build_anthropic_messages(self):
        messages = []
        for prompt in self.prompts:
            role = prompt.get("role")
            if role == "system":
                continue
            if role not in {"user", "assistant"}:
                raise ValueError(f"Unsupported Claude role: {role}")
            messages.append(
                {
                    "role": role,
                    "content": self._convert_content_to_anthropic(prompt["content"]),
                }
            )
        return messages

    def _extract_anthropic_text(self, response, response_format):
        text_parts = []
        expect_json = self._is_json_response(response_format)
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type is None and isinstance(block, dict):
                block_type = block.get("type")

            if expect_json and block_type == "tool_use":
                tool_input = getattr(block, "input", None)
                if tool_input is None and isinstance(block, dict):
                    tool_input = block.get("input")
                return json.dumps(tool_input)

            if block_type == "text":
                text = getattr(block, "text", None)
                if text is None and isinstance(block, dict):
                    text = block.get("text")
                if text:
                    text_parts.append(text)

        text_response = self._strip_code_fence("".join(text_parts))
        if expect_json and text_response:
            # Fallback when Claude returns raw JSON text instead of a tool call.
            json.loads(text_response)
        return text_response

    @staticmethod
    def _usage_get(usage, *names):
        for name in names:
            if usage is None:
                return None
            if isinstance(usage, dict):
                value = usage.get(name)
            else:
                value = getattr(usage, name, None)
            if value is not None:
                return int(value)
        return None

    def _record_usage(self, response):
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")

        prompt_tokens = self._usage_get(usage, "prompt_tokens", "input_tokens") or 0
        completion_tokens = (
            self._usage_get(usage, "completion_tokens", "output_tokens") or 0
        )
        total_tokens = self._usage_get(usage, "total_tokens")
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens

        self.total_api_calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += total_tokens
        self.last_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
        if total_tokens:
            print(
                "LLM token usage: "
                f"prompt={prompt_tokens}, completion={completion_tokens}, "
                f"total={total_tokens}",
                flush=True,
            )

    def usage_summary(self):
        return {
            "calls": self.total_api_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

    def _call_openai_compatible(self, response_format):
        if self.provider == "deepseek":
            response = self.deepseek_client.chat.completions.create(
                model=self.model_version,
                messages=self.prompts,
                stream=False,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                response_format=response_format,
            )
        elif "4o" in self.model_version:
            response = self.gpt_client.chat.completions.create(
                model=self.model_version,
                messages=self.prompts,
                stream=False,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                response_format=response_format,
            )
            logger.info(str(response))
        else:
            # GPT-5 style chat models reject non-default sampling values.
            # Omit sampling parameters so the API uses its model-required defaults.
            response = self.gpt_client.chat.completions.create(
                model=self.model_version,
                messages=self.prompts,
                max_completion_tokens=self.max_tokens,
                response_format=response_format,
            )
        return response, response.choices[0].message.content

    def _call_claude(self, response_format):
        request = {
            "model": self.model_version,
            "messages": self._build_anthropic_messages(),
            "max_tokens": self.max_tokens,
        }
        system_message = self._extract_system_message()
        if system_message:
            request["system"] = system_message
        request["temperature"] = self.temperature
        if self._is_json_response(response_format):
            request["tools"] = [
                {
                    "name": "return_json",
                    "description": "Return the final answer as a JSON object.",
                    "input_schema": {
                        "type": "object",
                        "additionalProperties": True,
                    },
                }
            ]
            request["tool_choice"] = {"type": "tool", "name": "return_json"}

        response = self.claude_client.messages.create(**request)
        return response, self._extract_anthropic_text(response, response_format)

    def _call_model(self, response_format):
        if self.provider == "offline":
            raise RuntimeError("PromptManager is in offline mode; no LLM calls are available.")
        if self.provider == "claude":
            return self._call_claude(response_format)
        return self._call_openai_compatible(response_format)

    def resetBufferPath(self, buffer_name):
        self.use_buffer = buffer_name
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.transcript_path = str(
            external_artifact_root() / "llm" / "transcripts" / f"{timestamp}_{uuid.uuid4().hex}.md"
        )
        self.file_path = None if buffer_name is None else str(resolve_artifact_path(
            Path("llm/responses") / buffer_name, "LLM response buffer",
        ))
        self._transcript_initialized = False

    def setSystemMessage(self, content):
        if len(self.prompts) == 0:
            self.prompts = [
                {"role": "system", "content": content},
            ]
        else:
            self.prompts[0] = {"role": "system", "content": content}
        self._append_transcript("system", content)

    def addQuery(self, content):
        new_query = {"role": "user", "content": content}
        self.prompts.append(new_query)

    def addResponse(self, content):
        new_response = {"role": "assistant", "content": content}
        self.prompts.append(new_response)

    def callGPT(self, content=None, response_format_str=None):
        if content is None:
            raise ("Called VLM with no content")
        elif isinstance(content, list):
            self.addQuery(content)

        if response_format_str is None:
            response_format = {"type": "text"}
        else:
            response_format = {"type": response_format_str}

        if self.use_buffer is None:
            self._saveGPTqueries(content[0]['text'])
            start = time.time()
            try:
                response, response_content = self._call_model(response_format)
                self._record_usage(response)
                self.responses.append(response)
                self.prompts.append(
                    {
                        "role": "assistant",
                        "content": response_content,
                    }
                )

                self._saveGPTresponses(self.prompts[-1]["content"])
                end = time.time()
                print("LLM call completed in", end - start)
            except Exception as e:
                logger.exception("LLM error during single-shot call")
                raise RuntimeError(f"LLM call failed: {e}") from e
        else:
            try:
                response = self._loadGPTresponses()
            except Exception as e:
                logger.info(f"[ERROR] buffer error: {e}")
                return None
            self.responses.append(response)
            self.prompts.append(
                {
                    "role": "assistant",
                    "content": response,
                }
            )

        self.api_called_times += 1

        return self.prompts[-1]["content"]

    def callGPTMultiShot(self, content=list[str], response_format_str=None):
        if content is None:
            raise ("Called VLM with no content")
        idx = 0
        while idx < len(content):
            if idx % 2 == 0:
                self.addQuery(content[idx])
            else:
                self.addResponse(content[idx])
            idx += 1

        if response_format_str is None:
            response_format = {"type": "text"}
        else:
            response_format = {"type": response_format_str}


        self._saveGPTqueries(content[-1])
        start = time.time()
        try:
            response, response_content = self._call_model(response_format)
            self._record_usage(response)
            self.responses.append(response)
            self.prompts.append(
                {
                    "role": "assistant",
                    "content": response_content,
                }
            )

            self._saveGPTresponses(self.prompts[-1]["content"])
            end = time.time()
            print(f"LLM multi-shot call completed in {end - start}s")
        except Exception as e:
            logger.exception("LLM error during multi-shot call")
            raise RuntimeError(f"LLM multi-shot call failed: {e}") from e

        self.api_called_times += 1

        return self.prompts[-1]["content"]


    def getLastResponse(self):
        last_response = self.responses[-1]
        if self.provider == "claude":
            return self._extract_anthropic_text(last_response, {"type": "text"})
        return last_response["choices"][0]["message"]["content"]

    def insertResponse(self, response):
        self.responses.append(response)
        self.prompts.append(
            {
                "role": "assistant",
                "content": response,
            }
        )

    def dropPastPrompt(self):
        self.prompts.remove(self.prompts[1])

    def dropLastPrompt(self):
        self.prompts.remove(self.prompts[-1])

    def dropPastResponse(self):
        self.responses.remove(self.responses[1])

    def dropLastResponse(self):
        self.responses.remove(self.responses[-1])

    def popPastPrompt(self):
        return self.prompts.pop(1)

    def popLastPrompt(self):
        return self.prompts.pop(-1)

    def popPastResponse(self):
        return self.responses.pop(1)

    def popLastResponse(self):
        return self.responses.pop(-1)

    def dropPastConversation(self, num_con=1):
        for _ in range(num_con * 2):
            self.dropPastPrompt()
        for _ in range(num_con):
            self.dropPastResponse()

    def dropLastConversation(self, num_con=1):
        for _ in range(num_con * 2):
            self.dropLastPrompt()
        for _ in range(num_con):
            self.dropLastResponse()

    def popPastConversation(self, num_con=1):
        prompts = []
        responses = []
        for _ in range(num_con * 2):
            prompts.append(self.popPastPrompt())
        for _ in range(num_con):
            responses.append(self.popPastResponse())
        return prompts, responses

    def popLastConversation(self, num_con=1):
        prompts = []
        responses = []
        for _ in range(num_con * 2):
            prompts.append(self.popLastPrompt())
        for _ in range(num_con):
            responses.append(self.popLastResponse())
        prompts.reverse()
        responses.reverse()
        return prompts, responses

    def dropAllConversation(self):
        self.prompts = []
        self.responses = []

    def _append_transcript(self, role: str, content: str):
        Path(self.transcript_path).parent.mkdir(parents=True, exist_ok=True)
        turn = self.api_called_times + 1
        with open(self.transcript_path, "a", encoding="utf-8") as f:
            if not self._transcript_initialized:
                f.write(
                    f"# LLM Conversation Transcript\n\n"
                    f"- **Model**: `{self.model_version}`\n"
                    f"- **Started**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                    f"---\n\n"
                )
                self._transcript_initialized = True
            if role == "system":
                f.write(f"## System\n\n{content}\n\n---\n\n")
            else:
                label = "User" if role == "user" else "Assistant"
                f.write(f"## Turn {turn} — {label}\n\n{content}\n\n---\n\n")

    def _saveGPTresponses(self, response):
        self._append_transcript("assistant", response)
        logger.info(response)

    def _saveGPTqueries(self, query):
        self._append_transcript("user", query)
        logger.info(query)

    def _loadGPTresponses(self):
        with open(
            os.path.join(self.file_path, f"gpt_res_{self.api_called_times}.txt"), "r"
        ) as f:
            response = f.read()
        logger.info("Used previous response.")
        return response
