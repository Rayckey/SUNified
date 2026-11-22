"""Helpers for extracting python dicts from LLM outputs."""

from __future__ import annotations

import ast
import json
import re


def extract_fenced_block(text: str, fence_lang: str) -> str:
    pattern = rf"```{re.escape(fence_lang)}(.*?)```"
    result = re.findall(pattern, text, re.DOTALL)
    if not result:
        return ""
    return result[-1].lstrip("\n")


def parse_python_dict(text: str) -> dict:
    """Parse a dict from LLM output (python fenced / JSON / literal)."""
    code = extract_fenced_block(text, "python")
    try:
        c = (code.strip() if code else text.strip())
        if not c:
            return {}
        # Try JSON first (Kuafu prompts often request pure JSON).
        try:
            j = json.loads(c)
            if isinstance(j, dict):
                if "Weights" in j and isinstance(j.get("Weights"), dict):
                    return j["Weights"]
                if "weights" in j and isinstance(j.get("weights"), dict):
                    return j["weights"]
                return j
        except json.JSONDecodeError as json_exc:
            # Fall through to Python literal parsing below.
            json_error: json.JSONDecodeError | None = json_exc
        else:
            json_error = None
        # allow `weights = {...}` or just `{...}`
        if "weights" in c and "=" in c:
            rhs = c.split("=", 1)[1].strip()
            parsed = ast.literal_eval(rhs)
        else:
            parsed = ast.literal_eval(c)
        return parsed if isinstance(parsed, dict) else {}
    except Exception as exc:
        raise ValueError(f"Failed to parse Python dict from LLM output: {text[:300]}") from exc


def parse_nested_stage_dict(text: str) -> dict:
    """Parse nested {stage_id: {idx: weight}} from output (python fenced / JSON / literal)."""
    code = extract_fenced_block(text, "python")
    try:
        c = (code.strip() if code else text.strip())
        if not c:
            return {}
        try:
            j = json.loads(c)
            return j if isinstance(j, dict) else {}
        except json.JSONDecodeError as json_exc:
            # Fall through to Python literal parsing below.
            json_error: json.JSONDecodeError | None = json_exc
        else:
            json_error = None
        parsed = ast.literal_eval(c)
        return parsed if isinstance(parsed, dict) else {}
    except Exception as exc:
        raise ValueError(f"Failed to parse nested stage dict from LLM output: {text[:300]}") from exc
