"""Remove structured settings and recognized annotations from outgoing model text.

Only prompt copies are filtered. Stored programs, geometry, runtime parameters,
and model responses returned to the application are not modified. Prose cleanup
is best-effort: unfamiliar wording never prevents a model request.
"""

from __future__ import annotations

from copy import deepcopy
import ast
import json
import re
from typing import Any


_MENTION = re.compile(r"toleran\w*|torleran\w*|tolleran\w*|completion[ _-]+(?:thresholds?|bounds?)", re.I)
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_MEASURE = rf"{_NUMBER}\s*(?:radians?|rad|degrees?|deg|meters?|metres?|cm|mm|m)\b"
_QUALIFIER = r"(?:a|an|the|their|stated|specified|fixed(?:-pose|-position|-orientation)?|small|tight|normal|tracking|sensing|controller|geometric|completion|Cartesian|translational|angular|position(?:al)?|orientation|pose|joint-state|contact(?:-grasp|-position)?|reach|support|placement|and|or)"
_LABEL = rf"(?:(?:{_QUALIFIER})\s+)*tolerances?"
_VALUE = rf"\s*(?:(?:of|is|=|:|no greater than)\s*)?(?:exactly\s+)?{_MEASURE}(?:\s*\(approximately\s+{_MEASURE}\))?"
_ANNOTATION = re.compile(
    rf"\s*\(\s*(?:±\s*{_MEASURE}|{_LABEL}{_VALUE}|within\s+(?:{_MEASURE}\s+)?{_LABEL})\s*\)", re.I
)
_QUANTIFIED = re.compile(
    rf"\s*\b(?:to\s+)?(?:within|together with|with|and using|using)\s+(?:(?:a|the)\s+)?(?:{_MEASURE}\s+{_LABEL}|{_LABEL}{_VALUE})", re.I
)
_RELATIVE = re.compile(rf"\bwithin\s+{_LABEL}\s+of\s+", re.I)
_TRACKING = re.compile(rf"\s*\bexcept for\s+{_LABEL}\b", re.I)
_WITHIN = re.compile(rf"\bwithin\s+{_LABEL}\b", re.I)
_BEYOND = re.compile(rf"\s*\bbeyond\s+{_LABEL}\b", re.I)
_VIOLATED = re.compile(rf"\b(?:the\s+)?fixed[- ](pose|position|orientation)\s+tolerance\s+is\s+(?:exceeded|violated)\b", re.I)


def _prose_for_prompt(text: str) -> str:
    if not _MENTION.search(text) and not re.search(rf"±\s*{_MEASURE}", text):
        return text
    text = _ANNOTATION.sub("", text)
    text = _QUANTIFIED.sub("", text)
    text = _RELATIVE.sub("", text)
    text = _TRACKING.sub("", text)
    text = _VIOLATED.sub(lambda match: f"the {match[1]} is no longer fixed", text)
    text = _WITHIN.sub("at the intended state", text)
    text = _BEYOND.sub("", text)
    # Preserve the required pose/alignment predicate, without its old numeric
    # acceptance-band vocabulary (e.g. "orientation ... remains satisfied").
    text = re.sub(r"\btolerance(s?)\b(?=\s+remain(?:s)?\s+satisfied)", r"requirement\1", text, flags=re.I)
    text = re.sub(r"\balignment\s+tolerances\b", "alignment requirements", text, flags=re.I)
    text = re.sub(
        r"\b(orientation|position|placement|contact target)\s+tolerance(s?)\b(?=\s+(?:and|at|(?:is|are)\s+(?:maintained|reached))\b)",
        r"\1 requirement\2", text, flags=re.I,
    )
    text = re.sub(rf"\bexceeds\s+{_LABEL}\b", "deviates from the required state", text, flags=re.I)
    # Traceback implementation details are not part of the geometric error.
    text = "\n".join(line for line in text.split("\n") if not (
        _MENTION.search(line) and (line.lstrip().startswith('File "') or line.lstrip().startswith('raise '))
    ))
    # Preserve unfamiliar prose: it may contain a geometric requirement.
    # Numerical threshold ownership is enforced by the constraint receiver,
    # so incidental wording must not abort formulation or correction.
    return text


def _json_for_prompt(value: Any) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if _MENTION.search(str(key)):
                continue
            # Free-form discussion of receiver settings is not needed to
            # correct the geometry or choose weights. Preserve the actual IR.
            if str(key).lower() in {"reasoning", "diagnosis"} and isinstance(item, str) and _MENTION.search(item):
                continue
            cleaned = _json_for_prompt(item)
            if str(key).lower() == "parameters" and not _has_leaf(cleaned):
                continue
            result[key] = cleaned
        return result
    if isinstance(value, list):
        return [_json_for_prompt(item) for item in value]
    return _prose_for_prompt(value) if isinstance(value, str) else value


def _has_leaf(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_has_leaf(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_leaf(item) for item in value)
    return True


def _literal_mapping(text: str, start: int) -> tuple[dict, int]:
    """Read Python-dict reprs in compiler errors without executing anything."""
    depth = 0
    quote = None
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = ast.literal_eval(text[start:index + 1])
                except (ValueError, SyntaxError):
                    break
                if isinstance(value, dict):
                    return value, index + 1 - start
                break
    raise ValueError("Not a literal mapping")


def text_for_model(text: str) -> str:
    """Filter prose and embedded JSON without changing geometric JSON values."""
    if not _MENTION.search(text) and not re.search(rf"±\s*{_MEASURE}", text):
        return text
    decoder = json.JSONDecoder()
    parts: list[str] = []
    cursor = 0
    # Prompt builders embed JSON after labels or inside code fences. Decode
    # those objects before touching prose, so vectors, indices and targets keep
    # their types and values and field removal never leaves malformed JSON.
    for match in re.finditer(r"\{", text):
        if match.start() < cursor:
            continue
        try:
            value, length = decoder.raw_decode(text[match.start():])
        except ValueError:
            try:
                value, length = _literal_mapping(text, match.start())
            except ValueError:
                continue
        parts.append(_prose_for_prompt(text[cursor:match.start()]))
        cleaned = _json_for_prompt(value)
        original = text[match.start():match.start() + length]
        parts.append(original if cleaned == value else json.dumps(cleaned, ensure_ascii=False))
        cursor = match.start() + length
    parts.append(_prose_for_prompt(text[cursor:]))
    return "".join(parts)


def content_for_model(content: Any) -> Any:
    """Filter chat text blocks; preserve image URLs/bytes and other attachments."""
    if isinstance(content, str):
        return text_for_model(content)
    copied = deepcopy(content)
    if isinstance(copied, list):
        for index, block in enumerate(copied):
            if isinstance(block, str):
                copied[index] = text_for_model(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                block["text"] = text_for_model(block["text"])
            elif isinstance(block, dict) and "role" in block and "content" in block:
                block["content"] = content_for_model(block["content"])
    return copied
