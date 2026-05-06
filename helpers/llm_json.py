import json
from collections.abc import Callable
from typing import Any


def parse_llm_json_response(
    response: str,
    dirty_parser: Callable[[str], Any] | None = None,
) -> Any:
    """Parse LLM JSON without silently dropping concatenated root values.

    Agent Zero's DirtyJson parser is intentionally forgiving, but for outputs like
    ``[...] [...]`` it returns only the first root. For memory extraction that loses
    data and hides the real provider bug. This parser first handles strict JSON
    roots, then falls back to the existing dirty parser for genuinely malformed
    but single-root responses.
    """
    if not isinstance(response, str) or not response.strip():
        return None

    text = _strip_code_fence(response.strip())
    roots = _decode_strict_json_roots(text)
    if roots:
        return _collapse_roots(roots)

    if dirty_parser is not None:
        return dirty_parser(text)

    raise ValueError("Could not parse LLM JSON response")


def _strip_code_fence(text: str) -> str:
    lines = text.strip().splitlines()
    if len(lines) >= 2 and lines[0].strip().startswith(("```", "~~~")):
        fence = lines[0].strip()[:3]
        if lines[-1].strip().startswith(fence):
            return "\n".join(lines[1:-1]).strip()
    return text


def _decode_strict_json_roots(text: str) -> list[Any]:
    decoder = json.JSONDecoder()
    roots: list[Any] = []
    index = _first_json_root_index(text, 0)

    while index is not None:
        try:
            value, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            return []

        roots.append(value)
        index = _first_json_root_index(text, end)

    return roots


def _first_json_root_index(text: str, start: int) -> int | None:
    for index in range(start, len(text)):
        if text[index].isspace():
            continue
        if text[index] in "[{":
            return index
        return None
    return None


def _collapse_roots(roots: list[Any]) -> Any:
    if len(roots) == 1:
        return roots[0]

    if all(isinstance(root, list) for root in roots):
        return _merge_lists(roots)

    first = roots[0]
    if all(root == first for root in roots):
        return first

    raise ValueError("LLM response contains multiple JSON roots with incompatible values")


def _merge_lists(roots: list[list[Any]]) -> list[Any]:
    merged: list[Any] = []
    seen: set[str] = set()
    for root in roots:
        for item in root:
            key = _stable_key(item)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged


def _stable_key(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return repr(value)
