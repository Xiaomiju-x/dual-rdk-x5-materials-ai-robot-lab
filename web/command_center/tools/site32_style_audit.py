#!/usr/bin/env python3
"""Deterministic CSS debt and performance gate for the Site32 release.

The legacy sheets are frozen against a checked-in baseline embedded below.  The
Site32 sheet additionally has strict tokenisation rules that cannot be waived by
supplying a more permissive baseline.  The default mode is read-only and emits a
single JSON document to stdout.
"""
from __future__ import annotations

import argparse
import ast
import bisect
from datetime import datetime, timezone
import gzip
import importlib.util
import json
import os
import re
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = "site32.style_audit.v1"
BASELINE_SCHEMA_VERSION = "site32.style_baseline.v1"
EXIT_PASS = 0
EXIT_GATE_FAILED = 1
EXIT_INPUT_ERROR = 2

TRACKED_FILES = (
    "static/style.css",
    "static/r4.css",
    "static/site32.css",
)
SITE32_FILE = "static/site32.css"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _candidate_identity(root: Path) -> dict[str, Any]:
    """Bind repository audits to the side-effect-free candidate manifest."""
    config_path = root / "cmdcenter" / "config.py"
    manifest_tool = root / "tools" / "site31_asset_manifest.py"
    if not config_path.exists() and not manifest_tool.exists():
        return {
            "release": None,
            "manifest_digest": None,
            "identity_status": "unbound_fixture",
        }
    if not config_path.is_file() or not manifest_tool.is_file():
        raise StyleAuditError("candidate identity requires config.py and site31_asset_manifest.py")

    try:
        tree = ast.parse(config_path.read_text(encoding="utf-8-sig"), filename=str(config_path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise StyleAuditError(f"cannot read candidate release identity: {exc}") from exc
    release = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "ASSET_VER" for target in node.targets):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            release = node.value.value.strip()
        break
    if not release:
        raise StyleAuditError("candidate ASSET_VER must be a non-empty string literal")

    spec = importlib.util.spec_from_file_location(
        f"_site32_style_manifest_{id(root)}", manifest_tool
    )
    if spec is None or spec.loader is None:
        raise StyleAuditError("cannot load candidate manifest builder")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        manifest = module.build_manifest(root)
    except Exception as exc:
        raise StyleAuditError(f"cannot build candidate manifest identity: {exc}") from exc
    manifest_release = manifest.get("release") if isinstance(manifest, dict) else None
    manifest_digest = manifest.get("manifest_digest") if isinstance(manifest, dict) else None
    if manifest_release != release:
        raise StyleAuditError(
            f"candidate release mismatch: config={release!r} manifest={manifest_release!r}"
        )
    if not isinstance(manifest_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", manifest_digest):
        raise StyleAuditError("candidate manifest digest is invalid")
    return {
        "release": release,
        "manifest_digest": manifest_digest,
        "identity_status": "bound",
    }

SAMPLE_LIMIT = 12
GATED_FILE_METRICS = (
    "bytes",
    "transition_all_declarations",
    "important_occurrences",
    "backdrop_filter_rules",
    "backdrop_filter_active_rules",
    "keyframe_definitions",
    "duplicate_keyframe_definitions",
    "naked_color_literals",
    "naked_radius_declarations",
    "naked_duration_literals",
)
BASELINE_FILE_METRICS = ("bytes", "gzip_bytes", *GATED_FILE_METRICS[1:])
GATED_AGGREGATE_METRICS = (
    "duplicate_keyframe_names",
    "duplicate_keyframe_definitions",
)
SITE32_STRICT_ZERO_METRICS = (
    "transition_all_declarations",
    "important_occurrences",
    "duplicate_keyframe_definitions",
    "naked_color_literals",
    "naked_radius_declarations",
    "naked_duration_literals",
)

# The legacy ceilings are replaced once, using --write-baseline, when the gate
# is deliberately ratcheted down.  Site32 has growth budgets for its modular
# sheet; strict-zero rules above remain authoritative regardless of baseline.
BUILTIN_BASELINE: dict[str, Any] = {
    "schema_version": BASELINE_SCHEMA_VERSION,
    "policy": "site32-round1-builtin",
    "files": {
        "static/style.css": {
            "bytes": 385096,
            "gzip_bytes": 70514,
            "transition_all_declarations": 2,
            "important_occurrences": 323,
            "backdrop_filter_rules": 103,
            "backdrop_filter_active_rules": 80,
            "keyframe_definitions": 63,
            "duplicate_keyframe_definitions": 9,
            "naked_color_literals": 2405,
            "naked_radius_declarations": 372,
            "naked_duration_literals": 161,
        },
        "static/r4.css": {
            "bytes": 9041,
            "gzip_bytes": 2682,
            "transition_all_declarations": 0,
            "important_occurrences": 62,
            "backdrop_filter_rules": 9,
            "backdrop_filter_active_rules": 4,
            "keyframe_definitions": 0,
            "duplicate_keyframe_definitions": 0,
            "naked_color_literals": 6,
            "naked_radius_declarations": 1,
            "naked_duration_literals": 0,
        },
        "static/site32.css": {
            "bytes": 65536,
            "gzip_bytes": 16384,
            "transition_all_declarations": 0,
            "important_occurrences": 0,
            "backdrop_filter_rules": 6,
            "backdrop_filter_active_rules": 4,
            "keyframe_definitions": 16,
            "duplicate_keyframe_definitions": 0,
            "naked_color_literals": 0,
            "naked_radius_declarations": 0,
            "naked_duration_literals": 0,
        },
    },
    "aggregate": {
        "duplicate_keyframe_names": 9,
        "duplicate_keyframe_definitions": 9,
    },
}


class StyleAuditError(RuntimeError):
    """Raised for missing, malformed, or unsupported audit input."""


@dataclass
class Block:
    header_start: int
    open_index: int
    header: str
    close_index: int | None = None
    children: list["Block"] = field(default_factory=list)


@dataclass(frozen=True)
class Declaration:
    block: Block
    property: str
    value: str
    masked_value: str
    line: int


_PROPERTY_RE = re.compile(r"^(?:--)?[-_a-zA-Z][-_a-zA-Z0-9]*$")
_IMPORTANT_RE = re.compile(r"!\s*important\b", re.IGNORECASE)
_TRANSITION_ALL_RE = re.compile(r"(?<![-_a-zA-Z0-9])all(?![-_a-zA-Z0-9])", re.IGNORECASE)
_TRANSITION_PROPERTY_RE = re.compile(
    r"^(?:-(?:webkit|moz|o)-)?transition(?:-property)?$", re.IGNORECASE
)
_KEYFRAMES_RE = re.compile(
    r"^@(?:-webkit-)?keyframes\s+((?:\\.|[-_a-zA-Z0-9])+)", re.IGNORECASE
)
_HEX_COLOR_RE = re.compile(r"(?<![-_a-zA-Z0-9])#[0-9a-fA-F]{3,8}\b")
_COLOR_FUNCTION_RE = re.compile(
    r"(?<![-_a-zA-Z0-9])(?:rgb|rgba|hsl|hsla|hwb|lab|lch|oklab|oklch|color)\s*\(",
    re.IGNORECASE,
)
_DIMENSION_RE = re.compile(
    r"(?<![-_a-zA-Z0-9.])(?:\d+(?:\.\d+)?|\.\d+)"
    r"(?:px|rem|em|%|vh|vw|vmin|vmax|ch|ex|cm|mm|in|pt|pc)\b",
    re.IGNORECASE,
)
_TIME_RE = re.compile(
    r"(?<![-_a-zA-Z0-9.])(?:\d+(?:\.\d+)?|\.\d+)(?:ms|s)\b",
    re.IGNORECASE,
)
_NAMED_COLORS = frozenset(
    """
    aliceblue antiquewhite aqua aquamarine azure beige bisque black blanchedalmond
    blue blueviolet brown burlywood cadetblue chartreuse chocolate coral
    cornflowerblue cornsilk crimson cyan darkblue darkcyan darkgoldenrod darkgray
    darkgreen darkgrey darkkhaki darkmagenta darkolivegreen darkorange darkorchid
    darkred darksalmon darkseagreen darkslateblue darkslategray darkslategrey
    darkturquoise darkviolet deeppink deepskyblue dimgray dimgrey dodgerblue
    firebrick floralwhite forestgreen fuchsia gainsboro ghostwhite gold goldenrod
    gray green greenyellow grey honeydew hotpink indianred indigo ivory khaki
    lavender lavenderblush lawngreen lemonchiffon lightblue lightcoral lightcyan
    lightgoldenrodyellow lightgray lightgreen lightgrey lightpink lightsalmon
    lightseagreen lightskyblue lightslategray lightslategrey lightsteelblue
    lightyellow lime limegreen linen magenta maroon mediumaquamarine mediumblue
    mediumorchid mediumpurple mediumseagreen mediumslateblue mediumspringgreen
    mediumturquoise mediumvioletred midnightblue mintcream mistyrose moccasin
    navajowhite navy oldlace olive olivedrab orange orangered orchid palegoldenrod
    palegreen paleturquoise palevioletred papayawhip peachpuff peru pink plum
    powderblue purple rebeccapurple red rosybrown royalblue saddlebrown salmon
    sandybrown seagreen seashell sienna silver skyblue slateblue slategray
    slategrey snow springgreen steelblue tan teal thistle tomato turquoise violet
    wheat white whitesmoke yellow yellowgreen
    """.split()
)
_COLOR_IDENT_RE = re.compile(r"(?<![-_a-zA-Z0-9])[-_a-zA-Z][-_a-zA-Z0-9]*(?![-_a-zA-Z0-9])")
_DURATION_PROPERTIES = frozenset(
    {
        "animation",
        "animation-delay",
        "animation-duration",
        "transition",
        "transition-delay",
        "transition-duration",
    }
)
_BACKDROP_PROPERTIES = frozenset({"backdrop-filter", "-webkit-backdrop-filter"})
_INACTIVE_FILTER_VALUES = frozenset(
    {"none", "initial", "inherit", "unset", "revert", "revert-layer"}
)


def _line_number(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def _line_offsets(text: str) -> list[int]:
    return [index for index, char in enumerate(text) if char == "\n"]


def _indexed_line_number(newlines: Sequence[int], index: int) -> int:
    return bisect.bisect_left(newlines, index) + 1


def _mask_comments_and_strings(text: str) -> str:
    """Mask comments and strings while preserving offsets and newlines."""
    chars = list(text)
    index = 0
    length = len(chars)
    while index < length:
        if chars[index] == "/" and index + 1 < length and chars[index + 1] == "*":
            start = index
            chars[index] = chars[index + 1] = " "
            index += 2
            while index + 1 < length and not (chars[index] == "*" and chars[index + 1] == "/"):
                if chars[index] != "\n":
                    chars[index] = " "
                index += 1
            if index + 1 >= length:
                raise StyleAuditError(
                    f"unterminated CSS comment at line {_line_number(text, start)}"
                )
            chars[index] = chars[index + 1] = " "
            index += 2
            continue
        if chars[index] in {'"', "'"}:
            quote = chars[index]
            start = index
            chars[index] = " "
            index += 1
            closed = False
            while index < length:
                current = chars[index]
                if current == "\\":
                    chars[index] = " "
                    if index + 1 < length:
                        index += 1
                        if chars[index] != "\n":
                            chars[index] = " "
                    index += 1
                    continue
                if current == quote:
                    chars[index] = " "
                    index += 1
                    closed = True
                    break
                if current != "\n":
                    chars[index] = " "
                index += 1
            if not closed:
                raise StyleAuditError(
                    f"unterminated CSS string at line {_line_number(text, start)}"
                )
            continue
        index += 1
    return "".join(chars)


def _is_escaped(text: str, index: int) -> bool:
    slashes = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        slashes += 1
        index -= 1
    return slashes % 2 == 1


def _parse_blocks(masked: str, source: str) -> list[Block]:
    roots: list[Block] = []
    stack: list[tuple[Block | None, int, list[Block]]] = [(None, 0, roots)]
    paren_depth = 0
    bracket_depth = 0

    for index, char in enumerate(masked):
        if _is_escaped(masked, index):
            continue
        if char == "(":
            paren_depth += 1
            continue
        if char == ")":
            paren_depth -= 1
            if paren_depth < 0:
                raise StyleAuditError(
                    f"{source}: unmatched ')' at line {_line_number(masked, index)}"
                )
            continue
        if char == "[":
            bracket_depth += 1
            continue
        if char == "]":
            bracket_depth -= 1
            if bracket_depth < 0:
                raise StyleAuditError(
                    f"{source}: unmatched ']' at line {_line_number(masked, index)}"
                )
            continue
        if paren_depth or bracket_depth:
            continue

        current, statement_start, children = stack[-1]
        if char == "{":
            header = masked[statement_start:index].strip()
            if not header:
                raise StyleAuditError(
                    f"{source}: block without selector/at-rule at line {_line_number(masked, index)}"
                )
            leading = len(masked[statement_start:index]) - len(masked[statement_start:index].lstrip())
            block = Block(statement_start + leading, index, header)
            children.append(block)
            stack.append((block, index + 1, block.children))
        elif char == ";":
            stack[-1] = (current, index + 1, children)
        elif char == "}":
            if len(stack) == 1:
                raise StyleAuditError(
                    f"{source}: unmatched '}}' at line {_line_number(masked, index)}"
                )
            block, _, _ = stack.pop()
            assert block is not None
            block.close_index = index
            parent, _, parent_children = stack[-1]
            stack[-1] = (parent, index + 1, parent_children)

    if paren_depth or bracket_depth:
        raise StyleAuditError(f"{source}: unbalanced parentheses or brackets")
    if len(stack) != 1:
        block = stack[-1][0]
        assert block is not None
        raise StyleAuditError(
            f"{source}: unclosed block at line {_line_number(masked, block.open_index)}"
        )
    return roots


def _walk_blocks(blocks: Iterable[Block]) -> Iterable[Block]:
    for block in blocks:
        yield block
        yield from _walk_blocks(block.children)


def _split_statements(masked: str, start: int, end: int) -> Iterable[tuple[int, int]]:
    statement_start = start
    paren_depth = 0
    bracket_depth = 0
    for index in range(start, end):
        char = masked[index]
        if _is_escaped(masked, index):
            continue
        if char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth = max(0, paren_depth - 1)
        elif char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth = max(0, bracket_depth - 1)
        elif char == ";" and not paren_depth and not bracket_depth:
            yield statement_start, index
            statement_start = index + 1
    if statement_start < end:
        yield statement_start, end


def _direct_segments(block: Block) -> Iterable[tuple[int, int]]:
    assert block.close_index is not None
    cursor = block.open_index + 1
    for child in block.children:
        if cursor < child.header_start:
            yield cursor, child.header_start
        assert child.close_index is not None
        cursor = child.close_index + 1
    if cursor < block.close_index:
        yield cursor, block.close_index


def _find_declaration_colon(masked_statement: str) -> int:
    paren_depth = 0
    bracket_depth = 0
    for index, char in enumerate(masked_statement):
        if _is_escaped(masked_statement, index):
            continue
        if char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth = max(0, paren_depth - 1)
        elif char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth = max(0, bracket_depth - 1)
        elif char == ":" and not paren_depth and not bracket_depth:
            return index
    return -1


def _parse_declarations(
    text: str,
    masked: str,
    roots: Sequence[Block],
    newlines: Sequence[int],
) -> list[Declaration]:
    declarations: list[Declaration] = []
    for block in _walk_blocks(roots):
        for segment_start, segment_end in _direct_segments(block):
            for start, end in _split_statements(masked, segment_start, segment_end):
                masked_statement = masked[start:end]
                colon = _find_declaration_colon(masked_statement)
                if colon < 0:
                    continue
                raw_property = masked_statement[:colon].strip()
                if not _PROPERTY_RE.fullmatch(raw_property):
                    continue
                property_offset = len(masked_statement[:colon]) - len(
                    masked_statement[:colon].lstrip()
                )
                value_start = start + colon + 1
                value = text[value_start:end].strip()
                masked_value = masked[value_start:end].strip()
                declarations.append(
                    Declaration(
                        block=block,
                        property=raw_property.lower(),
                        value=value,
                        masked_value=masked_value,
                        line=_indexed_line_number(newlines, start + property_offset),
                    )
                )
    return declarations


def _sample(declaration: Declaration) -> dict[str, Any]:
    compact = re.sub(r"\s+", " ", declaration.value).strip()
    return {
        "line": declaration.line,
        "property": declaration.property,
        "value": compact[:160],
    }


def _append_sample(samples: dict[str, list[dict[str, Any]]], key: str, item: dict[str, Any]) -> None:
    bucket = samples.setdefault(key, [])
    if len(bucket) < SAMPLE_LIMIT:
        bucket.append(item)


def _is_radius_property(property_name: str) -> bool:
    return property_name == "border-radius" or (
        property_name.startswith("border-") and property_name.endswith("-radius")
    )


def _mask_url_functions(value: str) -> str:
    """Remove URL payloads so fragments such as url(#icon) are not colors."""
    output = list(value)
    lower = value.lower()
    cursor = 0
    while True:
        match = re.search(r"(?<![-_a-zA-Z0-9])url\s*\(", lower[cursor:])
        if not match:
            break
        start = cursor + match.start()
        open_index = cursor + match.end() - 1
        depth = 1
        index = open_index + 1
        while index < len(value) and depth:
            if value[index] == "(" and not _is_escaped(value, index):
                depth += 1
            elif value[index] == ")" and not _is_escaped(value, index):
                depth -= 1
            index += 1
        end = index if depth == 0 else len(value)
        for item in range(start, end):
            if output[item] != "\n":
                output[item] = " "
        cursor = max(end, open_index + 1)
    return "".join(output)


def _property_accepts_named_color(property_name: str) -> bool:
    return (
        "color" in property_name
        or property_name.startswith(
            (
                "background",
                "border",
                "outline",
                "box-shadow",
                "text-shadow",
                "fill",
                "stroke",
                "caret",
                "accent",
                "column-rule",
                "text-decoration",
                "filter",
            )
        )
    )


def _color_literal_count(property_name: str, value: str) -> int:
    value = _mask_url_functions(value)
    count = len(_HEX_COLOR_RE.findall(value)) + len(_COLOR_FUNCTION_RE.findall(value))
    if _property_accepts_named_color(property_name):
        count += sum(
            1
            for match in _COLOR_IDENT_RE.finditer(value)
            if match.group(0).lower() in _NAMED_COLORS
        )
    return count


def _active_filter_value(value: str) -> bool:
    normalized = _IMPORTANT_RE.sub("", value).strip().lower()
    return normalized not in _INACTIVE_FILTER_VALUES


def analyze_css(path: Path, logical_name: str | None = None) -> dict[str, Any]:
    logical_name = logical_name or path.as_posix()
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise StyleAuditError(f"{logical_name}: cannot read CSS: {exc}") from exc
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise StyleAuditError(f"{logical_name}: CSS must be UTF-8: {exc}") from exc

    masked = _mask_comments_and_strings(text)
    roots = _parse_blocks(masked, logical_name)
    blocks = list(_walk_blocks(roots))
    newlines = _line_offsets(text)
    declarations = _parse_declarations(text, masked, roots, newlines)
    samples: dict[str, list[dict[str, Any]]] = {}

    transition_all = 0
    important = 0
    naked_colors = 0
    naked_radius = 0
    naked_duration = 0
    backdrop_by_block: dict[int, list[Declaration]] = {}

    for declaration in declarations:
        property_name = declaration.property
        value = declaration.masked_value

        if _TRANSITION_PROPERTY_RE.fullmatch(property_name) and _TRANSITION_ALL_RE.search(value):
            transition_all += 1
            _append_sample(samples, "transition_all", _sample(declaration))

        important_matches = len(_IMPORTANT_RE.findall(value))
        if important_matches:
            important += important_matches
            _append_sample(samples, "important", _sample(declaration))

        if property_name in _BACKDROP_PROPERTIES:
            backdrop_by_block.setdefault(id(declaration.block), []).append(declaration)

        is_custom_property = property_name.startswith("--")
        if not is_custom_property:
            color_count = _color_literal_count(property_name, value)
            if color_count:
                naked_colors += color_count
                _append_sample(samples, "naked_color", _sample(declaration))

            if _is_radius_property(property_name):
                dimensions = _DIMENSION_RE.findall(value)
                nonzero = [token for token in dimensions if float(re.match(r"(?:\d+(?:\.\d+)?|\.\d+)", token).group(0)) != 0]
                if nonzero:
                    naked_radius += 1
                    _append_sample(samples, "naked_radius", _sample(declaration))

            if property_name in _DURATION_PROPERTIES:
                time_count = len(_TIME_RE.findall(value))
                if time_count:
                    naked_duration += time_count
                    _append_sample(samples, "naked_duration", _sample(declaration))

    active_backdrop_rules = 0
    for rule_declarations in backdrop_by_block.values():
        if any(_active_filter_value(item.masked_value) for item in rule_declarations):
            active_backdrop_rules += 1
            _append_sample(samples, "backdrop_filter_active", _sample(rule_declarations[0]))

    keyframe_entries: list[tuple[str, int]] = []
    for block in blocks:
        match = _KEYFRAMES_RE.match(block.header)
        if match:
            keyframe_entries.append(
                (match.group(1), _indexed_line_number(newlines, block.header_start))
            )
    keyframe_counts = Counter(name for name, _ in keyframe_entries)
    duplicate_keyframes = {
        name: {"definitions": count, "lines": [line for item, line in keyframe_entries if item == name]}
        for name, count in sorted(keyframe_counts.items())
        if count > 1
    }

    return {
        "path": logical_name,
        "bytes": len(raw),
        "gzip_bytes": len(gzip.compress(raw, compresslevel=9, mtime=0)),
        "lines": text.count("\n") + (0 if not text or text.endswith("\n") else 1),
        "rule_blocks": len(blocks),
        "declarations": len(declarations),
        "transition_all_declarations": transition_all,
        "important_occurrences": important,
        "backdrop_filter_declarations": sum(len(items) for items in backdrop_by_block.values()),
        "backdrop_filter_rules": len(backdrop_by_block),
        "backdrop_filter_active_rules": active_backdrop_rules,
        "backdrop_filter_none_rules": len(backdrop_by_block) - active_backdrop_rules,
        "keyframe_definitions": len(keyframe_entries),
        "unique_keyframes": len(keyframe_counts),
        "duplicate_keyframe_names": len(duplicate_keyframes),
        "duplicate_keyframe_definitions": sum(count - 1 for count in keyframe_counts.values() if count > 1),
        "duplicate_keyframes": duplicate_keyframes,
        "keyframes": [
            {"name": name, "line": line} for name, line in keyframe_entries
        ],
        "naked_color_literals": naked_colors,
        "naked_radius_declarations": naked_radius,
        "naked_duration_literals": naked_duration,
        "samples": samples,
    }


def collect_metrics(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    all_keyframes: Counter[str] = Counter()
    keyframe_locations: dict[str, list[str]] = {}

    for logical_name in TRACKED_FILES:
        metrics = analyze_css(root / logical_name, logical_name)
        files[logical_name] = metrics
        for entry in metrics["keyframes"]:
            name = entry["name"]
            all_keyframes[name] += 1
            keyframe_locations.setdefault(name, []).append(
                f"{logical_name}:{entry['line']}"
            )

    cross_file_duplicates = {
        name: {"definitions": count, "locations": keyframe_locations[name]}
        for name, count in sorted(all_keyframes.items())
        if count > 1
    }
    aggregate = {
        "bytes": sum(item["bytes"] for item in files.values()),
        "gzip_bytes": sum(item["gzip_bytes"] for item in files.values()),
        "keyframe_definitions": sum(all_keyframes.values()),
        "unique_keyframes": len(all_keyframes),
        "duplicate_keyframe_names": len(cross_file_duplicates),
        "duplicate_keyframe_definitions": sum(
            count - 1 for count in all_keyframes.values() if count > 1
        ),
        "duplicate_keyframes": cross_file_duplicates,
        "backdrop_filter_rules": sum(item["backdrop_filter_rules"] for item in files.values()),
        "backdrop_filter_active_rules": sum(
            item["backdrop_filter_active_rules"] for item in files.values()
        ),
    }
    return files, aggregate


def build_baseline(files: dict[str, dict[str, Any]], aggregate: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "policy": "snapshot",
        "files": {
            path: {metric: int(files[path][metric]) for metric in BASELINE_FILE_METRICS}
            for path in TRACKED_FILES
        },
        "aggregate": {
            metric: int(aggregate[metric]) for metric in GATED_AGGREGATE_METRICS
        },
    }


def validate_baseline(payload: Any, source: str) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("schema_version") != BASELINE_SCHEMA_VERSION:
        raise StyleAuditError(
            f"{source}: expected baseline schema {BASELINE_SCHEMA_VERSION!r}"
        )
    files = payload.get("files")
    aggregate = payload.get("aggregate")
    if not isinstance(files, dict) or not isinstance(aggregate, dict):
        raise StyleAuditError(f"{source}: baseline requires files and aggregate objects")
    for logical_name in TRACKED_FILES:
        limits = files.get(logical_name)
        if not isinstance(limits, dict):
            raise StyleAuditError(f"{source}: missing baseline for {logical_name}")
        for metric in BASELINE_FILE_METRICS:
            value = limits.get(metric)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise StyleAuditError(
                    f"{source}: {logical_name}.{metric} must be a non-negative integer"
                )
    for metric in GATED_AGGREGATE_METRICS:
        value = aggregate.get(metric)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise StyleAuditError(f"{source}: aggregate.{metric} must be a non-negative integer")
    return payload


def load_baseline(path: Path | None) -> tuple[dict[str, Any], str]:
    if path is None:
        return validate_baseline(BUILTIN_BASELINE, "builtin"), "builtin"
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StyleAuditError(f"cannot load baseline {path}: {exc}") from exc
    return validate_baseline(payload, str(path)), str(path.resolve())


def evaluate_metrics(
    root: Path,
    files: dict[str, dict[str, Any]],
    aggregate: dict[str, Any],
    baseline: dict[str, Any],
    baseline_source: str = "provided",
) -> dict[str, Any]:
    root = root.resolve()
    violations: list[dict[str, Any]] = []

    for logical_name in TRACKED_FILES:
        current = files[logical_name]
        limits = baseline["files"][logical_name]
        for metric in GATED_FILE_METRICS:
            if logical_name == SITE32_FILE and metric in SITE32_STRICT_ZERO_METRICS:
                continue
            if current[metric] > limits[metric]:
                violations.append(
                    {
                        "code": "baseline_exceeded",
                        "path": logical_name,
                        "metric": metric,
                        "actual": current[metric],
                        "limit": limits[metric],
                    }
                )

    site32 = files[SITE32_FILE]
    for metric in SITE32_STRICT_ZERO_METRICS:
        if site32[metric] != 0:
            violations.append(
                {
                    "code": "site32_strict_zero",
                    "path": SITE32_FILE,
                    "metric": metric,
                    "actual": site32[metric],
                    "limit": 0,
                    "samples": site32["samples"].get(
                        {
                            "transition_all_declarations": "transition_all",
                            "important_occurrences": "important",
                            "naked_color_literals": "naked_color",
                            "naked_radius_declarations": "naked_radius",
                            "naked_duration_literals": "naked_duration",
                        }.get(metric, ""),
                        [],
                    ),
                }
            )

    legacy_keyframes = {
        entry["name"]
        for logical_name in TRACKED_FILES
        if logical_name != SITE32_FILE
        for entry in files[logical_name]["keyframes"]
    }
    site32_keyframes = {entry["name"] for entry in site32["keyframes"]}
    site32_cross_file_duplicates = sorted(site32_keyframes & legacy_keyframes)
    if site32_cross_file_duplicates:
        violations.append(
            {
                "code": "site32_cross_file_duplicate_keyframe",
                "path": SITE32_FILE,
                "metric": "cross_file_duplicate_keyframe_names",
                "actual": len(site32_cross_file_duplicates),
                "limit": 0,
                "names": site32_cross_file_duplicates,
            }
        )

    for metric in GATED_AGGREGATE_METRICS:
        limit = baseline["aggregate"][metric]
        if aggregate[metric] > limit:
            violations.append(
                {
                    "code": "aggregate_baseline_exceeded",
                    "path": "aggregate",
                    "metric": metric,
                    "actual": aggregate[metric],
                    "limit": limit,
                    "duplicates": aggregate["duplicate_keyframes"],
                }
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "ok": not violations,
        "exit_code": EXIT_PASS if not violations else EXIT_GATE_FAILED,
        "mode": "audit",
        "root": str(root),
        "baseline": {
            "source": baseline_source,
            "schema_version": baseline["schema_version"],
            "policy": baseline.get("policy", "unspecified"),
        },
        "policy": {
            "legacy": "baseline ceilings; debt and CSS size may decrease but may not increase",
            "compressed_size": "gzip bytes are reported and snapshotted but not gated because zlib output can vary by runtime",
            "site32_strict_zero": list(SITE32_STRICT_ZERO_METRICS),
            "site32_cross_file_keyframes": "must not reuse a keyframe name from legacy sheets",
            "backdrop_filter_counting": "standard and -webkit declarations in one block count as one logical rule",
        },
        "files": files,
        "aggregate": aggregate,
        "violations": violations,
    }


def run_audit(root: Path, baseline: dict[str, Any], baseline_source: str = "provided") -> dict[str, Any]:
    root = root.resolve()
    files, aggregate = collect_metrics(root)
    return evaluate_metrics(root, files, aggregate, baseline, baseline_source)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = None
    temporary_name = ""
    try:
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        )
        temporary_name = handle.name
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.chmod(temporary_name, 0o644)
        os.replace(temporary_name, path)
    except OSError as exc:
        if handle is not None and not handle.closed:
            handle.close()
        if temporary_name:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            except OSError:
                pass
        raise StyleAuditError(f"cannot write JSON to {path}: {exc}") from exc


def _emit(payload: dict[str, Any], pretty: bool) -> None:
    json.dump(
        payload,
        sys.stdout,
        ensure_ascii=False,
        indent=2 if pretty else None,
        sort_keys=True,
        separators=None if pretty else (",", ":"),
    )
    sys.stdout.write("\n")


def _parser() -> argparse.ArgumentParser:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Audit Site32 CSS debt, token use, effects, keyframes, and size."
    )
    parser.add_argument("--root", type=Path, default=default_root, help="cmdcenter root")
    parser.add_argument("--baseline", type=Path, help="baseline JSON; defaults to builtin ceilings")
    parser.add_argument(
        "--write-baseline",
        type=Path,
        metavar="PATH",
        help="write an exact snapshot baseline to PATH; no file is written by default",
    )
    parser.add_argument("--output", type=Path, help="also write the audit JSON to PATH")
    parser.add_argument("--pretty", action="store_true", help="pretty-print stdout JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.write_baseline and args.baseline:
            raise StyleAuditError("--write-baseline and --baseline are mutually exclusive")
        if args.write_baseline and args.output:
            if args.write_baseline.resolve() == args.output.resolve():
                raise StyleAuditError("--write-baseline and --output must use different paths")

        root = args.root.resolve()
        if args.write_baseline:
            files, aggregate = collect_metrics(root)
            strict_probe = evaluate_metrics(
                root,
                files,
                aggregate,
                {
                    "schema_version": BASELINE_SCHEMA_VERSION,
                    "policy": "write-baseline-probe",
                    "files": {
                        name: {metric: sys.maxsize for metric in GATED_FILE_METRICS}
                        for name in TRACKED_FILES
                    },
                    "aggregate": {metric: sys.maxsize for metric in GATED_AGGREGATE_METRICS},
                },
                "write-baseline-probe",
            )
            if not strict_probe["ok"]:
                strict_probe["mode"] = "write-baseline"
                strict_probe["error"] = "strict Site32 violations cannot be captured as baseline"
                _emit(strict_probe, args.pretty)
                return EXIT_GATE_FAILED
            snapshot = build_baseline(files, aggregate)
            _atomic_write_json(args.write_baseline, snapshot)
            payload = {
                "schema_version": SCHEMA_VERSION,
                "ok": True,
                "exit_code": EXIT_PASS,
                "mode": "write-baseline",
                "root": str(root),
                "baseline_written": str(args.write_baseline.resolve()),
                "baseline": snapshot,
            }
        else:
            baseline, baseline_source = load_baseline(args.baseline)
            payload = run_audit(root, baseline, baseline_source)
            payload.update(_candidate_identity(root))
            payload["generated_at"] = _utc_now()

        if args.output:
            _atomic_write_json(args.output, payload)
        _emit(payload, args.pretty)
        return int(payload["exit_code"])
    except StyleAuditError as exc:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "ok": False,
            "exit_code": EXIT_INPUT_ERROR,
            "mode": "input-error",
            "error": str(exc),
        }
        _emit(payload, args.pretty)
        return EXIT_INPUT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
