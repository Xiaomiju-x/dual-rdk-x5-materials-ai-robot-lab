#!/usr/bin/env python3
"""Check repository-local Markdown file links and Markdown heading anchors.

Only the Python standard library is used.  HTTP(S), mailto, and other explicit
URI schemes are outside this check.  Fenced code blocks and inline code spans
are masked so documentation examples do not become release blockers.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Sequence
from urllib.parse import unquote, urlsplit


MARKDOWN_SUFFIXES = frozenset({".markdown", ".md"})
SKIPPED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "venv",
    }
)

FENCE_RE = re.compile(r"^\s{0,3}(?P<fence>`{3,}|~{3,})")
INLINE_LINK_START_RE = re.compile(r"!?\[[^\]\n]*\]\(")
REFERENCE_DEFINITION_RE = re.compile(
    r"^\s{0,3}\[(?P<label>[^\]]+)\]:\s*(?:<(?P<angle>[^>]+)>|(?P<plain>\S+))"
)
REFERENCE_USE_RE = re.compile(
    r"!?\[(?P<text>[^\]\n]+)\]\[(?P<label>[^\]\n]*)\]"
)
ATX_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}(?:\s+|$)(?P<title>.*?)\s*#*\s*$")
SETEXT_RE = re.compile(r"^\s{0,3}(?:=+|-+)\s*$")
HTML_ANCHOR_RE = re.compile(
    r"<(?:a\b[^>]*\b(?:id|name)|[A-Za-z][^>]*\bid)\s*=\s*"
    r"(?P<quote>[\"'])(?P<anchor>.*?)(?P=quote)",
    re.IGNORECASE,
)
HTML_TAG_RE = re.compile(r"<[^>]+>")
MARKDOWN_INLINE_LINK_RE = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
MARKDOWN_REFERENCE_RE = re.compile(r"!?\[([^\]]*)\]\[[^\]]*\]")


@dataclass(frozen=True, order=True)
class Finding:
    rule: str
    source: str
    line: int
    target: str
    message: str


@dataclass
class CheckResult:
    markdown_files: int
    links_checked: int
    findings: list[Finding]

    @property
    def ok(self) -> bool:
        return not self.findings

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "status": "pass" if self.ok else "fail",
            "markdown_files": self.markdown_files,
            "links_checked": self.links_checked,
            "finding_count": len(self.findings),
            "findings": [asdict(item) for item in self.findings],
        }


@dataclass(frozen=True)
class Link:
    line: int
    destination: str | None
    reference_label: str | None = None


class _Collector:
    def __init__(self) -> None:
        self._items: list[Finding] = []
        self._keys: set[tuple[str, str, int, str]] = set()

    def add(
        self,
        rule: str,
        source: str,
        line: int,
        target: str,
        message: str,
    ) -> None:
        key = (rule, source, line, target)
        if key in self._keys:
            return
        self._keys.add(key)
        self._items.append(Finding(rule, source, line, target, message))

    def sorted(self) -> list[Finding]:
        return sorted(self._items)


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _iter_markdown_files(root: Path) -> Iterator[Path]:
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        retained: list[str] = []
        for name in sorted(directory_names):
            candidate = current_path / name
            if (
                name in SKIPPED_DIRECTORY_NAMES
                or name.startswith(".venv-")
                or candidate.is_symlink()
            ):
                continue
            retained.append(name)
        directory_names[:] = retained
        for name in sorted(file_names):
            candidate = current_path / name
            if candidate.is_symlink() or candidate.suffix.lower() not in MARKDOWN_SUFFIXES:
                continue
            yield candidate


def _mask_fenced_code(lines: Sequence[str]) -> list[str]:
    masked: list[str] = []
    fence_character: str | None = None
    fence_length = 0
    for line in lines:
        match = FENCE_RE.match(line)
        if fence_character is None:
            if match:
                fence = match.group("fence")
                fence_character = fence[0]
                fence_length = len(fence)
                masked.append("")
            else:
                masked.append(line)
            continue

        masked.append("")
        if match:
            fence = match.group("fence")
            if fence[0] == fence_character and len(fence) >= fence_length:
                fence_character = None
                fence_length = 0
    return masked


def _mask_inline_code(line: str) -> str:
    characters = list(line)
    cursor = 0
    while cursor < len(line):
        if line[cursor] != "`":
            cursor += 1
            continue
        run_end = cursor
        while run_end < len(line) and line[run_end] == "`":
            run_end += 1
        delimiter = line[cursor:run_end]
        closing = line.find(delimiter, run_end)
        if closing < 0:
            cursor = run_end
            continue
        for index in range(cursor, closing + len(delimiter)):
            characters[index] = " "
        cursor = closing + len(delimiter)
    return "".join(characters)


def _is_escaped(text: str, position: int) -> bool:
    backslashes = 0
    cursor = position - 1
    while cursor >= 0 and text[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return backslashes % 2 == 1


def _parse_inline_destination(line: str, opening_parenthesis: int) -> str | None:
    cursor = opening_parenthesis + 1
    while cursor < len(line) and line[cursor].isspace():
        cursor += 1
    if cursor >= len(line):
        return None

    if line[cursor] == "<":
        cursor += 1
        destination: list[str] = []
        while cursor < len(line):
            character = line[cursor]
            if character == ">" and not _is_escaped(line, cursor):
                return "".join(destination)
            if character == "\\" and cursor + 1 < len(line):
                cursor += 1
                destination.append(line[cursor])
            else:
                destination.append(character)
            cursor += 1
        return None

    destination = []
    nested_parentheses = 0
    while cursor < len(line):
        character = line[cursor]
        if character == "\\" and cursor + 1 < len(line):
            cursor += 1
            destination.append(line[cursor])
        elif character == "(" and not _is_escaped(line, cursor):
            nested_parentheses += 1
            destination.append(character)
        elif character == ")" and not _is_escaped(line, cursor):
            if nested_parentheses == 0:
                return "".join(destination)
            nested_parentheses -= 1
            destination.append(character)
        elif character.isspace() and nested_parentheses == 0:
            return "".join(destination)
        else:
            destination.append(character)
        cursor += 1
    return None


def _normalize_reference_label(label: str) -> str:
    return " ".join(label.strip().lower().split())


def _extract_links(lines: Sequence[str]) -> list[Link]:
    masked_lines = [_mask_inline_code(line) for line in _mask_fenced_code(lines)]
    definitions: dict[str, str] = {}
    definition_lines: set[int] = set()
    links: list[Link] = []

    for line_number, line in enumerate(masked_lines, start=1):
        match = REFERENCE_DEFINITION_RE.match(line)
        if not match:
            continue
        destination = match.group("angle") or match.group("plain") or ""
        definitions[_normalize_reference_label(match.group("label"))] = destination
        definition_lines.add(line_number)
        links.append(Link(line_number, destination))

    for line_number, line in enumerate(masked_lines, start=1):
        if line_number in definition_lines:
            continue
        for match in INLINE_LINK_START_RE.finditer(line):
            if _is_escaped(line, match.start()):
                continue
            opening_parenthesis = match.end() - 1
            destination = _parse_inline_destination(line, opening_parenthesis)
            if destination is not None:
                links.append(Link(line_number, destination))

        for match in REFERENCE_USE_RE.finditer(line):
            if _is_escaped(line, match.start()):
                continue
            label = match.group("label") or match.group("text")
            destination = definitions.get(_normalize_reference_label(label))
            links.append(Link(line_number, destination, label if destination is None else None))
    return links


def _github_slug(title: str) -> str:
    value = html.unescape(title).strip().lower()
    value = MARKDOWN_INLINE_LINK_RE.sub(r"\1", value)
    value = MARKDOWN_REFERENCE_RE.sub(r"\1", value)
    value = HTML_TAG_RE.sub("", value)
    value = value.replace("`", "")
    value = re.sub(r"[^\w\s-]", "", value, flags=re.UNICODE)
    value = re.sub(r"\s+", "-", value)
    return value


def _extract_anchors(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8-sig")
    lines = _mask_fenced_code(text.splitlines())
    anchors: set[str] = set()
    used_heading_slugs: set[str] = set()

    def add_heading(title: str) -> None:
        base = _github_slug(title)
        if not base:
            return
        candidate = base
        suffix = 0
        while candidate in used_heading_slugs:
            suffix += 1
            candidate = f"{base}-{suffix}"
        used_heading_slugs.add(candidate)
        anchors.add(candidate)

    previous_candidate: str | None = None
    for line in lines:
        for match in HTML_ANCHOR_RE.finditer(line):
            anchor = html.unescape(match.group("anchor")).strip()
            if anchor:
                anchors.add(anchor)

        atx_match = ATX_HEADING_RE.match(line)
        if atx_match:
            add_heading(atx_match.group("title"))
            previous_candidate = None
            continue
        if previous_candidate is not None and SETEXT_RE.match(line):
            add_heading(previous_candidate)
            previous_candidate = None
            continue
        stripped = line.strip()
        previous_candidate = stripped if stripped and not stripped.startswith("<") else None
    return anchors


def _safe_unquote(value: str) -> str | None:
    if re.search(r"%(?![0-9A-Fa-f]{2})", value):
        return None
    try:
        return unquote(value, encoding="utf-8", errors="strict")
    except UnicodeDecodeError:
        return None


def _display_target(path_text: str, fragment: str) -> str:
    target = path_text or "<same-page>"
    if fragment:
        target += "#" + fragment
    return target


def _resolve_target(
    root: Path,
    source: Path,
    destination: str,
) -> tuple[str, Path, str] | None:
    cleaned = html.unescape(destination.strip())
    if not cleaned:
        return None
    if cleaned.startswith("//"):
        return None
    parsed = urlsplit(cleaned)
    if parsed.scheme:
        return None

    path_text = _safe_unquote(parsed.path)
    fragment = _safe_unquote(parsed.fragment)
    if path_text is None or fragment is None:
        raise ValueError("invalid URL encoding")
    path_text = path_text.replace("\\", "/")
    if path_text.startswith("/"):
        candidate = root / path_text.lstrip("/")
    elif path_text:
        candidate = source.parent / path_text
    else:
        candidate = source
    return _display_target(path_text, fragment), candidate.resolve(), fragment


def _markdown_target_for_anchor(candidate: Path) -> Path | None:
    if candidate.is_file() and candidate.suffix.lower() in MARKDOWN_SUFFIXES:
        return candidate
    if candidate.is_dir():
        for name in ("README.md", "README.markdown"):
            readme = candidate / name
            if readme.is_file():
                return readme
    return None


def check_tree(root: Path | str) -> CheckResult:
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise ValueError("check root must be an existing directory")

    markdown_files = list(_iter_markdown_files(root_path))
    collector = _Collector()
    links_checked = 0
    anchor_cache: dict[Path, set[str]] = {}

    for source in markdown_files:
        source_relative = _relative(source, root_path)
        try:
            lines = source.read_text(encoding="utf-8-sig").splitlines()
        except (OSError, UnicodeError):
            collector.add(
                "unreadable_markdown",
                source_relative,
                1,
                source_relative,
                "Markdown is not readable UTF-8 text.",
            )
            continue

        for link in _extract_links(lines):
            if link.destination is None:
                links_checked += 1
                collector.add(
                    "undefined_reference",
                    source_relative,
                    link.line,
                    "[" + (link.reference_label or "") + "]",
                    "The Markdown reference label has no definition.",
                )
                continue
            try:
                resolved = _resolve_target(root_path, source, link.destination)
            except ValueError:
                links_checked += 1
                collector.add(
                    "invalid_url_encoding",
                    source_relative,
                    link.line,
                    "<invalid-encoding>",
                    "The local link contains invalid percent encoding.",
                )
                continue
            if resolved is None:
                continue
            links_checked += 1
            display_target, candidate, fragment = resolved
            try:
                candidate.relative_to(root_path)
            except ValueError:
                collector.add(
                    "outside_repository",
                    source_relative,
                    link.line,
                    display_target,
                    "The relative link resolves outside the repository.",
                )
                continue

            if not candidate.exists():
                collector.add(
                    "missing_file",
                    source_relative,
                    link.line,
                    display_target,
                    "The linked repository file does not exist.",
                )
                continue
            if not fragment:
                continue

            markdown_target = _markdown_target_for_anchor(candidate)
            if markdown_target is None:
                continue
            try:
                if markdown_target not in anchor_cache:
                    anchor_cache[markdown_target] = _extract_anchors(markdown_target)
                anchors = anchor_cache[markdown_target]
            except (OSError, UnicodeError):
                collector.add(
                    "unreadable_markdown",
                    source_relative,
                    link.line,
                    display_target,
                    "The linked Markdown file is not readable UTF-8 text.",
                )
                continue
            normalized_fragment = _github_slug(fragment)
            if fragment not in anchors and normalized_fragment not in anchors:
                collector.add(
                    "missing_anchor",
                    source_relative,
                    link.line,
                    display_target,
                    "The linked Markdown heading anchor does not exist.",
                )

    return CheckResult(len(markdown_files), links_checked, collector.sorted())


def _render_text(result: CheckResult) -> str:
    status = "PASS" if result.ok else "FAIL"
    lines = [
        f"MARKDOWN_LINK_CHECK={status}",
        f"markdown_files={result.markdown_files}",
        f"links_checked={result.links_checked}",
        f"finding_count={len(result.findings)}",
    ]
    for finding in result.findings:
        lines.append(
            f"[{finding.rule}] {finding.source}:{finding.line} -> "
            f"{finding.target} - {finding.message}"
        )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repository root to check (default: repository root)",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="concise text or machine-readable JSON",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = check_tree(args.root)
    except (OSError, ValueError) as error:
        print(f"MARKDOWN_LINK_CHECK=ERROR\nreason={type(error).__name__}", file=sys.stderr)
        return 2
    if args.format == "json":
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(_render_text(result))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
