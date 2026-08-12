#!/usr/bin/env python3
"""Static contract checks for the Site32 frontend module split."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"
SRC = STATIC / "src" / "site32"
INDEX = STATIC / "index.html"
ENTRY = STATIC / "site32.js"

RESPONSIBILITY_MODULES = {
    "router": "installSite32Router",
    "state": "createSite32State",
    "appearance": "installSite32Appearance",
    "search": "installSite32Search",
    "theater": "installSite32Theater",
    "motion": "installSite32Motion",
    "a11y": "installSite32A11y",
    "telemetry": "installSite32Telemetry",
}

INLINE_HANDLER_PATTERNS = [
    re.compile(r"\bon[a-z]+\s*=", re.IGNORECASE),
    re.compile(r"\.on[a-z]+\s*=", re.IGNORECASE),
    re.compile(r"setAttribute\s*\(\s*['\"]on[a-z]+['\"]", re.IGNORECASE),
    re.compile(r"insertAdjacentHTML\s*\(", re.IGNORECASE),
]

IMPORT_RE = re.compile(r"import\s+(?:[^'\"\n]+?\s+from\s+)?['\"]([^'\"]+)['\"]")


def fail(message: str) -> None:
    raise AssertionError(message)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def normalize_import(source: Path, spec: str) -> Path | None:
    spec = spec.split("?", 1)[0]
    if spec.startswith("/"):
      candidate = STATIC / spec.lstrip("/")
    elif spec.startswith("."):
      candidate = (source.parent / spec).resolve()
    else:
      return None
    try:
      return candidate.relative_to(STATIC.resolve())
    except ValueError:
      return None


def module_files() -> list[Path]:
    files = [ENTRY]
    files.extend(sorted(SRC.glob("*.js")))
    return files


def check_required_modules() -> None:
    entry = read(ENTRY)
    for name, export_name in RESPONSIBILITY_MODULES.items():
        path = SRC / f"{name}.js"
        if not path.exists():
            fail(f"missing Site32 responsibility module: {path}")
        text = read(path)
        if len(text) < 700:
            fail(f"module is too small to be a real responsibility module: {path}")
        if f"export function {export_name}" not in text:
            fail(f"{path} does not export {export_name}")
        if f"./src/site32/{name}.js" not in entry:
            fail(f"site32.js does not import {name}.js")
        if not re.search(rf"\b{name}\b", entry):
            fail(f"site32.js does not expose module key {name}")

    state_text = read(SRC / "state.js")
    if "subscribe" not in state_text or "update" not in state_text:
        fail("state.js must provide update and subscribe primitives")

    for name in RESPONSIBILITY_MODULES:
        if name == "state":
            continue
        text = read(SRC / f"{name}.js")
        if not any(token in text for token in ("addEventListener", "MutationObserver", "wrapGlobal", "sendBeacon")):
            fail(f"{name}.js is missing a callable adapter/listener surface")


def check_no_dynamic_inline_handlers() -> None:
    for path in module_files():
        text = read(path)
        for pattern in INLINE_HANDLER_PATTERNS:
            match = pattern.search(text)
            if match:
                fail(f"dynamic inline handler pattern in {path}: {match.group(0)}")


def build_graph() -> dict[Path, set[Path]]:
    files = {path.relative_to(STATIC) for path in module_files()}
    graph: dict[Path, set[Path]] = {path: set() for path in files}
    for path in module_files():
        rel = path.relative_to(STATIC)
        for spec in IMPORT_RE.findall(read(path)):
            imported = normalize_import(path, spec)
            if imported in files:
                graph[rel].add(imported)
    return graph


def check_no_cycles() -> None:
    graph = build_graph()
    visiting: set[Path] = set()
    visited: set[Path] = set()
    stack: list[Path] = []

    def visit(node: Path) -> None:
        if node in visited:
            return
        if node in visiting:
            cycle = stack[stack.index(node):] + [node]
            fail("module dependency cycle: " + " -> ".join(str(item) for item in cycle))
        visiting.add(node)
        stack.append(node)
        for child in graph[node]:
            visit(child)
        stack.pop()
        visiting.remove(node)
        visited.add(node)

    for node in graph:
        visit(node)


def check_single_site32_entry() -> None:
    html = read(INDEX)
    module_scripts = re.findall(r"<script\b[^>]*type=['\"]module['\"][^>]*>", html, re.IGNORECASE)
    site32_scripts = [tag for tag in module_scripts if re.search(r"\bsrc=['\"]/site32\.js(?:\?|['\"])", tag)]
    if len(site32_scripts) != 1:
        fail(f"expected exactly one /site32.js module entry, found {len(site32_scripts)}")
    if any("src/site32" in tag for tag in module_scripts):
        fail("index.html must not load Site32 source modules directly")


def check_observable_boot_contract() -> None:
    entry = read(ENTRY)
    required = (
        "window.Site32Boot",
        "document.body.dataset.site32Shell = 'degraded'",
        "document.body.dataset.site32BootError = bootStep",
        "state: 'ready'",
        "state: 'degraded'",
        "console.error('[Site32 boot]'",
    )
    missing = [token for token in required if token not in entry]
    if missing:
        fail(f"Site32 observable boot contract missing: {missing}")


def check_no_copied_state_machines() -> None:
    forbidden = {
        "router.js": ["ROUTE_CLUSTER", "let cur", "const PAGES"],
        "search.js": ["let MX", "const MX", "_materials="],
        "theater.js": ["let TH", "const TOUR", "TOUR=["],
        "appearance.js": ["let cur", "const PAGES", "const TOUR"],
    }
    for filename, needles in forbidden.items():
        text = read(SRC / filename)
        for needle in needles:
            if needle in text:
                fail(f"{filename} appears to copy app.js state machine token: {needle}")


def check_privacy_and_observer_budget() -> None:
    search = read(SRC / "search.js")
    theater = read(SRC / "theater.js")
    if "home: patch.homeQuery" in search or "atlas: patch.atlasQuery" in search:
        fail("search telemetry must not retain raw queries")
    for token in ("homeLength", "atlasLength"):
        if token not in search:
            fail(f"search telemetry is missing privacy-safe metric: {token}")
    if "attributeFilter: ['class', 'aria-pressed']" not in theater:
        fail("theater observer must ignore per-frame progress style mutations")


def check_view_transition_input_passthrough() -> None:
    css = read(STATIC / "site32.css") + "\n" + read(STATIC / "style.css")
    if "::view-transition" not in css or "pointer-events: none" not in css:
        fail("view-transition snapshots must not block newer navigation input")
    for token in (
        ":root{ view-transition-name:none; }",
        "#main{ view-transition-name:route-stage; }",
        "::view-transition-group(route-stage)",
        "pointer-events:none",
    ):
        if token not in css:
            fail(f"scoped view-transition contract missing: {token}")
    app = read(STATIC / "app.js")
    for token in (
        "function installRapidNavCommit()",
        "nav.addEventListener('mousedown'",
        "nav.addEventListener('mouseup'",
        "event.stopImmediatePropagation()",
        "installRapidNavCommit();",
        "active.classList.add('route-entering')",
        "bar.setAttribute('aria-hidden', off?'false':'true')",
        "routeTransitionDone(generation);",
    ):
        if token not in app:
            fail(f"rapid navigation commit contract missing: {token}")


def check_bilingual_route_title_sync() -> None:
    app = read(STATIC / "app.js")
    for token in (
        "Dual-RDK X5 Materials-Synthesis AI and Multi-Robot Laboratory Assistant · Materials Prediction and Closed-Loop Evidence",
        "label==='Dual-RDK X5 Materials-Synthesis AI and Multi-Robot Laboratory Assistant'",
        "queueMicrotask(()=>announceRouteChange(cur))",
        "lang==='en'?'Language switch':'语言切换'",
        "Research portal workspaces",
    ):
        if token not in app:
            fail(f"bilingual route title sync missing: {token}")


def main() -> int:
    checks = [
        check_required_modules,
        check_no_dynamic_inline_handlers,
        check_no_cycles,
        check_single_site32_entry,
        check_observable_boot_contract,
        check_no_copied_state_machines,
        check_privacy_and_observer_budget,
        check_view_transition_input_passthrough,
        check_bilingual_route_title_sync,
    ]
    for check in checks:
        check()
    print("site32 frontend module contract: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"site32 frontend module contract: FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
