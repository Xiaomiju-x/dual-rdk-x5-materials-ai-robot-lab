#!/usr/bin/env python3
"""Build a standalone, read-only HTML evidence report from shadow receipts."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def require_no_motion_authority(receipt: dict[str, Any]) -> None:
    authority = receipt.get("authority")
    if not isinstance(authority, dict):
        raise ValueError("receipt is missing authority")
    if authority.get("motion_authority") is not False:
        raise ValueError("receipt must set motion_authority=false")
    if authority.get("execution_allowed") is not False:
        raise ValueError("receipt must set execution_allowed=false")
    if authority.get("actuator_commands_issued") != 0:
        raise ValueError("receipt must prove actuator_commands_issued=0")


def scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "PASS" if value else "NOT PROVEN"
    if value is None:
        return "-"
    return str(value)


def build_html(receipt: dict[str, Any], model_receipts: list[dict[str, Any]]) -> str:
    require_no_motion_authority(receipt)
    prediction = receipt.get("prediction", {})
    evidence = receipt.get("evidence", {})
    graph = receipt.get("skill_graph", {})
    source = receipt.get("source", {})
    observed = graph.get("observed", [])
    expected = graph.get("expected", [])
    missing = set(graph.get("missing", []))
    model_rows = []
    for model in model_receipts:
        status = html.escape(str(model.get("status", model.get("maturity", "UNKNOWN"))))
        name = html.escape(str(model.get("model_name", model.get("name", "unnamed"))))
        digest = html.escape(str(model.get("sha256", model.get("checkpoint_sha256", "-")))[:16])
        model_rows.append(f"<tr><td>{name}</td><td>{status}</td><td><code>{digest}</code></td></tr>")
    if not model_rows:
        model_rows.append(
            "<tr><td>Learned policies</td><td>TRAINING_PENDING</td>"
            "<td><code>no checkpoint claimed</code></td></tr>"
        )

    phase_items = []
    observed_set = set(observed)
    for index, phase in enumerate(expected, 1):
        state = "missing" if phase in missing else ("seen" if phase in observed_set else "waiting")
        phase_items.append(
            f'<li class="{state}"><span>{index:02d}</span><strong>{html.escape(phase)}</strong>'
            f"<small>{state.upper()}</small></li>"
        )

    evidence_rows = []
    for key in (
        "apriltag_id2",
        "bag_present_cpu_authority",
        "bpu_auxiliary_forward",
        "closed_loop_done",
    ):
        value = evidence.get(key)
        cls = "ok" if value is True else "warn"
        evidence_rows.append(
            f'<tr><td>{html.escape(key)}</td><td class="{cls}">{html.escape(scalar(value))}</td></tr>'
        )

    verdict = html.escape(str(prediction.get("verdict", "UNKNOWN")))
    verdict_cls = "ok" if verdict == "AGREE" else "warn"
    source_hash = html.escape(str(source.get("sha256", "-")))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>X5-BiSkill Shadow Evidence</title>
<style>
:root {{
  color-scheme: dark;
  --bg: #0b1114;
  --panel: #121a1e;
  --line: #2d3b40;
  --text: #edf4f2;
  --muted: #a8b5b2;
  --cyan: #26c6b5;
  --green: #5fd18a;
  --amber: #f0b45a;
  --red: #ef6b6b;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
  letter-spacing: 0;
}}
header {{
  border-bottom: 1px solid var(--line);
  padding: 28px clamp(22px, 5vw, 72px) 22px;
  display: flex;
  justify-content: space-between;
  gap: 28px;
  align-items: end;
}}
h1 {{ margin: 0; font-size: 34px; line-height: 1.1; }}
header p {{ margin: 8px 0 0; color: var(--muted); }}
.authority {{
  color: var(--green);
  font: 700 13px Consolas, monospace;
  text-align: right;
  line-height: 1.7;
}}
main {{
  width: min(1240px, calc(100% - 40px));
  margin: 28px auto 52px;
}}
.summary {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  border: 1px solid var(--line);
}}
.metric {{ padding: 20px; border-right: 1px solid var(--line); }}
.metric:last-child {{ border-right: 0; }}
.metric small {{ display: block; color: var(--muted); margin-bottom: 8px; }}
.metric strong {{ font-size: 20px; overflow-wrap: anywhere; }}
.ok {{ color: var(--green); }}
.warn {{ color: var(--amber); }}
section {{ padding: 28px 0; border-bottom: 1px solid var(--line); }}
h2 {{ margin: 0 0 18px; font-size: 20px; }}
.phases {{
  list-style: none;
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 1px;
  background: var(--line);
  border: 1px solid var(--line);
  padding: 0;
  margin: 0;
}}
.phases li {{ background: var(--panel); min-height: 112px; padding: 16px; }}
.phases span {{ color: var(--cyan); font: 700 13px Consolas, monospace; }}
.phases strong {{ display: block; margin-top: 18px; font-size: 13px; overflow-wrap: anywhere; }}
.phases small {{ display: block; margin-top: 10px; color: var(--muted); }}
.phases .missing small {{ color: var(--red); }}
.tables {{ display: grid; grid-template-columns: 1fr 1.4fr; gap: 28px; }}
table {{ width: 100%; border-collapse: collapse; background: var(--panel); }}
th, td {{ padding: 13px 15px; border: 1px solid var(--line); text-align: left; font-size: 13px; }}
th {{ color: var(--muted); font-weight: 600; }}
code {{ font-family: Consolas, monospace; color: #b9e6df; overflow-wrap: anywhere; }}
.hash {{ padding: 16px; border-left: 3px solid var(--cyan); background: var(--panel); }}
.hash small {{ color: var(--muted); }}
.hash code {{ display: block; margin-top: 8px; }}
footer {{ color: var(--muted); font-size: 12px; margin-top: 24px; }}
@media (max-width: 850px) {{
  header {{ align-items: start; flex-direction: column; }}
  .authority {{ text-align: left; }}
  .summary {{ grid-template-columns: 1fr 1fr; }}
  .metric:nth-child(2) {{ border-right: 0; }}
  .metric:nth-child(-n+2) {{ border-bottom: 1px solid var(--line); }}
  .phases {{ grid-template-columns: 1fr 1fr; }}
  .tables {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<header>
  <div>
    <h1>X5-BiSkill Shadow</h1>
    <p>Dual-arm VLA-derived offline replay and evidence monitor</p>
  </div>
  <div class="authority">
    MOTION AUTHORITY = FROZEN V3<br>
    SHADOW ACTUATOR COMMANDS = 0
  </div>
</header>
<main>
  <div class="summary">
    <div class="metric"><small>Shadow verdict</small><strong class="{verdict_cls}">{verdict}</strong></div>
    <div class="metric"><small>Maturity</small><strong>OFFLINE_REPLAY</strong></div>
    <div class="metric"><small>Current phase</small><strong>{html.escape(scalar(prediction.get("current_phase")))}</strong></div>
    <div class="metric"><small>Next skill</small><strong>{html.escape(scalar(prediction.get("next_skill")))}</strong></div>
  </div>
  <section>
    <h2>Frozen skill graph</h2>
    <ol class="phases">{''.join(phase_items)}</ol>
  </section>
  <section class="tables">
    <div>
      <h2>Physical evidence</h2>
      <table><tbody>{''.join(evidence_rows)}</tbody></table>
    </div>
    <div>
      <h2>Learning candidates</h2>
      <table>
        <thead><tr><th>Model</th><th>Evidence status</th><th>Artifact hash</th></tr></thead>
        <tbody>{''.join(model_rows)}</tbody>
      </table>
    </div>
  </section>
  <section>
    <h2>Source integrity</h2>
    <div class="hash"><small>Frozen result SHA-256</small><code>{source_hash}</code></div>
  </section>
  <footer>
    Learning outputs are advisory evidence only. They cannot open cameras,
    serial ports, GPIO, SSH sessions, or robot SDKs.
  </footer>
</main>
</body>
</html>
"""


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--model-receipt", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = read_json(args.receipt)
    models = [read_json(path) for path in args.model_receipt]
    content = build_html(receipt, models)
    atomic_write(args.output, content)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "motion_authority": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
