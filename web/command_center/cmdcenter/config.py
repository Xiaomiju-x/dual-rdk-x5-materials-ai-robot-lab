"""Side-effect-free configuration parsing for the command center.

This module may be imported by tests and tooling.  It reads only the provided
mapping or process environment and never opens files, starts workers, touches
SQLite, probes the network, or writes state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from types import MappingProxyType


ASSET_VER = "site32-global-commercial-v1.13-20260720"
RELEASED_AT = "2026-07-20T00:14:11Z"

TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
DEFAULT_ALLOWED_HOST_SUFFIXES = (".xiaomiju.xyz",)
DEFAULT_ALLOWED_HOSTS = frozenset({"xiaomiju.xyz", "localhost", "127.0.0.1", "::1"})
DEFAULT_WEBHOOK_HOSTS = MappingProxyType({
    "wecom": frozenset({"qyapi.weixin.qq.com"}),
    "dingtalk": frozenset({"oapi.dingtalk.com"}),
    "feishu": frozenset({"open.feishu.cn", "open.larksuite.com"}),
})


@dataclass(frozen=True, slots=True)
class CmdcenterConfig:
    asset_version: str
    released_at: str
    cmd_test_mode: bool
    allowed_host_suffixes: tuple[str, ...]
    allowed_hosts: frozenset[str]
    webhook_hosts: Mapping[str, frozenset[str]]
    webhook_extra_hosts: frozenset[str]
    sse_max: int
    sse_lifetime_s: int
    auth_dir: str
    llm_key: str
    llm_model: str
    runtime_enabled: bool
    port: int


def env_bool(name: str, default: bool = False, environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    raw = env.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in TRUE_VALUES


def env_bounded_int(
    name: str,
    default: int,
    minimum: int,
    maximum: int,
    environ: Mapping[str, str] | None = None,
) -> int:
    env = os.environ if environ is None else environ
    try:
        value = int(str(env.get(name, default)).strip())
    except (TypeError, ValueError):
        value = default
    return min(max(value, minimum), maximum)


def parse_csv_hosts(value: str | None) -> frozenset[str]:
    return frozenset(
        host.strip().lower().rstrip(".")
        for host in str(value or "").split(",")
        if host.strip()
    )


def load_config(environ: Mapping[str, str] | None = None) -> CmdcenterConfig:
    env = os.environ if environ is None else environ
    auth_dir = str(env.get("XRD_AUTH_DIR") or "~/auth").strip() or "~/auth"
    llm_model = str(env.get("DEEPSEEK_MODEL") or "deepseek-chat").strip() or "deepseek-chat"
    webhook_hosts = MappingProxyType({key: frozenset(value) for key, value in DEFAULT_WEBHOOK_HOSTS.items()})
    return CmdcenterConfig(
        asset_version=ASSET_VER,
        released_at=RELEASED_AT,
        cmd_test_mode=env_bool("XRD_CMD_TEST_MODE", environ=env),
        allowed_host_suffixes=DEFAULT_ALLOWED_HOST_SUFFIXES,
        allowed_hosts=DEFAULT_ALLOWED_HOSTS,
        webhook_hosts=webhook_hosts,
        webhook_extra_hosts=parse_csv_hosts(env.get("XRD_WEBHOOK_HOST_ALLOWLIST")),
        sse_max=env_bounded_int("XRD_SSE_MAX", 2, 1, 8, env),
        sse_lifetime_s=env_bounded_int("XRD_SSE_LIFETIME_S", 55, 15, 300, env),
        auth_dir=os.path.expanduser(auth_dir),
        llm_key=str(env.get("DEEPSEEK_API_KEY") or "").strip(),
        llm_model=llm_model,
        runtime_enabled=env_bool("XRD_CMD_RUNTIME", environ=env),
        port=env_bounded_int("PORT", 29100, 1, 65535, env),
    )
