"""Release identity parsing shared by the Site32 application layer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re


_LEGACY_RELEASE_RE = re.compile(
    r"^site31-global-commercial-r(?P<version>\d+(?:\.\d+)?)-(?P<date>\d{8})$"
)
_SITE32_RELEASE_RE = re.compile(
    r"^site32-global-commercial-v(?P<version>\d+(?:\.\d+)?)-(?P<date>\d{8})$"
)


@dataclass(frozen=True, slots=True)
class ReleaseIdentity:
    value: str
    product: str
    version: str
    date: str
    generation: int

    @property
    def is_site32(self) -> bool:
        return self.product == "site32"

    @property
    def not_before(self) -> datetime:
        return datetime.strptime(self.date, "%Y%m%d").replace(tzinfo=timezone.utc)


def parse_release(value: str) -> ReleaseIdentity:
    """Parse a supported immutable release name or raise ``ValueError``."""

    for product, pattern, generation in (
        ("site32", _SITE32_RELEASE_RE, 32),
        ("site31", _LEGACY_RELEASE_RE, 31),
    ):
        match = pattern.fullmatch(value or "")
        if match:
            return ReleaseIdentity(
                value=value,
                product=product,
                version=match.group("version"),
                date=match.group("date"),
                generation=generation,
            )
    raise ValueError(f"unsupported release identity: {value!r}")


def is_supported_release(value: str) -> bool:
    try:
        parse_release(value)
    except ValueError:
        return False
    return True
