"""Explicit process-runtime startup for the command center."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable


class RuntimeController:
    """Start database initialization and background workers exactly once."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started = False
        self._threads: list[threading.Thread] = []

    @property
    def started(self) -> bool:
        return self._started

    def start(
        self,
        *,
        initialize: Callable[[], None],
        seed: Callable[[], None],
        workers: Iterable[tuple[str, Callable[[], None]]],
    ) -> bool:
        with self._lock:
            if self._started:
                return False
            initialize()
            seed()
            for name, target in workers:
                thread = threading.Thread(target=target, daemon=True, name=name)
                thread.start()
                self._threads.append(thread)
            self._started = True
            return True
