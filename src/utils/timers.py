import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class TimerResult:
    start: float
    end: float
    elapsed: float


class Timer:
    def __init__(self):
        self._start: Optional[float] = None
        self._end: Optional[float] = None

    def __enter__(self) -> "Timer":
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._end = time.time()

    @property
    def elapsed(self) -> float:
        if self._start is None:
            return 0.0
        end = self._end if self._end is not None else time.time()
        return end - self._start

