import threading
import time
from dataclasses import dataclass
from typing import Optional

import psutil


@dataclass
class MemorySample:
    peak_rss_bytes: int
    avg_cpu_percent: float


class PeakMemory:
    def __init__(self, interval_s: float = 0.1):
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._peak = 0
        self._cpu_total = 0.0
        self._cpu_samples = 0
        self._proc = psutil.Process()

    def _run(self) -> None:
        while not self._stop.is_set():
            rss = self._proc.memory_info().rss
            if rss > self._peak:
                self._peak = rss
            self._cpu_total += self._proc.cpu_percent(interval=None)
            self._cpu_samples += 1
            time.sleep(self.interval_s)

    def __enter__(self) -> "PeakMemory":
        self._peak = self._proc.memory_info().rss
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    @property
    def peak_rss_bytes(self) -> int:
        return self._peak

    @property
    def avg_cpu_percent(self) -> float:
        if self._cpu_samples == 0:
            return 0.0
        return self._cpu_total / self._cpu_samples
