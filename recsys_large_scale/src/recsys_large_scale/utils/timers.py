import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Timing:
    label: str
    seconds: float


@dataclass
class Timer:
    """Wall-clock timer. Use as a context manager; results accumulate in .timings."""

    timings: list[Timing] = field(default_factory=list)

    @contextmanager
    def measure(self, label: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.timings.append(Timing(label=label, seconds=elapsed))

    def report(self) -> str:
        return "\n".join(f"{t.label}: {t.seconds:.4f}s" for t in self.timings)
