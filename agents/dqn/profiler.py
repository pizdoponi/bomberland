"""
Detailed profiler for diagnosing performance bottlenecks in DQN training.

Usage:
    from profiler import profiler, profile_block

    # Profile a function
    @profiler.profile
    def my_function():
        ...

    # Profile a block
    with profile_block("my_operation"):
        ...

    # Print summary
    profiler.print_summary()
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Dict, List


@dataclass
class ProfileStats:
    """Statistics for a single profiled section."""
    total_time: float = 0.0
    call_count: int = 0
    min_time: float = float('inf')
    max_time: float = 0.0
    recent_times: List[float] = field(default_factory=list)

    def add_sample(self, elapsed: float) -> None:
        self.total_time += elapsed
        self.call_count += 1
        self.min_time = min(self.min_time, elapsed)
        self.max_time = max(self.max_time, elapsed)
        # Keep last 100 samples for recent average
        self.recent_times.append(elapsed)
        if len(self.recent_times) > 100:
            self.recent_times.pop(0)

    @property
    def avg_time(self) -> float:
        return self.total_time / self.call_count if self.call_count > 0 else 0.0

    @property
    def recent_avg(self) -> float:
        return sum(self.recent_times) / len(self.recent_times) if self.recent_times else 0.0


class Profiler:
    """Global profiler for tracking timing across the codebase."""

    def __init__(self):
        self._stats: Dict[str, ProfileStats] = defaultdict(ProfileStats)
        self._start_time = time.perf_counter()
        self._enabled = True
        self._stack: List[str] = []  # Track nested profiling

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def record(self, name: str, elapsed: float) -> None:
        """Record a timing sample for the given name."""
        if self._enabled:
            self._stats[name].add_sample(elapsed)

    @contextmanager
    def block(self, name: str):
        """Context manager for profiling a block of code."""
        if not self._enabled:
            yield
            return

        # Track nested calls with stack depth prefix
        full_name = name
        if self._stack:
            # Show nesting with indentation in the name
            full_name = f"{self._stack[-1]} > {name}"

        self._stack.append(name)
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._stats[full_name].add_sample(elapsed)
            self._stack.pop()

    def profile(self, func):
        """Decorator for profiling a function."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self._enabled:
                return func(*args, **kwargs)

            name = f"{func.__module__}.{func.__qualname__}"
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                self._stats[name].add_sample(elapsed)
        return wrapper

    def profile_async(self, func):
        """Decorator for profiling an async function."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not self._enabled:
                return await func(*args, **kwargs)

            name = f"{func.__module__}.{func.__qualname__}"
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                self._stats[name].add_sample(elapsed)
        return wrapper

    def get_summary(self) -> str:
        """Get a formatted summary of all profiling data."""
        if not self._stats:
            return "grepthis No profiling data collected."

        total_elapsed = time.perf_counter() - self._start_time

        lines = [
            "grepthis ========================================",
            "grepthis PROFILING SUMMARY",
            "grepthis ========================================",
            f"grepthis Total wall time: {total_elapsed:.2f}s",
            "grepthis ",
            f"grepthis {'Section':<55} {'Calls':>8} {'Total(s)':>10} {'Avg(ms)':>10} {'Recent(ms)':>10} {'%':>6}",
            "grepthis " + "-" * 105,
        ]

        # Sort by total time descending
        sorted_stats = sorted(
            self._stats.items(),
            key=lambda x: x[1].total_time,
            reverse=True
        )

        for name, stats in sorted_stats[:30]:  # Top 30
            if stats.call_count == 0:
                continue
            pct = (stats.total_time / total_elapsed) * 100 if total_elapsed > 0 else 0
            lines.append(
                f"grepthis {name:<55} "
                f"{stats.call_count:>8,} "
                f"{stats.total_time:>10.3f} "
                f"{stats.avg_time * 1000:>10.3f} "
                f"{stats.recent_avg * 1000:>10.3f} "
                f"{pct:>6.1f}"
            )

        lines.append("grepthis ========================================")
        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print the profiling summary to stdout."""
        print(self.get_summary())

    def reset(self) -> None:
        """Reset all profiling data."""
        self._stats.clear()
        self._start_time = time.perf_counter()


# Global profiler instance
profiler = Profiler()


# Convenience function for context manager usage
@contextmanager
def profile_block(name: str):
    """Context manager for profiling a block of code."""
    with profiler.block(name):
        yield
