"""Helpers shared by corpus builders."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class BuildReport:
    """Everything a builder dropped or decided, so exclusions are reported rather than silent."""
    corpus: str
    excluded: Counter = field(default_factory=Counter)
    notes: list[str] = field(default_factory=list)

    def drop(self, reason: str, n: int = 1) -> None:
        self.excluded[reason] += n

    def note(self, text: str) -> None:
        self.notes.append(text)

    def to_dict(self) -> dict:
        return {"corpus": self.corpus, "excluded": dict(self.excluded), "notes": self.notes}


class UnionFind:
    """Merges identifiers that refer to the same person."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Deterministic representative: lexicographically smallest id.
            lo, hi = sorted((ra, rb))
            self.parent[hi] = lo
