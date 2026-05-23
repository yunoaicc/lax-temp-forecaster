"""Layer 5a — fair-value pricing of Kalshi LAHIGH contracts from a distribution.

A Contract is one of three kinds, each mapping 1:1 to a tested DistributionSummary
method (strict >, strict <, inclusive between), so fair value re-derives nothing.
No market data here — comparing fair value to live Kalshi quotes is Layer 5b.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .climatology import DistributionSummary


@dataclass(frozen=True)
class Contract:
    kind: str                         # "greater" | "less" | "between"
    threshold: float | None = None    # greater/less
    lo: float | None = None           # between (inclusive)
    hi: float | None = None           # between (inclusive)
    label: str = ""

    @classmethod
    def greater(cls, threshold) -> "Contract":
        return cls(kind="greater", threshold=threshold, label=f"> {threshold}")

    @classmethod
    def less(cls, threshold) -> "Contract":
        return cls(kind="less", threshold=threshold, label=f"< {threshold}")

    @classmethod
    def between(cls, lo, hi) -> "Contract":
        if lo > hi:
            raise ValueError(f"between requires lo <= hi, got lo={lo}, hi={hi}")
        return cls(kind="between", lo=lo, hi=hi, label=f"{lo}-{hi}")

    def probability(self, dist: DistributionSummary) -> float:
        if self.kind == "greater":
            return dist.p_greater_than(self.threshold)
        if self.kind == "less":
            return dist.p_less_than(self.threshold)
        if self.kind == "between":
            return dist.p_between(self.lo, self.hi)
        raise ValueError(f"unknown contract kind: {self.kind!r}")


def price_book(dist: DistributionSummary, contracts: Iterable[Contract]) -> pd.DataFrame:
    """Fair value per contract. Columns: label, kind, fair_prob, fair_cents.

    fair_cents = round(fair_prob * 100); unclamped (this is a fair value, not a
    tradeable quote — a sub-1% tail reports 0, not Kalshi's 1¢ minimum)."""
    rows = []
    for c in contracts:
        p = c.probability(dist)
        rows.append({
            "label": c.label,
            "kind": c.kind,
            "fair_prob": p,
            "fair_cents": int(round(p * 100)),
        })
    return pd.DataFrame(rows, columns=["label", "kind", "fair_prob", "fair_cents"])


def lahigh_ladder(low_edge: int, high_edge: int, width: int = 1) -> list[Contract]:
    """Mutually-exclusive, exhaustive LAHIGH ladder over the integer °F line:

      - bottom tail: Contract.less(low_edge)         -> (-inf, low_edge-1]
      - middle:      Contract.between(a, a+width-1) tiling [low_edge, high_edge)
      - top tail:    Contract.greater(high_edge-1)   -> [high_edge, +inf)

    Disjoint and exhaustive, so fair probabilities sum to 1. Requires
    (high_edge - low_edge) to be a positive multiple of width."""
    if width < 1:
        raise ValueError(f"width must be >= 1, got {width}")
    if low_edge >= high_edge:
        raise ValueError(f"low_edge must be < high_edge, got {low_edge}, {high_edge}")
    if (high_edge - low_edge) % width != 0:
        raise ValueError(
            f"(high_edge - low_edge) must be a multiple of width; "
            f"got span {high_edge - low_edge}, width {width}"
        )
    contracts = [Contract.less(low_edge)]
    for a in range(low_edge, high_edge, width):
        contracts.append(Contract.between(a, a + width - 1))
    contracts.append(Contract.greater(high_edge - 1))
    return contracts
