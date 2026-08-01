"""Anchor rules: the smallest set of conditions that pins a prediction in place.

An attribution says how much each group pushed; an anchor says what would have to stay true
for the verdict to hold. Precision is measured by resampling every unanchored feature from
the observed data, so a rule only counts if the prediction survives the rest of the process
moving around it.

This is the greedy construction, not the KL-LUCB bandit search of the original paper: each
step keeps the single best predicate. It is cheaper and deterministic, and it can settle on
a longer rule than the optimal search would find.

A majority-class part often needs no predicates at all: precision is measured against
resampled data, so the empty rule already scores the class base rate. That is reported
honestly as an empty anchor rather than padded with conditions that do no work -- it means
the verdict survives the process moving anywhere, not that the search failed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

DEFAULT_THRESHOLD = 0.95
DEFAULT_BINS = 5
DEFAULT_SAMPLES = 500
MAX_PREDICATES = 4
MIN_EDGES = 2

PredictLabel = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class Predicate:
    """One interval condition on a single column."""

    index: int
    name: str
    lower: float
    upper: float

    def holds(self, features: np.ndarray) -> np.ndarray:
        column = features[:, self.index]
        return (column >= self.lower) & (column <= self.upper)

    def describe(self) -> str:
        return f"{self.name} in [{self.lower:.3g}, {self.upper:.3g}]"


@dataclass(frozen=True, slots=True)
class Anchor:
    """A rule, how often it holds, and how reliably it fixes the prediction."""

    predicates: tuple[Predicate, ...]
    prediction: int
    precision: float
    coverage: float

    def rule(self) -> str:
        if not self.predicates:
            return "(no anchor found)"
        return " AND ".join(predicate.describe() for predicate in self.predicates)

    def describe(self) -> str:
        verdict = "defect" if self.prediction == 1 else "ok"
        return (
            f"IF {self.rule()}\nTHEN {verdict}  "
            f"(precision {self.precision:.3f}, coverage {self.coverage:.3f})"
        )

    def holds(self, features: np.ndarray) -> np.ndarray:
        if not self.predicates:
            return np.ones(len(features), dtype=bool)
        return np.logical_and.reduce([p.holds(features) for p in self.predicates])


def anchor(
    predict_label: PredictLabel,
    instance: np.ndarray,
    background: np.ndarray,
    names: Sequence[str],
    *,
    candidates: Sequence[int] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    bins: int = DEFAULT_BINS,
    samples: int = DEFAULT_SAMPLES,
    seed: int = 42,
) -> Anchor:
    """Grow a rule around one part until its prediction survives perturbation."""
    rng = np.random.default_rng(seed)
    prediction = int(predict_label(instance.reshape(1, -1))[0])
    pool = list(range(len(names))) if candidates is None else list(candidates)
    search = _Search(predict_label, instance, background, names, prediction, bins, samples, rng)
    return search.run(pool, threshold)


@dataclass(frozen=True, slots=True)
class _Search:
    predict_label: PredictLabel
    instance: np.ndarray
    background: np.ndarray
    names: Sequence[str]
    prediction: int
    bins: int
    samples: int
    rng: np.random.Generator

    def run(self, pool: list[int], threshold: float) -> Anchor:
        chosen: list[Predicate] = []
        best = self._anchor_of(chosen)
        for _ in range(MAX_PREDICATES):
            if best.precision >= threshold or not pool:
                break
            candidate, precision = self._best_addition(chosen, pool)
            if candidate is None:
                break
            chosen.append(candidate)
            pool.remove(candidate.index)
            best = self._anchor_of(chosen, precision)
        return best

    def _best_addition(
        self, chosen: list[Predicate], pool: list[int]
    ) -> tuple[Predicate | None, float]:
        """Keep the single predicate that most improves precision; ties go to the first."""
        scored = [(self._precision([*chosen, p]), p) for p in map(self._predicate_for, pool)]
        if not scored:
            return None, 0.0
        precision, candidate = max(scored, key=lambda pair: pair[0])
        return candidate, precision

    def _predicate_for(self, index: int) -> Predicate:
        """The quantile bin the part already sits in; the rule must contain the part."""
        edges = _bin_edges(self.background[:, index], self.bins)
        value = float(self.instance[index])
        position = int(np.clip(np.searchsorted(edges, value, side="left") - 1, 0, len(edges) - 2))
        return Predicate(
            index, self.names[index], float(edges[position]), float(edges[position + 1])
        )

    def _precision(self, predicates: list[Predicate]) -> float:
        """Resample the unanchored columns; the anchored ones stay at the part's values."""
        rows = self.rng.choice(len(self.background), self.samples, replace=True)
        perturbed = self.background[rows].copy()
        for predicate in predicates:
            perturbed[:, predicate.index] = self.instance[predicate.index]
        return float(np.mean(self.predict_label(perturbed) == self.prediction))

    def _coverage(self, predicates: list[Predicate]) -> float:
        if not predicates:
            return 1.0
        holding = np.logical_and.reduce([p.holds(self.background) for p in predicates])
        return float(holding.mean())

    def _anchor_of(self, predicates: list[Predicate], precision: float | None = None) -> Anchor:
        measured = self._precision(predicates) if precision is None else precision
        return Anchor(tuple(predicates), self.prediction, measured, self._coverage(predicates))


def _bin_edges(column: np.ndarray, bins: int) -> np.ndarray:
    """Quantile edges widened at the ends so every observed value falls inside a bin."""
    edges = np.unique(np.quantile(column, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < MIN_EDGES:
        edges = np.array([column.min(), column.max() + 1e-9])
    edges[0], edges[-1] = -np.inf, np.inf
    return edges
