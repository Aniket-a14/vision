"""What a decision costs.

Two objects, one derived from the other. `CostMatrix` is the pair of numbers a threshold
search needs. `CostModel` is the prevention-appraisal-failure breakdown a quality
department actually keeps books in, and it derives the matrix.

The policy priced here is the one a real gate runs: a flagged part goes to manual
inspection, which is assumed perfect. So a false alarm costs the check and nothing more,
while a missed defect reaches the customer and costs the scrap value multiplied by M.
"""

from __future__ import annotations

from dataclasses import dataclass

# The 1-10-100 rule: a fault costs an order of magnitude more at each stage it survives.
# M is the least defensible number in this layer, so it is reported as a range everywhere
# a headline figure depends on it.
DEFAULT_ESCAPE_MULTIPLIER = 25.0
ESCAPE_MULTIPLIER_RANGE = (10.0, 50.0)


@dataclass(frozen=True, slots=True)
class CostMatrix:
    """The two numbers a threshold search needs: what each error type costs."""

    escape: float = 250.0
    overkill: float = 4.0

    @property
    def ratio(self) -> float:
        return self.escape / self.overkill


@dataclass(frozen=True, slots=True)
class CostModel:
    """Per-part costs for one HPDC cell, in the currency the line is budgeted in."""

    scrap: float = 12.0
    inspection: float = 3.0
    escape_multiplier: float = DEFAULT_ESCAPE_MULTIPLIER
    prevention_per_shot: float = 0.05

    @property
    def escape(self) -> float:
        """External failure: warranty, recall and reputation, priced off the scrap value."""
        return self.scrap * self.escape_multiplier

    @property
    def overkill(self) -> float:
        """Inspection clears a false alarm, so the part ships and only the check is paid for."""
        return self.inspection

    def matrix(self) -> CostMatrix:
        return CostMatrix(escape=self.escape, overkill=self.overkill)

    def with_multiplier(self, multiplier: float) -> CostModel:
        """Used by the sensitivity sweep; every other cost is held fixed."""
        return CostModel(self.scrap, self.inspection, multiplier, self.prevention_per_shot)
