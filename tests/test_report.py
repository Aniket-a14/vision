"""Figure rendering. Charts are checked for being produced, not for how they look."""

import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from defectlab.report import write_sweep_figures

SEVERITIES = (0.5, 1.0, 2.0)
SEEDS = (1, 2, 3)


@pytest.fixture
def results() -> pd.DataFrame:
    """Vision falls with severity, process is flat, fusion sits between them."""
    rows = []
    for seed in SEEDS:
        for severity in SEVERITIES:
            common = {"seed": seed, "severity": severity, "regime": "inline"}
            vision = 1.0 - 0.05 * severity
            rows.append({**common, "modality": "vision", "roc_auc": vision})
            rows.append({**common, "modality": "process", "roc_auc": 0.84})
            rows.append(
                {**common, "modality": "fusion", "roc_auc": vision + 0.01 * severity * seed}
            )
    return pd.DataFrame(rows)


def test_both_sweep_figures_are_written(results, tmp_path):
    written = write_sweep_figures(results, tmp_path)
    assert {path.name for path in written} == {"degradation_curve.png", "fusion_gain.png"}
    assert all(path.stat().st_size > 0 for path in written)


def test_figures_are_written_into_a_missing_directory(results, tmp_path):
    written = write_sweep_figures(results, tmp_path / "nested" / "figures")
    assert all(path.exists() for path in written)


def test_a_single_seed_still_renders(results, tmp_path):
    """Standard deviation is undefined for one seed; the ribbon must not crash the chart."""
    written = write_sweep_figures(results[results["seed"] == 1], tmp_path)
    assert all(path.stat().st_size > 0 for path in written)
