"""The dashboard data contract: grain, keys, and what a report is allowed to rely on."""

import json

import numpy as np
import pandas as pd
import pytest

from defectlab.export import (
    TABLES,
    ExportInputs,
    build_tables,
    risk_chart,
    spec,
    validate,
    write,
)
from defectlab.export.schema import (
    DIM_DATE,
    DIM_PARAMETER,
    FACT_ATTRIBUTION,
    FACT_PRODUCTION,
    FACT_SHOT,
    FACT_SPC,
    SECONDS_PER_SHOT,
)
from defectlab.twin import FEATURES, SETPOINTS, TwinConfig, run_line, score

SHOTS = 400
GROUPS = ("thermal", "pressure", "fill", "chemistry", "tooling", "image")


@pytest.fixture(scope="module")
def inputs():
    """A coherent run: the same shots scored, attributed and charted, in production order."""
    rng = np.random.default_rng(0)
    shots = score(run_line(SHOTS, TwinConfig(seed=3)), TwinConfig(seed=3), target_prevalence=0.05)
    risk = np.clip(shots["true_defect_prob"].to_numpy() + rng.normal(0, 0.02, SHOTS), 0.001, 0.999)
    curve = pd.DataFrame(
        {
            "threshold": np.linspace(0.01, 0.99, 50),
            "per_shot": np.linspace(2.0, 5.0, 50),
            "escape_rate": np.linspace(0.0, 1.0, 50),
            "overkill_rate": np.linspace(1.0, 0.0, 50),
            "alert_rate": np.linspace(1.0, 0.0, 50),
        }
    )
    attribution = pd.DataFrame(rng.normal(size=(SHOTS, len(GROUPS))), columns=list(GROUPS))
    line = score(run_line(SHOTS, TwinConfig(seed=9)), TwinConfig(seed=9), target_prevalence=0.03)
    line_risk = np.clip(line["true_defect_prob"].to_numpy(), 0.001, 0.999)
    return ExportInputs(
        shots, risk, 0.2, curve, attribution, risk_chart(line_risk), line, line_risk
    )


def test_every_contracted_table_is_built(inputs):
    assert set(build_tables(inputs)) == set(TABLES)


def test_each_table_matches_its_declared_columns(inputs):
    for name, frame in build_tables(inputs).items():
        assert list(frame.columns) == list(spec(name).columns) or set(frame.columns) == set(
            spec(name).columns
        )


def test_an_undeclared_column_is_rejected(inputs):
    frame = build_tables(inputs)[FACT_SHOT].assign(surprise=1)
    with pytest.raises(ValueError, match="undeclared columns"):
        validate(FACT_SHOT, frame)


def test_a_missing_column_is_rejected(inputs):
    frame = build_tables(inputs)[FACT_SHOT].drop(columns=["risk"])
    with pytest.raises(ValueError, match="missing contracted columns"):
        validate(FACT_SHOT, frame)


def test_a_duplicated_key_is_rejected(inputs):
    """A repeated key silently fans out every measure in the report; refuse it at build time."""
    frame = build_tables(inputs)[FACT_SHOT]
    with pytest.raises(ValueError, match="repeats on"):
        validate(FACT_SHOT, pd.concat([frame, frame.head(1)], ignore_index=True))


def test_the_shot_fact_is_one_row_per_shot(inputs):
    assert len(build_tables(inputs)[FACT_SHOT]) == SHOTS


def test_outcomes_are_named_not_numeric(inputs):
    """A dashboard legend reading 0/1/2/3 helps nobody."""
    outcomes = set(build_tables(inputs)[FACT_SHOT]["outcome"])
    assert outcomes <= {"true_negative", "false_positive", "false_negative", "true_positive"}


def test_outcomes_agree_with_the_label_and_the_flag(inputs):
    frame = build_tables(inputs)[FACT_SHOT]
    escaped = frame[frame["outcome"] == "false_negative"]
    assert (escaped["label"] == 1).all()
    assert (escaped["flagged"] == 0).all()


def test_timestamps_advance_one_cycle_per_shot(inputs):
    stamps = pd.to_datetime(build_tables(inputs)[FACT_PRODUCTION]["timestamp"])
    gaps = stamps.diff().dropna().dt.total_seconds().unique()
    assert gaps == pytest.approx([SECONDS_PER_SHOT])


def test_the_evaluation_set_carries_no_clock(inputs):
    """It is oversampled and grouped by label, so a timestamp on it would invent a time axis."""
    assert "timestamp" not in build_tables(inputs)[FACT_SHOT].columns


def test_the_production_run_is_absent_but_valid_when_not_supplied(inputs):
    bare = ExportInputs(inputs.shots, inputs.risk, inputs.threshold, inputs.cost_curve)
    assert build_tables(bare)[FACT_PRODUCTION].empty


def test_attribution_is_long_and_covers_every_shot(inputs):
    frame = build_tables(inputs)[FACT_ATTRIBUTION]
    assert len(frame) == SHOTS * len(GROUPS)
    assert set(frame["group"]) == set(GROUPS)


def test_attribution_is_empty_but_valid_when_not_supplied(inputs):
    """A process-only run has no image group; the table must still exist for the report to bind."""
    bare = ExportInputs(inputs.shots, inputs.risk, inputs.threshold, inputs.cost_curve)
    assert build_tables(bare)[FACT_ATTRIBUTION].empty


def test_the_parameter_dimension_covers_every_feature(inputs):
    frame = build_tables(inputs)[DIM_PARAMETER]
    assert set(frame["parameter"]) == set(FEATURES)


def test_only_setpoints_are_marked_as_levers(inputs):
    frame = build_tables(inputs)[DIM_PARAMETER]
    levers = set(frame.loc[frame["is_lever"] == 1, "parameter"])
    assert levers <= set(SETPOINTS)
    assert "si_content_pct" not in levers


def test_the_date_dimension_has_one_row_per_day(inputs):
    frame = build_tables(inputs)[DIM_DATE]
    assert frame["date"].is_unique
    assert len(frame) >= 1


def test_the_spc_fact_carries_its_limits_on_every_row(inputs):
    """Repeating the limits is redundant in a database and exactly right for a line chart."""
    frame = build_tables(inputs)[FACT_SPC]
    assert frame["centre"].nunique() == 1
    assert (frame["upper"] > frame["lower"]).all()


def test_every_spc_signal_names_a_rule(inputs):
    """The moving-range chart signals outside Nelson's eight; a signal the report cannot name is a bug."""
    frame = build_tables(inputs)[FACT_SPC]
    assert (frame[frame["signal"] == 1]["rule"] != "").all()
    assert (frame[frame["signal"] == 0]["rule"] == "").all()


def test_the_risk_chart_is_drawn_on_the_logit():
    """A probability bounded on [0, 1] and piled up near zero is not a Shewhart statistic."""
    from scipy import stats

    from defectlab.export.spc_view import risk_chart as chart

    rng = np.random.default_rng(0)
    risk = np.clip(rng.beta(0.3, 6.0, 800), 1e-4, 1 - 1e-4)
    drawn = chart(risk)
    assert abs(stats.skew(drawn["value"])) < abs(stats.skew(risk))
    assert drawn["signal"].mean() < 0.30


def test_write_produces_a_csv_per_table_and_a_manifest(inputs, tmp_path):
    written = write(inputs, tmp_path)
    assert set(written) == set(TABLES)
    assert all(path.exists() for path in written.values())
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert set(manifest["tables"]) == set(TABLES)


def test_the_manifest_records_row_counts_and_a_digest(inputs, tmp_path):
    """A stale refresh in the dashboard should be visible, not assumed."""
    write(inputs, tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["tables"][FACT_SHOT]["rows"] == SHOTS
    assert len(manifest["tables"][FACT_SHOT]["sha256"]) == 16


def test_the_written_csv_round_trips(inputs, tmp_path):
    written = write(inputs, tmp_path)
    reloaded = pd.read_csv(written[FACT_SHOT])
    assert list(reloaded.columns) == list(spec(FACT_SHOT).columns)
    assert len(reloaded) == SHOTS
