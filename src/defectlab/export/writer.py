"""Writes the star schema to disk and checks it against the declared contract.

Validation happens on write, not on read. A dashboard that silently loses a column fails at the
viva; a build that refuses to write fails in CI, where it is cheap.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from . import tables
from .schema import (
    DIM_DATE,
    DIM_GROUP,
    DIM_PARAMETER,
    DIM_RULE,
    FACT_ATTRIBUTION,
    FACT_COST_CURVE,
    FACT_PRODUCTION,
    FACT_SHOT,
    FACT_SPC,
    TABLES,
    spec,
)
from .tables import ExportInputs

BUILDERS = {
    FACT_SHOT: tables.fact_shot,
    FACT_PRODUCTION: tables.fact_production,
    FACT_ATTRIBUTION: tables.fact_attribution,
    FACT_SPC: tables.fact_spc,
    FACT_COST_CURVE: tables.fact_cost_curve,
    DIM_PARAMETER: tables.dim_parameter,
    DIM_GROUP: tables.dim_group,
    DIM_RULE: tables.dim_rule,
    DIM_DATE: tables.dim_date,
}

MANIFEST = "manifest.json"


def build_tables(inputs: ExportInputs) -> dict[str, pd.DataFrame]:
    """Every table in the contract, built and validated but not yet written."""
    built = {name: builder(inputs) for name, builder in BUILDERS.items()}
    for name, frame in built.items():
        validate(name, frame)
    return built


def validate(name: str, frame: pd.DataFrame) -> None:
    """Columns must match the contract exactly, and a key must actually be unique."""
    declared = spec(name)
    missing = set(declared.columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing contracted columns: {sorted(missing)}")
    extra = set(frame.columns) - set(declared.columns)
    if extra:
        raise ValueError(f"{name} has undeclared columns: {sorted(extra)}")
    _check_key(declared, frame)


def write(inputs: ExportInputs, destination: Path) -> dict[str, Path]:
    """CSV rather than parquet: Power BI reads it natively and a marker can inspect it."""
    destination.mkdir(parents=True, exist_ok=True)
    built = build_tables(inputs)
    written = {}
    for name, frame in built.items():
        path = destination / f"{name}.csv"
        frame.to_csv(path, index=False)
        written[name] = path
    _write_manifest(built, destination)
    return written


def _check_key(declared, frame: pd.DataFrame) -> None:
    """An empty table is allowed; a duplicated key is not, because it fans out every measure."""
    if frame.empty:
        return
    duplicated = int(frame.duplicated(subset=list(declared.key)).sum())
    if duplicated:
        raise ValueError(
            f"{declared.name} key {declared.key} repeats on {duplicated} rows; "
            f"grain is meant to be {declared.grain}"
        )


def _write_manifest(built: dict[str, pd.DataFrame], destination: Path) -> Path:
    """Row counts and digests, so a stale refresh in the dashboard is visible, not assumed."""
    payload = {
        "tables": {
            name: {
                "rows": len(frame),
                "columns": list(frame.columns),
                "grain": TABLES[name].grain,
                "sha256": _digest(frame),
            }
            for name, frame in built.items()
        }
    }
    path = destination / MANIFEST
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _digest(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()[:16]
