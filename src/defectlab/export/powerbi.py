"""Generates a Power BI project (PBIP) from the export contract.

PBIP is the plain-text form of a Power BI file: a TMDL semantic model plus a report definition.
Power BI Desktop opens it and can save it as a `.pbix`, which is a binary container nothing else
can author. Generating the model from `schema.TABLES` rather than hand-writing nine TMDL files
means the model cannot drift from the data that feeds it.

What is generated is the whole semantic model -- typed columns, relationships, a date table and
the DAX measures -- and a report with named but empty pages. Laying visuals out is a few minutes
of drag-and-drop in Desktop and is the part Desktop is actually good at; hand-writing visual
JSON blind is how you produce a file that will not open.
"""

from __future__ import annotations

import json
from pathlib import Path

from .schema import DIM_DATE, FACT_ATTRIBUTION, FACT_PRODUCTION, FACT_SHOT, FACT_SPC, TABLES

PROJECT = "DefectLab"
COMPATIBILITY_LEVEL = 1567
PAGES = ("Line overview", "Why this part", "Cost of quality", "Process control")

INTEGER_COLUMNS = frozenset(
    {
        "shot_id",
        "lot_id",
        "die_id",
        "shift_id",
        "label",
        "flagged",
        "is_lever",
        "signal",
        "number",
        "year",
        "month",
        "day",
    }
)
DATETIME_COLUMNS = frozenset({"timestamp", "date"})

# One relationship per line: from the many side to the one side.
RELATIONSHIPS = (
    (FACT_ATTRIBUTION, "shot_id", FACT_SHOT, "shot_id"),
    (FACT_ATTRIBUTION, "group", "dim_group", "group"),
    (FACT_SPC, "shot_id", FACT_PRODUCTION, "shot_id"),
    (FACT_SPC, "rule", "dim_rule", "rule"),
    (FACT_PRODUCTION, "date", DIM_DATE, "date"),
)

MEASURES: tuple[tuple[str, str, str], ...] = (
    ("Shots", "COUNTROWS(fact_shot)", "#,0"),
    ("Defects", "CALCULATE(COUNTROWS(fact_shot), fact_shot[label] = 1)", "#,0"),
    ("Flagged", "CALCULATE(COUNTROWS(fact_shot), fact_shot[flagged] = 1)", "#,0"),
    ("Escapes", 'CALCULATE(COUNTROWS(fact_shot), fact_shot[outcome] = "false_negative")', "#,0"),
    ("Overkills", 'CALCULATE(COUNTROWS(fact_shot), fact_shot[outcome] = "false_positive")', "#,0"),
    ("Escape Rate", "DIVIDE([Escapes], [Defects])", "0.0%"),
    ("Overkill Rate", "DIVIDE([Overkills], [Shots] - [Defects])", "0.0%"),
    ("Alert Rate", "DIVIDE([Flagged], [Shots])", "0.0%"),
    ("Selected Threshold", "SELECTEDVALUE(fact_cost_curve[threshold])", "0.000"),
    ("Cost per Shot", "AVERAGE(fact_cost_curve[per_shot])", "#,0.00"),
    ("COPQ per 1000 Shots", "[Cost per Shot] * 1000", "#,0"),
    ("Production Shots", "COUNTROWS(fact_production)", "#,0"),
    ("SPC Signals", "CALCULATE(COUNTROWS(fact_spc), fact_spc[signal] = 1)", "#,0"),
    ("SPC Signal Rate", "DIVIDE([SPC Signals], COUNTROWS(fact_spc))", "0.0%"),
    # One shot a minute, so shots / 60 is hours. Comparable to the ISA-18.2 6-12 per hour band.
    ("Alarms per Hour", "DIVIDE([SPC Signals], DIVIDE([Production Shots], 60))", "0.0"),
)


def write_project(destination: Path, exports: Path) -> dict[str, Path]:
    """Write the whole PBIP tree. `exports` is baked in as the default folder parameter."""
    model = destination / f"{PROJECT}.SemanticModel"
    report = destination / f"{PROJECT}.Report"
    written = {
        "pbip": _write(destination / f"{PROJECT}.pbip", _pbip()),
        "model_platform": _write(model / ".platform", _platform("SemanticModel")),
        "model_settings": _write(model / "definition.pbism", _pbism()),
        "database": _write(model / "definition" / "database.tmdl", _database()),
        "model": _write(model / "definition" / "model.tmdl", _model()),
        "expressions": _write(model / "definition" / "expressions.tmdl", _expressions(exports)),
        "relationships": _write(model / "definition" / "relationships.tmdl", _relationships()),
        "measures": _write(model / "definition" / "tables" / "Measures.tmdl", _measures()),
        "report_platform": _write(report / ".platform", _platform("Report")),
        "report_definition": _write(report / "definition.pbir", _pbir()),
        "report": _write(report / "report.json", _report()),
    }
    for name in TABLES:
        path = model / "definition" / "tables" / f"{name}.tmdl"
        written[name] = _write(path, _table(name))
    return written


def _write(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def _data_type(column: str) -> str:
    if column in INTEGER_COLUMNS:
        return "int64"
    if column in DATETIME_COLUMNS:
        return "dateTime"
    return "double" if _is_numeric(column) else "string"


def _is_numeric(column: str) -> bool:
    numeric = {
        "risk",
        "contribution",
        "value",
        "centre",
        "lower",
        "upper",
        "threshold",
        "per_shot",
        "escape_rate",
        "overkill_rate",
        "alert_rate",
        "nominal",
    }
    return column in numeric


def _table(name: str) -> str:
    declared = TABLES[name]
    lines = [f"table {name}", ""]
    for column in declared.columns:
        lines += _column(column)
    if name == FACT_PRODUCTION:
        lines += _calculated_date()
    lines += _partition(name, declared.columns)
    return "\n".join(lines) + "\n"


def _column(column: str) -> list[str]:
    kind = _data_type(column)
    return [
        f"\tcolumn {column}",
        f"\t\tdataType: {kind}",
        "\t\tsummarizeBy: none",
        f"\t\tsourceColumn: {column}",
        *(["\t\tformatString: General Date"] if kind == "dateTime" else []),
        "",
    ]


def _calculated_date() -> list[str]:
    """A relationship to the date table needs day granularity, not a timestamp."""
    return [
        "\tcolumn date = DATEVALUE(fact_production[timestamp])",
        "\t\tdataType: dateTime",
        "\t\tformatString: Short Date",
        "\t\tsummarizeBy: none",
        "",
    ]


def _partition(name: str, columns: tuple[str, ...]) -> list[str]:
    types = ", ".join(f'{{"{column}", {_m_type(column)}}}' for column in columns)
    return [
        f"\tpartition {name} = m",
        "\t\tmode: import",
        "\t\tsource =",
        "\t\t\t\tlet",
        f'\t\t\t\t    Source = Csv.Document(File.Contents(ExportFolder & "\\{name}.csv"), '
        '[Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]),',
        "\t\t\t\t    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),",
        f"\t\t\t\t    Typed = Table.TransformColumnTypes(Promoted, {{{types}}})",
        "\t\t\t\tin",
        "\t\t\t\t    Typed",
        "",
    ]


def _m_type(column: str) -> str:
    kind = _data_type(column)
    return {"int64": "Int64.Type", "double": "type number", "dateTime": "type datetime"}.get(
        kind, "type text"
    )


def _measures() -> str:
    lines = ["table Measures", ""]
    for name, expression, fmt in MEASURES:
        lines += [
            f"\tmeasure '{name}' = {expression}",
            f"\t\tformatString: {fmt}",
            "",
        ]
    lines += [
        "\tcolumn _placeholder",
        "\t\tisHidden",
        "\t\tdataType: string",
        "\t\tsummarizeBy: none",
        "\t\tsourceColumn: _placeholder",
        "",
        "\tpartition Measures = calculated",
        "\t\tmode: import",
        '\t\tsource = ROW("_placeholder", BLANK())',
        "",
    ]
    return "\n".join(lines) + "\n"


def _relationships() -> str:
    lines = []
    for index, (from_table, from_column, to_table, to_column) in enumerate(RELATIONSHIPS):
        lines += [
            f"relationship rel_{index}",
            f"\tfromColumn: {from_table}.{from_column}",
            f"\ttoColumn: {to_table}.{to_column}",
            "",
        ]
    return "\n".join(lines) + "\n"


def _model() -> str:
    refs = "\n".join(f"ref table {name}" for name in [*TABLES, "Measures"])
    return (
        "model Model\n"
        "\tculture: en-GB\n"
        "\tdefaultPowerBIDataSourceVersion: powerBI_V3\n"
        "\tdiscourageImplicitMeasures\n"
        "\tsourceQueryCulture: en-GB\n"
        "\n"
        f"{refs}\n"
        "\nref cultures en-GB\n"
    )


def _expressions(exports: Path) -> str:
    """The export folder is a parameter, so the model moves between machines without editing."""
    return (
        f'expression ExportFolder = "{exports.as_posix().replace("/", chr(92))}" meta '
        '[IsParameterQuery=true, Type="Text", IsParameterQueryRequired=true]\n'
        "\tlineageTag: export-folder\n"
        "\tqueryGroup: Parameters\n"
    )


def _database() -> str:
    return f"database\n\tcompatibilityLevel: {COMPATIBILITY_LEVEL}\n"


def _pbism() -> str:
    return json.dumps({"version": "1.0", "settings": {}}, indent=2) + "\n"


def _pbir() -> str:
    payload = {
        "version": "1.0",
        "datasetReference": {"byPath": {"path": f"../{PROJECT}.SemanticModel"}},
    }
    return json.dumps(payload, indent=2) + "\n"


def _platform(kind: str) -> str:
    payload = {
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/gitIntegration/platformProperties/2.0.0/schema.json",
        "metadata": {"type": kind, "displayName": PROJECT},
        "config": {"version": "2.0", "logicalId": f"{PROJECT.lower()}-{kind.lower()}"},
    }
    return json.dumps(payload, indent=2) + "\n"


def _pbip() -> str:
    payload = {
        "version": "1.0",
        "artifacts": [{"report": {"path": f"{PROJECT}.Report"}}],
        "settings": {"enableAutoRecovery": True},
    }
    return json.dumps(payload, indent=2) + "\n"


def _report() -> str:
    """Named but empty pages. Visual JSON written blind is how a PBIP fails to open."""
    sections = [
        {
            "name": f"page{index}",
            "displayName": title,
            "ordinal": index,
            "visualContainers": [],
            "config": "{}",
            "width": 1280,
            "height": 720,
            "displayOption": 1,
        }
        for index, title in enumerate(PAGES)
    ]
    payload = {
        "config": json.dumps({"version": "5.43", "activeSectionIndex": 0}),
        "layoutOptimization": 0,
        "resourcePackages": [],
        "sections": sections,
    }
    return json.dumps(payload, indent=2) + "\n"
