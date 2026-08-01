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
import uuid
from pathlib import Path

from .schema import DIM_DATE, FACT_ATTRIBUTION, FACT_PRODUCTION, FACT_SHOT, FACT_SPC, TABLES

PROJECT = "DefectLab"
COMPATIBILITY_LEVEL = 1567
# Fixed so `logical_id` is reproducible; any constant UUID would do.
LOGICAL_ID_NAMESPACE = uuid.UUID("6f8a1d2c-3b4e-5a6f-8c9d-0e1f2a3b4c5d")

# `definition.pbism` version 1.0 means the model must be TMSL in a model.bim; only 4.0 and above
# permit the TMDL \definition folder we write. Desktop reads this before the folder, so the wrong
# number here surfaces as "Missing required artifact 'model.bim'".
PBISM_VERSION = "4.0"
# 4.0 is the lowest that permits the PBIR \definition folder. The alternative, PBIR-Legacy, is a
# single report.json that Microsoft documents as unsupported for external editing -- writing one
# by hand produced a report with no theme, which Desktop crashed on rather than defaulting.
PBIR_VERSION = "4.0"
# The report definition's own version, separate from the .pbir format version above. It decides
# which files Desktop loads, and 1.0.0 loaded none of the pages -- the renderer then failed
# reading `visualContainers` off the page it had not got.
REPORT_DEFINITION_VERSION = "2.0.0"

_SCHEMA_HOST = "https://developer.microsoft.com/json-schemas/fabric"
_PBIR = f"{_SCHEMA_HOST}/item/report/definition"
# The PBIR versions are not interchangeable and not independent of each other. This set is copied
# from a report Power BI Desktop itself wrote, rather than picked from what the schema repo offers.
SCHEMAS = {
    "pbip": f"{_SCHEMA_HOST}/pbip/pbipProperties/1.0.0/schema.json",
    "pbism": f"{_SCHEMA_HOST}/item/semanticModel/definitionProperties/1.0.0/schema.json",
    "pbir": f"{_SCHEMA_HOST}/item/report/definitionProperties/2.0.0/schema.json",
    "platform": f"{_SCHEMA_HOST}/gitIntegration/platformProperties/2.0.0/schema.json",
    "version": f"{_PBIR}/versionMetadata/1.0.0/schema.json",
    "report": f"{_PBIR}/report/3.1.0/schema.json",
    "pages": f"{_PBIR}/pagesMetadata/1.0.0/schema.json",
    "page": f"{_PBIR}/page/2.0.0/schema.json",
}

# Shipped with Desktop, so the report only names it. `themeCollection` is required by the report
# schema, and its absence is what the renderer failed to read `customTheme` off.
BASE_THEME = {
    "name": "CY25SU12",
    "reportVersionAtImport": {"visual": "2.5.0", "report": "3.1.0", "page": "2.3.0"},
    "type": "SharedResources",
}

# Copied from the published schemas, not inferred from the URLs. Each definitionProperties schema
# marks `$schema` required and pattern-matches it, and Desktop enforces the pattern on open -- so a
# guessed URL is worse than none, which cost one failed open.
SCHEMA_PATTERNS = {
    "pbip": r"^https://developer\.microsoft\.com/json-schemas/fabric"
    r"/pbip/pbipProperties/1\.[0-9]+\.[0-9]+/schema\.json$",
    "pbism": r"^https://developer\.microsoft\.com/json-schemas/fabric"
    r"/item/semanticModel/definitionProperties/1\.[0-9]+\.[0-9]+/schema\.json$",
    "pbir": r"^https://developer\.microsoft\.com/json-schemas/fabric"
    r"/item/report/definitionProperties/2\.[0-9]+\.[0-9]+/schema\.json$",
}
PAGES = ("Line overview", "Why this part", "Cost of quality", "Process control")

# Not "Measures": that is the MDX measures dimension and the tabular schema reserves it, which
# Desktop reports as an unsupported table name.
MEASURE_TABLE = "Metrics"

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
        "measures": _write(model / "definition" / "tables" / f"{MEASURE_TABLE}.tmdl", _measures()),
        "report_platform": _write(report / ".platform", _platform("Report")),
        "report_definition": _write(report / "definition.pbir", _pbir()),
        **_write_report(report / "definition"),
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
    lines = [f"table {MEASURE_TABLE}", ""]
    for name, expression, fmt in MEASURES:
        lines += [
            f"\tmeasure '{name}' = {expression}",
            f"\t\tformatString: {fmt}",
            "",
        ]
    lines += [
        # A calculated table's columns come from its DAX, so the source column is a bracketed
        # reference into that result rather than a name in an external source.
        "\tcolumn _placeholder",
        "\t\tisHidden",
        "\t\tdataType: string",
        "\t\tisNameInferred",
        "\t\tsummarizeBy: none",
        "\t\tsourceColumn: [_placeholder]",
        "",
        f"\tpartition {MEASURE_TABLE} = calculated",
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
    """`ref` only fixes collection order, so there is one per table file and none for cultures,
    which we do not generate."""
    refs = "\n".join(f"ref table {name}" for name in [*TABLES, MEASURE_TABLE])
    return (
        "model Model\n"
        "\tculture: en-GB\n"
        "\tdefaultPowerBIDataSourceVersion: powerBI_V3\n"
        "\tdiscourageImplicitMeasures\n"
        "\tsourceQueryCulture: en-GB\n"
        "\n"
        f"{refs}\n"
    )


def _expressions(exports: Path) -> str:
    """The export folder is a parameter, so the model moves between machines without editing."""
    return (
        f'expression ExportFolder = "{exports.as_posix().replace("/", chr(92))}" meta '
        '[IsParameterQuery=true, Type="Text", IsParameterQueryRequired=true]\n'
    )


def _database() -> str:
    """Named, because every TMDL object is a type followed by a name."""
    return f"database {PROJECT}\n\tcompatibilityLevel: {COMPATIBILITY_LEVEL}\n"


def _pbism() -> str:
    """`version` is the format gate, not decoration: it decides whether Desktop looks for TMDL."""
    payload = {"$schema": SCHEMAS["pbism"], "version": PBISM_VERSION, "settings": {}}
    return json.dumps(payload, indent=2) + "\n"


def _pbir() -> str:
    payload = {
        "$schema": SCHEMAS["pbir"],
        "version": PBIR_VERSION,
        "datasetReference": {"byPath": {"path": f"../{PROJECT}.SemanticModel"}},
    }
    return json.dumps(payload, indent=2) + "\n"


def logical_id(kind: str) -> str:
    """A GUID, because Desktop parses this field as one and refuses the project otherwise.

    Derived by uuid5 rather than uuid4 so regenerating the project keeps the same identity --
    Fabric treats a changed logicalId as a different artifact.
    """
    return str(uuid.uuid5(LOGICAL_ID_NAMESPACE, f"{PROJECT}/{kind}"))


def _platform(kind: str) -> str:
    payload = {
        "$schema": SCHEMAS["platform"],
        "metadata": {"type": kind, "displayName": PROJECT},
        "config": {"version": "2.0", "logicalId": logical_id(kind)},
    }
    return json.dumps(payload, indent=2) + "\n"


def _pbip() -> str:
    """The .pbip shortcut is not an item definition, so its schema lives under a different path
    than the per-item ones -- under fabric/pbip rather than fabric/item."""
    payload = {
        "$schema": SCHEMAS["pbip"],
        "version": "1.0",
        "artifacts": [{"report": {"path": f"{PROJECT}.Report"}}],
        "settings": {"enableAutoRecovery": True},
    }
    return json.dumps(payload, indent=2) + "\n"


def _write_report(definition: Path) -> dict[str, Path]:
    """PBIR: one file per page, each with a published schema. The single-file PBIR-Legacy
    alternative is documented as unsupported for external editing, and behaved like it."""
    written = {
        "report_version": _write(definition / "version.json", _version()),
        "report": _write(definition / "report.json", _report()),
        "pages": _write(definition / "pages" / "pages.json", _pages()),
    }
    for title in PAGES:
        name = _page_name(title)
        written[f"page.{name}"] = _write(definition / "pages" / name / "page.json", _page(title))
    return written


def _page_name(title: str) -> str:
    """Page folder names may hold only word characters and hyphens."""
    return title.lower().replace(" ", "-")


def _version() -> str:
    payload = {"$schema": SCHEMAS["version"], "version": REPORT_DEFINITION_VERSION}
    return json.dumps(payload, indent=2) + "\n"


def _report() -> str:
    payload = {"$schema": SCHEMAS["report"], "themeCollection": {"baseTheme": BASE_THEME}}
    return json.dumps(payload, indent=2) + "\n"


def _pages() -> str:
    order = [_page_name(title) for title in PAGES]
    payload = {"$schema": SCHEMAS["pages"], "pageOrder": order, "activePageName": order[0]}
    return json.dumps(payload, indent=2) + "\n"


def _page(title: str) -> str:
    """Named but empty. Laying visuals out is the part Desktop is actually good at."""
    payload = {
        "$schema": SCHEMAS["page"],
        "name": _page_name(title),
        "displayName": title,
        "displayOption": "FitToPage",
        "width": 1280,
        "height": 720,
    }
    return json.dumps(payload, indent=2) + "\n"
