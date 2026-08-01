"""The generated Power BI project must stay in step with the export contract."""

import json
import re
import uuid

import pytest

from defectlab.export import TABLES, spec, write_project
from defectlab.export.powerbi import (
    MEASURE_TABLE,
    MEASURES,
    PAGES,
    PROJECT,
    RELATIONSHIPS,
    SCHEMA_PATTERNS,
    logical_id,
)


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("pbip")
    return root, write_project(root, root / "exports")


@pytest.mark.parametrize("kind", ["Report", "SemanticModel"])
def test_the_logical_id_is_a_guid(project, kind):
    """Desktop parses `config.logicalId` as a System.Guid and refuses to open the project if it
    is anything else. A readable slug there cost one failed open with a 400-line stack trace."""
    root, _ = project
    platform = json.loads((root / f"{PROJECT}.{kind}" / ".platform").read_text())
    assert uuid.UUID(platform["config"]["logicalId"])


def test_the_logical_id_is_stable_across_regeneration():
    """Fabric treats a changed logicalId as a different artifact, so uuid4 would be wrong."""
    assert logical_id("Report") == logical_id("Report")
    assert logical_id("Report") != logical_id("SemanticModel")


@pytest.mark.parametrize(
    ("kind", "path"),
    [
        ("pbip", "{p}.pbip"),
        ("pbism", "{p}.SemanticModel/definition.pbism"),
        ("pbir", "{p}.Report/definition.pbir"),
    ],
)
def test_every_schema_matches_the_pattern_its_own_schema_declares(project, kind, path):
    """Desktop refuses the project on a mismatch, and a guessed URL cost one failed open. The
    patterns are copied from the published schemas, so this test is the spec, not a guess."""
    root, _ = project
    schema = json.loads((root / path.format(p=PROJECT)).read_text())["$schema"]
    assert re.match(SCHEMA_PATTERNS[kind], schema), schema


def test_the_pbism_version_permits_tmdl(project):
    """Version 1.0 restricts the model to TMSL in a model.bim we do not write, which Desktop
    reports as a missing artifact rather than as the version problem it is."""
    root, _ = project
    model = root / f"{PROJECT}.SemanticModel"
    version = json.loads((model / "definition.pbism").read_text())["version"]
    assert float(version) >= 4.0, version
    assert (model / "definition" / "model.tmdl").exists()
    assert not (model / "model.bim").exists()


def test_the_measure_table_avoids_the_reserved_name(project):
    """`Measures` is the MDX measures dimension; the tabular schema reserves it and Desktop
    refuses the model outright rather than renaming it."""
    _, written = project
    assert MEASURE_TABLE != "Measures"
    assert written["measures"].stem == MEASURE_TABLE
    assert f"table {MEASURE_TABLE}" in written["measures"].read_text()


def test_the_model_refs_nothing_it_does_not_write(project):
    """A ref to a missing file is ignored, but the cultures folder was never generated at all."""
    _, written = project
    assert "ref culture" not in written["model"].read_text()


def test_every_contracted_table_becomes_a_tmdl_file(project):
    """The whole point of generating: a table added to the contract cannot be forgotten here."""
    _, written = project
    assert set(TABLES) <= set(written)
    assert all(written[name].exists() for name in TABLES)


def test_each_tmdl_declares_every_contracted_column(project):
    _, written = project
    for name, declared in TABLES.items():
        body = written[name].read_text()
        for column in declared.columns:
            assert f"column {column}" in body


def test_the_project_file_and_model_are_valid_json_where_they_should_be(project):
    root, written = project
    assert json.loads(written["pbip"].read_text())["version"] == "1.0"
    assert json.loads(written["report"].read_text())["themeCollection"]["baseTheme"]
    assert (root / f"{PROJECT}.SemanticModel" / "definition.pbism").exists()


def test_the_report_declares_a_theme(project):
    """The report schema requires `themeCollection`, and a report without one crashed the
    renderer reading `customTheme` off it rather than falling back to a default."""
    _, written = project
    assert "themeCollection" in json.loads(written["report"].read_text())


def test_the_report_names_every_page(project):
    """One folder per page under \\definition\\pages, ordered by pages.json."""
    _, written = project
    order = json.loads(written["pages"].read_text())["pageOrder"]
    titles = [json.loads(written[f"page.{name}"].read_text())["displayName"] for name in order]
    assert titles == list(PAGES)


@pytest.mark.parametrize("title", PAGES)
def test_every_page_name_is_a_legal_folder_name(project, title):
    """Desktop silently ignores a page folder whose name is not word characters or hyphens."""
    _, written = project
    name = json.loads(written[f"page.{title.lower().replace(' ', '-')}"].read_text())["name"]
    assert re.fullmatch(r"[\w-]+", name), name


def test_the_report_points_at_the_semantic_model(project):
    _, written = project
    reference = json.loads(written["report_definition"].read_text())
    assert reference["datasetReference"]["byPath"]["path"].endswith("SemanticModel")


def test_relationships_only_reference_real_columns():
    """A relationship onto a column that is not in the contract silently breaks the model."""
    for from_table, from_column, to_table, to_column in RELATIONSHIPS:
        assert to_column in spec(to_table).columns
        # fact_production gains `date` as a calculated column, so it is not in the contract.
        assert from_column in spec(from_table).columns or from_column == "date"


def test_the_export_folder_is_a_parameter(project):
    """The model must move between machines without editing every query."""
    _, written = project
    assert "IsParameterQuery=true" in written["expressions"].read_text()


def test_every_measure_has_a_format_string(project):
    _, written = project
    body = written["measures"].read_text()
    for name, _, fmt in MEASURES:
        assert f"measure '{name}'" in body
        assert f"formatString: {fmt}" in body


def test_the_headline_measures_exist(project):
    """These are the ones the four pages are built around."""
    _, written = project
    body = written["measures"].read_text()
    for name in ("Escape Rate", "COPQ per 1000 Shots", "Alarms per Hour", "SPC Signal Rate"):
        assert f"measure '{name}'" in body


def test_integer_columns_are_not_typed_as_text(project):
    _, written = project
    body = written["fact_shot"].read_text()
    assert "column shot_id\n\t\tdataType: int64" in body
    assert "column risk\n\t\tdataType: double" in body


def test_the_production_table_gets_a_day_grain_column(project):
    """A relationship to the date table needs day granularity, not a timestamp."""
    _, written = project
    assert (
        "column date = DATEVALUE(fact_production[timestamp])"
        in written["fact_production"].read_text()
    )
