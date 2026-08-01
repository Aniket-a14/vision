"""The generated Power BI project must stay in step with the export contract."""

import json

import pytest

from defectlab.export import TABLES, spec, write_project
from defectlab.export.powerbi import MEASURES, PAGES, PROJECT, RELATIONSHIPS


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("pbip")
    return root, write_project(root, root / "exports")


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
    assert json.loads(written["report"].read_text())["sections"]
    assert (root / f"{PROJECT}.SemanticModel" / "definition.pbism").exists()


def test_the_report_names_every_page(project):
    _, written = project
    titles = [s["displayName"] for s in json.loads(written["report"].read_text())["sections"]]
    assert titles == list(PAGES)


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
