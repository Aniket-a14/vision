"""The dashboard data contract: a star schema written to CSV and validated on write."""

from .powerbi import write_project
from .schema import TABLES, TableSpec, spec
from .spc_view import parameter_chart, risk_chart
from .tables import ExportInputs
from .writer import BUILDERS, build_tables, validate, write

__all__ = [
    "BUILDERS",
    "TABLES",
    "ExportInputs",
    "TableSpec",
    "build_tables",
    "parameter_chart",
    "risk_chart",
    "spec",
    "validate",
    "write",
    "write_project",
]
