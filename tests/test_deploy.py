"""The deployment manifests, checked against the package they deploy.

The Dockerfile lists its packages explicitly rather than installing `.[serve]`, because
resolving from the manifest alone is what keeps a source edit from re-downloading every wheel.
That duplication is a drift risk, so it is pinned here: add a runtime dependency and this fails
until the image learns about it.
"""

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "docker-compose.yml"
API_DOCKERFILE = ROOT / "deploy" / "api.Dockerfile"
NGINX = ROOT / "deploy" / "nginx" / "default.conf"


def _manifest() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _requirement_names(specifiers: list[str]) -> set[str]:
    """`uvicorn[standard]>=0.52` and `uvicorn` are the same package to this check."""
    return {re.split(r"[><=\[!~]", item)[0].strip() for item in specifiers}


def _dockerfile_packages() -> set[str]:
    text = API_DOCKERFILE.read_text(encoding="utf-8")
    install = re.search(r"uv pip install --no-cache \\\n(.*?)\n\n", text, re.S)
    assert install, "could not find the dependency install step"
    quoted = re.findall(r'"([^"]+)"|(\b[a-z0-9-]+(?:\[[a-z]+\])?)(?=\s|\\)', install.group(1))
    found = {a or b for a, b in quoted}
    return _requirement_names(sorted(found - {""}))


def test_the_image_installs_every_runtime_dependency():
    manifest = _manifest()["project"]
    required = _requirement_names(manifest["dependencies"])
    missing = required - _dockerfile_packages()
    assert not missing, f"api.Dockerfile is missing runtime dependencies: {sorted(missing)}"


def test_the_image_installs_the_serve_extra():
    extra = _manifest()["project"]["optional-dependencies"]["serve"]
    declared = _requirement_names(extra)
    installed = _dockerfile_packages()
    # xgboost-cpu is the same module without the CUDA runtime; the extra names the plain wheel.
    missing = {
        name for name in declared if name not in installed and f"{name}-cpu" not in installed
    }
    assert not missing, f"api.Dockerfile is missing serve dependencies: {sorted(missing)}"


def test_the_image_excludes_the_gpu_wheel():
    """`xgboost` on Linux pulls nvidia-nccl-cu12: 289 MB of CUDA the service never calls."""
    packages = _dockerfile_packages()
    assert "xgboost-cpu" in packages
    assert "xgboost" not in packages


def test_the_image_excludes_the_heavy_study_dependencies():
    """torch, shap, interpret and mlflow belong to the offline study, not the serving path."""
    packages = _dockerfile_packages()
    assert not packages & {"torch", "torchvision", "timm", "shap", "interpret", "mlflow"}


SERVING_MODULES = (
    "defectlab.api.app",
    "defectlab.api.scoring",
    "defectlab.api.explaining",
    "defectlab.edge.gate",
    "defectlab.models.pipeline",
    # The container runs `defectlab line`, so the CLI entry point is on the serving path too.
    "defectlab.cli.main",
    "defectlab.cli.commands",
)


@pytest.mark.parametrize("module", SERVING_MODULES)
def test_the_serving_path_never_reaches_opencv(module: str):
    """The container has no OpenCV, so an import chain that reaches it is a startup crash.

    It happened: `models/__init__` re-exported `run_cell`, which needed `imaging.Regime`, which
    imports cv2 -- so scoring process telemetry required a vision dependency. A unit test would
    not have caught it because the dev environment has cv2 installed.
    """
    import subprocess
    import sys

    probe = f"import sys, importlib;sys.modules['cv2'] = None;importlib.import_module({module!r})"
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, f"{module} reaches cv2:\n{result.stderr[-800:]}"


def _compose() -> str:
    return COMPOSE.read_text(encoding="utf-8")


def test_the_api_and_the_gate_do_not_share_an_audit_volume():
    """The hash chain is single-writer: two appenders each compute `previous` from their own
    in-memory head, so a shared file would not verify."""
    text = _compose()
    assert "api-audit:/srv/data/processed" in text
    assert "gate-audit:/srv/data/processed" in text


def test_the_line_containers_do_not_inherit_the_http_healthcheck():
    """The image probes /health. Neither line container serves HTTP, so an inherited check would
    mark them unhealthy forever and block anything waiting on `service_healthy`."""
    text = _compose()
    for service in ("linesim", "linegate"):
        block = text.split(f"  {service}:", 1)[1].split("\n\n", 1)[0]
        assert "healthcheck:\n      disable: true" in block, f"{service} inherits /health"


def test_unused_infrastructure_is_behind_a_profile():
    """Nothing in the build reads postgres or redis, so nothing should start them."""
    for service in ("postgres", "redis"):
        block = _compose().split(f"  {service}:", 1)[1]
        assert 'profiles: ["infra"]' in block.split("\n\n", 1)[0]


def test_the_proxy_does_not_buffer_the_event_stream():
    """Behind a buffering proxy the live line arrives in bursts when the buffer fills, or never."""
    assert "proxy_buffering off" in NGINX.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "path", [COMPOSE, API_DOCKERFILE, NGINX, ROOT / "deploy" / "app.Dockerfile"]
)
def test_the_deployment_files_exist(path: Path):
    assert path.is_file()
