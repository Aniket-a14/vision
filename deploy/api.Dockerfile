# The scoring service.
#
# Installs the `serve` extra, not `ml`: the process-only path never imports torch, shap,
# interpret or mlflow, and leaving them out is most of the image. The offline study still needs
# them -- it does not run in here.

FROM python:3.13-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv

WORKDIR /srv

# Dependencies resolve from the manifest alone, so a source edit does not re-download the wheels.
#
# `xgboost-cpu`, not `xgboost`: the Linux wheel of the latter depends on nvidia-nccl-cu12 for GPU
# training, which is 289 MB of CUDA runtime this service will never call. Same import name, same
# inference. Excluding torch and then shipping NCCL anyway would have been the same mistake twice.
COPY pyproject.toml README.md ./
RUN uv venv /opt/venv && VIRTUAL_ENV=/opt/venv uv pip install --no-cache \
    fastapi uvicorn[standard] paho-mqtt "scikit-learn>=1.9,<2" "xgboost-cpu>=3.3,<4" \
    "numpy>=2.1,<2.5" "pandas>=2.2,<3" "scipy>=1.18,<2" "pyarrow>=25.0" \
    "pandera>=0.32" "pydantic>=2.13" "pydantic-settings>=2.14"

COPY src ./src
RUN VIRTUAL_ENV=/opt/venv uv pip install --no-cache --no-deps -e .

ENV PATH="/opt/venv/bin:$PATH"

# Non-root: the service writes only the audit log, and that is a mounted volume.
RUN useradd --create-home --uid 10001 defectlab \
    && mkdir -p /srv/data/processed \
    && chown -R defectlab:defectlab /srv/data
USER defectlab

EXPOSE 8000

# The model is fitted at startup, so the container is not ready the moment the port opens.
HEALTHCHECK --interval=15s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "--factory", "defectlab.api.app:create_app", \
     "--host", "0.0.0.0", "--port", "8000"]
