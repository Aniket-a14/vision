# Deployment

```
docker compose up --build          # broker, API, line simulator, gate, UI
```

Then the operator app is on **http://localhost:8080** and the API on **http://localhost:8000**.

## What comes up

| Service | Role |
|---|---|
| `mosquitto` | the broker |
| `api` | FastAPI: score, explain, prescribe, SSE, audit |
| `app` | nginx serving the built React bundle, proxying `/api` |
| `linesim` | the machine — publishes telemetry, scores nothing |
| `linegate` | the gate — subscribes, scores, publishes verdicts |

`postgres` and `redis` are declared for the historian and the SSE fan-out but nothing in the
current build reads them, so they sit in the `infra` profile and stay down. Bring them up with
`docker compose --profile infra up`. Declaring a dependency the code does not use and letting it
start anyway is how a demo grows a component nobody can explain.

## Decisions worth defending

**The image installs the `serve` extra, not `ml`.** The process-only serving path never imports
torch, shap, interpret or mlflow. The offline study needs all four; it does not run in the
container.

**`xgboost-cpu`, not `xgboost`.** The Linux `xgboost` wheel depends on `nvidia-nccl-cu12` for GPU
training — **289 MB of CUDA runtime** this service never calls. The first build pulled it, which
made the `serve` extra pointless: excluding torch to save space and then shipping NCCL is the
same mistake twice. `xgboost-cpu` is the same import name and the same inference path.

Measured, both built from this Dockerfile:

| wheel | image | xgboost download |
|---|---|---|
| `xgboost` | **2.01 GB** | 94.1 MiB + 289.3 MiB nccl |
| `xgboost-cpu` | **1.02 GB** | 5.4 MiB |

Verified inside the image: `import xgboost` gives 3.3.0 and `site-packages` contains no `nvidia*`
package at all.

**The serving path must not reach OpenCV, and it did.** The first container start crashed with
`ModuleNotFoundError: No module named 'cv2'` — scoring process telemetry required a vision
dependency. The chain: `models/__init__` re-exported `run_cell`, which needed `imaging.Regime`,
which imports cv2. `AblationResult` and `run_cell` were ablation concerns living in
`pipeline.py`; they now sit in `ablation.py`, and `models/__init__` no longer re-exports them.

It then happened a second time through a different door. `cli/commands.py` imported `imaging` at
module scope for the `extract` command, and the CLI is the entry point for *every* command — so
`defectlab line` needed OpenCV as well, and both line containers crash-looped. Those imports are
now inside the commands that use them, with `TYPE_CHECKING` covering the `Regime` annotations.

No unit test could have caught either, because the dev environment has cv2 installed. Both are
now pinned by `test_the_serving_path_never_reaches_opencv`, which imports each module on the
serving path — including the CLI — in a subprocess with `sys.modules['cv2'] = None`.

**nginx proxies `/api`, so production uses no CORS at all.** CORS is a dev-server concession, and
the allowlist in `api/app.py` names only the Vite origins. `proxy_buffering off` is not optional:
behind a buffering proxy the SSE feed arrives in bursts when the buffer fills, or never.

**`start-period=180s` on the healthcheck.** The model is fitted at startup, so the container is
not ready the moment the port opens. A healthcheck that ignores this marks the service unhealthy
and restarts it forever.

**The API and the MQTT gate have separate audit volumes.** The hash chain is **single-writer**:
two processes appending to one file each compute `previous` from their own in-memory head, so
the result would not verify. One log per decision-maker is the honest model anyway — the HTTP
gate and the MQTT gate are two different ones.

**The line simulator and the gate are separate containers.** That is the split a real cell has,
and it is what makes `docker stop defectlab-linesim-1` demonstrate the last will rather than just
stopping a thread.

## Verified against a real broker

Not just the loopback:

- 20 telemetry, 20 verdicts and 2 status messages, read by an **independent subscriber** rather
  than by our own gate. Verdicts carried `audit_hash`, so the MQTT path audits.
- Retained status was delivered to a subscriber that connected **after** the run ended.
- **The last will fired 6.0 s after `kill`** on the publisher — no clean disconnect, and the
  broker published `offline` on the cell's status topic for it. This is the behaviour the
  loopback transport cannot test and the reason MQTT is here rather than a second SSE feed.

## Not done

No TLS, no broker auth (`allow_anonymous true`), no operator identity. Fine for a demo on
localhost, and all three are the difference between this and something that faces a plant
network. Say so before an examiner does.
