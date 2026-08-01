# The offline bundle

The viva demo has to run with no network. `docker compose up` does not, and the reasons are
invisible until you unplug.

```
python deploy/bundle.py --out dist/defectlab-offline           # build it
python deploy/bundle.py --out dist/defectlab-offline --verify  # check it
```

On the target machine:

```
docker load -i images.tar
docker compose -f docker-compose.yml up -d
```

Then open http://localhost:8080.

## What it contains

| File | Size | Why |
|---|---|---|
| `images.tar` | 260 MB | all three images, `docker save`d |
| `docker-compose.yml` | 4 KB | generated, no `build:` sections |
| `mosquitto.conf` | 1 KB | the broker config the repo bind-mounted |
| `manifest.json` | 1 KB | image list and sha256 digests |
| `README.md` | 1 KB | the two commands above |

260 MB against 1.02 GB of images because `docker save` stores the compressed layer blobs while
`docker images` reports the uncompressed size.

## The three things that break offline

**Every service is declared with `build:`.** Compose would try to build, which needs the base
images and PyPI. The bundle rewrites `build:` to `image:`, which is why compose now carries
explicit `image: defectlab/api:local` tags — without them compose names images after the project
directory and the bundle would save the wrong thing. `image_names()` refuses a service that has
no explicit tag rather than guessing.

**mosquitto bind-mounts its config from the repo**, by absolute path. That path does not exist on
the target. The config travels with the bundle and the mount is repointed at `./mosquitto.conf`.

**`docker compose config` emits `"command": null`** for any service that inherits the image's
CMD. Writing that back sets an *empty* command, and the container starts and does nothing. Nulls
are stripped. This one would have been invisible until the demo: the API would come up fine and
the two line containers would sit there doing nothing at all.

## Verified by deleting everything

Not by inspection. The images and every cached layer were removed —
`docker image prune -af` reclaimed **9.875 GB**, `docker images` showed nothing matching
`defectlab` or `mosquitto` — and then the stack was brought up from the tar alone:

```
docker load -i images.tar          -> 3 images loaded
docker compose up -d               -> 5 containers, api healthy
GET :8080/                         -> 200
GET :8080/api/health               -> {"status":"ok","model_version":"process-xgboost-2"}
GET :8080/api/stream?limit=3       -> 3 event: shot frames
python deploy/bundle.py --verify   -> bundle intact
```

Tampering with one byte of the compose file makes `--verify` report `BUNDLE CORRUPT` and exit 1,
so the manifest is doing work rather than decorating the directory.

## CI

`.github/workflows/ci.yml` had `branches: [main]` while the repository is on `master`, so **it
had never run on a push**. Fixed, and the workflow now has three jobs: `quality` (ruff,
import-linter, pytest), `app` (tsc, oxlint, vite build), and `bundle`, which builds the images,
assembles the bundle and verifies its manifest. The artefact keeps the manifest, compose file and
README — not the multi-gigabyte tar.

Building the bundle in CI is the point: the artefact that has to work on demo day is the bundle,
not the source tree.

## Not covered

The audit volumes are recreated empty on the target, so the chains start from genesis. That is
correct — a chain copied from another machine proves nothing about this one — but it means the
bundle carries no history, and a demo of "here is a month of decisions" would need the volume
exported too.
