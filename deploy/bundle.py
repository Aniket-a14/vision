"""Build a self-contained offline bundle: images, a compose file, and a manifest.

The viva demo has to run on a machine with no network. `docker compose up` does not, for two
reasons that are invisible until you unplug: every service is declared with `build:`, which needs
base images and PyPI, and mosquitto bind-mounts its config from the repo. The bundle rewrites
`build:` to `image:` and carries the config with it.

Deliberately importing nothing from `defectlab`. It must run on the machine that receives the
bundle, where the package is not installed and there may be no venv at all.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "docker-compose.yml"
MOSQUITTO_CONF = ROOT / "deploy" / "mosquitto" / "mosquitto.conf"
IMAGES_TAR = "images.tar"
MANIFEST = "manifest.json"
OFFLINE_COMPOSE = "docker-compose.yml"
DIGEST_CHUNK = 1 << 20


@dataclass(frozen=True, slots=True)
class Bundle:
    """Where the bundle went and what is in it."""

    destination: Path
    images: tuple[str, ...]
    digests: dict[str, str]


def run(command: list[str], capture: bool = True) -> str:
    """Surface docker's own message. A bare CalledProcessError hides the one useful line."""
    result = subprocess.run(command, capture_output=capture, text=True, check=False)
    if result.returncode:
        detail = (result.stderr or "").strip() or f"exit {result.returncode}"
        raise SystemExit(f"{' '.join(command[:3])} failed: {detail}")
    return result.stdout if capture else ""


def resolved_compose() -> dict:
    """Ask compose to resolve the file rather than parsing YAML ourselves."""
    return json.loads(run(["docker", "compose", "-f", str(COMPOSE), "config", "--format", "json"]))


def image_names(config: dict) -> tuple[str, ...]:
    """Every image the stack runs, deduplicated -- the two line services share one."""
    names = {service["image"] for service in config["services"].values() if service.get("image")}
    missing = [name for name, s in config["services"].items() if not s.get("image")]
    if missing:
        raise SystemExit(f"services without an explicit image tag: {missing}")
    return tuple(sorted(names))


def offline_compose(config: dict) -> dict:
    """Strip what only works with a network and a source tree."""
    services = {name: _offline_service(service) for name, service in config["services"].items()}
    return {"name": config.get("name", "defectlab"), "services": services, "volumes": _volumes()}


DROPPED_KEYS = ("build", "networks")


def _offline_service(service: dict) -> dict:
    """Drop `build`, strip nulls, and repoint the one bind mount at the copy travelling with us.

    The nulls matter: `docker compose config` emits `"command": null` for services that inherit
    the image's CMD, and writing that back sets an empty command instead of inheriting one.
    """
    trimmed = {
        key: value
        for key, value in service.items()
        if key not in DROPPED_KEYS and value is not None
    }
    if trimmed.get("volumes"):
        trimmed["volumes"] = [_offline_volume(volume) for volume in trimmed["volumes"]]
    return trimmed


def _offline_volume(volume: dict) -> dict:
    if volume.get("type") != "bind":
        return {key: value for key, value in volume.items() if key != "bind"}
    relative = Path(volume["source"]).name
    return {**volume, "source": f"./{relative}", "bind": volume.get("bind", {})}


def _volumes() -> dict:
    """Named volumes are recreated empty on the target; the audit chains start fresh."""
    return {"api-audit": None, "gate-audit": None}


def digest_of(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(DIGEST_CHUNK):
            sha.update(chunk)
    return sha.hexdigest()


def save_images(images: tuple[str, ...], destination: Path) -> Path:
    tar = destination / IMAGES_TAR
    print(f"saving {len(images)} images to {tar.name} (this is the slow part)", file=sys.stderr)
    run(["docker", "save", "-o", str(tar), *images], capture=False)
    return tar


def write_bundle(destination: Path) -> Bundle:
    destination.mkdir(parents=True, exist_ok=True)
    config = resolved_compose()
    images = image_names(config)
    tar = save_images(images, destination)
    compose_path = destination / OFFLINE_COMPOSE
    compose_path.write_text(json.dumps(offline_compose(config), indent=2), encoding="utf-8")
    shutil.copy2(MOSQUITTO_CONF, destination / MOSQUITTO_CONF.name)
    _write_readme(destination, images)
    digests = {path.name: digest_of(path) for path in (tar, compose_path)}
    _write_manifest(destination, images, digests, tar)
    return Bundle(destination, images, digests)


def _write_manifest(destination: Path, images: tuple, digests: dict, tar: Path) -> None:
    (destination / MANIFEST).write_text(
        json.dumps(
            {
                "images": list(images),
                "digests": digests,
                "images_bytes": tar.stat().st_size,
                "compose_file": OFFLINE_COMPOSE,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_readme(destination: Path, images: tuple[str, ...]) -> None:
    listing = "\n".join(f"  - {name}" for name in images)
    (destination / "README.md").write_text(
        "# DefectLab offline bundle\n\n"
        "Runs with no network. Docker is the only requirement.\n\n"
        "```\n"
        f"docker load -i {IMAGES_TAR}\n"
        f"docker compose -f {OFFLINE_COMPOSE} up -d\n"
        "```\n\n"
        "Then open http://localhost:8080.\n\n"
        f"Images:\n{listing}\n\n"
        "`docker compose down -v` removes the containers and the audit volumes.\n\n"
        "The compose file here has no `build:` sections on purpose: building needs base images\n"
        "and PyPI, which is exactly what is unavailable. It is JSON rather than YAML because it\n"
        "is generated; compose reads both.\n",
        encoding="utf-8",
    )


def verify(destination: Path) -> int:
    """Re-check the bundle against its own manifest. Cheap, and the failure mode is silent."""
    manifest = json.loads((destination / MANIFEST).read_text(encoding="utf-8"))
    problems = [
        f"{name}: expected {expected[:12]}, found {digest_of(destination / name)[:12]}"
        for name, expected in manifest["digests"].items()
        if digest_of(destination / name) != expected
    ]
    for problem in problems:
        print(problem, file=sys.stderr)
    print("bundle intact" if not problems else "BUNDLE CORRUPT")
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="build or verify the offline demo bundle")
    parser.add_argument("--out", type=Path, default=ROOT / "dist" / "defectlab-offline")
    parser.add_argument("--verify", action="store_true", help="check an existing bundle instead")
    args = parser.parse_args(argv)
    if args.verify:
        return verify(args.out)
    bundle = write_bundle(args.out)
    size = (bundle.destination / IMAGES_TAR).stat().st_size / 1e9
    print(f"wrote {bundle.destination}  ({len(bundle.images)} images, {size:.2f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
