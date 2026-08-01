"""The offline bundle transform, without invoking docker.

What matters here is the rewrite from a compose file that needs a network into one that does
not. The docker calls are exercised by actually building the bundle; these pin the logic that
is silent when it goes wrong -- a leftover `build:` section only fails once you are offline.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))

import bundle

CONFIG = {
    "name": "defectlab",
    "services": {
        "api": {
            "build": {"context": ".", "dockerfile": "deploy/api.Dockerfile"},
            "image": "defectlab/api:local",
            "command": None,
            "entrypoint": None,
            "networks": {"default": None},
            "volumes": [{"type": "volume", "source": "api-audit", "target": "/srv/data"}],
        },
        "mosquitto": {
            "image": "eclipse-mosquitto:2.0",
            "volumes": [
                {
                    "type": "bind",
                    "source": "C:\\repo\\deploy\\mosquitto\\mosquitto.conf",
                    "target": "/mosquitto/config/mosquitto.conf",
                    "read_only": True,
                }
            ],
        },
    },
}


@pytest.fixture
def offline() -> dict:
    return bundle.offline_compose(CONFIG)


def test_no_service_still_builds_from_source(offline):
    """Building needs base images and PyPI, which is exactly what is missing offline."""
    assert all("build" not in service for service in offline["services"].values())


def test_every_service_keeps_an_image(offline):
    assert all(service["image"] for service in offline["services"].values())


def test_a_service_without_an_image_tag_is_refused():
    """Compose would name it after the project and the bundle would save the wrong thing."""
    broken = {"services": {"api": {"build": {"context": "."}}}}
    with pytest.raises(SystemExit, match="without an explicit image tag"):
        bundle.image_names(broken)


def test_images_are_deduplicated():
    """The two line services run the same image; saving it twice would double the tar."""
    config = {
        "services": {
            "a": {"image": "defectlab/api:local"},
            "b": {"image": "defectlab/api:local"},
            "c": {"image": "eclipse-mosquitto:2.0"},
        }
    }
    assert bundle.image_names(config) == ("defectlab/api:local", "eclipse-mosquitto:2.0")


def test_inherited_commands_are_not_overwritten_with_null(offline):
    """`docker compose config` emits "command": null for a service that inherits the image CMD.
    Writing that back sets an empty command, and the container starts and does nothing."""
    assert "command" not in offline["services"]["api"]
    assert "entrypoint" not in offline["services"]["api"]


def test_the_bind_mount_points_at_the_copy_that_travels(offline):
    """An absolute path into the build machine's repo does not exist on the target."""
    mount = offline["services"]["mosquitto"]["volumes"][0]
    assert mount["source"] == "./mosquitto.conf"
    assert mount["read_only"]


@pytest.mark.parametrize(
    "source",
    ["C:\\repo\\deploy\\mosquitto\\mosquitto.conf", "/home/ci/repo/deploy/mosquitto.conf"],
)
def test_either_path_convention_is_stripped(source):
    """The bundle is built on Windows and unpacked on Linux, so the separator in the compose file
    is not the one the interpreter is using. `Path` split only one of them and CI caught it."""
    volume = {"type": "bind", "source": source, "target": "/mosquitto/config/mosquitto.conf"}
    assert bundle._offline_volume(volume)["source"] == "./mosquitto.conf"


def test_named_volumes_are_declared(offline):
    """Compose refuses to start a service referencing a volume the file never declares."""
    assert set(offline["volumes"]) == {"api-audit", "gate-audit"}


def test_the_generated_compose_is_json_serialisable(offline):
    assert json.loads(json.dumps(offline))["name"] == "defectlab"


def test_a_corrupt_bundle_is_reported(tmp_path):
    """The manifest is worth nothing if nothing ever checks it."""
    payload = tmp_path / "images.tar"
    payload.write_bytes(b"not really a tar")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"digests": {"images.tar": "0" * 64}}), encoding="utf-8"
    )
    assert bundle.verify(tmp_path) == 1


def test_an_intact_bundle_verifies(tmp_path):
    payload = tmp_path / "images.tar"
    payload.write_bytes(b"pretend this is an image layer")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"digests": {"images.tar": bundle.digest_of(payload)}}), encoding="utf-8"
    )
    assert bundle.verify(tmp_path) == 0
