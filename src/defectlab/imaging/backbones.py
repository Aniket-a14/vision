"""Frozen vision backbones used purely as feature extractors.

torch and timm are imported lazily so the camera model stays usable without them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class BackboneSpec:
    name: str
    timm_id: str
    dim: int
    input_size: int


BACKBONES: dict[str, BackboneSpec] = {
    "dinov2_s": BackboneSpec("dinov2_s", "vit_small_patch14_reg4_dinov2.lvd142m", 384, 224),
    "resnet18": BackboneSpec("resnet18", "resnet18.a1_in1k", 512, 224),
}

DEFAULT_BACKBONE = "resnet18"


def spec_for(name: str) -> BackboneSpec:
    if name not in BACKBONES:
        raise KeyError(f"unknown backbone {name!r}; available: {sorted(BACKBONES)}")
    return BACKBONES[name]


def build(name: str = DEFAULT_BACKBONE) -> tuple[Any, BackboneSpec]:
    """Load a pretrained backbone with its classifier head removed."""
    import timm

    spec = spec_for(name)
    model = timm.create_model(spec.timm_id, pretrained=True, num_classes=0)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, spec


def build_transform(spec: BackboneSpec) -> Any:
    import timm

    config = timm.data.resolve_data_config({})
    config["input_size"] = (3, spec.input_size, spec.input_size)
    return timm.data.create_transform(**config, is_training=False)


def configure_threads() -> None:
    """Four physical cores here; oversubscribing makes extraction slower, not faster."""
    import torch

    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
