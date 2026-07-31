"""Typed application settings loaded from environment or .env."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Paths(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DEFECTLAB_", env_file=".env", extra="ignore")

    raw: Path = ROOT / "data" / "raw"
    processed: Path = ROOT / "data" / "processed"
    exports: Path = ROOT / "data" / "exports"
    figures: Path = ROOT / "figures"

    @property
    def casting_root(self) -> Path:
        return self.raw / "casting_data"

    def ensure(self) -> None:
        for path in (self.processed, self.exports, self.figures):
            path.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DEFECTLAB_", env_file=".env", extra="ignore")

    seed: int = 42
    noise_sd: float = 1.0
    signal_gain: float = 3.0
    pca_components: int = 30
    target_defect_rate: float = 0.03
    paths: Paths = Field(default_factory=Paths)


settings = Settings()
