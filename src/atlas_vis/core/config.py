import shutil
from importlib import resources
from pathlib import Path
from typing import Any

import anyio
import yaml
from platformdirs import user_config_dir
from pydantic import BaseModel

from atlas_vis.core.exceptions import ConfigurationError

APP_NAME = "atlas_vis"
APP_AUTHOR = "atlas_vis_team"


class ProcessingSettings(BaseModel):
    """
    Configuration for data reduction, striding, and domain cropping during ingestion.

    Attributes:
        spatial_stride: Load every n-th spatial grid point.
        temporal_stride: Load every n-th time step.
        max_domain_x: Absolute maximum width in grid points.
        max_domain_y: Absolute maximum height in grid points.
        max_domain_z: Absolute maximum vertical levels.

    """

    spatial_stride: int
    temporal_stride: int
    max_domain_x: int
    max_domain_y: int
    max_domain_z: int


class UserSettings(BaseModel):
    """Master configuration definitions combining visual settings and parser-specific processing limits."""

    visible_variables: list[str]
    temporal_strategy: str
    spatial_bbox: list[float]
    parser_configs: dict[str, ProcessingSettings]


class ConfigurationManager:
    """
    Manages transactional file I/O safely away from the ASGI event loop.

    Utilizes platformdirs to guarantee cross-platform configuration persistence between CLI runs.
    """

    def __init__(self, override_dir: Path | None = None) -> None:
        """Initialize the configuration manager and seed defaults if necessary."""
        base_dir = override_dir or Path(user_config_dir(APP_NAME, APP_AUTHOR))
        self.config_path = base_dir / "config.yaml"
        self._lock = anyio.Lock()

        self._initialize_user_config()
        self._state = self._load_from_disk_sync()

    def _initialize_user_config(self) -> None:
        """Copy the default_settings.yaml bundled with the package to the user's config directory if missing."""
        if not self.config_path.exists():
            try:
                self.config_path.parent.mkdir(parents=True, exist_ok=True)
                with resources.path("atlas_vis.config", "default_settings.yaml") as default_path:
                    shutil.copy(default_path, self.config_path)
            except Exception as e:
                raise ConfigurationError(f"Failed to initialize user configuration directory: {e}") from e

    def _load_from_disk_sync(self) -> UserSettings:
        """Read the YAML file synchronously into the Pydantic state model."""
        try:
            with open(self.config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                return UserSettings(**data)
        except Exception as e:
            raise ConfigurationError(f"Failed to read cross-platform config.yaml payload: {e}") from e

    def get_parser_settings(self, parser_type: str) -> ProcessingSettings:
        """
        Retrieve sparse parameter settings specific to the requested parser type.

        Args:
            parser_type (str): The identifier string of the parser (e.g., 'WRF', 'UAS').

        Returns:
            ProcessingSettings: The configured spatial and domain limits.

        """
        if parser_type in self._state.parser_configs:
            return self._state.parser_configs[parser_type]
        return self._state.parser_configs["default"]

    async def get_snapshot(self) -> dict[str, Any]:
        """Provide a frozen dictionary snapshot for safe Dask computation graph compilation."""
        async with self._lock:
            return self._state.model_dump()

    async def update_settings(self, new_settings: dict[str, Any]) -> None:
        """Asynchronously validate nested model mutations and delegate writes to a background thread."""
        async with self._lock:
            try:
                updated = self._state.model_copy(update=new_settings)
                self._state = updated
            except Exception as e:
                raise ConfigurationError(f"Pydantic validation failed for settings change: {e}") from e
            await anyio.to_thread.run_sync(self._write_to_disk_sync, self._state.model_dump())

    def _write_to_disk_sync(self, data: dict[str, Any]) -> None:
        """Write file synchronously inside an AnyIO thread worker."""
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            raise ConfigurationError(f"Asynchronous disk write pipeline failed: {e}") from e
