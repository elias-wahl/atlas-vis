import shutil
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Optional

import anyio
import yaml
from platformdirs import user_config_dir
from pydantic import BaseModel, Field

from atlas_vis.core.exceptions import ConfigurationError

APP_NAME = "atlas_vis"
APP_AUTHOR = "atlas_vis_team"


class ProcessingSettings(BaseModel):
    """Configuration for data reduction, striding, and domain cropping during ingestion. Configurable per parser type."""

    spatial_stride: int
    temporal_stride: int
    max_domain_x: int
    max_domain_y: int
    max_domain_z: int


class UserSettings(BaseModel):
    """Master configuration definitions combining visual settings and parser-specific processing limits."""

    visible_variables: List[str]
    temporal_strategy: str
    spatial_bbox: List[float]
    parser_configs: Dict[str, ProcessingSettings]


class ConfigurationManager:
    """
    Manages transactional file I/O safely away from the ASGI event loop.

    Utilizes platformdirs to guarantee cross-platform configuration persistence between CLI runs.
    """

    def __init__(self, override_dir: Optional[Path] = None) -> None:
        base_dir = override_dir or Path(user_config_dir(APP_NAME, APP_AUTHOR))
        self.config_path = base_dir / "config.yaml"
        self._lock = anyio.Lock()

        # Ensure default configuration exists in the user directory
        self._initialize_user_config()
        self._state = self._load_from_disk_sync()

    def _initialize_user_config(self) -> None:
        """Copies the default_settings.yaml bundled with the package to the user's config directory if missing."""
        if not self.config_path.exists():
            try:
                self.config_path.parent.mkdir(parents=True, exist_ok=True)
                # Using importlib.resources to dynamically locate the bundled yaml file
                with resources.path("atlas_vis.config", "default_settings.yaml") as default_path:
                    shutil.copy(default_path, self.config_path)
            except Exception as e:
                raise ConfigurationError(f"Failed to initialize user configuration directory: {str(e)}")

    def _load_from_disk_sync(self) -> UserSettings:
        """Synchronously reads the YAML file into the Pydantic state model."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                return UserSettings(**data)
        except Exception as e:
            raise ConfigurationError(f"Failed to read cross-platform config.yaml payload: {str(e)}")

    def get_parser_settings(self, parser_type: str) -> ProcessingSettings:
        """
        Retrieves sparse parameter settings specific to the requested parser type.

        Args:
            parser_type (str): The identifier string of the parser (e.g., 'WRF', 'UAS').

        Returns:
            ProcessingSettings: The configured spatial and domain limits. Falls back to 'default' if not found.
        """
        return self._state.parser_configs.get(parser_type, self._state.parser_configs.get("default"))

    async def get_snapshot(self) -> Dict[str, Any]:
        """Provides a frozen dictionary snapshot for safe Dask computation graph compilation."""
        async with self._lock:
            return self._state.model_dump()

    async def update_settings(self, new_settings: Dict[str, Any]) -> None:
        """Asynchronously validates nested model mutations and delegates writes to a background thread."""
        async with self._lock:
            try:
                updated = self._state.model_copy(update=new_settings)
                self._state = updated
            except Exception as e:
                raise ConfigurationError(f"Pydantic validation failed for settings change: {str(e)}")
            await anyio.to_thread.run_sync(self._write_to_disk_sync, self._state.model_dump())

    def _write_to_disk_sync(self, data: Dict[str, Any]) -> None:
        """Synchronous file writing operation executed inside an AnyIO thread worker."""
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            raise ConfigurationError(f"Asynchronous disk write pipeline failed: {str(e)}")
