import json
from pathlib import Path
from typing import Any

from platformdirs import user_config_dir

APP_NAME = "atlas_vis"
APP_AUTHOR = "atlas_vis_team"


class MappingCache:
    """
    Manages a persistent JSON cache of variable mappings per filepath.
    Allows the application to remember fuzzy matching results and allows users to manually override mappings.
    """

    def __init__(self, override_dir: Path | None = None) -> None:
        base_dir = override_dir or Path(user_config_dir(APP_NAME, APP_AUTHOR))
        self.cache_path = base_dir / "variable_mappings.json"
        
        if not base_dir.exists():
            base_dir.mkdir(parents=True, exist_ok=True)
            
        self._cache = self._load()

    def _load(self) -> dict[str, dict[str, str]]:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save(self) -> None:
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=4)
        except Exception as e:
            print(f"Warning: Failed to save mapping cache: {e}")

    def get_mapping(self, filepath: str) -> dict[str, str]:
        """Return the known column mappings for a specific file."""
        return self._cache.get(filepath, {})

    def set_mapping(self, filepath: str, mapping: dict[str, str]) -> None:
        """Update and persist the column mappings for a specific file."""
        if self._cache.get(filepath) != mapping:
            self._cache[filepath] = mapping
            self._save()

# Global singleton instance for easy import
mapping_cache = MappingCache()
