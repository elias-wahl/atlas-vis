import yaml

from pathlib import Path
from platformdirs import user_config_dir
from anyio import Lock
from anyio.abc import ObjectSendStream
from anyio.to_thread import run_sync
from .aliases import aliases

APP_NAME = "atlas_vis"
APP_AUTHOR = "atlas_vis_team"


class AsyncNamespaceMapper:
    """Standardizes non-uniform meteorological variable names interactively without blocking the loop."""

    def __init__(
        self,
        ui_notify_queue: ObjectSendStream[dict[str, str]],
        override_dir: Path | None = None,
    ) -> None:
        """
        Initialize the mapping system with a UI WebSocket queue for human-in-the-loop resolution.

        Args:
            ui_notify_queue (anyio.abc.ObjectSendStream): Queue to send unmapped variables to the frontend.
            override_dir (Path | None): Directory override for testing purposes.

        """
        base_dir = override_dir or Path(user_config_dir(APP_NAME, APP_AUTHOR))
        self.mapping_file = base_dir / "namespace_mappings.yaml"
        self.ui_queue = ui_notify_queue
        self.lock = Lock()
        self.mappings = self._load_yaml_sync()

    def _load_yaml_sync(self) -> dict[str, str]:
        """Read existing variable mappings from the local disk synchronously during initialization."""
        if self.mapping_file.exists():
            with open(self.mapping_file, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    async def resolve_variable(self, var_name: str) -> tuple[str, bool]:
        """
        Asynchronously resolve variable aliases. Prompts UI on queue if variable is unmapped.

        Args:
            var_name (str): The raw variable name from the dataset.

        Returns:
            tuple[str, bool]: A tuple of the mapped name and a boolean indicating if it was already known.

        """
        async with self.lock:
            if var_name in self.mappings:
                return self.mappings[var_name], True

            # Attempt to resolve using the Aliases system before prompting the user.
            alias_match = aliases.get_match(var_name)
            if alias_match:
                return alias_match, True

            fallback_name = f"custom_field_{var_name.lower()}"
            self.mappings[var_name] = fallback_name

            await self.ui_queue.send(
                {
                    "event": "unmapped_variable",
                    "raw_name": var_name,
                    "fallback": fallback_name,
                }
            )
            await run_sync(self._save_yaml_sync)
            return fallback_name, False

    def _save_yaml_sync(self) -> None:
        """Write the updated mapping dictionary back to the configuration directory safely."""
        self.mapping_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.mapping_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.mappings, f, default_flow_style=False)
