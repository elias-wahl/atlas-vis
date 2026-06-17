import importlib.util
import sys
from pathlib import Path

from platformdirs import user_data_dir

from atlas_vis.core.exceptions import ParserRecognitionError
from atlas_vis.parsers.base import AbstractBaseParser

APP_NAME = "atlas_vis"
APP_AUTHOR = "atlas_vis_team"


class TrustedPluginRegistry:
    """
    Dynamically loads custom Bring Your Own Parser (BYOP) scripts written by the user.

    Reads locally from the OS-specific user data directory, making customization highly accessible.
    """

    def __init__(self, override_dir: Path | None = None) -> None:
        """
        Initialize the plugin directory structures using cross-platform data directories.

        Args:
            override_dir (Path | None): Directory override for testing purposes.

        """
        base_dir = override_dir or Path(user_data_dir(APP_NAME, APP_AUTHOR))
        self.plugin_dir = base_dir / "plugins" / "parsers"
        self.plugin_dir.mkdir(parents=True, exist_ok=True)
        self._loaded_parsers: dict[str, type[AbstractBaseParser]] = {}

    def load_plugins_from_directory(self) -> None:
        """Iterate through the designated user plugin directory and dynamically import any .py files found."""
        for py_file in self.plugin_dir.rglob("*.py"):
            module_name = f"atlas_vis.plugins.parsers.{py_file.stem}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec is None or spec.loader is None:
                    continue

                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, AbstractBaseParser)
                        and attr is not AbstractBaseParser
                    ):
                        parser_instance = attr()
                        self._loaded_parsers[parser_instance.parser_type] = attr

            except Exception as e:
                raise ParserRecognitionError(
                    f"Critical Plugin Failure: Could not load user parser module {py_file.name}. Reason: {e}"
                ) from e

    def get_parser(self, parser_type: str) -> type[AbstractBaseParser]:
        """
        Retrieve a registered parser class by its type identifier.

        Args:
            parser_type (str): The requested parser string identifier.

        Returns:
            type[AbstractBaseParser]: The instantiated parser class.

        Raises:
            ParserRecognitionError: If the requested type is not found in the loaded dictionary.

        """
        if parser_type not in self._loaded_parsers:
            raise ParserRecognitionError(
                f"Registry Miss: No loaded parser plugin matched parser type: {parser_type}"
            )
        return self._loaded_parsers[parser_type]
