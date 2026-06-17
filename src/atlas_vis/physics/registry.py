import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from platformdirs import user_data_dir

from atlas_vis.core.exceptions import PhysicsComputationError
from atlas_vis.physics.formulas import GENERAL_EQUATIONS

APP_NAME = "atlas_vis"
APP_AUTHOR = "atlas_vis_team"


class HierarchicalPhysicsRegistry:
    """
    Manages the strict Resolution Cascade for physical quantities.

    Level 1: Custom User Functions (Loaded dynamically from user local folder).
    Level 2: Parser-Specific Equations (Registered internally by specific dataset parsers).
    Level 3: General Fallback Equations (Functional MetPy lambdas).
    """

    def __init__(self, override_dir: Path | None = None) -> None:
        """
        Initialize the registry and load all functional lambda cascades.

        Args:
            override_dir (Path | None): Directory override for testing purposes.

        """
        base_dir = override_dir or Path(user_data_dir(APP_NAME, APP_AUTHOR))
        self.custom_physics_dir = base_dir / "plugins" / "physics"
        self.custom_physics_dir.mkdir(parents=True, exist_ok=True)

        self._registry: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}

        self._seed_level_3_fallbacks()
        self._load_user_custom_physics()

    def _seed_level_3_fallbacks(self) -> None:
        """Seed the registry with the default MetPy lambda dictionary."""
        for target_var, data in GENERAL_EQUATIONS.items():
            self.register_equation(
                target_var,
                data["dependencies"],
                data["func"],
                level=3,
                parser_type="global",
            )

    def _load_user_custom_physics(self) -> None:
        """Scan user directory for custom lambda equations and auto-register them at Level 1."""
        for py_file in self.custom_physics_dir.rglob("*.py"):
            module_name = f"atlas_vis.plugins.physics.{py_file.stem}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                if hasattr(module, "CUSTOM_EQUATIONS"):
                    for target_var, data in module.CUSTOM_EQUATIONS.items():
                        self.register_equation(
                            target_var,
                            data["dependencies"],
                            data["func"],
                            level=1,
                            parser_type="global",
                        )
            except Exception as e:
                raise PhysicsComputationError(
                    f"Failed to load custom physics script {py_file.name}: {e}"
                ) from e

    def register_equation(
        self,
        target_var: str,
        dependencies: list[str],
        func: Callable[..., Any],
        level: int,
        parser_type: str = "global",
    ) -> None:
        """
        Inject a lambda equation into the multi-tiered resolution cascade.

        Args:
            target_var (str): The target variable name.
            dependencies (list[str]): The input variable names required.
            func (Callable): The pure lambda function mapping Dask arrays to equations.
            level (int): The cascade level (1 = Custom, 2 = Parser, 3 = General).
            parser_type (str): The specific parser this applies to, or "global".

        """
        if target_var not in self._registry:
            self._registry[target_var] = {}
        if parser_type not in self._registry[target_var]:
            self._registry[target_var][parser_type] = {}

        self._registry[target_var][parser_type][level] = {
            "dependencies": dependencies,
            "func": func,
        }

    def get_best_equation(
        self, target_var: str, current_parser_type: str
    ) -> tuple[list[str], Callable[..., Any]]:
        """
        Execute the Resolution Cascade to find the most specific lambda available.

        Args:
            target_var (str): The physical variable requested for visualization.
            current_parser_type (str): The currently active dataset parser string.

        Returns:
            tuple[list[str], Callable[..., Any]]: The required dependencies and the resolved lambda equation.

        Raises:
            PhysicsComputationError: If no equation exists for the requested variable.

        """
        if target_var not in self._registry:
            raise PhysicsComputationError(
                f"No physics equations registered for target variable: {target_var}"
            )

        var_data = self._registry[target_var]

        if "global" in var_data and 1 in var_data["global"]:
            return var_data["global"][1]["dependencies"], var_data["global"][1]["func"]

        if current_parser_type in var_data and 2 in var_data[current_parser_type]:
            return var_data[current_parser_type][2]["dependencies"], var_data[
                current_parser_type
            ][2]["func"]

        if "global" in var_data and 3 in var_data["global"]:
            return var_data["global"][3]["dependencies"], var_data["global"][3]["func"]

        raise PhysicsComputationError(
            f"Cascade failed to find a valid lambda execution path for {target_var} under parser {current_parser_type}"
        )
