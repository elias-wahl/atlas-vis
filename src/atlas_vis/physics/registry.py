import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from platformdirs import user_data_dir

from atlas_vis.core.exceptions import PhysicsComputationError
from atlas_vis.physics.formulas import GENERAL_EQUATIONS

APP_NAME = "atlas_vis"
APP_AUTHOR = "atlas_vis_team"


class HierarchicalPhysicsRegistry:
    """
    Manages the strict Resolution Cascade for physical quantities:
    Level 1: Custom User Functions (Loaded dynamically from user local folder).
    Level 2: Parser-Specific Equations (Registered internally by specific dataset parsers).
    Level 3: General Fallback Equations (Functional MetPy lambdas).
    """

    def __init__(self, override_dir: Optional[Path] = None) -> None:
        base_dir = override_dir or Path(user_data_dir(APP_NAME, APP_AUTHOR))
        self.custom_physics_dir = base_dir / "plugins" / "physics"
        self.custom_physics_dir.mkdir(parents=True, exist_ok=True)

        # Structure: dict[target_variable, dict[parser_type, dict[level, function_data]]]
        self._registry: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]] = {}

        self._seed_level_3_fallbacks()
        self._load_user_custom_physics()

    def _seed_level_3_fallbacks(self) -> None:
        """Level 3: Seeds the registry with the default MetPy lambda dictionary."""
        for target_var, data in GENERAL_EQUATIONS.items():
            self.register_equation(target_var, data["dependencies"], data["func"], level=3, parser_type="global")

    def _load_user_custom_physics(self) -> None:
        """Level 1: Scans user directory for custom lambda equations and auto-registers them."""
        for py_file in self.custom_physics_dir.rglob("*.py"):
            module_name = f"atlas_vis.plugins.physics.{py_file.stem}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                # Assumes user files define a dictionary `CUSTOM_EQUATIONS` matching the lambda format
                if hasattr(module, "CUSTOM_EQUATIONS"):
                    for target_var, data in module.CUSTOM_EQUATIONS.items():
                        self.register_equation(
                            target_var, data["dependencies"], data["func"], level=1, parser_type="global"
                        )
            except Exception as e:
                raise PhysicsComputationError(f"Failed to load custom physics script {py_file.name}: {str(e)}")

    def register_equation(
        self, target_var: str, dependencies: List[str], func: Callable, level: int, parser_type: str = "global"
    ) -> None:
        """Injects a lambda equation into the multi-tiered resolution cascade."""
        if target_var not in self._registry:
            self._registry[target_var] = {}
        if parser_type not in self._registry[target_var]:
            self._registry[target_var][parser_type] = {}

        self._registry[target_var][parser_type][level] = {"dependencies": dependencies, "func": func}

    def get_best_equation(self, target_var: str, current_parser_type: str) -> Tuple[List[str], Callable]:
        """
        Executes the Resolution Cascade to find the most specific lambda available.
        Checks Level 1 (Global Custom) -> Level 2 (Specific Parser) -> Level 3 (Global Fallback).
        """
        if target_var not in self._registry:
            raise PhysicsComputationError(f"No physics equations registered for target variable: {target_var}")

        var_data = self._registry[target_var]

        if "global" in var_data and 1 in var_data["global"]:
            return var_data["global"][1]["dependencies"], var_data["global"][1]["func"]

        if current_parser_type in var_data and 2 in var_data[current_parser_type]:
            return var_data[current_parser_type][2]["dependencies"], var_data[current_parser_type][2]["func"]

        if "global" in var_data and 3 in var_data["global"]:
            return var_data["global"][3]["dependencies"], var_data["global"][3]["func"]

        raise PhysicsComputationError(
            f"Cascade failed to find a valid lambda execution path for {target_var} under parser {current_parser_type}"
        )
