import xarray as xr

from atlas_vis.core.exceptions import PhysicsComputationError
from atlas_vis.physics.registry import HierarchicalPhysicsRegistry


class PhysicsDependencyResolver:
    """
    Calculates the exact evaluation order for multi-layered physics dependencies.

    Queries the Hierarchical Physics Registry to enforce the Resolution Cascade,
    ensuring base variables are computed before the derived variables that depend upon them.
    """

    def __init__(
        self, registry: HierarchicalPhysicsRegistry, current_parser_type: str
    ) -> None:
        """
        Initialize the DAG resolver for a specific dataset paradigm.

        Args:
            registry (HierarchicalPhysicsRegistry): The fully loaded physics equation registry.
            current_parser_type (str): The active parser string to filter Level 2 specific equations.

        """
        self.registry = registry
        self.parser_type = current_parser_type

    def resolve_topological_sequence(self, target_vars: list[str]) -> list[str]:
        """
        Perform a recursive DFS topological sort to establish strict execution order.

        Args:
            target_vars (list[str]): The target variable names requested for visualization.

        Returns:
            list[str]: The ordered sequence of variables that must be calculated.

        Raises:
            PhysicsComputationError: If cyclic dependencies are detected.

        """
        visited: set[str] = set()
        temp_marks: set[str] = set()
        resolution_path: list[str] = []

        def dfs(node: str) -> None:
            if node in temp_marks:
                raise PhysicsComputationError(
                    f"Fatal Graph Failure: Cyclic dependency detected at physical variable node '{node}'."
                )
            if node not in visited:
                temp_marks.add(node)
                try:
                    deps, _ = self.registry.get_best_equation(node, self.parser_type)
                    for neighbor in deps:
                        dfs(neighbor)
                except PhysicsComputationError:
                    pass

                temp_marks.remove(node)
                visited.add(node)
                resolution_path.append(node)

        for var in target_vars:
            dfs(var)
        return resolution_path

    def evaluate_lazy_variables(self, ds: xr.Dataset, targets: list[str]) -> xr.Dataset:
        """
        Sequentially apply the selected mathematical physics functions based on the topological path.

        Mutates the xarray Dataset lazily in place, preserving out-of-core memory guarantees.

        Args:
            ds (xr.Dataset): The chunked target dataset.
            targets (list[str]): Variables to resolve and compute.

        Returns:
            xr.Dataset: The dataset augmented with the new lazy dask calculation graphs.

        Raises:
            PhysicsComputationError: If lambda execution fails during graph creation.

        """
        path = self.resolve_topological_sequence(targets)
        for node in path:
            if node in ds:
                continue

            try:
                deps, func = self.registry.get_best_equation(node, self.parser_type)
                ds[node] = func(ds, *deps)
            except Exception as e:
                raise PhysicsComputationError(
                    f"Failed to evaluate physics node '{node}': {e}"
                ) from e

        return ds
