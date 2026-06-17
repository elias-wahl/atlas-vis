import logging
from typing import Any

import dask
import psutil
from dask.distributed import Client, LocalCluster

from atlas_vis.core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class HPCExecutionEngine:
    """
    Manages the lifecycle, scaling, and memory thresholds of a hardened Dask cluster.

    Acts as the primary execution backbone for all Xarray out-of-core computations.
    """

    def __init__(self, cluster_config: dict[str, Any]) -> None:
        """
        Initialize the execution engine wrapper.

        Args:
            cluster_config (dict[str, Any]): General cluster connection settings.

        """
        self.config = cluster_config
        self.cluster: LocalCluster | None = None
        self.client: Client | None = None

    def _auto_tune_resources(self) -> dict[str, Any]:
        """
        Dynamically calculate optimal worker counts and memory limits based on host OS resources.

        Guarantees functionality on standard operating systems while exploiting Linux cluster potential.

        Returns:
            dict[str, Any]: Calculated safe thresholds for core execution arrays.

        """
        total_cores = psutil.cpu_count(logical=True)

        if total_cores is None:
            raise ConfigurationError("Failed to auto-detect CPU cores for Dask tuning.")

        worker_count = max(1, total_cores - 1)

        total_ram_gb = psutil.virtual_memory().total / (1024**3)
        usable_ram_gb = total_ram_gb * 0.80
        memory_per_worker = max(2.0, usable_ram_gb / worker_count)

        logger.info(
            f"Auto-Tuning Dask: {worker_count} workers, {memory_per_worker:.2f}GB per worker."
        )
        return {
            "n_workers": worker_count,
            "memory_limit": f"{memory_per_worker:.2f}GB",
        }

    def start_local_cluster(
        self, n_workers: int | None = None, memory_limit: str | None = None
    ) -> None:
        """
        Spawn an optimized local cluster.

        Enforces a strict 1-thread-per-worker rule to avoid Python Global Interpreter Lock (GIL) contention.

        Args:
            n_workers (int | None): Explicit worker override.
            memory_limit (str | None): Explicit memory limit per worker override.

        """
        resources = self._auto_tune_resources()
        final_workers = n_workers or resources["n_workers"]
        final_memory = memory_limit or resources["memory_limit"]

        dask.config.set(
            {
                "array.rechunk.method": "p2p",
                "optimization.fuse.active": True,
                "distributed.worker.memory.target": 0.60,
                "distributed.worker.memory.spill": 0.70,
                "distributed.worker.memory.pause": 0.80,
                "distributed.worker.memory.terminate": 0.95,
            }
        )

        self.cluster = LocalCluster(
            n_workers=final_workers,
            threads_per_worker=1,
            memory_limit=final_memory,
            dashboard_address=":8787",
        )
        self.client = Client(self.cluster)

    def shutdown(self) -> None:
        """Gracefully terminate worker nodes and close the active Dask scheduler."""
        if self.client:
            self.client.close()
        if self.cluster:
            self.cluster.close()
