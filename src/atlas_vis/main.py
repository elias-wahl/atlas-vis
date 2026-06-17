import multiprocessing
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import typer
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from atlas_vis.core.config import ConfigurationManager, ProcessingSettings
from atlas_vis.core.exceptions import AtlasVisError
from atlas_vis.processing.dask_manager import HPCExecutionEngine

# Instantiate singleton infrastructure controllers
config_manager = ConfigurationManager()
execution_engine = HPCExecutionEngine({"scheduler_port": 8786})
cli_app = typer.Typer(help="AtlasVis CLI Tool for managing HPC web render sessions.")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage the ASGI lifecycle events for background Dask provisioning.

    Executes cluster creation upon boot and tears it down gracefully upon shutdown
    to prevent zombie memory processes on the HPC node.

    Args:
        app (FastAPI): The executing web application application.

    """
    await run_in_threadpool(execution_engine.start_local_cluster)
    yield
    await run_in_threadpool(execution_engine.shutdown)


app = FastAPI(
    title="atlas_vis - Web Processing Service",
    description="High-performance asynchronous data orchestrator for out-of-core meteorological visualization.",
    version="0.1.0",
    lifespan=lifespan,
)


class SetupRequest(BaseModel):
    """Pydantic model representing incoming JSON payloads for system configuration updates."""

    visible_variables: list[str]
    temporal_strategy: str
    parser_type: str
    processing: ProcessingSettings


def run_api():
    """Run the API server - moved to module level for Windows multiprocessing compatibility."""
    uvicorn.run(app, host="0.0.0.0", port=8000)


def run_ui():
    """Run the UI server - moved to module level for Windows multiprocessing compatibility."""
    # We import the UI server locally to avoid initializing VTK in the main thread prematurely
    from atlas_vis.visualization.trame_server import TrameStreamingServer

    ui_server = TrameStreamingServer(port=8080)
    ui_server.start()


@app.post("/api/settings")
async def apply_mutations(req: SetupRequest) -> dict[str, Any]:
    """
    Safely validate and commit new user settings to platformdirs config without blocking loop.

    Args:
        req (SetupRequest): The requested mutation payload.

    Returns:
        dict[str, Any]: A success/failure status JSON payload.

    """
    try:
        current = await config_manager.get_snapshot()
        parser_configs = current.get("parser_configs", {})
        parser_configs[req.parser_type] = req.processing.model_dump()

        await config_manager.update_settings(
            {
                "visible_variables": req.visible_variables,
                "temporal_strategy": req.temporal_strategy,
                "parser_configs": parser_configs,
            }
        )
        return {"status": "success", "snapshot": await config_manager.get_snapshot()}
    except AtlasVisError as ave:
        return {"status": "error", "error": ave.to_dict()}
    except Exception as e:
        return {"status": "fatal", "message": str(e)}


@app.websocket("/ws/stream")
async def websocket_orchestrator(websocket: WebSocket) -> None:
    """
    Establish a long-lived bi-directional WebSocket connection for Trame and pipeline status.

    Args:
        websocket (WebSocket): The active client connection.

    """
    await websocket.accept()
    try:
        while True:
            await websocket.receive_text()
            await websocket.send_json({"event": "ack_command", "status": "processing"})
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        safe_error = AtlasVisError(
            message="An unhandled application exception occurred during active background pipeline execution.",
            context={"system_exception": str(exc)},
        )
        await websocket.send_json({"event": "toast_notification", "error": safe_error.to_dict()})


@app.get("/api/browse")
async def browse_directory(target_path: str | None = None) -> dict[str, Any]:
    """Safely enumerate subdirectories and files."""
    try:
        # FIX: Start at the User's Home Directory (~/) instead of script execution path
        current = Path(target_path) if target_path else Path.home()
        current = current.resolve()

        directories = []
        files = []

        if current.exists() and current.is_dir():
            for item in current.iterdir():
                # Skip hidden metadata or configuration paths
                if item.name.startswith("."):
                    continue
                if item.is_dir():
                    directories.append(item.name)
                elif item.is_file():
                    # FIX: Removed the suffix filter so ALL files appear in the UI
                    files.append(item.name)

        return {
            "status": "success",
            "current_path": str(current),
            "parent_path": str(current.parent) if current.parent != current else str(current),
            "directories": sorted(directories),
            "files": sorted(files),
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Filesystem isolation access failure: {str(e)}",
        }


@cli_app.command()
def start(host: str = "0.0.0.0", api_port: int = 8000, ui_port: int = 8080) -> None:
    """Start both the AtlasVis API backend and the Trame rendering frontend."""
    typer.echo(f"Starting AtlasVis API on port {api_port} and UI on port {ui_port}...")

    # Launch both services as separate processes
    api_process = multiprocessing.Process(target=run_api)
    ui_process = multiprocessing.Process(target=run_ui)

    api_process.start()
    ui_process.start()

    try:
        api_process.join()
        ui_process.join()
    except KeyboardInterrupt:
        typer.echo("\nShutting down AtlasVis services...")
        api_process.terminate()
        ui_process.terminate()


if __name__ == "__main__":
    cli_app()
