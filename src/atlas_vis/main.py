from typing import Any, Dict

import anyio
import typer
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from atlas_vis.core.config import ConfigurationManager, ProcessingSettings
from atlas_vis.core.exceptions import AtlasVisError
from atlas_vis.processing.dask_manager import HPCExecutionEngine

# Initialize the primary ASGI application router
app = FastAPI(
    title="atlas_vis - Web Processing Service",
    description="High-performance asynchronous data orchestrator for out-of-core meteorological visualization.",
    version="0.1.0",
)

# Instantiate singleton infrastructure controllers
config_manager = ConfigurationManager()
execution_engine = HPCExecutionEngine({"scheduler_port": 8786})
cli_app = typer.Typer(help="AtlasVis CLI Tool for managing HPC web render sessions.")


class SetupRequest(BaseModel):
    """Pydantic model representing incoming JSON payloads for system configuration updates."""

    visible_variables: list[str]
    temporal_strategy: str
    parser_type: str
    processing: ProcessingSettings


@app.on_event("startup")
async def startup_hpc() -> None:
    """ASGI Lifecycle Hook: Executes upon server boot to provision the auto-tuned local Dask worker pool."""
    await anyio.to_thread.run_sync(execution_engine.start_local_cluster)


@app.on_event("shutdown")
async def shutdown_hpc() -> None:
    """ASGI Lifecycle Hook: Safely tears down the Dask cluster, preventing memory zombie processes."""
    await anyio.to_thread.run_sync(execution_engine.shutdown)


@app.post("/api/settings")
async def apply_mutations(req: SetupRequest) -> Dict[str, Any]:
    """REST Endpoint: Safely validates and commits new user settings to platformdirs config without blocking loop."""
    try:
        # Save specific parser parameters into the config map
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
    """WebSocket Endpoint: Establishes a long-lived bi-directional connection."""
    await websocket.accept()
    try:
        while True:
            raw_message = await websocket.receive_text()
            await websocket.send_json({"event": "ack_command", "status": "processing"})
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        safe_error = AtlasVisError(
            message="An unhandled application exception occurred during active background pipeline execution.",
            context={"system_exception": str(exc)},
        )
        await websocket.send_json({"event": "toast_notification", "error": safe_error.to_dict()})


@cli_app.command()
def start(host: str = "0.0.0.0", port: int = 8000) -> None:
    """CLI Entrypoint: Starts the AtlasVis Uvicorn daemon and persistent config manager."""
    typer.echo(f"Starting AtlasVis server on {host}:{port} utilizing platformdirs persistent config.")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    cli_app()
