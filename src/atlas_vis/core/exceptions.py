import traceback
from typing import Any


class AtlasVisError(Exception):
    """
    Base exception class for all atlas_vis system failures.

    Allows for structured serialization of errors to push safely to the Trame frontend UI
    via ASGI WebSocket channels without breaking the continuous event loop.
    """

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        """Initialize the base error with a message and optional execution context."""
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the exception for WebSocket transmission.

        Returns:
            dict[str, Any]: A dictionary containing the error class name, the human-readable message,
                the execution context dictionary, and the full stack traceback.

        """
        return {
            "error_class": self.__class__.__name__,
            "message": self.message,
            "context": self.context,
            "traceback": traceback.format_exc(),
        }


class ConfigurationError(AtlasVisError):
    """Raised when synchronization, validation, or disk writing of system configuration fails."""


class ParserRecognitionError(AtlasVisError):
    """Raised when multifile parsing, metadata validation, or dynamic BYOP plugin loading fails."""


class DomainBoundsError(AtlasVisError):
    """Raised when requested coordinate ranges fall entirely outside native spatial boundaries."""


class PhysicsComputationError(AtlasVisError):
    """Raised when the hierarchical physics graph fails to resolve or calculations fail."""


class RenderingPipelineError(AtlasVisError):
    """Raised when PyVista StructuredGrid geometry generation fails due to buffer misalignment."""
