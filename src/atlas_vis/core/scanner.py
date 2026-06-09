from pathlib import Path
from typing import Generator, List, Optional

import xarray as xr

from atlas_vis.core.exceptions import ParserRecognitionError


class StandardDirectoryScanner:
    """
    Recursively scans directory trees and treats multi-file fragmented model outputs
    as a single, massive, consolidated multidimensional array.
    """

    def __init__(self, target_dir: Path) -> None:
        """
        Initializes the scanner and validates target directory existence.

        Args:
            target_dir (Path): The absolute or relative path to the dataset directory.
        Raises:
            ParserRecognitionError: If the directory does not exist on the filesystem.
        """
        self.target_dir = target_dir
        if not self.target_dir.exists():
            raise ParserRecognitionError(f"Fatal Initialization: Target directory does not exist: {self.target_dir}")

    def scan_files(self, extension: str = "*.nc") -> Generator[Path, None, None]:
        """
        Yields file paths matching the requested extension using standard recursive globbing.

        Args:
            extension (str): The file extension to match. Defaults to "*.nc".

        Yields:
            Path: Resolved file paths containing meteorological data.
        """
        for file_path in self.target_dir.rglob(extension):
            if file_path.is_file():
                yield file_path

    def open_unified_model_output(self, extension: str = "*.nc", concat_dim: str = "Time") -> xr.Dataset:
        """
        Opens discovered model outputs and consolidates them into a single, seamless xarray dataset.
        Applies chunking automatically to prevent loading massive multidimensional arrays into main memory.

        Args:
            extension (str): File extension to filter inside the directory.
            concat_dim (str): The dimension along which to concatenate the multifile dataset.

        Returns:
            xr.Dataset: The lazy, chunked out-of-core Dask dataset encompassing all files.

        Raises:
            ParserRecognitionError: If aggregation fails or no files are found.
        """
        files = list(self.scan_files(extension))
        if not files:
            raise ParserRecognitionError(f"Data Exhaustion: No files matching '{extension}' found in {self.target_dir}")
        try:
            # Parallelized multifile ingestion treating the entire directory sequence as one model instance
            return xr.open_mfdataset(
                files, combine="nested", concat_dim=concat_dim, parallel=True, chunks="auto", engine="netcdf4"
            )
        except Exception as e:
            raise ParserRecognitionError(f"Failed to consolidate multi-file model output into single dataset: {str(e)}")
