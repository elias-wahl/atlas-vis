from abc import ABC, abstractmethod

import xarray as xr


class AbstractBaseParser(ABC):
    """
    Core abstraction contract for integrating generic numerical datasets.
    Any custom BYOP (Bring Your Own Parser) script must inherit from this class
    to successfully register with the framework's processing pipeline.
    """

    @property
    @abstractmethod
    def parser_type(self) -> str:
        """
        Returns the internal string classification identifier.
        Must be strictly defined (e.g., 'WRF', 'UAS', 'Lidar', 'EULAG').
        This string maps directly to the sparse ProcessingSettings in the ConfigurationManager.

        Returns:
            str: The explicit parser identifier.
        """
        pass

    @abstractmethod
    def validate_dataset(self, ds: xr.Dataset) -> bool:
        """
        Interrogates the consolidated dataset's metadata, dimensions, and variables.
        Returns true if the dataset conforms to this specific parser's structural requirements.

        Args:
            ds (xr.Dataset): The lazy aggregated dataset.

        Returns:
            bool: True if the parser claims ownership of the dataset structure.
        """
        pass

    @abstractmethod
    def process_metadata(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Standardizes temporal, spatial, and vertical coordinate systems.
        Must return an xarray Dataset that conforms to the AtlasVis internal canonical variable names.
        Ensure all internal variables are explicitly quantified with Pint units during this phase.

        Args:
            ds (xr.Dataset): The raw, unstandardized dataset.

        Returns:
            xr.Dataset: The CF-compliant, dimensionally standardized dataset.
        """
        pass
