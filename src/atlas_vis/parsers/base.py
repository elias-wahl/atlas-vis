from abc import ABC, abstractmethod

import xarray as xr
from dask.rewrite import strategies


class AbstractBaseParser(ABC):
    """
    Core abstraction contract for integrating generic numerical datasets.

    Any custom BYOP (Bring Your Own Parser) script must inherit from this class
    to successfully register with the framework's processing pipeline.
    """

    @property
    @abstractmethod
    def parser_name(self) -> str:
        """
        Return the human-readable name of the parser.

        Returns:
            str: The display name of the parser (e.g., 'WRF Model Output').

        """
        pass

    @property
    @abstractmethod
    def default_fields(self) -> list[str]:
        """
        Return a list of default variable names that will be shown in the Trame UI if no user selection is made.

        Returns:
            List[str]: A list of default variable names (e.g., ['temperature', 'pressure']).
        """
        pass

    @property
    @abstractmethod
    def parser_type(self) -> str:
        """
        Return the internal string classification identifier.

        Must be strictly defined (e.g., 'wrf', 'uas', 'lidar', 'eulag').
        This string maps directly to the sparse ProcessingSettings in the ConfigurationManager.

        Returns:
            str: The explicit parser identifier.

        """
        pass

    @property
    @abstractmethod
    def key_words(self) -> list[str]:
        """
        Return a list of keywords that can be used to determine if a file is likely compatible with this parser.
        This is used to determine which parser(s) to apply for a given dataset.
        """
        pass

    @property
    @abstractmethod
    def input_data_format(self) -> list[str]:
        """
        Return a list of file extensions that this might be able to parse.
        This is used for preliminary file filtering before deeper validation.

        Returns:
            List[str]: A list of lowercase file extensions (e.g., ['.nc', '.hdf5']).

        """
        pass

    @abstractmethod
    def load_metadata(self, file_path: str) -> xr.Dataset | None:
        """
        Perform a lightweight read of the dataset to extract metadata and validate compatibility.
        Args:
            file_path (str): The path to the dataset file.
        Returns:
            xr.Dataset | None: A dataset containing only metadata if compatible, or None if incompatible.
        """
        pass

    @abstractmethod
    def load_dataset(self, file_path: str) -> xr.Dataset | None:
        """
        Perform a full read of the dataset, applying any necessary transformations to conform to the internal data model.

        Args:
            file_path (str): The path to the dataset file.

        Returns:
            xr.Dataset | None: The fully loaded and standardized dataset, or None if loading fails.
        """
        pass
