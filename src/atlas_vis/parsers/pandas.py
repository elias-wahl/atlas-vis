import re

import pandas as pd
import xarray as xr

from .base import AbstractBaseParser
from .util import pandas_to_metpy_xarray
from .aliases import aliases


class GeoSphere(AbstractBaseParser):
    @property
    def parser_name(self) -> str:
        return "geo_sphere"

    @property
    def parser_type(self) -> str:
        return "station"

    @property
    def default_fields(self) -> list[str]:
        return [
            "wind_direction",
            "wind_speed",
            "temperature",
            "pressure",
            "relative_humidity",
        ]

    @property
    def input_data_format(self) -> list[str]:
        return [".csv", ".xlsx", ".dat"]

    @property
    def key_words(self) -> list[str]:
        return ["geosphere"]

    def load_dataset(self, file_path: str, metadata_only: bool = False) -> xr.Dataset | None:
        column_names = []
        metadata = {}  # Dictionary to hold all our extracted header info

        with open(file_path, "r", encoding="latin-1") as file:
            for line in file:
                # 1. Extract Station and TAWES Number
                match_station = re.search(r"# Station:\s*(.*?),\s*Tawes-Nummer:\s*(\d+)", line)
                if match_station:
                    metadata["Station"] = match_station.group(1).strip()
                    metadata["TAWES_Nummer"] = int(match_station.group(2))

                # 2. Extract Position (Lat, Lon, Elevation)
                # Looks for: # Position: 47.456408 °N, 11.931428 °E, 509.0 m AGL
                match_pos = re.search(
                    r"# Position:\s*([\d.]+)\s*°[NS],\s*([\d.]+)\s*°[EW],\s*([\d.]+)\s*m",
                    line,
                )
                if match_pos:
                    metadata["Latitude"] = float(match_pos.group(1))
                    metadata["Longitude"] = float(match_pos.group(2))
                    metadata["Elevation_m"] = float(match_pos.group(3))

                # 3. Extract general info (Fehlwert, Quelle, Kontakt)
                match_kv = re.search(r"# (Fehlwert|Quelle|Kontakt):\s*(.*)", line)
                if match_kv:
                    key = match_kv.group(1).strip()
                    val = match_kv.group(2).strip()
                    metadata[key] = val

                # 4. Extract Column Names
                match_col = re.search(r"#\s*\d+\s+(.*)", line)
                if match_col:
                    clean_name = match_col.group(1).strip()
                    column_names.append(clean_name)

                # 5. Stop scanning once we hit the actual data
                if not line.startswith("#") and line.strip():
                    break

        # Map extracted column names using the persistent user cache and global aliases registry
        from atlas_vis.core.mapping_cache import mapping_cache
        
        file_mapping = mapping_cache.get_mapping(file_path)
        mapped_columns = []
        mapping_updated = False
        
        for col in column_names:
            if col in file_mapping:
                mapped_columns.append(file_mapping[col])
            else:
                matched = aliases.get_fuzzy_match(col)
                res = matched if matched else col
                mapped_columns.append(res)
                file_mapping[col] = res
                mapping_updated = True
                
        if mapping_updated:
            mapping_cache.set_mapping(file_path, file_mapping)

        if metadata_only:
            # If only metadata is requested, return an empty dataset configured with the parsed header info
            df = pd.DataFrame(columns=mapped_columns)
            df.attrs = metadata
            return pandas_to_metpy_xarray(df)

        # Read the data, ignoring the # comments, using Latin-1 encoding
        df = pd.read_csv(file_path, sep=r"\s+", comment="#", header=None, encoding="latin-1")

        # Apply the dynamically found column names
        if len(mapped_columns) == len(df.columns):
            df.columns = mapped_columns
        else:
            print("Warning: Detected columns do not match data width.")

        # Store the extracted metadata in the DataFrame's official attrs dictionary
        df.attrs = metadata

        # Convert to xarray Dataset using the utility function, which also applies CF-compliant coordinate handling
        xarray = pandas_to_metpy_xarray(df)

        return xarray

    def load_metadata(self, file_path: str) -> xr.Dataset | None:
        return self.load_dataset(file_path, metadata_only=True)
