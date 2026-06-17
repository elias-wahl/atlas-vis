import logging
from typing import Dict

import numpy as np
import pandas as pd
import xarray as xr

# Initialize a logger for this module
logger = logging.getLogger(__name__)


def pandas_to_metpy_xarray(
    df: pd.DataFrame, default_station_name: str = "Unknown_Station"
) -> xr.Dataset:
    """
    Convert a pandas DataFrame containing meteorological time-series data
    into a CF-compliant xarray Dataset.

    This function automatically detects spatial metadata (latitude, longitude,
    elevation) and station identifiers from either the DataFrame columns or
    the `df.attrs` dictionary, restructuring the data into a strict
    (location, time) multi-dimensional array.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame. Must contain a temporal dimension, either as a
        column named 'time' or as the DataFrame's index.
    default_station_name : str, optional
        Fallback identifier used for the 'location' dimension if no station
        name is found in the data, by default "Unknown_Station".

    Returns
    -------
    xr.Dataset
        A CF-compliant dataset with proper coordinate attributes for MetPy integration.

    Raises
    ------
    ValueError
        If no valid time column or index can be identified in the input DataFrame.
    """
    # Work on a copy to avoid mutating the user's original data
    df_clean: pd.DataFrame = df.copy()

    # ---------------------------------------------------------
    # 1. Temporal Dimension Handling
    # ---------------------------------------------------------
    if "time" in df_clean.columns:
        if not pd.api.types.is_datetime64_any_dtype(df_clean["time"]):
            logger.info("Converting 'time' column to datetime64.")
            df_clean["time"] = pd.to_datetime(df_clean["time"])
        df_clean = df_clean.set_index("time")
    elif df_clean.index.name != "time":
        raise ValueError(
            "Input DataFrame must have a column or index explicitly named 'time'."
        )

    # ---------------------------------------------------------
    # 2. Spatial Metadata Extraction
    # ---------------------------------------------------------
    spatial_keys: list[str] = ["latitude", "longitude", "elevation", "altitude"]
    spatial_data: Dict[str, float] = {}

    for key in spatial_keys:
        if key in df_clean.columns:
            # Extract first valid value and drop to prevent temporal bloat
            val = (
                df_clean[key].dropna().iloc[0]
                if not df_clean[key].dropna().empty
                else np.nan
            )
            spatial_data[key] = float(val)
            df_clean = df_clean.drop(columns=[key])
        else:
            # Fallback to the DataFrame's global attributes
            spatial_data[key] = float(df_clean.attrs.get(key, np.nan))

    # Normalize altitude to elevation for MetPy standards
    elev_val = (
        spatial_data["elevation"]
        if not np.isnan(spatial_data["elevation"])
        else spatial_data["altitude"]
    )

    # ---------------------------------------------------------
    # 3. Station Identifier Extraction
    # ---------------------------------------------------------
    station_name: str = default_station_name
    for name_key in ["station", "station_id", "site", "TAWES_Nummer"]:
        if name_key in df_clean.columns:
            station_name = str(df_clean[name_key].dropna().iloc[0])
            df_clean = df_clean.drop(columns=[name_key])
            break
        elif name_key in df_clean.attrs:
            station_name = str(df_clean.attrs[name_key])
            break

    if station_name == default_station_name:
        logger.warning(
            f"No station identifier found. Defaulting to '{default_station_name}'."
        )

    # ---------------------------------------------------------
    # 4. Xarray Dataset Construction
    # ---------------------------------------------------------
    ds: xr.Dataset = df_clean.to_xarray()
    ds = ds.expand_dims(location=[station_name])

    # ---------------------------------------------------------
    # 5. Apply CF-Compliant Coordinates
    # ---------------------------------------------------------
    # Using specific attribute dictionaries makes the dataset instantly
    # recognizable to cartographic libraries like Cartopy and MetPy.

    if not np.isnan(spatial_data["latitude"]):
        ds = ds.assign_coords(
            latitude=xr.DataArray(
                [spatial_data["latitude"]],
                dims=["location"],
                attrs={"standard_name": "latitude", "units": "degrees_north"},
            )
        )

    if not np.isnan(spatial_data["longitude"]):
        ds = ds.assign_coords(
            longitude=xr.DataArray(
                [spatial_data["longitude"]],
                dims=["location"],
                attrs={"standard_name": "longitude", "units": "degrees_east"},
            )
        )

    if not np.isnan(elev_val):
        ds = ds.assign_coords(
            elevation=xr.DataArray(
                [elev_val],
                dims=["location"],
                attrs={"standard_name": "surface_altitude", "units": "m"},
            )
        )

    # Preserve any remaining global attributes
    ds.attrs = df.attrs

    return ds
