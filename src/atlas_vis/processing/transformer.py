import dask.array as da
import numpy as np
import xarray as xr
from pyproj import Transformer

from atlas_vis.core.config import ProcessingSettings
from atlas_vis.core.exceptions import DomainBoundsError


class DomainSizeCropper:
    """Protects cluster node memory by strictly enforcing maximum dimension bounds."""

    def __init__(self, settings: ProcessingSettings) -> None:
        """Initialize the cropper with max bounds."""
        self.limits = {
            "west_east": settings.max_domain_x,
            "south_north": settings.max_domain_y,
            "bottom_top": settings.max_domain_z,
        }

    def apply_absolute_limits(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Slice dimensions strictly to the maximum bounds defined by configuration.

        Args:
            ds (xr.Dataset): The native dataset.

        Returns:
            xr.Dataset: Centrally cropped dataset respecting absolute hardware limits.

        """
        subset_dict = {}
        for dim, max_limit in self.limits.items():
            if dim in ds.dims and ds.sizes[dim] > max_limit:
                current_size = ds.sizes[dim]
                center_idx = current_size // 2
                half_window = max_limit // 2
                start_idx = max(0, center_idx - half_window)
                end_idx = min(current_size, center_idx + half_window)
                subset_dict[dim] = slice(start_idx, end_idx)

        if subset_dict:
            return ds.isel(**subset_dict)
        return ds


class DatasetThinner:
    """Applies explicit, parser-specific data thinning utilizing spatial and temporal strides."""

    def __init__(self, settings: ProcessingSettings) -> None:
        """Initialize the thinner with explicit user strides."""
        self.spatial_stride = settings.spatial_stride
        self.temporal_stride = settings.temporal_stride

    def apply_strides(
        self, ds: xr.Dataset, x_dim: str = "west_east", y_dim: str = "south_north", t_dim: str = "Time"
    ) -> xr.Dataset:
        """
        Downsample the array logically across the specified stride parameters.

        Args:
            ds (xr.Dataset): The cropped dataset.
            x_dim (str): East-West dimension name.
            y_dim (str): South-North dimension name.
            t_dim (str): Time dimension name.

        Returns:
            xr.Dataset: The explicitly thinned dataset to speed up rendering.

        """
        subset_dict = {}
        if x_dim in ds.dims:
            subset_dict[x_dim] = slice(None, None, self.spatial_stride)
        if y_dim in ds.dims:
            subset_dict[y_dim] = slice(None, None, self.spatial_stride)
        if t_dim in ds.dims:
            subset_dict[t_dim] = slice(None, None, self.temporal_stride)

        return ds.isel(**subset_dict)


class PyProjSpatialTransformer:
    """Translates arbitrary subset curvilinear coordinates into a standardized Cartesian projection space."""

    def __init__(self, bbox: list[float], crs_from: str = "epsg:4326", crs_to: str = "epsg:3857") -> None:
        """
        Initialize the spatial transformer to map PyProj across distributed blocks.

        Args:
            bbox (list[float]): Bounding box format [lat_min, lon_min, lat_max, lon_max].
            crs_from (str): Native coordinate reference system.
            crs_to (str): Target cartesian coordinate reference system.

        """
        self.lat_min, self.lon_min, self.lat_max, self.lon_max = bbox
        self.crs_from = crs_from
        self.crs_to = crs_to
        self.transformer = Transformer.from_crs(crs_from, crs_to, always_xy=True)

    def crop_native_grid_coordinates(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Apply coordinate boolean cropping inside native curvilinear grid space.

        Args:
            ds (xr.Dataset): The thinned dataset.

        Returns:
            xr.Dataset: The geographically masked dataset.

        Raises:
            DomainBoundsError: If bounding box completely misses the dataset area.

        """
        lat = ds["XLAT"] if "XLAT" in ds else ds["lat"]
        lon = ds["XLONG"] if "XLONG" in ds else ds["lon"]

        mask = (lat >= self.lat_min) & (lat <= self.lat_max) & (lon >= self.lon_min) & (lon <= self.lon_max)
        y_indices, x_indices = np.where(mask.values)

        if len(y_indices) == 0 or len(x_indices) == 0:
            raise DomainBoundsError("Execution Halt: Requested bounding box is entirely outside dataset domain bounds.")

        y_slice = slice(int(y_indices.min()), int(y_indices.max()) + 1)
        x_slice = slice(int(x_indices.min()), int(x_indices.max()) + 1)
        return ds.isel(south_north=y_slice, west_east=x_slice)

    def _dask_pyproj_wrapper(self, lon_chunk: np.ndarray, lat_chunk: np.ndarray) -> np.ndarray:
        """Apply PyProj transformations to localized numpy chunks."""
        x, y = self.transformer.transform(lon_chunk, lat_chunk)
        return np.stack([x, y], axis=0)

    def apply_lazy_cartesian_projection(self, ds: xr.Dataset, vertical_exaggeration: float = 10.0) -> xr.Dataset:
        """
        Convert coordinates lazily into a Cartesian projection space to feed the PyVista render engine.

        Uses dask.array.map_blocks to parallelize the PyProj transformation without blowing up system RAM.

        Args:
            ds (xr.Dataset): Geographically cropped dataset.
            vertical_exaggeration (float): Z-axis scale multiplier.

        Returns:
            xr.Dataset: The final dataset assigned with contiguous x_render, y_render, and z_render arrays.

        """
        lat_da = ds["XLAT"] if "XLAT" in ds else ds["lat"]
        lon_da = ds["XLONG"] if "XLONG" in ds else ds["lon"]

        lon_da, lat_da = xr.align(lon_da, lat_da, join="exact")

        xy_lazy = da.map_blocks(
            self._dask_pyproj_wrapper,
            lon_da.data,
            lat_da.data,
            dtype=float,
            drop_axis=None,
            new_axis=0,
            chunks=((2,) + lon_da.data.chunks),
        )

        x_lazy = xy_lazy[0]
        y_lazy = xy_lazy[1]
        z_lazy = ds["Z"].data * vertical_exaggeration if "Z" in ds else da.zeros_like(x_lazy)

        return ds.assign_coords(
            x_render=(["south_north", "west_east"], x_lazy),
            y_render=(["south_north", "west_east"], y_lazy),
            z_render=(["bottom_top", "south_north", "west_east"], z_lazy),
        )
