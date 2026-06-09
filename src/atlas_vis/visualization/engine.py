import numpy as np
import pyvista as pv
import xarray as xr

from atlas_vis.core.exceptions import RenderingPipelineError


class ZeroCopyVisualEngine:
    """
    Generates contiguous 3D geometries optimized for zero-copy memory pipelines.
    By enforcing the use of StructuredGrid, we bypass the extreme memory duplication
    overheads typically associated with VTK UnstructuredGrid generation.
    """

    @staticmethod
    def construct_structured_mesh(ds: xr.Dataset) -> pv.StructuredGrid:
        """
        Generates a structured mesh directly from evaluated array buffers.
        Maps the underlying NumPy memory directly to the VTK data arrays without a deep copy.

        Args:
            ds (xr.Dataset): The fully evaluated multidimensional dataset.

        Returns:
            pv.StructuredGrid: The zero-copy compatible VTK geometry mesh.
        """
        try:
            # Force evaluation of derived spatial coordinates into main memory
            x = ds["x_render"].values
            y = ds["y_render"].values
            z = ds["z_render"].values

            # Flatten and stack the 3D projection points contiguously into an Nx3 matrix
            pts = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

            grid = pv.StructuredGrid()

            # Direct pointer assignment of the underlying NumPy buffer to the VTK points matrix
            grid.points = pts
            grid.dimensions = (ds.sizes["west_east"], ds.sizes["south_north"], ds.sizes["bottom_top"])

            # Attach active meteorological physics variables to the points matrix
            for var in ds.data_vars:
                if "bottom_top" in ds[var].dims:
                    grid.point_data[var] = ds[var].values.ravel()

            return grid
        except Exception as e:
            raise RenderingPipelineError(f"Critical VTK Memory Error: Failed to assemble zero-copy mesh. {str(e)}")
