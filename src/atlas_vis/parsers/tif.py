import xarray as xr
from .base import AbstractBaseParser

class TifParser(AbstractBaseParser):
    @property
    def parser_name(self) -> str:
        return "geo_tiff"

    @property
    def parser_type(self) -> str:
        return "raster"

    @property
    def default_fields(self) -> list[str]:
        return ["height"]

    @property
    def input_data_format(self) -> list[str]:
        return [".tif", ".tiff"]

    @property
    def key_words(self) -> list[str]:
        return []

    def load_metadata(self, file_path: str) -> xr.Dataset | None:
        return self._load(file_path, metadata_only=True)

    def load_dataset(self, file_path: str) -> xr.Dataset | None:
        return self._load(file_path, metadata_only=False)

    def _load(self, file_path: str, metadata_only: bool) -> xr.Dataset | None:
        try:
            import rioxarray
            # open_rasterio opens lazily, perfect for both metadata and full dataset loading
            ds = rioxarray.open_rasterio(file_path)
            
            if isinstance(ds, xr.DataArray):
                ds = ds.to_dataset(name="height")
            else:
                vars_list = list(ds.data_vars)
                if len(vars_list) == 1:
                    ds = ds.rename({vars_list[0]: "height"})
            
            return ds
        except Exception:
            return None
