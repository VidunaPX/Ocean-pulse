# src/load_gocd_dataset.py
import xarray as xr

DATA_FILE = "data/gocd_a0002341_M0107.nc"

def load_dataset(file=DATA_FILE):
    """
    Load GCOD Copernicus ocean currents dataset and select surface layer.
    """
    ds = xr.open_dataset(file)
    print(ds)

    # Select surface layer if depth exists
    if 'z' in ds.dims:
        ds = ds.isel(z=0)

    # Variables are already named 'u' and 'v'
    # No need to rename

    return ds

if __name__ == "__main__":
    ds = load_dataset()
