# src/ingest.py
import xarray as xr
from dask.distributed import Client
from src.config import OISST_MONTHLY_URL, ZARR_OISST, BBOX, START, END

def run():
    # Start local dask client for parallel reads/writes (adjust workers locally)
    client = Client(n_workers=4, threads_per_worker=1, memory_limit="4GB")
    print("Dask client:", client)

    print("Opening OISST dataset:", OISST_MONTHLY_URL)
    ds = xr.open_dataset(OISST_MONTHLY_URL)
    # Optionally subset region:
    if BBOX:
        lon_min, lon_max, lat_min, lat_max = BBOX
        ds = ds.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
    ds = ds.sel(time=slice(START, END))
    # Keep only sst variable and chunk for performance
    ds = ds[["sst"]].chunk({'time': 12, 'lat': 180, 'lon': 360})
    print(ds)
    print("Writing to Zarr (this may take a while)...")
    ds.to_zarr(ZARR_OISST, consolidated=True)
    print("Saved:", ZARR_OISST)
    client.close()

if __name__ == "__main__":
    run()
