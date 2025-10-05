# src/load_dataset.py
import xarray as xr

DATA_FILE = "data/NOAAGlobalTemp_v6.0.0_gridded_s185001_e202508_c20250909T092005.nc"

def load_dataset(file=DATA_FILE):
    ds = xr.open_dataset(file)
    print(ds)
    return ds

if __name__ == "__main__":
    ds = load_dataset()
