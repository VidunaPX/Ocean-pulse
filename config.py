# src/config.py
OISST_MONTHLY_URL = "data/NOAAGlobalTemp_v6.0.0_gridded_s185001_e202508_c20250909T092005.nc"
ZARR_OISST = "data/oisst.zarr"
ZARR_SST_ANOM = "data/sst_anom.zarr"
BBOX = None   # None => global; or tuple(lon_min, lon_max, lat_min, lat_max)
START = "1982-01-01"
END = "2024-01-01"
COARSEN_FACTOR = 2  # 2 -> 0.5° if source is 0.25°
INPUT_LEN = 24      # months input
HORIZON = 3         # months ahead
