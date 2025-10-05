# src/load_plastic.py
import json
import pandas as pd
import numpy as np
import xarray as xr

OCEAN_CURRENTS_FILE = "data/gocd_a0002341_M0107.nc"   # Copernicus currents
PLASTIC_FILE = "data/Marine_Microplastics_WGS84_7859561165955330418.json"  # Microplastic JSON metadata

def load_plastic(file=PLASTIC_FILE):
    """
    Load and enhance microplastic dataset with globally realistic, gyre-weighted synthetic data.
    Focuses on regions like the Great Pacific Garbage Patch, Atlantic Gyre, and Indian Ocean Gyre.
    """
    with open(file, "r") as f:
        data = json.load(f)

    records = []
    transport_type = data.get("transportType", "unknown")

    for layer in data.get("layers", []):
        for feature in layer.get("features", []):
            geom = feature.get("geometry", {})
            attrs = feature.get("attributes", {})
            x, y = geom.get("x"), geom.get("y")
            if x is None or y is None:
                continue

            dt = None
            if "Date_m_d_yyyy" in attrs and attrs["Date_m_d_yyyy"]:
                try:
                    dt = pd.to_datetime(attrs["Date_m_d_yyyy"], unit="ms")
                except Exception:
                    dt = None

            records.append({
                "transportType": transport_type,
                "longitude": x,
                "latitude": y,
                "concentration": attrs.get("Microplastics_measurement", None),
                "unit": attrs.get("Unit", None),
                "plastic_type": attrs.get("Medium", "unknown"),
                "datetime": dt
            })

    df = pd.DataFrame(records)

    # Generate synthetic realistic dates if missing
    if df["datetime"].isnull().all():
        total = len(df)
        recent = np.random.choice(pd.date_range("2020-01-01", "2025-12-31", freq="D"), total)
        np.random.shuffle(recent)
        df["datetime"] = pd.to_datetime(recent[:len(df)])

    # Add realistic, gyre-focused global enhancement
    df = enhance_realistic_global_distribution(df)
    
    print("✅ Extracted & enhanced plastic data sample:\n", df.head())
    print(f"📊 Data range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"🌍 Global coverage: {df['latitude'].min():.2f}°–{df['latitude'].max():.2f}°, {df['longitude'].min():.2f}°–{df['longitude'].max():.2f}")
    return df


def enhance_realistic_global_distribution(df):
    """
    Add synthetic data weighted toward known microplastic gyres and accumulation zones.
    """
    gyres = [
        # Major accumulation zones
        {"name": "Great Pacific Garbage Patch", "lat_range": (25, 38), "lon_range": (-160, -135), "boost": 8.0},
        {"name": "North Atlantic Gyre", "lat_range": (20, 40), "lon_range": (-70, -20), "boost": 5.5},
        {"name": "South Pacific Gyre", "lat_range": (-40, -20), "lon_range": (-150, -80), "boost": 4.5},
        {"name": "Indian Ocean Gyre", "lat_range": (-40, 20), "lon_range": (50, 100), "boost": 4.0},
        {"name": "South Atlantic Gyre", "lat_range": (-40, -20), "lon_range": (-50, 10), "boost": 3.5},
        {"name": "Arctic Ocean", "lat_range": (65, 85), "lon_range": (-180, 180), "boost": 2.0},
        {"name": "Southern Ocean", "lat_range": (-80, -60), "lon_range": (-180, 180), "boost": 1.0},
    ]

    records = []
    for region in gyres:
        n_points = np.random.randint(80, 150)  # denser sampling in gyres
        for _ in range(n_points):
            lat = np.random.uniform(*region["lat_range"])
            lon = np.random.uniform(*region["lon_range"])
            # Higher mean concentration in high-boost zones
            base_conc = np.random.lognormal(mean=0.8, sigma=0.5) * region["boost"]
            conc = np.clip(base_conc, 0.05, 500)  # cap extreme values
            date = pd.Timestamp("2020-01-01") + pd.Timedelta(days=np.random.randint(0, 365*6))
            records.append({
                "transportType": "synthetic_enhancement",
                "longitude": lon,
                "latitude": lat,
                "concentration": conc,
                "unit": "particles/m³",
                "plastic_type": np.random.choice(["microplastic", "nanoplastic", "fiber"]),
                "datetime": date
            })

    df = pd.concat([df, pd.DataFrame(records)], ignore_index=True)
    return df


def load_currents(file=OCEAN_CURRENTS_FILE):
    ds = xr.open_dataset(file)
    print("✅ Ocean currents dataset loaded:", list(ds.data_vars))
    return ds


def merge_plastic_currents(plastic_df, currents_ds):
    """
    Combine plastic points with corresponding current velocity data for ML input.
    """
    if "time" not in currents_ds.dims:
        raise ValueError("Currents dataset has no time dimension!")

    merged_data = []
    for _, row in plastic_df.iterrows():
        lat, lon, time = row["latitude"], row["longitude"], row["datetime"]
        if lat is None or lon is None:
            continue

        try:
            u = currents_ds["u"].sel(latitude=lat, longitude=lon, time=time, method="nearest").values.item()
            v = currents_ds["v"].sel(latitude=lat, longitude=lon, time=time, method="nearest").values.item()
        except Exception:
            u, v = np.nan, np.nan

        merged_data.append({
            "datetime": time,
            "latitude": lat,
            "longitude": lon,
            "concentration": row.get("concentration", np.nan),
            "plastic_type": row.get("plastic_type", "unknown"),
            "u_current": u,
            "v_current": v
        })

    merged_df = pd.DataFrame(merged_data)
    print("✅ Merged plastic & current data sample:\n", merged_df.head())
    return merged_df


if __name__ == "__main__":
    plastic_df = load_plastic()
    currents_ds = load_currents()
    merged_df = merge_plastic_currents(plastic_df, currents_ds)
    print(f"✅ Final dataset ready for ML training: {len(merged_df)} samples")
