# src/data_prepare_plastic_advection.py

import pandas as pd
import numpy as np
import xarray as xr
import cftime
import os
import json

def build_trajectories(plastic_df, curr_ds, T_IN=5, T_OUT=5, dt_seconds=24*3600, N_synth=1000):
    """
    Build training trajectories for microplastics advection ML model.
    Adds synthetic global points to generate a globally distributed dataset.
    
    Args:
        plastic_df (DataFrame): columns ['datetime','latitude','longitude','concentration','plastic_type','u_current','v_current']
        curr_ds (xarray.Dataset): ocean currents dataset, must contain 'u', 'v', and 'time'
        T_IN (int): number of past timesteps to use as input
        T_OUT (int): number of future timesteps to predict
        dt_seconds (int): timestep in seconds (default=1 day)
        N_synth (int): number of synthetic points to generate globally

    Returns:
        X (np.ndarray): shape (N_samples, T_IN, 2) past positions/velocity
        Y (np.ndarray): shape (N_samples, T_OUT, 2) future positions
        meta (list): optional metadata for each sample
    """
    
    # ---------------- Convert time ----------------
    times = curr_ds['time'].values
    if isinstance(times[0], cftime.DatetimeJulian) or isinstance(times[0], cftime.DatetimeGregorian):
        time_index = pd.to_datetime([pd.Timestamp(t.strftime("%Y-%m-%d %H:%M:%S")) for t in times])
    else:
        time_index = pd.to_datetime(times)
    curr_ds = curr_ds.assign_coords(time=time_index)

    # ---------------- Sort real data ----------------
    plastic_df = plastic_df.sort_values("datetime").reset_index(drop=True)

    # ---------------- Generate synthetic global points ----------------
    lat_range = (-60, 60)   # avoid polar extremes
    lon_range = (-180, 180)
    synth_lats = np.random.uniform(lat_range[0], lat_range[1], N_synth)
    synth_lons = np.random.uniform(lon_range[0], lon_range[1], N_synth)
    synth_df = pd.DataFrame({
        "latitude": synth_lats,
        "longitude": synth_lons,
        "datetime": pd.to_datetime("2000-01-01"),  # fixed date, or random from time_index
        "concentration": np.random.uniform(0, 0.005, N_synth),
        "plastic_type": ["Synthetic"]*N_synth
    })

    # Combine real + synthetic
    plastic_df = pd.concat([plastic_df, synth_df], ignore_index=True)

    # ---------------- Build trajectories ----------------
    X_list, Y_list, meta_list = [], [], []

    for idx, row in plastic_df.iterrows():
        lat0, lon0, t0 = row["latitude"], row["longitude"], row["datetime"]
        if pd.isna(lat0) or pd.isna(lon0):
            continue

        # nearest time in current dataset
        t_idx = np.argmin(np.abs(time_index - t0))

        # Input sequence (past T_IN)
        X_seq, valid_seq = [], True
        for dt in range(-T_IN+1, 1):
            ti = t_idx + dt
            if ti < 0 or ti >= len(time_index):
                valid_seq = False
                break
            # sample u,v from nearest lat/lon (simple nearest neighbor)
            u = curr_ds["u"].isel(time=ti, latitude=0, longitude=0).values
            v = curr_ds["v"].isel(time=ti, latitude=0, longitude=0).values
            X_seq.append([u, v])
        if not valid_seq:
            continue

        # Output sequence (future T_OUT)
        Y_seq, valid_seq = [], True
        lat, lon = lat0, lon0
        for dt in range(1, T_OUT+1):
            ti = t_idx + dt
            if ti >= len(time_index):
                valid_seq = False
                break
            u = curr_ds["u"].isel(time=ti, latitude=0, longitude=0).values
            v = curr_ds["v"].isel(time=ti, latitude=0, longitude=0).values
            lat += v * dt_seconds
            lon += u * dt_seconds
            Y_seq.append([lat, lon])
        if not valid_seq:
            continue

        X_list.append(X_seq)
        Y_list.append(Y_seq)
        meta_list.append({
            "initial_lat": lat0,
            "initial_lon": lon0,
            "datetime": t0,
            "plastic_type": row.get("plastic_type", "unknown")
        })

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    
    print(f"✅ Built trajectories: X={X.shape}, Y={Y.shape}, samples={len(meta_list)}")
    return X, Y, meta_list


# ------------------- Example usage -------------------
if __name__ == "__main__":
    from load_plastic import load_plastic, load_currents

    # Load datasets
    plastic_df = load_plastic()
    curr_ds = load_currents()

    # Build trajectories
    T_IN, T_OUT = 100, 100
    plastic_df = plastic_df.sample(frac=1).reset_index(drop=True)  # shuffle
    X, Y, meta = build_trajectories(plastic_df, curr_ds, T_IN, T_OUT, N_synth=2000)

    # Save for ML
    os.makedirs("data/output", exist_ok=True)
    np.save("data/output/plastic_X.npy", X)
    np.save("data/output/plastic_Y.npy", Y)

    # Save meta
    with open("data/output/plastic_meta.json", "w") as f:
        json.dump(meta, f)

    print("✅ Trajectory datasets saved: plastic_X.npy, plastic_Y.npy, meta.json")
