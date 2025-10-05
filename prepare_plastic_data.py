# src/prepare_plastic_data.py
import os
import numpy as np
import pandas as pd
import xarray as xr
from load_plastic import load_plastic, load_currents, merge_plastic_currents

# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "data/output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- Hyperparameters ----------------
T_IN = 6   # input timesteps
T_OUT = 30  # output timesteps

# ---------------- Load & merge datasets ----------------
print("Loading plastic data...")
plastic_df = load_plastic()
print("Loading ocean currents...")
curr_ds = load_currents()
print("Merging plastic & currents...")
merged_df = merge_plastic_currents(plastic_df, curr_ds)

# NOTE: merged_df now includes enhanced east and west ocean points for realism from load_plastic.py

# Clean datetime column before filtering
merged_df = merged_df[pd.to_datetime(merged_df['datetime'], errors='coerce').notnull()]
merged_df['datetime'] = pd.to_datetime(merged_df['datetime'], errors='coerce')
merged_df = merged_df[merged_df['datetime'] >= pd.Timestamp('2020-01-01')].reset_index(drop=True)
print(f"After filtering, date range: {merged_df['datetime'].min()} to {merged_df['datetime'].max()}")
print(f"Latitude range: {merged_df['latitude'].min()} to {merged_df['latitude'].max()}")
print(f"Longitude range: {merged_df['longitude'].min()} to {merged_df['longitude'].max()}")

# ---------------- Build sequences ----------------
def build_trajectories(plastic_df, curr_ds, T_IN, T_OUT):
    """
    Convert plastic + currents data into LSTM sequences.
    X: past T_IN timesteps -> [lat, lon, u, v]
    Y: next T_OUT timesteps -> delta lat/lon
    """
    X_list, Y_list, meta = [], [], []

    # Sort by datetime
    plastic_df = plastic_df.sort_values("datetime").reset_index(drop=True)

    for i in range(len(plastic_df) - T_IN - T_OUT):
        # Input features
        X_seq = []
        for t in range(T_IN):
            row = plastic_df.iloc[i + t]
            lat, lon = row["latitude"], row["longitude"]
            dt = row["datetime"]

            # Handle missing currents
            try:
                u = curr_ds["u"].sel(latitude=lat, longitude=lon, time=dt, method="nearest").values.item()
                v = curr_ds["v"].sel(latitude=lat, longitude=lon, time=dt, method="nearest").values.item()
            except:
                u, v = 0.0, 0.0

            X_seq.append([lat, lon, u, v])
        X_list.append(np.array(X_seq))

        # Output deltas
        Y_seq = []
        for t in range(T_OUT):
            curr = plastic_df.iloc[i + T_IN + t]
            prev = plastic_df.iloc[i + T_IN + t - 1]
            dlat = curr["latitude"] - prev["latitude"]
            dlon = curr["longitude"] - prev["longitude"]
            Y_seq.append([dlat, dlon])
        Y_list.append(np.array(Y_seq))

        # Save metadata (middle of sequence)
        meta.append(plastic_df.iloc[i + T_IN]["datetime"])

    X = np.array(X_list)
    Y = np.array(Y_list)
    return X, Y, meta

# ---------------- Build & save ----------------
print("Building trajectories...")
X, Y, meta = build_trajectories(merged_df, curr_ds, T_IN, T_OUT)
print(f"Built X shape: {X.shape}, Y shape: {Y.shape}")

np.save(os.path.join(OUTPUT_DIR, "plastic_X.npy"), X)
np.save(os.path.join(OUTPUT_DIR, "plastic_Y.npy"), Y)
print(f"Saved plastic_X.npy and plastic_Y.npy to {OUTPUT_DIR}")
