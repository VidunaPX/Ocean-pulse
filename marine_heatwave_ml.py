# marine_heatwave_ml.py
# Predict Marine Heatwave Zones using SST anomalies

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ----------------------
# Dataset loader
# ----------------------
def load_dataset():
    """
    Dummy loader for SST anomalies dataset.
    Returns xarray Dataset with 'anom' [time, lat, lon], lat, lon
    Replace with your real dataset.
    """
    time = np.arange(0, 36)  # months
    lat = np.linspace(-90, 90, 36)
    lon = np.linspace(0, 360, 72)
    anom = np.random.rand(len(time), len(lat), len(lon)) * 2 - 1  # anomalies -1 to +1°C
    ds = xr.Dataset(
        {"anom": (("time", "lat", "lon"), anom)},
        coords={"time": time, "lat": lat, "lon": lon}
    )
    return ds

# ----------------------
# Label heatwave zones (for supervised learning)
# ----------------------
def label_heatwave_zones(sst_anom, threshold=0.8):
    """
    Label data: 1 if anomaly > threshold (heatwave), else 0
    """
    return (sst_anom > threshold).astype(int)

# ----------------------
# Prepare features & labels
# ----------------------
def prepare_ml_data(sst_anom, labels, lookback=3):
    """
    Convert time-series 3D data into ML features per grid cell
    Args:
        sst_anom: [time, lat, lon]
        labels: same shape, 0/1
        lookback: number of past months as features
    Returns:
        X: [samples, features], y: [samples]
    """
    time_len, lat_len, lon_len = sst_anom.shape
    features = []
    targets = []

    for t in range(lookback, time_len):
        past = sst_anom[t-lookback:t, :, :].reshape(-1, lookback)
        target = labels[t, :, :].flatten()
        features.append(past)
        targets.append(target)

    X = np.vstack(features)
    y = np.hstack(targets)
    return X, y

# ----------------------
# Train ML model
# ----------------------
def train_heatwave_model(X, y):
    """
    Train Random Forest Classifier
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("🌊 Heatwave Prediction Model Evaluation:")
    print(classification_report(y_test, y_pred, digits=3))
    return clf

# ----------------------
# Predict and visualize zones
# ----------------------
def predict_and_plot(model, sst_anom, lat, lon, lookback=3, save_path="data/output/heatwave_prediction.png"):
    """
    Predict heatwave zones for the last time step
    """
    # Use last lookback months as features
    features = sst_anom[-lookback:, :, :].reshape(-1, lookback)
    y_pred = model.predict(features).reshape(len(lat), len(lon))

    # Plot
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines()
    im = ax.pcolormesh(lon, lat, y_pred, cmap='Reds', shading='auto')
    ax.set_title("Predicted Marine Heatwave Zones (Next Month)")
    plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# ----------------------
# Main Execution
# ----------------------
if __name__ == "__main__":
    print("🌊 Loading SST dataset...")
    ds = load_dataset()
    sst_anom = ds['anom'].values  # shape [time, lat, lon]
    lat = ds['lat'].values
    lon = ds['lon'].values

    print("🔖 Labeling heatwave zones...")
    labels = label_heatwave_zones(sst_anom, threshold=0.8)

    print("📊 Preparing ML data...")
    X, y = prepare_ml_data(sst_anom, labels, lookback=3)

    print("🤖 Training heatwave prediction model...")
    model = train_heatwave_model(X, y)

    print("🌐 Predicting and visualizing heatwave zones...")
    predict_and_plot(model, sst_anom, lat, lon)

    print("✅ Heatwave prediction completed!")
