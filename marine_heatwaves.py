# marine_heatwaves.py
# Marine Heatwave Detection & Visualization Module
# Works standalone and when imported into OceanPulse

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy import ndimage

# -----------------------
# Dataset Loader
# -----------------------
def load_dataset():
    """
    Dummy loader for SST anomaly dataset.
    Returns:
        xr.Dataset with variables:
        - 'anom': SST anomaly [time, lat, lon]
        - 'lat', 'lon'
    """
    time = np.arange(0, 24)  # months
    lat = np.linspace(-90, 90, 36)
    lon = np.linspace(0, 360, 72)
    anom = np.random.rand(len(time), len(lat), len(lon)) * 2 - 1  # random anomalies
    ds = xr.Dataset(
        {"anom": (("time", "lat", "lon"), anom)},
        coords={"time": time, "lat": lat, "lon": lon}
    )
    return ds

# -----------------------
# Marine Heatwave Detection
# -----------------------
def detect_marine_heatwaves_advanced(sst_anom, baseline_period=5, duration_days=3, spatial_size=3):
    """
    Detect Marine Heatwave Zones.
    Returns a dict with zones, frequency, intensity, duration.
    """
    print("🌊 Detecting Marine Heatwaves...")

    # Rolling 90th percentile over time
    def percentile90(x, axis=None, **kwargs):
        return np.nanpercentile(x, 90, axis=axis)

    climatology = sst_anom.rolling(time=baseline_period, center=True).reduce(percentile90)

    # Identify events above threshold
    events = sst_anom > climatology

    # Duration filter
    duration_mask = events.rolling(time=duration_days, center=True).sum() >= duration_days

    # Spatial coherence filter
    zones = xr.zeros_like(duration_mask)
    for t in range(len(duration_mask.time)):
        slice_data = duration_mask.isel(time=t).values.astype(float)
        spatial_mask = ndimage.uniform_filter(slice_data, size=spatial_size, mode='constant') >= 0.5
        zones[t, :, :] = spatial_mask

    # Statistics
    frequency = zones.sum(dim='time')
    intensity = (sst_anom - climatology).where(zones).mean(dim='time')
    max_duration = zones.rolling(time=30, center=True).sum().max(dim='time')

    return {
        'zones': zones,
        'frequency': frequency,
        'intensity': intensity,
        'duration': max_duration
    }


# -----------------------
# Visualization
# -----------------------
def create_enhanced_heatwave_visualization(mhw_data, lat, lon, save_path=None):
    """
    Create a visualization of marine heatwaves.
    Returns: matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(18, 12))
    lon2d, lat2d = np.meshgrid(lon, lat)

    frequency = np.squeeze(mhw_data['frequency'].values)
    intensity = np.squeeze(mhw_data['intensity'].values)
    duration = np.squeeze(mhw_data['duration'].values)

    # Frequency
    ax1 = plt.subplot(2, 2, 1, projection=ccrs.PlateCarree())
    ax1.set_global()
    ax1.coastlines()
    im1 = ax1.pcolormesh(lon2d, lat2d, frequency, cmap='Reds', shading='auto')
    ax1.set_title("Marine Heatwave Frequency")
    plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.05)

    # Intensity
    ax2 = plt.subplot(2, 2, 2, projection=ccrs.PlateCarree())
    ax2.set_global()
    ax2.coastlines()
    im2 = ax2.pcolormesh(lon2d, lat2d, intensity, cmap='RdYlBu_r', shading='auto')
    ax2.set_title("Average Heatwave Intensity (°C above threshold)")
    plt.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.05)

    # Duration
    ax3 = plt.subplot(2, 2, 3, projection=ccrs.PlateCarree())
    ax3.set_global()
    ax3.coastlines()
    im3 = ax3.pcolormesh(lon2d, lat2d, duration, cmap='Oranges', shading='auto')
    ax3.set_title("Maximum Heatwave Duration (days)")
    plt.colorbar(im3, ax=ax3, orientation='horizontal', pad=0.05)

    plt.suptitle("🌊 Marine Heatwave Zones", fontsize=18, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig("Frontend/public", dpi=300, bbox_inches='tight')
        print(f"📊 Marine heatwave visualization saved to {save_path}")

    return fig

# -----------------------
# Standalone Execution
# -----------------------
if __name__ == "__main__":
    print("🌊 Loading dataset...")
    ds = load_dataset()
    sst_anom = ds['anom']
    lat = ds['lat'].values
    lon = ds['lon'].values

    print("🔍 Detecting Marine Heatwave Zones...")
    mhw_data = detect_marine_heatwaves_advanced(sst_anom)

    print("📊 Creating visualizations...")
    create_enhanced_heatwave_visualization(mhw_data, lat, lon, save_path="Frontend/public")

    print("✅ Marine Heatwave analysis completed!")
