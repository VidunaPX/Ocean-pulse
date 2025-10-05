# src/visualize_predictions.py

import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr

# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "data/output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
PRED_PATH = os.path.join(OUTPUT_DIR, "pred_next3.npy")
DATA_PATH = os.path.join(BASE_DIR, "../data/NOAAGlobalTemp_v6.0.0_gridded_s185001_e202508_c20250909T092005.nc")

# ---------------- Load Prediction ----------------
pred_next3 = np.load(PRED_PATH)  # shape: (3, lat, lon)
print(f"Loaded prediction array with shape: {pred_next3.shape}")

# ---------------- Load Coordinates ----------------
ds = xr.open_dataset(DATA_PATH)
lat = ds['lat'].values
lon = ds['lon'].values

# ---------------- Enhanced Visualization with Consistent Styling ----------------
def plot_global_anomaly(anomaly, month_idx):
    """
    Create enhanced visualization with consistent styling and professional appearance
    """
    # Create figure with consistent styling
    fig = plt.figure(figsize=(16, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    
    # Enhanced map features for professional appearance
    ax.coastlines(resolution='50m', linewidth=0.8, color='#2d3748', alpha=0.9)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, color='#4a5568', alpha=0.7)
    ax.add_feature(cfeature.OCEAN, color='#1a365d', alpha=0.1)
    ax.add_feature(cfeature.LAND, color='#2d3748', alpha=0.1)
    ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.6, color='#718096')
    
    # Consistent colormap and range across all visualizations
    cmap = plt.cm.RdYlBu_r  # Red-Yellow-Blue reversed for ocean temperatures
    vmin, vmax = -3.0, 3.0  # Consistent range across all months
    
    # Plot with consistent styling
    im = ax.pcolormesh(lon, lat, anomaly, cmap=cmap, vmin=vmin, vmax=vmax, 
                      shading='auto', alpha=0.85, transform=ccrs.PlateCarree())
    
    # Enhanced colorbar with consistent styling
    cbar = plt.colorbar(im, orientation='horizontal', pad=0.08, shrink=0.8, aspect=40)
    cbar.set_label('Predicted SST Anomaly (°C)', fontsize=14, fontweight='bold', 
                   color='#2d3748', labelpad=15)
    cbar.ax.tick_params(colors='#2d3748', labelsize=12)
    
    # Enhanced title with consistent branding
    plt.title(f'🌊 OceanPulse: Predicted SST Anomaly - Month {month_idx+1}\n'
              f'Planetary Health Tracker | {month_idx+1} Month Forecast', 
              fontsize=18, fontweight='bold', color='#1a365d', pad=25)
    
    # Add risk assessment subtitle
    risk_assessment = get_risk_assessment(anomaly, month_idx)
    plt.figtext(0.5, 0.02, risk_assessment, ha='center', fontsize=11, 
                color='#4a5568', style='italic', alpha=0.9)
    
    # Save with consistent high quality
    save_path = os.path.join("Frontend/public", f"predicted_sst_anomaly_month{month_idx+1}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"✅ Saved enhanced plot: {save_path}")

def get_risk_assessment(anomaly, month_idx):
    """
    Generate consistent risk assessment across all visualizations
    """
    max_anomaly = np.max(anomaly)
    min_anomaly = np.min(anomaly)
    mean_anomaly = np.mean(anomaly)
    std_anomaly = np.std(anomaly)
    
    # Risk assessment based on anomaly patterns
    if max_anomaly > 2.5:
        risk_level = "🔴 CRITICAL RISK"
        impact = "Severe marine heatwaves expected - immediate action required"
    elif max_anomaly > 2.0:
        risk_level = "🟠 HIGH RISK"
        impact = "Marine heatwaves likely - fisheries and coral reefs at risk"
    elif max_anomaly > 1.0:
        risk_level = "🟡 MODERATE RISK"
        impact = "Temperature stress possible - monitor marine ecosystems"
    else:
        risk_level = "🟢 LOW RISK"
        impact = "Normal conditions expected - continue monitoring"
    
    return (f"{risk_level} | {impact} | "
            f"Mean: {mean_anomaly:.2f}°C | "
            f"Range: {min_anomaly:.2f}°C to {max_anomaly:.2f}°C | "
            f"Std: {std_anomaly:.2f}°C")

# ---------------- Generate Plots ----------------
for i in range(pred_next3.shape[0]):
    plot_global_anomaly(pred_next3[i], i)

print("All predictions visualized successfully!")
