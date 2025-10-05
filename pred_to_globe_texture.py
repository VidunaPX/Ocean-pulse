# src/pred_to_globe_texture.py
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr

class GlobeTextureGenerator:
    """
    Generates enhanced globe textures and simple textures for 3D globe
    from SST anomaly predictions
    """
    def __init__(self):
        self.BASE_DIR = os.path.dirname(__file__)
        self.DATA_DIR = os.path.join(self.BASE_DIR, "data/output")
        self.OUT_DIR = os.path.join(self.BASE_DIR, "../Frontend/public")  # Save to Frontend/public

        self.PRED_PATH = os.path.join(self.DATA_DIR, "pred_next3.npy")
        self.DATA_PATH = os.path.join(self.BASE_DIR, "../data/NOAAGlobalTemp_v6.0.0_gridded_s185001_e202508_c20250909T092005.nc")

        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.OUT_DIR, exist_ok=True)

        # Load prediction
        self.pred = np.load(self.PRED_PATH)
        print(f"✅ Loaded prediction with shape: {self.pred.shape}")

        # Load coordinates
        ds = xr.open_dataset(self.DATA_PATH)
        self.lat = ds['lat'].values
        self.lon = ds['lon'].values

    def run(self):
        print("🌊 Generating enhanced globe textures for 3-month prediction...")
        for month_idx in range(3):
            print(f"\n📅 Processing Month {month_idx+1}...")
            month_data = self.pred[month_idx]

            # Enhanced globe texture with Orthographic projection
            self.create_enhanced_globe_texture(month_data, month_idx)

            # Simple 3D texture for mapping
            self.create_simple_texture(month_data, month_idx)
        print("\n🎉 All globe textures generated successfully!")

    def create_enhanced_globe_texture(self, anomaly_data, month_idx):
        # Orthographic projection for globe overlay
        fig = plt.figure(figsize=(12, 12))
        ax = plt.axes(projection=ccrs.Orthographic(central_longitude=0, central_latitude=0))
        ax.set_global()

        # Land, ocean, borders
        ax.add_feature(cfeature.LAND, facecolor='#2d3748', alpha=0.3)
        ax.add_feature(cfeature.OCEAN, facecolor='#1e3a8a', alpha=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, color='white', alpha=0.6)
        ax.coastlines(resolution='50m', linewidth=0.5, color='white', alpha=0.8)

        # Plot SST anomaly on globe
        cmap = plt.cm.RdYlBu_r
        vmin, vmax = -3, 3
        im = ax.pcolormesh(self.lon, self.lat, anomaly_data, cmap=cmap,
                           vmin=vmin, vmax=vmax, shading='auto',
                           alpha=0.8, transform=ccrs.PlateCarree())

        cbar = plt.colorbar(im, orientation='horizontal', pad=0.05, shrink=0.8, aspect=30)
        cbar.set_label(f'Predicted SST Anomaly (°C) - Month {month_idx+1}', fontsize=12, fontweight='bold', color='white')
        cbar.ax.tick_params(colors='white', labelsize=10)

        # Title and impact
        plt.title(f'🌊 OceanPulse: Predicted SST Anomaly - Month {month_idx+1}',
                  fontsize=16, fontweight='bold', color='white', pad=20)
        impact_text = self.get_impact_assessment(anomaly_data, month_idx)
        plt.figtext(0.5, 0.02, impact_text, ha='center', fontsize=10, color='white', style='italic', alpha=0.8)

        # Save enhanced globe texture
        output_path = os.path.join(self.OUT_DIR, f"predicted_sst_globe_month{month_idx+1}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black')
        plt.close()
        print(f"✅ Saved enhanced globe texture with globe overlay: {output_path}")

    def create_simple_texture(self, anomaly_data, month_idx):
        norm_data = (anomaly_data - np.min(anomaly_data)) / (np.max(anomaly_data) - np.min(anomaly_data))
        cmap = cm.get_cmap("RdYlBu_r")
        rgb = cmap(norm_data)[:, :, :3]
        rgb_uint8 = (rgb * 255).astype(np.uint8)
        rgb_uint8 = np.flipud(rgb_uint8)
        texture_path = os.path.join(self.OUT_DIR, f"globe_texture_month{month_idx+1}.png")
        Image.fromarray(rgb_uint8).save(texture_path)
        print(f"✅ Saved simple texture: {texture_path}")

    def get_impact_assessment(self, anomaly_data, month_idx):
        max_anom = np.max(anomaly_data)
        min_anom = np.min(anomaly_data)
        mean_anom = np.mean(anomaly_data)
        if max_anom > 2.0:
            return f"🔴 HIGH RISK | Marine heatwaves likely | Mean: {mean_anom:.2f}°C | Range: {min_anom:.2f}°C to {max_anom:.2f}°C"
        elif max_anom > 1.0:
            return f"🟡 MODERATE RISK | Monitor coral reefs | Mean: {mean_anom:.2f}°C | Range: {min_anom:.2f}°C to {max_anom:.2f}°C"
        else:
            return f"🟢 LOW RISK | Normal conditions | Mean: {mean_anom:.2f}°C | Range: {min_anom:.2f}°C to {max_anom:.2f}°C"

if __name__ == "__main__":
    generator = GlobeTextureGenerator()
    generator.run()
