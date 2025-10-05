import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# =====================================================
# PATHS
# =====================================================
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "../src/data/output")

PRED_PATH = os.path.join(DATA_DIR, "plastic_preds.npy")
X_PATH = os.path.join(DATA_DIR, "plastic_X.npy")

# =====================================================
# LOAD DATA
# =====================================================
if not os.path.exists(PRED_PATH):
    raise FileNotFoundError(f"Missing file: {PRED_PATH}")
if not os.path.exists(X_PATH):
    raise FileNotFoundError(f"Missing file: {X_PATH}")

# Predicted trajectories: shape (N, T_out, 2) = [lat, lon]
preds = np.load(PRED_PATH)
# Input sequences: shape (N, T_in, features)
X = np.load(X_PATH)

# Extract last known positions as starting points
start_points = [(X[i, -1, 0], X[i, -1, 1]) for i in range(len(preds))]

# =====================================================
# VISUALIZATION SETTINGS
# =====================================================
T_out = preds.shape[1]
Nplot = min(10, len(preds))
cmap = plt.get_cmap("plasma")

# Optional: crop long trajectories to display manageable amount
timesteps_to_show = min(T_out, 90)

# =====================================================
# CREATE MAP
# =====================================================
fig = plt.figure(figsize=(14, 7))
ax = plt.axes(projection=ccrs.PlateCarree())

ax.set_global()
ax.coastlines(resolution='110m', linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linewidth=0.4)
ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

# =====================================================
# PLOT TRAJECTORIES
# =====================================================
for i in range(Nplot):
    start_lat, start_lon = start_points[i]

    # Model output is *absolute lat/lon*, not deltas
    traj = preds[i, :timesteps_to_show]
    lats = traj[:, 0]
    lons = traj[:, 1]

    # Ensure longitudes are within [-180, 180]
    lons = (lons + 180) % 360 - 180

    # Draw trajectory line
    for j in range(len(lats) - 1):
        color = cmap(j / len(lats))
        alpha = 0.3 + 0.7 * (j / len(lats))
        ax.plot(
            lons[j:j+2],
            lats[j:j+2],
            color=color,
            alpha=alpha,
            linewidth=2.0,
            transform=ccrs.Geodetic(),
        )

    # Mark start and end points
    ax.plot(start_lon, start_lat, "go", markersize=5, transform=ccrs.Geodetic(), label="Start" if i == 0 else "")
    ax.plot(lons[-1], lats[-1], "ro", markersize=5, transform=ccrs.Geodetic(), label="End" if i == 0 else "")

# =====================================================
# ANNOTATIONS
# =====================================================
plt.title("🌊 Predicted Microplastic Drift (LSTM Model)", fontsize=16, pad=12)
plt.legend(loc="lower left", fontsize=8)
plt.tight_layout()

# Save & Show
save_path = os.path.join("Frontend/public")
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"✅ Saved visualization to: {save_path}")
plt.show()
