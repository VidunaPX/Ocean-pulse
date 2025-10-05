"""
Visualize predicted microplastic drift (absolute lat/lon trajectories)
with ocean current overlays.

Expected files:
- data/output/plastic_preds.npy  → (N, T_out, 2)
- data/output/plastic_X.npy      → (N, T_in, features)
- data/output/ocean_currents.npy → (H, W, 4) [lat_grid, lon_grid, u, v] (optional)
"""

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
CURR_PATH = os.path.join(DATA_DIR, "ocean_currents.npy")

# =====================================================
# LOAD DATA
# =====================================================
if not os.path.exists(PRED_PATH):
    raise FileNotFoundError(f"Missing file: {PRED_PATH}")
if not os.path.exists(X_PATH):
    raise FileNotFoundError(f"Missing file: {X_PATH}")

preds = np.load(PRED_PATH)  # (N, T_out, 2)
X = np.load(X_PATH)         # (N, T_in, F)
start_points = [(X[i, -1, 0], X[i, -1, 1]) for i in range(len(preds))]

# =====================================================
# LOAD OCEAN CURRENT DATA (optional)
# =====================================================
currents_available = os.path.exists(CURR_PATH)
if currents_available:
    data = np.load(CURR_PATH, allow_pickle=True).item() if CURR_PATH.endswith(".npy") else None
    if isinstance(data, dict):
        lat_grid = data["lat"]
        lon_grid = data["lon"]
        u = data["u"]
        v = data["v"]
    else:
        # assume standard layout: (H, W, 4) = [lat, lon, u, v]
        arr = np.load(CURR_PATH)
        lat_grid = arr[:, :, 0]
        lon_grid = arr[:, :, 1]
        u = arr[:, :, 2]
        v = arr[:, :, 3]
else:
    print("⚠️ No ocean current data found. Continuing without overlay.")
    lat_grid = lon_grid = u = v = None

# =====================================================
# VISUALIZATION SETTINGS
# =====================================================
T_out = preds.shape[1]
Nplot = min(10, len(preds))
cmap = plt.get_cmap("plasma")
timesteps_to_show = min(T_out, 90)

# =====================================================
# CREATE MAP
# =====================================================
fig = plt.figure(figsize=(14, 7))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()

ax.coastlines(resolution="110m", linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linewidth=0.4)
ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)
ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

# =====================================================
# OVERLAY OCEAN CURRENTS
# =====================================================
if currents_available:
    print("✅ Overlaying ocean current vectors...")
    # Downsample grid for readability
    step = max(1, int(lat_grid.shape[0] / 40))
    ax.quiver(
        lon_grid[::step, ::step],
        lat_grid[::step, ::step],
        u[::step, ::step],
        v[::step, ::step],
        transform=ccrs.PlateCarree(),
        scale=3,
        color="dodgerblue",
        alpha=0.5,
        width=0.0025,
        headwidth=3,
    )
else:
    print("⚠️ Ocean current vectors not available.")

# =====================================================
# PLOT TRAJECTORIES
# =====================================================
for i in range(Nplot):
    start_lat, start_lon = start_points[i]
    traj = preds[i, :timesteps_to_show]
    lats, lons = traj[:, 0], traj[:, 1]

    # Normalize longitudes
    lons = (lons + 180) % 360 - 180

    # Draw path
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

    # Markers
    ax.plot(start_lon, start_lat, "go", markersize=5, transform=ccrs.Geodetic(), label="Start" if i == 0 else "")
    ax.plot(lons[-1], lats[-1], "ro", markersize=5, transform=ccrs.Geodetic(), label="End" if i == 0 else "")

# =====================================================
# TITLE / LEGEND
# =====================================================
plt.title("🌊 Predicted Microplastic Drift with Ocean Currents", fontsize=16, pad=12)
plt.legend(loc="lower left", fontsize=8)
plt.tight_layout()

# Save & Show
save_path = os.path.join("Frontend/public")
plt.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"✅ Saved visualization to: {save_path}")
plt.show()
