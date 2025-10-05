# src/04_region_anomalies.py
import matplotlib.pyplot as plt
import os
from load_dataset import load_dataset

ds = load_dataset()
sst_anom = ds['anom']

# Pacific: lat -10 to 10, lon 120 to 280
pacific = sst_anom.sel(lat=slice(-10,10), lon=slice(120,280)).mean(dim=['lat','lon'])
# Indian Ocean: lat -30 to 30, lon 40 to 120
indian = sst_anom.sel(lat=slice(-30,30), lon=slice(40,120)).mean(dim=['lat','lon'])

plt.figure(figsize=(12,5))
pacific.plot(label='Pacific Ocean')
indian.plot(label='Indian Ocean')
plt.title("Regional Ocean Temperature Anomaly")
plt.ylabel("Temperature Anomaly (°C)")
plt.xlabel("Time")
plt.legend()
plt.grid(True)
os.makedirs("data/output", exist_ok=True)
plt.savefig("Frontend/public")
plt.show()
