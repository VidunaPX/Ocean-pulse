# src/02_basic_exploration.py
import matplotlib.pyplot as plt
import os
from load_dataset import load_dataset

ds = load_dataset()
sst_anom = ds['anom']

# Global mean anomaly
global_mean = sst_anom.mean(dim=['lat','lon'])

plt.figure(figsize=(12,5))
global_mean.plot()
plt.title("Global Ocean Temperature Anomaly (1850-2025)")
plt.ylabel("Temperature Anomaly (°C)")
plt.xlabel("Time")
plt.grid(True)
os.makedirs("data/output", exist_ok=True)
plt.savefig("Frontend/public")
plt.show()
