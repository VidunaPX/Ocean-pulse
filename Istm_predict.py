# src/05_lstm_predict.py
import torch
import xarray as xr
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from load_dataset import load_dataset
import os
import numpy as np

INPUT_LEN = 12  # past 12 months
HORIZON = 3     # predict next 3 months

# Folder to save models
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_oceanpulse.pth")

# Load dataset
ds = load_dataset()
sst_anom = ds['anom']

# Average over lat, lon, and z (your dataset has z dimension)
global_mean = sst_anom.mean(dim=['z', 'lat','lon']).values.astype('float32')


# Dataset class
class OceanSeqDataset(Dataset):
    def __init__(self, ts, input_len=INPUT_LEN, horizon=HORIZON):
        self.X, self.Y = [], []
        for i in range(len(ts) - input_len - horizon):
            seq = ts[i:i+input_len].squeeze()  # remove any extra dims
            target = ts[i+input_len:i+input_len+horizon].squeeze()
            self.X.append(seq)
            self.Y.append(target)
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.Y = torch.tensor(np.array(self.Y), dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


dataset = OceanSeqDataset(global_mean)
loader = DataLoader(dataset, batch_size=32, shuffle=True)


# LSTM model
class OceanLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, HORIZON)
    def forward(self, x):
        # x: (batch, seq_len)
        if x.dim() == 2:  # add feature dim
            x = x.unsqueeze(-1)  # now (batch, seq_len, 1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last time step
        out = self.fc(out)
        return out


model = OceanLSTM()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()


# Training loop
for epoch in range(64):
    epoch_loss = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
    print(f"Epoch {epoch+1}, Loss={epoch_loss/len(dataset):.4f}")


# Save model
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")


# Quick prediction
last_seq = torch.tensor(global_mean[-INPUT_LEN:], dtype=torch.float32).unsqueeze(0)
pred_next3 = model(last_seq).detach().numpy()[0]
print("Predicted next 3 months anomaly:", pred_next3)
