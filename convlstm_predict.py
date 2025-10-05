# src/convlstm_predict.py

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import sys

# ---------------- Parameters ----------------
INPUT_LEN = 12   # past 12 months
HORIZON = 3      # predict next 3 months
BATCH_SIZE = 256
EPOCHS = 100
LR = 0.001

# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "../data/NOAAGlobalTemp_v6.0.0_gridded_s185001_e202508_c20250909T092005.nc")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "convlstm_oceanpulse.pth")

OUTPUT_DIR = os.path.join(BASE_DIR, "data/output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
PRED_PATH = os.path.join(OUTPUT_DIR, "pred_next3.npy")

# ---------------- Load Dataset ----------------
ds = xr.open_dataset(DATA_PATH)
sst_anom = ds['anom'].squeeze('z')  # shape: (time, lat, lon)

# ---------------- Dataset Class ----------------
class OceanConvSeqDataset(Dataset):
    def __init__(self, data, input_len=12, horizon=3):
        self.X, self.Y = [], []
        data_np = data.values.astype('float32')
        for i in range(len(data_np) - input_len - horizon):
            x_seq = data_np[i:i+input_len]   # (input_len, lat, lon)
            y_seq = data_np[i+input_len:i+input_len+horizon]
            x_seq = x_seq[:, None, :, :]     # (input_len, 1, lat, lon)
            y_seq = y_seq[:, None, :, :]
            self.X.append(x_seq)
            self.Y.append(y_seq)
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.Y = torch.tensor(np.array(self.Y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

dataset = OceanConvSeqDataset(sst_anom, INPUT_LEN, HORIZON)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---------------- ConvLSTM ----------------
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_channels + hidden_channels,
                              4 * hidden_channels,
                              kernel_size,
                              padding=padding)
        self.hidden_channels = hidden_channels

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        conv_out = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_out, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class OceanConvLSTM(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=8, kernel_size=3, horizon=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.horizon = horizon
        self.cell = ConvLSTMCell(input_channels, hidden_channels, kernel_size)
        self.conv_out = nn.Conv2d(hidden_channels, horizon, kernel_size=1)

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        h = torch.zeros(batch_size, self.hidden_channels, H, W, device=x.device)
        c = torch.zeros(batch_size, self.hidden_channels, H, W, device=x.device)
        for t in range(seq_len):
            h, c = self.cell(x[:, t], h, c)
        out = self.conv_out(h)  # (batch, horizon, H, W)
        return out

# ---------------- Train ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OceanConvLSTM(hidden_channels=8).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# --- Checkpointing ---
checkpoint_path = os.path.join(MODEL_DIR, "convlstm_oceanpulse_checkpoint.pth")
start_epoch = 0
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint.get('epoch', 0)
    print(f"Resuming training from epoch {start_epoch+1}")

def print_progress_bar(iteration, total, prefix='', length=50):
    percent = iteration / total
    filled_len = int(length * percent)
    bar = '#' * filled_len + '-' * (length - filled_len)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent*100:3.0f}%')
    sys.stdout.flush()
    if iteration == total:
        print()  # newline on complete

print("Starting training...")
for epoch in range(start_epoch, EPOCHS):
    epoch_loss = 0
    for i, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb.squeeze(2))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()*xb.size(0)
        print_progress_bar(i+1, len(loader), prefix=f'Epoch {epoch+1} Batch {i+1}')
    print(f"Epoch {epoch+1}/{EPOCHS}, Avg Loss={epoch_loss/len(dataset):.4f}")
    # Save checkpoint after each epoch
    torch.save({
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)

# ---------------- Save Model ----------------
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# ---------------- Predict Next 3 Months ----------------
# Use last INPUT_LEN months as input
last_seq = sst_anom[-INPUT_LEN:].values.astype('float32')[None, :, None, :, :]  # shape (1, seq_len, 1, lat, lon)
last_seq_tensor = torch.tensor(last_seq, dtype=torch.float32).to(device)
model.eval()
with torch.no_grad():
    pred_next3 = model(last_seq_tensor).cpu().numpy()[0]  # shape: (HORIZON, lat, lon)

# ---------------- Save Prediction ----------------
np.save(PRED_PATH, pred_next3)
print(f"Predicted next 3 months anomalies saved to {PRED_PATH}")
