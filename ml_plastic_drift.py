import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.amp import autocast, GradScaler

# =====================================================
# CONFIGURATION
# =====================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 4        # e.g., [u_current, v_current, temp, salinity]
HIDDEN_DIM = 512
NUM_LAYERS = 2
OUTPUT_DIM = 2       # [latitude, longitude]
SEQ_LEN = 24         # timesteps per input
PRED_LEN = 24        # timesteps to predict
LR = 1e-3
EPOCHS = 30
BATCH_SIZE = 32

# =====================================================
# DUMMY DATASET LOADER (replace with real data)
# =====================================================
def load_dataset():
    # Generate synthetic input and target sequences for demo
    X = torch.randn(512, SEQ_LEN, INPUT_DIM)
    y = torch.randn(512, PRED_LEN, OUTPUT_DIM)
    return TensorDataset(X, y)

# =====================================================
# MODEL DEFINITION
# =====================================================
class PlasticDriftLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        self.enc = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.dec = nn.LSTM(output_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x, y=None, teacher_forcing_ratio=0.5):
        batch_size = x.size(0)

        # ----- Encoder -----
        _, (h, c) = self.enc(x)

        # ----- Decoder -----
        dec_input = torch.zeros(batch_size, 1, OUTPUT_DIM, device=x.device)
        outputs = []
        pred_len = y.size(1) if y is not None else PRED_LEN

        for t in range(pred_len):
            out, (h, c) = self.dec(dec_input, (h, c))
            pred = self.fc(out)
            outputs.append(pred)

            if y is not None and torch.rand(1).item() < teacher_forcing_ratio:
                dec_input = y[:, t:t+1, :]
            else:
                dec_input = pred

        return torch.cat(outputs, dim=1)

# =====================================================
# TRAINING FUNCTION
# =====================================================
def train_model():
    print(f"Training on {DEVICE}...")

    dataset = load_dataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = PlasticDriftLSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    scaler = GradScaler()

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0

        for xb, yb in dataloader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            optimizer.zero_grad()

            with autocast("cuda"):
                preds = model(xb, yb, teacher_forcing_ratio=0.5)
                loss = criterion(preds, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch}/{EPOCHS}]  Loss: {avg_loss:.6f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/plastic_drift_lstm.pth")
    print("✅ Model saved to models/plastic_drift_lstm.pth")

# =====================================================
# ENTRY POINT
# =====================================================
if __name__ == "__main__":
    train_model()
