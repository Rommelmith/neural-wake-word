import os, glob, random
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = "dataset"
POS_DIR = os.path.join(ROOT, "positive")
NEG_DIR = os.path.join(ROOT, "negative")

SR = 16000
N_MELS = 64
N_FFT = 400
HOP = 160

BATCH_SIZE = 8
EPOCHS = 300
LR = 1e-3
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)



def wav_to_mel_seq(path):
    y, sr = librosa.load(path, sr=SR)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP, power=2.0
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    return mel_db.T.astype(np.float32)  # (T, 64)


class WakeWordDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        seq = wav_to_mel_seq(path)
        x = torch.from_numpy(seq)
        y = torch.tensor(label, dtype=torch.float32)
        length = torch.tensor(x.shape[0], dtype=torch.long)
        return x, y, length

def collate_fn(batch):
    xs, ys, lengths = zip(*batch)
    lengths = torch.stack(lengths)
    max_len = int(lengths.max())
    feat_dim = xs[0].shape[1]

    x_pad = torch.zeros(len(xs), max_len, feat_dim, dtype=torch.float32)
    for i, x in enumerate(xs):
        x_pad[i, :x.shape[0]] = x

    y = torch.stack(ys)  # (B,)
    return x_pad, y, lengths

# Collect files
pos_files = sorted(glob.glob(os.path.join(POS_DIR, "*.wav")))
neg_files = sorted(glob.glob(os.path.join(NEG_DIR, "*.wav")))

print("Pos files:", len(pos_files), "| Neg files:", len(neg_files))

items = []

for p in pos_files:
    items.append((p, 1))

for n in neg_files:
    items.append((n, 0))
random.shuffle(items)

# Split 80/20
split = int(0.8 * len(items))
train_items = items[:split]
val_items = items[split:]

train_ds = WakeWordDataset(train_items)
val_ds = WakeWordDataset(val_items)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)


class LSTMWakeWord(nn.Module):
    def __init__(self, input_dim=64, hidden=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        last_h = h_n[-1]              # (B, hidden)
        logits = self.head(last_h).squeeze(1)  # (B,)
        return logits

model = LSTMWakeWord(input_dim=N_MELS).to(device)
model.load_state_dict(torch.load("wakeword_lstm.pt", map_location=device))
# criterion = nn.BCEWithLogitsLoss()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def acc_from_logits(logits, y):
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    return (preds == y).float().mean().item()


for epoch in range(1, EPOCHS + 1):
    model.train()
    tr_loss, tr_acc, tr_n = 0.0, 0.0, 0

    for x_batch, y_batch, lengths in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        logits = model(x_batch, lengths)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        b = y_batch.size(0)
        tr_loss += loss.item() * b
        tr_acc += acc_from_logits(logits.detach(), y_batch) * b
        tr_n += b

    model.eval()
    va_loss, va_acc, va_n = 0.0, 0.0, 0
    with torch.no_grad():
        for x_batch, y_batch, lengths in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            lengths = lengths.to(device)

            logits = model(x_batch, lengths)
            loss = criterion(logits, y_batch)

            b = y_batch.size(0)
            va_loss += loss.item() * b
            va_acc += acc_from_logits(logits, y_batch) * b
            va_n += b

    print(
        f"Epoch {epoch:02d} | "
        f"train loss {tr_loss/tr_n:.4f} acc {tr_acc/tr_n:.3f} | "
        f"val loss {va_loss/va_n:.4f} acc {va_acc/va_n:.3f}"
    )


SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wakeword_lstm.pt")
torch.save(model.state_dict(), SAVE_PATH)
print("✅ Saved:", SAVE_PATH, "bytes:", os.path.getsize(SAVE_PATH))