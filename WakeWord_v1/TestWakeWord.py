import queue
import time
import numpy as np
import sounddevice as sd
import librosa
import torch
import torch.nn as nn
import os

# -----------------------
# Same settings as training
# -----------------------
SR = 16000
N_MELS = 64
N_FFT = 400
HOP = 160

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)



def audio_to_mel_seq(audio_1d: np.ndarray):
    # audio_1d: float32, shape (samples,)
    mel = librosa.feature.melspectrogram(
        y=audio_1d, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP, power=2.0
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    return mel_db.T.astype(np.float32)  # (T, 64)

# -----------------------
# Model
# -----------------------
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
            nn.Sigmoid()
        )

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        last_h = h_n[-1]              # (B, hidden)
        logits = self.head(last_h).squeeze(1)  # (B,)
        return logits

model = LSTMWakeWord().to(device)
model.load_state_dict(torch.load("wakeword_lstm.pt", map_location=device))
model.eval()

# -----------------------
# Streaming settings
# -----------------------
WINDOW_SEC = 1.0          # model looks at last 1.0s audio
STEP_SEC = 0.10           # run model every 0.10s (10 times/sec)
THRESH = 0.65          # start with 0.70 (0.5 is usually too sensitive)
HITS_NEEDED = 0          # require N hits in a row to trigger
COOLDOWN_SEC = 1.0        # after trigger, ignore for 1s

window_samples = int(WINDOW_SEC * SR)
step_samples = int(STEP_SEC * SR)

audio_buffer = np.zeros(window_samples, dtype=np.float32)
q = queue.Queue()
last_trigger_time = 0.0
hit_streak = 0

def callback(indata, frames, time_info, status):
    if status:
        print(status)
    # indata shape: (frames, channels)
    q.put(indata[:, 0].copy())  # mono

print("Listening... say your wake word.")

with sd.InputStream(samplerate=SR, channels=1, blocksize=step_samples, dtype="float32", callback=callback):
    while True:
        chunk = q.get()  # shape (step_samples,)
        # slide buffer
        audio_buffer = np.roll(audio_buffer, -len(chunk))
        audio_buffer[-len(chunk):] = chunk

        # simple “silence gate” to avoid wasting compute
        rms = float(np.sqrt(np.mean(audio_buffer**2) + 1e-12))
        if rms < 0.005:
            hit_streak = 0
            continue

        # model inference
        seq = audio_to_mel_seq(audio_buffer)              # (T, 64)
        x = torch.from_numpy(seq).unsqueeze(0).to(device) # (1, T, 64)
        lengths = torch.tensor([seq.shape[0]]).to(device)

        with torch.no_grad():
            logit = model(x, lengths)
            prob = torch.sigmoid(logit).item()

        now = time.time()

        # debounce + cooldown
        if now - last_trigger_time < COOLDOWN_SEC:
            continue

        if prob >= THRESH:
            hit_streak += 1
        else:
            hit_streak = 0

        print(f"prob={prob:.3f} rms={rms:.4f} hits={hit_streak}", end="\r")

        if hit_streak >= HITS_NEEDED:
            last_trigger_time = now
            hit_streak = 0
            print("\n✅ WAKE WORD DETECTED!")