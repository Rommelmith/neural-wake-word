import librosa
import librosa.display
import numpy as np
import torch
import matplotlib.pyplot as plt

audio_path = "../dataset/negative/neg_0001.wav"
label = 0
y, sr = librosa.load(audio_path, sr=16000)
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, n_fft=400, hop_length=160)


mel_db = librosa.power_to_db(mel_spec, ref=np.max)
mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
seq = mel_db.T
X = torch.tensor(seq, dtype=torch.float32)
y = torch.tensor(label, dtype=torch.float32)
X_batch = X.unsqueeze(0)

plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_db, sr=sr, y_axis='mel', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.tight_layout()
plt.show()
print("X:", X.shape, "| X_batch:", X_batch.shape, "| y:", y.shape)
