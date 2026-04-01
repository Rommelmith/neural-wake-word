import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, random_split, DataLoader
import numpy as np
import os
import soundfile as sf
import librosa
import random
import torch.nn as nn



class Dataset(Dataset):
    def __init__(self, base_path):
        self.data = []

        for label, folder in [(0, "negative"), (1, "positive")]:
            folder_path = os.path.join(base_path, folder)

            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                self.data.append((file_path, label))


    def __len__(self):
        return len(self.data)

    @staticmethod
    def random_crop(waveform, target_length=16000):
        length = waveform.shape[-1]

        if length > target_length:
            start = random.randint(0, length - target_length)
            waveform = waveform[:, start:start + target_length]

        elif length < target_length:
            pad_amount = target_length - length
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

        return waveform

    def __getitem__(self, idx):
        path, label = self.data[idx]
        audio, sample_rate = sf.read(path)
        audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        length = audio.shape[-1]
        random_crop = self.random_crop(audio)
        mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80, n_fft=400, hop_length=160)
        spectogram = mel_transform(random_crop)
        transform = torchaudio.transforms.AmplitudeToDB(top_db=20,stype="power")
        spectogram = transform(spectogram)
        return spectogram, label

count = 0
ds  = Dataset(r"C:\Users\romme\PycharmProjects\neural-wake-word\dataset")
spec, label = ds[count]


######## Plotting the spectogram ################
'''while label != 1:
    spec, label = ds[count]
    count += 1
# print(spec)
# print(spec.shape, label)
import matplotlib.pyplot as plt
spec_np = spec.squeeze(0).detach().cpu().numpy()

plt.figure(figsize=(10, 10))
plt.imshow(spec_np, aspect="auto", origin="lower")
plt.colorbar(label="Intensity")
plt.title(f"Mel Spectrogram (label={label})")
plt.xlabel("Time (s)")
plt.ylabel("Frequency bands (Hz)")

plt.show()
'''
##########################################################

################### MODEL ################################


class WakeModel(nn.Module):
    def __init__(self):
        super(WakeModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(5, 5),stride=1)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(5, 5),stride=1)
        self.conv3 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(5, 5),stride=1)

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        x = torch.randn(1, 1, 80, 101)
        x = self.max_pool(self.relu(self.conv1(x)))
        x = self.max_pool(self.relu(self.conv2(x)))
        x = self.max_pool(self.relu(self.conv3(x)))
        flat_size = x.shape[1] * x.shape[2] * x.shape[3]

        print(flat_size)
        self.fc1 = nn.Linear(540, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x



model = WakeModel()
if os.path.exists("wake_model.pth"):
    model.load_state_dict(torch.load("wake_model.pth"))
    print("Model loaded")
else:
    print("No model found, starting fresh")

total = len(ds)
print(total)
train_size = int(0.8 * total)
val_size   = total - train_size

train_set, val_set = random_split(ds, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):

    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        labels = labels.float().unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_training_loss = total_loss / len(train_loader)

    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            labels = labels.float().unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

        avg_val_loss = total_loss / len(val_loader)

        print(f"Epoch [{epoch + 1:>2}/{num_epochs}]  "
              f"Train Loss: {avg_training_loss:.4f}  |  "
              f"Val Loss: {avg_val_loss:.4f}")


pos_correct = 0
pos_total = 0
neg_correct = 0
neg_total = 0

for images, labels in val_loader:
    labels = labels.float().unsqueeze(1)
    outputs = model(images)
    predictions = (outputs > 0.5).float()

    pos_mask = labels == 1
    neg_mask = labels == 0

    pos_correct += (predictions[pos_mask] == labels[pos_mask]).sum().item()
    pos_total += pos_mask.sum().item()
    neg_correct += (predictions[neg_mask] == labels[neg_mask]).sum().item()
    neg_total += neg_mask.sum().item()

print(f"Positive: {pos_correct}/{pos_total} | Negative: {neg_correct}/{neg_total}")

torch.save(model.state_dict(), "wake_model.pth")