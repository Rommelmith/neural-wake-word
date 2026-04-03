import sounddevice as sd
import torch
import torch.nn as nn
import torchaudio

device ='cuda' if torch.cuda.is_available() else 'cpu'
print(device)

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
model.to(device)
model.load_state_dict(torch.load('wake_model.pth'))
model.eval()
mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80, n_fft=400, hop_length=160)
db_transform = torchaudio.transforms.AmplitudeToDB(top_db=20, stype="power")
transform = torchaudio.transforms.AmplitudeToDB(top_db=20,stype="power")

while True:
    print("Linting...........")
    audio = sd.rec(frames=16000, samplerate=16000, channels=1, dtype='float32')
    sd.wait()
    audio = torch.tensor(audio, dtype=torch.float32).T
    spectrogram = mel_transform(audio)
    spectrogram = db_transform(spectrogram)
    spectrogram = spectrogram.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(spectrogram)
        confidence = output.item()
        print(f"Confidence: {confidence:.4f}")
        if confidence > 0.5:
            print("ALIRA detected!")