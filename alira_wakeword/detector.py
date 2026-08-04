import torch
import torchaudio
import sounddevice as sd
from pathlib import Path

from .model import WakeModel


class WakeWordDetector:
    def __init__(self, threshold=0.5, model_path=None, device="cpu"):
        torch.set_num_threads(1)

        self.threshold = threshold
        self.device = device

        if model_path is None:
            model_path = Path(__file__).parent / "wake_model.pth"

        self.model = WakeModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_mels=80,
            n_fft=400,
            hop_length=160
        )

        self.db_transform = torchaudio.transforms.AmplitudeToDB(
            top_db=20,
            stype="power"
        )

    def predict(self, audio):
        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio, dtype=torch.float32)

        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        spectrogram = self.mel_transform(audio)
        spectrogram = self.db_transform(spectrogram)
        spectrogram = spectrogram.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(spectrogram)
            confidence = output.item()

        return {
            "detected": confidence > self.threshold,
            "confidence": confidence
        }

    def listen_once(self, seconds=1):
        audio = sd.rec(
            frames=16000 * seconds,
            samplerate=16000,
            channels=1,
            dtype="float32"
        )

        sd.wait()

        audio = torch.tensor(audio, dtype=torch.float32).T

        return self.predict(audio)