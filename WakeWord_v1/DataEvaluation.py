import soundfile as sf
from pathlib import Path


def get_audio_properties_sf(file_path):
    with sf.SoundFile(file_path) as f:
        channels = f.channels
        sample_rate = f.samplerate
        duration_seconds = len(f) / f.samplerate

        print(f"File: {file_path}")
        print(f"Channels: {channels}")
        print(f"Sampling Rate: {sample_rate} Hz")
        print(f"Length: {duration_seconds:.2f} seconds")

# Example usage:
path = Path("../dataset/negative/")
for file in path.iterdir():
    get_audio_properties_sf(file)