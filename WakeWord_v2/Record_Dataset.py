import sounddevice as sd
import soundfile as sf
import os
import time

DST = r"C:\Users\romme\PycharmProjects\neural-wake-word\dataset\negative"
NUM_CLIPS = 150
DURATION = 1  # seconds
SR = 16000

os.makedirs(DST, exist_ok=True)

print("Recording ambient noise clips. Stay quiet or make normal room sounds.")
print("Starting in 3 seconds...")
time.sleep(3)

for i in range(NUM_CLIPS):
    audio = sd.rec(frames=SR * DURATION, samplerate=SR, channels=1, dtype='float32')
    sd.wait()
    filename = os.path.join(DST, f"random_{i:04d}.wav")
    sf.write(filename, audio, SR)
    print(f"Recorded {i + 1}/{NUM_CLIPS}")

print(f"Done. Saved {NUM_CLIPS} clips to {DST}")