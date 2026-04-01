import time
from pathlib import Path
import numpy as np
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000
CHANNELS = 1

# How long the final saved clip should be
SAVE_DURATION = 1.0

# How long to actually record after the cue
RECORD_DURATION = 2.0

# Small audio captured before the cue so the start of the word isn't missed
PRE_ROLL = 0.25

OUTPUT_DIR = Path("../dataset/positive")   # change if needed
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def countdown(seconds=3):
    for i in range(seconds, 0, -1):
        print(f"{i}...")
        time.sleep(1)


def record_with_cue():
    print("\nGet ready.")
    countdown(3)
    print("START — speak now!")

    total_duration = PRE_ROLL + RECORD_DURATION
    audio = sd.rec(
        int(total_duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32"
    )
    sd.wait()
    print("Recording finished.\n")

    return audio.squeeze()


def trim_centered(audio: np.ndarray, save_duration=SAVE_DURATION):
    target_len = int(save_duration * SAMPLE_RATE)

    if len(audio) < target_len:
        padded = np.zeros(target_len, dtype=np.float32)
        padded[:len(audio)] = audio
        return padded

    energy = np.abs(audio)
    peak_index = int(np.argmax(energy))

    start = max(0, peak_index - target_len // 3)
    end = start + target_len

    if end > len(audio):
        end = len(audio)
        start = end - target_len

    return audio[start:end].astype(np.float32)


def record_sample(filename: str):
    filepath = OUTPUT_DIR / filename
    raw_audio = record_with_cue()
    final_audio = trim_centered(raw_audio, SAVE_DURATION)

    sf.write(filepath, final_audio, SAMPLE_RATE)
    print(f"Saved: {filepath}")
    print(f"Saved duration: {len(final_audio) / SAMPLE_RATE:.2f} sec\n")


def main():
    print("Wake-word recorder")
    print("Press Enter to prepare a recording.")
    print("You will get a 3-second countdown.")
    print("Speak exactly when you see: START — speak now")
    print("Type 'q' to quit.\n")

    count = 54

    while True:
        cmd = input(f"[Sample {count}] Press Enter to continue: ").strip().lower()
        if cmd == "q":
            print("Done.")
            break

        filename = f"alira_{count:04d}.wav"
        record_sample(filename)
        count += 1


if __name__ == "__main__":
    main()