import os
import shutil
import random

SRC = r"C:\Users\romme\PycharmProjects\neural-wake-word\speech_commands_v0.02"
DST = r"C:\Users\romme\PycharmProjects\neural-wake-word\dataset\positive"
SAMPLES_PER_WORD = 50  # adjust if you want more/less

os.makedirs(DST, exist_ok=True)

# skip non-word folders and background noise
SKIP = {"_background_noise_", "."}

count = 0
for word_folder in os.listdir(SRC):
    word_path = os.path.join(SRC, word_folder)
    if not os.path.isdir(word_path) or word_folder in SKIP:
        continue

    wavs = [f for f in os.listdir(word_path) if f.endswith(".wav")]
    chosen = random.sample(wavs, min(SAMPLES_PER_WORD, len(wavs)))

    for wav in chosen:
        new_name = f"{word_folder}_{wav}"
        shutil.copy2(os.path.join(word_path, wav), os.path.join(DST, new_name))
        count += 1

print(f"Done. Copied {count} files to {DST}")

