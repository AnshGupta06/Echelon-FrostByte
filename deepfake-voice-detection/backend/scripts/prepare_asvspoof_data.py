import os
import shutil

# CHANGE THESE PATHS
ASVSPOOF_ROOT = r"C:\Users\anshg\Downloads\archive (1).zip\LA\LA\ASVspoof2019_LA_train\flac"   # where you downloaded ASVspoof
AUDIO_DIR = os.path.join(
    ASVSPOOF_ROOT,
    "ASVspoof2019_LA_train",
    "flac"
)
PROTOCOL_FILE = os.path.join(
    ASVSPOOF_ROOT,
    "ASVspoof2019_LA_train",
    "protocol",
    "ASVspoof2019.LA.cm.train.trn.txt"
)

# YOUR PROJECT TARGET FOLDERS
HUMAN_DIR = "data/human"
SYNTHETIC_DIR = "data/synthetic"

os.makedirs(HUMAN_DIR, exist_ok=True)
os.makedirs(SYNTHETIC_DIR, exist_ok=True)

with open(PROTOCOL_FILE, "r") as f:
    for line in f:
        parts = line.strip().split()
        file_id = parts[0]
        label = parts[-1]

        src_file = os.path.join(AUDIO_DIR, file_id + ".flac")

        if not os.path.exists(src_file):
            continue

        if label == "bonafide":
            dst = os.path.join(HUMAN_DIR, file_id + ".flac")
        else:
            dst = os.path.join(SYNTHETIC_DIR, file_id + ".flac")

        shutil.copy(src_file, dst)

print("ASVspoof data copied successfully.")
