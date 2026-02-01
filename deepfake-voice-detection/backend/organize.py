import os
import shutil
import glob

# --- CONFIG ---
SOURCE_ROOT = "temp_source"  # Where you dragged the files
DEST_ROOT = "data"           # Where train.py looks

def organize():
    print("🧹 Starting Data Cleanup...")
    
    # 1. Setup Destination Folders
    os.makedirs(os.path.join(DEST_ROOT, "real"), exist_ok=True)
    os.makedirs(os.path.join(DEST_ROOT, "fake"), exist_ok=True)

    # 2. Locate Protocol File (The Labels)
    # Looking for: ASVspoof2019.LA.cm.train.trn.txt
    proto_path = os.path.join(SOURCE_ROOT, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.train.trn.txt")
    
    if not os.path.exists(proto_path):
        # Fallback: Search for it recursively
        found = glob.glob(os.path.join(SOURCE_ROOT, "**", "*.cm.train.trn.txt"), recursive=True)
        if not found:
            print(f"❌ Error: Could not find protocol file in {SOURCE_ROOT}")
            print("Did you copy the 'ASVspoof2019_LA_cm_protocols' folder correctly?")
            return
        proto_path = found[0]

    print(f"✅ Found Labels: {proto_path}")

    # 3. Locate Audio Files
    audio_dir = os.path.join(SOURCE_ROOT, "ASVspoof2019_LA_train", "flac")
    if not os.path.exists(audio_dir):
        # Fallback Search
        found = glob.glob(os.path.join(SOURCE_ROOT, "**", "flac"), recursive=True)
        if not found:
            print("❌ Error: Could not find 'flac' folder.")
            return
        audio_dir = found[0]

    print(f"✅ Found Audio: {audio_dir}")

    # 4. Sort Files
    print("🚀 Sorting files... (This takes 1-2 mins)")
    with open(proto_path, "r") as f:
        lines = f.readlines()

    count = 0
    # Hackathon Optimization: Only take first 4000 files to save training time
    # Remove [:4000] if you want to use the WHOLE dataset (takes longer)
    for line in lines[:4000]: 
        parts = line.strip().split(" ")
        filename = parts[1]       # e.g., LA_T_1234567
        label = parts[4]          # "bonafide" or "spoof"

        src = os.path.join(audio_dir, filename + ".flac")
        
        if not os.path.exists(src):
            continue

        if label == "bonafide":
            dst = os.path.join(DEST_ROOT, "real", filename + ".flac")
        else:
            dst = os.path.join(DEST_ROOT, "fake", filename + ".flac")

        shutil.copy2(src, dst)
        count += 1
        
        if count % 500 == 0:
            print(f"   Sorted {count} files...")

    print(f"🎉 Done! {count} files are ready in 'backend/data/'.")
    print("👉 Next Step: Run 'python train.py'")

if __name__ == "__main__":
    organize()