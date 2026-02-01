import kagglehub
import os
import shutil
import glob

def setup_dataset():
    print("🚀 Initializing KaggleHub Download...")
    
    # 1. Download (Automatic Caching & Unzipping)
    # This returns the local path to where kagglehub stored the files
    try:
        path = kagglehub.dataset_download("awsaf49/asvpoof-2019-dataset")
        print(f"✅ Raw Dataset downloaded to: {path}")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you are logged in! Run 'kagglehub.login()' in python if needed, or setup ~/.kaggle/kaggle.json")
        return

    # 2. Define Destination
    TARGET_DIR = "data"
    REAL_DIR = os.path.join(TARGET_DIR, "real")
    FAKE_DIR = os.path.join(TARGET_DIR, "fake")
    
    # Create clean folders if they don't exist
    os.makedirs(REAL_DIR, exist_ok=True)
    os.makedirs(FAKE_DIR, exist_ok=True)

    print("🧹 Organizing files for training (this takes 10-20 seconds)...")

    # 3. Locate the specific training folder inside the mess
    # The dataset structure is: <path>/LA/LA/ASVspoof2019_LA_train/flac/
    # We search for it dynamically to be safe
    
    # Find the 'flac' folder
    flac_search = glob.glob(os.path.join(path, "**", "ASVspoof2019_LA_train", "flac"), recursive=True)
    
    if not flac_search:
        print("❌ Could not find the 'flac' folder inside the downloaded dataset.")
        print("Structure might have changed. Check the path manually.")
        return
        
    source_flac_dir = flac_search[0]
    
    # Find the protocol file (Labels)
    # It is usually in: .../ASVspoof2019_LA_cm_protocols/
    protocol_search = glob.glob(os.path.join(path, "**", "ASVspoof2019.LA.cm.train.trn.txt"), recursive=True)
    
    if not protocol_search:
        print("❌ Could not find the protocol label file.")
        return

    protocol_path = protocol_search[0]

    # 4. Sort Files based on Labels
    with open(protocol_path, "r") as f:
        lines = f.readlines()
        
    count = 0
    limit = 3000 # Hackathon Speed Limit: Use 3000 files (1500 each approx) for fast training
    
    for line in lines[:limit]: 
        parts = line.strip().split(" ")
        filename = parts[1]
        label = parts[4] # "bonafide" (Real) or "spoof" (Fake)
        
        src_file = os.path.join(source_flac_dir, filename + ".flac")
        
        if not os.path.exists(src_file):
            continue
            
        if label == "bonafide":
            dst_file = os.path.join(REAL_DIR, filename + ".flac")
        else:
            dst_file = os.path.join(FAKE_DIR, filename + ".flac")
            
        # Copy instead of move (keeps the cached kagglehub data intact)
        if not os.path.exists(dst_file):
            shutil.copy2(src_file, dst_file)
            count += 1
            
        if count % 500 == 0:
            print(f"   Processed {count} files...")

    print(f"🎉 Success! {count} audio files ready in 'data/real' and 'data/fake'.")
    print("👉 Now run: python backend/train.py")

if __name__ == "__main__":
    setup_dataset()