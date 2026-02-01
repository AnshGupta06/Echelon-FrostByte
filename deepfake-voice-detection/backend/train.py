import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import glob
import sys

# Ensure we can import from local folders
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.features import extract_log_mel_spectrogram
    from models.model import ResNetDeepFake
except ImportError:
    print("❌ Critical Error: Could not import 'utils' or 'models'.")
    print("Make sure you have created 'backend/utils/features.py' and 'backend/models/model.py'")
    sys.exit(1)

# --- CONFIGURATION ---
DATA_DIR = "data"     # Folder containing 'real' and 'fake' subfolders
BATCH_SIZE = 16       # Reduce to 8 if you run out of Memory
EPOCHS = 5            # 5 Epochs is usually enough f    or a Hackathon demo
LR = 0.001            # Learning Rate
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"⚙️  Training Configuration: Device={DEVICE}, Batch={BATCH_SIZE}, Epochs={EPOCHS}")

# --- DATASET LOADER ---
class VoiceDataset(Dataset):
    def __init__(self, root_dir):
        # Allow both .wav (if you recorded your own) and .flac (ASVspoof dataset)
        self.real_files = glob.glob(os.path.join(root_dir, "real", "*"))
        self.fake_files = glob.glob(os.path.join(root_dir, "fake", "*"))
        
        # Filter to ensure we only pick audio files
        self.real_files = [f for f in self.real_files if f.endswith(('.wav', '.flac', '.mp3'))]
        self.fake_files = [f for f in self.fake_files if f.endswith(('.wav', '.flac', '.mp3'))]

        self.all_files = self.real_files + self.fake_files
        # Label 0 = REAL, Label 1 = FAKE
        self.labels = [0] * len(self.real_files) + [1] * len(self.fake_files)
        
        print(f"📊 Dataset Stats: {len(self.real_files)} Real | {len(self.fake_files)} Fake")

    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self, idx):
        file_path = self.all_files[idx]
        label = self.labels[idx]
        
        try:
            # Extract Feature (Log-Mel Spectrogram)
            # Returns Tensor of shape [1, Freq, Time]
            spec = extract_log_mel_spectrogram(file_path)
            return spec, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"⚠️ Error loading {file_path}: {e}")
            # Return a dummy tensor to prevent crashing (Hackathon fix)
            return torch.zeros((1, 128, 128)), torch.tensor(label, dtype=torch.long)

# --- TRAINING LOOP ---
def train():
    # 1. Prepare Data
    if not os.path.exists(os.path.join(DATA_DIR, "real")) or not os.path.exists(os.path.join(DATA_DIR, "fake")):
        print("❌ Error: 'data/real' or 'data/fake' folders missing!")
        print("👉 Run 'python backend/data/loader.py' first.")
        return

    dataset = VoiceDataset(DATA_DIR)
    if len(dataset) == 0:
        print("❌ Error: No audio files found in 'data/'")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Initialize Model
    print("🧠 Initializing ResNet18 (Customized for Audio)...")
    model = ResNetDeepFake().to(DEVICE)
    
    # 3. Setup Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # 4. Train
    print("🚀 Starting Training...")
    model.train()
    
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + Backward + Optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 10 == 0: # Print every 10 batches
                print(f"   [Epoch {epoch+1}, Batch {i}] Loss: {loss.item():.4f}")

        epoch_acc = 100 * correct / total
        epoch_loss = running_loss / len(dataloader)
        print(f"✅ Epoch {epoch+1}/{EPOCHS} Finished | Accuracy: {epoch_acc:.2f}% | Loss: {epoch_loss:.4f}")
        
        # Save Checkpoint
        save_path = os.path.join("models", "weights.pth")
        torch.save(model.state_dict(), save_path)
        print(f"💾 Model saved to {save_path}")

    print("🎉 Training Complete! You can now restart the backend to use the new AI.")

if __name__ == "__main__":
    train()