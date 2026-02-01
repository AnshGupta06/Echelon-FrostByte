import torch
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import ResNetDeepFake
from utils.features import extract_log_mel_spectrogram

class DeepfakeDetector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔌 Loading AI Model on {self.device}...")
        
        self.model = ResNetDeepFake()
        model_path = os.path.join("models", "weights.pth")
        
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                print("✅ Model loaded successfully!")
            except Exception as e:
                print(f"❌ Error loading weights: {e}")
                self.model = None
        else:
            print("⚠️ WARNING: weights.pth not found.")
            self.model = None

    def predict(self, audio_buffer):
        # Default safe response
        result = {
            "label": "ERROR", 
            "confidence": 0.0, 
            "energy": 0.0, 
            "artifacts": 0.0
        }

        if self.model is None:
            return result

        try:
            # 1. Calculate Energy (Volume)
            # Simple Root Mean Square (RMS) calculation
            energy = float(np.mean(audio_buffer**2)) * 1000 
            
            # 2. AI Inference
            spec_tensor = extract_log_mel_spectrogram(audio_buffer).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = self.model(spec_tensor)
                probs = torch.nn.functional.softmax(logits, dim=1)
                fake_score = probs[0][1].item()
            
            label = "FAKE" if fake_score > 0.5 else "REAL"
            
            return {
                "label": label,
                "confidence": float(fake_score),
                "energy": round(energy, 4),             
                "artifacts": round(fake_score * 10, 2)  
            }
            
        except Exception as e:
            print(f"Inference Error: {e}")
            return result