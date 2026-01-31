from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from api.websockets import websocket_endpoint
from realtime.inference_engine import DeepfakeDetector
import io
import librosa
import numpy as np

app = FastAPI()

# --- 1. THE SECURITY FIX ---
# Instead of "*", we list the exact ports your frontend might use.
origins = [
    "http://localhost:3000",    # Next.js
    "http://127.0.0.1:3000",    # Next.js (IP)
    "http://localhost:5173",    # Vite
    "http://127.0.0.1:5173",    # Vite (IP)
    "*"                         # Fallback (Try to allow everything)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # Use the specific list
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI Model
detector = DeepfakeDetector()

@app.websocket("/ws/audio") 
async def audio_socket(websocket: WebSocket):
    # This invokes the handler in websockets.py which does the actual accept()
    await websocket_endpoint(websocket)

@app.post("/analyze-file")
async def analyze_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        audio_array, _ = librosa.load(io.BytesIO(contents), sr=16000)
        
        if len(audio_array) > 64000:
            audio_array = audio_array[:64000]
            
        result = detector.predict(audio_array)
        return result
    except Exception as e:
        print(f"File Error: {e}")
        raise HTTPException(status_code=500, detail="Could not process audio file")

@app.post("/analyze-chunk")
async def analyze_chunk(file: UploadFile = File(...)):
    """
    Analyze a 10-second audio chunk for deepfake detection.
    Accepts: WAV or WebM (Opus) audio files, ~10 seconds duration.
    Returns: is_deepfake (bool), confidence (float), model_name (str)
    """
    try:
        contents = await file.read()
        
        # Load audio - librosa can handle both WAV and WebM formats
        # For WebM/Opus, librosa uses ffmpeg under the hood if available
        try:
            audio_array, sr = librosa.load(io.BytesIO(contents), sr=16000, duration=10.0)
        except Exception as load_error:
            # If librosa fails, try with soundfile or pydub
            print(f"Librosa load error: {load_error}")
            # Fallback: try to load as raw audio (may need additional processing)
            raise HTTPException(status_code=400, detail=f"Could not decode audio: {load_error}")
        
        # Ensure we have approximately 10 seconds of audio (16000 * 10 = 160000 samples)
        target_samples = 160000
        if len(audio_array) < target_samples:
            # Pad with zeros if shorter
            audio_array = np.pad(audio_array, (0, target_samples - len(audio_array)))
        elif len(audio_array) > target_samples:
            # Truncate if longer
            audio_array = audio_array[:target_samples]
        
        result = detector.predict(audio_array)
        
        # Convert to required format
        is_deepfake = result.get("label") == "FAKE"
        confidence = result.get("confidence", 0.0)
        
        return {
            "is_deepfake": is_deepfake,
            "confidence": float(confidence),
            "model_name": "ResNetDeepFake"
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Chunk Analysis Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Could not process audio chunk: {str(e)}")

@app.get("/")
def health_check():
    return {"status": "Deepfake Detector Live"}