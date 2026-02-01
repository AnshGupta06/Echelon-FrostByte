from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, Form
from fastapi.middleware.cors import CORSMiddleware
from api.websockets import websocket_endpoint
from realtime.inference_engine import DeepfakeDetector
from realtime.call_stats import call_stats
import io
import librosa
import numpy as np
import os
import shutil
import glob
import sys

app = FastAPI()

# --- FFmpeg CONFIGURATION ---
# Check if ffmpeg is available
if not shutil.which("ffmpeg"):
    print("⚠️ FFmpeg not found in PATH. Attempting to locate...")
    try:
        # Search in WinGet directory
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        search_pattern = os.path.join(local_app_data, "Microsoft", "WinGet", "Packages", "**", "ffmpeg.exe")
        found_files = glob.glob(search_pattern, recursive=True)
        
        if found_files:
            ffmpeg_path = os.path.dirname(found_files[0])
            os.environ["PATH"] += os.pathsep + ffmpeg_path
            print(f"✅ Added FFmpeg to PATH: {ffmpeg_path}")
        else:
            print("❌ FFmpeg executable not found in standard locations.")
    except Exception as e:
        print(f"Error locating FFmpeg: {e}")

# --- 1. CORS SECURITY ---
# Allow all origins for Hackathon ease, or specifically configured ones
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "*"  # Allow all for hackathon/demo to prevent CORS headaches
]

# Add production domains from ENV
if os.getenv("FRONTEND_URL"):
    origins.append(os.getenv("FRONTEND_URL"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
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
async def analyze_chunk(
    file: UploadFile = File(...),
    call_id: str = Form(None)
):
    """
    Analyze a 4-second audio chunk for deepfake detection.
    Accepts: WAV or WebM (Opus) audio files.
    Returns: Chunk result + Call Summary (if call_id provided).
    """
    temp_filename = None
    try:
        contents = await file.read()
        
        # Write to temp file to help librosa/ffmpeg detect format
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            tmp.write(contents)
            temp_filename = tmp.name
        
        # Load audio from file path
        try:
            audio_array, sr = librosa.load(temp_filename, sr=16000, duration=4.0)
        except Exception as load_error:
            print(f"Librosa load error: {load_error}")
            raise HTTPException(status_code=400, detail=f"Could not decode audio: {load_error}")
        
        # Ensure we have approximately 4 seconds of audio (16000 * 4 = 64000 samples)
        target_samples = 64000
        if len(audio_array) < target_samples:
            # Pad with zeros if shorter
            audio_array = np.pad(audio_array, (0, target_samples - len(audio_array)))
        elif len(audio_array) > target_samples:
            # Truncate if longer
            audio_array = audio_array[:target_samples]
        
        # 1. Run Exact Same Inference as File Upload
        result = detector.predict(audio_array)
        
        # 2. Extract Labels
        is_deepfake = result.get("label") == "FAKE"
        confidence = result.get("confidence", 0.0)
        
        response_data = {
            "is_deepfake": is_deepfake,
            "confidence": float(confidence),
            "model_name": "ResNetDeepFake"
        }

        # 3. Update Rolling Stats (If call_id exists)
        if call_id:
            summary = call_stats.update_stats(call_id, result.get("label"), confidence)
            response_data["summary"] = summary

        return response_data

    except HTTPException:
        raise
    except Exception as e:
        print(f"Chunk Analysis Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Could not process audio chunk: {str(e)}")
    finally:
        # Cleanup temp file
        if temp_filename and os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except Exception as e:
                print(f"Error removing temp file: {e}")

@app.get("/")
def health_check():
    return {"status": "Deepfake Detector Live"}