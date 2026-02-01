from fastapi import WebSocket, WebSocketDisconnect
import numpy as np
import sys
import os

# Ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from realtime.sliding_window import SlidingWindowBuffer
from realtime.inference_engine import DeepfakeDetector

# ⚡ CRITICAL FIX: Initialize the model here so this file can use it
print("🔌 Initializing AI for WebSockets...")
detector = DeepfakeDetector()

async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ Client Connected to WebSocket")
    
    # Window size: 2.5s (Matches the frontend/backend logic we discussed)
    buffer = SlidingWindowBuffer(window_size_seconds=2.5) 
    
    session_scores = [] 
    chunks_received = 0 
    
    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message:
                data = message["bytes"]
                buffer.add_chunk(data)
                chunks_received += 1
                
                # Process only when buffer is full
                if buffer.is_ready():
                    audio_input = buffer.get_buffer()
                    
                   # 🔍 RUN INFERENCE
                    result = detector.predict(audio_input)
                    
                    # Store verdict
                    is_fake = 1 if result.get("label") == "FAKE" else 0
                    session_scores.append(is_fake)
                    
                    # Send Live Updates (Safe Mode)
                    await websocket.send_json({
                        "status": "processing",
                        "live_label": result.get("label", "ANALYZING"),
                        "live_confidence": result.get("confidence", 0.0),
                        "energy": result.get("energy", 0.0),       # .get() prevents crash
                        "artifacts": result.get("artifacts", 0.0)  # .get() prevents crash
                    })

            elif "text" in message:
                if message["text"] == "STOP":
                    print(f"🛑 Call Ended. Predictions: {len(session_scores)}")
                    
                    if not session_scores:
                        final_verdict = {
                            "status": "complete", 
                            "label": "INCONCLUSIVE", 
                            "confidence": 0.0
                        }
                    else:
                        avg_score = np.mean(session_scores)
                        label = "FAKE" if avg_score > 0.5 else "REAL"
                        # Confidence logic: How sure are we?
                        conf = avg_score if label == "FAKE" else (1.0 - avg_score)
                        
                        final_verdict = {
                            "status": "complete", 
                            "label": label, 
                            "confidence": round(float(conf), 2)
                        }
                    
                    await websocket.send_json(final_verdict)
                    break 

    except WebSocketDisconnect:
        print("❌ Client disconnected")
    except Exception as e:
        print(f"🔥 CRITICAL ERROR in WebSocket: {e}")
        await websocket.close()