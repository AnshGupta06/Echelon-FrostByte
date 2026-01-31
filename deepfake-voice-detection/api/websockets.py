from fastapi import WebSocket, WebSocketDisconnect
from realtime.sliding_window import SlidingWindowBuffer
# from realtime.inference_engine import predict_voice (You will build this later)
import random # Placeholder for now
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

manager = ConnectionManager()

async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    # Initialize a buffer specifically for THIS caller
    buffer = SlidingWindowBuffer(window_size_seconds=4.0)
    
    try:
        while True:
            # 1. Receive Audio Chunk (Bytes)
            data = await websocket.receive_bytes()
            
            # 2. Add to Buffer
            buffer.add_chunk(data)
            
            # 3. Check if we can predict
            if buffer.is_ready():
                audio_input = buffer.get_buffer()
                
                # --- [MID-EVAL SHORTCUT] ---
                # Real model isn't ready? Fake it to test the UI!
                # Replace this line with your actual model inference later.
                fake_score = random.random() # 0.0 to 1.0
                label = "FAKE" if fake_score > 0.5 else "REAL"
                # ---------------------------
                
                response = {
                    "status": "processed",
                    "label": label,
                    "confidence": f"{fake_score:.2f}"
                }
                
                await websocket.send_text(json.dumps(response))
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)