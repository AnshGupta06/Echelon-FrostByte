from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from api.websockets import websocket_endpoint

app = FastAPI()

# Allow Next.js to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, change to your Next.js URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws/audio") 
async def audio_socket(websocket: WebSocket):
    await websocket_endpoint(websocket)

@app.get("/")
def health_check():
    return {"status": "Deepfake Detector Live"}