# Real-Time Voice Calling System with Deepfake Detection

A fully functional real-time voice calling system with integrated deepfake detection. Two authenticated users can call each other directly in the browser, and the system analyzes audio in real-time to detect deepfake voices.

## 🏗️ Architecture

### Frontend (Next.js + React)
- **Location**: `frontend/frostbyte/`
- **Tech Stack**: Next.js 16, React 19, TypeScript, TailwindCSS
- **Features**:
  - JWT-based authentication
  - WebRTC peer-to-peer calling
  - Real-time audio recording and 10-second chunking
  - Live deepfake warning UI
  - Socket.IO client for signaling

### Signaling Server (Node.js + Express)
- **Location**: `signaling-server/`
- **Tech Stack**: Node.js, Express, Socket.IO, PostgreSQL
- **Features**:
  - WebRTC signaling (offer/answer/ICE candidates)
  - User authentication and management
  - Call state management
  - Real-time deepfake warning broadcasting

### ML Service (Python + FastAPI)
- **Location**: `backend/`
- **Tech Stack**: FastAPI, PyTorch, librosa
- **Features**:
  - Deepfake voice detection model (ResNetDeepFake)
  - REST API for audio chunk analysis
  - WebSocket support for real-time streaming

## 🚀 Setup Instructions

### Prerequisites
- Node.js 18+ and npm
- Python 3.8+
- FFmpeg (for audio processing - optional, for better audio format support)

**Note**: This version uses **in-memory storage** - no PostgreSQL required! See `SIMPLE_SETUP.md` for the easiest setup.

### 1. Signaling Server Setup

```bash
cd signaling-server
npm install

# No .env file needed! (Optional: create one to customize PORT or FRONTEND_URL)

# Start server
npm start
# Server runs on http://localhost:3001
# Data is stored in memory and saved to data.json (optional)
```

### 3. ML Service Setup

```bash
cd backend

# Create virtual environment (if not exists)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Ensure model weights exist
# Place weights.pth in backend/models/ directory

# Start FastAPI server
uvicorn api.app:app --reload --port 8000
# Server runs on http://localhost:8000
```

### 4. Frontend Setup

```bash
cd frontend/frostbyte
npm install

# Create .env.local file
echo "NEXT_PUBLIC_API_URL=http://localhost:3001" > .env.local
echo "NEXT_PUBLIC_SIGNALING_URL=http://localhost:3001" >> .env.local
echo "NEXT_PUBLIC_ML_API_URL=http://localhost:8000" >> .env.local

# Start development server
npm run dev
# App runs on http://localhost:3000
```

## 📱 Usage

### 1. Register/Login
- Navigate to `http://localhost:3000`
- Register a new account or login
- You'll be redirected to the dashboard

### 2. Make a Call
- Register two users (in different browser windows/tabs)
- Note the User IDs shown on each dashboard
- In one window, enter the other user's ID and click "Call"
- The other user will receive an incoming call notification
- Once accepted, the call starts

### 3. Real-Time Detection
- During the call, audio is recorded in 10-second chunks
- Each chunk is analyzed by the ML model
- If deepfake probability exceeds 70%, a warning is displayed
- Recent analysis results are shown in real-time

### 4. End Call
- Click "End Call" to terminate
- All audio chunks and predictions are stored in the database

## 🔐 Authentication

- **Simplified**: No passwords required!
- Just choose a username to register
- User IDs are automatically generated
- User data stored in memory (with optional JSON file persistence)

## 🎙️ Audio Processing

- **Format**: WebM (Opus codec) from browser, converted to WAV for ML
- **Sample Rate**: 16kHz
- **Chunk Duration**: Exactly 10 seconds
- **Analysis**: Each chunk is sent to ML service for real-time detection

## 🗄️ Data Storage

- **In-Memory**: Fast, no database required
- **JSON File**: Optional persistence to `signaling-server/data.json`
- **Users**: Stored in memory with username and User ID
- **Calls**: Tracked in memory during active sessions
- **Predictions**: Stored in memory (can be exported if needed)

## 🔧 Configuration

### Environment Variables

**Signaling Server** (`signaling-server/.env`):
```
PORT=3001
FRONTEND_URL=http://localhost:3000
JWT_SECRET=your-secret-key
DB_USER=postgres
DB_HOST=localhost
DB_NAME=deepfake_calls
DB_PASSWORD=postgres
DB_PORT=5432
```

**Frontend** (`frontend/frostbyte/.env.local`):
```
NEXT_PUBLIC_API_URL=http://localhost:3001
NEXT_PUBLIC_SIGNALING_URL=http://localhost:3001
NEXT_PUBLIC_ML_API_URL=http://localhost:8000
```

## 🧪 Testing

1. Open two browser windows/tabs
2. Register two different users
3. Note the user IDs
4. In one window, call the other user's ID
5. Accept the call in the second window
6. Speak into the microphone
7. Observe real-time deepfake detection results

## 📊 ML Model

- **Model**: ResNetDeepFake (CNN-based)
- **Input**: Log-Mel Spectrogram (128x128)
- **Output**: Binary classification (REAL/FAKE) with confidence score
- **Features**: MFCC, Mel Spectrogram extraction

## 🚨 Deepfake Warning Threshold

- Default threshold: **70% confidence**
- Warnings appear when `is_deepfake = true` AND `confidence >= 0.7`
- Real-time warnings displayed during active calls
- Historical predictions stored for analysis

## 🔒 Security Notes

- **Production**: Use bcrypt for password hashing (currently plaintext for demo)
- **HTTPS**: Required for WebRTC in production
- **JWT Secret**: Change default secret in production
- **CORS**: Configure allowed origins properly

## 📝 API Endpoints

### Signaling Server
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login
- `GET /api/users/search` - Search users

### ML Service
- `POST /analyze-chunk` - Analyze 10-second audio chunk
- `POST /analyze-file` - Analyze uploaded audio file
- `WebSocket /ws/audio` - Real-time audio streaming

## 🐛 Troubleshooting

1. **WebRTC not connecting**: Check STUN server configuration
2. **Audio not recording**: Verify microphone permissions
3. **ML analysis failing**: Ensure model weights exist and FFmpeg is installed
4. **Database errors**: Verify PostgreSQL connection and schema

## 📄 License

© 2024 Echelon-FrostByte. Real-Time Deepfake Detection System.

