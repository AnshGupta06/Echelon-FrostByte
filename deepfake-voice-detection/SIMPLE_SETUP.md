# Simple Setup Guide (No Database Required!)

This version uses **in-memory storage** with optional JSON file persistence. No PostgreSQL needed!

## 🚀 Quick Start

### Step 1: Start Signaling Server
```powershell
cd signaling-server
npm install
npm start
```
Server runs on: http://localhost:3001

**No database setup needed!** Data is stored in memory and optionally saved to `data.json`.

### Step 2: Start ML Service
```powershell
cd backend
.\venv\Scripts\activate.ps1
uvicorn api.app:app --reload --port 8000
```
Server runs on: http://localhost:8000

### Step 3: Start Frontend
```powershell
cd frontend/frostbyte
npm install
npm run dev
```
App runs on: http://localhost:3000

## 🎯 How It Works

### Authentication
- **No passwords required!** Just choose a username
- User IDs are automatically generated (timestamp-based)
- Users are stored in memory and saved to `signaling-server/data.json`

### User Registration
1. Go to http://localhost:3000
2. Click "Create Account"
3. Enter a username (e.g., "alice")
4. You'll get a User ID automatically
5. That's it! No password needed.

### Making Calls
1. Register two users in different browser windows
2. Note the User IDs shown on the dashboard
3. In one window, enter the other user's ID and click "Call"
4. Accept the call in the other window
5. Start talking - deepfake detection runs in real-time!

## 📝 Data Storage

- **In-Memory**: All data stored in server memory (fast, but lost on restart)
- **JSON File**: Optional persistence to `signaling-server/data.json`
- **No Database**: PostgreSQL not required!

## 🔧 Features

✅ No database setup required
✅ No authentication/passwords
✅ Simple username-based registration
✅ Automatic User ID generation
✅ Data persists to JSON file (optional)
✅ All WebRTC calling features work
✅ Real-time deepfake detection

## 📊 API Endpoints

- `POST /api/auth/register` - Register with just username
- `POST /api/auth/login` - Login with User ID or Username
- `GET /api/users/search?query=username` - Search users
- `GET /api/users` - List all users
- `GET /api/stats` - Server statistics

## 🎉 That's It!

No complex setup needed. Just start the three services and you're ready to go!

