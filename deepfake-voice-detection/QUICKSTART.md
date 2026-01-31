# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Database Setup

### Step 5: Test the System

1. Open `http://localhost:3000` in two different browsers (or incognito windows)
2. Register two users:
   - User 1: username "alice", password "test123"
   - User 2: username "bob", password "test123"
3. Note the user IDs shown on the dashboard
4. In User 1's window, enter User 2's ID and click "Call"
5. User 2 will see an incoming call - click "Accept"
6. Speak into the microphone - you'll see real-time deepfake detection results!

## 🔧 Troubleshooting

### "Connection refused" errors
- Ensure all three servers are running (signaling: 3001, ML: 8000, frontend: 3000)
- Check firewall settings

### "Model not found" errors
- Ensure `backend/models/weights.pth` exists
- If not, you may need to train the model first (see backend/README.md)

### WebRTC not connecting
- Check browser console for errors
- Ensure microphone permissions are granted
- Try different browsers (Chrome/Firefox recommended)

### Audio analysis failing
- Install FFmpeg: `brew install ffmpeg` (Mac) or `apt-get install ffmpeg` (Linux)
- Check ML service logs for errors

## 📝 Notes

- **Password Security**: Current implementation uses plaintext passwords for demo. Use bcrypt in production.
- **HTTPS Required**: WebRTC requires HTTPS in production (or localhost for development).
- **Model Weights**: The system expects a trained model. If weights.pth doesn't exist, the system will still work but return default values.

## 🎯 Next Steps

1. Train your own model (see `backend/train.py`)
2. Configure production environment variables
3. Set up SSL certificates for HTTPS
4. Deploy to production servers

