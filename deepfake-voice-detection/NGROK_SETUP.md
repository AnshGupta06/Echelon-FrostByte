# Ngrok Setup Guide for WebRTC Calling

## Problem: "Failed to fetch" with Ngrok

When using ngrok to connect two different PCs, you may encounter connection issues. Here's how to fix it:

## Solution 1: Update Environment Variables

### For Signaling Server

1. Start ngrok tunnel:
```bash
ngrok http 3001
```

2. Copy the HTTPS URL (e.g., `https://abc123.ngrok.io`)

3. Update `frontend/frostbyte/.env.local`:
```env
NEXT_PUBLIC_SIGNALING_URL=https://abc123.ngrok.io
NEXT_PUBLIC_ML_API_URL=http://localhost:8000
NEXT_PUBLIC_API_URL=https://abc123.ngrok.io
```

4. **Important**: Update signaling server CORS settings in `signaling-server/server.js`:
```javascript
const io = new Server(server, {
  cors: {
    origin: [
      "http://localhost:3000",
      "https://your-frontend-ngrok-url.ngrok.io", // Add your frontend ngrok URL
      "*" // For testing only
    ],
    methods: ["GET", "POST"],
    credentials: true
  }
});
```

### For ML Service (if needed)

1. Start ngrok tunnel for ML service:
```bash
ngrok http 8000
```

2. Update `frontend/frostbyte/.env.local`:
```env
NEXT_PUBLIC_ML_API_URL=https://xyz789.ngrok.io
```

## Solution 2: Fix CORS Issues

The signaling server needs to allow your frontend ngrok URL. Update `signaling-server/server.js`:

```javascript
app.use(cors({
  origin: [
    "http://localhost:3000",
    "https://your-frontend-ngrok.ngrok.io",
    /\.ngrok\.io$/, // Allow all ngrok URLs
    "*" // For testing
  ],
  credentials: true
}));
```

## Solution 3: WebSocket Connection Issues

Ngrok supports WebSockets, but you need to:

1. Use **HTTPS** ngrok URLs (not HTTP)
2. Ensure both signaling server and frontend use the same ngrok URL
3. Check ngrok dashboard for connection status

## Solution 4: Testing Connection

Add this to your frontend to test:

```javascript
// Test signaling server connection
fetch(`${SIGNALING_URL}/api/stats`)
  .then(res => res.json())
  .then(data => console.log('Server stats:', data))
  .catch(err => console.error('Connection failed:', err));
```

## Common Issues

### Issue 1: "Failed to fetch"
- **Cause**: CORS or ngrok URL mismatch
- **Fix**: Update CORS settings and ensure URLs match

### Issue 2: WebSocket connection fails
- **Cause**: Using HTTP instead of HTTPS
- **Fix**: Use HTTPS ngrok URLs

### Issue 3: Connection works but calls fail
- **Cause**: STUN/TURN server issues
- **Fix**: Add TURN server for NAT traversal:
```javascript
const configuration = {
  iceServers: [
    { urls: "stun:stun.l.google.com:19302" },
    { 
      urls: "turn:your-turn-server.com:3478",
      username: "user",
      credential: "pass"
    }
  ],
};
```

## Quick Test

1. Start signaling server: `cd signaling-server && npm start`
2. Start ngrok: `ngrok http 3001`
3. Update `.env.local` with ngrok URL
4. Restart frontend: `npm run dev`
5. Check browser console for connection status

## Alternative: Use Same Network

If ngrok is too complex, use:
- Same WiFi network
- Local IP addresses (e.g., `http://192.168.1.100:3001`)
- Update `.env.local` with local IP

## Debugging

Check browser console for:
- Connection errors
- WebSocket status
- CORS errors
- Network tab for failed requests

The enhanced WebRTC component now shows connection status in the UI!

