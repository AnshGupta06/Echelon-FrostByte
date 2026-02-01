# WebRTC Signaling Server

Node.js signaling server for WebRTC peer-to-peer voice calling with deepfake detection.

## Setup

1. Install dependencies:
```bash
npm install
```

2. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your settings
```

3. Ensure PostgreSQL is running and database is created:
```bash
psql -d deepfake_calls -f database.sql
```

4. Start the server:
```bash
npm start
# or for development:
npm run dev
```

## Features

- JWT authentication
- WebRTC signaling (offer/answer/ICE)
- Call state management
- Real-time deepfake warning broadcasting
- User online/offline status

## API

### REST Endpoints
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login user
- `GET /api/users/search?query=username` - Search users

### Socket.IO Events

**Client → Server:**
- `call-user` - Initiate call to user
- `accept-call` - Accept incoming call
- `reject-call` - Reject incoming call
- `offer` - WebRTC offer
- `answer` - WebRTC answer
- `ice-candidate` - ICE candidate
- `audio-chunk-analysis` - Send ML analysis result
- `end-call` - End active call

**Server → Client:**
- `call-initiated` - Call initiated successfully
- `incoming-call` - Incoming call notification
- `call-accepted` - Call accepted
- `call-rejected` - Call rejected
- `call-ended` - Call ended
- `call-error` - Call error
- `offer` - WebRTC offer
- `answer` - WebRTC answer
- `ice-candidate` - ICE candidate
- `deepfake-warning` - Real-time deepfake warning
- `user-online` - User came online
- `user-offline` - User went offline

