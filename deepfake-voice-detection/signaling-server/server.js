const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const cors = require('cors');
const fs = require('fs');
const path = require('path');

const app = express();
const server = http.createServer(app);

// CORS configuration - allow ngrok and localhost
const allowedOrigins = [
  "http://localhost:3000",
  "http://127.0.0.1:3000",
  process.env.FRONTEND_URL,
].filter(Boolean);

// Allow all ngrok URLs
const corsOptions = {
  origin: function (origin, callback) {
    // Allow requests with no origin (mobile apps, Postman, etc.)
    if (!origin) return callback(null, true);
    
    // Allow localhost and ngrok
    if (allowedOrigins.includes(origin) || 
        origin.includes('ngrok.io') || 
        origin.includes('ngrok-free.app') ||
        origin.includes('localhost')) {
      callback(null, true);
    } else {
      callback(null, true); // Allow all for development
    }
  },
  credentials: true,
  methods: ["GET", "POST", "OPTIONS"],
  allowedHeaders: ["Content-Type", "Authorization"]
};

const io = new Server(server, {
  cors: corsOptions,
  transports: ["websocket", "polling"],
  allowEIO3: true
});

app.use(cors(corsOptions));
app.use(express.json());

// In-memory storage (can optionally persist to JSON)
const users = new Map(); // userId -> { userId, username, createdAt }
const activeUsers = new Map(); // userId -> socketId
const activeCalls = new Map(); // callId -> { callerId, calleeId, startTime }
const audioChunkPredictions = []; // Store predictions in memory

// JSON file for persistence (optional)
const DATA_FILE = path.join(__dirname, 'data.json');

// Load data from JSON file if it exists
function loadData() {
  try {
    if (fs.existsSync(DATA_FILE)) {
      const data = JSON.parse(fs.readFileSync(DATA_FILE, 'utf8'));
      if (data.users) {
        data.users.forEach(user => {
          users.set(user.userId, user);
        });
      }
      console.log(`✅ Loaded ${users.size} users from ${DATA_FILE}`);
    }
  } catch (error) {
    console.error('Error loading data:', error);
  }
}

// Save data to JSON file
function saveData() {
  try {
    const data = {
      users: Array.from(users.values()),
      lastUpdated: new Date().toISOString()
    };
    fs.writeFileSync(DATA_FILE, JSON.stringify(data, null, 2));
  } catch (error) {
    console.error('Error saving data:', error);
  }
}

// Initialize: Load existing data
loadData();

// No authentication middleware - allow all connections
io.on('connection', (socket) => {
  console.log(`✅ Client connected: ${socket.id}`);
  
  // Handle user registration/login
  socket.on('register-user', ({ userId, username }) => {
    if (!userId || !username) {
      socket.emit('register-error', { message: 'userId and username required' });
      return;
    }

    // Store or update user
    users.set(userId, {
      userId: parseInt(userId),
      username,
      createdAt: users.get(userId)?.createdAt || new Date().toISOString()
    });
    
    // Register as online
    activeUsers.set(parseInt(userId), socket.id);
    socket.userId = parseInt(userId);
    socket.username = username;
    
    // Save to file
    saveData();
    
    socket.emit('user-registered', {
      userId: parseInt(userId),
      username
    });
    
    // Broadcast user online
    socket.broadcast.emit('user-online', { userId: parseInt(userId), username });
    
    console.log(`👤 User registered: ${username} (${userId})`);
  });

  // Handle call initiation
  socket.on('call-user', ({ targetUserId }) => {
    if (!socket.userId) {
      socket.emit('call-error', { message: 'Please register first' });
      return;
    }

    const targetUserIdInt = parseInt(targetUserId);
    const targetSocketId = activeUsers.get(targetUserIdInt);
    
    if (!targetSocketId) {
      socket.emit('call-error', { message: 'User is not online' });
      return;
    }

    // Check if user exists
    if (!users.has(targetUserIdInt)) {
      socket.emit('call-error', { message: 'User not found' });
      return;
    }

    // Create call record
    const callId = `call_${Date.now()}_${socket.userId}`;
    activeCalls.set(callId, {
      callerId: socket.userId,
      calleeId: targetUserIdInt,
      startTime: new Date()
    });

    // Notify target user
    io.to(targetSocketId).emit('incoming-call', {
      callId,
      callerId: socket.userId,
      callerUsername: socket.username
    });

    socket.emit('call-initiated', { callId, targetUserId: targetUserIdInt });
    console.log(`📞 Call initiated: ${socket.userId} -> ${targetUserIdInt}`);
  });

  // Handle call acceptance
  socket.on('accept-call', ({ callId }) => {
    const call = activeCalls.get(callId);
    if (!call || call.calleeId !== socket.userId) {
      socket.emit('call-error', { message: 'Invalid call' });
      return;
    }

    const callerSocketId = activeUsers.get(call.callerId);
    if (callerSocketId) {
      io.to(callerSocketId).emit('call-accepted', { callId });
    }
    socket.emit('call-accepted', { callId });
    console.log(`✅ Call accepted: ${callId}`);
  });

  // Handle call rejection
  socket.on('reject-call', ({ callId }) => {
    const call = activeCalls.get(callId);
    if (call) {
      const callerSocketId = activeUsers.get(call.callerId);
      if (callerSocketId) {
        io.to(callerSocketId).emit('call-rejected', { callId });
      }
      activeCalls.delete(callId);
    }
  });

  // WebRTC signaling: Offer
  socket.on('offer', ({ callId, offer }) => {
    const call = activeCalls.get(callId);
    if (!call) return;

    const targetSocketId = call.callerId === socket.userId 
      ? activeUsers.get(call.calleeId)
      : activeUsers.get(call.callerId);

    if (targetSocketId) {
      io.to(targetSocketId).emit('offer', { callId, offer });
    }
  });

  // WebRTC signaling: Answer
  socket.on('answer', ({ callId, answer }) => {
    const call = activeCalls.get(callId);
    if (!call) return;

    const targetSocketId = call.callerId === socket.userId 
      ? activeUsers.get(call.calleeId)
      : activeUsers.get(call.callerId);

    if (targetSocketId) {
      io.to(targetSocketId).emit('answer', { callId, answer });
    }
  });

  // WebRTC signaling: ICE candidate
  socket.on('ice-candidate', ({ callId, candidate }) => {
    const call = activeCalls.get(callId);
    if (!call) return;

    const targetSocketId = call.callerId === socket.userId 
      ? activeUsers.get(call.calleeId)
      : activeUsers.get(call.callerId);

    if (targetSocketId) {
      io.to(targetSocketId).emit('ice-candidate', { callId, candidate });
    }
  });

  // Handle audio chunk analysis result
  socket.on('audio-chunk-analysis', ({ callId, userId, chunkId, result }) => {
    // Store prediction in memory
    audioChunkPredictions.push({
      callId,
      userId,
      chunkId,
      isDeepfake: result.is_deepfake,
      confidence: result.confidence,
      modelName: result.model_name || 'ResNetDeepFake',
      timestamp: new Date().toISOString()
    });

    // Broadcast to both users in the call
    const call = activeCalls.get(callId);
    if (call) {
      const callerSocketId = activeUsers.get(call.callerId);
      const calleeSocketId = activeUsers.get(call.calleeId);

      const warningData = {
        callId,
        userId,
        isDeepfake: result.is_deepfake,
        confidence: result.confidence,
        threshold: 0.7
      };

      if (callerSocketId) {
        io.to(callerSocketId).emit('deepfake-warning', warningData);
      }
      if (calleeSocketId) {
        io.to(calleeSocketId).emit('deepfake-warning', warningData);
      }
    }
  });

  // Handle call end
  socket.on('end-call', ({ callId }) => {
    const call = activeCalls.get(callId);
    if (call) {
      const callerSocketId = activeUsers.get(call.callerId);
      const calleeSocketId = activeUsers.get(call.calleeId);

      if (callerSocketId) {
        io.to(callerSocketId).emit('call-ended', { callId });
      }
      if (calleeSocketId) {
        io.to(calleeSocketId).emit('call-ended', { callId });
      }

      activeCalls.delete(callId);
      console.log(`📴 Call ended: ${callId}`);
    }
  });

  // Handle disconnect
  socket.on('disconnect', () => {
    if (socket.userId) {
      console.log(`❌ User disconnected: ${socket.username} (${socket.userId})`);
      activeUsers.delete(socket.userId);
      socket.broadcast.emit('user-offline', { userId: socket.userId });

      // End all active calls for this user
      for (const [callId, call] of activeCalls.entries()) {
        if (call.callerId === socket.userId || call.calleeId === socket.userId) {
          activeCalls.delete(callId);
        }
      }
    } else {
      console.log(`❌ Client disconnected: ${socket.id}`);
    }
  });
});

// REST API endpoints (simplified, no database)
app.post('/api/auth/register', (req, res) => {
  try {
    const { username } = req.body;
    
    if (!username) {
      return res.status(400).json({ error: 'Username required' });
    }

    // Generate a simple user ID (timestamp-based)
    const userId = Date.now();
    
    // Store user
    users.set(userId, {
      userId,
      username,
      createdAt: new Date().toISOString()
    });
    
    // Save to file
    saveData();
    
    res.json({ 
      userId, 
      username,
      message: 'Registration successful (no password required)'
    });
  } catch (error) {
    console.error('Registration error:', error);
    res.status(500).json({ error: 'Registration failed', details: error.message });
  }
});

app.post('/api/auth/login', (req, res) => {
  try {
    const { userId, username } = req.body;
    
    if (!userId && !username) {
      return res.status(400).json({ error: 'userId or username required' });
    }

    let user;
    if (userId) {
      user = users.get(parseInt(userId));
    } else if (username) {
      // Find user by username
      for (const [id, u] of users.entries()) {
        if (u.username === username) {
          user = u;
          break;
        }
      }
    }

    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }

    res.json({ 
      userId: user.userId, 
      username: user.username,
      message: 'Login successful (no password required)'
    });
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ error: 'Login failed', details: error.message });
  }
});

app.get('/api/users/search', (req, res) => {
  try {
    const { query } = req.query;
    if (!query) {
      return res.json([]);
    }

    const results = Array.from(users.values())
      .filter(user => user.username.toLowerCase().includes(query.toLowerCase()))
      .slice(0, 10)
      .map(user => ({ user_id: user.userId, username: user.username }));

    res.json(results);
  } catch (error) {
    console.error('User search error:', error);
    res.status(500).json({ error: 'Search failed' });
  }
});

app.get('/api/users', (req, res) => {
  try {
    const userList = Array.from(users.values()).map(user => ({
      userId: user.userId,
      username: user.username
    }));
    res.json(userList);
  } catch (error) {
    console.error('Error fetching users:', error);
    res.status(500).json({ error: 'Failed to fetch users' });
  }
});

app.get('/api/stats', (req, res) => {
  res.json({
    totalUsers: users.size,
    activeUsers: activeUsers.size,
    activeCalls: activeCalls.size,
    totalPredictions: audioChunkPredictions.length
  });
});

const PORT = process.env.PORT || 3001;
server.listen(PORT, () => {
  console.log(`🚀 Signaling server running on port ${PORT}`);
  console.log(`📝 Using in-memory storage (data saved to ${DATA_FILE})`);
  console.log(`🔓 No authentication required`);
});
