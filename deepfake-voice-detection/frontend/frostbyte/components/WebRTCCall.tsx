"use client";
import { useState, useRef, useEffect } from "react";
import { io, Socket } from "socket.io-client";
import { authService } from "@/lib/auth";
import { Phone, PhoneOff, AlertTriangle, CheckCircle2, Wifi, WifiOff, Database, Mic, MicOff } from "lucide-react";

const SIGNALING_URL = process.env.NEXT_PUBLIC_SIGNALING_URL || "http://localhost:3001";
const ML_API_URL = process.env.NEXT_PUBLIC_ML_API_URL || "http://localhost:8000";

interface CallState {
  status: "idle" | "calling" | "ringing" | "active" | "ended";
  callId: string | null;
  remoteUserId: string | null;
  remoteUsername: string | null;
}

interface DeepfakeWarning {
  userId: number;
  isDeepfake: boolean;
  confidence: number;
  timestamp: Date;
}

interface AudioChunk {
  id: string;
  timestamp: Date;
  blob: Blob;
  url: string;
  analysis?: {
    isDeepfake: boolean;
    confidence: number;
  };
}

export default function WebRTCCall() {
  const [callState, setCallState] = useState<CallState>({
    status: "idle",
    callId: null,
    remoteUserId: null,
    remoteUsername: null,
  });
  const [targetUserId, setTargetUserId] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<Array<{ user_id: number; username: string }>>([]);
  const [incomingCall, setIncomingCall] = useState<{
    callId: string;
    callerId: string;
    callerUsername: string;
  } | null>(null);
  const [warnings, setWarnings] = useState<DeepfakeWarning[]>([]);
  const [onlineUsers, setOnlineUsers] = useState<Array<{ userId: number; username: string }>>([]);
  const [connectionStatus, setConnectionStatus] = useState<"connecting" | "connected" | "disconnected">("disconnected");
  const [isRecording, setIsRecording] = useState(false);
  const [recordedChunks, setRecordedChunks] = useState<AudioChunk[]>([]);
  const [currentRiskScore, setCurrentRiskScore] = useState<number | null>(null);
  const [isStoring, setIsStoring] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);

  // Refs
  const socketRef = useRef<Socket | null>(null);
  const localStreamRef = useRef<MediaStream | null>(null);
  const remoteStreamRef = useRef<MediaStream | null>(null);
  const peerConnectionRef = useRef<RTCPeerConnection | null>(null);
  const localAudioRef = useRef<HTMLAudioElement | null>(null);
  const remoteAudioRef = useRef<HTMLAudioElement | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunkIndexRef = useRef(0);
  const audioChunksRef = useRef<Blob[]>([]);
  const chunkTimerRef = useRef<NodeJS.Timeout | null>(null);
  const audioChunksStorageRef = useRef<AudioChunk[]>([]);

  useEffect(() => {
    const user = authService.getUser();
    if (!user) return;

    // Connect to signaling server with better error handling
    const connectSocket = () => {
      setConnectionStatus("connecting");
      setConnectionError(null);

      try {
        socketRef.current = io(SIGNALING_URL, {
          transports: ["websocket", "polling"],
          reconnection: true,
          reconnectionAttempts: 5,
          reconnectionDelay: 1000,
          timeout: 20000,
        });

        const socket = socketRef.current;

        socket.on("connect", () => {
          console.log("✅ Connected to signaling server");
          setConnectionStatus("connected");
          setConnectionError(null);
          socket.emit("register-user", {
            userId: user.userId,
            username: user.username
          });
        });

        socket.on("disconnect", (reason) => {
          console.log("❌ Disconnected:", reason);
          setConnectionStatus("disconnected");
          if (reason === "io server disconnect") {
            // Server disconnected, try to reconnect
            socket.connect();
          }
        });

        socket.on("connect_error", (error) => {
          console.error("Connection error:", error);
          setConnectionStatus("disconnected");
          setConnectionError(`Failed to connect: ${error.message}`);

          // If using ngrok or external URL, provide helpful message
          if (SIGNALING_URL.includes("ngrok") || SIGNALING_URL.includes("http://") && !SIGNALING_URL.includes("localhost")) {
            setConnectionError("Cannot connect to server. Check if ngrok tunnel is active and URL is correct.");
          }
        });

        socket.on("user-registered", () => {
          console.log("✅ User registered with server");
        });

        socket.on("register-error", ({ message }) => {
          console.error("Registration error:", message);
          setConnectionError(message);
        });

        socket.on("call-initiated", ({ callId }) => {
          setCallState((prev) => ({ ...prev, status: "ringing", callId }));
        });

        socket.on("incoming-call", ({ callId, callerId, callerUsername }) => {
          setIncomingCall({ callId, callerId, callerUsername });
          setCallState((prev) => ({
            ...prev,
            status: "ringing",
            callId,
            remoteUserId: callerId,
            remoteUsername: callerUsername,
          }));
        });

        socket.on("call-accepted", async ({ callId }) => {
          if (callState.status === "ringing") {
            await startCall(callId);
          }
        });

        socket.on("call-rejected", () => {
          setCallState({ status: "idle", callId: null, remoteUserId: null, remoteUsername: null });
          setIncomingCall(null);
          endCall();
        });

        socket.on("call-ended", () => {
          endCall();
          setCallState({ status: "idle", callId: null, remoteUserId: null, remoteUsername: null });
        });

        socket.on("call-error", ({ message }) => {
          alert(`Call error: ${message}`);
          setCallState({ status: "idle", callId: null, remoteUserId: null, remoteUsername: null });
        });

        socket.on("offer", async ({ callId, offer }) => {
          if (!peerConnectionRef.current) {
            await createPeerConnection(callId);
          }
          await peerConnectionRef.current!.setRemoteDescription(new RTCSessionDescription(offer));
          const answer = await peerConnectionRef.current!.createAnswer();
          await peerConnectionRef.current!.setLocalDescription(answer);
          socket.emit("answer", { callId, answer });
        });

        socket.on("answer", async ({ answer }) => {
          if (peerConnectionRef.current) {
            await peerConnectionRef.current.setRemoteDescription(new RTCSessionDescription(answer));
          }
        });

        socket.on("ice-candidate", async ({ candidate }) => {
          if (peerConnectionRef.current && candidate) {
            await peerConnectionRef.current.addIceCandidate(new RTCIceCandidate(candidate));
          }
        });

        socket.on("deepfake-warning", ({ userId, isDeepfake, confidence }) => {
          setWarnings((prev) => [
            ...prev,
            { userId, isDeepfake, confidence, timestamp: new Date() },
          ]);
          setCurrentRiskScore(confidence);
        });

        socket.on("user-online", ({ userId, username }) => {
          setOnlineUsers((prev) => {
            if (prev.find((u) => u.userId === userId)) return prev;
            return [...prev, { userId, username }];
          });
        });

        socket.on("user-offline", ({ userId }) => {
          setOnlineUsers((prev) => prev.filter((u) => u.userId !== userId));
        });
      } catch (error: any) {
        console.error("Socket setup error:", error);
        setConnectionError(error.message);
        setConnectionStatus("disconnected");
      }
    };

    connectSocket();

    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, []);

  const createPeerConnection = async (callId: string) => {
    const configuration = {
      iceServers: [
        { urls: "stun:stun.l.google.com:19302" },
        { urls: "stun:stun1.l.google.com:19302" },
      ],
    };

    const pc = new RTCPeerConnection(configuration);
    peerConnectionRef.current = pc;

    // Add local stream tracks
    if (localStreamRef.current) {
      localStreamRef.current.getTracks().forEach((track) => {
        pc.addTrack(track, localStreamRef.current!);
      });
    }

    // Handle remote stream
    pc.ontrack = (event) => {
      if (remoteAudioRef.current) {
        remoteStreamRef.current = event.streams[0];
        remoteAudioRef.current.srcObject = event.streams[0];
      }
    };

    // Handle ICE candidates
    pc.onicecandidate = (event) => {
      if (event.candidate && socketRef.current) {
        socketRef.current.emit("ice-candidate", { callId, candidate: event.candidate });
      }
    };

    return pc;
  };

  const startCall = async (callId: string) => {
    try {
      // Get user media
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      localStreamRef.current = stream;

      if (localAudioRef.current) {
        localAudioRef.current.srcObject = stream;
      }

      // Create peer connection
      await createPeerConnection(callId);

      // Create offer if we're the caller
      if (callState.status === "calling" || callState.status === "ringing") {
        const offer = await peerConnectionRef.current!.createOffer();
        await peerConnectionRef.current!.setLocalDescription(offer);
        socketRef.current?.emit("offer", { callId, offer });
      }

      // Start recording and chunking
      startAudioRecording(stream, callId);

      setCallState((prev) => ({ ...prev, status: "active" }));
      setIsRecording(true);
    } catch (error) {
      console.error("Error starting call:", error);
      alert("Failed to start call. Please check microphone permissions.");
    }
  };

  const startAudioRecording = (stream: MediaStream, callId: string) => {
    // mimeType check
    let mimeType = "audio/webm;codecs=opus";
    if (!MediaRecorder.isTypeSupported(mimeType)) {
      mimeType = "audio/webm";
    }

    const mediaRecorder = new MediaRecorder(stream, { mimeType });
    mediaRecorderRef.current = mediaRecorder;

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        audioChunksRef.current.push(event.data);
      }
    };

    mediaRecorder.onstop = async () => {
      // 1. Process the accumulated data from this segment
      if (audioChunksRef.current.length > 0) {
        const chunkBlob = new Blob(audioChunksRef.current, { type: mimeType });
        const chunkUrl = URL.createObjectURL(chunkBlob);

        const chunkId = `chunk_${chunkIndexRef.current}_${Date.now()}`;
        const audioChunk: AudioChunk = {
          id: chunkId,
          timestamp: new Date(),
          blob: chunkBlob,
          url: chunkUrl,
        };

        // Update UI immediately
        audioChunksStorageRef.current.push(audioChunk);
        setRecordedChunks([...audioChunksStorageRef.current]);

        // Send to Backend
        await processAudioChunk(callId, chunkIndexRef.current, chunkBlob, audioChunk);

        chunkIndexRef.current++;
        audioChunksRef.current = []; // Clear for next run
      }

      // 2. Restart if call is still active (indicated by ref existing)
      if (mediaRecorderRef.current === mediaRecorder && localStreamRef.current) {
        // Small delay to ensure clean state
        setTimeout(() => {
          if (mediaRecorderRef.current === mediaRecorder && mediaRecorder.state === "inactive") {
            try {
              mediaRecorder.start();
              // Schedule next Stop
              chunkTimerRef.current = setTimeout(() => {
                if (mediaRecorder.state === "recording") {
                  mediaRecorder.stop();
                }
              }, 4000);
            } catch (e) {
              console.error("Failed to restart recorder:", e);
            }
          }
        }, 100);
      }
    };

    // Initial Start
    try {
      mediaRecorder.start();
      chunkTimerRef.current = setTimeout(() => {
        if (mediaRecorder.state === "recording") {
          mediaRecorder.stop();
        }
      }, 4000);
    } catch (e) {
      console.error("Failed to start recorder:", e);
    }
  };

  const processAudioChunk = async (callId: string, chunkIndex: number, audioBlob: Blob, audioChunk: AudioChunk) => {
    if (!audioBlob || audioBlob.size === 0) return;

    setIsStoring(true);
    try {
      const formData = new FormData();
      formData.append("file", audioBlob, `chunk_${chunkIndex}.webm`);
      if (callId) {
        formData.append("call_id", callId);
      }

      // Send to ML service
      const response = await fetch(`${ML_API_URL}/analyze-chunk`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`ML analysis failed: ${errorText}`);
      }

      const result = await response.json();
      const user = authService.getUser();

      // Update chunk with analysis
      audioChunk.analysis = {
        isDeepfake: result.is_deepfake,
        confidence: result.confidence,
      };

      // Update recorded chunks
      setRecordedChunks([...audioChunksStorageRef.current]);

      // Send result to signaling server
      if (socketRef.current && user) {
        socketRef.current.emit("audio-chunk-analysis", {
          callId,
          userId: user.userId,
          chunkId: `chunk_${chunkIndex}`,
          result,
        });
      }

      setCurrentRiskScore(result.confidence);
    } catch (error) {
      console.error("Error processing audio chunk:", error);
    } finally {
      setIsStoring(false);
    }
  };

  const initiateCall = async () => {
    if (!targetUserId) {
      alert("Please enter a user ID");
      return;
    }

    setCallState((prev) => ({
      ...prev,
      status: "calling",
      remoteUserId: targetUserId,
    }));

    socketRef.current?.emit("call-user", { targetUserId });
  };

  const acceptCall = async () => {
    if (!incomingCall) return;

    socketRef.current?.emit("accept-call", { callId: incomingCall.callId });
    await startCall(incomingCall.callId);
    setIncomingCall(null);
  };

  const rejectCall = () => {
    if (incomingCall) {
      socketRef.current?.emit("reject-call", { callId: incomingCall.callId });
    }
    setIncomingCall(null);
    setCallState({ status: "idle", callId: null, remoteUserId: null, remoteUsername: null });
  };

  const endCall = () => {
    // Stop recording
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }

    // Stop tracks
    localStreamRef.current?.getTracks().forEach((track) => track.stop());
    remoteStreamRef.current?.getTracks().forEach((track) => track.stop());

    // Close peer connection
    if (peerConnectionRef.current) {
      peerConnectionRef.current.close();
      peerConnectionRef.current = null;
    }

    // Clear refs
    localStreamRef.current = null;
    remoteStreamRef.current = null;
    mediaRecorderRef.current = null;
    audioChunksRef.current = [];
    chunkIndexRef.current = 0;
    setIsRecording(false);

    if (chunkTimerRef.current) {
      clearTimeout(chunkTimerRef.current);
    }

    // Notify server
    if (callState.callId) {
      socketRef.current?.emit("end-call", { callId: callState.callId });
    }

    setWarnings([]);
    setCurrentRiskScore(null);
  };

  const searchUsers = async (query: string) => {
    try {
      const response = await fetch(
        `${SIGNALING_URL}/api/users/search?query=${encodeURIComponent(query)}`
      );
      if (response.ok) {
        const results = await response.json();
        setSearchResults(results);
      }
    } catch (error) {
      console.error("User search error:", error);
    }
  };

  const user = authService.getUser();
  const DEEPFAKE_THRESHOLD = 0.7;
  const recentWarnings = warnings.filter(
    (w) => Date.now() - w.timestamp.getTime() < 30000
  );
  const hasHighRiskWarning = recentWarnings.some(
    (w) => w.isDeepfake && w.confidence >= DEEPFAKE_THRESHOLD
  );

  // Calculate average risk score
  const avgRiskScore = recentWarnings.length > 0
    ? recentWarnings.reduce((sum, w) => sum + w.confidence, 0) / recentWarnings.length
    : currentRiskScore || 0;

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gray-900/50 backdrop-blur-sm border border-gray-700 rounded-xl">
      {/* Connection Status */}
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          {connectionStatus === "connected" ? (
            <><Wifi className="text-green-500 w-5 h-5" /><span className="text-green-400 text-sm">Connected</span></>
          ) : connectionStatus === "connecting" ? (
            <><Wifi className="text-yellow-500 w-5 h-5 animate-pulse" /><span className="text-yellow-400 text-sm">Connecting...</span></>
          ) : (
            <><WifiOff className="text-red-500 w-5 h-5" /><span className="text-red-400 text-sm">Disconnected</span></>
          )}
        </div>
        {connectionError && (
          <div className="text-red-400 text-xs bg-red-500/20 px-3 py-1 rounded">
            {connectionError}
          </div>
        )}
      </div>

      <div className="mb-6">
        <h2 className="text-2xl font-bold text-white mb-2">Voice Call System</h2>
        <p className="text-gray-400">
          {user && `Logged in as: ${user.username} (ID: ${user.userId})`}
        </p>
      </div>

      {/* Risk Score Display */}
      {callState.status === "active" && (
        <div className="mb-6 p-4 bg-gray-800 rounded-lg border border-gray-600">
          <div className="flex items-center justify-between mb-2">
            <div className="text-white font-bold">Current Risk Score</div>
            <div className={`text-2xl font-black ${avgRiskScore >= DEEPFAKE_THRESHOLD ? "text-red-500" : avgRiskScore >= 0.5 ? "text-yellow-500" : "text-green-500"}`}>
              {(avgRiskScore * 100).toFixed(1)}%
            </div>
          </div>
          <div className="w-full bg-gray-700 h-3 rounded-full overflow-hidden">
            <div
              className={`h-full transition-all duration-300 ${avgRiskScore >= DEEPFAKE_THRESHOLD ? "bg-red-500" : avgRiskScore >= 0.5 ? "bg-yellow-500" : "bg-green-500"
                }`}
              style={{ width: `${Math.min(avgRiskScore * 100, 100)}%` }}
            />
          </div>
          <div className="mt-2 text-xs text-gray-400">
            Threshold: {(DEEPFAKE_THRESHOLD * 100).toFixed(0)}% | {avgRiskScore >= DEEPFAKE_THRESHOLD ? "⚠️ HIGH RISK" : avgRiskScore >= 0.5 ? "⚠️ MODERATE RISK" : "✅ LOW RISK"}
          </div>
        </div>
      )}

      {/* Deepfake Warning Banner */}
      {hasHighRiskWarning && (
        <div className="mb-6 p-4 bg-red-500/20 border-2 border-red-500 rounded-lg flex items-center gap-3 animate-pulse">
          <AlertTriangle className="text-red-500 w-6 h-6" />
          <div>
            <div className="text-red-400 font-bold">⚠️ HIGH RISK: Potential Deepfake Detected</div>
            <div className="text-red-300 text-sm">
              Recent audio chunks show deepfake probability above {DEEPFAKE_THRESHOLD * 100}%
            </div>
          </div>
        </div>
      )}

      {/* Storage Status */}
      {callState.status === "active" && (
        <div className="mb-4 p-3 bg-gray-800 rounded-lg flex items-center justify-between">
          <div className="flex items-center gap-2">
            {isRecording ? (
              <><Mic className="text-green-500 w-5 h-5 animate-pulse" /><span className="text-green-400 text-sm">Recording Active</span></>
            ) : (
              <><MicOff className="text-gray-500 w-5 h-5" /><span className="text-gray-400 text-sm">Not Recording</span></>
            )}
          </div>
          <div className="flex items-center gap-2">
            {isStoring ? (
              <><Database className="text-blue-500 w-5 h-5 animate-spin" /><span className="text-blue-400 text-sm">Storing...</span></>
            ) : (
              <><Database className="text-green-500 w-5 h-5" /><span className="text-green-400 text-sm">Stored</span></>
            )}
          </div>
          <div className="text-gray-400 text-sm">
            Chunks: {recordedChunks.length} | Total: {recordedChunks.reduce((sum, c) => sum + (c.blob.size / 1024), 0).toFixed(1)} KB
          </div>
        </div>
      )}

      {/* Call Interface */}
      {callState.status === "idle" && (
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Search Users or Enter User ID
            </label>
            <div className="flex gap-2 mb-2">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => {
                  setSearchQuery(e.target.value);
                  if (e.target.value.length > 0) {
                    searchUsers(e.target.value);
                  } else {
                    setSearchResults([]);
                  }
                }}
                placeholder="Search by username..."
                className="flex-1 px-4 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white"
              />
            </div>

            {searchResults.length > 0 && (
              <div className="mb-4 p-3 bg-gray-800 rounded-lg border border-gray-600 max-h-48 overflow-y-auto">
                {searchResults.map((user) => (
                  <button
                    key={user.user_id}
                    onClick={() => {
                      setTargetUserId(user.user_id.toString());
                      setSearchQuery("");
                      setSearchResults([]);
                    }}
                    className="w-full text-left px-3 py-2 hover:bg-gray-700 rounded text-white mb-1 last:mb-0"
                  >
                    {user.username} (ID: {user.user_id})
                  </button>
                ))}
              </div>
            )}

            <div className="flex gap-2">
              <input
                type="text"
                value={targetUserId}
                onChange={(e) => setTargetUserId(e.target.value)}
                placeholder="Or enter User ID directly"
                className="flex-1 px-4 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white"
              />
              <button
                onClick={initiateCall}
                disabled={!targetUserId || connectionStatus !== "connected"}
                className="px-6 py-2 bg-cyan-600 hover:bg-cyan-500 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-bold rounded-lg flex items-center gap-2"
              >
                <Phone className="w-5 h-5" />
                Call
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Incoming Call */}
      {incomingCall && callState.status === "ringing" && (
        <div className="p-6 bg-blue-500/20 border border-blue-500 rounded-lg text-center">
          <div className="text-white text-xl font-bold mb-2">Incoming Call</div>
          <div className="text-gray-300 mb-4">
            From: {incomingCall.callerUsername} (ID: {incomingCall.callerId})
          </div>
          <div className="flex gap-4 justify-center">
            <button
              onClick={acceptCall}
              className="px-6 py-3 bg-green-600 hover:bg-green-500 text-white font-bold rounded-lg flex items-center gap-2"
            >
              <Phone className="w-5 h-5" />
              Accept
            </button>
            <button
              onClick={rejectCall}
              className="px-6 py-3 bg-red-600 hover:bg-red-500 text-white font-bold rounded-lg flex items-center gap-2"
            >
              <PhoneOff className="w-5 h-5" />
              Reject
            </button>
          </div>
        </div>
      )}

      {/* Active Call */}
      {callState.status === "active" && (
        <div className="space-y-4">
          <div className="p-4 bg-green-500/20 border border-green-500 rounded-lg text-center">
            <div className="text-green-400 font-bold text-lg mb-2">Call Active</div>
            <div className="text-gray-300">
              Connected to: {callState.remoteUsername || callState.remoteUserId}
            </div>
          </div>

          {/* Audio elements (hidden, for playback) */}
          <audio ref={localAudioRef} autoPlay muted />
          <audio ref={remoteAudioRef} autoPlay />

          {/* Recorded Audio Chunks */}
          {recordedChunks.length > 0 && (
            <div className="mt-4">
              <div className="text-sm font-medium text-gray-300 mb-2">Recorded Audio Chunks ({recordedChunks.length}):</div>
              <div className="grid grid-cols-2 gap-2 max-h-64 overflow-y-auto">
                {recordedChunks.map((chunk, idx) => (
                  <div
                    key={chunk.id}
                    className={`p-3 rounded-lg border ${chunk.analysis?.isDeepfake && chunk.analysis.confidence >= DEEPFAKE_THRESHOLD
                      ? "bg-red-500/20 border-red-500"
                      : chunk.analysis?.isDeepfake
                        ? "bg-yellow-500/20 border-yellow-500"
                        : "bg-gray-800 border-gray-600"
                      }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="text-white text-xs font-mono">
                        Chunk {idx + 1}
                      </div>
                      {chunk.analysis ? (
                        <div className={`text-xs font-bold ${chunk.analysis.isDeepfake && chunk.analysis.confidence >= DEEPFAKE_THRESHOLD
                          ? "text-red-400"
                          : chunk.analysis.isDeepfake
                            ? "text-yellow-400"
                            : "text-green-400"
                          }`}>
                          {(chunk.analysis.confidence * 100).toFixed(0)}%
                        </div>
                      ) : (
                        <div className="text-xs text-gray-400">Analyzing...</div>
                      )}
                    </div>
                    <audio
                      src={chunk.url}
                      controls
                      className="w-full h-8"
                      preload="metadata"
                    />
                    <div className="text-xs text-gray-400 mt-1">
                      {chunk.timestamp.toLocaleTimeString()} | {(chunk.blob.size / 1024).toFixed(1)} KB
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Recent Warnings */}
          {recentWarnings.length > 0 && (
            <div className="mt-4">
              <div className="text-sm font-medium text-gray-300 mb-2">Recent Analysis:</div>
              <div className="space-y-2">
                {recentWarnings.slice(-5).map((warning, idx) => (
                  <div
                    key={idx}
                    className={`p-3 rounded-lg flex items-center gap-2 ${warning.isDeepfake && warning.confidence >= DEEPFAKE_THRESHOLD
                      ? "bg-red-500/20 border border-red-500"
                      : "bg-gray-800 border border-gray-600"
                      }`}
                  >
                    {warning.isDeepfake && warning.confidence >= DEEPFAKE_THRESHOLD ? (
                      <AlertTriangle className="text-red-500 w-5 h-5" />
                    ) : (
                      <CheckCircle2 className="text-green-500 w-5 h-5" />
                    )}
                    <div className="flex-1">
                      <div className="text-white text-sm">
                        {warning.isDeepfake ? "DEEPFAKE" : "REAL"} - Confidence:{" "}
                        {(warning.confidence * 100).toFixed(1)}%
                      </div>
                      <div className="text-gray-400 text-xs">
                        {warning.timestamp.toLocaleTimeString()}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <button
            onClick={() => {
              endCall();
              setCallState({ status: "idle", callId: null, remoteUserId: null, remoteUsername: null });
              setRecordedChunks([]);
              audioChunksStorageRef.current = [];
            }}
            className="w-full px-6 py-3 bg-red-600 hover:bg-red-500 text-white font-bold rounded-lg flex items-center justify-center gap-2"
          >
            <PhoneOff className="w-5 h-5" />
            End Call
          </button>
        </div>
      )}

      {/* Calling Status */}
      {callState.status === "calling" && (
        <div className="p-6 bg-yellow-500/20 border border-yellow-500 rounded-lg text-center">
          <div className="text-yellow-400 font-bold text-lg">Calling...</div>
          <div className="text-gray-300 mt-2">Waiting for user to answer</div>
        </div>
      )}
    </div>
  );
}
