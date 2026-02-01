"use client";
import { useState, useRef } from "react";

export default function LiveCallStreamer() {
  const [mode, setMode] = useState<"live" | "upload">("live");
  const [isRecording, setIsRecording] = useState(false);
  const [status, setStatus] = useState("Idle");
  
  // METRICS
  const [finalVerdict, setFinalVerdict] = useState<{ label: string; confidence: number } | null>(null);
  const [liveData, setLiveData] = useState({ label: "--", confidence: 0, energy: 0, artifacts: 0 });
  
  // REFS
  const socketRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);

  // --- AUDIO UTILS ---
  const downsampleBuffer = (buffer: Float32Array, inputRate: number, outputRate: number) => {
    if (outputRate >= inputRate) return buffer;
    const ratio = inputRate / outputRate;
    const newLength = Math.round(buffer.length / ratio);
    const result = new Float32Array(newLength);
    for (let i = 0; i < newLength; i++) {
      result[i] = buffer[Math.round(i * ratio)];
    }
    return result;
  };

  // --- HANDLERS ---
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setStatus("Uploading & Analyzing...");
    setFinalVerdict(null);
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await fetch("http://localhost:8000/analyze-file", { method: "POST", body: formData });
      const data = await res.json();
      setFinalVerdict({ label: data.label, confidence: data.confidence });
      setStatus("Analysis Complete");
    } catch (err) { console.error(err); setStatus("Error: Upload Failed"); }
  };

  const startStream = async () => {
    setFinalVerdict(null);
    setLiveData({ label: "--", confidence: 0, energy: 0, artifacts: 0 });

    try {
      setStatus("Requesting Mic...");
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      setStatus("Connecting...");
      socketRef.current = new WebSocket("ws://localhost:8000/ws/audio");
      
      socketRef.current.onopen = async () => {
        setStatus("Connected! Initializing Audio...");
        
        // 1. Create Context (Browser decides Sample Rate)
        const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
        audioContextRef.current = new AudioContextClass();
        
        // 2. Resume if suspended
        if (audioContextRef.current.state === "suspended") {
            await audioContextRef.current.resume();
        }

        const inputRate = audioContextRef.current.sampleRate;
        console.log(`🎤 Mic Sample Rate: ${inputRate}Hz (Target: 16000Hz)`);

        sourceRef.current = audioContextRef.current.createMediaStreamSource(stream);
        processorRef.current = audioContextRef.current.createScriptProcessor(4096, 1, 1);
        
        processorRef.current.onaudioprocess = (e) => {
          if (socketRef.current?.readyState === WebSocket.OPEN) {
            const rawData = e.inputBuffer.getChannelData(0);
            
            // ⚡ FIX: DOWNSAMPLE TO 16000HZ
            const finalData = downsampleBuffer(rawData, inputRate, 16000);
            
            socketRef.current.send(finalData.buffer);
          }
        };

        sourceRef.current.connect(processorRef.current);
        processorRef.current.connect(audioContextRef.current.destination);
        setIsRecording(true);
        setStatus("Monitoring Live Audio...");
      };
      
      socketRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.status === "processing") {
            // Update Real-Time Stats
            setLiveData({
                label: data.live_label,
                confidence: data.live_confidence,
                energy: data.energy,
                artifacts: data.artifacts
            });
        }
        else if (data.status === "complete") {
          setFinalVerdict({ label: data.label, confidence: data.confidence });
          setStatus("Call Ended");
          closeResources();
        }
      };
    } catch (err) { console.error(err); setStatus("Error: Mic Permission Denied"); }
  };

  const requestStop = () => {
    setIsRecording(false);
    setStatus("Finalizing...");
    if (socketRef.current?.readyState === WebSocket.OPEN) socketRef.current.send("STOP");
    else closeResources();
  };

  const closeResources = () => {
    socketRef.current?.close();
    sourceRef.current?.disconnect();
    processorRef.current?.disconnect();
    audioContextRef.current?.close();
  };

  // --- UI COMPONENTS ---
  return (
    <div className="p-6 border border-gray-700 rounded-xl bg-gray-900/50 text-white max-w-2xl mx-auto backdrop-blur-sm">
      <div className="flex justify-center gap-4 mb-6">
        <button onClick={() => setMode("live")} className={`px-4 py-2 rounded-lg font-bold ${mode === "live" ? "bg-cyan-600" : "bg-gray-800 text-gray-400"}`}>Live Monitor</button>
        <button onClick={() => setMode("upload")} className={`px-4 py-2 rounded-lg font-bold ${mode === "upload" ? "bg-cyan-600" : "bg-gray-800 text-gray-400"}`}>File Upload</button>
      </div>

      {/* --- LIVE STATS DASHBOARD --- */}
      {mode === "live" && isRecording && (
        <div className="grid grid-cols-2 gap-4 mb-6">
            <div className="p-4 bg-gray-800 rounded-lg text-center border border-gray-600">
                <div className="text-gray-400 text-xs uppercase">Current Fragment</div>
                <div className={`text-2xl font-bold ${liveData.label === "FAKE" ? "text-red-500" : "text-green-500"}`}>
                    {liveData.label}
                </div>
            </div>
            <div className="p-4 bg-gray-800 rounded-lg border border-gray-600">
                <div className="flex justify-between text-xs text-gray-400 mb-1">
                    <span>Artifact Level</span>
                    <span>{liveData.artifacts}</span>
                </div>
                {/* Artifact Bar */}
                <div className="w-full bg-gray-700 h-2 rounded-full overflow-hidden">
                    <div 
                        className="bg-red-500 h-full transition-all duration-300" 
                        style={{ width: `${Math.min(liveData.artifacts * 10, 100)}%` }} 
                    />
                </div>
                
                <div className="flex justify-between text-xs text-gray-400 mt-3 mb-1">
                    <span>Voice Energy</span>
                    <span>{liveData.energy.toFixed(1)}</span>
                </div>
                {/* Energy Bar */}
                <div className="w-full bg-gray-700 h-2 rounded-full overflow-hidden">
                    <div 
                        className="bg-cyan-400 h-full transition-all duration-100" 
                        style={{ width: `${Math.min(liveData.energy * 20, 100)}%` }} 
                    />
                </div>
            </div>
        </div>
      )}

      {/* --- MAIN VERDICT DISPLAY --- */}
      <div className={`h-32 flex flex-col items-center justify-center rounded-lg mb-6 transition-all duration-500
        ${finalVerdict 
            ? (finalVerdict.label === "FAKE" ? "bg-red-500/20 border-2 border-red-500" : "bg-green-500/20 border-2 border-green-500") 
            : "bg-gray-800 border-2 border-gray-700"}`}>
        
        {!finalVerdict ? (
          <div className="flex flex-col items-center">
            <div className="text-gray-400 animate-pulse text-lg font-mono mb-2">{status}</div>
          </div>
        ) : (
          <>
            <div className="text-sm uppercase tracking-widest text-gray-300 mb-1">Final Analysis</div>
            <div className={`text-4xl font-black ${finalVerdict.label === "FAKE" ? "text-red-500" : "text-green-400"}`}>
              {finalVerdict.label}
            </div>
            <div className="text-sm mt-2 text-gray-400">Avg Confidence: {(finalVerdict.confidence * 100).toFixed(1)}%</div>
          </>
        )}
      </div>

      {/* --- CONTROLS --- */}
      {mode === "live" ? (
        !isRecording ? (
          <button onClick={startStream} className="w-full bg-cyan-600 hover:bg-cyan-500 text-white py-4 rounded-lg font-bold text-lg shadow-lg">INITIALIZE LIVE SCAN</button>
        ) : (
          <button onClick={requestStop} className="w-full bg-red-600 hover:bg-red-500 text-white py-4 rounded-lg font-bold text-lg animate-pulse">STOP & GENERATE REPORT</button>
        )
      ) : (
        <div className="relative w-full">
            <input type="file" accept=".wav,.mp3" onChange={handleFileUpload} className="hidden" id="file-upload" />
            <label htmlFor="file-upload" className="w-full flex items-center justify-center bg-gray-700 hover:bg-gray-600 text-white py-4 rounded-lg font-bold text-lg cursor-pointer border-2 border-dashed border-gray-500">
                Select Forensic Audio File
            </label>
        </div>
      )}
    </div>
  );
}