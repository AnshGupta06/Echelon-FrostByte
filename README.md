# Echelon – FrostByte
### Real-Time Deepfake Voice Detection & Forensic Security System

![Security](https://img.shields.io/badge/Domain-AI_Security-blue?style=for-the-badge) ![Python](https://img.shields.io/badge/Python-FastAPI-green?style=for-the-badge) ![NextJS](https://img.shields.io/badge/Frontend-Next.js-black?style=for-the-badge)

## 1. Project Overview

**Echelon – FrostByte** is a real-time AI security system designed to detect deepfake voice attacks (AI-generated speech) during live P2P voice calls and in uploaded audio files.

### The Problem
Generative AI voice clones (Voice Conversion and TTS) have become indistinguishable from human speech, enabling new vectors for fraud, social engineering, and impersonation attacks. Traditional security measures rely on metadata or watermarks, which are easily bypassed.

### Our Solution
FrostByte implements an active **Audio Forensic Defense** layer that sits parallel to voice communications. By analyzing high-frequency anomalies and vocoder artifacts in the Mel-spectrogram domain, our system can detect synthetic speech in real-time without needing to identify the speaker or store the audio.

---

## 2. Key Features

- **Real-Time Detection:** Instantly flags AI-generated voices during live WebRTC calls.
- **Chunk-Based Analysis:** Processes audio in 4-10 second sliding windows for continuous monitoring.
- **Visual Audio Forensics:** Uses a customized CNN (ResNet18) to analyze Mel-spectrogram spectral patterns.
- **Low-Latency Inference:** Detection runs parallel to the call, ensuring no lag in voice communication.
- **Robustness:** Capable of detecting artifacts from major vocoders (HiFi-GAN, MelGAN) even under lossy compression.
- **Privacy-First:** No speaker verification or biometric database is used; we look for *generation artifacts*, not identities.

---

## 3. System Architecture

The active defense system operates in parallel to the standard WebRTC P2P voice stream. This ensures that security analysis never introduces latency to the conversation itself.

```ascii
                    [ P2P WebRTC Voice Call (User A <-> User B) ]
                                          |
                      +-------------------+-------------------+
                      |                                       |
             (1) Audio Tap via                           (1) Audio Tap via 
             AudioWorklet/ScriptProcessor                AudioWorklet/ScriptProcessor
                      |                                       |
                      v                                       v
            [ Frontend (Next.js) ]                  [ Frontend (Next.js) ]
                      |                                       |
           (2) WebSocket Stream (Blob)             (2) WebSocket Stream (Blob)
                      |                                       |
                      +-------------------+-------------------+
                                          |
                                          v
                              [ Backend API (FastAPI) ]
                                          |
                               (3) Feature Extraction
                                 (Mel-Spectrogram)
                                          |
                                          v
                             [ AI Inference Engine ]
                             (ResNet18 CNN Model)
                                          |
                                          v
                              [ Risk Score & Alert ]
                 (Sent back to Client via WebSocket in < 200ms)
```

1.  **Audio Tap:** The Next.js frontend captures the audio stream using the Web Audio API without interrupting the WebRTC connection.
2.  **Streaming:** Audio chunks are sent via secure WebSockets to the Python backend.
3.  **Inference:** The backend converts raw audio to Mel-spectrograms and feeds them into the CNN. If the "Fake" probability exceeds the threshold, a visual alert is triggered on the UI.

---

## 4. Machine Learning Approach

Our detection engine is based on **DeeperForensics** principles, focusing on the artifacts left behind by Neural Vocoders.

-   **Input Data:** **Mel-Spectrograms**. We convert 1D raw audio waveforms into 2D time-frequency representations. Deepfakes often struggle to reproduce high-frequency phase coherence, which appears as distinct artifacts in the spectrogram.
-   **Model Architecture:** A **ResNet18 CNN** (Customized).
    -   *Input Layer:* Modified to accept 1-channel grayscale spectrograms.
    -   *4-Layer (Stage) Feature Extractor:* The ResNet body consists of 4 main residual stages that progressively extract features from simple edges (spectral lines) to complex textures (vocoder noise patterns).
    -   *Output:* Binary classification (Real vs. Fake) with confidence scoring.
-   **Generalization:** By training on the structural artifacts of vocoders (the "engine" that generates the sound) rather than specific voices, the system generalizes well to unseen TTS and Voice Conversion attacks.

---

## 5. Real-Time Detection Pipeline

1.  **Capture:** Browser records audio in continuous buffers.
2.  **Chunking:** Audio is sliced into ~4 second segments to ensure sufficient temporal resolution for the spectrogram.
3.  **Preprocessing:** 
    -   Resampling to 16kHz.
    -   Short-Time Fourier Transform (STFT) -> Mel-Scale conversion.
4.  **Inference:** The CNN allows for batch processing of these images.
5.  **Scoring:** A rolling risk score is calculated. If multiple consecutive chunks are flagged as "Fake", the call is marked as High Risk.
6.  **UI Feedback:** The user sees a "SAFE" (Green) or "DETECTED" (Red) status overlay immediately.

---

## 6. Tech Stack

-   **Frontend:** Next.js (React), TailwindCSS, WebRTC, Socket.io-client
-   **Backend:** Python 3.10+, FastAPI, Uvicorn, WebSockets
-   **ML & Audio:** PyTorch, Torchaudio, Librosa, NumPy
-   **Signaling:** Node.js, Express, Socket.io (for P2P connection establishment)
-   **Infrastructure:** LocalHost / Docker ready

---

## 7. How to Run Locally

### Prerequisites
-   Python 3.10+
-   Node.js 18+
-   **FFmpeg** installed and added to system PATH.

### A. Backend (Inference Server)
```bash
cd backend
# Create virtual environment
python -m venv venv
# Activate (Windows)
venv\Scripts\activate
# Install dependencies
pip install -r requirements.txt

# Start the API server
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

### B. Signaling Server (For P2P Calls)
```bash
cd signaling-server
npm install
npm start
# Runs on port 3001
```

### C. Frontend (Client)
```bash
cd frontend/frostbyte
npm install
npm run dev
# Opens at http://localhost:3000
```
*Note: You must allow microphone permissions in the browser. For P2P testing on separate devices, you may need to serve the frontend via HTTPS (using storage tunneling or mkcert) as modern browsers block microphone access on non-secure HTTP (except localhost).*

---

## 8. Demo Instructions

### Test 1: File Upload Forensics
1.  Go to the "Analyze" tab.
2.  Upload a recorded `.wav` or `.mp3` file.
3.  Wait for the system to generate the spectrogram and run the CNN.
4.  Observe the "Deepfake Probability" score.

### Test 2: Live P2P Defense
1.  Open the app in two browser windows (or two devices on the same network).
2.  Enter the same **Room ID** (e.g., "demo123") on both.
3.  Click **"Join Call"** to establish the WebRTC video/audio connection.
4.  While speaking, observe the "Real-Time Analysis" indicator.
5.  (Optional) Play a TTS sample (e.g., from ElevenLabs) into the microphone and watch the detector trigger a "FAKE DETECTED" alert.

---

## 9. Security & Privacy Considerations

-   **No Storage:** Audio chunks are processed in RAM and discarded immediately after inference. We do not save user conversations.
-   **Local Inference:** The model allows for privacy-preserving deployment where data never leaves the premise if deployed on-edge.
-   **Artifact-Based:** By avoiding speaker identification systems, we protect user biometric privacy while still validating the *authenticity* of the media.

---

## 10. Limitations & Future Work

-   **Adversarial Attacks:** Sophisticated attacks adding Gaussian noise may reduce detection accuracy (currently mitigating with augmentation).
-   **Latency:** Network jitter can delay the WebSocket stream relative to the WebRTC call (currently ~200ms delay).
-   **Scale:** Current WebSocket implementation handles 1:1 calls; migration to a scalable TURN/SFU architecture is needed for group calls.
-   **Model:** Future updates will include Vision Transformers (ViT) for improved spectrogram analysis.

---

## 11. Team / Hackathon Note

**Built for [Insert Hackathon Name]** 
This project is a functional prototype demonstrating the feasibility of parallel deepfake detection in live communications. While the UI is polished, the underlying ML model is optimized for the hackathon constraints and is not yet a production-grade security appliance.

---
*Echelon – FrostByte © 2026*
