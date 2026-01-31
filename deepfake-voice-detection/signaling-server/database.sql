-- Database schema for deepfake voice calling system

-- Users table
CREATE TABLE IF NOT EXISTS users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    ip_address VARCHAR(45),
    device_fingerprint VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    last_seen TIMESTAMP DEFAULT NOW()
);

-- Calls table
CREATE TABLE IF NOT EXISTS calls (
    call_id SERIAL PRIMARY KEY,
    caller_id INTEGER NOT NULL REFERENCES users(user_id),
    callee_id INTEGER NOT NULL REFERENCES users(user_id),
    status VARCHAR(20) DEFAULT 'initiating', -- initiating, active, ended, rejected
    started_at TIMESTAMP,
    ended_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Audio chunks table
CREATE TABLE IF NOT EXISTS audio_chunks (
    chunk_id SERIAL PRIMARY KEY,
    call_id INTEGER NOT NULL REFERENCES calls(call_id),
    user_id INTEGER NOT NULL REFERENCES users(user_id),
    chunk_index INTEGER NOT NULL,
    audio_file_path VARCHAR(500),
    duration_seconds DECIMAL(5,2) DEFAULT 10.0,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Audio chunk predictions table
CREATE TABLE IF NOT EXISTS audio_chunk_predictions (
    prediction_id SERIAL PRIMARY KEY,
    call_id INTEGER NOT NULL REFERENCES calls(call_id),
    user_id INTEGER NOT NULL REFERENCES users(user_id),
    chunk_id INTEGER REFERENCES audio_chunks(chunk_id),
    is_deepfake BOOLEAN NOT NULL,
    confidence DECIMAL(5,4) NOT NULL, -- 0.0000 to 1.0000
    model_name VARCHAR(100) DEFAULT 'ResNetDeepFake',
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_calls_caller ON calls(caller_id);
CREATE INDEX IF NOT EXISTS idx_calls_callee ON calls(callee_id);
CREATE INDEX IF NOT EXISTS idx_calls_status ON calls(status);
CREATE INDEX IF NOT EXISTS idx_audio_chunks_call ON audio_chunks(call_id);
CREATE INDEX IF NOT EXISTS idx_audio_chunks_user ON audio_chunks(user_id);
CREATE INDEX IF NOT EXISTS idx_predictions_call ON audio_chunk_predictions(call_id);
CREATE INDEX IF NOT EXISTS idx_predictions_user ON audio_chunk_predictions(user_id);
CREATE INDEX IF NOT EXISTS idx_predictions_chunk ON audio_chunk_predictions(chunk_id);

