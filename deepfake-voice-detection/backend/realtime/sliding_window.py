import numpy as np
import collections

class SlidingWindowBuffer:
    def __init__(self, window_size_seconds=4.0, sr=16000):
        self.window_size = int(window_size_seconds * sr)
        self.buffer = collections.deque(maxlen=self.window_size)
        
    def add_chunk(self, chunk_bytes):
        """
        Ingests raw bytes (float32 or int16), converts to numpy, 
        and adds to the sliding buffer.
        """
        new_data = np.frombuffer(chunk_bytes, dtype=np.float32)
        
        self.buffer.extend(new_data)
        
    def is_ready(self):
        """Returns True if buffer is full enough to predict."""
        return len(self.buffer) == self.window_size

    def get_buffer(self):
        """Returns the current numpy array for the model."""
        return np.array(self.buffer, dtype=np.float32)