import numpy as np


class SampleBuffer:
    def __init__(self, channels: int, max_history: int):
        self.channels = channels
        self.max_history = max_history
        self._buffer = np.zeros((channels, max_history), dtype=np.float32)
        self.total_samples = 0
        self.current_position = 0

    def __len__(self):
        return min(self.total_samples, self.max_history)

    def get_slice(self, start: int, end: int) -> np.ndarray:
        effective_start = max(start - (self.total_samples - self.max_history), 0)
        effective_end = min(end - (self.total_samples - self.max_history), self.max_history)
        start_index = (self.current_position + effective_start) % self.max_history
        end_index = (self.current_position + effective_end) % self.max_history

        if start_index < end_index or effective_end == 0:
            return self._buffer[:, start_index:end_index]
        else:
            return np.hstack((self._buffer[:, start_index:], self._buffer[:, :end_index]))

    def push(self, audio_chunk: np.ndarray):
        chunk_length = audio_chunk.shape[1]
        end_position = (self.current_position + chunk_length) % self.max_history
        if self.current_position < end_position:
            self._buffer[:, self.current_position:end_position] = audio_chunk
        else:
            space_at_end = self.max_history - self.current_position
            self._buffer[:, self.current_position:] = audio_chunk[:, :space_at_end]
            self._buffer[:, :end_position] = audio_chunk[:, space_at_end:]

        self.current_position = end_position
        self.total_samples += chunk_length
