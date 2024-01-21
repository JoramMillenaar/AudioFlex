from typing import Iterable, Protocol

import numpy as np


class Buffer(Protocol):
    def get_slice(self, start: int, end: int) -> np.ndarray:
        pass


class CircularBuffer(Buffer):
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


class BufferedInput(Buffer):
    def __init__(self, max_history: int, audio_input: Iterable[np.ndarray], channels: int):
        self.audio_input = iter(audio_input)
        self.circular_buffer = CircularBuffer(channels=channels, max_history=max_history)

    def _fetch_and_store(self, required_samples: int):
        while required_samples > 0:
            try:
                chunk = next(self.audio_input)
                self.circular_buffer.push(chunk)
                required_samples -= chunk.shape[1]
            except StopIteration:
                break  # No more data available from the input

    def get_slice(self, start: int, end: int) -> np.ndarray:
        total_samples = self.circular_buffer.total_samples
        if end > total_samples:
            self._fetch_and_store(end - total_samples)
        return self.circular_buffer.get_slice(start, end)
