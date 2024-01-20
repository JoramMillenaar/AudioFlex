from collections import deque

import numpy as np
from numpy._typing import NDArray


class BufferEnd(Exception):
    pass


class SampleBuffer:
    def __init__(self, channels: int, max_history: int):
        self.channels = channels
        self.max_history = max_history
        self._buffer = deque(maxlen=max_history)
        self.total_samples = 0

    def __len__(self):
        return len(self._buffer)

    def get_slice(self, start: int, end: int) -> NDArray:
        buffer_start = max(start - self.total_samples + len(self._buffer), 0)
        buffer_end = min(end - self.total_samples + len(self._buffer), len(self._buffer))
        return np.array(self._buffer).T[:, buffer_start: buffer_end]  # TODO: improve performance

    def push(self, audio_chunk: NDArray):
        self._buffer.extend(audio_chunk.T)
        self.total_samples += audio_chunk.shape[1]
