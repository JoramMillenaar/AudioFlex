from typing import Iterable

import numpy as np
from AudioIO.buffers import CircularBuffer, NotEnoughSamples

from audioflex.services import overlap_add


class OverlapAdd:
    def __init__(self, channels: int, chunk_size: int, frame_size: int):
        assert not chunk_size % frame_size // 2, 'chunk_size must be divisible by half the frame_size'
        self.chunk_size = chunk_size
        self.channels = channels
        self.frame_size = frame_size
        self.hop_size = frame_size // 2

        self.window = np.kaiser(self.frame_size, beta=6)
        self.window = np.array(channels * [self.window], dtype=np.float32)
        self.bottom_window, self.top_window = np.split(self.window, 2, axis=1)
        self.buffer = CircularBuffer(channels=channels, max_history=chunk_size + frame_size * 2)
        self.last_frame = np.zeros((channels, frame_size), dtype=np.float32)
        self.buffer.push(self.last_frame)

        self.input_position = 0

    def hop_distance(self, stretch_factor: float):
        return self.hop_size + self.get_stretch_offset(stretch_factor)

    def get_stretch_offset(self, stretch_factor: float) -> int:
        """Returns number of samples the frame needs to shift to stretch the audio according to the stretch_factor"""
        if 0 >= stretch_factor > 2:
            raise ValueError("stretch_factor must be between zero and 2")
        return int(self.hop_size * (stretch_factor - 1))

    def fetch_frame(self, start_position: int):
        return self.buffer[start_position:start_position + self.frame_size]

    def process(self, audio_chunk: np.ndarray, stretch_factor: float = 1):
        self.buffer.push(audio_chunk)
        windowed_frames = (frame * self.window for frame in self.fetch_frames(stretch_factor))
        return self.overlap_add_frames(frames=windowed_frames)

    def get_frame_offset(self, audio: np.ndarray, frame: np.ndarray) -> int:
        return self.hop_size

    def fetch_frames(self, stretch_factor: float):
        while True:
            try:
                yield self.fetch_frame(start_position=self.input_position)
            except NotEnoughSamples:
                return
            self.input_position += self.hop_distance(stretch_factor)

    def overlap_add_frames(self, frames: Iterable[np.ndarray]) -> np.ndarray:
        buffer = self.last_frame
        for frame in frames:
            offset = self.get_frame_offset(audio=self.last_frame, frame=frame)
            buffer = overlap_add(audio=buffer, frame=frame, offset=offset)
            self.last_frame = buffer[:, -self.frame_size:]
        return buffer[:, self.hop_size: -self.hop_size]
