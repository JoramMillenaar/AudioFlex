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
        self.overlap_factor = 2

        self.input_pointer = 0
        self.rate = 1

    def get_stretch_offset(self) -> int:
        """Returns by how many samples the frame needs to be shifted by to stretch the audio according to the rate"""
        if 0 >= self.rate > 2:
            raise ValueError("Rate must be between zero and 2")
        return int(self.hop_size * (self.rate - 1))

    def get_overlap_offset(self) -> int:
        return self.frame_size // self.overlap_factor

    def get_default_offset(self):
        return self.get_overlap_offset() + self.get_stretch_offset()

    def get_frame_from_input(self, input_index: int):
        return self.buffer[input_index:input_index + self.frame_size]

    def process(self, audio_chunk, rate: float = 1):
        self.buffer.push(audio_chunk)
        self.rate = rate
        windowed_frames = (frame * self.window for frame in self.fetch_frames())
        return self.overlap_add_frames(frames=windowed_frames)

    def get_frame_offset(self, frame) -> int:
        return self.frame_size // self.overlap_factor

    def fetch_frames(self):
        while True:
            try:
                yield self.get_frame_from_input(self.input_pointer)
            except NotEnoughSamples:
                return
            self.input_pointer += self.get_default_offset()

    def overlap_add_frames(self, frames: Iterable[np.ndarray]) -> np.ndarray:
        for frame in frames:
            offset = self.get_frame_offset(frame)
            buffer = overlap_add(audio=self.last_frame, frame=frame, offset=offset)
            self.last_frame = buffer[:, -self.frame_size:]

        hop_size = self.frame_size // 2
        return buffer[:, hop_size:-hop_size]
