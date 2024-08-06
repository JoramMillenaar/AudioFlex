from typing import Iterable

import numpy as np

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
        self.last_frame = np.zeros((channels, frame_size), dtype=np.float32)
        self.leftover_samples = np.ndarray((channels, 0), dtype=np.float32)

    def get_hop_distance(self, stretch_factor: float):
        return self.hop_size + self.get_stretch_offset(stretch_factor)

    def get_stretch_offset(self, stretch_factor: float) -> int:
        """Returns number of samples the frame needs to shift to stretch the audio according to the stretch_factor"""
        if 0 >= stretch_factor > 2:
            raise ValueError("stretch_factor must be between zero and 2")
        return int(self.hop_size * (stretch_factor - 1))

    def process(self, audio_chunk: np.ndarray, stretch_factor: float = 1):
        hop_distance = self.get_hop_distance(stretch_factor)
        windowed_frames = (frame * self.window for frame in self.to_frames(audio_chunk, hop_distance))
        return self.overlap_add_frames(frames=windowed_frames)

    def get_frame_offset(self, audio: np.ndarray, frame: np.ndarray) -> int:
        return self.hop_size

    def to_frames(self, audio_chunk: np.ndarray, hop_distance: float) -> Iterable[np.ndarray]:
        audio_chunk = np.concatenate((self.leftover_samples, audio_chunk), axis=1)

        current_position = 0
        while current_position + self.frame_size < self.chunk_size:
            yield audio_chunk[:, current_position:current_position + self.frame_size]
            current_position += hop_distance

        self.leftover_samples = audio_chunk[:, current_position:]

    def overlap_add_frames(self, frames: Iterable[np.ndarray]) -> np.ndarray:
        buffer = self.last_frame
        for frame in frames:
            offset = self.get_frame_offset(audio=self.last_frame, frame=frame)
            buffer = overlap_add(audio=buffer, frame=frame, offset=offset)
            self.last_frame = buffer[:, -self.frame_size:]
        return buffer[:, self.hop_size: -self.hop_size]
