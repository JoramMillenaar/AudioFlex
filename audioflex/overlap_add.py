from typing import Iterable

import numpy as np
from AudioIO.buffers import CircularBuffer, NotEnoughSamples

SemiBlockPairs = list[tuple[np.ndarray, np.ndarray]]


class OverlapAdd:
    def __init__(self, channels: int, chunk_size: int, block_size: int):
        assert not chunk_size % block_size // 2, 'chunk_size must be divisible by half the block_size'
        self.chunk_size = chunk_size
        self.channels = channels
        self.block_size = block_size

        self.window = np.kaiser(self.block_size, beta=6)
        self.window = np.array(channels * [self.window], dtype=np.float32)
        self.bottom_window, self.top_window = np.split(self.window, 2, axis=1)
        self.buffer = CircularBuffer(channels=channels, max_history=chunk_size + block_size * 2)
        self.previous_block = np.zeros((channels, block_size), dtype=np.float32)
        self.buffer.push(self.previous_block)
        self.overlap_factor = 2

        self.input_pointer = 0
        self.rate = 1

    def get_stretch_offset(self) -> int:
        """Returns by how many samples the block needs to be shifted by to stretch the audio according to the rate"""
        if 0 >= self.rate > 2:
            raise ValueError("Rate must be between zero and 2")
        return int((self.block_size // 2) * (self.rate - 1))

    def get_overlap_offset(self) -> int:
        return self.block_size // self.overlap_factor

    def get_default_offset(self):
        return self.get_overlap_offset() + self.get_stretch_offset()

    def get_block_from_input(self, input_index: int):
        return self.buffer[input_index:input_index + self.block_size]

    def process(self, audio_chunk, rate: float = 1):
        self.buffer.push(audio_chunk)
        self.rate = rate
        windowed_blocks = (block * self.window for block in self.fetch_blocks())
        return self.overlap_add_blocks(blocks=windowed_blocks)

    def get_block_offset(self, block) -> int:
        return self.block_size // self.overlap_factor

    def fetch_blocks(self):
        while True:
            try:
                yield self.get_block_from_input(self.input_pointer)
            except NotEnoughSamples:
                return
            self.input_pointer += self.get_default_offset()

    def overlap_add_block(self, block: np.ndarray, offset: int) -> np.ndarray:
        s = np.sum((self.previous_block[:, offset:], block[:, :-offset]), axis=0)
        return np.concatenate((self.previous_block[:, :offset], s, block[:, -offset:]), axis=1)

    def overlap_add_blocks(self, blocks: Iterable[np.ndarray]) -> np.ndarray:
        for block in blocks:
            offset = self.get_block_offset(block)
            buffer = self.overlap_add_block(block, offset)
            self.previous_block = buffer[:, -self.block_size:]

        cutoff = self.block_size // 2
        r = buffer[:, cutoff:-cutoff]
        return r
