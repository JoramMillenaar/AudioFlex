import numpy as np
from AudioIO.buffers import CircularBuffer

SemiBlockPairs = list[tuple[np.ndarray, np.ndarray]]


class OverlapAdd:
    def __init__(self, channels: int, chunk_size: int, block_size: int):
        assert not chunk_size % block_size // 2, 'chunk_size must be divisible by half the block_size'
        self.chunk_size = chunk_size
        self.channels = channels
        self.block_size = block_size

        self.window = np.hanning(self.block_size)
        self.window = np.array(channels * [self.window], dtype=np.float32)
        self.bottom_window, self.top_window = np.split(self.window, 2, axis=1)
        self.buffer = CircularBuffer(channels=channels, max_history=chunk_size + block_size)
        self.last_semi_block = np.zeros((channels, block_size // 2), dtype=np.float32)
        self.buffer.push(self.last_semi_block)

        self.input_block_index = 0
        self.rate = 1

    def get_sample_offset(self) -> int:
        if 0 >= self.rate > 2:
            raise ValueError("Rate must be between zero and 2")
        return int((self.block_size // 2) * (self.rate - 1))

    @property
    def current_block(self) -> np.ndarray:
        return self.buffer[self.input_block_index:self.input_block_index + self.block_size]

    @staticmethod
    def get_semi_block_pairs(blocks: list[np.ndarray], last_semi_block: np.ndarray) -> (SemiBlockPairs, np.ndarray):
        pairs = []
        for block in blocks:
            left, right = np.split(block, 2, axis=1)
            pairs.append((left, last_semi_block))
            last_semi_block = right
        return pairs, last_semi_block

    def process(self, audio_chunk, rate: float = 1):
        self.buffer.push(audio_chunk)
        self.rate = rate
        return self.get_output()

    def get_blocks(self) -> list[np.ndarray]:
        blocks = []
        while self.input_block_index + self.block_size <= self.buffer.pushed_samples:
            blocks.append(self.current_block)
            self.input_block_index += self.get_sample_offset() + self.block_size // 2
        return blocks

    def get_output(self):
        blocks = self.get_blocks()
        semi_block_pairs, self.last_semi_block = self.get_semi_block_pairs(blocks, self.last_semi_block)
        semi_block_pairs = [(a * self.bottom_window, b * self.top_window) for a, b in semi_block_pairs]
        output_semi_blocks = [np.sum((a, b), axis=0) for a, b in semi_block_pairs]
        output = np.concatenate(output_semi_blocks, axis=1)
        return output
