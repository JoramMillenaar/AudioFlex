import numpy as np
from numpy._typing import NDArray

from buffer_handlers import SampleBuffer


class OverlapAdd:
    def __init__(self, channels: int, block_size: int, chunk_size: int, time_percentage: float):
        self.block_size = block_size
        self.buffer = SampleBuffer(channels=channels, max_history=self.block_size * 2)
        self.channels = channels
        self.inv_time_factor = time_percentage
        self.output_samples = int(np.round(chunk_size * (1 / self.inv_time_factor)))
        self.window = np.hanning(self.block_size)
        self.window = np.array(channels * [self.window], dtype=np.float32)

        self.semi_block_samples = self.block_size // 2
        self.semi_block_index = 0
        self.input_block_index = 0
        self.sum_buffer_index = 0
        self.output_index = 0

    @property
    def _buffer_full_enough(self) -> bool:
        return len(self.buffer) >= self.output_samples

    def _increment_indices(self, length):
        self.input_block_index += length
        self.sum_buffer_index += length
        self.output_index += length
        self.semi_block_index += length

    def _take_from_semi_block(self, samples: int):
        cur = self.buffer.get_slice(start=self.input_block_index, end=self.input_block_index + samples)
        add = self.buffer.get_slice(start=self.sum_buffer_index, end=self.sum_buffer_index + samples)

        length = min(cur.shape[1], add.shape[1])  # Due to rounding errors some buffers might be off by one or two
        cur = self._apply_window(cur[:, :length], window_start=self.semi_block_index)
        add = self._apply_window(add[:, :length], window_start=self.semi_block_samples + self.semi_block_index)
        self._increment_indices(length)
        return np.sum((cur, add), axis=0)

    def _take(self, samples: int):
        if self.semi_block_index + samples <= self.semi_block_samples:
            return self._take_from_semi_block(samples)
        elif self.semi_block_index == self.semi_block_samples:
            self._update_block_indices()
            return self._take(samples)
        block_end = self._take(self.semi_block_samples - self.semi_block_index)
        self._update_block_indices()
        new_block = self._take(samples - block_end.shape[1])
        return np.concatenate((block_end, new_block), axis=1)

    def _update_block_indices(self):
        self.sum_buffer_index = self.input_block_index
        self.input_block_index = int(round(self.output_index * self.inv_time_factor))
        self.semi_block_index = 0

    def _apply_window(self, chunk: NDArray, window_start: int):
        window_end = window_start + chunk.shape[1]
        window_slice = self.window[:, window_start:window_end]
        return chunk * window_slice

    def process(self, audio_chunk):
        self.buffer.push(audio_chunk)
        if self._buffer_full_enough:
            return self._take(self.output_samples)
        else:
            return np.zeros((self.channels, self.output_samples), dtype=np.float32)
