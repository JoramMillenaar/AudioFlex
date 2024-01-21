import numpy as np
from numpy._typing import NDArray

from audioflex.buffer_handlers import Buffer


class OverlapAdd:
    """
    Algorithm that exposes the process method to stretch multiple-channel audio by factor of 'time_percentage'
     1000 samples as input with a time_percentage of 0.8 would result in 800 outputted samples
    :param input_buffer: Buffer to take the input samples of (An interface with 'get_slice' method)
    :param channels: Amount of channels expected for the audio processor input
    :param block_size: Amount of samples to divide the input in to overlap (typically between 64 and 1024)


    Visual Explanation attempt (where time_percentage is 1):
    Every symbol, except for the boundary markers '|', is a sample

    ############   -> Input
    |          |
    BBBBBB     |   -> (BBBBBB) is a block (so are the rows below it)
    |  111222  |   -> (111) & (222) are also represented as 'semi-blocks'
    |     XXXYYY   -> (XXX) is also named 'current block' where (222) would be the 'summing buffer'
    |          |
    ############   -> Output

    The time_percentage only affects where the block is taken from the input.

    This approach iterates over 'semi-blocks', which is half of a block. This semi-block has a consistent
     'current block' and 'summing buffer', which are summed together after windowing to form the output.
    When the output is asking for more samples than the semi-block holds, the current semi-block remainder
     is fetched before updating the block assignments/indices. Then the rest of the samples are fetched recursively.
    """

    def __init__(self, input_buffer: Buffer, block_size: int, channels: int):
        self.block_size = block_size
        self.channels = channels
        self.buffer = input_buffer
        self.window = np.hanning(self.block_size)
        self.window = np.array(channels * [self.window], dtype=np.float32)
        self.inv_time_factor = 1

        self._semi_block_samples = self.block_size // 2
        self._semi_block_index = 0
        self._input_block_index = 0
        self._sum_buffer_index = 0
        self._output_index = 0

    def set_rate(self, rate: float):
        self.inv_time_factor = rate

    def set_time_factor(self, time_factor: float):
        self.inv_time_factor = 1 / time_factor

    def _increment_indices(self, samples: int):
        self._input_block_index += samples
        self._sum_buffer_index += samples
        self._output_index += samples
        self._semi_block_index += samples

    def _take_from_semi_block(self, samples: int) -> NDArray:
        cur = self.buffer.get_slice(start=self._input_block_index, end=self._input_block_index + samples)
        add = self.buffer.get_slice(start=self._sum_buffer_index, end=self._sum_buffer_index + samples)

        length = min(cur.shape[1], add.shape[1])  # Due to rounding one of the buffers might be off by one sample
        cur = self._apply_window(cur[:, :length], window_start=self._semi_block_index)
        add = self._apply_window(add[:, :length], window_start=self._semi_block_samples + self._semi_block_index)
        self._increment_indices(length)
        self._process_current_block(cur)
        return np.sum((cur, add), axis=0)

    def pull(self, samples) -> NDArray:
        """
        Either:
            - Simply take from the current semi-block if the semi-block has enough samples left
            - If the block has been exhausted; update the block positions (and buffers) and try again (recurse)
            - If we need more samples than the block holds, first get the blocks remainder, then get the rest (recurse)
        :param samples: Amount of samples that should be outputted
        :return: An audio chunk of length 'samples'
        """
        if self._semi_block_index + samples <= self._semi_block_samples:
            return self._take_from_semi_block(samples)
        elif self._semi_block_index == self._semi_block_samples:
            self._update_block_indices()
            return self.pull(samples)
        block_end = self.pull(self._semi_block_samples - self._semi_block_index)
        self._update_block_indices()
        new_block = self.pull(samples - block_end.shape[1])
        return np.concatenate((block_end, new_block), axis=1)

    def _update_block_indices(self):
        self._sum_buffer_index = self._input_block_index
        self._input_block_index = int(round(self._output_index * self.inv_time_factor))
        self._semi_block_index = 0

    def _apply_window(self, chunk: NDArray, window_start: int) -> NDArray:
        window_end = window_start + chunk.shape[1]
        window_slice = self.window[:, window_start:window_end]
        return chunk * window_slice

    def _process_current_block(self, chunk):
        return chunk
