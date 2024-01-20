import numpy as np
from numpy._typing import NDArray

from audioflex.buffer_handlers import SampleBuffer


class OverlapAdd:
    def __init__(self, channels: int, block_size: int, time_percentage: float):
        """
        Algorithm that exposes the process method to stretch multiple-channel audio by factor of 'time_percentage'
         1000 samples as input with a time_percentage of 0.8 would result in 800 outputted samples
        :param channels: Amount of channels expected for the audio processor input
        :param block_size: Amount of samples to divide the input in to overlap (typically between 64 and 1024)
        :param time_percentage: How much to stretch the input audio by


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
        self.block_size = block_size
        self.buffer = SampleBuffer(channels=channels, max_history=self.block_size * 2)
        self.channels = channels
        self.inv_time_factor = time_percentage
        self.window = np.hanning(self.block_size)
        self.window = np.array(channels * [self.window], dtype=np.float32)

        self.semi_block_samples = self.block_size // 2
        self.semi_block_index = 0
        self.input_block_index = 0
        self.sum_buffer_index = 0
        self.output_index = 0
        self.chunk_size = 0

    @property
    def _buffer_full_enough(self) -> bool:
        return len(self.buffer) >= self.output_samples

    @property
    def output_samples(self):
        return int(np.round(self.chunk_size * (1 / self.inv_time_factor)))

    def _increment_indices(self, length: int):
        self.input_block_index += length
        self.sum_buffer_index += length
        self.output_index += length
        self.semi_block_index += length

    def _take_from_semi_block(self, samples: int) -> NDArray:
        cur = self.buffer.get_slice(start=self.input_block_index, end=self.input_block_index + samples)
        add = self.buffer.get_slice(start=self.sum_buffer_index, end=self.sum_buffer_index + samples)

        length = min(cur.shape[1], add.shape[1])  # Due to rounding one of the buffers might be off by one sample
        cur = self._apply_window(cur[:, :length], window_start=self.semi_block_index)
        add = self._apply_window(add[:, :length], window_start=self.semi_block_samples + self.semi_block_index)
        self._increment_indices(length)
        self._process_current_block(cur)
        return np.sum((cur, add), axis=0)

    def _take(self, samples: int) -> NDArray:
        """
        Either:
            - Simply take from the current semi-block if the semi-block has enough samples left
            - If the block has been exhausted; update the block positions (and buffers) and try again (recurse)
            - If we need more samples than the block holds, first get the blocks remainder, then get the rest (recurse)
        :param samples: Amount of samples that should be outputted
        :return: An audio chunk of length 'samples'
        """
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

    def _apply_window(self, chunk: NDArray, window_start: int) -> NDArray:
        window_end = window_start + chunk.shape[1]
        window_slice = self.window[:, window_start:window_end]
        return chunk * window_slice

    def process(self, audio_chunk: NDArray) -> NDArray:
        """
        Stretch the duration of the given 'audio_chunk' by the factor of the instance's given time_percentage
         without altering the pitch
        :param audio_chunk: Multiple-channel audio data to stretch in duration.
            The shape should equal: (instance's channels, instance's chunk_size)
        :return: Stretched audio data (chunk_size will be altered by a factor of the instance's time_percentage)
        """
        self.chunk_size = audio_chunk.shape[1]
        self.buffer.push(audio_chunk)
        if self._buffer_full_enough:
            return self._take(self.output_samples)
        else:
            return np.zeros((self.channels, self.output_samples), dtype=np.float32)

    def _process_current_block(self, chunk):
        return chunk
