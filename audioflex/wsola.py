import numpy as np
from numpy._typing import NDArray
from scipy.signal import correlate

from audioflex.overlap_add import OverlapAdd


class WSOLA(OverlapAdd):
    def __init__(self, chunk_size: int, block_size: int, channels: int, search_window: int):
        """
        WSOLA algorithm for timescale modification of audio signals without affecting pitch.
        :param chunk_size: Expected length of the inputted audio_chunks
        :param channels: Amount of channels expected for the audio processor input
        :param block_size: Amount of samples to divide the input in to overlap
        :param search_window: Size of the search window to find the best overlap position
        """
        super().__init__(channels, chunk_size, block_size)
        self.search_window = search_window
        self.previous_block = None

    def _find_best_overlap_position(self, target_block: NDArray) -> int:
        """
        Find the best overlap position using cross-correlation.
        """
        correlation = correlate(self.previous_block, target_block, mode='full', method='auto')
        mid_point = correlation.shape[1] // 2
        search_start = max(0, mid_point - self.search_window // 2)
        search_end = min(correlation.shape[1], mid_point + self.search_window // 2)
        best_offset = np.argmax(correlation[:, search_start:search_end]) - self.search_window
        return best_offset

    def get_sample_offset(self):
        offset = super().get_sample_offset()
        if self.previous_block is not None:
            offset += self._find_best_overlap_position(self.current_block)
        self.previous_block = self.current_block
        return offset

