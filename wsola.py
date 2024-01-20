import numpy as np
from scipy.signal import correlate

from overlap_add import OverlapAdd


class WSOLA(OverlapAdd):
    def __init__(self, channels: int, block_size: int, chunk_size: int, time_percentage: float, search_window: int):
        """
        WSOLA algorithm for time-scale modification of audio signals without affecting pitch.
        :param channels: Amount of channels expected for the audio processor input
        :param block_size: Amount of samples to divide the input in to overlap
        :param chunk_size: Amount of samples per channel expected for the audio processor input
        :param time_percentage: How much to stretch the input audio by
        :param search_window: Size of the search window to find the best overlap position
        """
        super().__init__(channels, block_size, chunk_size, time_percentage)
        self.search_window = search_window
        self.previous_block = None

    def _find_best_overlap_position(self, target_block):
        """
        Find the best overlap position using cross-correlation.
        """
        correlation = correlate(self.previous_block, target_block, mode='full', method='auto')
        mid_point = len(correlation) // 2
        search_start = max(0, mid_point - self.search_window)
        search_end = min(len(correlation), mid_point + self.search_window)
        best_offset = np.argmax(correlation[search_start:search_end]) - self.search_window
        return best_offset

    def _process_current_block(self, chunk):
        if self.previous_block is not None:
            offset = self._find_best_overlap_position(chunk)
            chunk = np.roll(chunk, offset, axis=1)
        self.previous_block = chunk
        return chunk
