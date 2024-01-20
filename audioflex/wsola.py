import numpy as np
from numpy._typing import NDArray
from scipy.signal import correlate

from audioflex.overlap_add import OverlapAdd


class WSOLA(OverlapAdd):
    def __init__(self, channels: int, block_size: int, time_percentage: float, search_window: int):
        """
        WSOLA algorithm for timescale modification of audio signals without affecting pitch.
        :param channels: Amount of channels expected for the audio processor input
        :param block_size: Amount of samples to divide the input in to overlap
        :param time_percentage: How much to stretch the input audio by
        :param search_window: Size of the search window to find the best overlap position
        """
        super().__init__(channels, block_size, time_percentage)
        self.search_window = search_window
        self.previous_block = None

    def _find_best_overlap_position(self, target_block: NDArray) -> int:
        """
        Find the best overlap position using cross-correlation.
        """
        correlation = correlate(self.previous_block, target_block, mode='full', method='auto')
        mid_point = len(correlation) // 2
        search_start = max(0, mid_point - self.search_window)
        search_end = min(len(correlation), mid_point + self.search_window)
        best_offset = np.argmax(correlation[search_start:search_end]) - self.search_window
        return best_offset

    def _process_current_block(self, audio_chunk: NDArray) -> NDArray:
        if self.previous_block is not None:
            offset = self._find_best_overlap_position(audio_chunk)
            audio_chunk = np.roll(audio_chunk, offset, axis=1)
        self.previous_block = audio_chunk
        return audio_chunk
