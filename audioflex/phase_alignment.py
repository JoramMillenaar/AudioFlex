from typing import Protocol

import numpy as np
from scipy.signal import correlate


class PhaseAligner(Protocol):
    def get_closest_alignment_offset(self, a: np.ndarray, b: np.ndarray) -> int:
        pass


class PointBiasPhaseAligner(PhaseAligner):
    def __init__(self, point_bias_factor: float, preferred_offset: int):
        self.point_bias_factor = point_bias_factor
        self.preferred_offset = preferred_offset

    def get_bias_window(self, length: int):
        index_array = np.arange(length)
        if self.point_bias_factor == 0:
            std_dev = length
        else:
            std_dev = 1 / self.point_bias_factor

        bias_window = np.exp(-0.5 * ((index_array - self.preferred_offset) / std_dev) ** 2)
        bias_window /= np.max(bias_window)
        return bias_window

    def get_closest_alignment_offset(self, a: np.ndarray, b: np.ndarray) -> int | None:
        correlation = correlate(a, b, mode='same')
        if not correlation.any():
            return  # No correlations

        bias = self.get_bias_window(len(correlation))

        biased_correlation = correlation * bias
        max_index = np.argmax(biased_correlation)
        return max_index
