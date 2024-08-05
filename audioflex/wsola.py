import numpy as np

from audioflex.overlap_add import OverlapAdd
from audioflex.phase_alignment import PointBiasPhaseAligner


class WSOLA(OverlapAdd):
    def __init__(self, chunk_size: int, frame_size: int, channels: int):
        super().__init__(channels, chunk_size, frame_size)
        self.phase_aligner = PointBiasPhaseAligner(0.5, preferred_offset=self.hop_size)

    def get_frame_offset(self, audio: np.ndarray, frame: np.ndarray) -> int:
        adjustment = self.phase_aligner.get_closest_alignment_offset(self.last_frame[0], frame[0])
        return adjustment or self.hop_size
