import numpy as np

from audioflex.overlap_add import OverlapAdd
from audioflex.phase_alignment import PointBiasPhaseAligner


class WSOLA(OverlapAdd):
    def __init__(self, chunk_size: int, frame_size: int, channels: int, search_range: int):
        super().__init__(channels, chunk_size, frame_size)
        self.search_range = search_range
        self.agg_adjustment = 0
        self.phase_aligner = PointBiasPhaseAligner(0.5, preferred_offset=self.frame_size // 2)

    def get_frame_offset(self, frame: np.ndarray) -> int:
        default_offset = super().get_frame_offset(frame)
        self.phase_aligner.preferred_offset = default_offset
        adjustment = self.phase_aligner.get_closest_alignment_offset(self.last_frame[0], frame[0])
        return adjustment or default_offset
