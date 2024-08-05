import numpy as np

from audioflex.overlap_add import OverlapAdd
from audioflex.phase_alignment import PointBiasPhaseAligner


class WSOLA(OverlapAdd):
    def __init__(self, chunk_size: int, block_size: int, channels: int, search_range: int):
        super().__init__(channels, chunk_size, block_size)
        self.search_range = search_range
        self.agg_adjustment = 0
        self.phase_aligner = PointBiasPhaseAligner(0.5, preferred_offset=self.block_size // 2)

    def get_block_offset(self, block: np.ndarray) -> int:
        default_offset = super().get_block_offset(block)
        self.phase_aligner.preferred_offset = default_offset
        adjustment = self.phase_aligner.get_closest_alignment_offset(self.previous_block[0], block[0])
        return adjustment or default_offset
