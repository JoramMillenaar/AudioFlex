import numpy as np

from audioflex.overlap_add import OverlapAdd


class WSOLA(OverlapAdd):
    def __init__(self, chunk_size: int, block_size: int, channels: int, search_range: int, frame_length: int):
        """
        WSOLA algorithm for timescale modification of audio signals without affecting pitch.
        :param chunk_size: Expected length of the inputted audio_chunks
        :param channels: Amount of channels expected for the audio processor input
        :param block_size: Amount of samples to divide the input in to overlap
        :param search_range: Amount of samples to search through to find the best overlap position
        :param frame_length: Amount samples to include in the frame that finds the best overlap position
        """
        super().__init__(channels, chunk_size, block_size)
        self.search_range = search_range
        self.frame_length = frame_length
        self.previous_block = None

    def get_adjustment(self, audio_chunk, frame, center_affinity=0.5):
        frame_length = frame.shape[1]
        center = audio_chunk.shape[1] // 2
        start = max(center - self.search_range // 2 - frame_length // 2, 0)
        end = min(start + self.search_range, audio_chunk.shape[1] - frame_length)

        # Normalize the frames for comparison
        norm_audio_chunk = (audio_chunk - np.mean(audio_chunk, axis=1, keepdims=True)) / np.std(audio_chunk, axis=1, keepdims=True)
        norm_frame = (frame - np.mean(frame, axis=1, keepdims=True)) / np.std(frame, axis=1, keepdims=True)

        # Calculate cross-correlation
        correlations = np.array([
            np.sum(norm_audio_chunk[:, i:i + frame_length] * norm_frame)
            for i in range(start, end)
        ])

        # Apply weighting to balance similarity and proximity
        affinities = np.bartlett(self.search_range)
        weighted_scores = (1 - center_affinity) * correlations + center_affinity * affinities

        # Find the position of maximum weighted score (maximize correlation)
        best_adjustment = np.argmax(weighted_scores) - (self.search_range // 2)

        return best_adjustment + start

    def get_sample_offset(self):
        offset = super().get_sample_offset()
        if self.previous_block is not None:
            edge = self.block_size // 2 - self.search_range // 2
            adjustment = self.get_adjustment(self.previous_block[:, edge:], self.current_block[:, :edge])
            adjustment -= self.search_range // 2
            offset += adjustment
        self.previous_block = self.current_block
        return offset

