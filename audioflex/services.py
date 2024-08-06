import numpy as np


def overlap_add(audio: np.ndarray, frame: np.ndarray, offset: int) -> np.ndarray:
    """
    Overlap add the 'frame' to 'audio', 'offset' amount of samples from the end of 'audio'
    :param audio: Audio to append (overlap-add) the block on
    :param frame: Frame of audio to append to the audio
    :param offset: Amount of samples that 'frame' should overlap unto the end of 'audio'
    :return: Extended 'audio'
    """
    summed_overlap = audio[:, -offset:] + frame[:, :offset]
    return np.concatenate((audio[:, :-offset], summed_overlap, frame[:, offset:]), axis=1)
