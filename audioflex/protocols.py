from typing import Protocol

import numpy as np


class SliceableArray(Protocol):
    def __getitem__(self, item: slice) -> np.ndarray: ...
