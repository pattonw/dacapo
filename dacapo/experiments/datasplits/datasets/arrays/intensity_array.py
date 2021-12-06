from .array import Array

from funlib.geometry import Coordinate, Roi

import numpy as np


class IntensitiesArray(Array):
    """
    This is wrapper another array that will normalize intensities to
    the range (0, 1) and convert to float32. Use this if you have your
    intensities stored as uint8 or similar and want your model to
    have floats as input.   
    """

    def __init__(self, array_config):
        super().__init__()
        self._source_array = array_config.source_array_config.array_type(
            array_config.source_array_config
        )

        self._min = array_config.min
        self._max = array_config.max

    @property
    def axes(self):
        return self._source_array.axes

    @property
    def dims(self) -> int:
        return self._source_array.dims

    @property
    def voxel_size(self) -> Coordinate:
        return self._source_array.voxel_size

    @property
    def roi(self) -> Roi:
        return self._source_array.roi

    @property
    def writable(self) -> bool:
        return False

    @property
    def dtype(self):
        return np.float32

    @property
    def num_channels(self) -> int:
        return self._source_array.num_channels

    @property
    def data(self):
        raise ValueError(
            "Cannot get a writable view of this array because it is a virtual "
            "array created by modifying another array on demand."
        )

    def __getitem__(self, roi: Roi) -> np.ndarray:
        intensities = self._source_array[roi]
        normalized = (intensities.astype(np.float32) - self._min) / (self._max - self._min)
        return normalized
