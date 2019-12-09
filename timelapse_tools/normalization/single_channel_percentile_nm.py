#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Tuple

import dask.array as da
import numpy as np
from aicspylibczi import CziFile

from .normalization_manager import NormalizationManager

###############################################################################


class SingleChannelPercentileNM(NormalizationManager):
    """
    A NormalizationManager that uses intensity percentiles and clipping to normalize the
    delayed dask array.
    """

    def __init__(
        self,
        min_percentile: float = 50.0,
        max_percentile: float = 99.8,
    ):
        self._min_percentile = min_percentile
        self._max_percentile = max_percentile

    def process_data(
        self,
        data: da,
        dimensions: List[Tuple[str, int]],
        file_pointer: CziFile,
        read_dims: Dict[str, int],
        computation_results: Any
    ):
        # Get the norm by values
        norm_by = np.percentile(
            data.flatten(),
            [self._min_percentile, self._max_percentile]
        )

        # Norm
        normed = (data - norm_by[0]) / (norm_by[1] - norm_by[0])

        # Clip any values outside of 0 and 1
        clipped = np.clip(normed, 0, 1)

        # Scale them between 0 and 255
        return clipped * 255
