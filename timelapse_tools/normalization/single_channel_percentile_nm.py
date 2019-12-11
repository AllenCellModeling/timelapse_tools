#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Optional

import dask.array as da
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
        data: da.core.Array,
        file_pointer: CziFile,
        computation_results: Optional[Any] = None
    ) -> da.core.Array:
        # Get the norm by values
        norm_by = da.percentile(
            data.flatten(),
            [self._min_percentile, self._max_percentile]
        )

        # Norm
        normed = (data - norm_by[0]) / (norm_by[1] - norm_by[0])

        # Clip any values outside of 0 and 1
        clipped = da.clip(normed, 0, 1)

        # Scale them between 0 and 255
        return clipped * 255
