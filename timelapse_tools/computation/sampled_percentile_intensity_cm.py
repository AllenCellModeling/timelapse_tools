#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, List, Tuple

import numpy as np
from aicspylibczi import CziFile

from .computation_manager import ComputationManager

###############################################################################


class SampledPercentileIntensityCM(ComputationManager):
    """
    Find the minimum and maximum intensity values from the entire series of data cubes
    and store them in state.
    """

    def __init__(
        self,
        min_percentile: float = 50.0,
        max_percentile: float = 99.8,
        sample_percent: float = 0.05,
    ):
        self._sampled_values = None

    def process_data(
        self,
        data: np.ndarray,
        dimensions: List[Tuple[str, int]],
        file_pointer: CziFile,
        read_dims: Dict[str, int],
        last_iteration: bool = False,
    ):
        pass
