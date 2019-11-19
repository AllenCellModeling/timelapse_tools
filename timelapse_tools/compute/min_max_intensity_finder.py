#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Tuple

import numpy as np
from aicspylibczi import CziFile

from .computation_manager import ComputationManager

###############################################################################


class MinMaxIntensityFinder(ComputationManager):
    """
    Find the minimum and maximum intensity values from the entire series of data cubes and store them in state.
    """

    def __init__(self):
        self.min_intensity = None
        self.max_intensity = None

    def process_data(self, data: np.ndarray, dimensions: List[Tuple[str, int]], file_pointer: CziFile):
        # Get min and max
        min = data.min()
        max = data.max()

        # Update the processors state if required
        if self.min_intensity is None or self.min_intensity > min:
            self.min_intensity = min
        if self.max_intensity is None or self.max_intensity < max:
            self.max_intensity = max
