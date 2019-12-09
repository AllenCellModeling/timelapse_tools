#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, List, Tuple

import numpy as np
from aicspylibczi import CziFile

from .computation_manager import ComputationManager

###############################################################################


class PassthroughCM(ComputationManager):
    """
    A ComputationManager that does no operation and simply passes through the data.
    """

    def process_data(
        self,
        data: np.ndarray,
        dimensions: List[Tuple[str, int]],
        file_pointer: CziFile,
        read_dims: Dict[str, int],
        last_iteration: bool = False,
    ):
        pass
