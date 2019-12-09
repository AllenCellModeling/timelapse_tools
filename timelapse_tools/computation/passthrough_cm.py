#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, List, Tuple

import dask.array as da
from aicspylibczi import CziFile

from .computation_manager import ComputationManager

###############################################################################


class PassthroughCM(ComputationManager):
    """
    A ComputationManager that does no operation and simply passes through the data.
    """

    def process_data(
        self,
        data: da,
        dimensions: List[Tuple[str, int]],
        file_pointer: CziFile,
        read_dims: Dict[str, int],
    ):
        pass
