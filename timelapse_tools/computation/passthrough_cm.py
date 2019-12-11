#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
        data: da.core.Array,
        file_pointer: CziFile
    ):
        pass
