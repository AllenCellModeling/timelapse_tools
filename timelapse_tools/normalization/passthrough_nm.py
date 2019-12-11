#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Optional

import dask.array as da
from aicspylibczi import CziFile

from .normalization_manager import NormalizationManager

###############################################################################


class PassthroughNM(NormalizationManager):
    """
    A NormalizationManager that does no operation and simply passes through the data.
    """

    def process_data(
        self,
        data: da.core.Array,
        file_pointer: CziFile,
        computation_results: Optional[Any] = None
    ) -> da.core.Array:
        return data
