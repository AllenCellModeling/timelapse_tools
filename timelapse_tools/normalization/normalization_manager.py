#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Any, Optional

import dask.array as da
from aicspylibczi import CziFile

###############################################################################


class NormalizationManager(ABC):
    """
    NormalizationManagers are simple objects used to manage normalization of the delayed
    dask array. The only requirement of NormalizationManager sub-classes is that they
    must have a `process_data` function which has a specific set of parameters. Look at
    the documentation for `process_data` for more details.

    Why not just provide a function, why specify a whole class?
    By using a class it allows for a couple things: better code splitting capabilities,
    i.e. it is possible to include small utility functions as underscore methods on the
    object for example. Additionally, if you have a lot of computation to do you may
    want some sort of state management that a class provides.
    """

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def process_data(
        self,
        data: da.core.Array,
        file_pointer: CziFile,
        computation_results: Optional[Any] = None
    ) -> da.core.Array:
        """
        Normalize a data cube.

        Parameters
        ----------
        data: dask.array.core.Array
            The data cube to normalize.
        file_pointer: CziFile
            The file pointer (or buffer reference) in the case you want to explicitly
            access more information from the file during the process operation.
        computation_results: Optional[Any]
            The results generated from the previously ran ComputationManager.

        Returns
        -------
        data: dask.array.core.Array
            The normalized data cube.

        Notes
        -----
        You do not need to call `compute` on the dask array during this function.
        We will call it at the end of the entire workflow.
        """
        pass
