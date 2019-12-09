#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

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
        data: da,
        dimensions: List[Tuple[str, int]],
        file_pointer: CziFile,
        read_dims: Dict[str, int],
        computation_results: Any
    ) -> Any:
        """
        Process the data cube read from the file. The outputs from processing should be
        stored as attributes on the ComputationManager instance.

        Parameters
        ----------
        data: da
            The read data cube to process.
        dimensions: List[Tuple[str, int]]
            The dimensions object returned during the read data operation.
        file_pointer: CziFile
            The file pointer (or buffer reference) in the case you want to explicitly
            access more information from the file during each process operation.
        read_dims: Dict[str, int]
            Which dimensions and indices were used to read the provided data cube.
        """
        pass
