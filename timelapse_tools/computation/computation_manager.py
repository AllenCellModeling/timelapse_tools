#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Any

import dask.array as da
from aicspylibczi import CziFile

###############################################################################


class ComputationManager(ABC):
    """
    ComputationManagers are simple objects used to manage computation of a delayed dask
    array. The only requirement of ComputationManager sub-classes is that they must have
    a `process_data` function which has a specific set of parameters. Look at the
    documentation for `process_data` for more details.

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
        file_pointer: CziFile
    ) -> Any:
        """
        Process the data cube read from the file.

        Parameters
        ----------
        data: dask.array.core.Array
            The read data cube to process.
        file_pointer: CziFile
            The file pointer (or buffer reference) in the case you want to explicitly
            access more information from the file during each process operation.
        """
        pass
