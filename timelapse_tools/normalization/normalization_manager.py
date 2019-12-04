#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
from aicspylibczi import CziFile

from .. import exceptions
from ..computation.computation_manager import ComputationManager

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

AVAILABLE_OPERATING_DIMENSIONS = ["S", "T", "C", "Z"]

###############################################################################


class NormalizationManager(ABC):
    """
    NormalizationManagers are simple objects that can be used to normalize each frame (or data cube) of an out of
    memory image provided. You can construct them anyway you would like, but they must all have a `process_data`
    function. That function has a specific set of parameters that it muct accept. Details on that function.

    Why not just provide a function to some loop, why provide a whole class?
    By using an object it allows for a couple things. Better code splitting capabilities, you can include small utility
    functions as underscore methods on the object for example, but the largest benefit is that it allows full control to
    the user over how to store the outputs of each `process_data` iteration.
    """

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def process_data(
        self,
        data: np.ndarray,
        dimensions: List[Tuple[str, int]],
        file_pointer: CziFile,
        read_dims: Dict[str, int],
        computation_manager: Optional[ComputationManager] = None
    ):
        """
        Process a single data cube read from the file. You should store the outputs from your processing on as
        attributes on this objects. So that you have access to them on the next iteration if required.

        Parameters
        ----------
        data: np.ndarray
            The read data cube to process.
        dimensions: List[Tuple[str, int]]
            The dimensions object returned during the read data operation.
        file_pointer: CziFile
            The file pointer (or buffer reference) in the case you want to explicitely access more information from the
            file during each process operation.
        read_dims: Dict[str, int]
            The dimensions and which indices that were used to read this data cube from the file.
        """
        pass
