#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
from aicspylibczi import CziFile


class ComputationManager(ABC):
    """
    ComputationManagers are simple objects that can be used to store computation state
    as each frame (or data cube) of an out-of-memory image is provided. The only
    requirement of ComputationManager sub-classes is that they must have a
    `process_data` function which has a specific set of parameters. Look at the
    documentation for `process_data` for more details.

    Why not just provide a function to some loop, why specify a whole class as a base?
    By using a class it allows for a couple things: better code splitting capabilities,
    i.e. it is possible to include small utility functions as underscore methods on the
    object for example, but the largest benefit is that it allows full control to the
    user over how to store the outputs of each `process_data` iteration.
    """

    def __init__(self):
        pass

    @abstractmethod
    def process_data(
        self,
        data: np.ndarray,
        dimensions: List[Tuple[str, int]],
        file_pointer: CziFile,
        read_dims: Dict[str, int],
        last_iteration: bool = False,
    ):
        """
        Process a single data cube read from the file. The outputs from each processing
        iteration should be stored as attributes on the ComputationManager instance so
        that they are accessible to future iterations if required.

        Parameters
        ----------
        data: np.ndarray
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
