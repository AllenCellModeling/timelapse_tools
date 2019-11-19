#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from aicspylibczi import CziFile
from tqdm import tqdm

###############################################################################


class ComputationManager(ABC):
    """
    ComputationManagers are simple objects that can be used to store computation state as each frame (or data cube) of
    an out of memory image is provided. You can construct them anyway you would like, but they must all have a
    `process_data` function. That function has a specific set of parameters that it muct accept. Details on that
    function.

    Why not just provide a function to some loop, why provide a whole class?
    By using an object it allows for a couple things. Better code splitting capabilities, you can include small utility
    functions as underscore methods on the object for example, but the largest benefit is that it allows full control to
    the user over how to store the outputs of each `process_data` iteration.
    """

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def process_data(self, data: np.ndarray, dimensions: List[Tuple[str, int]], file_pointer: CziFile):
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
        """
        pass


def compute(
    input_file: Union[str, Path, CziFile],
    computation_manager: ComputationManager,
    operating_dimension: str = "T",
    series_range: slice = slice(None, None, None),
    S: Optional[int] = None,
    C: Optional[int] = None,
    Z: Optional[int] = None,
    T: Optional[int] = None,
    show_progress: bool = False
):
    # Convert pathlike to CziFile
    if isinstance(input_file, (str, Path)):
        # Resolve path
        input_file = Path(input_file).expanduser().resolve(strict=True)

        # Check path
        if input_file.is_dir():
            raise IsADirectoryError(
                f"Please provide a single file to the `input_file` parameter. Receieved directory: {input_file}"
            )

        # Init czi
        input_file = CziFile(input_file)

    # Check that no other type was provided
    if not isinstance(input_file, CziFile):
        raise TypeError(
            f"Please provide either a string, pathlib.Path, or aicspylibczi.CziFile to the input_file parameter. "
            f"Received {type(input_file)}"
        )

    # Iterate over time dim
    len_T = input_file.dims()[operating_dimension][1]

    # Generate iterator
    iterator = range(len_T)[series_range]

    # Change iterator over to tqdm if desired
    if show_progress:
        iterator = tqdm(iterator)

    # Process each frame
    for i in iterator:
        # Read slice
        if C:
            read_slice, dims = input_file.read_image(T=i, B=0, S=S, C=C)
        else:
            read_slice, dims = input_file.read_image(T=i, B=0, S=S)

        # Compute
        computation_manager.process_data(data=read_slice, dimensions=dims, file_pointer=input_file)

    return computation_manager
