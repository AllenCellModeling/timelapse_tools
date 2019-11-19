#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from aicspylibczi import CziFile
from tqdm import tqdm

from .. import exceptions

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

AVAILABLE_OPERATING_DIMENSIONS = ["S", "T", "C", "Z"]

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
    def process_data(
        self,
        data: np.ndarray,
        dimensions: List[Tuple[str, int]],
        file_pointer: CziFile,
        read_dims: Dict[str, int]
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


def compute(
    input_file: Union[str, Path, CziFile],
    computation_manager: ComputationManager,
    operating_dimension: str = "T",
    series_range: slice = slice(None, None, None),
    S: Optional[int] = None,
    C: Optional[int] = None,
    T: Optional[int] = None,
    Z: Optional[int] = None,
    show_progress: bool = False
):
    """
    Run a computation across each data cube retrieved over the operating dimension and store it in the computation
    managers state.

    Parameters
    ----------
    input_file: Union[str, Path, CziFile]
        The file to process.
    computation_manager: ComputationManager
        An initialized computation manager object that will be handed the read data cube on each iteration.
    operating_dimension: str
        Which dimension to iterate over.
        Default: "T" (Time)
    series_range: slice
        A slice object to inform which indices to read over the available indices from the operating dimension.
        Default: all, slice(None, None, None)
    S: Optional[int]
        Which scene to read at each iteration.
        Default: None, read all scenes
    C: Optional[int]
        Which channel to read at each iteration.
        Default: None, read all channels
    T: Optional[int]
        Which timepoint to read at each iteration.
        Default: None, read all timepoints.
    Z: Optional[int]
        Which spatial Z plane to read at each iteration.
        Default: None, read all spatial Z planes.
    show_progress: bool
        Display a progress bar.
        Default: False

    Returns
    -------
    computation_manager: ComputationManager
        The provided computation manager after it has processed each iteration.

    Notes
    -----
    You may notice that the default `operating_dimension` is "T" and that the default `T` is None. This behavior is
    enforced. You cannot specify an index to read on each iteration for a dimension that is acting as the operating
    dimension. As an example, any value besides `T=None`, for `operating_dimension="T"` will raise a
    `ConflictingArgumentsError`. Similary, any value besides `Z=None`, for `operating_dimension="Z"` will raise a
    `ConflictingArgumentsError`.
    """
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

    # Upper operating dimension
    operating_dimension = operating_dimension.upper()

    # Check that the operating dimension exists in the dimensions for the file
    if operating_dimension not in input_file.dims():
        raise ValueError(
            f"Operating dimension: '{operating_dimension}' not found in file dimensions: {input_file.dims()}"
        )

    # Handle operating dimension provided is specified as read specific
    specified_read_dim = locals().get(operating_dimension)
    if specified_read_dim:
        raise exceptions.ConflictingArgumentsError(
            f"Cannot specify reading only '{operating_dimension}': {specified_read_dim} "
            f"because '{operating_dimension}' is currently the operating dimension as well."
        )

    # Iterate over operating dim
    len_operating_dim = input_file.dims()[operating_dimension][1]

    # Generate iterator
    iterator = range(len_operating_dim)[series_range]

    # Change iterator over to tqdm if desired
    if show_progress:
        iterator = tqdm(iterator)

    # Create dictionary to store which dims we will need to read on each iteration
    read_dims = {}
    if "B" in input_file.dims():
        read_dims["B"] = 0

    # Set the available dims left to retrieve
    available_dims = AVAILABLE_OPERATING_DIMENSIONS.copy()
    available_dims.remove(operating_dimension)

    # Set the rest of the passed dims
    for dim in available_dims:
        # Only add the dimension if it exists in the file
        if dim in input_file.dims():
            if locals().get(dim) is not None:
                read_dims[dim] = locals().get(dim)
            else:
                log.warn(
                    f"All '{dim}' dimension data will be read on each iteration. This may increase time and memory."
                )

    # Process each frame
    for i in iterator:
        # Set the operating dimension current index
        read_dims[operating_dimension] = i

        # Read slice
        data, dims = input_file.read_image(**read_dims)

        # Compute
        computation_manager.process_data(data=data, dimensions=dims, file_pointer=input_file, read_dims=read_dims)

    return computation_manager
