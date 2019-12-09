#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import dask.array as da
import numpy as np
from aicspylibczi import CziFile
from dask import delayed

from .. import exceptions
from ..constants import AVAILABLE_OPERATING_DIMENSIONS

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


def _read_image(
    img: CziFile,
    read_dims: Optional[Dict[str, int]] = None
) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
    # Catch optional read dim
    if read_dims is None:
        read_dims = {}

    # Read image
    data, dims = img.read_image(**read_dims)

    # Drop dims that shouldn't be provided back
    ops = []
    real_dims = []
    for i, dim_info in enumerate(dims):
        # Expand dimension info
        dim, size = dim_info

        # If the dim was provided in the read dims we know a single plane for that
        # dimension was requested so remove it
        if dim in read_dims:
            ops.append(0)
        # Otherwise just read the full slice
        else:
            ops.append(slice(None, None, None))
            real_dims.append(dim_info)

    # Convert ops and run
    return data[tuple(ops)], real_dims


def _imread(img: CziFile, read_dims: Optional[Dict[str, int]] = None) -> np.ndarray:
    data, dims = _read_image(img, read_dims)
    return data


def daread(
    img: Union[str, Path, CziFile],
    S: Optional[int] = None,
    T: Optional[int] = None,
    C: Optional[int] = None,
    Z: Optional[int] = None,
    operating_dimension: str = None
) -> da:
    # Convert pathlike to CziFile
    if isinstance(img, (str, Path)):
        # Resolve path
        img = Path(img).expanduser().resolve(strict=True)

        # Check path
        if img.is_dir():
            raise IsADirectoryError(
                f"Please provide a single file to the `img` parameter. "
                f"Received directory: {img}"
            )

        # Init czi
        img = CziFile(img)

    # Check that no other type was provided
    if not isinstance(img, CziFile):
        raise TypeError(
            f"Please provide a path to a file as a string, pathlib.Path, or an already "
            f"initialized aicspylibczi.CziFile to the `img` parameters. "
            f"Received type: {type(img)}"
        )

    # Upper operating dimension and check valid
    operating_dimension = operating_dimension.upper()
    if operating_dimension not in AVAILABLE_OPERATING_DIMENSIONS:
        raise ValueError(
            f"Operating dimension: '{operating_dimension}' is a not a valid operating "
            f"dimension. Possible options: {AVAILABLE_OPERATING_DIMENSIONS}"
        )

    # Get image dims
    image_dims = img.dims()

    # Check that the operating_dimension is in the dimensions for the file
    if operating_dimension not in image_dims:
        raise ValueError(
            f"Operating dimension: '{operating_dimension}' not found in file "
            f"dimensions: {image_dims}"
        )

    # Handle operating dimension provided is specified as read specific
    specified_read_dim = locals().get(operating_dimension)
    if specified_read_dim:
        raise exceptions.ConflictingArgumentsError(
            f"Cannot specify reading only '{operating_dimension}': "
            f"{specified_read_dim} because '{operating_dimension}' is currently the "
            f"operating dimension."
        )

    # Get the bounds of the time dimension
    op_dim_begin_index, op_dim_len = image_dims[operating_dimension]

    # Create the reading arguments
    read_dims = {operating_dimension: op_dim_begin_index}
    if "B" in image_dims:
        read_dims["B"] = 0

    # Set the available dims left to retrieve
    available_dims = AVAILABLE_OPERATING_DIMENSIONS.copy()
    available_dims.remove(operating_dimension)

    # Set the rest of the passed dims
    for dim in available_dims:
        # Only add the dimension if it exists in the file
        if dim in image_dims:
            if locals().get(dim) is not None:
                read_dims[dim] = locals().get(dim)
            else:
                log.warn(
                    f"All '{dim}' dimension data will be read on each iteration. "
                    f"This may increase time and memory."
                )

    # Read first plane for information used by dask.array.from_delayed
    sample, sample_dims = _read_image(img, read_dims)

    # Create delayed functions
    lazy_arrays = []
    for i in range(op_dim_begin_index, op_dim_begin_index + op_dim_len):
        read_dims[operating_dimension] = i
        lazy_arrays.append(
            delayed(_imread)(img, read_dims)
        )

    # Create dask arrays from delayed
    dask_arrays = [
        # Only use the last two dimensions (YX) from shape
        da.from_delayed(delayed_reader, shape=sample.shape, dtype=sample.dtype)
        for delayed_reader in lazy_arrays
    ]

    # Stack into one large dask.array
    stack = da.stack(dask_arrays, axis=0)

    return stack
