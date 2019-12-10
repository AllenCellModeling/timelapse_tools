#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import dask.array as da
import numpy as np
from aicspylibczi import CziFile
from dask import delayed

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


def daread(img: Union[str, Path, CziFile]) -> da:
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

    # Get image dims
    image_dims = img.dims()
    if "B" in image_dims:
        image_dims.pop("B")

    # Setup the read dimensions dictionary for reading the first plane
    read_dims = {}
    for dim, dim_info in image_dims.items():
        # Unpack dimension info
        dim_begin_index, dim_len = dim_info

        # Add to read dims
        read_dims[dim] = dim_begin_index

    # Read first plane for information used by dask.array.from_delayed
    sample, sample_dims = img.read_image(**read_dims)

    # We want these readers in the true dim order.
    # Because `image_dims` is a dictionary we can't trust the order like we can
    # the `sample_dims`. We use both to have the full positional and size information
    # Before we do this though we need to remove the Y and X dimensions because those
    # are the base operating planes of the CZI format
    operating_dims = [
        dim_info for dim_info in sample_dims if dim_info[0] not in ["Y", "X"]
    ]

    # Just the dim characters
    dims = [dim_info[0] for dim_info in operating_dims]

    # Create operating shape
    operating_shape = []
    for dim_info in operating_dims:
        dim, size = dim_info
        operating_shape.append(image_dims[dim][1])

    operating_shape = tuple(operating_shape)

    # Create empty numpy array to be filled with delayed dask arrays
    correctly_shaped_array = np.ndarray(operating_shape, np.object)
    lazy_arrays = []

    # Iter through the dimensions and add the readers with those indices
    it = np.nditer(correctly_shaped_array, flags=["multi_index", "refs_ok"])
    while not it.finished:
        read_dims = {
            dim: image_dims[dim][0] + it.multi_index[dims.index(dim)]
            for dim in dims
        }
        lazy_arrays.append(da.from_delayed(
            delayed(_imread)(img, read_dims),
            shape=sample.shape[-2:],
            dtype=sample.dtype
        ))
        it.iternext()

    # Convert to dask array
    # We flatten then to list so that we can stack and reshape properly
    real_shape = tuple(list(operating_shape) + list(sample.shape[-2:]))
    chunk_shape = tuple(list(np.ones(len(operating_shape))) + list(sample.shape[-2:]))
    stacked = da.stack(lazy_arrays).reshape(real_shape).rechunk(chunk_shape)

    # Convert to dask array and return
    dims = dims + ["Y", "X"]
    return stacked, "".join(dims)
