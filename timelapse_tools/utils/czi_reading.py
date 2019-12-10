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
    # Remove B from dims because `read_image` never returns B even if it is in the file
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

    # The Y and X dimensions are always the last two dimensions, in that order
    # These dimensions cannot be operated over but the shape information is used
    # in multiple places so we pull out for storage
    sample_YX_shape = sample.shape[-2:]

    # We want these readers in the true dim order.
    # Because `image_dims` is a dictionary we can't trust the order like we can
    # the `sample_dims`. We use both to have the full positional and size information
    # Before we do this though we need to remove the Y and X dimensions because those
    # are the base storage planes of the CZI format and we can't operate over them
    # At the end of this we are going to produce a `lazy_arrays` list of delayed dask
    # arrays that are of the 2D YX planes found in the file. Everything else is simply
    # reshaping of the array to fit the normal image model
    operating_dims = [
        dim_info for dim_info in sample_dims if dim_info[0] not in ["Y", "X"]
    ]

    # Create operating shape
    # This will result in a tuple of dim size integers
    operating_shape = []
    for dim_info in operating_dims:
        dim, size = dim_info
        operating_shape.append(image_dims[dim][1])

    # Convert to tuple
    operating_shape = tuple(operating_shape)

    # Create empty numpy array with the operating shape so that we can iter through
    # and use the multi_index to create the readers
    correctly_shaped_array = np.ndarray(operating_shape)

    # Create a flat list of all the delayed dask arrays
    lazy_arrays = []

    # Create an array of just the dimension characters
    # This list of dim characters will be in the same order as the operating_shape
    dims = [dim_info[0] for dim_info in operating_dims]

    # Iter through the dimensions and add the readers with iteration indices
    it = np.nditer(correctly_shaped_array, flags=["multi_index", "refs_ok"])
    while not it.finished:
        # Construct read dims from the multi index for this iteration
        # Because the operating shape and the dims are in the same order
        # We can basically zip the multi index with the dim character but as a dict
        read_dims = {
            dim: image_dims[dim][0] + it.multi_index[i]
            for i, dim in enumerate(dims)
        }

        # Append the delayed dask array to the flat list of arrays
        lazy_arrays.append(da.from_delayed(
            # In short terms, this creates a partial function of _imread
            # with parameters im and read_dims.
            delayed(_imread)(img, read_dims),

            # Because we are using the _imread function and the hyper specific
            # read_dims, we know we are expecting a 2D YX plane from this
            # from_delayed needs to know the shape of the expecation
            # we use the last two values from the sample to pass through because
            # the sample shape will also have YX on the end
            shape=sample_YX_shape,
            dtype=sample.dtype
        ))
        it.iternext()

    # Convert to dask array
    # We create the real shape by simply adding the YX shape to the operating shape
    real_shape = tuple(list(operating_shape) + list(sample_YX_shape))
    # Stack and reshape to get what would normally be returned by any old imread
    stacked = da.stack(lazy_arrays).reshape(real_shape)

    # Because dimensions outside of Y and X can be in any order and present or not
    # we also return the dimension order string
    dims = dims + ["Y", "X"]
    return stacked, "".join(dims)
