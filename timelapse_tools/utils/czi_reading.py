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

    # This produces a list of tuples of dim character and index such as:
    # [("S", 0), ("T", 0), ("C", 0), ("Z", 0)]
    operating_dims = [
        dim_info for dim_info in sample_dims if dim_info[0] not in ["Y", "X"]
    ]

    # Create operating shape and dim order list
    # Using the image_dims pulled at the start and the operating_dims,
    # match them up to get the true shape of the data without the YX plane sizes
    # This will result in a list of dim size integers such as:
    # [3, 100, 2, 70]
    # ["S", "T", "C", "Z"]
    operating_shape = []
    dims = []
    for dim_info in operating_dims:
        dim, size = dim_info
        operating_shape.append(image_dims[dim][1])
        dims.append(dim)

    # Convert to tuple
    operating_shape = tuple(operating_shape)

    # Create empty numpy array with the operating shape so that we can iter through
    # and use the multi_index to create the readers
    # We add empty dimensions of size one to fake being the Y and X dimensions
    lazy_arrays = np.ndarray(operating_shape + (1, 1), dtype=object)

    # We can enumerate over the multi-indexed array and construct read_dims
    # dictionaries by simply zipping together the ordered dims list and the current
    # multi-index. We then pass set the value of the array at that multi-index to
    # the delayed reader using the constructed read_dims dictionary.
    for i, _ in np.ndenumerate(lazy_arrays):
        read_dims = dict(zip(dims, i))
        lazy_arrays[i] = da.from_delayed(
            delayed(_imread)(img, read_dims),
            shape=sample_YX_shape,
            dtype=sample.dtype
        )

    # Convert the numpy array of lazy readers into a dask array
    merged = da.block(lazy_arrays.tolist())

    # Because dimensions outside of Y and X can be in any order and present or not
    # we also return the dimension order string
    dims = dims + ["Y", "X"]
    return merged, "".join(dims)
