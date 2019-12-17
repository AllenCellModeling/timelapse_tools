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
    img: Path, read_dims: Optional[Dict[str, int]] = None
) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
    # Catch optional read dim
    if read_dims is None:
        read_dims = {}

    # Init czi
    czi = CziFile(img)

    # Read image
    log.debug(f"Reading dimensions: {read_dims}")
    data, dims = czi.read_image(**read_dims)

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

    # Convert ops and run getitem
    return data[tuple(ops)], real_dims


def _imread(img: Path, read_dims: Optional[Dict[str, int]] = None) -> np.ndarray:
    data, dims = _read_image(img, read_dims)
    return data


def daread(img: Union[str, Path]) -> da.core.Array:
    """
    Read a CZI image file as a delayed dask array where each YX plane will be read on
    request.

    Parameters
    ----------
    img: Union[str, Path]
        The filepath to read.

    Returns
    -------
    img: dask.array.core.Array
        The constructed dask array where each YX plane is a delayed read.
    """
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

    # Check that no other type was provided
    if not isinstance(img, Path):
        raise TypeError(
            f"Please provide a path to a file as a string, or an pathlib.Path, to the "
            f"`img` parameter. "
            f"Received type: {type(img)}"
        )

    # Init temp czi
    czi = CziFile(img)

    # Get image dims shape
    image_dims = czi.dims_shape()

    # Setup the read dimensions dictionary for reading the first plane
    first_plane_read_dims = {}
    for dim, dim_info in image_dims.items():
        # Unpack dimension info
        dim_begin_index, dim_end_index = dim_info

        # Add to read dims
        first_plane_read_dims[dim] = dim_begin_index

    # Read first plane for information used by dask.array.from_delayed
    sample, sample_dims = czi.read_image(**first_plane_read_dims)

    # The Y and X dimensions are always the last two dimensions, in that order.
    # These dimensions cannot be operated over but the shape information is used
    # in multiple places so we pull them out for easier access.
    sample_YX_shape = sample.shape[-2:]

    # Create operating shape and dim order list
    operating_shape = czi.size[:-2]
    dims = [dim for dim in czi.dims[:-2]]

    # Create empty numpy array with the operating shape so that we can iter through
    # and use the multi_index to create the readers.
    # We add empty dimensions of size one to fake being the Y and X dimensions.
    lazy_arrays = np.ndarray(operating_shape + (1, 1), dtype=object)

    # We can enumerate over the multi-indexed array and construct read_dims
    # dictionaries by simply zipping together the ordered dims list and the current
    # multi-index plus the begin index for that plane.
    # We then set the value of the array at the same multi-index to
    # the delayed reader using the constructed read_dims dictionary.
    begin_indicies = tuple(image_dims[dim][0] for dim in dims)
    for i, _ in np.ndenumerate(lazy_arrays):
        this_plane_read_indicies = (
            current_dim_begin_index + curr_dim_index
            for current_dim_begin_index, curr_dim_index in zip(begin_indicies, i)
        )
        this_plane_read_dims = dict(zip(dims, this_plane_read_indicies))
        lazy_arrays[i] = da.from_delayed(
            delayed(_imread)(img, this_plane_read_dims),
            shape=sample_YX_shape,
            dtype=sample.dtype,
        )

    # Convert the numpy array of lazy readers into a dask array
    merged = da.block(lazy_arrays.tolist())

    # Because dimensions outside of Y and X can be in any order and present or not
    # we also return the dimension order string.
    dims = dims + ["Y", "X"]
    return merged, "".join(dims)
