#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Tuple

import numpy as np

###############################################################################


def im2proj(
    data: np.ndarray,
    shape: List[Tuple[str, int]],
    C: int = 0,
    min_percent_intensity: float = 50.0,
    max_percent_intensity: float = 99.8,
    project_axis: str = "Z",
    **kwargs
) -> np.ndarray:
    """
    Generate a 2D (YX) projection for the ndarray.

    Parameters
    ----------
    data: np.ndarray
        The image data to create a projection for.
    shape: List[Tuple[str, int]]
        The dimension and shape information to infer projection processing with.
    C: int
        Which channel to use for the projection.
    min_percent_intensity: float
        The minimum percent intensity to allow through before projecting.
    max_percent_intensity: float
        The maximum percent intensity to allow through before projecting.

    Returns
    -------
    projection: np.ndarray
        The 2D (YX) projection.
    """
    # Get shape operations
    ops = []
    kept_dims = []
    for i, dim_and_size in enumerate(shape):
        # Uppercase for safety
        dim, size = dim_and_size
        dim = dim.upper()

        # Set operation for that dimension
        if dim == "S" or dim == "T":
            ops.append(0)
        elif dim == "C":
            ops.append(C)
        else:
            ops.append(slice(None, None, None))
            kept_dims.append(dim)

    # Convert to tuple
    ops = tuple(ops)

    # These percentiles generally work pretty well for our data
    norm_by = np.percentile(data[ops], [min_percent_intensity, max_percent_intensity])

    # Normalize by min and max of the percentiles found
    normed = (data - norm_by[0]) / (norm_by[1] - norm_by[0])

    # Clip any values outside of 0 and 1
    clipped = np.clip(normed, 0, 1)

    # Scale them between 0 and 255
    scaled = clipped * 255

    # Prepare for projection
    project_axis = project_axis.upper()
    projection_index = kept_dims.index(project_axis)

    # Return the max project through Z
    return scaled[ops].max(axis=projection_index).astype(np.uint8)


def im2proj_all_axes(
    data: np.ndarray,
    shape: List[Tuple[str, int]],
    C: int = 0,
    min_percent_intensity: float = 50.0,
    max_percent_intensity: float = 99.8,
    **kwargs
) -> np.ndarray:
    """
    Generate 2D projections for all axes and stack them together for the ndarray.

    Parameters
    ----------
    data: np.ndarray
        The image data to create a projection for.
    shape: List[Tuple[str, int]]
        The dimension and shape information to infer projection processing with.
    C: int
        Which channel to use for the projection.
    min_percent_intensity: float
        The minimum percent intensity to allow through before projecting.
    max_percent_intensity: float
        The maximum percent intensity to allow through before projecting.

    Returns
    -------
    projection: np.ndarray
        The stacked 2D projections as a single 2D array.
    """
    # Gets the YX projection
    z_projection = im2proj(
        data=data,
        shape=shape,
        C=C,
        min_percent_intensity=min_percent_intensity,
        max_percent_intensity=max_percent_intensity,
        project_axis="Z",
        **kwargs
    )

    # Gets the ZX projection
    y_projection = im2proj(
        data=data,
        shape=shape,
        C=C,
        min_percent_intensity=min_percent_intensity,
        max_percent_intensity=max_percent_intensity,
        project_axis="Y",
        **kwargs
    )

    # Gets the ZY projection
    x_projection = im2proj(
        data=data,
        shape=shape,
        C=C,
        min_percent_intensity=min_percent_intensity,
        max_percent_intensity=max_percent_intensity,
        project_axis="X",
        **kwargs
    )

    # Combine upper portion by combining x.T and z projections
    upper = np.hstack((x_projection.T, z_projection))

    # Combine lower portion by combining zeros with the flipped y projection
    zeros = np.zeros((min(y_projection.shape), min(x_projection.shape)))

    # Check which axis to flip
    if len(y_projection.shape) == 3:
        y_projection = np.flip(y_projection, axis=1)
    else:
        y_projection = np.flip(y_projection, axis=0)

    # Stack
    lower = np.hstack((zeros, y_projection))

    # Combine upper and lower portions
    return np.vstack((upper, lower))
