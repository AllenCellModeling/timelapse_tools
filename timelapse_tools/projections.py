#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime
from typing import List, Tuple

import numpy as np
from lxml.etree import _Element

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

    # Combine lower portion by combining zeros with y projection
    zeros = np.zeros((min(y_projection.shape), min(x_projection.shape)))
    lower = np.hstack((zeros, y_projection))

    # Combine upper and lower portions
    return np.vstack((upper, lower))


def t_index_labeler(metadata: _Element, start_T: int, current_T: int, shape: List[Tuple[str, int]]) -> str:
    return f"T: {current_T}"


def t_plus_duration_labeler(metadata: _Element, start_T: int, current_T: int, shape: List[Tuple[str, int]]) -> str:
    # Get timeline elements from metadata
    timeline_elements = metadata.xpath(".//TimelineElement/Time")

    # Get the first and current time point
    begin = datetime.strptime(timeline_elements[start_T].text, "%Y-%m-%dT%H:%M:%S.%fZ")
    current = datetime.strptime(timeline_elements[current_T].text, "%Y-%m-%dT%H:%M:%S.%fZ")

    # Get duration
    duration = str(current - begin)

    # Return the duration from start
    return f"Duration: {duration}"
