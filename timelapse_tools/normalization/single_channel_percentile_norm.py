#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dask.array as da

from .. import exceptions

###############################################################################


def percentile_norm(
    data: da.core.Array,
    min_p: float = 50.0,
    max_p: float = 99.8,
    **kwargs
) -> da.core.Array:
    # Enforce shape
    if len(data.shape) > 4:
        raise exceptions.InvalidShapeError(len(data.shape), 4)

    # Get the norm by values
    norm_by = da.percentile(
        data.flatten(),
        [min_p, min_p]
    )

    # Norm
    normed = (data - norm_by[0]) / (norm_by[1] - norm_by[0])

    # Clip any values outside of 0 and 1
    clipped = da.clip(normed, 0, 1)

    # Scale them between 0 and 255
    return clipped * 255
