#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dask.array as da

from .. import exceptions

###############################################################################


def single_channel_max_project(
    data: da.core.Array, dims: str, max_project_dim: str = "Z", **kwargs
) -> da.core.Array:
    # Check shape
    if len(data.shape) > 3:
        raise exceptions.InvalidShapeError(len(data.shape), 3)

    # If shape is three, we know we need to project
    if len(data.shape) == 3:
        return data.max(dims.index(max_project_dim))

    # If it is less than three, it's two and just return
    return data
