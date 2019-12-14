#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dask.array as da

from .. import exceptions

###############################################################################


def single_channel_max_project(
    data: da.core.Array,
    dims: str,
    max_project_dim: str = "Z",
    **kwargs
) -> da.core.Array:
    if len(data.shape) > 3:
        raise exceptions.InvalidShapeError(len(data.shape), 3)

    return data.max(dims.index(max_project_dim))
