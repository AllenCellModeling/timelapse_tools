#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dask.array as da
from prefect import task

from .. import exceptions

###############################################################################


@task
def single_channel_max_project(data: da.core.Array, dims: str) -> da.core.Array:
    if len(data.shape) > 3:
        raise exceptions.InvalidShapeError(len(data.shape), 3)

    max_project_axis = dims.index("Z")
    return da.max(data, axis=max_project_axis)[0, :]
