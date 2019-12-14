#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest
from dask.diagnostics import Profiler

from timelapse_tools import daread

###############################################################################


@pytest.mark.parametrize("img, expected_shape, expected_chunksize, expected_dims", [
    (
        "s_1_t_5_c_1_z_1.czi",
        (1, 5, 1, 1, 624, 924),
        (1, 1, 1, 1, 624, 924),
        "STCZYX"
    ),
    (
        "s_None_t_5_c_1_z_None.czi",
        (5, 1, 1248, 1848),
        (1, 1, 1248, 1848),
        "TCYX"
    ),
    pytest.param(
        "does_not_exist.czi",
        None,
        None,
        None,
        marks=pytest.mark.raises(exception=FileNotFoundError)
    ),
    pytest.param(
        "",
        None,
        None,
        None,
        marks=pytest.mark.raises(exception=IsADirectoryError)
    )
])
def test_daread(data_dir, img, expected_shape, expected_chunksize, expected_dims):
    # Read the data into a dask array
    data, dims = daread(data_dir / img)

    # Do basic checking of shape, chunksize, and dims
    assert data.shape == expected_shape
    assert data.chunksize == expected_chunksize
    assert dims == expected_dims

    # Check that when a single plane is selected, only two tasks run
    getitem_ops = []
    for dim in dims:
        if dim not in ["Y", "X"]:
            getitem_ops.append(0)
        else:
            getitem_ops.append(slice(None, None, None))

    # Run through profiler
    with Profiler() as prof:
        assert isinstance(data[tuple(getitem_ops)].compute(), np.ndarray)
        assert len(prof.results) == 2
