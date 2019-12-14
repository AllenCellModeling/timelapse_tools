#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest

from timelapse_tools import daread

###############################################################################


@pytest.mark.parametrize(
    "img, expected_shape, expected_chunksize, expected_dims, getitem_ops",
    [
        (
            "s_1_t_5_c_1_z_1.czi",
            (1, 5, 1, 1, 624, 924),
            (1, 1, 1, 1, 624, 924),
            "STCZYX",
            (0, 0, 0, 0, slice(None, None, None), slice(None, None, None))
        ),
        (
            "s_None_t_5_c_1_z_None.czi",
            (5, 1, 1248, 1848),
            (1, 1, 1248, 1848),
            "TCYX",
            (0, 0, slice(None, None, None), slice(None, None, None))
        ),
        pytest.param(
            "does_not_exist.czi",
            None,
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
            None,
            marks=pytest.mark.raises(exception=IsADirectoryError)
        )
    ]
)
def test_daread(
    data_dir,
    img,
    expected_shape,
    expected_chunksize,
    expected_dims,
    getitem_ops
):
    data, dims = daread(data_dir / img)
    assert data.shape == expected_shape
    assert data.chunksize == expected_chunksize
    assert dims == expected_dims
    assert len(data[getitem_ops].shape) == 2
    assert isinstance(data[getitem_ops].compute(), np.ndarray)
